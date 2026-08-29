mod bitwise_polyair;
pub use bitwise_polyair::*;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapter::{ALUTypeRegisterOp, CPUState},
    air::DTCoreAirBuilder,
};
use dt_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ALUTypeRecord, ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};

use crate::utils::{next_power_of_two, padded_rows_threshold};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the opcodes XOR, OR, and AND.
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct BitwiseCols<T> {
    ///cpu state
    pub cpu_state: CPUState<T>,
    /// register read-write operations
    pub mem_ops: ALUTypeRegisterOp<T>,
    ///is Xor instruction
    pub is_xor: T,
    /// is Or instruction
    pub is_or: T,
    /// is And instruction
    pub is_and: T,
    /// is real row
    pub is_real: T,
}

impl<F: Field> MachineAir<F> for BitwiseChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Bitwise".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let shard = input.execution_shard();
        let rows: Vec<[F; NUM_BITWISE_COLS]> = input
            .bitwise_events
            .par_iter()
            .map(|(record, event)| {
                let mut row = [F::zero(); NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(record, event, cols, &mut blu, shard);
                row
            })
            .collect::<Vec<_>>();

        let real_nb_rows = rows.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(real_nb_rows, size_log2));

        let main =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_BITWISE_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_BITWISE_COLS }, padded_nb_rows)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let blu_batches = input
            .bitwise_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(record, event)| {
                    let mut row = [F::zero(); NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.bitwise_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}
impl BitwiseChip {
    fn event_to_row<F: Field>(
        &self,
        record: &ALUTypeRecord,
        event: &AluEvent,
        cols: &mut BitwiseCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);

        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);
        cols.is_real = F::one();

        if !event.op_a_0 {
            let a = event.a.to_le_bytes();
            let b = event.b.to_le_bytes();
            let c = event.c.to_le_bytes();
            let byte_opcode = match event.opcode {
                Opcode::XOR => ByteOpcode::XOR,
                Opcode::OR => ByteOpcode::OR,
                Opcode::AND => ByteOpcode::AND,
                _ => panic!("Invalid bitwise opcode"),
            };

            for i in 0..4 {
                blu.add_byte_lookup_event(ByteLookupEvent {
                    opcode: byte_opcode,
                    a1: a[i] as u16,
                    a2: 0,
                    b: b[i],
                    c: c[i],
                });
            }
        }
    }
}

impl<F> BaseAir<F> for BitwiseChip {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}
impl<AB> Air<AB> for BitwiseChip
where
    AB: DTCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseCols<AB::Var> = (*local).borrow();

        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let one: AB::Expr = AB::F::one().into();

        builder.assert_bool(local.is_real);

        // cpu state transition
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );

        // get opcode and byte opcode
        let (opcode, byte_opcode) = {
            let xor_op = AB::F::from_canonical_u8(Opcode::XOR as u8);
            let or_op = AB::F::from_canonical_u8(Opcode::OR as u8);
            let and_op = AB::F::from_canonical_u8(Opcode::AND as u8);

            let xor_byte = AB::F::from_canonical_u8(ByteOpcode::XOR as u8);
            let or_byte = AB::F::from_canonical_u8(ByteOpcode::OR as u8);
            let and_byte = AB::F::from_canonical_u8(ByteOpcode::AND as u8);
            builder.assert_bool(local.is_and);
            builder.assert_bool(local.is_xor);
            builder.assert_bool(local.is_or);
            builder.when(local.is_real).assert_one(local.is_xor + local.is_or + local.is_and);

            let op = local.is_xor * xor_op + local.is_or * or_op + local.is_and * and_op;
            let b_op = local.is_xor * xor_byte + local.is_or * or_byte + local.is_and * and_byte;
            (op, b_op)
        };

        //bit operations
        let perform_calc = local.is_real - local.mem_ops.op_a_zero;
        let a = *local.mem_ops.op_a_value();
        let b = *local.mem_ops.op_b_value();
        let c = *local.mem_ops.op_c_value();

        for i in 0..4 {
            builder.send_byte(byte_opcode.clone(), a[i], b[i], c[i], perform_calc.clone());
        }

        ALUTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            local.is_real.into(),
        );

        builder.when(one - local.is_real).assert_zero(local.mem_ops.op_a_zero);
    }
}
