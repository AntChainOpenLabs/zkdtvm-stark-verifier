mod lt_polyair;
pub use lt_polyair::*;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use dt_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ALUTypeRecord, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};

use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    Word,
};

use crate::{
    adapter::{ALUTypeRegisterOp, CPUState},
    air::{DTCoreAirBuilder, WordAirBuilder},
    operations::LtOperationSigned,
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};
/// The number of main trace columns for `LtChip`.
pub const NUM_LT_COLS: usize = size_of::<LtCols<u8>>();

/// A chip that implements comparison operations for the opcodes SLT and SLTU.
#[derive(Default)]
pub struct LtChip;
/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LtCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub cpu_state: CPUState<T>,

    /// The adapter to read program and register information.
    pub mem_ops: ALUTypeRegisterOp<T>,

    /// If the opcode is SLT.
    pub is_slt: T,

    /// If the opcode is SLTU.
    pub is_sltu: T,

    /// Instance of `LtOperationSigned` to handle comparison logic in `LtChip`'s ALU operations.
    pub lt_operation: LtOperationSigned<T>,
}
impl<F: Field> MachineAir<F> for LtChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Lt".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.lt_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));

        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let mut values = zeroed_f_vec(nb_rows * NUM_LT_COLS);
        values.chunks_mut(chunk_size * NUM_LT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_LT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LtCols<F> = row.borrow_mut();
                    let mut byte_lookup_events = Vec::new();
                    let (record, event) = &input.lt_events[idx];
                    self.event_to_row(record, event, cols, &mut byte_lookup_events, shard);
                });
            },
        );

        let main = RowMajorMatrix::new(values, NUM_LT_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_LT_COLS }, padded_nb_rows)
    }
    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.lt_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();
        let blu_batches = input
            .lt_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(record, event)| {
                    let mut row = [F::zero(); NUM_LT_COLS];
                    let cols: &mut LtCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.lt_events.is_empty()
        }
    }
    fn local_only(&self) -> bool {
        true
    }
}
impl LtChip {
    /// Create a row from an event.
    fn event_to_row<F: Field>(
        &self,
        record: &ALUTypeRecord,
        event: &AluEvent,
        cols: &mut LtCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);

        // When rd=x0, executor sets event.a=0 (x0 hardwired to 0),
        // but LtOperation needs the actual comparison result to populate correctly.
        let a = if event.op_a_0 {
            if event.opcode == Opcode::SLT {
                ((event.b as i32) < (event.c as i32)) as u32
            } else {
                (event.b < event.c) as u32
            }
        } else {
            event.a
        };
        cols.lt_operation.populate(blu, a, event.b, event.c, event.opcode == Opcode::SLT);

        cols.is_slt = F::from_bool(event.opcode == Opcode::SLT);
        cols.is_sltu = F::from_bool(event.opcode == Opcode::SLTU);
    }
}
impl<F> BaseAir<F> for LtChip {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }
}

impl<AB> Air<AB> for LtChip
where
    AB: DTCoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LtCols<AB::Var> = (*local).borrow();
        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let op_a_word = *local.mem_ops.op_a_value();
        let op_b_word = *local.mem_ops.op_b_value();
        let op_c_word = *local.mem_ops.op_c_value();
        // SAFETY: All selectors `is_slt`, `is_sltu` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_slt + is_sltu` is
        // boolean. Therefore, the `opcode` matches the corresponding opcode.
        let is_real = local.is_slt + local.is_sltu;
        builder.assert_bool(local.is_slt);
        builder.assert_bool(local.is_sltu);
        builder.assert_bool(is_real.clone());

        // Evaluate the LT operation.
        LtOperationSigned::<AB::F>::eval(
            builder,
            op_b_word.map(Into::into),
            op_c_word.map(Into::into),
            local.lt_operation,
            local.is_slt.into(),
            is_real.clone(),
        );
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            is_real.clone(),
            execution_shard,
        );

        // Get the opcode for the operation.
        let opcode = local.is_slt * AB::F::from_canonical_u32(Opcode::SLT as u32) +
            local.is_sltu * AB::F::from_canonical_u32(Opcode::SLTU as u32);
        //alu type mem_ops
        ALUTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            is_real.clone(),
        );
        // Constraint comparison result: skip when rd=x0 (op_a_zero=1),
        // because op_a_word is forced to 0 but result may be 1.
        let expected_result: Word<AB::Expr> =
            Word::extend_var::<AB>(local.lt_operation.result.result);
        let perform_calc = is_real.clone() - local.mem_ops.op_a_zero.into();
        builder.when(perform_calc).assert_word_eq(op_a_word, expected_result);
    }
}
