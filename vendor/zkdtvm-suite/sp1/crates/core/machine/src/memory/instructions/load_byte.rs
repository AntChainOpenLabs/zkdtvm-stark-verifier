use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ByteOpcode, ExecutionRecord, ITypeRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    Word,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

use crate::{
    adapter::{CPUState, ITypeRegisterOp},
    air::{DTCoreAirBuilder, WordAirBuilder},
    memory::{instructions::operations::AddressOperation, MemoryCols, MemoryReadCols},
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};
#[derive(Default)]
pub struct LoadByteChip;

pub const NUM_LOAD_BYTE_COLUMNS: usize = size_of::<LoadByteCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadByteCols<T> {
    pub cpu_state: CPUState<T>,

    /// op_b <- rs1, op_c <- imm, op_a ->rd
    pub mem_ops: ITypeRegisterOp<T>,
    // op_b + op_c
    pub address_operation: AddressOperation<T>,
    /// memory read
    pub memory_access: MemoryReadCols<T>,
    //offset_bits inner_product with load_word
    pub selected_byte: T,
    /*
    degree 3 constraint:
    let padding_byte = msb * lb * 0xff
    load_word = [selected_byte,padding_byte,padding_byte,padding_byte]
    builder.when(is_real - cols.mem_ops.op_a_zero).assert_word_eq(load_word, cols.mem_ops.op_a_value());
    */
    pub msb: T,

    pub is_lb: T,
    pub is_lbu: T,
    pub is_real: T,
}

impl<F> BaseAir<F> for LoadByteChip {
    fn width(&self) -> usize {
        NUM_LOAD_BYTE_COLUMNS
    }
}

impl<F: Field> MachineAir<F> for LoadByteChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadByte".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.load_byte_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(nb_rows * NUM_LOAD_BYTE_COLUMNS);
        let shard = input.execution_shard();
        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_BYTE_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_BYTE_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadByteCols<F> = row.borrow_mut();
                    let (record, event) = &input.load_byte_events[idx];
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_LOAD_BYTE_COLUMNS);
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_LOAD_BYTE_COLUMNS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.load_byte_events.is_empty()
        }
    }
}

impl LoadByteChip {
    pub(crate) fn event_to_row<F: Field>(
        &self,
        record: &ITypeRecord,
        event: &MemInstrEvent,
        cols: &mut LoadByteCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);
        // Populate memory accesses for reading from memory.
        cols.memory_access
            .populate(event.mem_access.read_record().expect("load event reads from memory"), blu);

        let mem_addr = cols.address_operation.populate(blu, event.b, event.c);

        //select byte from memory value (not from address)
        let mem_value = event.mem_access.read_record().expect("load event reads from memory").value;
        let mem_value_bytes = mem_value.to_le_bytes();
        let byte_offset_in_word = (mem_addr & 0b11) as usize;
        let selected_byte = mem_value_bytes[byte_offset_in_word];
        cols.selected_byte = F::from_canonical_u8(selected_byte);

        //msb (0 or 1, not 0 or 128)
        let msb = (selected_byte >> 7) as u16;
        cols.msb = F::from_canonical_u8(msb as u8);
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::MSB,
            a1: msb,
            a2: 0,
            b: selected_byte,
            c: 0,
        });
        cols.is_lb = F::from_bool(event.opcode == Opcode::LB);
        cols.is_lbu = F::from_bool(event.opcode == Opcode::LBU);
        cols.is_real = F::from_bool(true);
    }
}

impl<AB> Air<AB> for LoadByteChip
where
    AB: DTCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadByteCols<AB::Var> = (*local).borrow();

        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_lbu);
        builder.assert_bool(local.is_lb);

        // cpu state transition
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );
        // register op + program lookup
        let opcode = local.is_lb * AB::F::from_canonical_u8(Opcode::LB as u8) +
            local.is_lbu * AB::F::from_canonical_u8(Opcode::LBU as u8);
        builder.assert_eq(local.is_real, local.is_lb + local.is_lbu);
        ITypeRegisterOp::<AB::F>::eval(
            builder,
            shard.clone(),
            clk.clone(),
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            local.is_real.into(),
        );
        //address operation
        let addr_base = local.mem_ops.op_b_value();
        let addr_offset = local.mem_ops.op_c_value();
        AddressOperation::<AB::F>::eval(
            builder,
            *addr_base,
            *addr_offset,
            local.address_operation,
            local.is_real.into(),
        );
        // read memory
        builder.eval_memory_access(
            shard,
            clk,
            local.address_operation.aligned_address,
            &local.memory_access,
            local.is_real,
        );
        //selected byte
        let expected_unsigned_byte = local
            .memory_access
            .value()
            .0
            .iter()
            .zip(local.address_operation.offset_bit.iter())
            .fold(AB::Expr::zero(), |acc, (byte, bit)| acc + *byte * *bit);
        // if is not real, 0 equals 0
        builder.assert_eq(expected_unsigned_byte, local.selected_byte);

        //msb
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::MSB as u8),
            local.msb,
            local.selected_byte,
            AB::F::zero(),
            local.is_real.into(),
        );

        let padding_byte = local.msb * local.is_lb * AB::Expr::from_canonical_u8(0xff);
        let load_word: Word<AB::Expr> = Word([
            local.selected_byte.into(),
            padding_byte.clone(),
            padding_byte.clone(),
            padding_byte,
        ]);
        builder
            .when(local.is_real.into() - local.mem_ops.op_a_zero)
            .assert_word_eq(load_word, *local.mem_ops.op_a_value());
    }
}
