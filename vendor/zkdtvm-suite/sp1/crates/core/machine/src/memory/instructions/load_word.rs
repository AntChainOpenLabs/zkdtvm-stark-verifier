use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ExecutionRecord, ITypeRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
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
pub struct LoadWordChip;

pub const NUM_LOAD_WORD_COLUMNS: usize = size_of::<LoadWordCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadWordCols<T> {
    pub cpu_state: CPUState<T>,

    /// op_b <- rs1, op_c <- imm, op_a ->rd
    pub mem_ops: ITypeRegisterOp<T>,
    // op_b + op_c
    pub address_operation: AddressOperation<T>,
    /// memory read
    pub memory_access: MemoryReadCols<T>,

    pub is_real: T,
}

impl<F> BaseAir<F> for LoadWordChip {
    fn width(&self) -> usize {
        NUM_LOAD_WORD_COLUMNS
    }
}

impl<F: Field> MachineAir<F> for LoadWordChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadWord".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.load_word_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(nb_rows * NUM_LOAD_WORD_COLUMNS);
        let shard = input.execution_shard();
        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_WORD_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_WORD_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadWordCols<F> = row.borrow_mut();
                    let (record, event) = &input.load_word_events[idx];
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_LOAD_WORD_COLUMNS);
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_LOAD_WORD_COLUMNS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.load_word_events.is_empty()
        }
    }
}

impl LoadWordChip {
    pub(crate) fn event_to_row<F: Field>(
        &self,
        record: &ITypeRecord,
        event: &MemInstrEvent,
        cols: &mut LoadWordCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);
        // Populate memory accesses for reading from memory.
        cols.memory_access
            .populate(event.mem_access.read_record().expect("load event reads from memory"), blu);

        cols.address_operation.populate(blu, event.b, event.c);

        cols.is_real = F::from_bool(true);
    }
}

impl<AB> Air<AB> for LoadWordChip
where
    AB: DTCoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadWordCols<AB::Var> = (*local).borrow();

        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
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
        // register op + program lookup
        let opcode = AB::F::from_canonical_u8(Opcode::LW as u8);
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
        builder.assert_eq(local.is_real, local.address_operation.offset_bit[0]);
        let load_word = local.memory_access.value();
        builder
            .when(local.is_real.into() - local.mem_ops.op_a_zero)
            .assert_word_eq(*load_word, *local.mem_ops.op_a_value());
    }
}
