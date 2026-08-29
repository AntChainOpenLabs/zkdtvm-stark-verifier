use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, JumpEvent},
    ExecutionRecord, JTypeRecord, Opcode, Program, DEFAULT_PC_INC,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::MachineAir,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    Word,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapter::{CPUState, JTypeRegisterOp},
    air::DTCoreAirBuilder,
    operations::{AddOperation, BabyBearWordRangeChecker},
    utils::{next_power_of_two, padded_rows_threshold, zeroed_f_vec},
};
pub const NUM_JAL_COLS: usize = size_of::<JalCols<u8>>();
#[derive(Default)]
pub struct JalChip;

impl<F> BaseAir<F> for JalChip {
    fn width(&self) -> usize {
        NUM_JAL_COLS
    }
}
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JalCols<T> {
    /// state
    pub cpu_state: CPUState<T>,
    /// j type register op
    pub mem_ops: JTypeRegisterOp<T>,
    /// add op: pc + op_b
    pub add_op: AddOperation<T>,
    /// pc word
    pub pc: Word<T>,
    /// BabyBear range checker for the program counter.
    pub pc_range_checker: BabyBearWordRangeChecker<T>,
    ///op_a range checker
    pub op_a_range_checker: BabyBearWordRangeChecker<T>,
    /// next_pc_range_checker
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,
    /// Whether this is one real row
    pub is_real: T,
}
impl<F: Field> MachineAir<F> for JalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Jal".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let nb_rows = input.jal_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 1);
        let mut values = zeroed_f_vec(nb_rows * NUM_JAL_COLS);
        let shard = input.execution_shard();
        let blu_events = values
            .chunks_mut(chunk_size * NUM_JAL_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_JAL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut JalCols<F> = row.borrow_mut();
                    let (record, event) = &input.jal_events[idx];
                    self.event_to_row(record, event, cols, &mut blu, shard);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        let main = RowMajorMatrix::new(values, NUM_JAL_COLS);
        CompressedMatrix::new(main, PaddingRow::Zero { width: NUM_JAL_COLS }, padded_nb_rows)
    }
    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.jal_events.is_empty()
        }
    }
    fn local_only(&self) -> bool {
        true
    }
}

impl JalChip {
    fn event_to_row<F: Field>(
        &self,
        record: &JTypeRecord,
        event: &JumpEvent,
        cols: &mut JalCols<F>,
        blu: &mut impl ByteRecord,
        shard: u32,
    ) {
        cols.cpu_state.populate(blu, record.clk, event.pc, shard);
        cols.mem_ops.populate(blu, *record);
        cols.add_op.populate(blu, event.b, event.pc);
        cols.pc = Word::from(event.pc);
        cols.pc_range_checker.populate(cols.pc, blu);
        cols.op_a_range_checker.populate(*cols.mem_ops.op_a_value(), blu);
        cols.next_pc_range_checker.populate(cols.add_op.value, blu);
        cols.is_real = F::one();
    }
}

impl<AB> Air<AB> for JalChip
where
    AB: DTCoreAirBuilder,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JalCols<AB::Var> = (*local).borrow();
        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let a_word = local.mem_ops.op_a_value();
        let b_word = local.mem_ops.op_b_value();
        // SAFETY: All selectors `is_jal`, `is_jalr` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_jal + is_jalr` is
        // boolean. Therefore, the `opcode` matches the corresponding opcode.

        builder.assert_bool(local.is_real);
        //cpu state
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.add_op.value.reduce::<AB>(),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
            execution_shard,
        );
        let opcode = Opcode::JAL.as_field::<AB::F>();
        JTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            local.is_real.into(),
        );
        //if op_a_zero,then is_real,avoid 0 - 1 case for is_real - op_a_zero
        builder.when(local.mem_ops.op_a_zero).assert_one(local.is_real);
        builder.when(local.is_real - local.mem_ops.op_a_zero).assert_eq(
            a_word.reduce::<AB>(),
            //write curr_pc + 4 to op_a
            local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
        );
        builder.assert_eq(local.cpu_state.pc, local.pc.reduce::<AB>());
        //constraint next_pc via add op
        AddOperation::<AB::F>::eval(builder, *b_word, local.pc, local.add_op, local.is_real.into());

        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            *a_word,
            local.op_a_range_checker,
            local.is_real.into(),
        );

        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.pc,
            local.pc_range_checker,
            local.is_real.into(),
        );
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.add_op.value,
            local.next_pc_range_checker,
            local.is_real.into(),
        );
    }
}
