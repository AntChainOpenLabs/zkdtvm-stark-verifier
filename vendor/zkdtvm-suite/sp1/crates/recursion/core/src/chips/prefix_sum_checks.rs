#![allow(clippy::needless_range_loop)]

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};
use crate::{
    air::Block, builder::DTRecursionAirBuilder, runtime::ExecutionRecord, Address, Instruction,
    PrefixSumChecksInstr,
};
use core::borrow::Borrow;
use dt_core_machine::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{BaseAirBuilder, ChallengeExtension, DTAirBuilder, ExtensionAirBuilder, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::borrow::BorrowMut;
use tracing::instrument;

pub const NUM_PREFIX_SUM_CHECKS_COLS: usize = size_of::<PrefixSumChecksCols<u8>>();
pub const NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS: usize =
    size_of::<PrefixSumChecksPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct PrefixSumChecksChip;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct PrefixSumChecksPreprocessedCols<T: Copy> {
    pub x1_addr: Address<T>,
    pub x2_addr: Address<T>,
    pub prev_acc_addr: Address<T>,
    pub acc_mem: MemoryAccessColsChips<T>,
    pub is_real: T,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PrefixSumChecksCols<F: Copy> {
    pub x1: Block<F>,
    pub x2: Block<F>,
    pub eq_val: Block<F>,
    pub prev_acc: Block<F>,
    pub acc: Block<F>,
}

impl<F> BaseAir<F> for PrefixSumChecksChip {
    fn width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_COLS
    }
}

impl<F: Field> MachineAir<F> for PrefixSumChecksChip {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "PrefixSumChecks".to_string()
    }

    #[instrument(name = "generate prefix sum checks trace", level = "debug", skip_all,
        fields(rows = input.prefix_sum_checks_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        let mut overall_rows = Vec::new();

        input.prefix_sum_checks_events.iter().for_each(|event| {
            let num_steps = event.x1_vec.len();
            let mut rows = vec![vec![F::zero(); NUM_PREFIX_SUM_CHECKS_COLS]; num_steps];

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut PrefixSumChecksCols<F> = row.as_mut_slice().borrow_mut();

                cols.x1 = event.x1_vec[i];
                cols.x2 = event.x2_vec[i];
                cols.prev_acc = event.prev_acc_vec[i];
                cols.acc = event.acc_vec[i];

                let x1_ext = ChallengeExtension(cols.x1.0);
                let x2_ext = ChallengeExtension(cols.x2.0);
                let one = ChallengeExtension::from_base(F::one());

                let eq_val = one - x1_ext - x2_ext + x1_ext * x2_ext + x1_ext * x2_ext;
                cols.eq_val = eq_val.0.into();
            });
            overall_rows.extend(rows);
        });

        let real_nb_rows = overall_rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, input.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_PREFIX_SUM_CHECKS_COLS,
        );

        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_PREFIX_SUM_CHECKS_COLS },
            total_height,
        )
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let mut rows: Vec<[F; NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::PrefixSumChecks(x) => Some(x.as_ref()),
                _ => None,
            })
            .for_each(|instruction: &PrefixSumChecksInstr<F>| {
                let PrefixSumChecksInstr { addrs, mult } = instruction;
                let num_steps = addrs.x1_vec.len();
                let mut row_add =
                    vec![[F::zero(); NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS]; num_steps];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut PrefixSumChecksPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.is_real = F::one();
                    row.x1_addr = addrs.x1_vec[i];
                    row.x2_addr = addrs.x2_vec[i];
                    row.prev_acc_addr = addrs.prev_acc_vec[i];
                    row.acc_mem = MemoryAccessCols {
                        addr: addrs.acc_vec[i],
                        mult: if i == num_steps - 1 { *mult } else { F::one() },
                    };
                });
                rows.extend(row_add);
            });

        let real_nb_rows = rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, program.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS },
            total_height,
        ))
    }
}

impl PrefixSumChecksChip {
    pub fn eval_prefix_sum_checks<
        AB: BaseAirBuilder + ExtensionAirBuilder + DTRecursionAirBuilder + DTAirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &PrefixSumChecksCols<AB::Var>,
        local_prepr: &PrefixSumChecksPreprocessedCols<AB::Var>,
    ) {
        let one = ChallengeExtension::<<AB as AirBuilder>::Expr>::from_base(AB::Expr::one());
        let two = ChallengeExtension::<<AB as AirBuilder>::Expr>::from_base(
            AB::Expr::from_canonical_u32(2),
        );
        let local_x1 = local.x1.as_extension::<AB>();
        let local_x2 = local.x2.as_extension::<AB>();
        let local_eq_val = local.eq_val.as_extension::<AB>();
        let local_acc = local.acc.as_extension::<AB>();
        let prev_acc = local.prev_acc.as_extension::<AB>();

        // Receive x1 and x2 from memory.
        builder.receive_block(local_prepr.x1_addr, local.x1, local_prepr.is_real);
        builder.receive_block(local_prepr.x2_addr, local.x2, local_prepr.is_real);

        // Receive prev_acc from memory.
        builder.receive_block(local_prepr.prev_acc_addr, local.prev_acc, local_prepr.is_real);

        // eq_val = 1 - x1 - x2 + 2·x1·x2
        let expected_eq = one - local_x1.clone() - local_x2.clone() + two * local_x1 * local_x2;
        builder.when(local_prepr.is_real).assert_ext_eq(local_eq_val.clone(), expected_eq);

        // acc = prev_acc * eq_val
        builder.when(local_prepr.is_real).assert_ext_eq(local_acc, prev_acc * local_eq_val);

        // Send acc to memory.
        builder.send_block(local_prepr.acc_mem.addr, local.acc, local_prepr.acc_mem.mult);
    }
}

impl<AB> Air<AB> for PrefixSumChecksChip
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PrefixSumChecksCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &PrefixSumChecksPreprocessedCols<_> = (*prep_local).borrow();
        self.eval_prefix_sum_checks::<AB>(builder, local, prep_local);
    }
}
