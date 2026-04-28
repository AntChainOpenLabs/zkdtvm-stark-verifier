#![allow(clippy::needless_range_loop)]

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};
use crate::air::BinomialExtensionUtils;
use crate::air::Block;
use crate::{
    builder::DTRecursionAirBuilder, runtime::ExecutionRecord, Instruction, PrefixSumChecksInstr,
};
use core::borrow::Borrow;
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::air::{
    BaseAirBuilder, BinomialExtension, DTAirBuilder, ExtensionAirBuilder, MachineAir,
};
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::AbstractField;
use p3_field::Field;
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
    pub x1_mem: MemoryAccessColsChips<T>,
    pub x2_mem: MemoryAccessColsChips<T>,
    pub acc_mem: MemoryAccessColsChips<T>,
    pub out_mem: MemoryAccessColsChips<T>,
    pub iteration_num: T,
    pub is_first: T,
    pub is_last: T,
    pub is_real: T,
    pub chain_acc_out: MemoryAccessColsChips<T>,
    pub chain_acc_in: MemoryAccessColsChips<T>,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PrefixSumChecksCols<F: Copy> {
    pub x1: Block<F>,
    pub x2: Block<F>,
    pub eq_val: Block<F>,
    pub x1_mul_x2: Block<F>,
    pub acc: Block<F>,
    pub new_acc: Block<F>,
    pub prev_new_acc: Block<F>,
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
            let mut acc = event.init_acc;
            let mut prev_new_acc = Block::<F>::default();

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut PrefixSumChecksCols<F> = row.as_mut_slice().borrow_mut();

                cols.x1 = event.x1_vec[i];
                cols.x2 = event.x2_vec[i];

                let x1_ext = BinomialExtension(cols.x1.0);
                let x2_ext = BinomialExtension(cols.x2.0);
                let one = BinomialExtension::from_base(F::one());

                let x1_mul_x2 = x1_ext * x2_ext;
                cols.x1_mul_x2 = x1_mul_x2.0.into();

                let eq_val = one - x1_ext - x2_ext + x1_mul_x2 + x1_mul_x2;
                cols.eq_val = eq_val.0.into();

                cols.prev_new_acc = prev_new_acc;

                let acc_ext = BinomialExtension(acc.0);
                cols.acc = acc;

                let new_acc = acc_ext * eq_val;
                cols.new_acc = new_acc.0.into();
                prev_new_acc = cols.new_acc;
                acc = cols.new_acc;
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
                let PrefixSumChecksInstr { addrs, mult, chain_acc_addrs } = instruction;
                let num_steps = addrs.x1_vec.len();
                let mut row_add =
                    vec![[F::zero(); NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS]; num_steps];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut PrefixSumChecksPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.iteration_num = F::from_canonical_u32(i as u32);
                    row.is_first = F::from_bool(i == 0);
                    row.is_last = F::from_bool(i == num_steps - 1);
                    row.is_real = F::one();
                    row.x1_mem = MemoryAccessCols { addr: addrs.x1_vec[i], mult: F::neg_one() };
                    row.x2_mem = MemoryAccessCols { addr: addrs.x2_vec[i], mult: F::neg_one() };
                    row.acc_mem =
                        MemoryAccessCols { addr: addrs.init_acc, mult: -F::from_bool(i == 0) };
                    row.out_mem = MemoryAccessCols {
                        addr: addrs.result,
                        mult: *mult * F::from_bool(i == num_steps - 1),
                    };
                    if i < num_steps - 1 {
                        row.chain_acc_out =
                            MemoryAccessCols { addr: chain_acc_addrs[i], mult: F::one() };
                    }
                    if i > 0 {
                        row.chain_acc_in =
                            MemoryAccessCols { addr: chain_acc_addrs[i - 1], mult: F::neg_one() };
                    }
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
        let one = BinomialExtension::<<AB as AirBuilder>::Expr>::from_block(Block([
            AB::Expr::one(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ]));
        let two = BinomialExtension::<<AB as AirBuilder>::Expr>::from_block(Block([
            AB::Expr::from_canonical_u32(2),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ]));
        let local_x1 = local.x1.as_extension::<AB>();
        let local_x2 = local.x2.as_extension::<AB>();
        let local_x1_mul_x2 = local.x1_mul_x2.as_extension::<AB>();
        let local_eq_val = local.eq_val.as_extension::<AB>();
        let local_acc = local.acc.as_extension::<AB>();
        let local_new_acc = local.new_acc.as_extension::<AB>();
        let prev_new_acc = local.prev_new_acc.as_extension::<AB>();

        builder.send_block(local_prepr.x1_mem.addr, local.x1, local_prepr.x1_mem.mult);
        builder.send_block(local_prepr.x2_mem.addr, local.x2, local_prepr.x2_mem.mult);

        builder.send_block(local_prepr.acc_mem.addr, local.acc, local_prepr.acc_mem.mult);

        builder
            .when(local_prepr.is_real)
            .assert_ext_eq(local_x1_mul_x2.clone(), local_x1.clone() * local_x2.clone());

        let expected_eq = one - local_x1 - local_x2 + two * local_x1_mul_x2;
        builder.when(local_prepr.is_real).assert_ext_eq(local_eq_val.clone(), expected_eq);

        builder
            .when(local_prepr.is_real)
            .assert_ext_eq(local_new_acc.clone(), local_acc.clone() * local_eq_val);

        // On non-first rows, constrain that acc = prev_new_acc (using chain).
        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_first)
            .assert_ext_eq(local_acc, prev_new_acc.clone());

        builder.send_block(local_prepr.out_mem.addr, local.new_acc, local_prepr.out_mem.mult);

        // Chain interactions for accumulator.
        builder.send_block(
            local_prepr.chain_acc_out.addr,
            local.new_acc,
            local_prepr.chain_acc_out.mult,
        );
        builder.send_block(
            local_prepr.chain_acc_in.addr,
            local.prev_new_acc,
            local_prepr.chain_acc_in.mult,
        );
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
