#![allow(clippy::needless_range_loop)]

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};
use crate::air::Block;
use crate::{
    builder::DTRecursionAirBuilder, runtime::ExecutionRecord, ExtExpReverseBitsInstr, Instruction,
};
use core::borrow::Borrow;
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::air::{
    BaseAirBuilder, BinomialExtension, DTAirBuilder, ExtensionAirBuilder, MachineAir,
};
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::borrow::BorrowMut;
use tracing::instrument;

pub const NUM_EXT_EXP_REVERSE_BITS_COLS: usize = size_of::<ExtExpReverseBitsCols<u8>>();
pub const NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS: usize =
    size_of::<ExtExpReverseBitsPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]

pub struct ExtExpReverseBitsChip<const DEGREE: usize>;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct ExtExpReverseBitsPreprocessedCols<T: Copy> {
    pub x_mem: MemoryAccessColsChips<T>,
    pub exponent_mem: MemoryAccessColsChips<T>,
    pub result_mem: MemoryAccessColsChips<T>,
    pub iteration_num: T,
    pub is_first: T,
    pub is_last: T,
    pub is_real: T,
    pub chain_accum_out: MemoryAccessColsChips<T>,
    pub chain_accum_in: MemoryAccessColsChips<T>,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ExtExpReverseBitsCols<T: Copy> {
    /// The base of the exponentiation.
    pub x: Block<T>,

    /// The current bit of the exponent. This is read from memory.
    pub current_bit: T,

    /// The previous accumulator squared.
    pub prev_accum_squared: Block<T>,

    /// Is set to the value local.prev_accum_squared * local.multiplier.
    pub prev_accum_squared_times_multiplier: Block<T>,

    /// The accumulator of the current iteration.
    pub accum: Block<T>,

    /// The accumulator squared.
    pub accum_squared: Block<T>,

    /// A column which equals x if `current_bit` is on, and 1 otherwise.
    pub multiplier: Block<T>,
}

impl<F, const DEGREE: usize> BaseAir<F> for ExtExpReverseBitsChip<DEGREE> {
    fn width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_COLS
    }
}

impl<F: Field, const DEGREE: usize> MachineAir<F> for ExtExpReverseBitsChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("ExtExpReverseBitsDeg{DEGREE}")
    }

    #[instrument(name = "generate ext exp reverse bits trace", level = "debug", skip_all, fields(rows = input.ext_exp_reverse_bits_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        let mut overall_rows = Vec::new();

        input.ext_exp_reverse_bits_events.iter().for_each(|event| {
            let mut rows = vec![vec![F::zero(); NUM_EXT_EXP_REVERSE_BITS_COLS]; event.exp.len()];
            let mut accum = Block::<F>::from(F::one());

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut ExtExpReverseBitsCols<F> = row.as_mut_slice().borrow_mut();

                cols.x = event.base;
                cols.current_bit = event.exp[i];
                let one_block = Block::<F>::from(F::one());
                if cols.current_bit == F::one() {
                    cols.multiplier = cols.x;
                } else {
                    cols.multiplier = one_block;
                }

                let prev_accum = accum;
                accum = (BinomialExtension(prev_accum.0)
                    * BinomialExtension(prev_accum.0)
                    * BinomialExtension(cols.multiplier.0))
                .0
                .into();

                cols.accum = accum;
                cols.accum_squared =
                    (BinomialExtension(accum.0) * BinomialExtension(accum.0)).0.into();
                cols.prev_accum_squared =
                    (BinomialExtension(prev_accum.0) * BinomialExtension(prev_accum.0)).0.into();
                cols.prev_accum_squared_times_multiplier =
                    (BinomialExtension(cols.prev_accum_squared.0)
                        * BinomialExtension(cols.multiplier.0))
                    .0
                    .into();
            });
            overall_rows.extend(rows);
        });

        let real_nb_rows = overall_rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, input.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_EXT_EXP_REVERSE_BITS_COLS,
        );

        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_EXT_EXP_REVERSE_BITS_COLS },
            total_height,
        )
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let mut rows: Vec<[F; NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ExtExpReverseBits(x) => Some(x),
                _ => None,
            })
            .for_each(|instruction: &ExtExpReverseBitsInstr<F>| {
                let ExtExpReverseBitsInstr { addrs, mult, chain_accum_addrs } = instruction;
                let num_bits = addrs.exp.len();
                let mut row_add =
                    vec![[F::zero(); NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS]; num_bits];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut ExtExpReverseBitsPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.iteration_num = F::from_canonical_u32(i as u32);
                    row.is_first = F::from_bool(i == 0);
                    row.is_last = F::from_bool(i == num_bits - 1);
                    row.is_real = F::one();
                    row.x_mem = MemoryAccessCols { addr: addrs.base, mult: -F::from_bool(i == 0) };
                    row.exponent_mem = MemoryAccessCols { addr: addrs.exp[i], mult: F::neg_one() };
                    row.result_mem = MemoryAccessCols {
                        addr: addrs.result,
                        mult: *mult * F::from_bool(i == num_bits - 1),
                    };
                    if i < num_bits - 1 {
                        row.chain_accum_out =
                            MemoryAccessCols { addr: chain_accum_addrs[i], mult: F::one() };
                    }
                    if i > 0 {
                        row.chain_accum_in =
                            MemoryAccessCols { addr: chain_accum_addrs[i - 1], mult: F::neg_one() };
                    }
                });
                rows.extend(row_add);
            });

        let real_nb_rows = rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, program.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_EXT_EXP_REVERSE_BITS_PREPROCESSED_COLS },
            total_height,
        ))
    }
}

impl<const DEGREE: usize> ExtExpReverseBitsChip<DEGREE> {
    pub fn eval_ext_exp_reverse_bits<
        AB: BaseAirBuilder + ExtensionAirBuilder + DTRecursionAirBuilder + DTAirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &ExtExpReverseBitsCols<AB::Var>,
        local_prepr: &ExtExpReverseBitsPreprocessedCols<AB::Var>,
    ) {
        // Dummy constraints to normalize to DEGREE when DEGREE > 3.
        if DEGREE > 3 {
            let lhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            let rhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            builder.assert_eq(lhs, rhs);
        }

        let local_accum = local.accum.as_extension::<AB>();
        let local_multiplier = local.multiplier.as_extension::<AB>();
        let local_prev_accum_squared_times_multiplier =
            local.prev_accum_squared_times_multiplier.as_extension::<AB>();
        let local_prev_accum_squared = local.prev_accum_squared.as_extension::<AB>();
        let local_accum_squared = local.accum_squared.as_extension::<AB>();

        // Read x from memory (only on first row).
        builder.send_block(local_prepr.x_mem.addr, local.x, local_prepr.x_mem.mult);

        // Read exponent bit from memory.
        builder.send_single(
            local_prepr.exponent_mem.addr,
            local.current_bit,
            local_prepr.exponent_mem.mult,
        );

        // On first row: accum = multiplier.
        builder
            .when(local_prepr.is_first)
            .assert_ext_eq(local_accum.clone(), local_multiplier.clone());

        // multiplier = 1 + bit*(x − 1).
        let bit: AB::Expr = local.current_bit.into();
        for k in 0..4 {
            let mul_k: AB::Expr = local.multiplier.0[k].into();
            let x_k: AB::Expr = local.x.0[k].into();
            let one_k = if k == 0 { AB::Expr::one() } else { AB::Expr::zero() };
            builder
                .when(local_prepr.is_real)
                .assert_eq(mul_k - one_k.clone(), bit.clone() * (x_k - one_k));
        }

        // prev_accum_squared_times_multiplier = prev_accum_squared * multiplier.
        builder.when(local_prepr.is_real).assert_ext_eq(
            local_prev_accum_squared_times_multiplier.clone(),
            local_prev_accum_squared.clone() * local_multiplier.clone(),
        );

        // On non-first rows: accum = prev_accum_squared * multiplier.
        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_first)
            .assert_ext_eq(local_accum.clone(), local_prev_accum_squared_times_multiplier.clone());

        // accum_squared = accum * accum.
        builder
            .when(local_prepr.is_real)
            .assert_ext_eq(local_accum_squared.clone(), local_accum.clone() * local_accum.clone());

        // Write result.
        builder.send_block(local_prepr.result_mem.addr, local.accum, local_prepr.result_mem.mult);

        // Chain interactions for accum_squared.
        builder.send_block(
            local_prepr.chain_accum_out.addr,
            local.accum_squared,
            local_prepr.chain_accum_out.mult,
        );
        builder.send_block(
            local_prepr.chain_accum_in.addr,
            local.prev_accum_squared,
            local_prepr.chain_accum_in.mult,
        );
    }
}

impl<AB, const DEGREE: usize> Air<AB> for ExtExpReverseBitsChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ExtExpReverseBitsCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &ExtExpReverseBitsPreprocessedCols<_> = (*prep_local).borrow();
        self.eval_ext_exp_reverse_bits::<AB>(builder, local, prep_local);
    }
}
