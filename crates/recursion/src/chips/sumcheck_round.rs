#![allow(clippy::needless_range_loop)]

use super::mem::{MemoryAccessCols, MemoryAccessColsChips};
use crate::air::Block;
use crate::{
    builder::DTRecursionAirBuilder, runtime::ExecutionRecord, Instruction, SumcheckRoundInstr,
};
use core::borrow::Borrow;
use crate::utils::{next_power_of_two, padded_rows_threshold};
use dt_derive::AlignedBorrow;
use dt_stark::air::{
    BaseAirBuilder, BinomialExtension, DTAirBuilder, ExtensionAirBuilder, MachineAir,
};
use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use std::borrow::BorrowMut;
use tracing::instrument;

pub const NUM_SUMCHECK_ROUND_COLS: usize = size_of::<SumcheckRoundCols<u8>>();
pub const NUM_SUMCHECK_ROUND_PREPROCESS_COLS: usize =
    size_of::<SumcheckRoundPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct SumcheckRoundChip;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct SumcheckRoundPreprocessedCols<T: Copy> {
    pub challenge_mem: MemoryAccessColsChips<T>,
    pub coeff_mem: MemoryAccessColsChips<T>,
    pub claim_mem: MemoryAccessColsChips<T>,
    pub out_mem: MemoryAccessColsChips<T>,
    pub iteration_num: T,
    pub is_first: T,
    pub is_last: T,
    pub is_real: T,
    pub is_claim_check: T,
    pub chain_rs_out: MemoryAccessColsChips<T>,
    pub chain_rs_in: MemoryAccessColsChips<T>,
    pub chain_ha_out: MemoryAccessColsChips<T>,
    pub chain_ha_in: MemoryAccessColsChips<T>,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct SumcheckRoundCols<F: Copy> {
    pub challenge: Block<F>,
    pub current_coeff: Block<F>,
    pub running_sum: Block<F>,
    pub claim: Block<F>,
    pub horner_accum: Block<F>,
    pub horner_accum_mul_challenge: Block<F>,
    pub prev_running_sum: Block<F>,
    pub prev_horner_mul_challenge: Block<F>,
}

impl<F> BaseAir<F> for SumcheckRoundChip {
    fn width(&self) -> usize {
        NUM_SUMCHECK_ROUND_COLS
    }
}

impl<F: Field> MachineAir<F> for SumcheckRoundChip {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "SumcheckRound".to_string()
    }

    #[instrument(name = "generate sumcheck round trace", level = "debug", skip_all,
        fields(rows = input.sumcheck_round_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> CompressedMatrix<F> {
        let mut overall_rows = Vec::new();

        input.sumcheck_round_events.iter().for_each(|event| {
            let num_coeffs = event.coeffs.len();
            let mut rows = vec![vec![F::zero(); NUM_SUMCHECK_ROUND_COLS]; num_coeffs];
            let mut running_sum = Block::<F>::default();
            let mut horner_accum = Block::<F>::default();
            let mut prev_horner_mul_challenge = Block::<F>::default();
            let mut prev_running_sum = Block::<F>::default();

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut SumcheckRoundCols<F> = row.as_mut_slice().borrow_mut();

                cols.challenge = event.challenge;
                cols.current_coeff = event.coeffs[i];
                cols.claim = event.claim;

                cols.prev_running_sum = prev_running_sum;
                cols.prev_horner_mul_challenge = prev_horner_mul_challenge;

                if i == 0 {
                    running_sum = cols.current_coeff;
                    horner_accum = cols.current_coeff;
                } else {
                    running_sum = (BinomialExtension(prev_running_sum.0)
                        + BinomialExtension(cols.current_coeff.0))
                    .0
                    .into();
                    horner_accum = (BinomialExtension(prev_horner_mul_challenge.0)
                        + BinomialExtension(cols.current_coeff.0))
                    .0
                    .into();
                }

                cols.running_sum = running_sum;
                cols.horner_accum = horner_accum;
                cols.horner_accum_mul_challenge = (BinomialExtension(horner_accum.0)
                    * BinomialExtension(cols.challenge.0))
                .0
                .into();
                prev_horner_mul_challenge = cols.horner_accum_mul_challenge;
                prev_running_sum = running_sum;
            });
            overall_rows.extend(rows);
        });

        let real_nb_rows = overall_rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, input.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_SUMCHECK_ROUND_COLS,
        );

        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SUMCHECK_ROUND_COLS },
            total_height,
        )
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        NUM_SUMCHECK_ROUND_PREPROCESS_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        let mut rows: Vec<[F; NUM_SUMCHECK_ROUND_PREPROCESS_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::SumcheckRound(x) => Some(x.as_ref()),
                _ => None,
            })
            .for_each(|instruction: &SumcheckRoundInstr<F>| {
                let SumcheckRoundInstr { addrs, mult, chain_rs_addrs, chain_ha_addrs } =
                    instruction;
                let num_coeffs = addrs.coeffs.len();
                let mut row_add = vec![[F::zero(); NUM_SUMCHECK_ROUND_PREPROCESS_COLS]; num_coeffs];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut SumcheckRoundPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.iteration_num = F::from_canonical_u32(i as u32);
                    row.is_first = F::from_bool(i == 0);
                    row.is_last = F::from_bool(i == num_coeffs - 1);
                    row.is_real = F::one();
                    row.is_claim_check = F::from_bool(i == num_coeffs - 1);
                    row.challenge_mem =
                        MemoryAccessCols { addr: addrs.challenge, mult: -F::from_bool(i == 0) };
                    row.coeff_mem = MemoryAccessCols { addr: addrs.coeffs[i], mult: F::neg_one() };
                    row.claim_mem =
                        MemoryAccessCols { addr: addrs.claim, mult: -F::from_bool(i == 0) };
                    row.out_mem = MemoryAccessCols {
                        addr: addrs.new_claim,
                        mult: *mult * F::from_bool(i == num_coeffs - 1),
                    };
                    if i < num_coeffs - 1 {
                        row.chain_rs_out =
                            MemoryAccessCols { addr: chain_rs_addrs[i], mult: F::one() };
                    }
                    if i > 0 {
                        row.chain_rs_in =
                            MemoryAccessCols { addr: chain_rs_addrs[i - 1], mult: F::neg_one() };
                    }
                    if i < num_coeffs - 1 {
                        row.chain_ha_out =
                            MemoryAccessCols { addr: chain_ha_addrs[i], mult: F::one() };
                    }
                    if i > 0 {
                        row.chain_ha_in =
                            MemoryAccessCols { addr: chain_ha_addrs[i - 1], mult: F::neg_one() };
                    }
                });
                rows.extend(row_add);
            });

        let real_nb_rows = rows.len();
        let total_height =
            padded_rows_threshold(next_power_of_two(real_nb_rows, program.fixed_log2_rows(self)));

        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<F>>(),
            NUM_SUMCHECK_ROUND_PREPROCESS_COLS,
        );
        Some(CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_SUMCHECK_ROUND_PREPROCESS_COLS },
            total_height,
        ))
    }
}

impl SumcheckRoundChip {
    pub fn eval_sumcheck_round<
        AB: BaseAirBuilder + ExtensionAirBuilder + DTRecursionAirBuilder + DTAirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &SumcheckRoundCols<AB::Var>,
        local_prepr: &SumcheckRoundPreprocessedCols<AB::Var>,
    ) {
        let local_challenge = local.challenge.as_extension::<AB>();
        let local_coeff = local.current_coeff.as_extension::<AB>();
        let local_running_sum = local.running_sum.as_extension::<AB>();
        let local_horner = local.horner_accum.as_extension::<AB>();
        let local_horner_mul = local.horner_accum_mul_challenge.as_extension::<AB>();
        let local_claim = local.claim.as_extension::<AB>();
        let prev_running_sum = local.prev_running_sum.as_extension::<AB>();
        let prev_horner_mul = local.prev_horner_mul_challenge.as_extension::<AB>();

        // Read challenge from memory (only on first row).
        builder.send_block(
            local_prepr.challenge_mem.addr,
            local.challenge,
            local_prepr.challenge_mem.mult,
        );

        // Read claim from memory (only on first row).
        builder.send_block(local_prepr.claim_mem.addr, local.claim, local_prepr.claim_mem.mult);

        // Read coefficient from memory.
        builder.send_block(
            local_prepr.coeff_mem.addr,
            local.current_coeff,
            local_prepr.coeff_mem.mult,
        );

        // On first row: running_sum = coeff, horner = coeff.
        builder
            .when(local_prepr.is_first)
            .assert_ext_eq(local_running_sum.clone(), local_coeff.clone());

        // On non-first row: running_sum = prev_running_sum + coeff.
        builder.when(local_prepr.is_real).when_not(local_prepr.is_first).assert_ext_eq(
            local_running_sum.clone(),
            prev_running_sum.clone() + local_coeff.clone(),
        );

        builder.when(local_prepr.is_first).assert_ext_eq(local_horner.clone(), local_coeff.clone());

        // On non-first row: horner = prev_horner_mul_challenge + coeff.
        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_first)
            .assert_ext_eq(local_horner.clone(), prev_horner_mul.clone() + local_coeff.clone());

        // horner_mul_challenge = horner * challenge (always).
        builder
            .when(local_prepr.is_real)
            .assert_ext_eq(local_horner_mul.clone(), local_horner.clone() * local_challenge);

        // Claim check: claim = coeff + running_sum (on last row).
        builder
            .when(local_prepr.is_claim_check)
            .assert_ext_eq(local_claim, local_coeff + local_running_sum);

        // Write result.
        builder.send_block(local_prepr.out_mem.addr, local.horner_accum, local_prepr.out_mem.mult);

        // Chain interactions for running_sum.
        builder.send_block(
            local_prepr.chain_rs_out.addr,
            local.running_sum,
            local_prepr.chain_rs_out.mult,
        );
        builder.send_block(
            local_prepr.chain_rs_in.addr,
            local.prev_running_sum,
            local_prepr.chain_rs_in.mult,
        );

        // Chain interactions for horner accumulator.
        builder.send_block(
            local_prepr.chain_ha_out.addr,
            local.horner_accum_mul_challenge,
            local_prepr.chain_ha_out.mult,
        );
        builder.send_block(
            local_prepr.chain_ha_in.addr,
            local.prev_horner_mul_challenge,
            local_prepr.chain_ha_in.mult,
        );
    }
}

impl<AB> Air<AB> for SumcheckRoundChip
where
    AB: DTRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SumcheckRoundCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &SumcheckRoundPreprocessedCols<_> = (*prep_local).borrow();
        self.eval_sumcheck_round::<AB>(builder, local, prep_local);
    }
}
