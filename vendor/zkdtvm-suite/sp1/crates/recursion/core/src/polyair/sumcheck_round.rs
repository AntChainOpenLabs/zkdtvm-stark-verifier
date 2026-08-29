use crate::{
    chips::sumcheck_round::{
        SumcheckRoundChip, NUM_SUMCHECK_ROUND_COLS, NUM_SUMCHECK_ROUND_PREPROCESS_COLS,
    },
    *,
};
use dt_stark::{
    air::{ChallengeExtension, FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::ops::Deref;

// Preprocessed column indices (SumcheckRoundPreprocessedCols layout):
//   challenge_mem(addr,mult), coeff_mem(addr,mult), claim_mem(addr,mult), out_mem(addr,mult),
//   iteration_num, is_first, is_last, is_real, is_claim_check,
//   chain_rs_out(addr,mult), chain_rs_in(addr,mult), chain_ha_out(addr,mult),
// chain_ha_in(addr,mult)
const PREP_CHALLENGE_ADDR: usize = 0;
const PREP_CHALLENGE_MULT: usize = 1;
const PREP_COEFF_ADDR: usize = 2;
const PREP_COEFF_MULT: usize = 3;
const PREP_CLAIM_ADDR: usize = 4;
const PREP_CLAIM_MULT: usize = 5;
const PREP_OUT_ADDR: usize = 6;
const PREP_OUT_MULT: usize = 7;
const PREP_IS_FIRST: usize = 9;
const PREP_IS_REAL: usize = 11;
const PREP_IS_CLAIM_CHECK: usize = 12;
const PREP_CHAIN_RS_OUT_ADDR: usize = 13;
const PREP_CHAIN_RS_OUT_MULT: usize = 14;
const PREP_CHAIN_RS_IN_ADDR: usize = 15;
const PREP_CHAIN_RS_IN_MULT: usize = 16;
const PREP_CHAIN_HA_OUT_ADDR: usize = 17;
const PREP_CHAIN_HA_OUT_MULT: usize = 18;
const PREP_CHAIN_HA_IN_ADDR: usize = 19;
const PREP_CHAIN_HA_IN_MULT: usize = 20;

// Main trace column offsets (SumcheckRoundCols layout):
//   challenge[D], current_coeff[D], running_sum[D], claim[D],
//   horner_accum[D], horner_accum_mul_challenge[D], prev_running_sum[D],
// prev_horner_mul_challenge[D] Each Block<F> occupies D fields.
fn main_challenge(_d: usize) -> usize {
    0
}
fn main_coeff(d: usize) -> usize {
    d
}
fn main_running_sum(d: usize) -> usize {
    2 * d
}
fn main_claim(d: usize) -> usize {
    3 * d
}
fn main_horner_accum(d: usize) -> usize {
    4 * d
}
fn main_horner_mul_challenge(d: usize) -> usize {
    5 * d
}
fn main_prev_running_sum(d: usize) -> usize {
    6 * d
}
fn main_prev_horner_mul_challenge(d: usize) -> usize {
    7 * d
}

// Reserved poly layout:
//   [0 .. 8*D): all main columns
//   [8*D .. 8*D+3): prep flags: is_first, is_real, is_claim_check
//   [8*D+3 .. 8*D+11): prep mults (8 total, one per lookup)
fn reserved_is_first(d: usize) -> usize {
    8 * d
}
fn reserved_is_real(d: usize) -> usize {
    8 * d + 1
}
fn reserved_is_claim_check(d: usize) -> usize {
    8 * d + 2
}
fn reserved_mults_start(d: usize) -> usize {
    8 * d + 3
}

#[derive(Default, Clone, Copy)]
pub struct SumcheckRoundChipPolyAir;

fn ext_poly_mul<AB: FullAirBuilder>(
    a: &[AB::VarMaybeExt],
    b: &[AB::VarMaybeExt],
) -> Vec<AB::VarMaybeExt> {
    let a_ext = ChallengeExtension(core::array::from_fn(|i| a[i].clone()));
    let b_ext = ChallengeExtension(core::array::from_fn(|i| b[i].clone()));
    (a_ext * b_ext).0.to_vec()
}

impl<AB: FullAirBuilder> FullAir<AB> for SumcheckRoundChipPolyAir {
    fn width(&self) -> usize {
        NUM_SUMCHECK_ROUND_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let d = runtime::D;
        let mut cols = Vec::new();

        // All 8*D main trace columns
        for i in 0..(8 * d) {
            cols.push(PairCol::Main(i));
        }

        // Prep flags: is_first, is_real, is_claim_check
        cols.push(PairCol::Prep(PREP_IS_FIRST));
        cols.push(PairCol::Prep(PREP_IS_REAL));
        cols.push(PairCol::Prep(PREP_IS_CLAIM_CHECK));

        // Prep mults (8 total, matching precompute_lc order)
        cols.push(PairCol::Prep(PREP_CHALLENGE_MULT));
        cols.push(PairCol::Prep(PREP_CLAIM_MULT));
        cols.push(PairCol::Prep(PREP_COEFF_MULT));
        cols.push(PairCol::Prep(PREP_OUT_MULT));
        cols.push(PairCol::Prep(PREP_CHAIN_RS_OUT_MULT));
        cols.push(PairCol::Prep(PREP_CHAIN_RS_IN_MULT));
        cols.push(PairCol::Prep(PREP_CHAIN_HA_OUT_MULT));
        cols.push(PairCol::Prep(PREP_CHAIN_HA_IN_MULT));

        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        // Pre-build all 8 interaction value vectors to avoid borrow conflicts.
        let interactions: Vec<Vec<AB::VarMaybeExt>> = [
            (PREP_CHALLENGE_ADDR, main_challenge(d)),
            (PREP_CLAIM_ADDR, main_claim(d)),
            (PREP_COEFF_ADDR, main_coeff(d)),
            (PREP_OUT_ADDR, main_horner_accum(d)),
            (PREP_CHAIN_RS_OUT_ADDR, main_running_sum(d)),
            (PREP_CHAIN_RS_IN_ADDR, main_prev_running_sum(d)),
            (PREP_CHAIN_HA_OUT_ADDR, main_horner_mul_challenge(d)),
            (PREP_CHAIN_HA_IN_ADDR, main_prev_horner_mul_challenge(d)),
        ]
        .iter()
        .map(|&(addr_idx, block_start)| {
            let mut vals = vec![prep[addr_idx].clone()];
            for i in 0..d {
                vals.push(main[block_start + i].clone());
            }
            vals
        })
        .collect();

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let challenge: Vec<_> = (0..d).map(|i| local[main_challenge(d) + i].clone()).collect();
        let coeff: Vec<_> = (0..d).map(|i| local[main_coeff(d) + i].clone()).collect();
        let running_sum: Vec<_> = (0..d).map(|i| local[main_running_sum(d) + i].clone()).collect();
        let claim: Vec<_> = (0..d).map(|i| local[main_claim(d) + i].clone()).collect();
        let horner_accum: Vec<_> =
            (0..d).map(|i| local[main_horner_accum(d) + i].clone()).collect();
        let horner_mul_chal: Vec<_> =
            (0..d).map(|i| local[main_horner_mul_challenge(d) + i].clone()).collect();
        let prev_rs: Vec<_> = (0..d).map(|i| local[main_prev_running_sum(d) + i].clone()).collect();
        let prev_hmc: Vec<_> =
            (0..d).map(|i| local[main_prev_horner_mul_challenge(d) + i].clone()).collect();

        let is_first = local[reserved_is_first(d)].clone();
        let is_real = local[reserved_is_real(d)].clone();
        let is_claim_check = local[reserved_is_claim_check(d)].clone();

        let one = AB::VarMaybeExt::one();
        let is_real_not_first = is_real.clone() * (one - is_first.clone());

        // Constraint 1: when(is_first): running_sum = coeff
        for i in 0..d {
            builder.when(is_first.clone()).assert_zero(running_sum[i].clone() - coeff[i].clone());
        }

        // Constraint 2: when(is_real && !is_first): running_sum = prev_running_sum + coeff
        for i in 0..d {
            builder
                .when(is_real_not_first.clone())
                .assert_zero(running_sum[i].clone() - prev_rs[i].clone() - coeff[i].clone());
        }

        // Constraint 3: when(is_first): horner_accum = coeff
        for i in 0..d {
            builder.when(is_first.clone()).assert_zero(horner_accum[i].clone() - coeff[i].clone());
        }

        // Constraint 4: when(is_real && !is_first): horner_accum = prev_horner_mul_challenge +
        // coeff
        for i in 0..d {
            builder
                .when(is_real_not_first.clone())
                .assert_zero(horner_accum[i].clone() - prev_hmc[i].clone() - coeff[i].clone());
        }

        // Constraint 5: when(is_real): horner_mul_challenge = horner_accum * challenge
        let expected_mul = ext_poly_mul::<AB>(&horner_accum, &challenge);
        for i in 0..d {
            builder
                .when(is_real.clone())
                .assert_zero(horner_mul_chal[i].clone() - expected_mul[i].clone());
        }

        // Constraint 6: when(is_claim_check): claim = coeff + running_sum
        for i in 0..d {
            builder
                .when(is_claim_check.clone())
                .assert_zero(claim[i].clone() - coeff[i].clone() - running_sum[i].clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let mults_start = reserved_mults_start(d);

        // 8 lookups matching precompute_lc order.
        // Mult sign determines send vs recv (positive = send, negative = recv).
        for i in 0..8 {
            builder.send(local[mults_start + i].clone());
        }
    }
}

impl<F: Field> BaseAir<F> for SumcheckRoundChipPolyAir {
    fn width(&self) -> usize {
        NUM_SUMCHECK_ROUND_COLS
    }
}

impl<F: Field> MachineAir<F> for SumcheckRoundChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "SumcheckRound".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_SUMCHECK_ROUND_PREPROCESS_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        SumcheckRoundChip.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        SumcheckRoundChip.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        SumcheckRoundChip.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        SumcheckRoundChip.generate_trace(input, output)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_analysis() {
        use ::polyair::Chip;

        #[cfg(feature = "koalabear")]
        {
            let chip = Chip::<SumcheckRoundChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                SumcheckRoundChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 8);
            println!(
                "SumcheckRoundChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<SumcheckRoundChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
                SumcheckRoundChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 8);
            println!(
                "SumcheckRoundChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
