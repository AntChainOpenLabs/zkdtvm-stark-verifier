use crate::{
    chips::prefix_sum_checks::{
        PrefixSumChecksChip, NUM_PREFIX_SUM_CHECKS_COLS, NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS,
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

#[derive(Default, Clone, Copy)]
pub struct PrefixSumChecksChipPolyAir;

// Preprocessed column indices (PrefixSumChecksPreprocessedCols layout):
//   x1_addr[0], x2_addr[1], prev_acc_addr[2], acc_mem.addr[3], acc_mem.mult[4], is_real[5]
const PREP_X1_ADDR: usize = 0;
const PREP_X2_ADDR: usize = 1;
const PREP_PREV_ACC_ADDR: usize = 2;
const PREP_ACC_ADDR: usize = 3;
const PREP_ACC_MULT: usize = 4;
const PREP_IS_REAL: usize = 5;

// Main trace column offsets (PrefixSumChecksCols layout):
//   x1[0..D], x2[D..2D], eq_val[2D..3D], prev_acc[3D..4D], acc[4D..5D]
fn main_x1(_d: usize) -> usize {
    0
}
fn main_x2(d: usize) -> usize {
    d
}
fn main_eq_val(d: usize) -> usize {
    2 * d
}
fn main_prev_acc(d: usize) -> usize {
    3 * d
}
fn main_acc(d: usize) -> usize {
    4 * d
}

// Reserved poly layout:
//   [0 .. 5*D): all main columns
//   [5*D]: prep acc_mult
//   [5*D+1]: prep is_real
fn reserved_acc_mult(d: usize) -> usize {
    5 * d
}
fn reserved_is_real(d: usize) -> usize {
    5 * d + 1
}

impl<AB: FullAirBuilder> FullAir<AB> for PrefixSumChecksChipPolyAir {
    fn width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();

        // All 5*D main trace columns
        for i in 0..NUM_PREFIX_SUM_CHECKS_COLS {
            cols.push(PairCol::Main(i));
        }

        // Prep: acc_mult, is_real
        cols.push(PairCol::Prep(PREP_ACC_MULT));
        cols.push(PairCol::Prep(PREP_IS_REAL));

        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        // Build interaction value vectors for 4 memory lookups.
        let interactions: Vec<Vec<AB::VarMaybeExt>> = [
            (PREP_X1_ADDR, main_x1(d)),
            (PREP_X2_ADDR, main_x2(d)),
            (PREP_PREV_ACC_ADDR, main_prev_acc(d)),
            (PREP_ACC_ADDR, main_acc(d)),
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

        let x1: Vec<_> = (0..d).map(|i| local[main_x1(d) + i].clone()).collect();
        let x2: Vec<_> = (0..d).map(|i| local[main_x2(d) + i].clone()).collect();
        let eq_val: Vec<_> = (0..d).map(|i| local[main_eq_val(d) + i].clone()).collect();
        let prev_acc: Vec<_> = (0..d).map(|i| local[main_prev_acc(d) + i].clone()).collect();
        let acc: Vec<_> = (0..d).map(|i| local[main_acc(d) + i].clone()).collect();
        let is_real = local[reserved_is_real(d)].clone();

        let x1_ext = ChallengeExtension(core::array::from_fn(|i| x1[i].clone()));
        let x2_ext = ChallengeExtension(core::array::from_fn(|i| x2[i].clone()));
        let eq_val_ext = ChallengeExtension(core::array::from_fn(|i| eq_val[i].clone()));
        let prev_acc_ext = ChallengeExtension(core::array::from_fn(|i| prev_acc[i].clone()));
        let acc_ext = ChallengeExtension(core::array::from_fn(|i| acc[i].clone()));

        // eq_val = 1 - x1 - x2 + 2·x1·x2
        let one = ChallengeExtension::<AB::VarMaybeExt>::from_base(AB::VarMaybeExt::one());
        let two = ChallengeExtension::<AB::VarMaybeExt>::from_base(AB::VarMaybeExt::from(
            AB::F::from_canonical_u32(2),
        ));
        let expected_eq = one - x1_ext.clone() - x2_ext.clone() + two * x1_ext * x2_ext;
        for i in 0..d {
            builder
                .when(is_real.clone())
                .assert_zero(eq_val_ext.0[i].clone() - expected_eq.0[i].clone());
        }

        // acc = prev_acc * eq_val
        let expected_acc = prev_acc_ext * eq_val_ext;
        for i in 0..d {
            builder
                .when(is_real.clone())
                .assert_zero(acc_ext.0[i].clone() - expected_acc.0[i].clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let acc_mult = local[reserved_acc_mult(d)].clone();
        let is_real = local[reserved_is_real(d)].clone();

        // Order matches precompute_lc: recv(x1), recv(x2), recv(prev_acc), send(acc)
        builder.recv(is_real.clone());
        builder.recv(is_real.clone());
        builder.recv(is_real);
        builder.send(acc_mult);
    }
}

impl<F: Field> BaseAir<F> for PrefixSumChecksChipPolyAir {
    fn width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_COLS
    }
}

impl<F: Field> MachineAir<F> for PrefixSumChecksChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "PrefixSumChecks".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_PREPROCESS_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        PrefixSumChecksChip.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        PrefixSumChecksChip.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        PrefixSumChecksChip.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        PrefixSumChecksChip.generate_trace(input, output)
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
            let chip = Chip::<PrefixSumChecksChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                PrefixSumChecksChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 4);
            println!(
                "PrefixSumChecksChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<PrefixSumChecksChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
                PrefixSumChecksChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 4);
            println!(
                "PrefixSumChecksChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
