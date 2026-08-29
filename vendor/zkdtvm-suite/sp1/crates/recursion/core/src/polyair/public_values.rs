use crate::{
    air::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS},
    chips::public_values::{
        PublicValuesChip, NUM_PUBLIC_VALUES_COLS, NUM_PUBLIC_VALUES_PREPROCESSED_COLS,
        PUB_VALUES_LOG_HEIGHT,
    },
    *,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::{borrow::Borrow, ops::Deref};

#[derive(Default, Clone, Copy)]
pub struct PublicValuesChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for PublicValuesChipPolyAir {
    fn width(&self) -> usize {
        NUM_PUBLIC_VALUES_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();

        // Main: pv_element (1 col)
        cols.push(PairCol::Main(0));

        // Preprocessed layout: pv_idx[0..DIGEST_SIZE], pv_mem.addr, pv_mem.mult
        // pv_idx needed in eval() for gate constraints
        for i in 0..DIGEST_SIZE {
            cols.push(PairCol::Prep(i));
        }
        // pv_mem.mult needed in lookup()
        cols.push(PairCol::Prep(DIGEST_SIZE + 1));

        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        // send_single: [addr, pv_element, 0, 0, ..., 0]
        // Zero padding is unnecessary since zero terms don't affect the denominator.
        let addr = prep[DIGEST_SIZE].clone(); // pv_mem.addr
        let val = main[0].clone(); // pv_element
        let vals = vec![addr, val];

        builder.retain_precomputed(builder.lookup_denominator(mem_kind, vals));
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly layout: [pv_element, pv_idx[0..DIGEST_SIZE], pv_mem.mult]
        let pv_element = local[0].clone();
        let pv_idx: Vec<_> = (0..DIGEST_SIZE).map(|i| local[1 + i].clone()).collect();

        // Public values
        let pv = builder.public();
        let pv_elms: [AB::VarMaybeExt; RECURSIVE_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| pv[i].clone().into());
        let public_values: &RecursionPublicValues<AB::VarMaybeExt> = pv_elms.as_slice().borrow();

        // For each digest element, when the corresponding pv_idx flag is set,
        // constrain that pv_element equals the public value digest element.
        for i in 0..DIGEST_SIZE {
            builder
                .when(pv_idx[i].clone())
                .assert_zero(public_values.digest[i].clone() - pv_element.clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly layout: [pv_element, pv_idx[0..DIGEST_SIZE], pv_mem.mult]
        let mult = local[1 + DIGEST_SIZE].clone();

        // 1 send matching precompute_lc order
        builder.send(mult);
    }
}

impl<F: Field> BaseAir<F> for PublicValuesChipPolyAir {
    fn width(&self) -> usize {
        NUM_PUBLIC_VALUES_COLS
    }
}

impl<F: Field> MachineAir<F> for PublicValuesChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "PublicValues".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PUBLIC_VALUES_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        PublicValuesChip.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        PublicValuesChip.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        PublicValuesChip.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        PublicValuesChip.generate_trace(input, output)
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
            let chip = Chip::<PublicValuesChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                PublicValuesChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 1);
            println!(
                "PublicValuesChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<PublicValuesChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
                PublicValuesChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), 1);
            println!(
                "PublicValuesChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
