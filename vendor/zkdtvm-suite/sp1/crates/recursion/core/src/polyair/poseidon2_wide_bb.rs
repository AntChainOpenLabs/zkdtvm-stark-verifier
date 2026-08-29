use std::ops::Deref;

use crate::{
    chips::poseidon2_wide::{
        columns::preprocessed::Poseidon2PreprocessedColsWide, Poseidon2WideChip,
    },
    *,
};
use dt_core_machine::operations::poseidon2::{
    air::{eval_poseidon2_full, Poseidon2ColsViewBb},
    permutation::NUM_POSEIDON2_DEGREE3_COLS,
    WIDTH,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

const NUM_COLS: usize = NUM_POSEIDON2_DEGREE3_COLS;
const NUM_PREPROCESSED_COLS: usize = core::mem::size_of::<Poseidon2PreprocessedColsWide<u8>>();
const OUTPUT_OFFSET: usize = {
    let ext_state = 8 * WIDTH; // 128
    let int_state = WIDTH; // 16
    let int_s0 = 12; // NUM_INTERNAL_ROUNDS - 1
    ext_state + int_state + int_s0 // 156
};

#[derive(Default, Clone, Copy)]
pub struct Poseidon2WideBbChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2WideBbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();
        for i in 0..NUM_COLS {
            cols.push(PairCol::Main(i));
        }
        // Prep: input[0..16], output[0..16](addr+mult), is_real_neg
        // Need output mults and is_real_neg in lookup
        for i in 0..WIDTH {
            cols.push(PairCol::Prep(16 + 2 * i + 1));
        }
        cols.push(PairCol::Prep(NUM_PREPROCESSED_COLS - 1));
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));
        let main = builder.main();
        let prep = builder.preprocessed();

        let mut interactions = Vec::new();
        for i in 0..WIDTH {
            interactions.push(vec![prep[i].clone(), main[i].clone()]);
        }
        for i in 0..WIDTH {
            interactions.push(vec![prep[16 + 2 * i].clone(), main[OUTPUT_OFFSET + i].clone()]);
        }
        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let view = Poseidon2ColsViewBb::from_degree3_slice(&local[..NUM_COLS]);
        eval_poseidon2_full(builder, &view);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let is_real_neg = local[NUM_COLS + WIDTH].clone();
        for _ in 0..WIDTH {
            builder.send(is_real_neg.clone());
        }
        for i in 0..WIDTH {
            builder.send(local[NUM_COLS + i].clone());
        }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2WideBbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for Poseidon2WideBbChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2WideDeg9".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        Poseidon2WideChip::<9>.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Poseidon2WideChip::<9>.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Poseidon2WideChip::<9>.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Poseidon2WideChip::<9>.generate_trace(input, output)
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

        let chip = Chip::<Poseidon2WideBbChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
            Poseidon2WideBbChipPolyAir,
        );
        assert_eq!(chip.num_lookup(), 32);
        println!(
            "Poseidon2WideBbChipPolyAir: degree={}, num_alpha={}, num_lookup={}",
            chip.degree,
            chip.num_alpha,
            chip.num_lookup()
        );
    }
}
