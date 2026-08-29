use crate::*;
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::ops::Deref;

use crate::chips::mem::constant::{
    MemoryChip as MemoryConstChip, NUM_CONST_MEM_ENTRIES_PER_ROW, NUM_MEM_INIT_COLS,
    NUM_MEM_PREPROCESSED_INIT_COLS,
};

#[derive(Default, Clone, Copy)]
pub struct MemoryConstChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for MemoryConstChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEM_INIT_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let d = runtime::D;
        // Only mult fields are needed in later rounds (for lookup()).
        // val[0..D] and addr are only used in precompute_lc() which runs once in Round 0.
        (0..NUM_CONST_MEM_ENTRIES_PER_ROW)
            .map(|entry| PairCol::Prep(entry * (d + 2) + d + 1))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;

        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let prep = builder.preprocessed();
        let interactions: Vec<Vec<AB::VarMaybeExt>> = (0..NUM_CONST_MEM_ENTRIES_PER_ROW)
            .map(|entry| {
                let offset = entry * (d + 2);
                let addr = prep[offset + d].clone();
                let mut vals = vec![addr];
                for i in 0..d {
                    vals.push(prep[offset + i].clone());
                }
                vals
            })
            .collect();

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, _builder: &mut AB) {
        // MemoryConstChip has no gate constraints — only lookups.
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly only contains mult fields, one per entry
        for entry in 0..NUM_CONST_MEM_ENTRIES_PER_ROW {
            let mult = local[entry].clone();
            builder.send(mult);
        }
    }
}

impl<F: Field> BaseAir<F> for MemoryConstChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEM_INIT_COLS
    }
}

impl<F: Field> MachineAir<F> for MemoryConstChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "MemoryConst".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_MEM_PREPROCESSED_INIT_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        MemoryConstChip::<F>::default().preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        MemoryConstChip::<F>::default().generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        MemoryConstChip::<F>::default().num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        MemoryConstChip::<F>::default().generate_trace(input, output)
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
            let chip = Chip::<MemoryConstChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                MemoryConstChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), NUM_CONST_MEM_ENTRIES_PER_ROW);
            println!(
                "MemoryConstChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<MemoryConstChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
                MemoryConstChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), NUM_CONST_MEM_ENTRIES_PER_ROW);
            println!(
                "MemoryConstChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
