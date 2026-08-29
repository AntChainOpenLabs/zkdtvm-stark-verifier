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

use crate::chips::mem::variable::{
    MemoryChip as MemoryVarChip, NUM_MEM_INIT_COLS, NUM_MEM_PREPROCESSED_INIT_COLS,
    NUM_VAR_MEM_ENTRIES_PER_ROW,
};

#[derive(Default, Clone, Copy)]
pub struct MemoryVarChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for MemoryVarChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEM_INIT_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only mult fields are needed in later rounds (for lookup()).
        // val[0..D] and addr are only used in precompute_lc() which runs once in Round 0.
        (0..NUM_VAR_MEM_ENTRIES_PER_ROW).map(|entry| PairCol::Prep(entry * 2 + 1)).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;

        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();
        let interactions: Vec<Vec<AB::VarMaybeExt>> = (0..NUM_VAR_MEM_ENTRIES_PER_ROW)
            .map(|entry| {
                let main_offset = entry * d;
                let prep_offset = entry * 2;

                let addr = prep[prep_offset].clone();
                let mut vals = vec![addr];
                for i in 0..d {
                    vals.push(main[main_offset + i].clone());
                }
                vals
            })
            .collect();

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, _builder: &mut AB) {
        // MemoryVarChip has no gate constraints — only lookups.
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly only contains mult fields, one per entry
        for entry in 0..NUM_VAR_MEM_ENTRIES_PER_ROW {
            let mult = local[entry].clone();
            builder.send(mult);
        }
    }
}

impl<F: Field> BaseAir<F> for MemoryVarChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEM_INIT_COLS
    }
}

impl<F: Field> MachineAir<F> for MemoryVarChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "MemoryVar".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_MEM_PREPROCESSED_INIT_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        MemoryVarChip::<F>::default().preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        MemoryVarChip::<F>::default().generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        MemoryVarChip::<F>::default().num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        MemoryVarChip::<F>::default().generate_trace(input, output)
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
            let chip = Chip::<MemoryVarChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
                MemoryVarChipPolyAir,
            );
            assert_eq!(chip.num_lookup(), NUM_VAR_MEM_ENTRIES_PER_ROW);
            println!(
                "MemoryVarChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip =
                Chip::<MemoryVarChipPolyAir, p3_baby_bear::BabyBear, 4>::new(MemoryVarChipPolyAir);
            assert_eq!(chip.num_lookup(), NUM_VAR_MEM_ENTRIES_PER_ROW);
            println!(
                "MemoryVarChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
