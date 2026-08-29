use crate::{
    chips::alu_base::{
        BaseAluAccessCols, BaseAluChip, BaseAluValueCols, NUM_BASE_ALU_ACCESS_COLS,
        NUM_BASE_ALU_ENTRIES_PER_ROW, NUM_BASE_ALU_VALUE_COLS,
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
use std::ops::Deref;

pub struct BaseAluChipPolyAir<const N: usize = NUM_BASE_ALU_ENTRIES_PER_ROW>;

impl<const N: usize> Default for BaseAluChipPolyAir<N> {
    fn default() -> Self {
        Self
    }
}

impl<const N: usize> Clone for BaseAluChipPolyAir<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize> Copy for BaseAluChipPolyAir<N> {}

impl<AB: FullAirBuilder, const N: usize> FullAir<AB> for BaseAluChipPolyAir<N> {
    fn width(&self) -> usize {
        NUM_BASE_ALU_VALUE_COLS * N
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();
        for entry in 0..N {
            // Main: out, in1, in2 (3 fields per entry)
            let main_offset = entry * NUM_BASE_ALU_VALUE_COLS;
            cols.push(PairCol::Main(main_offset)); // out
            cols.push(PairCol::Main(main_offset + 1)); // in1
            cols.push(PairCol::Main(main_offset + 2)); // in2

            // Preprocessed: is_add, is_sub, is_mul, is_div, mult
            // BaseAluAccessCols layout: addrs(3 Address) + is_add + is_sub + is_mul + is_div + mult
            // = 3 + 4 + 1 = 8 fields per entry
            let prep_offset = entry * NUM_BASE_ALU_ACCESS_COLS;
            cols.push(PairCol::Prep(prep_offset + 3)); // is_add
            cols.push(PairCol::Prep(prep_offset + 4)); // is_sub
            cols.push(PairCol::Prep(prep_offset + 5)); // is_mul
            cols.push(PairCol::Prep(prep_offset + 6)); // is_div
            cols.push(PairCol::Prep(prep_offset + 7)); // mult
        }
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        let interactions: Vec<Vec<AB::VarMaybeExt>> = (0..N)
            .flat_map(|entry| {
                let main_offset = entry * NUM_BASE_ALU_VALUE_COLS;
                let prep_offset = entry * NUM_BASE_ALU_ACCESS_COLS;

                let addr_out = prep[prep_offset].clone();
                let addr_in1 = prep[prep_offset + 1].clone();
                let addr_in2 = prep[prep_offset + 2].clone();

                let out = main[main_offset].clone();
                let in1 = main[main_offset + 1].clone();
                let in2 = main[main_offset + 2].clone();

                vec![vec![addr_in1, in1], vec![addr_in2, in2], vec![addr_out, out]]
            })
            .collect();

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly layout per entry: [out, in1, in2, is_add, is_sub, is_mul, is_div, mult]
        // = 8 fields per entry
        let entry_size = 8;

        for entry in 0..N {
            let base = entry * entry_size;
            let out = local[base].clone();
            let in1 = local[base + 1].clone();
            let in2 = local[base + 2].clone();
            let is_add = local[base + 3].clone();
            let is_sub = local[base + 4].clone();
            let is_mul = local[base + 5].clone();
            let is_div = local[base + 6].clone();

            // is_real = is_add + is_sub + is_mul + is_div
            let is_real = is_add.clone() + is_sub.clone() + is_mul.clone() + is_div.clone();
            // is_real must be boolean
            builder.assert_zero(
                is_real.clone() * (is_real.clone() - AB::VarMaybeExt::from(AB::F::one())),
            );

            // is_add: in1 + in2 = out
            builder.when(is_add).assert_zero(in1.clone() + in2.clone() - out.clone());
            // is_sub: in1 = in2 + out → in1 - in2 - out = 0
            builder.when(is_sub).assert_zero(in1.clone() - in2.clone() - out.clone());
            // is_mul: in1 * in2 = out
            builder.when(is_mul).assert_zero(in1.clone() * in2.clone() - out.clone());
            // is_div: in2 * out = in1
            builder.when(is_div).assert_zero(in2 * out - in1);
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let entry_size = 8;

        for entry in 0..N {
            let base = entry * entry_size;
            let is_add = local[base + 3].clone();
            let is_sub = local[base + 4].clone();
            let is_mul = local[base + 5].clone();
            let is_div = local[base + 6].clone();
            let mult = local[base + 7].clone();

            let is_real = is_add + is_sub + is_mul + is_div;

            // Order matches precompute_lc: recv(in1), recv(in2), send(out)
            builder.recv(is_real.clone());
            builder.recv(is_real);
            builder.send(mult);
        }
    }
}

impl<F: Field, const N: usize> BaseAir<F> for BaseAluChipPolyAir<N> {
    fn width(&self) -> usize {
        NUM_BASE_ALU_VALUE_COLS * N
    }
}

impl<F: Field, const N: usize> MachineAir<F> for BaseAluChipPolyAir<N> {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        if N == NUM_BASE_ALU_ENTRIES_PER_ROW {
            "BaseAlu".to_string()
        } else {
            format!("BaseAlu<{}>", N)
        }
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BASE_ALU_ACCESS_COLS * N
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        BaseAluChip::<N>.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        BaseAluChip::<N>.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        BaseAluChip::<N>.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        BaseAluChip::<N>.generate_trace(input, output)
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
            let chip =
                Chip::<BaseAluChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(BaseAluChipPolyAir);
            // 4 entries × 3 lookups = 12 lookups
            assert_eq!(chip.num_lookup(), 12);
            println!(
                "BaseAluChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip =
                Chip::<BaseAluChipPolyAir, p3_baby_bear::BabyBear, 4>::new(BaseAluChipPolyAir);
            assert_eq!(chip.num_lookup(), 12);
            println!(
                "BaseAluChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
