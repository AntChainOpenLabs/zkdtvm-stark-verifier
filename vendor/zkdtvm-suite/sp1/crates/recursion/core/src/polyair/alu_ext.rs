use crate::{
    chips::alu_ext::{
        ExtAluAccessCols, ExtAluChip, ExtAluValueCols, NUM_EXT_ALU_ACCESS_COLS, NUM_EXT_ALU_COLS,
        NUM_EXT_ALU_ENTRIES_PER_ROW,
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

const NUM_EXT_ALU_VALUE_COLS: usize = core::mem::size_of::<ExtAluValueCols<u8>>();

pub struct ExtAluChipPolyAir<const N: usize = NUM_EXT_ALU_ENTRIES_PER_ROW>;

impl<const N: usize> Default for ExtAluChipPolyAir<N> {
    fn default() -> Self {
        Self
    }
}

impl<const N: usize> Clone for ExtAluChipPolyAir<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize> Copy for ExtAluChipPolyAir<N> {}

fn ext_poly_mul<AB: FullAirBuilder>(
    a: &[AB::VarMaybeExt],
    b: &[AB::VarMaybeExt],
) -> Vec<AB::VarMaybeExt> {
    let a_ext = ChallengeExtension(core::array::from_fn(|i| a[i].clone()));
    let b_ext = ChallengeExtension(core::array::from_fn(|i| b[i].clone()));
    (a_ext * b_ext).0.to_vec()
}

impl<AB: FullAirBuilder, const N: usize> FullAir<AB> for ExtAluChipPolyAir<N> {
    fn width(&self) -> usize {
        NUM_EXT_ALU_VALUE_COLS * N
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let d = runtime::D;
        let mut cols = Vec::new();
        for entry in 0..N {
            // Main: ExtAluValueCols { vals: ExtAluIo<Block<F>> { out, in1, in2 } }
            // Each Block has D fields, so 3*D fields per entry
            let main_offset = entry * NUM_EXT_ALU_VALUE_COLS;
            // out[0..D]
            for i in 0..d {
                cols.push(PairCol::Main(main_offset + i));
            }
            // in1[0..D]
            for i in 0..d {
                cols.push(PairCol::Main(main_offset + d + i));
            }
            // in2[0..D]
            for i in 0..d {
                cols.push(PairCol::Main(main_offset + 2 * d + i));
            }

            // Preprocessed: ExtAluAccessCols { addrs: ExtAluIo<Address> { out, in1, in2 },
            //                                  is_add, is_sub, is_mul, is_div, mult }
            // = 3 + 4 + 1 = 8 fields per entry
            let prep_offset = entry * NUM_EXT_ALU_ACCESS_COLS;
            cols.push(PairCol::Prep(prep_offset + 3)); // is_add
            cols.push(PairCol::Prep(prep_offset + 4)); // is_sub
            cols.push(PairCol::Prep(prep_offset + 5)); // is_mul
            cols.push(PairCol::Prep(prep_offset + 6)); // is_div
            cols.push(PairCol::Prep(prep_offset + 7)); // mult
        }
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let d = runtime::D;
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        let main = builder.main();
        let prep = builder.preprocessed();

        // For each entry: 3 lookups (recv in1, recv in2, send out)
        // denominator = α + Memory + β¹·addr + β²·val[0] + β³·val[1] + ... + β^(D+1)·val[D-1]
        let interactions: Vec<Vec<AB::VarMaybeExt>> = (0..N)
            .flat_map(|entry| {
                let main_offset = entry * NUM_EXT_ALU_VALUE_COLS;
                let prep_offset = entry * NUM_EXT_ALU_ACCESS_COLS;

                // Addrs: ExtAluIo<Address> { out, in1, in2 }
                let addr_out = prep[prep_offset].clone();
                let addr_in1 = prep[prep_offset + 1].clone();
                let addr_in2 = prep[prep_offset + 2].clone();

                // Values: ExtAluIo<Block<F>> { out[0..D], in1[0..D], in2[0..D] }
                let out_vals: Vec<_> = (0..d).map(|i| main[main_offset + i].clone()).collect();
                let in1_vals: Vec<_> = (0..d).map(|i| main[main_offset + d + i].clone()).collect();
                let in2_vals: Vec<_> =
                    (0..d).map(|i| main[main_offset + 2 * d + i].clone()).collect();

                // recv in1: [addr_in1, in1[0], in1[1], ..., in1[D-1]]
                let mut vals_in1 = vec![addr_in1];
                vals_in1.extend(in1_vals);

                // recv in2: [addr_in2, in2[0], in2[1], ..., in2[D-1]]
                let mut vals_in2 = vec![addr_in2];
                vals_in2.extend(in2_vals);

                // send out: [addr_out, out[0], out[1], ..., out[D-1]]
                let mut vals_out = vec![addr_out];
                vals_out.extend(out_vals);

                vec![vals_in1, vals_in2, vals_out]
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

        let entry_size = 3 * d + 5;

        for entry in 0..N {
            let base = entry * entry_size;
            let out: Vec<_> = (0..d).map(|i| local[base + i].clone()).collect();
            let in1: Vec<_> = (0..d).map(|i| local[base + d + i].clone()).collect();
            let in2: Vec<_> = (0..d).map(|i| local[base + 2 * d + i].clone()).collect();
            let is_add = local[base + 3 * d].clone();
            let is_sub = local[base + 3 * d + 1].clone();
            let is_mul = local[base + 3 * d + 2].clone();
            let is_div = local[base + 3 * d + 3].clone();

            let is_real = is_add.clone() + is_sub.clone() + is_mul.clone() + is_div.clone();
            builder.assert_zero(is_real.clone() * (is_real - AB::VarMaybeExt::one()));

            // add: out = in1 + in2
            for i in 0..d {
                builder
                    .when(is_add.clone())
                    .assert_zero(in1[i].clone() + in2[i].clone() - out[i].clone());
            }
            // sub: out = in1 - in2
            for i in 0..d {
                builder
                    .when(is_sub.clone())
                    .assert_zero(in1[i].clone() - in2[i].clone() - out[i].clone());
            }
            // mul: out = in1 * in2 (extension field via ChallengeExtension)
            let mul_result = ext_poly_mul::<AB>(&in1, &in2);
            for i in 0..d {
                builder.when(is_mul.clone()).assert_zero(mul_result[i].clone() - out[i].clone());
            }
            // div: in1 = in2 * out
            let div_result = ext_poly_mul::<AB>(&in2, &out);
            for i in 0..d {
                builder.when(is_div.clone()).assert_zero(div_result[i].clone() - in1[i].clone());
            }
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let d = runtime::D;
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let entry_size = 3 * d + 5;

        for entry in 0..N {
            let base = entry * entry_size;
            let is_add = local[base + 3 * d].clone();
            let is_sub = local[base + 3 * d + 1].clone();
            let is_mul = local[base + 3 * d + 2].clone();
            let is_div = local[base + 3 * d + 3].clone();
            let mult = local[base + 3 * d + 4].clone();

            let is_real = is_add + is_sub + is_mul + is_div;

            // Order matches precompute_lc: recv(in1), recv(in2), send(out)
            builder.recv(is_real.clone());
            builder.recv(is_real);
            builder.send(mult);
        }
    }
}

impl<F: Field, const N: usize> BaseAir<F> for ExtAluChipPolyAir<N> {
    fn width(&self) -> usize {
        NUM_EXT_ALU_VALUE_COLS * N
    }
}

impl<F: Field, const N: usize> MachineAir<F> for ExtAluChipPolyAir<N> {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        if N == NUM_EXT_ALU_ENTRIES_PER_ROW {
            "ExtAlu".to_string()
        } else {
            format!("ExtAlu<{}>", N)
        }
    }

    fn preprocessed_width(&self) -> usize {
        NUM_EXT_ALU_ACCESS_COLS * N
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        ExtAluChip::<N>.preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        ExtAluChip::<N>.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        ExtAluChip::<N>.num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ExtAluChip::<N>.generate_trace(input, output)
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
                Chip::<ExtAluChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(ExtAluChipPolyAir);
            // 4 entries × 3 lookups = 12 lookups
            assert_eq!(chip.num_lookup(), 12);
            println!(
                "ExtAluChipPolyAir (eth): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
        #[cfg(feature = "babybear")]
        {
            let chip = Chip::<ExtAluChipPolyAir, p3_baby_bear::BabyBear, 4>::new(ExtAluChipPolyAir);
            assert_eq!(chip.num_lookup(), 12);
            println!(
                "ExtAluChipPolyAir (legacy): degree={}, num_alpha={}, num_lookup={}",
                chip.degree,
                chip.num_alpha,
                chip.num_lookup()
            );
        }
    }
}
