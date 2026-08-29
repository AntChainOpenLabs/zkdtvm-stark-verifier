use std::ops::Deref;

use crate::{
    chips::poseidon2_skinny_kb::{
        columns::{preprocessed::Poseidon2PreprocessedColsSkinnyKb, NUM_POSEIDON2_COLS},
        external_linear_layer, internal_linear_layer, Poseidon2SkinnyKbChip, NUM_INTERNAL_ROUNDS,
        WIDTH,
    },
    *,
};
use dt_primitives::{
    KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::Matrix;

const NUM_COLS: usize = NUM_POSEIDON2_COLS;
const NUM_PREPROCESSED_COLS: usize = core::mem::size_of::<Poseidon2PreprocessedColsSkinnyKb<u8>>();
const ROWS_PER_PERMUTE: usize = 5;

#[derive(Default, Clone, Copy)]
pub struct Poseidon2SkinnyKbChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2SkinnyKbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();

        // All main trace columns (51 cols: state_in[16] + round_witness[19] + state_out[16])
        for i in 0..NUM_COLS {
            cols.push(PairCol::Main(i));
        }

        // Preprocessed: is_round[0..5](5) + state_in_addrs[0..16](16) + state_out_mem[0..16](32) =
        // 53 Need in eval: is_round[0..5] for selectors
        for i in 0..ROWS_PER_PERMUTE {
            cols.push(PairCol::Prep(i));
        }
        // Need in lookup: state_out_mem[i].mult at prep offset 5 + 16 + 2*i + 1
        for i in 0..WIDTH {
            cols.push(PairCol::Prep(5 + 16 + 2 * i + 1));
        }

        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));

        // Collect values from main/prep before mutably borrowing builder
        let main = builder.main();
        let prep = builder.preprocessed();

        let mut interactions = Vec::new();

        // 16 input sends: [state_in_addr[i], state_in[i]]
        for i in 0..WIDTH {
            let addr = prep[5 + i].clone();
            let val = main[i].clone();
            interactions.push(vec![addr, val]);
        }

        // 16 output sends: [state_out_addr[i], state_out[i]]
        for i in 0..WIDTH {
            let addr = prep[5 + 16 + 2 * i].clone();
            let val = main[35 + i].clone();
            interactions.push(vec![addr, val]);
        }

        for vals in interactions {
            builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), vals));
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // reserved_poly layout:
        //   [state_in(16), round_witness(19), state_out(16),  (= 51 main cols)
        //    is_round(5), output_mults(16)]                   (= 21 prep cols)
        let state_in = |i: usize| local[i].clone();
        let round_witness = |i: usize| local[16 + i].clone();
        let state_out = |i: usize| local[35 + i].clone();

        let is_round: [AB::VarMaybeExt; 5] = core::array::from_fn(|r| local[NUM_COLS + r].clone());

        let is_internal = is_round[2].clone();

        // ------------------------------------------------------------------
        // External rounds: each row folds two rounds.
        // ------------------------------------------------------------------
        let external_pairs: [(usize, bool, usize); 4] =
            [(0, true, 0), (1, true, 2), (3, false, 0), (4, false, 2)];

        for (round_sel_idx, first_half, table_idx) in external_pairs {
            let selector = is_round[round_sel_idx].clone();

            let first_rc = |i: usize| -> AB::VarMaybeExt {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx][i]
                };
                AB::VarMaybeExt::from_canonical_u32(c.as_canonical_u32())
            };
            let second_rc = |i: usize| -> AB::VarMaybeExt {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx + 1][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx + 1][i]
                };
                AB::VarMaybeExt::from_canonical_u32(c.as_canonical_u32())
            };

            let mut first_input: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| state_in(i));
            if round_sel_idx == 0 {
                external_linear_layer(&mut first_input);
            }

            let first_add_rc: [AB::VarMaybeExt; WIDTH] =
                core::array::from_fn(|i| first_input[i].clone() + first_rc(i));
            let first_sbox: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
                first_add_rc[i].clone() * first_add_rc[i].clone() * first_add_rc[i].clone()
            });
            let mut first_out = first_sbox;
            external_linear_layer(&mut first_out);
            for i in 0..WIDTH {
                builder.when(selector.clone()).assert_zero(round_witness(i) - first_out[i].clone());
            }

            let second_add_rc: [AB::VarMaybeExt; WIDTH] =
                core::array::from_fn(|i| round_witness(i) + second_rc(i));
            let second_sbox: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
                second_add_rc[i].clone() * second_add_rc[i].clone() * second_add_rc[i].clone()
            });
            let mut second_out = second_sbox;
            external_linear_layer(&mut second_out);
            for i in 0..WIDTH {
                builder.when(selector.clone()).assert_zero(state_out(i) - second_out[i].clone());
            }
        }

        // ------------------------------------------------------------------
        // Internal rounds: all 20 folded into one row.
        // ------------------------------------------------------------------
        {
            let mut state: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| state_in(i));

            for k in 0..(NUM_INTERNAL_ROUNDS - 1) {
                let rc_k = AB::VarMaybeExt::from_canonical_u32(
                    KoalaBear_PARTIAL_CONSTS[k].as_canonical_u32(),
                );
                let sbox_in = state[0].clone() + rc_k;
                let sbox_out = sbox_in.clone() * sbox_in.clone() * sbox_in.clone();
                builder.when(is_internal.clone()).assert_zero(round_witness(k) - sbox_out);
                state[0] = round_witness(k);
                internal_linear_layer(&mut state);
            }

            // Last internal round: inline without witness.
            {
                let rc_last = AB::VarMaybeExt::from_canonical_u32(
                    KoalaBear_PARTIAL_CONSTS[NUM_INTERNAL_ROUNDS - 1].as_canonical_u32(),
                );
                let sbox_in = state[0].clone() + rc_last;
                state[0] = sbox_in.clone() * sbox_in.clone() * sbox_in.clone();
                internal_linear_layer(&mut state);
            }

            for i in 0..WIDTH {
                builder.when(is_internal.clone()).assert_zero(state_out(i) - state[i].clone());
            }
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_round: [AB::VarMaybeExt; 5] = core::array::from_fn(|r| local[NUM_COLS + r].clone());
        let neg_is_real: AB::VarMaybeExt =
            AB::VarMaybeExt::zero() - is_round.iter().cloned().reduce(|a, b| a + b).unwrap();

        // Order matches precompute_lc: 16 input sends, 16 output sends
        for _ in 0..WIDTH {
            builder.send(neg_is_real.clone());
        }
        for i in 0..WIDTH {
            let mult = local[NUM_COLS + ROWS_PER_PERMUTE + i].clone();
            builder.send(mult);
        }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2SkinnyKbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for Poseidon2SkinnyKbChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2SkinnyKbDeg3".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        Poseidon2SkinnyKbChip::<3>::default().preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Poseidon2SkinnyKbChip::<3>::default().generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Poseidon2SkinnyKbChip::<3>::default().num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Poseidon2SkinnyKbChip::<3>::default().generate_trace(input, output)
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

        let chip = Chip::<Poseidon2SkinnyKbChipPolyAir, p3_koala_bear::KoalaBear, 5>::new(
            Poseidon2SkinnyKbChipPolyAir,
        );
        assert_eq!(chip.num_lookup(), 32);
        println!(
            "Poseidon2SkinnyKbChipPolyAir: degree={}, num_alpha={}, num_lookup={}",
            chip.degree,
            chip.num_alpha,
            chip.num_lookup()
        );
    }
}
