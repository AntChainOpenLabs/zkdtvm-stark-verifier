use std::ops::Deref;

use crate::{
    chips::poseidon2_skinny::{
        columns::{preprocessed::Poseidon2PreprocessedColsSkinny, NUM_POSEIDON2_COLS},
        external_linear_layer, internal_linear_layer, Poseidon2SkinnyChip, WIDTH,
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

const NUM_COLS: usize = NUM_POSEIDON2_COLS; // 32
const NUM_PREPROCESSED_COLS: usize = core::mem::size_of::<Poseidon2PreprocessedColsSkinny<u8>>(); // 68

// Preprocessed layout offsets:
// round_kind(0), is_first_round(1), is_real(2), state_in_neg_mult(3),
// round_constants[0..16](4..20), state_in_addrs[0..16](20..36), state_out_mem[0..16](36..68)
const PREP_ROUND_KIND: usize = 0;
const PREP_IS_FIRST: usize = 1;
const PREP_IS_REAL: usize = 2;
const PREP_NEG_MULT: usize = 3;
const PREP_RC_OFFSET: usize = 4;
const PREP_IN_ADDR_OFFSET: usize = 20;
const PREP_OUT_MEM_OFFSET: usize = 36;

#[derive(Default, Clone, Copy)]
pub struct Poseidon2SkinnyBbChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2SkinnyBbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::new();
        // All 32 main cols
        for i in 0..NUM_COLS {
            cols.push(PairCol::Main(i));
        }
        // Prep: round_kind, is_first_round, is_real, round_constants[0..16]
        cols.push(PairCol::Prep(PREP_ROUND_KIND));
        cols.push(PairCol::Prep(PREP_IS_FIRST));
        cols.push(PairCol::Prep(PREP_IS_REAL));
        for i in 0..WIDTH {
            cols.push(PairCol::Prep(PREP_RC_OFFSET + i));
        }
        // output mults for lookup
        for i in 0..WIDTH {
            cols.push(PairCol::Prep(PREP_OUT_MEM_OFFSET + 2 * i + 1));
        }
        // state_in_neg_mult for lookup
        cols.push(PairCol::Prep(PREP_NEG_MULT));
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let mem_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));
        let main = builder.main();
        let prep = builder.preprocessed();

        let mut interactions = Vec::new();
        // 16 input sends
        for i in 0..WIDTH {
            let addr = prep[PREP_IN_ADDR_OFFSET + i].clone();
            let val = main[i].clone(); // state_in[i]
            interactions.push(vec![addr, val]);
        }
        // 16 output sends
        for i in 0..WIDTH {
            let addr = prep[PREP_OUT_MEM_OFFSET + 2 * i].clone();
            let val = main[WIDTH + i].clone(); // state_out[i]
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
        //   main: state_in[0..16], state_out[16..32]    (32 cols)
        //   prep: round_kind, is_first, is_real, rc[0..16], out_mults[0..16], neg_mult
        //         (3 + 16 + 16 + 1 = 36 cols)
        let state_in = |i: usize| local[i].clone();
        let state_out = |i: usize| local[WIDTH + i].clone();

        let mut prep_idx = NUM_COLS;
        let round_kind = local[prep_idx].clone();
        prep_idx += 1;
        let is_first = local[prep_idx].clone();
        prep_idx += 1;
        let is_real = local[prep_idx].clone();
        prep_idx += 1;
        let rc: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| local[prep_idx + i].clone());
        prep_idx += WIDTH;
        // out_mults and neg_mult are only used in lookup, skip here
        let _ = prep_idx;

        let is_internal = round_kind.clone();
        let is_external = AB::VarMaybeExt::one() - is_internal.clone();
        let not_first = AB::VarMaybeExt::one() - is_first.clone();

        // External round, is_first: initial linear layer + RC + S-box(x^7) + linear layer
        {
            let mut pre: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| state_in(i));
            external_linear_layer(&mut pre);
            let add_rc: [AB::VarMaybeExt; WIDTH] =
                core::array::from_fn(|i| pre[i].clone() + rc[i].clone());
            let sbox: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
                let deg3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
                deg3.clone() * deg3.clone() * add_rc[i].clone()
            });
            let mut out = sbox;
            external_linear_layer(&mut out);
            let selector = is_real.clone() * is_external.clone() * is_first.clone();
            for i in 0..WIDTH {
                builder.when(selector.clone()).assert_zero(state_out(i) - out[i].clone());
            }
        }

        // External round, !is_first: RC + S-box(x^7) + linear layer
        {
            let add_rc: [AB::VarMaybeExt; WIDTH] =
                core::array::from_fn(|i| state_in(i) + rc[i].clone());
            let sbox: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
                let deg3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
                deg3.clone() * deg3.clone() * add_rc[i].clone()
            });
            let mut out = sbox;
            external_linear_layer(&mut out);
            let selector = is_real.clone() * is_external.clone() * not_first.clone();
            for i in 0..WIDTH {
                builder.when(selector.clone()).assert_zero(state_out(i) - out[i].clone());
            }
        }

        // Internal round: only state[0] through S-box
        {
            let mut state: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| state_in(i));
            let add_rc0 = state[0].clone() + rc[0].clone();
            let deg3 = add_rc0.clone() * add_rc0.clone() * add_rc0.clone();
            state[0] = deg3.clone() * deg3.clone() * add_rc0.clone();
            internal_linear_layer(&mut state);
            let selector = is_real.clone() * is_internal.clone();
            for i in 0..WIDTH {
                builder.when(selector.clone()).assert_zero(state_out(i) - state[i].clone());
            }
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // Layout after main(32) + round_kind(1) + is_first(1) + is_real(1) + rc(16):
        let out_mults_start = NUM_COLS + 3 + WIDTH; // 32 + 19 = 51
        let neg_mult = local[out_mults_start + WIDTH].clone(); // last reserved col

        for _ in 0..WIDTH {
            builder.send(neg_mult.clone());
        }
        for i in 0..WIDTH {
            builder.send(local[out_mults_start + i].clone());
        }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2SkinnyBbChipPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for Poseidon2SkinnyBbChipPolyAir {
    type Record = ExecutionRecord<F>;
    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2SkinnyDeg9".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        Poseidon2SkinnyChip::<9>::default().preprocessed_num_rows(program, instrs_len)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Poseidon2SkinnyChip::<9>::default().generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {}

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Poseidon2SkinnyChip::<9>::default().num_rows(input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Poseidon2SkinnyChip::<9>::default().generate_trace(input, output)
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

        let chip = Chip::<Poseidon2SkinnyBbChipPolyAir, p3_baby_bear::BabyBear, 4>::new(
            Poseidon2SkinnyBbChipPolyAir,
        );
        assert_eq!(chip.num_lookup(), 32);
        println!(
            "Poseidon2SkinnyBbChipPolyAir: degree={}, num_alpha={}, num_lookup={}",
            chip.degree,
            chip.num_alpha,
            chip.num_lookup()
        );
    }
}
