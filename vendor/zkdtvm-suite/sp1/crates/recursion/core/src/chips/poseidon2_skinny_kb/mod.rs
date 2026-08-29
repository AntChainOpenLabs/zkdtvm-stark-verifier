use std::marker::PhantomData;

use dt_primitives::{MONTY_INVERSE_KOALABEAR, POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY};
use p3_field::{AbstractField, PrimeField32};

pub mod air;
pub mod columns;
pub mod trace;

use p3_poseidon2::matmul_internal;

/// The width of the permutation.
pub const WIDTH: usize = 16;
pub const RATE: usize = WIDTH / 2;

pub const NUM_EXTERNAL_ROUNDS: usize = 8;
/// cbindgen:ignore
pub const NUM_INTERNAL_ROUNDS: usize = 20;
/// cbindgen:ignore
pub const NUM_ROUNDS: usize = NUM_EXTERNAL_ROUNDS + NUM_INTERNAL_ROUNDS;

/// Half of the external rounds (first half = 4, second half = 4).
pub const HALF_EXTERNAL_ROUNDS: usize = NUM_EXTERNAL_ROUNDS / 2;

/// Rows produced per permutation in the KoalaBear skinny chip's "5-row" layout:
///   - rows 0..1: first half of external rounds (2 rounds per row, folding)
///   - row 2: ALL `NUM_INTERNAL_ROUNDS` (= 20) internal rounds folded into one row
///   - rows 3..4: second half of external rounds (2 rounds per row, folding)
pub const EXTERNAL_ROWS_PER_HALF: usize = 2;
pub const EXTERNAL_ROUNDS_PER_ROW: usize = HALF_EXTERNAL_ROUNDS / EXTERNAL_ROWS_PER_HALF;
/// cbindgen:ignore
pub const ROWS_PER_PERMUTE: usize = EXTERNAL_ROWS_PER_HALF * 2 + 1; // 5

/// Index of the single "internal-rounds" row inside a 5-row permutation block.
/// cbindgen:ignore
pub const INTERNAL_ROW_IDX: usize = EXTERNAL_ROWS_PER_HALF; // 2

/// Number of cross-row scratch address groups carried by `Poseidon2SkinnyInstr` for the
/// KoalaBear path. Equals `ROWS_PER_PERMUTE - 1 = 4` (one group between each pair of
/// adjacent rows of a permutation block).
/// cbindgen:ignore
pub const SKINNY_NUM_SCRATCH_KB: usize = ROWS_PER_PERMUTE - 1; // 4

/// KoalaBear variant of the Poseidon2 skinny chip. Uses SBOX_DEGREE=3 (x^3) instead of
/// BabyBear's SBOX_DEGREE=7, which means no intermediate temp columns are needed to reduce
/// the constraint degree.
///
/// Layout: 9 rows per permutation. The 4 + 4 external rounds occupy one row each; all
/// 20 internal rounds are folded into a single "internal" row whose state-update chain is
/// closed within a single AIR row using the per-internal-round `internal_rounds_s0[k]`
/// witness columns. This keeps constraint degree at 3 while shrinking the row count from
/// 28 (one-round-per-row) to 9.
pub struct Poseidon2SkinnyKbChip<const DEGREE: usize>(PhantomData<()>);

impl<const DEGREE: usize> Default for Poseidon2SkinnyKbChip<DEGREE> {
    fn default() -> Self {
        assert!(DEGREE >= 3);
        Self(PhantomData)
    }
}

pub fn apply_m_4<AF>(x: &mut [AF])
where
    AF: AbstractField,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123.clone() + x[3].clone();
    x[3] = t01233.clone() + x[0].double();
    x[1] = t01123.clone() + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

pub(crate) fn external_linear_layer<AF: AbstractField>(state: &mut [AF; WIDTH]) {
    for j in (0..WIDTH).step_by(4) {
        apply_m_4(&mut state[j..j + 4]);
    }
    let sums: [AF; 4] =
        core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k].clone()).sum::<AF>());

    for j in 0..WIDTH {
        state[j] = state[j].clone() + sums[j % 4].clone();
    }
}

pub(crate) fn internal_linear_layer<F: AbstractField>(state: &mut [F; WIDTH]) {
    let matmul_constants: [<F as AbstractField>::F; WIDTH] =
        POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY
            .iter()
            .map(|x| <F as AbstractField>::F::from_wrapped_u32(x.as_canonical_u32()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    matmul_internal(state, matmul_constants);
    let monty_inverse = F::from_wrapped_u32(MONTY_INVERSE_KOALABEAR.as_canonical_u32());
    state.iter_mut().for_each(|i| *i = i.clone() * monty_inverse.clone());
}
