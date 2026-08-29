//! PolyAir gate constraints for the inner Poseidon2 permutation (KoalaBear, degree-3 S-box).
//!
//! The inner Poseidon2 AIR for KoalaBear has **0 lookup interactions** — pure gate constraints.
//! This module mirrors `air.rs::eval()` in `VarMaybeExt` arithmetic.
//!
//! ## Constraint Summary
//!
//! - `WIDTH` boundary constraints linking `inputs` to `external_rounds_state[0]`.
//! - `NUM_EXTERNAL_ROUNDS` × `WIDTH` post-state constraints across external rounds.
//! - `NUM_INTERNAL_ROUNDS - 1` constraints linking `internal_rounds_s0[r]` to the running state[0]
//!   after each non-final internal round.
//! - `WIDTH` constraints linking the final internal state to `external_rounds_state[HFROUNDS]`.

use dt_stark::air::FullAirBuilder;
use p3_field::AbstractField;

use super::{
    air::INTERNAL_DIAG_KB_24,
    columns::{Poseidon2Cols, HFROUNDS, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH},
    constants::RoundConstants,
};

// ============================================================================
// Poseidon2 inner gate constraints (KoalaBear)
// ============================================================================

/// Enforce all gate constraints from the KoalaBear inner Poseidon2 AIR.
///
/// Reproduces every `assert_eq` from `air.rs::eval()`. No lookup interactions.
pub fn poseidon2_inner_gate_constraints<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::VarMaybeExt>,
    constants: &RoundConstants<AB::F>,
) where
    AB::VarMaybeExt: Clone,
{
    // external_rounds_state[0] == inputs
    for i in 0..WIDTH {
        builder.assert_zero(local.external_rounds_state[0][i].clone() - local.inputs[i].clone());
    }

    // First half of external rounds.
    for r in 0..HFROUNDS {
        eval_external_round_gate::<AB>(builder, local, constants, r);
    }

    // Internal rounds.
    eval_internal_rounds_gate::<AB>(builder, local, constants);

    // Second half of external rounds.
    for r in HFROUNDS..NUM_EXTERNAL_ROUNDS {
        eval_external_round_gate::<AB>(builder, local, constants, r);
    }
}

// ============================================================================
// Round evaluation helpers
// ============================================================================

fn eval_external_round_gate<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::VarMaybeExt>,
    constants: &RoundConstants<AB::F>,
    r: usize,
) where
    AB::VarMaybeExt: Clone,
{
    let mut state: [AB::VarMaybeExt; WIDTH] = local.external_rounds_state[r].clone();

    if r == 0 {
        external_linear_layer_full_air::<AB>(&mut state);
    }

    let round_constants = if r < HFROUNDS {
        &constants.beginning_full_round_constants[r]
    } else {
        &constants.ending_full_round_constants[r - HFROUNDS]
    };

    // Degree-3 S-box: (state[i] + rc)^3.
    let mut sbox_out: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
        let add_rc = state[i].clone() + AB::VarMaybeExt::from(round_constants[i]);
        add_rc.clone() * add_rc.clone() * add_rc
    });

    external_linear_layer_full_air::<AB>(&mut sbox_out);

    let next_state: &[AB::VarMaybeExt; WIDTH] = if r == HFROUNDS - 1 {
        &local.internal_rounds_state
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        &local.output_state
    } else {
        &local.external_rounds_state[r + 1]
    };

    for i in 0..WIDTH {
        builder.assert_zero(next_state[i].clone() - sbox_out[i].clone());
    }
}

fn eval_internal_rounds_gate<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::VarMaybeExt>,
    constants: &RoundConstants<AB::F>,
) where
    AB::VarMaybeExt: Clone,
{
    let mut state: [AB::VarMaybeExt; WIDTH] = local.internal_rounds_state.clone();

    for r in 0..NUM_INTERNAL_ROUNDS {
        let s0 = if r == 0 { state[0].clone() } else { local.internal_rounds_s0[r - 1].clone() };
        let add_rc = s0 + AB::VarMaybeExt::from(constants.partial_round_constants[r]);

        state[0] = add_rc.clone() * add_rc.clone() * add_rc;
        internal_linear_layer_full_air::<AB>(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_zero(local.internal_rounds_s0[r].clone() - state[0].clone());
        }
    }

    // After all internal rounds, link to start of second-half external rounds.
    for i in 0..WIDTH {
        builder.assert_zero(local.external_rounds_state[HFROUNDS][i].clone() - state[i].clone());
    }
}

// ============================================================================
// Linear layer helpers (FullAirBuilder compatible)
// ============================================================================

#[inline]
fn apply_mat4_full_air<AB: FullAirBuilder>(x: &mut [AB::VarMaybeExt; 4])
where
    AB::VarMaybeExt: Clone,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123 + x[3].clone();
    x[3] = t01233.clone() + x[0].clone() + x[0].clone();
    x[1] = t01123.clone() + x[2].clone() + x[2].clone();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

fn external_linear_layer_full_air<AB: FullAirBuilder>(state: &mut [AB::VarMaybeExt; WIDTH])
where
    AB::VarMaybeExt: Clone,
{
    for i in (0..WIDTH).step_by(4) {
        let mut chunk =
            [state[i].clone(), state[i + 1].clone(), state[i + 2].clone(), state[i + 3].clone()];
        apply_mat4_full_air::<AB>(&mut chunk);
        state[i] = chunk[0].clone();
        state[i + 1] = chunk[1].clone();
        state[i + 2] = chunk[2].clone();
        state[i + 3] = chunk[3].clone();
    }

    let sums: [AB::VarMaybeExt; 4] = core::array::from_fn(|k| {
        let mut acc = state[k].clone();
        for j in (4..WIDTH).step_by(4) {
            acc = acc + state[j + k].clone();
        }
        acc
    });
    for i in 0..WIDTH {
        state[i] = state[i].clone() + sums[i % 4].clone();
    }
}

fn internal_linear_layer_full_air<AB: FullAirBuilder>(input: &mut [AB::VarMaybeExt; WIDTH])
where
    AB::VarMaybeExt: Clone,
{
    let mut part_sum = input[1].clone();
    for i in 2..WIDTH {
        part_sum = part_sum + input[i].clone();
    }
    let full_sum = part_sum.clone() + input[0].clone();

    // Uniform formula: input[i] = full_sum + INTERNAL_DIAG_KB_24[i] * input[i]
    // (special-cased for i=0,1,2 to avoid a field multiplication by a small constant).
    input[0] = part_sum - input[0].clone(); // diag[0] = -2 ⇒ full_sum - 2*input[0]
    input[1] = full_sum.clone() + input[1].clone(); // diag[1] = 1
    input[2] = full_sum.clone() + input[2].clone() + input[2].clone(); // diag[2] = 2

    for i in 3..WIDTH {
        let diag = AB::VarMaybeExt::from(AB::F::from_canonical_u32(INTERNAL_DIAG_KB_24[i]));
        input[i] = full_sum.clone() + input[i].clone() * diag;
    }
}
