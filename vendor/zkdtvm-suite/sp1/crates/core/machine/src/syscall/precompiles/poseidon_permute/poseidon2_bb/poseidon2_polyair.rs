//! PolyAir gate constraints for the inner Poseidon2 permutation.
//!
//! The inner Poseidon2 AIR has **0 lookup interactions** — it is pure gate constraints.
//! This module provides a single function `poseidon2_inner_gate_constraints` that
//! reproduces every `assert_eq` from the original `air.rs::eval()`.
//!
//! ## Constraint Summary
//!
//! - **Full rounds** (4 beginning + 4 ending = 8): Each full round has WIDTH=24 S-box constraints +
//!   24 post-state assertions = 48 per round. Total: 8 × 48 = 384 constraints.
//!
//! - **Partial rounds** (21): Each partial round has 1 S-box constraint + 1 post_sbox assertion = 2
//!   per round. Total: 21 × 2 = 42 constraints.
//!
//! - **Grand total**: 426 gate constraints, 0 interactions.

use dt_stark::air::FullAirBuilder;
use p3_field::AbstractField;

use super::{
    air::INTERNAL_DIAG_BB_24,
    columns::{Poseidon2Cols, HFROUNDS, PROUNDS, WIDTH},
    constants::RoundConstants,
};

// ============================================================================
// Poseidon2 inner gate constraints
// ============================================================================

pub fn poseidon2_inner_gate_constraints<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::VarMaybeExt>,
    constants: &RoundConstants<AB::F>,
) where
    AB::VarMaybeExt: Clone,
{
    let mut state: [AB::VarMaybeExt; WIDTH] = local.inputs.clone();
    external_linear_layer_full_air::<AB>(&mut state);

    for round in 0..HFROUNDS {
        eval_full_round_gate::<AB>(
            builder,
            &mut state,
            &local.beginning_full_rounds[round],
            &constants.beginning_full_round_constants[round],
        );
    }

    for round in 0..PROUNDS {
        eval_partial_round_gate::<AB>(
            builder,
            &mut state,
            &local.partial_rounds[round],
            &constants.partial_round_constants[round],
        );
    }

    for round in 0..HFROUNDS {
        eval_full_round_gate::<AB>(
            builder,
            &mut state,
            &local.ending_full_rounds[round],
            &constants.ending_full_round_constants[round],
        );
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

    input[0] = part_sum - input[0].clone();
    input[1] = full_sum.clone() + input[1].clone();
    input[2] = full_sum.clone() + input[2].clone() + input[2].clone();

    for i in 3..WIDTH {
        let diag = AB::VarMaybeExt::from(AB::F::from_canonical_u32(INTERNAL_DIAG_BB_24[i]));
        input[i] = full_sum.clone() + input[i].clone() * diag;
    }
}

// ============================================================================
// Round evaluation helpers
// ============================================================================

fn eval_full_round_gate<AB: FullAirBuilder>(
    builder: &mut AB,
    state: &mut [AB::VarMaybeExt; WIDTH],
    full_round: &super::FullRound<AB::VarMaybeExt>,
    round_constants: &[AB::F; WIDTH],
) where
    AB::VarMaybeExt: Clone,
{
    for i in 0..WIDTH {
        let x = state[i].clone() + AB::VarMaybeExt::from(round_constants[i]);

        let committed_x3 = full_round.sbox[i].0[0].clone();
        let x_cubed = x.clone() * x.clone() * x.clone();
        builder.assert_zero(committed_x3.clone() - x_cubed);

        state[i] = committed_x3.clone() * committed_x3 * x;
    }

    external_linear_layer_full_air::<AB>(state);

    for i in 0..WIDTH {
        builder.assert_zero(state[i].clone() - full_round.post[i].clone());
        state[i] = full_round.post[i].clone();
    }
}

fn eval_partial_round_gate<AB: FullAirBuilder>(
    builder: &mut AB,
    state: &mut [AB::VarMaybeExt; WIDTH],
    partial_round: &super::PartialRound<AB::VarMaybeExt>,
    round_constant: &AB::F,
) where
    AB::VarMaybeExt: Clone,
{
    let x = state[0].clone() + AB::VarMaybeExt::from(*round_constant);

    let committed_x3 = partial_round.sbox.0[0].clone();
    let x_cubed = x.clone() * x.clone() * x.clone();
    builder.assert_zero(committed_x3.clone() - x_cubed);

    state[0] = committed_x3.clone() * committed_x3 * x;

    builder.assert_zero(state[0].clone() - partial_round.post_sbox.clone());
    state[0] = partial_round.post_sbox.clone();

    internal_linear_layer_full_air::<AB>(state);
}
