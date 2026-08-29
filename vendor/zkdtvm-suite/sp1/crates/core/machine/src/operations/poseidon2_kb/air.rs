//! The operations that comprise the Poseidon2 hash function.

use std::array;

use dt_primitives::{
    KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
    MONTY_INVERSE_KOALABEAR, POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY,
};
use dt_stark::air::{FullAirBuilder, MachineAirBuilder};
use p3_air::PairBuilder;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_poseidon2::matmul_internal;

use crate::operations::poseidon2_kb::{
    permutation::Poseidon2Cols, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH,
};

pub trait Poseidon2ConstraintBuilder {
    type F: Field;
    type Var: Clone;
    type Expr: Clone
        + From<Self::F>
        + core::ops::Add<Output = Self::Expr>
        + core::ops::Sub<Output = Self::Expr>
        + core::ops::Mul<Output = Self::Expr>
        + core::ops::Mul<Self::F, Output = Self::Expr>;

    fn lift_var(value: &Self::Var) -> Self::Expr;
    fn assert_eq(&mut self, left: Self::Expr, right: Self::Expr);
}

struct MachineAirAdapter<'a, AB>(&'a mut AB);

impl<AB> Poseidon2ConstraintBuilder for MachineAirAdapter<'_, AB>
where
    AB: MachineAirBuilder + PairBuilder,
{
    type F = AB::F;
    type Var = AB::Var;
    type Expr = AB::Expr;

    fn lift_var(value: &Self::Var) -> Self::Expr {
        value.clone().into()
    }

    fn assert_eq(&mut self, left: Self::Expr, right: Self::Expr) {
        self.0.assert_eq(left, right);
    }
}

struct FullAirAdapter<'a, AB>(&'a mut AB);

impl<AB> Poseidon2ConstraintBuilder for FullAirAdapter<'_, AB>
where
    AB: FullAirBuilder,
{
    type F = AB::F;
    type Var = AB::VarMaybeExt;
    type Expr = AB::VarMaybeExt;

    fn lift_var(value: &Self::Var) -> Self::Expr {
        value.clone()
    }

    fn assert_eq(&mut self, left: Self::Expr, right: Self::Expr) {
        self.0.assert_eq(left, right);
    }
}

fn apply_m_4_generic<AF>(x: &mut [AF; 4])
where
    AF: Clone + core::ops::Add<Output = AF> + core::ops::Sub<Output = AF>,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123.clone() + x[3].clone();
    x[3] = t01233.clone() + x[0].clone() + x[0].clone();
    x[1] = t01123.clone() + x[2].clone() + x[2].clone();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

fn external_linear_layer_generic<AF>(state: &mut [AF; WIDTH])
where
    AF: Clone + core::ops::Add<Output = AF> + core::ops::Sub<Output = AF>,
{
    for j in (0..WIDTH).step_by(4) {
        let mut chunk: [AF; 4] = core::array::from_fn(|i| state[j + i].clone());
        apply_m_4_generic(&mut chunk);
        for i in 0..4 {
            state[j + i] = chunk[i].clone();
        }
    }

    let sums: [AF; 4] = core::array::from_fn(|k| {
        let mut sum = state[k].clone();
        for j in (4..WIDTH).step_by(4) {
            sum = sum + state[j + k].clone();
        }
        sum
    });

    for j in 0..WIDTH {
        state[j] = state[j].clone() + sums[j % 4].clone();
    }
}

fn internal_linear_layer_generic<AF, F>(state: &mut [AF; WIDTH])
where
    AF: Clone
        + From<F>
        + core::ops::Add<Output = AF>
        + core::ops::Mul<Output = AF>
        + core::ops::Mul<F, Output = AF>,
    F: Field,
{
    // Keep constants in base field F so that AF * F uses the efficient
    // component-wise path (7 base muls) instead of AF * AF (61 base muls).
    let diag_constants: [F; WIDTH] = core::array::from_fn(|i| {
        F::from_canonical_u32(
            POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY[i].as_canonical_u32(),
        )
    });
    let monty_inverse: F = F::from_canonical_u32(MONTY_INVERSE_KOALABEAR.as_canonical_u32());

    // Compute sum of ORIGINAL state values first (matching matmul_internal's (1 + diag(v)) * state)
    let mut sum = AF::from(F::zero());
    for i in 0..WIDTH {
        sum = sum + state[i].clone();
    }
    // result[i] = (state[i] * diag[i] + sum) * monty_inverse
    for i in 0..WIDTH {
        state[i] = (state[i].clone() * diag_constants[i] + sum.clone()) * monty_inverse;
    }
}

/// Apply the Poseidon2 `M_4` matrix to `x`.
pub fn apply_m_4_mut<AF>(x: &mut [AF])
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

/// Apply a Poseidon2 external linear layer in-place.
pub fn external_linear_layer_mut<AF: AbstractField>(state: &mut [AF; WIDTH]) {
    external_linear_layer_generic(state);
}

/// Apply a Poseidon2 external linear layer.
pub fn external_linear_layer<AF: AbstractField + Copy>(state: &[AF; WIDTH]) -> [AF; WIDTH] {
    let mut state = *state;
    external_linear_layer_mut(&mut state);
    state
}

/// Apply a Poseidon2 internal linear layer in-place.
///
/// This must produce the same result as `DiffusionMatrixKoalaBear::permute_mut`.
/// The diagonal matrix constants are stored in Montgomery form (`_MONTY` suffix).
/// We convert them to canonical form first via `as_canonical_u32()`, then construct
/// the target field element via `from_canonical_u32()` (not `from_wrapped_u32()`).
/// Using `from_wrapped_u32()` would incorrectly treat the canonical value as a raw
/// Montgomery representation, producing wrong constants.
pub fn internal_linear_layer_mut<F: AbstractField>(state: &mut [F; WIDTH]) {
    let matmul_constants: [<F as AbstractField>::F; WIDTH] =
        POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY
            .iter()
            .map(|x| <F as AbstractField>::F::from_canonical_u32(x.as_canonical_u32()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    matmul_internal(state, matmul_constants);
    let monty_inverse = F::from_canonical_u32(MONTY_INVERSE_KOALABEAR.as_canonical_u32());
    for i in state {
        *i = i.clone() * monty_inverse.clone();
    }
}

pub fn eval_external_round_core<B>(builder: &mut B, local_row: &dyn Poseidon2Cols<B::Var>, r: usize)
where
    B: Poseidon2ConstraintBuilder,
{
    let mut local_state: [B::Expr; WIDTH] =
        array::from_fn(|i| B::lift_var(&local_row.external_rounds_state()[r][i]));

    // For the first round, apply the linear layer.
    if r == 0 {
        external_linear_layer_generic(&mut local_state);
    }

    // Add the round constants.
    let add_rc: [B::Expr; WIDTH] = array::from_fn(|i| {
        local_state[i].clone() +
            if r < NUM_EXTERNAL_ROUNDS / 2 {
                B::Expr::from(B::F::from_canonical_u32(
                    KoalaBear_BEGIN_EXT_CONSTS[r][i].as_canonical_u32(),
                ))
            } else {
                B::Expr::from(B::F::from_canonical_u32(
                    KoalaBear_END_EXT_CONSTS[r - NUM_EXTERNAL_ROUNDS / 2][i].as_canonical_u32(),
                ))
            }
    });

    // Apply the sboxes.
    // See `populate_external_round` for why we don't have columns for the sbox output here.
    let mut sbox_deg_3: [B::Expr; WIDTH] = core::array::from_fn(|_| B::Expr::from(B::F::zero()));
    for i in 0..WIDTH {
        sbox_deg_3[i] = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
    }

    // Apply the linear layer.
    let mut state = sbox_deg_3;
    external_linear_layer_generic(&mut state);

    let next_state: [B::Expr; WIDTH] = if r == (NUM_EXTERNAL_ROUNDS / 2) - 1 {
        array::from_fn(|i| B::lift_var(&local_row.internal_rounds_state()[i]))
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        array::from_fn(|i| B::lift_var(&local_row.perm_output()[i]))
    } else {
        array::from_fn(|i| B::lift_var(&local_row.external_rounds_state()[r + 1][i]))
    };

    for i in 0..WIDTH {
        builder.assert_eq(next_state[i].clone(), state[i].clone());
    }
}

pub fn eval_internal_rounds_core<B>(builder: &mut B, local_row: &dyn Poseidon2Cols<B::Var>)
where
    B: Poseidon2ConstraintBuilder,
{
    let mut state: [B::Expr; WIDTH] =
        core::array::from_fn(|i| B::lift_var(&local_row.internal_rounds_state()[i]));
    for r in 0..NUM_INTERNAL_ROUNDS {
        // Add the round constant.
        let add_rc = if r == 0 {
            state[0].clone()
        } else {
            B::lift_var(&local_row.internal_rounds_s0()[r - 1])
        } + B::Expr::from(B::F::from_canonical_u32(
            KoalaBear_PARTIAL_CONSTS[r].as_canonical_u32(),
        ));

        // Apply the linear layer.
        // See `populate_internal_rounds` for why we don't have columns for the new state here.
        state[0] = add_rc.clone() * add_rc.clone() * add_rc.clone();
        internal_linear_layer_generic::<B::Expr, B::F>(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_eq(B::lift_var(&local_row.internal_rounds_s0()[r]), state[0].clone());
        }
    }

    for i in 0..WIDTH {
        builder.assert_eq(
            B::lift_var(&local_row.external_rounds_state()[NUM_EXTERNAL_ROUNDS / 2][i]),
            state[i].clone(),
        );
    }
}

pub fn eval_external_round<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>, r: usize)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let mut adapter = MachineAirAdapter(builder);
    eval_external_round_core(&mut adapter, local_row, r);
}

pub fn eval_internal_rounds<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let mut adapter = MachineAirAdapter(builder);
    eval_internal_rounds_core(&mut adapter, local_row);
}

pub fn eval_poseidon2<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>)
where
    AB: MachineAirBuilder + PairBuilder,
{
    for r in 0..NUM_EXTERNAL_ROUNDS {
        eval_external_round(builder, local_row, r);
    }
    eval_internal_rounds(builder, local_row);
}

pub fn eval_external_round_full<AB>(
    builder: &mut AB,
    local_row: &dyn Poseidon2Cols<AB::VarMaybeExt>,
    r: usize,
) where
    AB: FullAirBuilder,
{
    let mut adapter = FullAirAdapter(builder);
    eval_external_round_core(&mut adapter, local_row, r);
}

pub fn eval_internal_rounds_full<AB>(
    builder: &mut AB,
    local_row: &dyn Poseidon2Cols<AB::VarMaybeExt>,
) where
    AB: FullAirBuilder,
{
    let mut adapter = FullAirAdapter(builder);
    eval_internal_rounds_core(&mut adapter, local_row);
}

pub fn eval_poseidon2_full<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::VarMaybeExt>)
where
    AB: FullAirBuilder,
{
    for r in 0..NUM_EXTERNAL_ROUNDS {
        eval_external_round_full(builder, local_row, r);
    }
    eval_internal_rounds_full(builder, local_row);
}
