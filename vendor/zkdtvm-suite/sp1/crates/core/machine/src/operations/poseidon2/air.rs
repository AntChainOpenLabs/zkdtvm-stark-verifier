use std::array;

use dt_primitives::RC_16_30_U32;
use dt_stark::air::MachineAirBuilder;
use p3_air::PairBuilder;
use p3_baby_bear::{MONTY_INVERSE, POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY};
use p3_field::{AbstractField, PrimeField32};
use p3_poseidon2::matmul_internal;

use super::{permutation::Poseidon2Cols, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH};

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

pub fn external_linear_layer_mut<AF: AbstractField>(state: &mut [AF; WIDTH]) {
    for j in (0..WIDTH).step_by(4) {
        apply_m_4_mut(&mut state[j..j + 4]);
    }
    let sums: [AF; 4] =
        core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k].clone()).sum::<AF>());

    for j in 0..WIDTH {
        state[j] = state[j].clone() + sums[j % 4].clone();
    }
}

pub fn external_linear_layer<AF: AbstractField + Copy>(state: &[AF; WIDTH]) -> [AF; WIDTH] {
    let mut state = *state;
    external_linear_layer_mut(&mut state);
    state
}

pub fn internal_linear_layer_mut<F: AbstractField>(state: &mut [F; WIDTH]) {
    let matmul_constants: [<F as AbstractField>::F; WIDTH] =
        POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY
            .iter()
            .map(|x| <F as AbstractField>::F::from_wrapped_u32(x.as_canonical_u32()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    matmul_internal(state, matmul_constants);
    let monty_inverse = F::from_wrapped_u32(MONTY_INVERSE.as_canonical_u32());
    state.iter_mut().for_each(|i| *i = i.clone() * monty_inverse.clone());
}

/// Eval the constraints for the external rounds.
pub fn eval_external_round<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>, r: usize)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let mut local_state: [AB::Expr; WIDTH] =
        array::from_fn(|i| local_row.external_rounds_state()[r][i].into());

    // For the first round, apply the linear layer.
    if r == 0 {
        external_linear_layer_mut(&mut local_state);
    }

    // Add the round constants.
    let round = if r < NUM_EXTERNAL_ROUNDS / 2 { r } else { r + NUM_INTERNAL_ROUNDS };
    let add_rc: [AB::Expr; WIDTH] = array::from_fn(|i| {
        local_state[i].clone() + AB::F::from_wrapped_u32(RC_16_30_U32[round][i])
    });

    // Apply the sboxes.
    // See `populate_external_round` for why we don't have columns for the sbox output here.
    let mut sbox_deg_7: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::zero());
    let mut sbox_deg_3: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::zero());
    for i in 0..WIDTH {
        let calculated_sbox_deg_3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();

        if let Some(external_sbox) = local_row.external_rounds_sbox() {
            builder.assert_eq(external_sbox[r][i].into(), calculated_sbox_deg_3);
            sbox_deg_3[i] = external_sbox[r][i].into();
        } else {
            sbox_deg_3[i] = calculated_sbox_deg_3;
        }

        sbox_deg_7[i] = sbox_deg_3[i].clone() * sbox_deg_3[i].clone() * add_rc[i].clone();
    }

    // Apply the linear layer.
    let mut state = sbox_deg_7;
    external_linear_layer_mut(&mut state);

    let next_state = if r == (NUM_EXTERNAL_ROUNDS / 2) - 1 {
        local_row.internal_rounds_state()
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        local_row.perm_output()
    } else {
        &local_row.external_rounds_state()[r + 1]
    };

    for i in 0..WIDTH {
        builder.assert_eq(next_state[i], state[i].clone());
    }
}

/// Eval the constraints for the internal rounds.
pub fn eval_internal_rounds<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let state = &local_row.internal_rounds_state();
    let s0 = local_row.internal_rounds_s0();
    let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| state[i].into());
    for r in 0..NUM_INTERNAL_ROUNDS {
        // Add the round constant.
        let round = r + NUM_EXTERNAL_ROUNDS / 2;
        let add_rc = if r == 0 { state[0].clone() } else { s0[r - 1].into() } +
            AB::Expr::from_wrapped_u32(RC_16_30_U32[round][0]);

        let mut sbox_deg_3 = add_rc.clone() * add_rc.clone() * add_rc.clone();
        if let Some(internal_sbox) = local_row.internal_rounds_sbox() {
            builder.assert_eq(internal_sbox[r], sbox_deg_3);
            sbox_deg_3 = internal_sbox[r].into();
        }

        // See `populate_internal_rounds` for why we don't have columns for the sbox output
        // here.
        let sbox_deg_7 = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc.clone();

        // Apply the linear layer.
        // See `populate_internal_rounds` for why we don't have columns for the new state here.
        state[0] = sbox_deg_7.clone();
        internal_linear_layer_mut(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_eq(s0[r], state[0].clone());
        }
    }

    let external_state = local_row.external_rounds_state()[NUM_EXTERNAL_ROUNDS / 2];
    for i in 0..WIDTH {
        builder.assert_eq(external_state[i], state[i].clone())
    }
}

use dt_stark::air::FullAirBuilder;

pub struct Poseidon2ColsViewBb<T: Clone> {
    pub external_rounds_state: Vec<[T; WIDTH]>,
    pub internal_rounds_state: [T; WIDTH],
    pub internal_rounds_s0: [T; NUM_INTERNAL_ROUNDS - 1],
    pub output_state: [T; WIDTH],
    pub external_rounds_sbox: Option<Vec<[T; WIDTH]>>,
    pub internal_rounds_sbox: Option<Vec<T>>,
}

impl<T: Clone> Poseidon2ColsViewBb<T> {
    pub fn from_degree3_slice(s: &[T]) -> Self {
        let ext_state: Vec<[T; WIDTH]> = (0..NUM_EXTERNAL_ROUNDS)
            .map(|r| core::array::from_fn(|i| s[r * WIDTH + i].clone()))
            .collect();
        let int_offset = NUM_EXTERNAL_ROUNDS * WIDTH;
        let int_state = core::array::from_fn(|i| s[int_offset + i].clone());
        let s0_offset = int_offset + WIDTH;
        let int_s0 = core::array::from_fn(|i| s[s0_offset + i].clone());
        let out_offset = s0_offset + (NUM_INTERNAL_ROUNDS - 1);
        let output_state = core::array::from_fn(|i| s[out_offset + i].clone());
        let sbox_offset = out_offset + WIDTH;
        let ext_sbox: Vec<[T; WIDTH]> = (0..NUM_EXTERNAL_ROUNDS)
            .map(|r| core::array::from_fn(|i| s[sbox_offset + r * WIDTH + i].clone()))
            .collect();
        let int_sbox_offset = sbox_offset + NUM_EXTERNAL_ROUNDS * WIDTH;
        let int_sbox: Vec<T> =
            (0..NUM_INTERNAL_ROUNDS).map(|i| s[int_sbox_offset + i].clone()).collect();
        Self {
            external_rounds_state: ext_state,
            internal_rounds_state: int_state,
            internal_rounds_s0: int_s0,
            output_state,
            external_rounds_sbox: Some(ext_sbox),
            internal_rounds_sbox: Some(int_sbox),
        }
    }

    pub fn from_degree9_slice(s: &[T]) -> Self {
        let ext_state: Vec<[T; WIDTH]> = (0..NUM_EXTERNAL_ROUNDS)
            .map(|r| core::array::from_fn(|i| s[r * WIDTH + i].clone()))
            .collect();
        let int_offset = NUM_EXTERNAL_ROUNDS * WIDTH;
        let int_state = core::array::from_fn(|i| s[int_offset + i].clone());
        let s0_offset = int_offset + WIDTH;
        let int_s0 = core::array::from_fn(|i| s[s0_offset + i].clone());
        let out_offset = s0_offset + (NUM_INTERNAL_ROUNDS - 1);
        let output_state = core::array::from_fn(|i| s[out_offset + i].clone());
        Self {
            external_rounds_state: ext_state,
            internal_rounds_state: int_state,
            internal_rounds_s0: int_s0,
            output_state,
            external_rounds_sbox: None,
            internal_rounds_sbox: None,
        }
    }
}

fn external_linear_layer_generic_bb<T>(state: &mut [T; WIDTH])
where
    T: Clone + core::ops::Add<Output = T>,
{
    for j in (0..WIDTH).step_by(4) {
        let t01 = state[j].clone() + state[j + 1].clone();
        let t23 = state[j + 2].clone() + state[j + 3].clone();
        let t0123 = t01.clone() + t23.clone();
        let t01123 = t0123.clone() + state[j + 1].clone();
        let t01233 = t0123.clone() + state[j + 3].clone();
        state[j + 3] = t01233.clone() + state[j].clone() + state[j].clone();
        state[j + 1] = t01123.clone() + state[j + 2].clone() + state[j + 2].clone();
        state[j] = t01123 + t01;
        state[j + 2] = t01233 + t23;
    }
    let sums: [T; 4] = core::array::from_fn(|k| {
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

fn internal_linear_layer_full_bb<T: AbstractField>(state: &mut [T; WIDTH]) {
    let matmul_constants: [T::F; WIDTH] = POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY
        .iter()
        .map(|x| T::F::from_wrapped_u32(x.as_canonical_u32()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    matmul_internal(state, matmul_constants);
    let monty_inverse = T::from_wrapped_u32(MONTY_INVERSE.as_canonical_u32());
    state.iter_mut().for_each(|i| *i = i.clone() * monty_inverse.clone());
}

pub fn eval_external_round_full<AB>(
    builder: &mut AB,
    view: &Poseidon2ColsViewBb<AB::VarMaybeExt>,
    r: usize,
) where
    AB: FullAirBuilder,
{
    let mut local_state: [AB::VarMaybeExt; WIDTH] =
        core::array::from_fn(|i| view.external_rounds_state[r][i].clone());

    if r == 0 {
        external_linear_layer_generic_bb(&mut local_state);
    }

    let round = if r < NUM_EXTERNAL_ROUNDS / 2 { r } else { r + NUM_INTERNAL_ROUNDS };
    let add_rc: [AB::VarMaybeExt; WIDTH] = core::array::from_fn(|i| {
        local_state[i].clone() + AB::VarMaybeExt::from_wrapped_u32(RC_16_30_U32[round][i])
    });

    let mut sbox_deg_7: [AB::VarMaybeExt; WIDTH] =
        core::array::from_fn(|_| AB::VarMaybeExt::zero());
    let mut sbox_deg_3: [AB::VarMaybeExt; WIDTH] =
        core::array::from_fn(|_| AB::VarMaybeExt::zero());
    for i in 0..WIDTH {
        let calculated_sbox_deg_3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();

        if let Some(ref external_sbox) = view.external_rounds_sbox {
            builder.assert_zero(external_sbox[r][i].clone() - calculated_sbox_deg_3);
            sbox_deg_3[i] = external_sbox[r][i].clone();
        } else {
            sbox_deg_3[i] = calculated_sbox_deg_3;
        }

        sbox_deg_7[i] = sbox_deg_3[i].clone() * sbox_deg_3[i].clone() * add_rc[i].clone();
    }

    let mut state = sbox_deg_7;
    external_linear_layer_generic_bb(&mut state);

    let next_state: [AB::VarMaybeExt; WIDTH] = if r == (NUM_EXTERNAL_ROUNDS / 2) - 1 {
        core::array::from_fn(|i| view.internal_rounds_state[i].clone())
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        core::array::from_fn(|i| view.output_state[i].clone())
    } else {
        core::array::from_fn(|i| view.external_rounds_state[r + 1][i].clone())
    };

    for i in 0..WIDTH {
        builder.assert_zero(next_state[i].clone() - state[i].clone());
    }
}

pub fn eval_internal_rounds_full<AB>(builder: &mut AB, view: &Poseidon2ColsViewBb<AB::VarMaybeExt>)
where
    AB: FullAirBuilder,
{
    let s0 = &view.internal_rounds_s0;
    let mut state: [AB::VarMaybeExt; WIDTH] =
        core::array::from_fn(|i| view.internal_rounds_state[i].clone());

    for r in 0..NUM_INTERNAL_ROUNDS {
        let round = r + NUM_EXTERNAL_ROUNDS / 2;
        let add_rc = if r == 0 { state[0].clone() } else { s0[r - 1].clone() } +
            AB::VarMaybeExt::from_wrapped_u32(RC_16_30_U32[round][0]);

        let mut sbox_deg_3 = add_rc.clone() * add_rc.clone() * add_rc.clone();
        if let Some(ref internal_sbox) = view.internal_rounds_sbox {
            builder.assert_zero(internal_sbox[r].clone() - sbox_deg_3);
            sbox_deg_3 = internal_sbox[r].clone();
        }

        let sbox_deg_7 = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc.clone();

        state[0] = sbox_deg_7;
        internal_linear_layer_full_bb(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_zero(s0[r].clone() - state[0].clone());
        }
    }

    for i in 0..WIDTH {
        builder.assert_zero(
            view.external_rounds_state[NUM_EXTERNAL_ROUNDS / 2][i].clone() - state[i].clone(),
        );
    }
}

pub fn eval_poseidon2_full<AB>(builder: &mut AB, view: &Poseidon2ColsViewBb<AB::VarMaybeExt>)
where
    AB: FullAirBuilder,
{
    for r in 0..NUM_EXTERNAL_ROUNDS {
        eval_external_round_full(builder, view, r);
    }
    eval_internal_rounds_full(builder, view);
}
