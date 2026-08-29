use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{num_cols, Poseidon2Cols, HFROUNDS, PROUNDS, WIDTH},
    FullRound, PartialRound, RoundConstants, SBox,
};

/// BabyBear Poseidon2 internal diagonal for width 24 (canonical u32 values).
/// BabyBear::ORDER_U32 = 0x78000001 = 2013265921
pub(super) const INTERNAL_DIAG_BB_24: [u32; 24] = {
    const O: u32 = 0x78000001;
    [
        O - 2,
        1,
        2,
        (O + 1) >> 1,
        3,
        4,
        (O - 1) >> 1,
        O - 3,
        O - 4,
        O - ((O - 1) >> 8),
        O - ((O - 1) >> 2),
        O - ((O - 1) >> 3),
        O - ((O - 1) >> 4),
        O - ((O - 1) >> 7),
        O - ((O - 1) >> 9),
        O - 15,
        (O - 1) >> 8,
        (O - 1) >> 2,
        (O - 1) >> 3,
        (O - 1) >> 4,
        (O - 1) >> 5,
        (O - 1) >> 6,
        (O - 1) >> 7,
        15,
    ]
};

#[derive(Debug)]
pub struct Poseidon2Air<F: Field> {
    pub(crate) constants: RoundConstants<F>,
}

impl<F: Field> Default for Poseidon2Air<F> {
    fn default() -> Self {
        Self { constants: RoundConstants::<F>::default() }
    }
}

impl<F: Field> Poseidon2Air<F> {
    pub const fn new(constants: RoundConstants<F>) -> Self {
        Self { constants }
    }
}

impl<F: Field> BaseAir<F> for Poseidon2Air<F> {
    fn width(&self) -> usize {
        num_cols()
    }
}

pub(crate) fn eval<AB: AirBuilder>(
    air: &Poseidon2Air<AB::F>,
    builder: &mut AB,
    local: &Poseidon2Cols<AB::Var>,
) {
    let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

    external_linear_layer(&mut state);

    for round in 0..HFROUNDS {
        eval_full_round(
            &mut state,
            &local.beginning_full_rounds[round],
            &air.constants.beginning_full_round_constants[round],
            builder,
        );
    }

    for round in 0..PROUNDS {
        eval_partial_round(
            &mut state,
            &local.partial_rounds[round],
            &air.constants.partial_round_constants[round],
            builder,
        );
    }

    for round in 0..HFROUNDS {
        eval_full_round(
            &mut state,
            &local.ending_full_rounds[round],
            &air.constants.ending_full_round_constants[round],
            builder,
        );
    }
}

impl<AB: AirBuilder> Air<AB> for Poseidon2Air<AB::F> {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2Cols<AB::Var> = (*local).borrow();
        eval::<AB>(self, builder, local);
    }
}

#[inline]
fn eval_full_round<AB: AirBuilder>(
    state: &mut [AB::Expr; WIDTH],
    full_round: &FullRound<AB::Var>,
    round_constants: &[AB::F; WIDTH],
    builder: &mut AB,
) {
    for (i, (s, r)) in state.iter_mut().zip(round_constants.iter()).enumerate() {
        *s = s.clone() + *r;
        eval_sbox(&full_round.sbox[i], s, builder);
    }
    external_linear_layer(state);
    for (state_i, post_i) in state.iter_mut().zip(full_round.post) {
        builder.assert_eq(state_i.clone(), post_i);
        *state_i = post_i.into();
    }
}

#[inline]
fn eval_partial_round<AB: AirBuilder>(
    state: &mut [AB::Expr; WIDTH],
    partial_round: &PartialRound<AB::Var>,
    round_constant: &AB::F,
    builder: &mut AB,
) {
    state[0] = state[0].clone() + *round_constant;
    eval_sbox(&partial_round.sbox, &mut state[0], builder);

    builder.assert_eq(state[0].clone(), partial_round.post_sbox);
    state[0] = partial_round.post_sbox.into();

    internal_linear_layer(state);
}

/// Degree-7 S-box with 1 intermediate register storing x^3.
#[inline]
fn eval_sbox<AB: AirBuilder>(sbox: &SBox<AB::Var>, x: &mut AB::Expr, builder: &mut AB) {
    let committed_x3: AB::Expr = sbox.0[0].into();
    builder.assert_eq(committed_x3.clone(), x.cube());
    *x = committed_x3.square() * x.clone();
}

/// External linear layer: M4 circ on width-24 state.
fn external_linear_layer<AF: AbstractField>(state: &mut [AF; WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        let t01 = chunk[0].clone() + chunk[1].clone();
        let t23 = chunk[2].clone() + chunk[3].clone();
        let t0123 = t01.clone() + t23.clone();
        let t01123 = t0123.clone() + chunk[1].clone();
        let t01233 = t0123.clone() + chunk[3].clone();
        chunk[3] = t01233.clone() + chunk[0].double();
        chunk[1] = t01123.clone() + chunk[2].double();
        chunk[0] = t01123 + t01;
        chunk[2] = t01233 + t23;
    }
    let sums: [AF; 4] =
        core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k].clone()).sum::<AF>());
    state.iter_mut().enumerate().for_each(|(i, elem)| *elem += sums[i % 4].clone());
}

/// Internal linear layer using BabyBear diagonal.
fn internal_linear_layer<AF: AbstractField>(input: &mut [AF; WIDTH]) {
    let part_sum: AF = input[1..].iter().cloned().sum();
    let full_sum = part_sum.clone() + input[0].clone();

    input[0] = part_sum - input[0].clone();
    input[1] = full_sum.clone() + input[1].clone();
    input[2] = full_sum.clone() + input[2].double();

    input.iter_mut().zip(INTERNAL_DIAG_BB_24).skip(3).for_each(|(val, diag_elem)| {
        *val = full_sum.clone() + val.clone() * AF::from_canonical_u32(diag_elem);
    });
}
