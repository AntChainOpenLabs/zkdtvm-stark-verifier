use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{num_cols, Poseidon2Cols, HFROUNDS, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH},
    RoundConstants,
};

/// KoalaBear Poseidon2 internal diagonal for width 24 (mat_diag_minus_1 values).
/// KoalaBear::ORDER_U32 = 0x7F000001 = 2130706433
pub(super) const INTERNAL_DIAG_KB_24: [u32; 24] = {
    const O: u32 = 0x7F000001;
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
        O - ((O - 1) >> 5),
        O - ((O - 1) >> 6),
        O - ((O - 1) >> 24),
        (O - 1) >> 8,
        (O - 1) >> 3,
        (O - 1) >> 4,
        (O - 1) >> 5,
        (O - 1) >> 6,
        (O - 1) >> 7,
        (O - 1) >> 9,
        (O - 1) >> 24,
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
    // external_rounds_state[0] must equal inputs
    for i in 0..WIDTH {
        let lhs: AB::Expr = local.external_rounds_state[0][i].into();
        let rhs: AB::Expr = local.inputs[i].into();
        builder.assert_eq(lhs, rhs);
    }

    // First half of external rounds
    for r in 0..HFROUNDS {
        eval_external_round(builder, local, &air.constants, r);
    }

    // Internal rounds
    eval_internal_rounds(builder, local, &air.constants);

    // Second half of external rounds
    for r in HFROUNDS..NUM_EXTERNAL_ROUNDS {
        eval_external_round(builder, local, &air.constants, r);
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

fn eval_external_round<AB: AirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::Var>,
    constants: &RoundConstants<AB::F>,
    r: usize,
) {
    let mut state: [AB::Expr; WIDTH] =
        core::array::from_fn(|i| local.external_rounds_state[r][i].into());

    if r == 0 {
        external_linear_layer(&mut state);
    }

    let round_constants = if r < HFROUNDS {
        &constants.beginning_full_round_constants[r]
    } else {
        &constants.ending_full_round_constants[r - HFROUNDS]
    };

    // Add round constants and apply degree-3 S-box
    let sbox_out: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
        let add_rc = state[i].clone() + round_constants[i];
        add_rc.cube()
    });

    let mut state = sbox_out;
    external_linear_layer(&mut state);

    let next_state: &[AB::Var; WIDTH] = if r == HFROUNDS - 1 {
        &local.internal_rounds_state
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        &local.output_state
    } else {
        &local.external_rounds_state[r + 1]
    };

    for i in 0..WIDTH {
        builder.assert_eq(next_state[i], state[i].clone());
    }
}

fn eval_internal_rounds<AB: AirBuilder>(
    builder: &mut AB,
    local: &Poseidon2Cols<AB::Var>,
    constants: &RoundConstants<AB::F>,
) {
    let mut state: [AB::Expr; WIDTH] =
        core::array::from_fn(|i| local.internal_rounds_state[i].into());

    for r in 0..NUM_INTERNAL_ROUNDS {
        let add_rc: AB::Expr =
            if r == 0 { state[0].clone() } else { local.internal_rounds_s0[r - 1].into() } +
                constants.partial_round_constants[r];

        state[0] = add_rc.cube();
        internal_linear_layer(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_eq(local.internal_rounds_s0[r], state[0].clone());
        }
    }

    // After all internal rounds, constrain against second-half external state
    for i in 0..WIDTH {
        builder.assert_eq(local.external_rounds_state[HFROUNDS][i], state[i].clone());
    }
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

/// Internal linear layer using KoalaBear diagonal.
fn internal_linear_layer<AF: AbstractField>(input: &mut [AF; WIDTH]) {
    let part_sum: AF = input[1..].iter().cloned().sum();
    let full_sum = part_sum.clone() + input[0].clone();

    input[0] = part_sum - input[0].clone();
    input[1] = full_sum.clone() + input[1].clone();
    input[2] = full_sum.clone() + input[2].double();

    input.iter_mut().zip(INTERNAL_DIAG_KB_24).skip(3).for_each(|(val, diag_elem)| {
        *val = full_sum.clone() + val.clone() * AF::from_canonical_u32(diag_elem);
    });
}
