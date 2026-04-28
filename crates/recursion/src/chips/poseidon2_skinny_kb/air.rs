use std::{array, borrow::Borrow};

use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{builder::DTRecursionAirBuilder, chips::poseidon2_skinny_kb::columns::Poseidon2};

use super::{
    columns::{preprocessed::Poseidon2PreprocessedCols, NUM_POSEIDON2_COLS},
    external_linear_layer, internal_linear_layer, Poseidon2SkinnyKbChip, NUM_INTERNAL_ROUNDS,
    WIDTH,
};

impl<F, const DEGREE: usize> BaseAir<F> for Poseidon2SkinnyKbChip<DEGREE> {
    fn width(&self) -> usize {
        assert!(DEGREE >= 3);
        NUM_POSEIDON2_COLS
    }
}

impl<AB, const DEGREE: usize> Air<AB> for Poseidon2SkinnyKbChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
    AB::Var: 'static,
{
    fn eval(&self, builder: &mut AB) {
        assert!(DEGREE >= 3);

        let main = builder.main();
        let local_row = main.row_slice(0);
        let next_row_idx = if main.height() > 1 { 1 } else { 0 };
        let next_row = main.row_slice(next_row_idx);
        let local_row: &Poseidon2<_> = (*local_row).borrow();
        let next_row: &Poseidon2<_> = (*next_row).borrow();
        let prepr = builder.preprocessed();
        let prep_local = prepr.row_slice(0);
        let prep_local: &Poseidon2PreprocessedCols<_> = (*prep_local).borrow();

        let lhs = (0..DEGREE).map(|_| local_row.state_var[0].into()).product::<AB::Expr>();
        let rhs = (0..DEGREE).map(|_| local_row.state_var[0].into()).product::<AB::Expr>();
        builder.assert_eq(lhs, rhs);

        (0..WIDTH).for_each(|i| {
            builder.send_single(
                prep_local.memory_preprocessed[i].addr,
                local_row.state_var[i],
                prep_local.memory_preprocessed[i].mult,
            )
        });

        self.eval_input_round(builder, local_row, prep_local, next_row);
        self.eval_external_round(builder, local_row, prep_local, next_row);
        self.eval_internal_rounds(
            builder,
            local_row,
            next_row,
            prep_local.round_counters_preprocessed.round_constants,
            prep_local.round_counters_preprocessed.is_internal_round,
        );
    }
}

impl<const DEGREE: usize> Poseidon2SkinnyKbChip<DEGREE> {
    fn eval_input_round<AB: DTRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        prep_local: &Poseidon2PreprocessedCols<AB::Var>,
        next_row: &Poseidon2<AB::Var>,
    ) {
        let mut state: [AB::Expr; WIDTH] = array::from_fn(|i| local_row.state_var[i].into());
        external_linear_layer(&mut state);

        let next_state = next_row.state_var;
        for i in 0..WIDTH {
            builder
                .when_not(builder.is_last_row())
                .when(prep_local.round_counters_preprocessed.is_input_round)
                .assert_eq(next_state[i], state[i].clone());
        }
    }

    fn eval_external_round<AB: DTRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        prep_local: &Poseidon2PreprocessedCols<AB::Var>,
        next_row: &Poseidon2<AB::Var>,
    ) {
        let local_state = local_row.state_var;

        let add_rc: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
            local_state[i].into() + prep_local.round_counters_preprocessed.round_constants[i]
        });

        // KoalaBear S-box: x^3
        let mut sbox_result: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::zero());
        for i in 0..WIDTH {
            sbox_result[i] = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
        }

        let mut state = sbox_result;
        external_linear_layer(&mut state);

        let next_state = next_row.state_var;
        for i in 0..WIDTH {
            builder
                .when_not(builder.is_last_row())
                .when(prep_local.round_counters_preprocessed.is_external_round)
                .assert_eq(next_state[i], state[i].clone());
        }
    }

    fn eval_internal_rounds<AB: DTRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        next_row: &Poseidon2<AB::Var>,
        round_constants: [AB::Var; WIDTH],
        is_internal_row: AB::Var,
    ) {
        let local_state = local_row.state_var;

        let s0 = local_row.internal_rounds_s0;
        let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| local_state[i].into());
        for r in 0..NUM_INTERNAL_ROUNDS {
            let add_rc =
                if r == 0 { state[0].clone() } else { s0[r - 1].into() } + round_constants[r];

            // KoalaBear S-box: x^3
            let sbox_result = add_rc.clone() * add_rc.clone() * add_rc.clone();

            state[0] = sbox_result.clone();
            internal_linear_layer(&mut state);

            if r < NUM_INTERNAL_ROUNDS - 1 {
                builder.when(is_internal_row).assert_eq(s0[r], state[0].clone());
            }
        }

        let next_state = next_row.state_var;
        for i in 0..WIDTH {
            builder.when(is_internal_row).assert_eq(next_state[i], state[i].clone())
        }
    }
}
