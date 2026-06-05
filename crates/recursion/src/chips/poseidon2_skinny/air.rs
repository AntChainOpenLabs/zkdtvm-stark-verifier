use std::{array, borrow::Borrow};

use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{builder::DTRecursionAirBuilder, chips::poseidon2_skinny::columns::Poseidon2};

use super::{
    columns::{preprocessed::Poseidon2PreprocessedCols, NUM_POSEIDON2_COLS},
    external_linear_layer, internal_linear_layer, Poseidon2SkinnyChip, WIDTH,
};

impl<F, const DEGREE: usize> BaseAir<F> for Poseidon2SkinnyChip<DEGREE> {
    fn width(&self) -> usize {
        assert!(DEGREE >= 9);
        NUM_POSEIDON2_COLS
    }
}

impl<AB, const DEGREE: usize> Air<AB> for Poseidon2SkinnyChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
    AB::Var: 'static,
{
    /// Single-row evaluation of one Poseidon2 round.
    ///
    /// Each row carries `state_in` and `state_out` of one round. Cross-round chaining
    /// (i.e. `row[r].state_out == row[r+1].state_in`) is enforced through memory lookups,
    /// not constraints; here we only constrain the per-row transition
    /// `state_out = round(state_in)`.
    fn eval(&self, builder: &mut AB) {
        assert!(DEGREE >= 9);

        let main = builder.main();
        let local_row = main.row_slice(0);
        let local_row: &Poseidon2<_> = (*local_row).borrow();
        let prepr = builder.preprocessed();
        let prep_local = prepr.row_slice(0);
        let prep_local: &Poseidon2PreprocessedCols<_> = (*prep_local).borrow();

        // ------------------------------------------------------------------
        // 1. Memory interactions on the LogUp bus (send-only convention).
        // ------------------------------------------------------------------
        for i in 0..WIDTH {
            builder.send_single(
                prep_local.state_in_addrs[i],
                local_row.state_in[i],
                prep_local.state_in_neg_mult,
            );
            builder.send_single(
                prep_local.state_out_mem[i].addr,
                local_row.state_out[i],
                prep_local.state_out_mem[i].mult,
            );
        }

        let is_real: AB::Expr = prep_local.is_real.into();
        let is_internal: AB::Expr = prep_local.round_kind.into();
        let is_external: AB::Expr = AB::Expr::one() - is_internal.clone();
        let is_first: AB::Expr = prep_local.is_first_round.into();
        let not_first: AB::Expr = AB::Expr::one() - is_first.clone();

        // ------------------------------------------------------------------
        // 2. External round (split into is_first vs !is_first).
        //    BabyBear uses SBOX_DEGREE=7: x^7 = x^3 * x^3 * x.
        // ------------------------------------------------------------------
        // 2a. Branch: external && is_first (absorbs initial linear layer)
        {
            let mut pre: [AB::Expr; WIDTH] = array::from_fn(|i| local_row.state_in[i].into());
            external_linear_layer(&mut pre);
            let add_rc: [AB::Expr; WIDTH] =
                array::from_fn(|i| pre[i].clone() + prep_local.round_constants[i]);
            let sbox: [AB::Expr; WIDTH] = array::from_fn(|i| {
                let deg3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
                deg3.clone() * deg3.clone() * add_rc[i].clone()
            });
            let mut out = sbox;
            external_linear_layer(&mut out);
            let selector = is_real.clone() * is_external.clone() * is_first.clone();
            for i in 0..WIDTH {
                builder
                    .when(selector.clone())
                    .assert_eq(local_row.state_out[i].into(), out[i].clone());
            }
        }
        // 2b. Branch: external && !is_first
        {
            let add_rc: [AB::Expr; WIDTH] = array::from_fn(|i| {
                local_row.state_in[i].into() + prep_local.round_constants[i]
            });
            let sbox: [AB::Expr; WIDTH] = array::from_fn(|i| {
                let deg3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
                deg3.clone() * deg3.clone() * add_rc[i].clone()
            });
            let mut out = sbox;
            external_linear_layer(&mut out);
            let selector = is_real.clone() * is_external.clone() * not_first.clone();
            for i in 0..WIDTH {
                builder
                    .when(selector.clone())
                    .assert_eq(local_row.state_out[i].into(), out[i].clone());
            }
        }

        // ------------------------------------------------------------------
        // 3. Internal round. Only state_in[0] goes through the S-box.
        // ------------------------------------------------------------------
        {
            let mut state: [AB::Expr; WIDTH] = array::from_fn(|i| local_row.state_in[i].into());
            let add_rc0 = state[0].clone() + prep_local.round_constants[0];
            let deg3 = add_rc0.clone() * add_rc0.clone() * add_rc0.clone();
            state[0] = deg3.clone() * deg3.clone() * add_rc0.clone();
            internal_linear_layer(&mut state);
            let selector = is_real.clone() * is_internal.clone();
            for i in 0..WIDTH {
                builder
                    .when(selector.clone())
                    .assert_eq(local_row.state_out[i].into(), state[i].clone());
            }
        }
    }
}
