use std::{array, borrow::Borrow};

use dt_primitives::{
    KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{AbstractField, PrimeField32};
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
        let local_row: &Poseidon2<_> = (*local_row).borrow();
        let prepr = builder.preprocessed();
        let prep_local = prepr.row_slice(0);
        let prep_local: &Poseidon2PreprocessedCols<_> = (*prep_local).borrow();

        // One-hot row selectors.
        let is_round: [AB::Expr; 5] = array::from_fn(|r| prep_local.is_round[r].into());

        // Derived selectors.
        let is_real: AB::Expr = is_round.iter().cloned().sum();
        let is_internal: AB::Expr = is_round[2].clone();

        // ------------------------------------------------------------------
        // 1. Memory interactions (send-only convention).
        // ------------------------------------------------------------------
        let neg_is_real: AB::Expr = AB::Expr::zero() - is_real.clone();
        for i in 0..WIDTH {
            builder.send_single(
                prep_local.state_in_addrs[i],
                local_row.state_in[i],
                neg_is_real.clone(),
            );
            builder.send_single(
                prep_local.state_out_mem[i].addr,
                local_row.state_out[i],
                prep_local.state_out_mem[i].mult,
            );
        }

        // ------------------------------------------------------------------
        // 2. External rounds: each external row folds two rounds.
        //    Row mapping:
        //      is_round[0] → external pair 0 (BEGIN_EXT_CONSTS[0], [1])
        //      is_round[1] → external pair 1 (BEGIN_EXT_CONSTS[2], [3])
        //      is_round[3] → external pair 2 (END_EXT_CONSTS[0], [1])
        //      is_round[4] → external pair 3 (END_EXT_CONSTS[2], [3])
        // ------------------------------------------------------------------
        let external_pairs: [(usize, bool, usize); 4] = [
            (0, true, 0),  // is_round[0], first_half, table_idx=0
            (1, true, 2),  // is_round[1], first_half, table_idx=2
            (3, false, 0), // is_round[3], second_half, table_idx=0
            (4, false, 2), // is_round[4], second_half, table_idx=2
        ];

        for (round_sel_idx, first_half, table_idx) in external_pairs {
            let selector = is_round[round_sel_idx].clone();

            let first_rc = |i: usize| -> AB::Expr {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx][i]
                };
                AB::Expr::from_canonical_u32(c.as_canonical_u32())
            };
            let second_rc = |i: usize| -> AB::Expr {
                let c = if first_half {
                    KoalaBear_BEGIN_EXT_CONSTS[table_idx + 1][i]
                } else {
                    KoalaBear_END_EXT_CONSTS[table_idx + 1][i]
                };
                AB::Expr::from_canonical_u32(c.as_canonical_u32())
            };

            // First round input: on the very first row (is_round[0]) apply initial linear layer.
            let mut first_input: [AB::Expr; WIDTH] =
                array::from_fn(|i| local_row.state_in[i].into());
            if round_sel_idx == 0 {
                external_linear_layer(&mut first_input);
            }

            // First round: +RC → S-box(x^3) → external_linear_layer → round_witness
            let first_add_rc: [AB::Expr; WIDTH] =
                array::from_fn(|i| first_input[i].clone() + first_rc(i));
            let first_sbox: [AB::Expr; WIDTH] = array::from_fn(|i| {
                first_add_rc[i].clone() * first_add_rc[i].clone() * first_add_rc[i].clone()
            });
            let mut first_out = first_sbox;
            external_linear_layer(&mut first_out);
            for i in 0..WIDTH {
                builder
                    .when(selector.clone())
                    .assert_eq(local_row.round_witness[i].into(), first_out[i].clone());
            }

            // Second round: round_witness → +RC → S-box(x^3) → external_linear_layer → state_out
            let second_add_rc: [AB::Expr; WIDTH] =
                array::from_fn(|i| local_row.round_witness[i].into() + second_rc(i));
            let second_sbox: [AB::Expr; WIDTH] = array::from_fn(|i| {
                second_add_rc[i].clone() * second_add_rc[i].clone() * second_add_rc[i].clone()
            });
            let mut second_out = second_sbox;
            external_linear_layer(&mut second_out);
            for i in 0..WIDTH {
                builder
                    .when(selector.clone())
                    .assert_eq(local_row.state_out[i].into(), second_out[i].clone());
            }
        }

        // ------------------------------------------------------------------
        // 3. Internal rounds: fold all 20 rounds into one row.
        //    Rounds 0..18: use round_witness[k] as witness for (state[0]+RC[k])^3
        //    Round 19: compute inline (no witness needed, degree stays at 3).
        // ------------------------------------------------------------------
        {
            let internal_selector = is_internal.clone();
            let mut state: [AB::Expr; WIDTH] = array::from_fn(|i| local_row.state_in[i].into());

            for k in 0..(NUM_INTERNAL_ROUNDS - 1) {
                let rc_k: AB::Expr =
                    AB::Expr::from_canonical_u32(KoalaBear_PARTIAL_CONSTS[k].as_canonical_u32());
                let sbox_in = state[0].clone() + rc_k;
                let sbox_out_expected: AB::Expr =
                    sbox_in.clone() * sbox_in.clone() * sbox_in.clone();
                builder
                    .when(internal_selector.clone())
                    .assert_eq(local_row.round_witness[k].into(), sbox_out_expected);
                state[0] = local_row.round_witness[k].into();
                internal_linear_layer(&mut state);
            }

            // Last internal round (k = NUM_INTERNAL_ROUNDS - 1): inline without witness.
            {
                let rc_last: AB::Expr = AB::Expr::from_canonical_u32(
                    KoalaBear_PARTIAL_CONSTS[NUM_INTERNAL_ROUNDS - 1].as_canonical_u32(),
                );
                let sbox_in = state[0].clone() + rc_last;
                state[0] = sbox_in.clone() * sbox_in.clone() * sbox_in.clone();
                internal_linear_layer(&mut state);
            }

            for i in 0..WIDTH {
                builder
                    .when(internal_selector.clone())
                    .assert_eq(local_row.state_out[i].into(), state[i].clone());
            }
        }
    }
}
