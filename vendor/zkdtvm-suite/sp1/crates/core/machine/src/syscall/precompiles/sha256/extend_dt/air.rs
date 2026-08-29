use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope},
    InteractionKind,
};
use p3_air::{Air, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use typenum::{U3, U4};

use crate::{
    air::MemoryAirBuilder,
    memory::MemoryCols,
    operations_dt::{
        AddNOperationWithoutResult, CompactWord, FixedRotateRightOperation,
        FixedShiftRightOperation, XorNOperation,
    },
    syscall::precompiles::sha256::extend_dt::{ShaExtendChip, ShaExtendCols, NUM_SHA_EXTEND_COLS},
};

use core::borrow::Borrow;
use std::iter::once;

impl<F> BaseAir<F> for ShaExtendChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_COLS
    }
}

impl<AB> Air<AB> for ShaExtendChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendCols<AB::Var> = (*local).borrow();

        let i_start = AB::F::from_canonical_u32(16);
        let num_bytes_in_word = AB::F::from_canonical_u32(size_of::<u32>() as u32);

        // Receive the state
        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(local.i.into()))
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );

        // Send the next state
        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(local.i + AB::Expr::one()))
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaExtend),
            InteractionScope::Local,
        );

        // Read w[i-15].
        builder.eval_memory_access(
            local.shard,
            local.clk + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(15)) * num_bytes_in_word,
            &local.w_i_minus_15,
            local.is_real,
        );

        // Read w[i-2].
        builder.eval_memory_access(
            local.shard,
            local.clk + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(2)) * num_bytes_in_word,
            &local.w_i_minus_2,
            local.is_real,
        );

        // Read w[i-16].
        builder.eval_memory_access(
            local.shard,
            local.clk + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(16)) * num_bytes_in_word,
            &local.w_i_minus_16,
            local.is_real,
        );

        // Read w[i-7].
        builder.eval_memory_access(
            local.shard,
            local.clk + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(7)) * num_bytes_in_word,
            &local.w_i_minus_7,
            local.is_real,
        );

        let w_i_minus_15: CompactWord<AB::Expr> = (*local.w_i_minus_15.value()).into();

        // Compute `s0`.
        // w[i-15] rightrotate 7.
        let w_i_minus_15_rr_7 = FixedRotateRightOperation::<AB::F>::eval(
            local.w_i_minus_15_rr_7,
            builder,
            w_i_minus_15.clone(),
            7,
            local.is_real,
        );

        // w[i-15] rightrotate 18.
        let w_i_minus_15_rr_18 = FixedRotateRightOperation::<AB::F>::eval(
            local.w_i_minus_15_rr_18,
            builder,
            w_i_minus_15.clone(),
            18,
            local.is_real,
        );

        // w[i-15] rightshift 3.
        let w_i_minus_15_rs_3 = FixedShiftRightOperation::<AB::F>::eval(
            &local.w_i_minus_15_rs_3,
            builder,
            w_i_minus_15,
            3,
            local.is_real,
        );

        // s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)
        let s0 = XorNOperation::<AB::F, U3>::eval(
            &local.s0,
            builder,
            [
                CompactWord::<AB::F>::into_word(w_i_minus_15_rr_7, local.w_i_minus_15_rr_7_witness),
                CompactWord::<AB::F>::into_word(
                    w_i_minus_15_rr_18,
                    local.w_i_minus_15_rr_18_witness,
                ),
                CompactWord::<AB::F>::into_word(w_i_minus_15_rs_3, local.w_i_minus_15_rs_3_witness),
            ],
            local.is_real,
        );

        let w_i_minus_2: CompactWord<AB::Expr> = (*local.w_i_minus_2.value()).into();

        // Compute `s1`.
        // w[i-2] rightrotate 17.
        let w_i_minus_2_rr_17 = FixedRotateRightOperation::<AB::F>::eval(
            local.w_i_minus_2_rr_17,
            builder,
            w_i_minus_2.clone(),
            17,
            local.is_real,
        );

        // w[i-2] rightrotate 19.
        let w_i_minus_2_rr_19 = FixedRotateRightOperation::<AB::F>::eval(
            local.w_i_minus_2_rr_19,
            builder,
            w_i_minus_2.clone(),
            19,
            local.is_real,
        );

        // w[i-2] rightshift 10.
        let w_i_minus_2_rs_10 = FixedShiftRightOperation::<AB::F>::eval(
            &local.w_i_minus_2_rs_10,
            builder,
            w_i_minus_2,
            10,
            local.is_real,
        );

        // s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
        let s1 = XorNOperation::<AB::F, U3>::eval(
            &local.s1,
            builder,
            [
                CompactWord::<AB::F>::into_word(w_i_minus_2_rr_17, local.w_i_minus_2_rr_17_witness),
                CompactWord::<AB::F>::into_word(w_i_minus_2_rr_19, local.w_i_minus_2_rr_19_witness),
                CompactWord::<AB::F>::into_word(w_i_minus_2_rs_10, local.w_i_minus_2_rs_10_witness),
            ],
            local.is_real,
        );

        let w_i_minus_16 = (*local.w_i_minus_16.value()).into();
        let s0 = s0.into();
        let w_i_minus_7 = (*local.w_i_minus_7.value()).into();
        let s1 = s1.into();

        // s2 := w[i-16] + s0 + w[i-7] + s1.
        AddNOperationWithoutResult::<AB::F, U4>::eval(
            builder,
            [w_i_minus_16, s0, w_i_minus_7, s1],
            local.w_i.access.value.into(),
            local.is_real,
        );

        // Write `s2` to `w[i]`.
        builder.eval_memory_access(
            local.shard,
            local.clk + (local.i - i_start),
            local.w_ptr + local.i * num_bytes_in_word,
            &local.w_i,
            local.is_real,
        );

        // Assert that is_real is a bool.
        builder.assert_bool(local.is_real);
    }
}
