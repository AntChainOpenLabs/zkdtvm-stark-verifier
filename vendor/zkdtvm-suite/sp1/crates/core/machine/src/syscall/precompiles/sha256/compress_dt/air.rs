use core::borrow::Borrow;
use std::iter::once;

use dt_stark::{
    air::{AirInteraction, DTAirBuilder, InteractionScope},
    InteractionKind, Word,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use typenum::{U2, U3, U5};

use crate::{
    air::MemoryAirBuilder,
    operations_dt::{
        AddNOperation, AndNOperation, CompactWord, FixedRotateRightOperation, NotOperation,
        XorNOperation,
    },
    syscall::precompiles::sha256::{
        compress_dt::{ShaCompressChip, SHA_COMPRESS_K},
        ShaCompressCols, NUM_SHA_COMPRESS_COLS,
    },
};

impl<F> BaseAir<F> for ShaCompressChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_COLS
    }
}

impl<AB> Air<AB> for ShaCompressChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressCols<AB::Var> = (*local).borrow();

        self.eval_control_flow_flags(builder, local);

        self.eval_memory(builder, local);

        self.eval_compression_ops(builder, local);
    }
}

impl ShaCompressChip {
    fn eval_control_flow_flags<AB: DTAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        local
            .i_low_one_hot
            .iter()
            .chain(local.i_high_one_hot.iter())
            .for_each(|b| builder.assert_bool(*b));
        builder.when(local.is_real).assert_eq(
            local.i,
            local
                .i_low_one_hot
                .iter()
                .enumerate()
                .map(|(i, b)| *b * AB::F::from_canonical_u32(i as u32))
                .chain(
                    local
                        .i_high_one_hot
                        .iter()
                        .enumerate()
                        .map(|(i, b)| *b * AB::F::from_canonical_u32((i << 3) as u32)),
                )
                .sum::<AB::Expr>(),
        );
        builder.assert_bool(
            local
                .i_low_one_hot
                .iter()
                .map(|b| <AB::Var as Into<AB::Expr>>::into(*b))
                .sum::<AB::Expr>(),
        );
        builder.assert_bool(
            local
                .i_high_one_hot
                .iter()
                .map(|b| <AB::Var as Into<AB::Expr>>::into(*b))
                .sum::<AB::Expr>(),
        );

        let receive_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(local.i.into()))
            .chain(
                [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h]
                    .into_iter()
                    .flat_map(|h| h.0.into_iter().map(|h| h.into())),
            )
            .collect::<Vec<_>>();
        builder.receive(
            AirInteraction::new(receive_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );

        let send_values = once(local.shard.into())
            .chain(once(local.clk.into()))
            .chain(once(local.w_ptr.into()))
            .chain(once(local.i + AB::F::one()))
            .chain(
                [
                    local.temp1_add_temp2.value,
                    local.a,
                    local.b,
                    local.c,
                    local.d_add_temp1.value,
                    local.e,
                    local.f,
                    local.g,
                ]
                .into_iter()
                .flat_map(|h| h.0.into_iter().map(|h| h.into())),
            )
            .collect::<Vec<_>>();
        builder.send(
            AirInteraction::new(send_values, local.is_real.into(), InteractionKind::ShaCompress),
            InteractionScope::Local,
        );
    }

    /// Constrains that memory address is correct and that memory is correctly written/read.
    fn eval_memory<AB: DTAirBuilder>(&self, builder: &mut AB, local: &ShaCompressCols<AB::Var>) {
        builder.eval_memory_access(
            local.shard,
            local.clk,
            local.w_ptr + local.i * AB::F::from_canonical_u32(size_of::<u32>() as u32),
            &local.w_access,
            local.is_real,
        );

        let one_hot: [AB::Expr; 64] =
            std::array::from_fn(|i| local.i_low_one_hot[i & 0x7] * local.i_high_one_hot[i >> 3]);

        builder.assert_eq(
            one_hot
                .iter()
                .zip(SHA_COMPRESS_K.iter())
                .map(|(b, k)| b.clone() * AB::F::from_canonical_u32(k & 0xFFFFu32))
                .sum::<AB::Expr>(),
            local.k[0],
        );
        builder.assert_eq(
            one_hot
                .iter()
                .zip(SHA_COMPRESS_K.iter())
                .map(|(b, k)| b.clone() * AB::F::from_canonical_u32(k >> 16))
                .sum::<AB::Expr>(),
            local.k[1],
        );
    }

    fn eval_compression_ops<AB: DTAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25).
        // Calculate e rightrotate 6.
        let e_rr_6 = FixedRotateRightOperation::<AB::F>::eval(
            local.e_rr_6,
            builder,
            local.e,
            6,
            local.is_real,
        );
        // Calculate e rightrotate 11.
        let e_rr_11 = FixedRotateRightOperation::<AB::F>::eval(
            local.e_rr_11,
            builder,
            local.e,
            11,
            local.is_real,
        );
        // Calculate e rightrotate 25.
        let e_rr_25 = FixedRotateRightOperation::<AB::F>::eval(
            local.e_rr_25,
            builder,
            local.e,
            25,
            local.is_real,
        );

        // Calculate S1 := ((e rightrotate 6) xor (e rightrotate 11)) xor (e rightrotate 25).
        let s1 = XorNOperation::<AB::F, U3>::eval(
            &local.s1,
            builder,
            [
                CompactWord::<AB::F>::into_word(e_rr_6, local.e_rr_6_witness),
                CompactWord::<AB::F>::into_word(e_rr_11, local.e_rr_11_witness),
                CompactWord::<AB::F>::into_word(e_rr_25, local.e_rr_25_witness),
            ],
            local.is_real,
        );

        let e_word = CompactWord::<AB::F>::into_word(local.e, local.e_witness);

        // Calculate ch := (e and f) xor ((not e) and g).
        // Calculate e and f.
        let e_and_f = AndNOperation::<AB::F, U2>::eval(
            &local.e_and_f,
            builder,
            [e_word.clone(), CompactWord::<AB::F>::into_word(local.f, local.f_witness)],
            local.is_real,
        );
        // Calculate not e.
        let e_not = NotOperation::<AB::F>::eval_word::<AB::Expr>(e_word);
        // Calculate (not e) and g.
        let e_not_and_g = AndNOperation::<AB::F, U2>::eval(
            &local.e_not_and_g,
            builder,
            [e_not, CompactWord::<AB::F>::into_word(local.g, local.g_witness)],
            local.is_real,
        );

        // Calculate ch := (e and f) xor ((not e) and g).
        let ch = XorNOperation::<AB::F, U2>::eval(
            &local.ch,
            builder,
            [e_and_f, e_not_and_g],
            local.is_real,
        );

        // Calculate temp1 := h + S1 + ch + k[i] + w[i].
        AddNOperation::<AB::F, U5>::eval(
            local.temp1,
            builder,
            [
                CompactWord(local.h.0.map(<AB::Var as Into<AB::Expr>>::into)),
                s1.into(),
                ch.into(),
                CompactWord(local.k.0.map(<AB::Var as Into<AB::Expr>>::into)),
                local.w_access.access.value.into(),
            ],
            local.is_real,
        );

        // Calculate S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22).
        // Calculate a rightrotate 2.
        let a_rr_2 = FixedRotateRightOperation::<AB::F>::eval(
            local.a_rr_2,
            builder,
            local.a,
            2,
            local.is_real,
        );
        // Calculate a rightrotate 13.
        let a_rr_13 = FixedRotateRightOperation::<AB::F>::eval(
            local.a_rr_13,
            builder,
            local.a,
            13,
            local.is_real,
        );
        // Calculate a rightrotate 22.
        let a_rr_22 = FixedRotateRightOperation::<AB::F>::eval(
            local.a_rr_22,
            builder,
            local.a,
            22,
            local.is_real,
        );

        // Calculate S0 := ((a rightrotate 2) xor (a rightrotate 13)) xor (a rightrotate 22).
        let s0 = XorNOperation::<AB::F, U3>::eval(
            &local.s0,
            builder,
            [
                CompactWord::<AB::F>::into_word(a_rr_2, local.a_rr_2_witness),
                CompactWord::<AB::F>::into_word(a_rr_13, local.a_rr_13_witness),
                CompactWord::<AB::F>::into_word(a_rr_22, local.a_rr_22_witness),
            ],
            local.is_real,
        );

        let a_word = CompactWord::<AB::F>::into_word(local.a, local.a_witness);
        let b_word = CompactWord::<AB::F>::into_word(local.b, local.b_witness);
        let c_word = CompactWord::<AB::F>::into_word(local.c, local.c_witness);

        // Calculate maj := (a and b) xor (a and c) xor (b and c).
        // Calculate a and b.
        let a_and_b = AndNOperation::<AB::F, U2>::eval(
            &local.a_and_b,
            builder,
            [a_word.clone(), b_word.clone()],
            local.is_real,
        );
        // Calculate a and c.
        let a_and_c = AndNOperation::<AB::F, U2>::eval(
            &local.a_and_c,
            builder,
            [a_word, c_word.clone()],
            local.is_real,
        );
        // Calculate b and c.
        let b_and_c = AndNOperation::<AB::F, U2>::eval(
            &local.b_and_c,
            builder,
            [b_word, c_word],
            local.is_real,
        );

        // Calculate maj := ((a and b) xor (a and c)) xor (b and c).
        let maj = XorNOperation::<AB::F, U3>::eval(
            &local.maj,
            builder,
            [a_and_b, a_and_c, b_and_c],
            local.is_real,
        );

        // Calculate temp2 := s0 + maj.
        AddNOperation::<AB::F, U2>::eval(
            local.temp2,
            builder,
            [<Word<AB::Expr> as Into<CompactWord<AB::Expr>>>::into(s0), maj.into()],
            local.is_real,
        );

        // Calculate d + temp1 for the new value of e.
        AddNOperation::<AB::F, U2>::eval(
            local.d_add_temp1,
            builder,
            [local.d, local.temp1.value],
            local.is_real,
        );

        // Calculate temp1 + temp2 for the new value of a.
        AddNOperation::<AB::F, U2>::eval(
            local.temp1_add_temp2,
            builder,
            [local.temp1.value, local.temp2.value],
            local.is_real,
        );
    }
}
