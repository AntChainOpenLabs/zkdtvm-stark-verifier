use core::ops::Deref;
use std::any::{Any, TypeId};

use dt_core_executor::{ByteOpcode, ExecutionRecord, Program};
use dt_stark::{
    air::{
        AirInteraction, DTAirBuilder, FullAir, FullAirBuilder, InteractionScope, MachineAir,
        PairCol,
    },
    global_d11::PROJECTIVE_CHAIN_BLOCK_WIDTH,
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;

use crate::bytes::polyair::{
    bit_range_precompute_lc, u16_range_precompute_lc, u8_range_pair_precompute_lc,
};

use super::{
    columns::{GlobalCols, NUM_GLOBAL_COLS},
    constraints::{for_each_constraint_residual, header},
    interaction::{
        projective_chain_denominator, projective_chain_payload, LookupDirection,
        GLOBAL_INTERACTION_DESCRIPTORS,
    },
    writer::global_padded_rows,
};

pub(crate) const NUM_GLOBAL_LOOKUPS: usize = GLOBAL_INTERACTION_DESCRIPTORS.len();
pub(crate) const GLOBAL_LOOKUP_BATCH_SIZE: usize = 2;
pub(crate) const GLOBAL_PERMUTATION_LOOKUP_WIDTH: usize =
    NUM_GLOBAL_LOOKUPS.div_ceil(GLOBAL_LOOKUP_BATCH_SIZE);
pub(crate) const GLOBAL_PLC_WIDTH: usize = 32;
pub(crate) const GLOBAL_RESERVED_WIDTH: usize = 5;
pub(crate) const GLOBAL_EFFECTIVE_BASE_CELLS: usize =
    GLOBAL_RESERVED_WIDTH + 5 * GLOBAL_PLC_WIDTH + 5 * GLOBAL_PERMUTATION_LOOKUP_WIDTH;
pub(crate) const GLOBAL_MAX_BETA_POWER: usize = 11;

/// `Projective228QIntervalV4` Global main AIR.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalChip;

impl<F: Field> BaseAir<F> for GlobalChip {
    fn width(&self) -> usize {
        NUM_GLOBAL_COLS
    }
}

impl<F: Field> MachineAir<F> for GlobalChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Global".to_string()
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> CompressedMatrix<F> {
        assert_eq!(
            TypeId::of::<F>(),
            TypeId::of::<KoalaBear>(),
            "Global trace generation is supported only for the canonical KoalaBear base field"
        );
        let trace = input.take_global_trace_artifact().expect(
            "canonical Global trace artifact missing; generate_dependencies must run exactly once",
        );
        let erased: Box<dyn Any> = Box::new(trace);
        *erased
            .downcast::<CompressedMatrix<F>>()
            .expect("field identity was checked before Global trace downcast")
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        if !<Self as MachineAir<F>>::included(self, input) {
            return;
        }
        if input.has_global_trace_artifact() {
            return;
        }
        let prepared =
            super::writer::prepare_global_trace(input).expect("Global trace preparation failed");
        let retained = prepared.consume_byte_delta(output);
        output.install_global_trace_artifact(retained.trace, retained.reducer_trace);
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        if TypeId::of::<F>() != TypeId::of::<KoalaBear>() {
            return None;
        }
        let raw_rows = crate::global::sources::global_endpoint_count(input);
        Some(global_padded_rows(raw_rows).expect("Global physical height exceeds h22"))
    }

    fn included(&self, input: &Self::Record) -> bool {
        TypeId::of::<F>() == TypeId::of::<KoalaBear>() &&
            crate::global::sources::global_endpoint_count(input) > 0
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Global
    }

    fn local_only(&self) -> bool {
        true
    }

    fn global_boundary_owner(&self) -> Option<dt_stark::global_d11::StableChipId> {
        None
    }
}

impl<AB> Air<AB> for GlobalChip
where
    AB: DTAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_binding = main.row_slice(0);
        let local_vars = local_binding.deref();
        let local_exprs: [AB::Expr; NUM_GLOBAL_COLS] =
            core::array::from_fn(|index| local_vars[index].into());
        // SAFETY: `local_exprs` has the compile-time checked repr(C) width and
        // exact flat field order of `GlobalCols`.
        let local = unsafe { &*local_exprs.as_ptr().cast::<GlobalCols<AB::Expr>>() };

        for_each_constraint_residual(local, |residual| builder.assert_zero(residual));

        let h = header(local);
        let is_real = local.is_real.clone();
        let zero = AB::Expr::zero();

        // #1: unchanged production ten-value endpoint bus.
        builder.receive(
            AirInteraction::new(
                h.message
                    .iter()
                    .cloned()
                    .chain([h.is_send.clone(), local.is_receive.clone(), h.kind.clone()])
                    .collect(),
                is_real.clone(),
                InteractionKind::Global,
            ),
            InteractionScope::Local,
        );

        // #2--#8: the same integer-domain interactions as the FullAir path.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            local.m0_lo16.clone(),
            zero.clone(),
            zero.clone(),
            is_real.clone(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            zero.clone(),
            local.m0_hi8.clone(),
            local.message_rest[4].clone(),
            is_real.clone(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            zero.clone(),
            local.message_rest[5].clone(),
            h.kind.clone(),
            is_real.clone(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::BitRange as u8),
            h.tweak.clone(),
            AB::Expr::from_canonical_u8(9),
            zero.clone(),
            is_real.clone(),
        );
        for value in [
            local.w_lo16.clone(),
            local.w_hi.clone(),
            AB::Expr::from_canonical_u16(16_255) - local.w_hi.clone(),
        ] {
            builder.send_byte(
                AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
                value,
                zero.clone(),
                zero.clone(),
                is_real.clone(),
            );
        }

        // #9--#10: direct metadata explicitly retains 34 affine base
        // expressions but beta-compresses seven Ext5 values.  The sole spare
        // coefficient is structural zero in the framework codec.
        let chain_kind = GLOBAL_INTERACTION_DESCRIPTORS[8].kind;
        let receive_payload = projective_chain_payload(local.index.clone(), &local.input).to_vec();
        builder.receive(
            AirInteraction::new_extension_blocks(
                receive_payload,
                PROJECTIVE_CHAIN_BLOCK_WIDTH,
                is_real.clone(),
                chain_kind,
            ),
            InteractionScope::Local,
        );
        let send_payload =
            projective_chain_payload(local.index.clone() + AB::Expr::one(), &local.cumulative)
                .to_vec();
        builder.send(
            AirInteraction::new_extension_blocks(
                send_payload,
                PROJECTIVE_CHAIN_BLOCK_WIDTH,
                is_real,
                chain_kind,
            ),
            InteractionScope::Local,
        );
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for GlobalChip
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_GLOBAL_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        GLOBAL_MAX_BETA_POWER
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(super::columns::GLOBAL_COL_MAP.is_real),
            PairCol::Main(super::columns::GLOBAL_COL_MAP.is_receive),
            PairCol::Main(super::columns::GLOBAL_COL_MAP.input.z[10]),
            PairCol::Main(super::columns::GLOBAL_COL_MAP.input.x[10]),
            PairCol::Main(super::columns::GLOBAL_COL_MAP.products.u4[10]),
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        // SAFETY: the typed repr(C) layout is compile-time checked to contain
        // exactly 228 consecutive elements and `main` has that exact width.
        let local: &GlobalCols<AB::VarMaybeExt> =
            unsafe { &*(main.as_ptr().cast::<GlobalCols<AB::VarMaybeExt>>()) };
        let h = header(local);
        let is_real = local.is_real.clone();

        // #1: receive the unchanged ten-value production Global payload.
        let global_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));
        builder.retain_precomputed(builder.lookup_denominator(
            global_kind,
            h.message.iter().cloned().chain([
                h.is_send.clone(),
                local.is_receive.clone(),
                h.kind.clone(),
            ]),
        ));

        // #2--#8: integer-domain ownership.
        u16_range_precompute_lc(builder, local.m0_lo16.clone());
        u8_range_pair_precompute_lc(builder, local.m0_hi8.clone(), local.message_rest[4].clone());
        u8_range_pair_precompute_lc(builder, local.message_rest[5].clone(), h.kind.clone());
        bit_range_precompute_lc(
            builder,
            h.tweak.clone(),
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(9)),
        );
        u16_range_precompute_lc(builder, local.w_lo16.clone());
        u16_range_precompute_lc(builder, local.w_hi.clone());
        u16_range_precompute_lc(
            builder,
            AB::VarMaybeExt::from(AB::F::from_canonical_u16(16_255)) - local.w_hi.clone(),
        );

        // #9--#10: indexed projective chain, seven Ext5 blocks each.
        builder.retain_precomputed(projective_chain_denominator(
            builder,
            local.index.clone(),
            &local.input,
        ));
        builder.retain_precomputed(projective_chain_denominator(
            builder,
            local.index.clone() + AB::one_maybe(),
            &local.cumulative,
        ));

        let beta = builder.beta_powers().to_vec();
        let project = |values: &[AB::VarMaybeExt]| {
            let mut out = beta[0].clone() * values[0].clone();
            for (power, value) in values.iter().enumerate().skip(1) {
                out = out + beta[power].clone() * value.clone();
            }
            out
        };
        let endpoint_x = super::constraints::packed_x(local, &h);
        builder.retain_precomputed(project(&endpoint_x));
        builder.retain_precomputed(project(&h.signed_y));
        for values in [
            local.input.x.as_slice(),
            local.input.y.as_slice(),
            local.input.z.as_slice(),
            local.products.u0.as_slice(),
            local.products.u1.as_slice(),
            local.products.u3.as_slice(),
            local.products.u4.as_slice(),
            local.products.u5.as_slice(),
            local.cumulative.x.as_slice(),
            local.cumulative.y.as_slice(),
            local.cumulative.z.as_slice(),
            local.quotient.map.as_slice(),
            local.quotient.u0.as_slice(),
            local.quotient.u1.as_slice(),
            local.quotient.u3.as_slice(),
            local.quotient.u4.as_slice(),
            local.quotient.u5.as_slice(),
            local.quotient.output_x.as_slice(),
            local.quotient.output_y.as_slice(),
            local.quotient.output_z.as_slice(),
        ] {
            builder.retain_precomputed(project(values));
        }

        let _ = is_real;
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let r = reserved.row_slice(0);
        let precomputed = builder.precomputed();
        let p = precomputed.row_slice(0);
        let is_real = r[0].clone();
        let is_receive = r[1].clone();
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));
        builder.assert_zero(is_receive.clone() * (one.clone() - is_receive.clone()));
        builder.assert_zero(is_receive.clone() * (one.clone() - is_real.clone()));

        let x = p[10].clone();
        let y = p[11].clone();
        let input_x = p[12].clone();
        let input_y = p[13].clone();
        let input_z = p[14].clone();
        let u0 = p[15].clone();
        let u1 = p[16].clone();
        let u3 = p[17].clone();
        let u4 = p[18].clone();
        let u5 = p[19].clone();
        let cumulative_x = p[20].clone();
        let cumulative_y = p[21].clone();
        let cumulative_z = p[22].clone();
        let q_map = p[23].clone();
        let q_u0 = p[24].clone();
        let q_u1 = p[25].clone();
        let q_u3 = p[26].clone();
        let q_u4 = p[27].clone();
        let q_u5 = p[28].clone();
        let q_x = p[29].clone();
        let q_y = p[30].clone();
        let q_z = p[31].clone();
        let beta = builder.beta_powers()[1].clone();
        let f_beta = builder.beta_powers()[11].clone() - builder.beta_powers()[3].clone() -
            AB::from_ef(AB::EF::from_canonical_u8(2));
        let three = AB::VarMaybeExt::from(AB::F::from_canonical_u8(3));
        let thirty_six = AB::from_ef(AB::EF::from_canonical_u8(36));

        builder.assert_zero_ext(
            y.clone() * y.clone() - x.clone() * x.clone() * x.clone() +
                x.clone() * three.clone() -
                beta.clone() -
                thirty_six -
                q_map * f_beta.clone(),
        );
        builder.assert_zero_ext(input_x.clone() * x.clone() - u0.clone() - q_u0 * f_beta.clone());
        builder.assert_zero_ext(input_y.clone() * y.clone() - u1.clone() - q_u1 * f_beta.clone());
        builder.assert_zero_ext(
            (input_x.clone() + input_y.clone()) * (x.clone() + y.clone()) -
                u3.clone() -
                q_u3 * f_beta.clone(),
        );
        builder.assert_zero_ext(input_z.clone() * x - u4.clone() - q_u4 * f_beta.clone());
        builder.assert_zero_ext(input_z.clone() * y - u5.clone() - q_u5 * f_beta.clone());

        let sxy = u3 - u0.clone() - u1.clone();
        let sxz = input_x.clone() + u4.clone();
        let syz = input_y.clone() + u5;
        let b_input_z = (beta.clone() + AB::from_ef(AB::EF::from_canonical_u8(36))) *
            input_z.clone() -
            f_beta.clone() * r[2].clone();
        let b_sxz = (beta + AB::from_ef(AB::EF::from_canonical_u8(36))) * sxz.clone() -
            f_beta.clone() * (r[3].clone() + r[4].clone());
        let delta = (sxz - b_input_z) * three.clone();
        let l0 = u1.clone() + delta.clone();
        let l3 = u1 - delta;
        let l2 = (u0.clone() - input_z.clone()) * three.clone();
        let l1 = (b_sxz - u0 - input_z.clone() * three.clone()) * three;
        let raw_x = sxy.clone() * l0.clone() - syz.clone() * l1.clone();
        let raw_y = l2.clone() * l1 + l3.clone() * l0;
        let raw_z = syz * l3 + sxy * l2;
        let not_real = one - is_real.clone();
        builder.assert_zero_ext(
            raw_x * is_real.clone() + input_x * not_real.clone() - cumulative_x -
                q_x * f_beta.clone(),
        );
        builder.assert_zero_ext(
            raw_y * is_real.clone() + input_y * not_real.clone() - cumulative_y -
                q_y * f_beta.clone(),
        );
        builder.assert_zero_ext(
            raw_z * is_real + input_z * not_real - cumulative_z - q_z * f_beta,
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local_slice = local_binding.deref();
        let is_real = local_slice[0].clone();

        for descriptor in GLOBAL_INTERACTION_DESCRIPTORS {
            debug_assert_eq!(descriptor.slots_per_row, 1);
            match descriptor.direction {
                LookupDirection::Send => builder.send(is_real.clone()),
                LookupDirection::Receive => builder.recv(is_real.clone()),
            }
        }
    }
}

const _: () = {
    assert!(NUM_GLOBAL_LOOKUPS == 10);
    assert!(GLOBAL_PERMUTATION_LOOKUP_WIDTH == 5);
    assert!(GLOBAL_PLC_WIDTH == 32);
    assert!(GLOBAL_RESERVED_WIDTH == 5);
    assert!(GLOBAL_EFFECTIVE_BASE_CELLS == 190);
};
