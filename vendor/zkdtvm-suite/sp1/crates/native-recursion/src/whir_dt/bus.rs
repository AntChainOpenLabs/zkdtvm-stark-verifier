use dt_stark::air::FullAirBuilder;

use crate::{
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        WHIR_EVAL_CHAIN_SCHEMA, WHIR_FINAL_ROOT_CHAIN_SCHEMA, WHIR_GROUP_CLAIM_SCHEMA,
        WHIR_LEAF_CHAIN_SCHEMA, WHIR_LEAF_POW_SEED_SCHEMA, WHIR_OPENED_EVAL_SCHEMA,
        WHIR_QUERY_CHAIN_SCHEMA, WHIR_QUERY_INIT_SCHEMA, WHIR_QUERY_LEAF_SUM_SCHEMA,
        WHIR_ROUND_BCAST_SCHEMA, WHIR_ROUND_CHAIN_SCHEMA, WHIR_SAMPLE_BAND_SCHEMA,
        WHIR_TWIDDLE_POW_SCHEMA,
    },
};

macro_rules! global_bus {
    ($name:ident, $schema:ident, [$($arg:ident),+], $cap:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            bus: RecursionFullAirBus,
        }

        impl $name {
            pub const fn new() -> Self {
                Self { bus: RecursionFullAirBus::new($schema) }
            }

            pub const fn required_max_beta_power_floor(&self) -> usize {
                self.bus.required_max_beta_power_floor()
            }

            pub fn denominator<AB>(
                &self,
                builder: &AB,
                $($arg: AB::VarMaybeExt),+
            ) -> AB::VarExt
            where
                AB: FullAirBuilder,
            {
                let values = Vec::from([$($arg),+]);
                debug_assert_eq!(values.len(), $cap);
                self.bus.denominator(builder, values)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

macro_rules! per_proof_bus {
    ($name:ident, $schema:ident, [$($arg:ident),+], $cap:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            bus: RecursionFullAirBus,
        }

        impl $name {
            pub const fn new() -> Self {
                Self { bus: RecursionFullAirBus::new($schema) }
            }

            pub const fn required_max_beta_power_floor(&self) -> usize {
                self.bus.required_max_beta_power_floor()
            }

            pub fn denominator<AB>(
                &self,
                builder: &AB,
                proof_idx: AB::VarMaybeExt,
                $($arg: AB::VarMaybeExt),+
            ) -> AB::VarExt
            where
                AB: FullAirBuilder,
            {
                let values = Vec::from([$($arg),+]);
                debug_assert_eq!(values.len(), $cap);
                self.bus.denominator_for_proof(builder, proof_idx, values)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

global_bus!(WhirTwiddlePowBus, WHIR_TWIDDLE_POW_SCHEMA, [table_id, byte, value], 3);
global_bus!(
    WhirSampleBandBus,
    WHIR_SAMPLE_BAND_SCHEMA,
    [query_bits, shift, high_max, high_bits],
    4
);

per_proof_bus!(
    WhirRoundBcastBus,
    WHIR_ROUND_BCAST_SCHEMA,
    [
        round,
        r0,
        r1,
        r2,
        r3,
        r4,
        is_merge,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        eq0,
        eq1,
        eq2,
        eq3,
        eq4,
        emit_prep_seed,
        merge_log_height
    ],
    19
);
per_proof_bus!(WhirGroupClaimBus, WHIR_GROUP_CLAIM_SCHEMA, [log_height, c0, c1, c2, c3, c4], 6);
// The leaf sum is keyed per (height-group, truncated index); the first slot is the
// fold-bound leaf index, not the query ordinal.
per_proof_bus!(
    WhirQueryLeafSumBus,
    WHIR_QUERY_LEAF_SUM_SCHEMA,
    [idx, log_height, s0, s1, s2, s3, s4],
    7
);
per_proof_bus!(
    WhirQueryChainBus,
    WHIR_QUERY_CHAIN_SCHEMA,
    [query_idx, cursor, query_bits, r_rounds, idx, idx_bit, x, acc, ipw, f0, f1, f2, f3, f4],
    14
);
per_proof_bus!(
    WhirEvalChainBus,
    WHIR_EVAL_CHAIN_SCHEMA,
    [
        cursor,
        log_height,
        batch_id,
        batch_pos,
        value_idx,
        segment_element_count,
        alpha0,
        alpha1,
        alpha2,
        alpha3,
        alpha4,
        pow0,
        pow1,
        pow2,
        pow3,
        pow4,
        acc0,
        acc1,
        acc2,
        acc3,
        acc4,
        group_base0,
        group_base1,
        group_base2,
        group_base3,
        group_base4
    ],
    26
);
// The intra-group chain is keyed by the group instance (idx, log_height), not per query.
per_proof_bus!(
    WhirLeafChainBus,
    WHIR_LEAF_CHAIN_SCHEMA,
    [
        idx, cursor, log_height, batch_id, alpha0, alpha1, alpha2, alpha3, alpha4, pow0, pow1,
        pow2, pow3, pow4, acc0, acc1, acc2, acc3, acc4
    ],
    19
);
// WhirAlphaBcastBus (1029) is retired; alpha rides WhirLeafPowSeed (1044).
per_proof_bus!(
    WhirLeafPowSeedBus,
    WHIR_LEAF_POW_SEED_SCHEMA,
    [log_height, a0, a1, a2, a3, a4, p0, p1, p2, p3, p4],
    11
);
per_proof_bus!(
    WhirQueryInitBus,
    WHIR_QUERY_INIT_SCHEMA,
    [w_qbase, query_bits, r_rounds, cfr0, cfr1, cfr2, cfr3, cfr4],
    8
);
per_proof_bus!(
    WhirOpenedEvalBus,
    WHIR_OPENED_EVAL_SCHEMA,
    [batch_id, batch_pos, chip_idx, value_idx, v0, v1, v2, v3, v4],
    9
);
per_proof_bus!(
    WhirRoundChainBus,
    WHIR_ROUND_CHAIN_SCHEMA,
    [
        round,
        tidx,
        claim0,
        claim1,
        claim2,
        claim3,
        claim4,
        eq0,
        eq1,
        eq2,
        eq3,
        eq4,
        pending_is_merge,
        beta0,
        beta1,
        beta2,
        beta3,
        beta4,
        pending_eq0,
        pending_eq1,
        pending_eq2,
        pending_eq3,
        pending_eq4
    ],
    23
);
per_proof_bus!(
    WhirFinalRootChainBus,
    WHIR_FINAL_ROOT_CHAIN_SCHEMA,
    [step, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15],
    17
);
