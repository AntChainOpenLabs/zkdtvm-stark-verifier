use dt_stark::air::FullAirBuilder;
use p3_field::AbstractField;

use crate::{
    interaction_full_air_dt::RecursionFullAirBus,
    interaction_registry_dt::{
        NATIVE_CHIP_METADATA_SCHEMA, PROOF_SHAPE_BATCH_DIM_SCHEMA, PROOF_SHAPE_CHAIN_SCHEMA,
        PROOF_SHAPE_CHIP_META_SCHEMA, PROOF_SHAPE_HEIGHT_GROUP_SCHEMA,
        PROOF_SHAPE_HEIGHT_MEMBER_SCHEMA, PROOF_SHAPE_HEIGHT_RANK_SCHEMA,
        PROOF_SHAPE_GLOBAL_PACKED_SCHEMA, PROOF_SHAPE_SUMMARY_SCHEMA, PROOF_SHAPE_VALUES_SCHEMA,
    },
};

pub const PROOF_SHAPE_COMMIT_VK: usize = 0;
pub const PROOF_SHAPE_COMMIT_MAIN: usize = 1;
pub const PROOF_SHAPE_COMMIT_PERMUTATION: usize = 2;

pub const PROOF_SHAPE_BATCH_PREPROCESSED: usize = 0;
pub const PROOF_SHAPE_BATCH_MAIN: usize = 1;
pub const PROOF_SHAPE_BATCH_PERMUTATION: usize = 2;

pub const PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES: usize = 0;
pub const PROOF_SHAPE_NAMESPACE_VK_META: usize = 1;

pub const PROOF_SHAPE_VK_META_COMMIT_BASE: usize = 0;
pub const PROOF_SHAPE_VK_META_COMMIT_ELTS: usize = 8;
pub const PROOF_SHAPE_VK_META_PC_START: usize = 8;
pub const PROOF_SHAPE_VK_META_BOUNDARY_BASE: usize = 9;
pub const PROOF_SHAPE_VK_META_BOUNDARY_KIND: usize = PROOF_SHAPE_VK_META_BOUNDARY_BASE;
pub const PROOF_SHAPE_VK_META_BOUNDARY_X_BASE: usize = PROOF_SHAPE_VK_META_BOUNDARY_BASE + 1;
pub const PROOF_SHAPE_VK_META_BOUNDARY_Y_BASE: usize = PROOF_SHAPE_VK_META_BOUNDARY_X_BASE + 11;
pub const PROOF_SHAPE_VK_META_BOUNDARY_ELTS: usize = 23;
pub const PROOF_SHAPE_CORE_VK_META_VALUE_COUNT: usize = 32;
pub const PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT: usize = 8;
pub const PROOF_SHAPE_VK_META_VALUE_COUNT: usize = PROOF_SHAPE_CORE_VK_META_VALUE_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeChipMetadataBus {
    bus: RecursionFullAirBus,
}

impl NativeChipMetadataBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(NATIVE_CHIP_METADATA_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        role_id: AB::VarMaybeExt,
        chip_id: AB::VarMaybeExt,
        stable_air_id_lo: AB::VarMaybeExt,
        stable_air_id_hi: AB::VarMaybeExt,
        prep_width: AB::VarMaybeExt,
        main_width: AB::VarMaybeExt,
        perm_width: AB::VarMaybeExt,
        constraint_count: AB::VarMaybeExt,
        gate_count: AB::VarMaybeExt,
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator(
            builder,
            [
                role_id,
                chip_id,
                stable_air_id_lo,
                stable_air_id_hi,
                prep_width,
                main_width,
                perm_width,
                constraint_count,
                gate_count,
            ],
        )
    }
}

impl Default for NativeChipMetadataBus {
    fn default() -> Self {
        Self::new()
    }
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

per_proof_bus!(
    ProofShapeChipMetaBus,
    PROOF_SHAPE_CHIP_META_SCHEMA,
    [chip_idx, static_chip_id, log_height, gate_count, batch_count],
    5
);
per_proof_bus!(
    ProofShapeBatchDimBus,
    PROOF_SHAPE_BATCH_DIM_SCHEMA,
    // Provisional seam for downstream PCS/constraint consumers: static_chip_id and width are
    // carried here for now; the payload is expected to be revisited when whir_dt lands.
    [batch_id, batch_pos, chip_idx, static_chip_id, width, log_height],
    6
);
per_proof_bus!(ProofShapeValuesBus, PROOF_SHAPE_VALUES_SCHEMA, [namespace, idx, value], 3);

/// Authenticates one eight-field Binder public-value row as two Ext5 blocks.
///
/// Keeping the original Binder row boundary avoids extra transcript rows and carry
/// columns.  The high block contains three live limbs and two canonical zero limbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofShapeGlobalPackedBus {
    bus: RecursionFullAirBus,
}

impl ProofShapeGlobalPackedBus {
    pub const fn new() -> Self {
        Self { bus: RecursionFullAirBus::new(PROOF_SHAPE_GLOBAL_PACKED_SCHEMA) }
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    pub fn denominator<AB>(
        &self,
        builder: &AB,
        proof_idx: AB::VarMaybeExt,
        shape_idx_base: AB::VarMaybeExt,
        values: &[AB::VarMaybeExt; 8],
    ) -> AB::VarExt
    where
        AB: FullAirBuilder,
    {
        self.bus.denominator_ext_blocks_for_proof(
            builder,
            proof_idx,
            [
                AB::from_ef(AB::EF::zero()) + shape_idx_base,
                AB::pack_ext_limbs(&values[..5]),
                AB::pack_ext_limbs(&values[5..]),
            ],
        )
    }
}

impl Default for ProofShapeGlobalPackedBus {
    fn default() -> Self {
        Self::new()
    }
}
per_proof_bus!(
    ProofShapeHeightGroupBus,
    PROOF_SHAPE_HEIGHT_GROUP_SCHEMA,
    [group_idx, log_height],
    2
);
per_proof_bus!(
    ProofShapeChainBus,
    PROOF_SHAPE_CHAIN_SCHEMA,
    [
        chip_idx,
        log_height,
        static_chip_id,
        tidx_acc,
        prep_matrix_idx,
        first_log_height,
        shape_chip_count
    ],
    7
);
per_proof_bus!(ProofShapeHeightMemberBus, PROOF_SHAPE_HEIGHT_MEMBER_SCHEMA, [log_height], 1);
per_proof_bus!(ProofShapeHeightRankBus, PROOF_SHAPE_HEIGHT_RANK_SCHEMA, [height_cursor, rank], 2);
per_proof_bus!(
    ProofShapeSummaryBus,
    PROOF_SHAPE_SUMMARY_SCHEMA,
    [num_rounds, c_chips, num_public_values, static_chip_id_base],
    4
);
