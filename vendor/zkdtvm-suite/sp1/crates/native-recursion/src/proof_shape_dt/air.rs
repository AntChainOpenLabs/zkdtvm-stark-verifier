use core::{
    borrow::Borrow,
    ops::{Add, Deref, Mul, Sub},
};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    batch_constraint_dt::columns::{
        batch_seed_prefix_limbs, BATCH_VK_TAG_V1, BATCH_VK_TAG_VERSION_LIMBS, BATCH_VK_VERSION_V1,
    },
    config::{D_EF, F},
    constraint_replay_dt::ConstraintFoldPlanChainBus,
    primitives_dt::bus::{RangeCheckerBus, RangeCheckerBusMessage},
    proof_shape_dt::{
        bus::{
            NativeChipMetadataBus, ProofShapeBatchDimBus, ProofShapeChainBus,
            ProofShapeChipMetaBus, ProofShapeGlobalPackedBus, ProofShapeHeightGroupBus,
            ProofShapeHeightMemberBus, ProofShapeHeightRankBus, ProofShapeSummaryBus,
            ProofShapeValuesBus,
            PROOF_SHAPE_BATCH_MAIN, PROOF_SHAPE_BATCH_PERMUTATION, PROOF_SHAPE_BATCH_PREPROCESSED,
            PROOF_SHAPE_COMMIT_MAIN, PROOF_SHAPE_COMMIT_PERMUTATION, PROOF_SHAPE_COMMIT_VK,
            PROOF_SHAPE_VK_META_COMMIT_BASE,
        },
        columns::{
            NativeChipMetadataPreprocessedCols, ProofHeightSetCols, ProofShapeBinderCols,
            NUM_NATIVE_CHIP_METADATA_COLS, NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS,
            NUM_PROOF_HEIGHT_SET_COLS, NUM_PROOF_SHAPE_BINDER_COLS,
        },
        trace::{
            NativeChipMetadataTraceGenerator, ProofHeightSetTraceGenerator,
            ProofShapeBinderTraceGenerator,
        },
    },
    system_dt::{RecursionNativeChipMetadataRequest, RecursionNativeProgram, RecursionRecord},
    transcript_dt::{merkle_path::MerkleCommitmentRootBus, sponge::TranscriptEventBus},
    whir_dt::WhirRoleConfig,
};

#[derive(Debug, Clone)]
pub struct NativeChipMetadataAir {
    pub bus: NativeChipMetadataBus,
    pub metadata: Vec<RecursionNativeChipMetadataRequest>,
}

impl NativeChipMetadataAir {
    pub fn new(metadata: Vec<RecursionNativeChipMetadataRequest>) -> Self {
        Self { bus: NativeChipMetadataBus::new(), metadata }
    }
}

impl BaseAir<F> for NativeChipMetadataAir {
    fn width(&self) -> usize {
        NUM_NATIVE_CHIP_METADATA_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for NativeChipMetadataAir {
    fn width(&self) -> usize {
        NUM_NATIVE_CHIP_METADATA_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_NATIVE_CHIP_METADATA_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominator = {
            let prep = builder.preprocessed();
            let local: &NativeChipMetadataPreprocessedCols<AB::VarMaybeExt> = prep.borrow();
            self.bus.denominator(
                builder,
                local.role_id.clone(),
                local.chip_id.clone(),
                local.stable_air_id_lo.clone(),
                local.stable_air_id_hi.clone(),
                local.prep_width.clone(),
                local.main_width.clone(),
                local.perm_width.clone(),
                local.constraint_count.clone(),
                local.gate_count.clone(),
            )
        };
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, _builder: &mut AB) {}

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let mult = local[NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS].clone();
        builder.send(mult);
    }
}

impl MachineAir<F> for NativeChipMetadataAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeChipMetadata".to_string()
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(NativeChipMetadataTraceGenerator::trace_height(&self.metadata))
    }

    fn preprocessed_width(&self) -> usize {
        NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(NativeChipMetadataTraceGenerator::trace_height(&program.native_chip_metadata))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Some(NativeChipMetadataTraceGenerator::generate_preprocessed_trace(
            &program.native_chip_metadata,
        ))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        NativeChipMetadataTraceGenerator::generate_trace_compressed(input, &self.metadata)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProofShapeBinderAir {
    pub num_public_values: usize,
    pub seed_prefix_limbs: usize,
    pub transcript_event_bus: TranscriptEventBus,
    pub metadata_bus: NativeChipMetadataBus,
    pub commitment_root_bus: MerkleCommitmentRootBus,
    pub chip_meta_bus: ProofShapeChipMetaBus,
    pub batch_dim_bus: ProofShapeBatchDimBus,
    pub values_bus: ProofShapeValuesBus,
    pub global_packed_bus: ProofShapeGlobalPackedBus,
    pub chain_bus: ProofShapeChainBus,
    pub summary_bus: ProofShapeSummaryBus,
    pub height_member_bus: ProofShapeHeightMemberBus,
    pub fold_plan_chain_bus: ConstraintFoldPlanChainBus,
    pub range_bus: RangeCheckerBus,
    pub role_config: WhirRoleConfig,
    pub contains_global_bus: bool,
}

impl ProofShapeBinderAir {
    pub const fn new(
        num_public_values: usize,
        role_config: WhirRoleConfig,
        contains_global_bus: bool,
    ) -> Self {
        Self {
            num_public_values,
            seed_prefix_limbs: batch_seed_prefix_limbs(contains_global_bus),
            transcript_event_bus: TranscriptEventBus::new(),
            metadata_bus: NativeChipMetadataBus::new(),
            commitment_root_bus: MerkleCommitmentRootBus::new(),
            chip_meta_bus: ProofShapeChipMetaBus::new(),
            batch_dim_bus: ProofShapeBatchDimBus::new(),
            values_bus: ProofShapeValuesBus::new(),
            global_packed_bus: ProofShapeGlobalPackedBus::new(),
            chain_bus: ProofShapeChainBus::new(),
            summary_bus: ProofShapeSummaryBus::new(),
            height_member_bus: ProofShapeHeightMemberBus::new(),
            fold_plan_chain_bus: ConstraintFoldPlanChainBus::new(),
            range_bus: RangeCheckerBus::new(),
            role_config,
            contains_global_bus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LookupDirection {
    Send,
    Recv,
}

pub(crate) fn binder_lookup_ops<T>(
    local: &ProofShapeBinderCols<T>,
    whir_num_queries: T,
    chip_meta_mult: T,
    contains_global_bus: bool,
) -> Vec<(LookupDirection, T)>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    use LookupDirection::{Recv, Send};

    let mut ops = Vec::with_capacity(34 + usize::from(contains_global_bus));
    for mask in &local.event_recv_mask {
        ops.push((Recv, mask.clone()));
    }
    ops.push((Recv, local.is_vk_commit.clone()));
    ops.push((Recv, local.is_vk_commit.clone()));
    ops.push((Recv, local.is_chip.clone()));
    ops.push((Send, local.whir_role_config_recv_mult.clone() * whir_num_queries));
    ops.push((Send, chip_meta_mult));
    ops.push((Send, local.batch_dim_prep_send_mult.clone()));
    ops.push((Send, local.is_chip.clone() + local.is_chip.clone()));
    ops.push((Send, local.batch_dim_perm_send_mult.clone()));
    for mult in &local.shape_value_send_mults {
        ops.push((Send, mult.clone()));
    }
    ops.push((Send, local.has_prep.clone() + local.is_chip.clone() + local.is_chip.clone()));
    ops.push((
        Recv,
        local.is_active_shape_header.clone() + local.is_chip.clone() + local.is_e5.clone(),
    ));
    ops.push((
        Send,
        local.is_e1.clone() + local.is_active_shape_header.clone() + local.is_chip.clone(),
    ));
    ops.push((Send, local.summary_send_mult.clone()));
    ops.push((Recv, local.is_chip.clone()));
    // 26..30: range and segment-band recvs (id band on chip rows; prev band on chip + E5).
    // 31: the E5 FoldPlan/c_chips authority source.
    ops.push((Recv, local.is_chip.clone()));
    ops.push((Recv, local.is_chip.clone()));
    ops.push((Recv, local.is_chip.clone() + local.is_e5.clone()));
    ops.push((Recv, local.is_chip.clone() + local.is_e5.clone()));
    ops.push((Send, local.fold_plan_source_mult.clone()));
    if contains_global_bus {
        // `commit_id` is overlaid with the packed-row bit on public-value rows.
        // Subtract the two non-zero commitment-row ids to recover a linear
        // multiplicity, as required by the lookup argument.
        ops.push((
            Send,
            local.commit_id.clone()
                - local.is_e1.clone()
                - local.is_e5.clone()
                - local.is_e5.clone(),
        ));
    }
    debug_assert_eq!(ops.len(), 34 + usize::from(contains_global_bus));
    ops
}

impl BaseAir<F> for ProofShapeBinderAir {
    fn width(&self) -> usize {
        NUM_PROOF_SHAPE_BINDER_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ProofShapeBinderAir {
    fn width(&self) -> usize {
        NUM_PROOF_SHAPE_BINDER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.transcript_event_bus.required_max_beta_power_floor(),
            self.metadata_bus.required_max_beta_power_floor(),
            self.commitment_root_bus.required_max_beta_power_floor(),
            self.chip_meta_bus.required_max_beta_power_floor(),
            self.batch_dim_bus.required_max_beta_power_floor(),
            self.values_bus.required_max_beta_power_floor(),
            self.global_packed_bus.required_max_beta_power_floor(),
            self.chain_bus.required_max_beta_power_floor(),
            self.summary_bus.required_max_beta_power_floor(),
            self.height_member_bus.required_max_beta_power_floor(),
            self.fold_plan_chain_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_PROOF_SHAPE_BINDER_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &ProofShapeBinderCols<AB::VarMaybeExt> = main.borrow();
            binder_denominators(self, builder, local)
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ProofShapeBinderCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        assert_bool(builder, local.is_valid.clone());
        let selectors = [
            local.is_vk_commit.clone(),
            local.is_vk_meta.clone(),
            local.is_public_values.clone(),
            local.is_e1.clone(),
            local.is_active_shape_header.clone(),
            local.is_chip.clone(),
            local.is_e5.clone(),
        ];
        let mut selector_sum = AB::zero_maybe();
        for selector in selectors {
            assert_bool(builder, selector.clone());
            selector_sum = selector_sum + selector;
        }
        builder.assert_eq(selector_sum, local.is_valid.clone());

        for mask in local.event_recv_mask.iter().chain(local.shape_value_send_mask.iter()) {
            assert_bool(builder, mask.clone());
        }
        // Shape-value sends carry downstream demand counts. Public values can be consumed by
        // multiple terminal equations, so these multiplicities are not boolean.
        assert_bool(builder, local.whir_role_config_recv_mult.clone());
        assert_bool(builder, local.has_prep.clone());
        assert_bool(builder, local.is_group_start.clone());
        assert_bool(builder, local.batch_dim_prep_send_mult.clone());
        assert_bool(builder, local.batch_dim_perm_send_mult.clone());
        assert_bool(builder, local.is_first_chip.clone());
        builder.assert_zero(
            local.summary_send_mult.clone()
                * (local.summary_send_mult.clone() - const_maybe::<AB>(4))
                * (local.summary_send_mult.clone() - const_maybe::<AB>(3)),
        );

        let event_selector = local.is_vk_commit.clone()
            + local.is_vk_meta.clone()
            + local.is_public_values.clone()
            + local.is_e1.clone()
            + local.is_active_shape_header.clone()
            + local.is_chip.clone()
            + local.is_e5.clone();
        let full_event_selector =
            local.is_vk_commit.clone() + local.is_e1.clone() + local.is_e5.clone();
        let shape_selector =
            local.is_vk_commit.clone() + local.is_vk_meta.clone() + local.is_public_values.clone();
        for i in 0..8 {
            builder.assert_zero(
                (AB::one_maybe() - event_selector.clone()) * local.event_recv_mask[i].clone(),
            );
            builder.assert_zero(
                full_event_selector.clone() * (local.event_recv_mask[i].clone() - AB::one_maybe()),
            );
            builder.assert_zero(
                (AB::one_maybe() - shape_selector.clone()) * local.shape_value_send_mask[i].clone(),
            );
            builder.assert_zero(
                local.shape_value_send_mults[i].clone()
                    * (AB::one_maybe() - local.shape_value_send_mask[i].clone()),
            );
            builder.assert_zero(
                local.is_public_values.clone()
                    * (local.shape_value_send_mask[i].clone() - local.event_recv_mask[i].clone()),
            );
            builder.assert_zero(
                local.is_vk_meta.clone()
                    * (local.shape_value_send_mask[i].clone() - local.event_recv_mask[i].clone()),
            );
            builder.assert_zero(
                local.is_vk_meta.clone()
                    * (AB::one_maybe() - local.shape_value_send_mask[i].clone())
                    * local.event_values[i].clone(),
            );
            builder.assert_zero(
                local.is_vk_commit.clone()
                    * (local.shape_value_send_mask[i].clone() - AB::one_maybe()),
            );
        }
        for i in 0..7 {
            builder.assert_zero(
                local.is_public_values.clone()
                    * local.event_recv_mask[i + 1].clone()
                    * (AB::one_maybe() - local.event_recv_mask[i].clone()),
            );
        }

        builder.assert_zero(
            local.is_vk_commit.clone()
                * (local.tidx_base.clone() - const_maybe::<AB>(BATCH_VK_TAG_VERSION_LIMBS)),
        );
        builder.assert_zero(
            local.is_vk_commit.clone()
                * (local.shape_idx_base.clone()
                    - const_maybe::<AB>(PROOF_SHAPE_VK_META_COMMIT_BASE)),
        );
        builder.assert_zero(
            local.is_vk_meta.clone()
                * (local.shape_idx_base.clone() - local.tidx_base.clone()
                    + const_maybe::<AB>(BATCH_VK_TAG_VERSION_LIMBS)),
        );
        builder.assert_zero(
            local.is_public_values.clone()
                * (local.shape_idx_base.clone() - local.tidx_base.clone()
                    + const_maybe::<AB>(self.seed_prefix_limbs)),
        );
        let commit_selector =
            local.is_vk_commit.clone() + local.is_e1.clone() + local.is_e5.clone();
        builder.assert_zero(
            local.is_vk_commit.clone()
                * (local.commit_id.clone() - const_maybe::<AB>(PROOF_SHAPE_COMMIT_VK)),
        );
        builder.assert_zero(
            local.is_e1.clone()
                * (local.commit_id.clone() - const_maybe::<AB>(PROOF_SHAPE_COMMIT_MAIN)),
        );
        builder.assert_zero(
            local.is_e5.clone()
                * (local.commit_id.clone() - const_maybe::<AB>(PROOF_SHAPE_COMMIT_PERMUTATION)),
        );
        builder.assert_zero(
            (AB::one_maybe()
                - commit_selector.clone()
                - local.is_public_values.clone()) *
                local.commit_id.clone(),
        );
        builder.assert_zero(
            local.is_public_values.clone()
                * local.commit_id.clone()
                * (local.commit_id.clone() - AB::one_maybe()),
        );
        if !self.contains_global_bus {
            builder.assert_zero(local.is_public_values.clone() * local.commit_id.clone());
        }
        builder.assert_zero(
            local.whir_role_config_recv_mult.clone() * (AB::one_maybe() - commit_selector.clone()),
        );
        builder.assert_zero(
            local.batch_dim_prep_send_mult.clone() * (AB::one_maybe() - local.has_prep.clone()),
        );
        builder.assert_zero(
            local.batch_dim_perm_send_mult.clone() * (AB::one_maybe() - local.is_chip.clone()),
        );
        let batch_count =
            AB::mul_base(local.perm_width.clone(), AB::F::from_canonical_usize(D_EF).inverse());
        builder.assert_zero(
            local.chip_meta_send_mult.clone()
                - local.is_chip.clone()
                    * (local.gate_count.clone() + batch_count + AB::one_maybe()),
        );
        builder
            .assert_zero(local.summary_send_mult.clone() * (AB::one_maybe() - local.is_e5.clone()));
        builder.assert_zero(
            local.fold_plan_source_mult.clone() * (AB::one_maybe() - local.is_e5.clone()),
        );
        builder.assert_zero(
            local.is_e5.clone()
                * (local.fold_plan_source_mult.clone()
                    - local.prev_chip_idx.clone()
                    - const_maybe::<AB>(2)),
        );

        let chip = local.is_chip.clone();
        for i in 0..8 {
            if i < 5 {
                builder.assert_zero(
                    chip.clone() * (local.event_recv_mask[i].clone() - AB::one_maybe()),
                );
            } else {
                builder.assert_zero(chip.clone() * local.event_recv_mask[i].clone());
                builder.assert_zero(chip.clone() * local.event_values[i].clone());
            }
        }
        for i in 0..8 {
            let expected_mask = if i < 3 { AB::one_maybe() } else { AB::zero_maybe() };
            builder.assert_zero(
                local.is_active_shape_header.clone()
                    * (local.event_recv_mask[i].clone() - expected_mask),
            );
        }
        builder.assert_zero(
            chip.clone() * (local.event_values[0].clone() - local.stable_air_id_lo.clone()),
        );
        builder.assert_zero(
            chip.clone() * (local.event_values[1].clone() - local.stable_air_id_hi.clone()),
        );
        builder
            .assert_zero(chip.clone() * (local.event_values[2].clone() - local.log_height.clone()));
        builder
            .assert_zero(chip.clone() * (local.event_values[3].clone() - local.main_width.clone()));
        builder
            .assert_zero(chip.clone() * (local.event_values[4].clone() - local.chip_idx.clone()));
        builder.assert_zero(
            local.is_active_shape_header.clone()
                * (local.event_values[0].clone()
                    - AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                        dt_stark::air::ACTIVE_SHAPE_TAG_V1,
                    ))),
        );
        builder.assert_zero(
            local.is_active_shape_header.clone()
                * (local.event_values[1].clone()
                    - AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                        dt_stark::air::ACTIVE_SHAPE_VERSION_V2,
                    ))),
        );
        builder.assert_zero(
            local.is_active_shape_header.clone()
                * (local.event_values[2].clone() - local.prev_shape_chip_count.clone()),
        );
        builder.assert_zero(
            local.is_active_shape_header.clone()
                * (local.tidx_base.clone() - local.prev_tidx_acc.clone()),
        );
        builder.assert_zero(chip.clone() * (local.tidx_base.clone() - local.prev_tidx_acc.clone()));
        for i in 3..8 {
            builder
                .assert_zero(local.is_active_shape_header.clone() * local.event_values[i].clone());
        }
        builder.assert_zero(
            chip.clone()
                * (local.prep_width.clone() * local.prep_width_inv.clone()
                    - local.has_prep.clone()),
        );
        builder.assert_zero(
            chip.clone() * (AB::one_maybe() - local.has_prep.clone()) * local.prep_width.clone(),
        );
        builder.assert_zero((AB::one_maybe() - chip.clone()) * local.has_prep.clone());
        builder.assert_zero(chip.clone() * (local.chip_idx.clone() - local.prev_chip_idx.clone()));

        let one = AB::one_maybe();
        let group_start = local.is_group_start.clone();
        builder.assert_zero(
            chip.clone()
                * (one.clone() - group_start.clone())
                * (local.prev_log_height.clone() - local.log_height.clone()),
        );
        let height_range = local.prev_log_height.clone() - local.log_height.clone() - one.clone();
        let id_range =
            local.static_chip_id.clone() - local.prev_static_chip_id.clone() - one.clone();
        let expected_range = group_start.clone() * height_range + (one - group_start) * id_range;
        builder.assert_zero(chip * (local.range_val.clone() - expected_range));

        builder.assert_zero(local.is_e1.clone() * local.chain_send_chip_idx.clone());
        builder.assert_zero(
            local.is_e1.clone() * (local.chain_send_log_height.clone() - const_maybe::<AB>(25)),
        );
        builder.assert_zero(local.is_e1.clone() * local.chain_send_static_chip_id.clone());
        builder.assert_zero(
            local.is_e1.clone()
                * (local.chain_send_tidx_acc.clone()
                    - local.tidx_base.clone()
                    - const_maybe::<AB>(8)),
        );
        builder.assert_zero(local.is_e1.clone() * local.chain_send_prep_matrix_idx.clone());
        builder.assert_zero(local.is_e1.clone() * local.chain_send_first_log_height.clone());
        let active_shape_header = local.is_active_shape_header.clone();
        builder.assert_zero(
            active_shape_header.clone()
                * (local.prev_chip_idx.clone() - local.chain_send_chip_idx.clone()),
        );
        builder.assert_zero(
            active_shape_header.clone()
                * (local.prev_log_height.clone() - local.chain_send_log_height.clone()),
        );
        builder.assert_zero(
            active_shape_header.clone()
                * (local.prev_static_chip_id.clone() - local.chain_send_static_chip_id.clone()),
        );
        builder.assert_zero(
            active_shape_header.clone()
                * (local.chain_send_tidx_acc.clone()
                    - local.prev_tidx_acc.clone()
                    - const_maybe::<AB>(
                        crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_HEADER_LIMBS,
                    )),
        );
        builder.assert_zero(
            active_shape_header.clone()
                * (local.prev_prep_matrix_idx.clone() - local.chain_send_prep_matrix_idx.clone()),
        );
        builder.assert_zero(
            active_shape_header.clone()
                * (local.prev_first_log_height.clone() - local.chain_send_first_log_height.clone()),
        );
        builder.assert_zero(
            active_shape_header
                * (local.prev_shape_chip_count.clone() - local.chain_send_shape_chip_count.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_chip_idx.clone() - local.chip_idx.clone() - AB::one_maybe()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_log_height.clone() - local.log_height.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_static_chip_id.clone() - local.static_chip_id.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_tidx_acc.clone()
                    - local.prev_tidx_acc.clone()
                    - const_maybe::<AB>(
                        crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_ENTRY_LIMBS,
                    )),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_prep_matrix_idx.clone()
                    - local.prev_prep_matrix_idx.clone()
                    - local.has_prep.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.prev_chip_idx.clone() * local.prev_chip_idx_inv.clone() - AB::one_maybe()
                    + local.is_first_chip.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone() * local.is_first_chip.clone() * local.prev_chip_idx.clone(),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * local.is_first_chip.clone()
                * (local.chain_send_first_log_height.clone() - local.log_height.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (AB::one_maybe() - local.is_first_chip.clone())
                * (local.chain_send_first_log_height.clone() - local.prev_first_log_height.clone()),
        );
        builder.assert_zero(
            local.is_chip.clone()
                * (local.chain_send_shape_chip_count.clone() - local.prev_shape_chip_count.clone()),
        );

        // Segment binding: seg bits are booleans forced to bit7 of their ids by the
        // band lookups; adjacency along the 1012 chain keeps one segment per proof
        // (first chip exempt — its prev is the E1 seed); the E5 row pins the 1022 id_base
        // payload to the last chip's segment.
        assert_bool(builder, local.seg_bit.clone());
        assert_bool(builder, local.prev_seg_bit.clone());
        builder.assert_zero(
            local.is_chip.clone()
                * (AB::one_maybe() - local.is_first_chip.clone())
                * (local.seg_bit.clone() - local.prev_seg_bit.clone()),
        );
        builder.assert_zero(
            local.is_e1.clone()
                * (local.tidx_base.clone()
                    - const_maybe::<AB>(self.seed_prefix_limbs + self.num_public_values)),
        );
        builder.assert_zero(
            local.is_e5.clone()
                * (local.tidx_base.clone() - local.prev_tidx_acc.clone() - const_maybe::<AB>(10)),
        );
        builder.assert_zero(
            local.is_e5.clone()
                * (local.prev_shape_chip_count.clone() - local.prev_chip_idx.clone()),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ProofShapeBinderCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        // Order matches precompute_lc / binder_denominators:
        // 0..8 transcript payload recv; 8..10 GKV1 tag/version recv; 10 metadata recv;
        // 11 commit-root send; 12 WHIR role-config recv; 13 chip-meta send;
        // 14..16 batch-dim sends; 17..24 shape-value sends; 25 height-member send;
        // 26 chain recv; 27 chain send; 28 proof-shape-summary send; 29 range recv.
        for (direction, mult) in binder_lookup_ops(
            local,
            const_maybe::<AB>(self.role_config.num_queries),
            local.chip_meta_send_mult.clone(),
            self.contains_global_bus,
        ) {
            match direction {
                LookupDirection::Send => builder.send(mult),
                LookupDirection::Recv => builder.recv(mult),
            }
        }
    }
}

impl MachineAir<F> for ProofShapeBinderAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        // Legacy diagnostic name; wire identity comes from NativeAirId::wire_name.
        "NativeProofShapeBinder".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ProofShapeBinderTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ProofShapeBinderTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProofHeightSetAir {
    pub height_member_bus: ProofShapeHeightMemberBus,
    pub height_group_bus: ProofShapeHeightGroupBus,
    pub height_rank_bus: ProofShapeHeightRankBus,
}

impl ProofHeightSetAir {
    pub const fn new() -> Self {
        Self {
            height_member_bus: ProofShapeHeightMemberBus::new(),
            height_group_bus: ProofShapeHeightGroupBus::new(),
            height_rank_bus: ProofShapeHeightRankBus::new(),
        }
    }
}

impl Default for ProofHeightSetAir {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseAir<F> for ProofHeightSetAir {
    fn width(&self) -> usize {
        NUM_PROOF_HEIGHT_SET_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ProofHeightSetAir {
    fn width(&self) -> usize {
        NUM_PROOF_HEIGHT_SET_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.height_member_bus.required_max_beta_power_floor(),
            self.height_group_bus.required_max_beta_power_floor(),
            self.height_rank_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_PROOF_HEIGHT_SET_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &ProofHeightSetCols<AB::VarMaybeExt> = main.borrow();
            vec![
                self.height_member_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    local.height_cursor.clone(),
                ),
                self.height_group_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    local.rank.clone(),
                    local.height_cursor.clone(),
                ),
                self.height_rank_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    local.height_cursor.clone(),
                    local.rank.clone(),
                ),
                self.height_rank_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    local.height_cursor.clone() - AB::one_maybe(),
                    local.rank.clone() + local.present.clone(),
                ),
            ]
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ProofHeightSetCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_first.clone());
        assert_bool(builder, local.is_last.clone());
        assert_bool(builder, local.present.clone());
        builder.assert_zero(
            local.is_first.clone() * (local.height_cursor.clone() - const_maybe::<AB>(24)),
        );
        builder.assert_zero(local.is_first.clone() * local.rank.clone());
        builder.assert_zero(local.is_last.clone() * local.height_cursor.clone());
        builder.assert_zero(
            local.member_count.clone() * local.member_count_inv.clone() - local.present.clone(),
        );
        builder.assert_zero((AB::one_maybe() - local.present.clone()) * local.member_count.clone());
        assert_bool(builder, local.height_group_send_mult.clone());
        builder.assert_zero(
            local.height_group_send_mult.clone() * (AB::one_maybe() - local.present.clone()),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ProofHeightSetCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        builder.recv(local.member_count.clone());
        builder.send(local.height_group_send_mult.clone());
        builder.recv(local.is_valid.clone() - local.is_first.clone());
        builder.send(local.is_valid.clone() - local.is_last.clone());
    }
}

impl MachineAir<F> for ProofHeightSetAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeProofHeightSet".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ProofHeightSetTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ProofHeightSetTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn binder_denominators<AB: FullAirBuilder>(
    air: &ProofShapeBinderAir,
    builder: &AB,
    local: &ProofShapeBinderCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    let mut denominators = Vec::with_capacity(34 + usize::from(air.contains_global_bus));
    for i in 0..8 {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            local.tidx_base.clone() + const_maybe::<AB>(i),
            AB::zero_maybe(),
            local.event_values[i].clone(),
        ));
    }
    denominators.push(air.transcript_event_bus.denominator(
        builder,
        proof_idx.clone(),
        AB::zero_maybe(),
        AB::zero_maybe(),
        const_maybe::<AB>(BATCH_VK_TAG_V1 as usize),
    ));
    denominators.push(air.transcript_event_bus.denominator(
        builder,
        proof_idx.clone(),
        AB::one_maybe(),
        AB::zero_maybe(),
        const_maybe::<AB>(BATCH_VK_VERSION_V1 as usize),
    ));
    denominators.push(air.metadata_bus.denominator(
        builder,
        local.role_id.clone(),
        local.static_chip_id.clone(),
        local.stable_air_id_lo.clone(),
        local.stable_air_id_hi.clone(),
        local.prep_width.clone(),
        local.main_width.clone(),
        local.perm_width.clone(),
        local.constraint_count.clone(),
        local.gate_count.clone(),
    ));
    denominators.push(air.commitment_root_bus.denominator(
        builder,
        proof_idx.clone(),
        local.commit_id.clone(),
        local.event_values.clone(),
    ));
    denominators.push(air.chip_meta_bus.denominator(
        builder,
        proof_idx.clone(),
        local.chip_idx.clone(),
        local.static_chip_id.clone(),
        local.log_height.clone(),
        local.gate_count.clone(),
        AB::mul_base(local.perm_width.clone(), AB::F::from_canonical_usize(D_EF).inverse()),
    ));
    denominators.push(air.batch_dim_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(PROOF_SHAPE_BATCH_PREPROCESSED),
        local.prev_prep_matrix_idx.clone(),
        local.chip_idx.clone(),
        local.static_chip_id.clone(),
        local.prep_width.clone(),
        local.log_height.clone(),
    ));
    denominators.push(air.batch_dim_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(PROOF_SHAPE_BATCH_MAIN),
        local.chip_idx.clone(),
        local.chip_idx.clone(),
        local.static_chip_id.clone(),
        local.main_width.clone(),
        local.log_height.clone(),
    ));
    denominators.push(air.batch_dim_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(PROOF_SHAPE_BATCH_PERMUTATION),
        local.chip_idx.clone(),
        local.chip_idx.clone(),
        local.static_chip_id.clone(),
        local.perm_width.clone(),
        local.log_height.clone(),
    ));
    let shape_namespace = local.is_vk_commit.clone() + local.is_vk_meta.clone();
    for i in 0..8 {
        denominators.push(air.values_bus.denominator(
            builder,
            proof_idx.clone(),
            shape_namespace.clone(),
            local.shape_idx_base.clone() + const_maybe::<AB>(i),
            local.event_values[i].clone(),
        ));
    }
    denominators.push(air.height_member_bus.denominator(
        builder,
        proof_idx.clone(),
        local.log_height.clone(),
    ));
    denominators.push(air.chain_bus.denominator(
        builder,
        proof_idx.clone(),
        local.prev_chip_idx.clone(),
        local.prev_log_height.clone(),
        local.prev_static_chip_id.clone(),
        local.prev_tidx_acc.clone(),
        local.prev_prep_matrix_idx.clone(),
        local.prev_first_log_height.clone(),
        local.prev_shape_chip_count.clone(),
    ));
    denominators.push(air.chain_bus.denominator(
        builder,
        proof_idx,
        local.chain_send_chip_idx.clone(),
        local.chain_send_log_height.clone(),
        local.chain_send_static_chip_id.clone(),
        local.chain_send_tidx_acc.clone(),
        local.chain_send_prep_matrix_idx.clone(),
        local.chain_send_first_log_height.clone(),
        local.chain_send_shape_chip_count.clone(),
    ));
    denominators.push(air.summary_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.prev_first_log_height.clone(),
        local.prev_chip_idx.clone(),
        const_maybe::<AB>(air.num_public_values),
        local.prev_seg_bit.clone() * const_maybe::<AB>(128),
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage { value: local.range_val.clone(), max_bits: const_maybe::<AB>(8) },
    ));
    // Segment band (dual range8 per side, exact [128*b, 128*b + 128) membership):
    // chip rows bind their own id; chip rows AND the E5 row bind the chain-recv'd prev id
    // (on E5 that is the LAST chip, which pins the 1022 id_base payload).
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.static_chip_id.clone() - local.seg_bit.clone() * const_maybe::<AB>(128),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: const_maybe::<AB>(127)
                - (local.static_chip_id.clone() - local.seg_bit.clone() * const_maybe::<AB>(128)),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.prev_static_chip_id.clone()
                - local.prev_seg_bit.clone() * const_maybe::<AB>(128),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: const_maybe::<AB>(127)
                - (local.prev_static_chip_id.clone()
                    - local.prev_seg_bit.clone() * const_maybe::<AB>(128)),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators.push(air.fold_plan_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        AB::zero_maybe(),
        local.prev_chip_idx.clone(),
        AB::zero_maybe(),
    ));
    if air.contains_global_bus {
        denominators.push(air.global_packed_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.shape_idx_base.clone(),
            &local.event_values,
        ));
    }
    denominators
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::D_EF;
    use polyair::Chip;

    #[test]
    fn symbolic_analysis() {
        let metadata = NativeChipMetadataAir::new(Vec::new());
        let metadata_chip = Chip::<NativeChipMetadataAir, F, D_EF>::new(metadata);
        assert_eq!(metadata_chip.num_lookup(), 1);
        assert!(metadata_chip.required_max_beta_power() >= 13);
        assert!(metadata_chip.degree <= 3);

        let binder = Chip::<ProofShapeBinderAir, F, D_EF>::new(ProofShapeBinderAir::new(
            0,
            crate::whir_dt::whir_role_config(crate::whir_dt::WHIR_ROLE_CORE),
            true,
        ));
        assert_eq!(binder.num_lookup(), 35);
        assert!(binder.required_max_beta_power() >= 14);
        assert!(binder.degree <= 3);

        let heights = Chip::<ProofHeightSetAir, F, D_EF>::new(ProofHeightSetAir::new());
        assert_eq!(heights.num_lookup(), 4);
        assert!(heights.degree <= 3);
    }
}
