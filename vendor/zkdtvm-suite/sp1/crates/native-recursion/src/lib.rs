//! Native recursion has 11 layer-invariant AIR families and 17 families with
//! distinct L1/L2/L3/L4 identities.
//!
//! For auditing, the layer-specific families are documented as ten whose
//! symbolic AIR changes and seven whose symbolic eval is stable while fixed
//! program/preprocessed data changes. This subdivision is documentation and
//! test taxonomy, not runtime dispatch or cache identity.
//!
//! Dynamic L2 round/depth and trace height are not AIR identity dimensions.

pub mod batch_constraint_dt;
pub mod bus;
pub mod child_views;
pub mod compress_dt;
pub mod config;
pub mod constraint_replay_dt;
pub mod interaction;
pub mod interaction_full_air_dt;
pub mod interaction_registry_dt;
pub mod machine_dt;
pub mod native_air_dt;
pub mod primitives_dt;
pub mod proof_shape_dt;
pub mod statement_boundary_air_dt;
pub mod statement_config_air_dt;
pub mod statement_dt;
pub mod statement_hash_air_dt;
pub mod symbolic_expr_adapter_dt;
pub mod symbolic_expr_fixed_dt;
pub mod symbolic_ir_dt;
pub mod system_dt;
mod tracegen_backend;
pub mod transcript_dt;
pub mod validate;
pub mod verifier_dt;
pub mod whir_dt;

pub use tracegen_backend::TracegenAuthorityHandle;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Self
    }

    pub(crate) fn elapsed(self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    pub(crate) fn saturating_duration_since(self, _earlier: Self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

pub(crate) fn env_var(name: &str) -> Result<String, std::env::VarError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        Err(std::env::VarError::NotPresent)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var(name)
    }
}

/// Gates all diagnostic stdout prints (chip profiles, dedup split counters,
/// proof-size slices, pool statistics). Off by default; set
/// `DT_NATIVE_RECURSION_DEBUG=1` to enable.
pub fn debug_prints_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            env_var("DT_NATIVE_RECURSION_DEBUG").map(|v| v == "1").unwrap_or(false)
        }
    })
}

pub mod prelude {
    pub use crate::{
        batch_constraint_dt::{
            batch_sumcheck_rows, batch_transcript_input_rows,
            record_batch_constraint_materials_from_views, BatchConstraintRecordError,
            BatchOpeningPointBus, BatchSumcheckAir, BatchSumcheckClaimChainBus, BatchSumcheckCols,
            BatchSumcheckTraceGenerator, BatchTranscriptInputsAir, BatchTranscriptInputsCols,
            BatchTranscriptInputsTraceGenerator, BatchTranscriptLayout, SumcheckOutBus,
            BATCH_ROUND_EVENT_LIMBS, BATCH_SUMCHECK_EVALS, NUM_BATCH_SUMCHECK_COLS,
            NUM_BATCH_TRANSCRIPT_INPUTS_COLS, SUMCHECK_OUT_ALPHA, SUMCHECK_OUT_EQ,
            SUMCHECK_OUT_PERM_ALPHA, SUMCHECK_OUT_PERM_BETA,
        },
        child_views::{
            NativeAirAuthority, NativeChildColumnKind, NativeChildMetadataView,
            NativeChildProofView, NativeChildRole, NativeChildVerifierConfigView,
            NativeChildViewError, NativeChildViewResult, NativeChildViews, NativeChipMetadata,
            NativeOpenedChipView, NativePcsBatchKind, NativePcsBatchView, NativePcsOpeningView,
            NativeVerifierRoundShape, NativeWhirConfigView, VerifiedChildLayout,
            KOALABEAR_MAX_TRACE_LOG_HEIGHT,
        },
        config::{
            Challenge, Challenger, ChildMlChallenger, ChildMlCommitment, ChildMlPcsOpeningProof,
            ChildMlPcsProverData, Digest, Mlpcs, Perm, Val, CHUNK, DIGEST_SIZE, D_EF, EF, F,
            POSEIDON2_WIDTH, SC,
        },
        constraint_replay_dt::{
            annotate_constraint_replay_publications, constraint_beta_ladder_rows,
            constraint_challenge_rows, constraint_dag_rows, constraint_fold_rows,
            constraint_program_rows, constraint_replay_bus_residual_report,
            constraint_root_table_rows, constraint_terminal_rows, ConstraintBetaLadderAir,
            ConstraintBetaLadderCols, ConstraintBetaLadderRow, ConstraintBetaLadderTraceGenerator,
            ConstraintBoundaryAir, ConstraintBoundaryCols, ConstraintBoundaryTraceGenerator,
            ConstraintChallengeAir, ConstraintChallengeBus, ConstraintChallengeCols,
            ConstraintChallengeRow, ConstraintChallengeTraceGenerator, ConstraintDagEvalAir,
            ConstraintDagEvalCols, ConstraintDagEvalTraceGenerator, ConstraintDagRow,
            ConstraintEqChainBus, ConstraintFoldAir, ConstraintFoldChainBus, ConstraintFoldCols,
            ConstraintFoldRow, ConstraintFoldTraceGenerator, ConstraintNodeValueBus,
            ConstraintProgramBus, ConstraintProgramCols, ConstraintProgramPreprocessedCols,
            ConstraintProgramRow, ConstraintProgramTableAir, ConstraintProgramTraceGenerator,
            ConstraintReplayBusResidualReport, ConstraintRootTableAir, ConstraintRootTableBus,
            ConstraintRootTableCols, ConstraintRootTablePreprocessedCols, ConstraintRootTableRow,
            ConstraintRootTableTraceGenerator, ConstraintTerminalAir, ConstraintTerminalCols,
            ConstraintTerminalRow, ConstraintTerminalTraceGenerator,
            CONSTRAINT_CHALLENGE_BETA_POWER, CONSTRAINT_CHALLENGE_BETA_SEPTIX,
            CONSTRAINT_CHALLENGE_IS_FIRST, CONSTRAINT_CHALLENGE_IS_LAST,
            CONSTRAINT_CHALLENGE_PERM_ALPHA, CONSTRAINT_FOLD_BATCH_SIZE,
            CONSTRAINT_FOLD_ROOT_SLOTS, CONSTRAINT_LEAF_BETA_POWER, CONSTRAINT_LEAF_BETA_SEPTIX,
            CONSTRAINT_LEAF_IS_FIRST_ROW, CONSTRAINT_LEAF_IS_LAST_ROW, CONSTRAINT_LEAF_KIND_COUNT,
            CONSTRAINT_LEAF_MAIN, CONSTRAINT_LEAF_PERM_ALPHA, CONSTRAINT_LEAF_PRECOMPUTED,
            CONSTRAINT_LEAF_PREPROCESSED, CONSTRAINT_LEAF_PUBLIC, CONSTRAINT_LEAF_RESERVED_POLY,
            CONSTRAINT_MAX_BETA_POWERS, CONSTRAINT_ROOT_GATE, CONSTRAINT_ROOT_MULTIPLICITY,
            CONSTRAINT_ROOT_PRECOMPUTE_DENOM, CONSTRAINT_TERMINAL_LCS_LIMBS,
            NUM_CONSTRAINT_BETA_LADDER_COLS, NUM_CONSTRAINT_BOUNDARY_COLS,
            NUM_CONSTRAINT_CHALLENGE_COLS,
            NUM_CONSTRAINT_DAG_EVAL_COLS, NUM_CONSTRAINT_FOLD_COLS, NUM_CONSTRAINT_PROGRAM_COLS,
            NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS, NUM_CONSTRAINT_ROOT_TABLE_COLS,
            NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS, NUM_CONSTRAINT_TERMINAL_COLS,
        },
        interaction::{
            validate_recursion_interaction_budget, RecursionInteractionBudget,
            RecursionInteractionBudgetError, RecursionInteractionBuilder, RecursionInteractionIdx,
            RecursionInteractionIndexSpace, RecursionInteractionKind,
            RecursionInteractionLoweringSpec, RecursionLookupInteraction,
            RecursionPermutationInteraction,
        },
        interaction_full_air_dt::RecursionFullAirBus,
        interaction_registry_dt::{
            validate_recursion_interaction_registry,
            validate_registered_recursion_interaction_budget, RecursionInteractionSchema,
            RecursionRegisteredInteractionBudget, BATCH_OPENING_POINT_SCHEMA,
            BATCH_SUMCHECK_CLAIM_CHAIN_SCHEMA, BETA_LADDER_CHAIN_SCHEMA,
            CONSTRAINT_CHALLENGE_SCHEMA, CONSTRAINT_EQ_CHAIN_SCHEMA, CONSTRAINT_FOLD_CHAIN_SCHEMA,
            CONSTRAINT_NODE_VALUE_SCHEMA, CONSTRAINT_PROGRAM_SCHEMA,
            GLOBAL_RECURSION_INTERACTION_IDX_START, MERKLE_COMMITMENT_ROOT_SCHEMA,
            MERKLE_DIGEST_CHAIN_SCHEMA, MERKLE_LEAF_BLOCK_SCHEMA, MERKLE_SPONGE_STATE_CHAIN_SCHEMA,
            NATIVE_CHIP_METADATA_SCHEMA, NATIVE_RECURSION_SCHEMAS,
            PER_PROOF_RECURSION_INTERACTION_IDX_START, POSEIDON2_PERMUTE_SCHEMA,
            PROOF_SHAPE_BATCH_DIM_SCHEMA, PROOF_SHAPE_CHAIN_SCHEMA, PROOF_SHAPE_CHIP_META_SCHEMA,
            PROOF_SHAPE_HEIGHT_GROUP_SCHEMA, PROOF_SHAPE_HEIGHT_MEMBER_SCHEMA,
            PROOF_SHAPE_HEIGHT_RANK_SCHEMA, PROOF_SHAPE_SUMMARY_SCHEMA, PROOF_SHAPE_VALUES_SCHEMA,
            RANGE_CHECKER_SCHEMA, STATEMENT_CHILD_FACTS_SCHEMA, STATEMENT_CONFIG_SCHEMA,
            STATEMENT_DIGEST_CHAIN_SCHEMA, STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA,
            STATEMENT_HASH_CHAIN_SCHEMA,
            STATEMENT_SCALAR_CHAIN_SCHEMA, STATEMENT_VK_DIGEST_SCHEMA, SUMCHECK_OUT_SCHEMA,
            WHIR_EVAL_CHAIN_SCHEMA, WHIR_GROUP_CLAIM_SCHEMA, WHIR_LEAF_CHAIN_SCHEMA,
            WHIR_OPENED_EVAL_SCHEMA, WHIR_QUERY_CHAIN_SCHEMA, WHIR_QUERY_INIT_SCHEMA,
            WHIR_QUERY_LEAF_SUM_SCHEMA, WHIR_ROUND_BCAST_SCHEMA, WHIR_ROUND_CHAIN_SCHEMA,
            WHIR_SAMPLE_BAND_SCHEMA, WHIR_TWIDDLE_POW_SCHEMA,
        },
        machine_dt::{
            assert_machine_record_fully_published, assert_native_recursion_record_residuals,
            build_core_native_recursion_program, build_dual_segment_reduce_program,
            build_mixed_reduce_program, build_native_recursion_program, build_root_shrink_program,
            core_recording_machine, native_child_verifier_config,
            native_child_verifier_config_for_role, native_metadata_for_shard,
            native_metadata_from_machine, native_recording_machine,
            native_recording_machine_for_stage,
            prepare_recursion_tracegen_record_compact_with_timing,
            prepare_recursion_tracegen_record_with_timing, proof_shape_static_chip_id_map,
            record_native_proof_shard, record_native_proof_shard_in_segment, verify_recursion,
            CoreRecordingChip, CoreRecordingMachine, NativeProverFor, NativeRecordingMachine,
            NativeRecursionAir, NativeRecursionAssemblyError, NativeRecursionAssemblyResult,
            NativeRecursionMachine, NativeRecursionProver, PreparedCompactRecursionTracegenRecord,
            PreparedRecursionTracegenRecord, ProveRecursionMetrics, ProveRecursionTimings,
            RecursionTraceCost, MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
        },
        primitives_dt::{
            bus::{RangeCheckerBus, RangeCheckerBusMessage},
            pow::PowerCheckerCounts,
            range::{RangeCheckerAir, RangeCheckerCols, RangeCheckerTraceGenerator},
        },
        proof_shape_dt::{
            metadata_universe_from_view, native_chip_metadata_trace_rows, proof_height_set_rows,
            proof_shape_binder_rows, record_proof_shape_from_views, NativeChipMetadataAir,
            NativeChipMetadataBus, NativeChipMetadataCols, NativeChipMetadataPreprocessedCols,
            NativeChipMetadataTraceGenerator, ProofHeightSetAir, ProofHeightSetCols,
            ProofHeightSetTraceGenerator, ProofShapeBatchDimBus, ProofShapeBinderAir,
            ProofShapeBinderCols, ProofShapeBinderTraceGenerator, ProofShapeChainBus,
            ProofShapeChipMetaBus, ProofShapeHeightGroupBus, ProofShapeHeightMemberBus,
            ProofShapeHeightRankBus, ProofShapeRecordError, ProofShapeSummaryBus,
            ProofShapeGlobalPackedBus, ProofShapeValuesBus, NUM_NATIVE_CHIP_METADATA_COLS,
            NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS, NUM_PROOF_HEIGHT_SET_COLS,
            NUM_PROOF_SHAPE_BINDER_COLS, PROOF_SHAPE_BATCH_MAIN, PROOF_SHAPE_BATCH_PERMUTATION,
            PROOF_SHAPE_BATCH_PREPROCESSED, PROOF_SHAPE_COMMIT_MAIN,
            PROOF_SHAPE_COMMIT_PERMUTATION, PROOF_SHAPE_COMMIT_VK,
            PROOF_SHAPE_CORE_VK_META_VALUE_COUNT, PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES,
            PROOF_SHAPE_NAMESPACE_VK_META, PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT,
            PROOF_SHAPE_VK_META_COMMIT_BASE, PROOF_SHAPE_VK_META_COMMIT_ELTS,
            PROOF_SHAPE_VK_META_BOUNDARY_BASE, PROOF_SHAPE_VK_META_BOUNDARY_ELTS,
            PROOF_SHAPE_VK_META_BOUNDARY_KIND, PROOF_SHAPE_VK_META_BOUNDARY_X_BASE,
            PROOF_SHAPE_VK_META_BOUNDARY_Y_BASE, PROOF_SHAPE_VK_META_PC_START,
            PROOF_SHAPE_VK_META_VALUE_COUNT,
        },
        statement_boundary_air_dt::{
            annotate_statement_publications, StatementBoundaryAir, StatementBoundaryCols,
            StatementChildFactsBus, StatementDigestChainBus, StatementGlobalIntervalChainBus,
            StatementHashChainBus, StatementScalarChainBus, StatementVkDigestBus,
            NUM_STATEMENT_BOUNDARY_COLS,
        },
        statement_config_air_dt::{
            StatementConfigAir, StatementConfigBus, StatementConfigCols,
            StatementConfigPreprocessedCols, NUM_STATEMENT_CONFIG_COLS,
            NUM_STATEMENT_CONFIG_PREPROCESSED_COLS,
        },
        statement_dt::{
            child_vk_digest, native_vk_statement_digest, poseidon2_hash_slice,
            resolve_child_vk_class, ChildVkClass, NativeRecursionPublicValues, SpecStatement,
            SpecStatementError, CORE_CHILD_NUM_PUBLIC_VALUES, CORE_PV_COMMITTED_VALUE_DIGEST_ELTS,
            CORE_PV_COMMITTED_VALUE_DIGEST_START, CORE_PV_DEFERRED_PROOFS_DIGEST_ELTS,
            CORE_PV_DEFERRED_PROOFS_DIGEST_START, CORE_PV_EMPTY, CORE_PV_EXECUTION_SHARD,
            CORE_PV_EXIT_CLK, CORE_PV_EXIT_CODE, CORE_PV_LAST_FINALIZE_ADDR,
            CORE_PV_LAST_INIT_ADDR, CORE_PV_NEXT_PC, CORE_PV_PREVIOUS_FINALIZE_ADDR,
            CORE_PV_PREVIOUS_INIT_ADDR, CORE_PV_SHARD, CORE_PV_START_CLK, CORE_PV_START_PC,
            NATIVE_PV_COMMITTED_VALUE_DIGEST_ELTS, NATIVE_PV_COMMITTED_VALUE_DIGEST_START,
            NATIVE_PV_CONTAINS_EXECUTION_SHARD, NATIVE_PV_DEFERRED_PROOFS_DIGEST_ELTS,
            NATIVE_PV_DEFERRED_PROOFS_DIGEST_START, NATIVE_PV_DIGEST_START,
            NATIVE_PV_DT_VK_DIGEST_START, NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START,
            NATIVE_PV_EXIT_CODE, NATIVE_PV_GLOBAL_INTERVAL_END, NATIVE_PV_GLOBAL_INTERVAL_START,
            NATIVE_PV_GLOBAL_STATE_ELTS, NATIVE_PV_IS_COMPLETE, NATIVE_PV_LAST_FINALIZE_ADDR,
            NATIVE_PV_LAST_INIT_ADDR, NATIVE_PV_NEXT_EXECUTION_SHARD, NATIVE_PV_NEXT_PC,
            NATIVE_PV_NEXT_SHARD, NATIVE_PV_PREVIOUS_FINALIZE_ADDR, NATIVE_PV_PREVIOUS_INIT_ADDR,
            NATIVE_PV_START_EXECUTION_SHARD, NATIVE_PV_START_PC,
            NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START, NATIVE_PV_START_SHARD,
            NATIVE_PV_VK_ROOT_START, NATIVE_RECURSION_NUM_PV_ELMS_TO_HASH,
            NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
            STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        },
        statement_hash_air_dt::{
            root_digest_hash_input, root_digest_input_pv_indices, root_public_values_digest,
            StatementDigestMode, StatementHashAir, StatementHashCols, StatementHashRootCols,
            StatementHashTraceGenerator, NUM_STATEMENT_HASH_COLS, NUM_STATEMENT_HASH_ROOT_COLS,
            STATEMENT_HASH_KIND_ROOT_DIGEST, STATEMENT_HASH_KIND_SELF_DIGEST,
            STATEMENT_HASH_KIND_VK_DIGEST, STATEMENT_ROOT_DIGEST_BLOCKS,
        },
        symbolic_expr_adapter_dt::{
            RecursionAdaptedRoot, RecursionAdaptedRootStreams, RecursionAdapterError,
            RecursionOpMix, RecursionPolyAirLeaf, RecursionPolyAirNode, RecursionPolyAirOp,
            RecursionRootKind, RecursionSymbolicExprAdapter,
        },
        symbolic_expr_fixed_dt::{
            RecursionChildRole, RecursionFixedSymbolicChip, RecursionFixedSymbolicProgram,
            RecursionFixedSymbolicProgramError, RecursionSymbolicBuilderSnapshot,
        },
        symbolic_ir_dt::{
            evaluate_chip_node_values, evaluate_chip_replay, evaluate_derived_roots,
            evaluate_gate_roots, evaluate_lookup_batches, evaluate_node_table,
            evaluate_precomputed_lc, evaluate_reserved_poly_values,
            evaluate_signed_lookup_multiplicities, fold_gate_values, RecursionD0CostLedger,
            RecursionEvaluatedDerivedRoot, RecursionPolyAirChipEval, RecursionPolyAirChipIr,
            RecursionPolyAirConstraintRoot, RecursionPolyAirDerivedRoot, RecursionPolyAirEnv,
            RecursionPolyAirEvaluationError, RecursionPolyAirLookupBatchEval,
            RecursionPolyAirLookupRoot, RecursionPolyAirVerifierProgram, RecursionPolyAirWidths,
            RecursionStaticChipBinding,
        },
        system_dt::{
            BuildingRecord, CoreRecordingChallenger, RecordingSC, RecordingStage,
            RecursionBatchConstraintRecord, RecursionBatchCumSumRecord, RecursionConstraintEvent,
            RecursionConstraintRecord, RecursionMerklePathEvent, RecursionMerklePathOp,
            RecursionMerklePathRecord, RecursionMerklePathRow, RecursionNativeChipMetadataPool,
            RecursionNativeChipMetadataRequest, RecursionNativeProgram, RecursionPoseidon2Pool,
            RecursionPoseidon2Request, RecursionPowerPool, RecursionPowerRequest,
            RecursionProofRecord, RecursionProofShapeChip, RecursionProofShapeRecord,
            RecursionRangePool, RecursionRangeRequest, RecursionRecord,
            RecursionRecordingChallenger, RecursionStatementRole, RecursionSumcheckRoundRecord,
            RecursionTranscriptBitsEvent, RecursionTranscriptEvent, RecursionTranscriptEventKind,
            RecursionTranscriptRecord, RecursionWhirBatchEvalRow, RecursionWhirLeafExtStreamRow,
            RecursionWhirLeafExtStreamTraceRow, RecursionWhirLeafStreamRow,
            RecursionWhirQueryFoldRow, RecursionWhirRecord, RecursionWhirRoundRow, SpecSponge,
            SpecSpongeBlock, SpecSpongeError, StatementConfigRow, WhirQueryPairSource,
            WhirSpecFoldError, WhirSpecFoldSeed, WhirSpecFoldShape,
        },
        transcript_dt::{
            bus::Poseidon2PermuteBus,
            merkle_path::{
                trace_row as merkle_path_trace_row, MerkleCommitmentRootBus, MerkleDigestChainBus,
                MerkleLeafBlockBus, MerklePathAir, MerklePathCols, MerklePathTraceGenerator,
                MerkleSpongeStateChainBus, NUM_MERKLE_PATH_COLS,
            },
            poseidon2::{
                poseidon2_permute, Poseidon2ColsView, Poseidon2PermuteAir,
                Poseidon2PermuteTraceGenerator, NUM_POSEIDON2_PERMUTE_COLS,
                NUM_POSEIDON2_PERMUTE_DENOMINATOR_VALUES, NUM_POSEIDON2_PERMUTE_PAYLOAD_VALUES,
            },
            sponge::{
                transcript_sponge_row_count, transcript_sponge_rows, transcript_sponge_trace_row,
                TranscriptEventBus, TranscriptSpongeAir, TranscriptSpongeChainBus,
                TranscriptSpongeCols, TranscriptSpongeTraceGenerator, NUM_TRANSCRIPT_SPONGE_COLS,
            },
        },
        validate::{
            check_lookup_residuals, check_provider_pools, check_real_trace_constraints,
            check_traces_match_plan, exact_pre_trace_gate, ExactTraceActiveRowOverride,
            ExactTracePlan, NativeRecursionPoolStats, NativeRecursionValidationError,
            NativeRecursionValidationResult, PlannedChipTrace, NATIVE_RECURSION_ALLOWED_RANGE_BITS,
        },
        whir_dt::{
            sample_band_for_query_bits, whir_batch_eval_rows, whir_bus_residual_report,
            whir_leaf_ext_stream_rows, whir_leaf_stream_rows, whir_query_fold_rows,
            whir_role_config, whir_role_configs, whir_round_rows, whir_sample_band_rows,
            WhirBatchEvalAir, WhirBatchEvalCols, WhirBatchEvalTraceGenerator,
            WhirBusResidualReport, WhirEvalChainBus, WhirGroupClaimBus, WhirLeafChainBus,
            WhirLeafExtStreamAir, WhirLeafExtStreamCols, WhirLeafExtStreamTraceGenerator,
            WhirLeafPowSeedBus, WhirLeafStreamAir, WhirLeafStreamCols,
            WhirLeafStreamTraceGenerator, WhirOpenedEvalBus, WhirQueryChainBus, WhirQueryFoldAir,
            WhirQueryFoldCols, WhirQueryFoldPackedCols, WhirQueryFoldReservedCols,
            WhirQueryFoldTraceGenerator, WhirQueryInitBus, WhirQueryLeafSumBus, WhirRecordError,
            WhirRoleConfig, WhirRoundAir, WhirRoundBcastBus, WhirRoundChainBus, WhirRoundCols,
            WhirRoundTraceGenerator, WhirSampleBandAir, WhirSampleBandBus, WhirSampleBandCols,
            WhirSampleBandConfig, WhirSampleBandPreprocessedCols, WhirSampleBandTraceGenerator,
            WhirTwiddleCols, WhirTwiddlePowBus, WhirTwiddlePreprocessedCols, WhirTwiddleTableAir,
            WhirTwiddleTraceGenerator, NUM_WHIR_BATCH_EVAL_COLS, NUM_WHIR_LEAF_EXT_STREAM_COLS,
            NUM_WHIR_LEAF_STREAM_COLS, NUM_WHIR_QUERY_FOLD_COLS,
            NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS, NUM_WHIR_QUERY_FOLD_PACKED_COLS,
            NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS, NUM_WHIR_QUERY_FOLD_RESERVED_COLS,
            NUM_WHIR_ROUND_COLS, NUM_WHIR_SAMPLE_BAND_COLS, NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS,
            NUM_WHIR_TWIDDLE_COLS, NUM_WHIR_TWIDDLE_PREPROCESSED_COLS, WHIR_ROLE_COMPRESS,
            WHIR_ROLE_CORE, WHIR_ROLE_COUNT, WHIR_ROLE_SHRINK, WHIR_SAMPLE_BAND_ROWS,
            WHIR_TWIDDLE_ROWS, WHIR_TWIDDLE_TABLES,
        },
        TracegenAuthorityHandle,
    };
}
