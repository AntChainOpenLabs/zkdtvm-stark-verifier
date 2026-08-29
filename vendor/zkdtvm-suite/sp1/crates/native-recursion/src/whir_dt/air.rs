use core::{borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{ChallengeExtension, FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    batch_constraint_dt::BatchOpeningPointBus,
    config::{D_EF, F, POSEIDON2_WIDTH},
    primitives_dt::bus::{RangeCheckerBus, RangeCheckerBusMessage},
    proof_shape_dt::{ProofShapeBatchDimBus, ProofShapeHeightGroupBus, ProofShapeSummaryBus},
    system_dt::{RecursionNativeProgram, RecursionRecord, WHIR_BATCH_PERMUTATION},
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        merkle_path::{MerkleCommitmentRootBus, MerkleLeafBlockBus},
        sponge::TranscriptEventBus,
    },
    whir_dt::{
        bus::{
            WhirEvalChainBus, WhirFinalRootChainBus, WhirGroupClaimBus, WhirLeafChainBus,
            WhirLeafPowSeedBus, WhirOpenedEvalBus, WhirQueryChainBus, WhirQueryInitBus,
            WhirQueryLeafSumBus, WhirRoundBcastBus, WhirRoundChainBus, WhirSampleBandBus,
            WhirTwiddlePowBus,
        },
        columns::{
            whir_unit_key, WhirBatchEvalCols, WhirLeafExtStreamCols,
            WhirLeafExtStreamDenominatorCols, WhirLeafExtStreamPackedCols,
            WhirLeafExtStreamPrecomputedCols, WhirLeafExtStreamReservedCols, WhirLeafStreamCols,
            WhirQueryFoldCols, WhirQueryFoldPackedCols, WhirQueryFoldReservedCols, WhirRoundCols,
            NUM_WHIR_BATCH_EVAL_COLS, NUM_WHIR_LEAF_EXT_STREAM_COLS, NUM_WHIR_LEAF_STREAM_COLS,
            NUM_WHIR_QUERY_FOLD_COLS, NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS,
            NUM_WHIR_QUERY_FOLD_PACKED_COLS, NUM_WHIR_ROUND_COLS, NUM_WHIR_SAMPLE_BAND_COLS,
            NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS, NUM_WHIR_TWIDDLE_COLS,
            NUM_WHIR_TWIDDLE_PREPROCESSED_COLS, WHIR_BATCHING_POW_HIGH_MAX,
            WHIR_BATCHING_POW_SHIFT, WHIR_FINAL_ROOT_DIGEST_LANES, WHIR_FINAL_ROOT_POSEIDON2_PERMS,
            WHIR_INPUT_PERMUTATION_PATH_SLOT, WHIR_LEAF_BASE_LIMBS_PER_ROW,
            WHIR_LEAF_BLOCKS_PER_ROW, WHIR_LEAF_RLC_SLOTS, WHIR_PAIRED_RANGE_BITS,
            WHIR_QUERY_PAIR_LEAF_BLOCKS, WHIR_QUERY_POW_HIGH_MAX, WHIR_QUERY_POW_SHIFT,
            WHIR_ROUND_MAX_TRANSCRIPT_EVENTS, WHIR_TWIDDLE_TABLES, WHIR_UNIT_KEY_SLOT_STRIDE,
        },
        trace::{
            whir_role_config, WhirBatchEvalTraceGenerator, WhirLeafExtStreamTraceGenerator,
            WhirLeafStreamTraceGenerator, WhirQueryFoldTraceGenerator, WhirRoleConfig,
            WhirRoundTraceGenerator, WhirSampleBandTraceGenerator, WhirTwiddleTraceGenerator,
        },
        WHIR_ROLE_CORE,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct WhirTwiddleTableAir {
    pub bus: WhirTwiddlePowBus,
}

impl Default for WhirTwiddleTableAir {
    fn default() -> Self {
        Self { bus: WhirTwiddlePowBus::new() }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirTwiddleTableAir {
    fn width(&self) -> usize {
        NUM_WHIR_TWIDDLE_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirTwiddleTableAir {
    fn width(&self) -> usize {
        NUM_WHIR_TWIDDLE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_WHIR_TWIDDLE_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_WHIR_TWIDDLE_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let prep = builder.preprocessed();
            let local: &crate::whir_dt::columns::WhirTwiddlePreprocessedCols<AB::VarMaybeExt> =
                prep.borrow();
            (0..WHIR_TWIDDLE_TABLES)
                .map(|table_id| {
                    self.bus.denominator(
                        builder,
                        const_maybe::<AB>(table_id),
                        local.byte.clone(),
                        local.values[table_id].clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, _builder: &mut AB) {}

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        // Order matches precompute_lc: table_id 0, 1, 2.
        for table_id in 0..WHIR_TWIDDLE_TABLES {
            builder.send(local[NUM_WHIR_TWIDDLE_PREPROCESSED_COLS + table_id].clone());
        }
    }
}

impl MachineAir<F> for WhirTwiddleTableAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "WhirTwiddleTable".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_WHIR_TWIDDLE_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(WhirTwiddleTraceGenerator::trace_height())
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Some(WhirTwiddleTraceGenerator::generate_preprocessed_trace())
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(WhirTwiddleTraceGenerator::trace_height())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirTwiddleTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirSampleBandAir {
    pub bus: WhirSampleBandBus,
}

impl Default for WhirSampleBandAir {
    fn default() -> Self {
        Self { bus: WhirSampleBandBus::new() }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirSampleBandAir {
    fn width(&self) -> usize {
        NUM_WHIR_SAMPLE_BAND_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirSampleBandAir {
    fn width(&self) -> usize {
        NUM_WHIR_SAMPLE_BAND_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_WHIR_SAMPLE_BAND_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominator = {
            let prep = builder.preprocessed();
            let local: &crate::whir_dt::columns::WhirSampleBandPreprocessedCols<AB::VarMaybeExt> =
                prep.borrow();
            self.bus.denominator(
                builder,
                local.query_bits.clone(),
                local.shift.clone(),
                local.high_max.clone(),
                local.high_bits.clone(),
            )
        };
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, _builder: &mut AB) {}

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        // Order matches precompute_lc: WhirSampleBand.
        builder.send(local[NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS].clone());
    }
}

impl MachineAir<F> for WhirSampleBandAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "WhirSampleBand".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(WhirSampleBandTraceGenerator::trace_height())
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Some(WhirSampleBandTraceGenerator::generate_preprocessed_trace())
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(WhirSampleBandTraceGenerator::trace_height())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirSampleBandTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirRoundAir {
    pub num_public_values: usize,
    pub round_bcast_bus: WhirRoundBcastBus,
    pub query_init_bus: WhirQueryInitBus,
    pub round_chain_bus: WhirRoundChainBus,
    pub final_root_chain_bus: WhirFinalRootChainBus,
    pub role_config: WhirRoleConfig,
    pub summary_bus: ProofShapeSummaryBus,
    pub opening_point_bus: BatchOpeningPointBus,
    pub height_group_bus: ProofShapeHeightGroupBus,
    pub group_claim_bus: WhirGroupClaimBus,
    pub commitment_root_bus: MerkleCommitmentRootBus,
    pub poseidon2_bus: Poseidon2PermuteBus,
    pub range_bus: RangeCheckerBus,
    pub transcript_event_bus: TranscriptEventBus,
}

impl Default for WhirRoundAir {
    fn default() -> Self {
        Self::new(whir_role_config(WHIR_ROLE_CORE), dt_stark::air::DT_PROOF_NUM_PV_ELTS)
    }
}

impl WhirRoundAir {
    pub fn new(role_config: WhirRoleConfig, num_public_values: usize) -> Self {
        Self {
            num_public_values,
            round_bcast_bus: WhirRoundBcastBus::new(),
            query_init_bus: WhirQueryInitBus::new(),
            round_chain_bus: WhirRoundChainBus::new(),
            final_root_chain_bus: WhirFinalRootChainBus::new(),
            role_config,
            summary_bus: ProofShapeSummaryBus::new(),
            opening_point_bus: BatchOpeningPointBus::new(),
            height_group_bus: ProofShapeHeightGroupBus::new(),
            group_claim_bus: WhirGroupClaimBus::new(),
            commitment_root_bus: MerkleCommitmentRootBus::new(),
            poseidon2_bus: Poseidon2PermuteBus::new(),
            range_bus: RangeCheckerBus::new(),
            transcript_event_bus: TranscriptEventBus::new(),
        }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirRoundAir {
    fn width(&self) -> usize {
        NUM_WHIR_ROUND_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirRoundAir {
    fn width(&self) -> usize {
        NUM_WHIR_ROUND_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.round_bcast_bus.required_max_beta_power_floor(),
            self.query_init_bus.required_max_beta_power_floor(),
            self.round_chain_bus.required_max_beta_power_floor(),
            self.final_root_chain_bus.required_max_beta_power_floor(),
            self.summary_bus.required_max_beta_power_floor(),
            self.opening_point_bus.required_max_beta_power_floor(),
            self.height_group_bus.required_max_beta_power_floor(),
            self.group_claim_bus.required_max_beta_power_floor(),
            self.commitment_root_bus.required_max_beta_power_floor(),
            self.poseidon2_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
            self.transcript_event_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_WHIR_ROUND_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &WhirRoundCols<AB::VarMaybeExt> = main.borrow();
        for denominator in whir_round_denominators(self, builder, local) {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirRoundCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_pow_batch.clone());
        assert_bool(builder, local.is_preamble.clone());
        assert_bool(builder, local.is_round.clone());
        assert_bool(builder, local.is_final.clone());
        assert_bool(builder, local.is_final_perm.clone());
        assert_bool(builder, local.is_merge.clone());
        assert_bool(builder, local.round_has_oracle.clone());
        assert_bool(builder, local.emit_prep_seed.clone());
        assert_bool(builder, local.chain_recv_pending_is_merge.clone());
        assert_bool(builder, local.chain_send_pending_is_merge.clone());
        for flag in &local.final_root_perm_step_flags {
            assert_bool(builder, flag.clone());
        }
        builder.assert_eq(
            local.is_pow_batch.clone() +
                local.is_preamble.clone() +
                local.is_round.clone() +
                local.is_final.clone() +
                local.is_final_perm.clone(),
            local.is_valid.clone(),
        );
        let role_num_queries = const_maybe::<AB>(self.role_config.num_queries);
        let role_log_blowup = const_maybe::<AB>(self.role_config.log_blowup);
        builder.assert_zero(
            local.round_has_oracle.clone() * (AB::one_maybe() - local.is_round.clone()),
        );
        builder.assert_zero(local.is_merge.clone() * (AB::one_maybe() - local.is_round.clone()));
        // No separate bcast/query_init/commitment_root mult=>flag constraints: the ==
        // pins below (mult == flag*Q with boolean flags) already imply them.
        builder.assert_zero(
            local.is_valid.clone() *
                (local.query_bits.clone() - local.r_rounds.clone() - role_log_blowup),
        );
        builder
            .assert_eq(local.bcast_mult.clone(), local.is_round.clone() * role_num_queries.clone());
        builder.assert_eq(
            local.query_init_mult.clone(),
            local.is_final.clone() * role_num_queries.clone(),
        );
        builder.assert_eq(
            local.commitment_root_send_mult.clone(),
            (local.is_preamble.clone() + local.round_has_oracle.clone()) * role_num_queries,
        );
        constrain_round_claim_chain(builder, local, self.role_config.log_blowup);
        constrain_round_pow_samples(builder, local);
        constrain_round_final_root_poseidon2(builder, local, self.role_config.log_blowup);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirRoundCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        // Order matches whir_round_denominators: internal WHIR buses, authenticated input seams,
        // commitment-root send, final-root Poseidon2/chain, PoW ranges, then W-events.
        builder.send(local.bcast_mult.clone());
        builder.send(local.query_init_mult.clone());
        let round_chain_mult = whir_round_chain_mult::<AB>(local);
        builder.recv(round_chain_mult.clone());
        builder.send(round_chain_mult);
        builder.recv(local.is_pow_batch.clone());
        builder.recv(local.is_round.clone());
        builder.recv(local.is_preamble.clone() + local.is_merge.clone());
        builder.recv(local.is_preamble.clone() + local.is_merge.clone());
        builder.send(local.commitment_root_send_mult.clone());
        builder.recv(local.final_root_poseidon2_recv_mult.clone());
        let final_root_chain_mult = whir_final_root_chain_mult::<AB>(local);
        builder.recv(final_root_chain_mult.clone());
        builder.send(final_root_chain_mult);
        let pow_sample_mult = whir_round_pow_sample_mult::<AB>(local);
        builder.recv(pow_sample_mult.clone());
        builder.recv(pow_sample_mult);
        for idx in 0..WHIR_ROUND_MAX_TRANSCRIPT_EVENTS {
            builder.recv(whir_round_event_mult::<AB>(local, idx));
        }
    }
}

impl MachineAir<F> for WhirRoundAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        // Legacy diagnostic name; wire identity comes from NativeAirId::wire_name.
        "WhirRound".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(WhirRoundTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirRoundTraceGenerator::generate_trace_compressed(input)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        for row in crate::whir_dt::trace::whir_round_rows(input) {
            let mult = row.final_root_poseidon2_recv_mult;
            if row.is_final_perm && mult != 0 {
                output.poseidon2.record_poseidon2_count(row.final_root_poseidon2_input, mult);
            }
        }
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirBatchEvalAir {
    pub batch_dim_bus: ProofShapeBatchDimBus,
    pub role_config: WhirRoleConfig,
    pub group_claim_bus: WhirGroupClaimBus,
    pub eval_chain_bus: WhirEvalChainBus,
    pub leaf_pow_seed_bus: WhirLeafPowSeedBus,
    pub opened_eval_bus: WhirOpenedEvalBus,
    pub transcript_event_bus: TranscriptEventBus,
    pub range_bus: RangeCheckerBus,
}

impl Default for WhirBatchEvalAir {
    fn default() -> Self {
        Self::new(whir_role_config(WHIR_ROLE_CORE))
    }
}

impl WhirBatchEvalAir {
    pub fn new(role_config: WhirRoleConfig) -> Self {
        Self {
            batch_dim_bus: ProofShapeBatchDimBus::new(),
            role_config,
            group_claim_bus: WhirGroupClaimBus::new(),
            eval_chain_bus: WhirEvalChainBus::new(),
            leaf_pow_seed_bus: WhirLeafPowSeedBus::new(),
            opened_eval_bus: WhirOpenedEvalBus::new(),
            transcript_event_bus: TranscriptEventBus::new(),
            range_bus: RangeCheckerBus::new(),
        }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirBatchEvalAir {
    fn width(&self) -> usize {
        NUM_WHIR_BATCH_EVAL_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirBatchEvalAir {
    fn width(&self) -> usize {
        NUM_WHIR_BATCH_EVAL_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.batch_dim_bus.required_max_beta_power_floor(),
            self.group_claim_bus.required_max_beta_power_floor(),
            self.eval_chain_bus.required_max_beta_power_floor(),
            self.leaf_pow_seed_bus.required_max_beta_power_floor(),
            self.opened_eval_bus.required_max_beta_power_floor(),
            self.transcript_event_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_WHIR_BATCH_EVAL_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &WhirBatchEvalCols<AB::VarMaybeExt> = main.borrow();
        for denominator in whir_batch_eval_denominators(self, builder, local) {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirBatchEvalCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_start.clone());
        assert_bool(builder, local.is_group_end.clone());
        assert_bool(builder, local.is_value.clone());
        assert_bool(builder, local.is_segment_start.clone());
        assert_bool(builder, local.is_segment_end.clone());
        assert_bool(builder, local.is_first_value.clone());
        assert_bool(builder, local.is_group_start.clone());
        assert_bool(builder, local.is_perm_batch.clone());
        assert_bool(builder, local.batch_dim_recv_mult.clone());
        assert_flag_implies(builder, local.is_start.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_group_end.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_value.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_segment_start.clone(), local.is_value.clone());
        assert_flag_implies(builder, local.is_segment_end.clone(), local.is_value.clone());
        assert_flag_implies(builder, local.is_first_value.clone(), local.is_segment_start.clone());
        // `is_first_value => is_value` and `opened_eval_send_mult => is_valid` are
        // implied transitively by the flag chain, so they are not asserted here.
        assert_flag_implies(builder, local.is_group_start.clone(), local.is_segment_start.clone());
        assert_flag_implies(builder, local.batch_dim_recv_mult.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.opened_eval_send_mult.clone(), local.is_value.clone());
        builder.assert_zero(local.batch_dim_recv_mult.clone() * local.value_idx.clone());
        assert_flag_implies(
            builder,
            local.is_segment_start.clone(),
            local.batch_dim_recv_mult.clone(),
        );
        assert_flag_implies(builder, local.is_group_end.clone(), local.is_segment_end.clone());
        // The pow-seed publication fires only on group-start rows; its count is
        // balance-forced to the number of deduped leaf group instances.
        builder.assert_zero(
            local.pow_seed_cnt.clone() * (AB::one_maybe() - local.is_group_start.clone()),
        );
        constrain_batch_eval_segment_order(builder, local);
        constrain_batch_eval_prefix_walk(builder, local);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirBatchEvalCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        // Order matches whir_batch_eval_denominators: BatchDim recv, EvalChain recv/send,
        // GroupClaim send, LeafPowSeed send, OpenedEval send, alpha event recvs.
        builder.recv(local.batch_dim_recv_mult.clone());
        builder.recv(local.is_start.clone() + local.is_value.clone());
        builder.send(local.is_start.clone() + local.is_value.clone());
        builder.send(local.is_group_end.clone());
        builder.send(local.pow_seed_cnt.clone());
        builder.send(local.opened_eval_send_mult.clone());
        for _ in 0..D_EF {
            builder.recv(local.is_start.clone());
        }
        builder.recv(local.is_group_start.clone());
    }
}

impl MachineAir<F> for WhirBatchEvalAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        // Legacy diagnostic name; wire identity comes from NativeAirId::wire_name.
        "WhirBatchEval".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(WhirBatchEvalTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirBatchEvalTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirQueryFoldAir {
    pub twiddle_bus: WhirTwiddlePowBus,
    pub sample_band_bus: WhirSampleBandBus,
    pub round_bcast_bus: WhirRoundBcastBus,
    pub query_leaf_sum_bus: WhirQueryLeafSumBus,
    pub query_chain_bus: WhirQueryChainBus,
    pub query_init_bus: WhirQueryInitBus,
    pub range_bus: RangeCheckerBus,
    pub merkle_leaf_block_bus: MerkleLeafBlockBus,
    pub transcript_event_bus: TranscriptEventBus,
}

impl Default for WhirQueryFoldAir {
    fn default() -> Self {
        Self {
            twiddle_bus: WhirTwiddlePowBus::new(),
            sample_band_bus: WhirSampleBandBus::new(),
            round_bcast_bus: WhirRoundBcastBus::new(),
            query_leaf_sum_bus: WhirQueryLeafSumBus::new(),
            query_chain_bus: WhirQueryChainBus::new(),
            query_init_bus: WhirQueryInitBus::new(),
            range_bus: RangeCheckerBus::new(),
            merkle_leaf_block_bus: MerkleLeafBlockBus::new(),
            transcript_event_bus: TranscriptEventBus::new(),
        }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirQueryFoldAir {
    fn width(&self) -> usize {
        NUM_WHIR_QUERY_FOLD_COLS
    }
}

fn whir_query_fold_reserved_main_indices() -> Vec<usize> {
    let mut indices = vec![
        core::mem::offset_of!(WhirQueryFoldCols<u8>, is_seed),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, is_round),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, cursor),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample_raw),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample_high),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample_shift),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample_high_max),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, query_sample_high_gap_inv),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, idx),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, idx_bit),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, idx_tail_bit0),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, idx_tail_bit1),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, x),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, acc),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, ipw),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_cursor),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_idx),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_idx_bit),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_x),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_acc),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, chain_send_ipw),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, is_merge),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, is_assign),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, merge_cursor_inv),
        core::mem::offset_of!(WhirQueryFoldCols<u8>, emit_prep_seed),
    ];
    let twiddle_bytes = core::mem::offset_of!(WhirQueryFoldCols<u8>, twiddle_bytes);
    indices.extend((0..WHIR_TWIDDLE_TABLES).map(|idx| twiddle_bytes + idx));
    let twiddle_values = core::mem::offset_of!(WhirQueryFoldCols<u8>, twiddle_values);
    indices.extend((0..WHIR_TWIDDLE_TABLES).map(|idx| twiddle_values + idx));
    indices.push(core::mem::offset_of!(WhirQueryFoldCols<u8>, twiddle_product_01));
    indices
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirQueryFoldAir {
    fn width(&self) -> usize {
        NUM_WHIR_QUERY_FOLD_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.twiddle_bus.required_max_beta_power_floor(),
            self.sample_band_bus.required_max_beta_power_floor(),
            self.round_bcast_bus.required_max_beta_power_floor(),
            self.query_leaf_sum_bus.required_max_beta_power_floor(),
            self.query_chain_bus.required_max_beta_power_floor(),
            self.query_init_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
            self.merkle_leaf_block_bus.required_max_beta_power_floor(),
            self.transcript_event_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        whir_query_fold_reserved_main_indices().into_iter().map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let values = {
            let main = builder.main();
            let local: &WhirQueryFoldCols<AB::VarMaybeExt> = main.borrow();
            let mut values = whir_query_fold_denominators(self, builder, local);
            values.extend([
                AB::pack_ext_limbs(&local.folded),
                AB::pack_ext_limbs(&local.f0),
                AB::pack_ext_limbs(&local.f1),
                AB::pack_ext_limbs(&local.chain_send_folded),
                AB::pack_ext_limbs(&local.r_fold),
                AB::pack_ext_limbs(&local.merge_beta),
                AB::pack_ext_limbs(&local.merge_eq),
                AB::pack_ext_limbs(&local.cfr),
                AB::pack_ext_limbs(&local.leaf_sum),
            ]);
            values
        };
        debug_assert_eq!(
            values.len(),
            NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS + NUM_WHIR_QUERY_FOLD_PACKED_COLS
        );
        for value in values {
            builder.retain_precomputed(value);
        }
    }

    fn eval(&self, builder: &mut AB) {
        eval_query_fold_constraints(builder, true);
    }

    fn lookup(&self, builder: &mut AB) {
        lookup_query_fold(builder, false);
    }
}

fn eval_query_fold_constraints<AB: FullAirBuilder>(
    builder: &mut AB,
    enforce_exact_assignment: bool,
) {
    let reserved = builder.reserved_poly();
    let local_binding = reserved.row_slice(0);
    let local: &WhirQueryFoldReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
    let precomputed = builder.precomputed();
    let precomputed_binding = precomputed.row_slice(0);
    let packed: &WhirQueryFoldPackedCols<AB::VarExt> = precomputed_binding.deref()
        [NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS..
            NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS + NUM_WHIR_QUERY_FOLD_PACKED_COLS]
        .borrow();
    assert_bool(builder, local.is_seed.clone());
    assert_bool(builder, local.is_round.clone());
    assert_bool(builder, local.is_seed.clone() + local.is_round.clone());
    assert_bool(builder, local.is_merge.clone());
    assert_bool(builder, local.is_assign.clone());
    assert_bool(builder, local.emit_prep_seed.clone());
    assert_bool(builder, local.idx_bit.clone());
    assert_bool(builder, local.idx_tail_bit0.clone());
    assert_bool(builder, local.idx_tail_bit1.clone());
    assert_bool(builder, local.chain_send_idx_bit.clone());
    builder.assert_zero(local.is_merge.clone() * (AB::one_maybe() - local.is_round.clone()));
    assert_flag_implies(builder, local.is_assign.clone(), local.is_merge.clone());
    assert_flag_implies(builder, local.emit_prep_seed.clone(), local.is_round.clone());
    builder.assert_zero(local.is_assign.clone() * local.cursor.clone());
    if enforce_exact_assignment {
        builder.assert_zero(
            local.cursor.clone() * local.merge_cursor_inv.clone() - local.is_merge.clone() +
                local.is_assign.clone(),
        );
    }
    constrain_query_chain_scaffold(builder, local, packed);
    constrain_query_twiddle_seed(builder, local);
    constrain_query_fold_arithmetic(builder, local, packed);
}

fn lookup_query_fold<AB: FullAirBuilder>(builder: &mut AB, mirror: bool) {
    let reserved = builder.reserved_poly();
    let local_binding = reserved.row_slice(0);
    let local: &WhirQueryFoldReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
    let is_valid = local.is_seed.clone() + local.is_round.clone();
    let recv = |builder: &mut AB, mult| {
        if mirror {
            builder.send(mult);
        } else {
            builder.recv(mult);
        }
    };
    let send = |builder: &mut AB, mult| {
        if mirror {
            builder.recv(mult);
        } else {
            builder.send(mult);
        }
    };
    for _ in 0..WHIR_TWIDDLE_TABLES {
        recv(builder, local.is_seed.clone());
    }
    recv(builder, local.is_seed.clone());
    recv(builder, local.is_round.clone());
    recv(builder, local.is_merge.clone());
    recv(builder, is_valid.clone());
    send(builder, is_valid);
    recv(builder, local.is_seed.clone());
    recv(builder, local.is_seed.clone());
    recv(builder, local.is_seed.clone());
    for _ in 0..WHIR_QUERY_PAIR_LEAF_BLOCKS {
        send(builder, local.is_round.clone());
    }
    recv(builder, local.is_seed.clone());
}

#[cfg(test)]
pub(crate) fn eval_query_fold_historical<AB: FullAirBuilder>(builder: &mut AB) {
    eval_query_fold_constraints(builder, false);
}

#[cfg(test)]
pub(crate) fn lookup_query_fold_mirror<AB: FullAirBuilder>(builder: &mut AB) {
    lookup_query_fold(builder, true);
}

impl MachineAir<F> for WhirQueryFoldAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "WhirQueryFold".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(WhirQueryFoldTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirQueryFoldTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirLeafStreamAir {
    pub leaf_pow_seed_bus: WhirLeafPowSeedBus,
    pub leaf_chain_bus: WhirLeafChainBus,
    pub merkle_leaf_block_bus: MerkleLeafBlockBus,
    pub query_leaf_sum_bus: WhirQueryLeafSumBus,
    pub range_bus: RangeCheckerBus,
}

impl Default for WhirLeafStreamAir {
    fn default() -> Self {
        Self {
            leaf_pow_seed_bus: WhirLeafPowSeedBus::new(),
            leaf_chain_bus: WhirLeafChainBus::new(),
            merkle_leaf_block_bus: MerkleLeafBlockBus::new(),
            query_leaf_sum_bus: WhirQueryLeafSumBus::new(),
            range_bus: RangeCheckerBus::new(),
        }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirLeafStreamAir {
    fn width(&self) -> usize {
        NUM_WHIR_LEAF_STREAM_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirLeafStreamAir {
    fn width(&self) -> usize {
        NUM_WHIR_LEAF_STREAM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.leaf_pow_seed_bus.required_max_beta_power_floor(),
            self.leaf_chain_bus.required_max_beta_power_floor(),
            self.merkle_leaf_block_bus.required_max_beta_power_floor(),
            self.query_leaf_sum_bus.required_max_beta_power_floor(),
            self.range_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_WHIR_LEAF_STREAM_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &WhirLeafStreamCols<AB::VarMaybeExt> = main.borrow();
        for denominator in whir_leaf_stream_denominators(self, builder, local) {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirLeafStreamCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        assert_bool(builder, local.is_unit_start.clone());
        assert_bool(builder, local.is_unit_end.clone());
        assert_bool(builder, local.is_unit_key_start.clone());
        for bit in &local.chunk_mask {
            assert_bool(builder, bit.clone());
        }
        // `is_unit_start => is_valid` is implied transitively via
        // is_unit_start => is_unit_key_start => is_valid, so it is not asserted here.
        assert_flag_implies(builder, local.is_unit_end.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_unit_key_start.clone(), local.is_valid.clone());
        assert_flag_implies(builder, local.is_unit_start.clone(), local.is_unit_key_start.clone());
        // The 1025 publication fires only at the group end; its value is
        // balance-forced to the consuming-merge count.
        builder
            .assert_zero(local.serve_cnt.clone() * (AB::one_maybe() - local.is_unit_end.clone()));
        assert_prefix_mask(builder, &local.chunk_mask);
        builder
            .assert_zero(local.chunk_mask[0].clone() * (AB::one_maybe() - local.is_valid.clone()));
        constrain_leaf_base_rlc(builder, local);
        constrain_leaf_base_unit_key_order(builder, local);
        builder.assert_zero(
            local.chunk_mask[0].clone() *
                (local.unit_key.clone() -
                    const_maybe::<AB>(WHIR_UNIT_KEY_SLOT_STRIDE) * local.batch_id.clone() -
                    local.log_height.clone()),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirLeafStreamCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        // Order matches whir_leaf_stream_denominators: LeafPowSeed recv,
        // LeafChain recv/send (boundary-gated), MerkleLeafBlock send, QueryLeafSum
        // send (count), unit-key gap range recv.
        builder.recv(local.is_unit_start.clone());
        builder.recv(local.is_valid.clone() - local.is_unit_start.clone());
        builder.send(local.is_valid.clone() - local.is_unit_end.clone());
        builder.send(local.chunk_mask[0].clone());
        builder.send(local.serve_cnt.clone());
        builder.recv(local.is_unit_key_start.clone() - local.is_unit_start.clone());
    }
}

impl MachineAir<F> for WhirLeafStreamAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "WhirLeafStream".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(WhirLeafStreamTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirLeafStreamTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WhirLeafExtStreamAir {
    pub leaf_chain_bus: WhirLeafChainBus,
    pub merkle_leaf_block_bus: MerkleLeafBlockBus,
    pub query_leaf_sum_bus: WhirQueryLeafSumBus,
}

impl Default for WhirLeafExtStreamAir {
    fn default() -> Self {
        Self {
            leaf_chain_bus: WhirLeafChainBus::new(),
            merkle_leaf_block_bus: MerkleLeafBlockBus::new(),
            query_leaf_sum_bus: WhirQueryLeafSumBus::new(),
        }
    }
}

impl<Fld: Field> BaseAir<Fld> for WhirLeafExtStreamAir {
    fn width(&self) -> usize {
        NUM_WHIR_LEAF_EXT_STREAM_COLS
    }
}

fn whir_leaf_ext_main_indices<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|index| start + index)
}

fn whir_leaf_ext_reserved_main_indices() -> WhirLeafExtStreamReservedCols<usize> {
    WhirLeafExtStreamReservedCols {
        is_unit_end: core::mem::offset_of!(WhirLeafExtStreamCols<u8>, is_unit_end),
        serve_cnt: core::mem::offset_of!(WhirLeafExtStreamCols<u8>, serve_cnt),
        is_unit_key_start: core::mem::offset_of!(WhirLeafExtStreamCols<u8>, is_unit_key_start),
        element_masks: whir_leaf_ext_main_indices(core::mem::offset_of!(
            WhirLeafExtStreamCols<u8>,
            element_masks
        )),
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for WhirLeafExtStreamAir {
    fn width(&self) -> usize {
        NUM_WHIR_LEAF_EXT_STREAM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.leaf_chain_bus.required_max_beta_power_floor(),
            self.merkle_leaf_block_bus.required_max_beta_power_floor(),
            self.query_leaf_sum_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        whir_leaf_ext_reserved_main_indices()
            .as_slice()
            .iter()
            .copied()
            .map(PairCol::Main)
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let precomputed = {
            let main = builder.main();
            let local: &WhirLeafExtStreamCols<AB::VarMaybeExt> = main.borrow();
            whir_leaf_ext_stream_precomputed(self, builder, local)
        };
        for value in precomputed.as_slice() {
            builder.retain_precomputed(value.clone());
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirLeafExtStreamReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let precomputed = builder.precomputed();
        let precomputed_binding = precomputed.row_slice(0);
        let precomputed: &WhirLeafExtStreamPrecomputedCols<AB::VarExt> =
            precomputed_binding.deref().borrow();

        let is_valid = local.element_masks[0].clone();
        assert_bool(builder, is_valid.clone());
        for elem_idx in 1..WHIR_LEAF_RLC_SLOTS {
            builder.assert_zero(
                local.element_masks[elem_idx].clone() *
                    (local.element_masks[elem_idx].clone() -
                        local.element_masks[elem_idx - 1].clone()),
            );
        }
        builder.assert_zero(
            local.is_unit_end.clone() * (local.is_unit_end.clone() - is_valid.clone()),
        );
        builder.assert_zero(
            local.is_unit_key_start.clone() * (local.is_unit_key_start.clone() - is_valid),
        );
        builder
            .assert_zero(local.serve_cnt.clone() * (AB::one_maybe() - local.is_unit_end.clone()));
        constrain_leaf_ext_rlc(builder, local, &precomputed.packed);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &WhirLeafExtStreamReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let is_valid = local.element_masks[0].clone();
        builder.recv(is_valid.clone());
        builder.send(is_valid - local.is_unit_end.clone());
        for block in 0..WHIR_LEAF_BLOCKS_PER_ROW {
            builder.send(local.element_masks[block * WHIR_LEAF_BASE_LIMBS_PER_ROW / D_EF].clone());
        }
        builder.send(local.serve_cnt.clone());
    }
}

impl MachineAir<F> for WhirLeafExtStreamAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "WhirLeafExtStream".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(WhirLeafExtStreamTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WhirLeafExtStreamTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn whir_round_denominators<AB: FullAirBuilder>(
    air: &WhirRoundAir,
    builder: &AB,
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let mut denominators = Vec::with_capacity(14 + WHIR_ROUND_MAX_TRANSCRIPT_EVENTS);
    denominators.push(air.round_bcast_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.round.clone(),
        local.r_fold[0].clone(),
        local.r_fold[1].clone(),
        local.r_fold[2].clone(),
        local.r_fold[3].clone(),
        local.r_fold[4].clone(),
        local.chain_recv_pending_is_merge.clone(),
        local.chain_recv_pending_beta[0].clone(),
        local.chain_recv_pending_beta[1].clone(),
        local.chain_recv_pending_beta[2].clone(),
        local.chain_recv_pending_beta[3].clone(),
        local.chain_recv_pending_beta[4].clone(),
        local.chain_recv_pending_eq[0].clone(),
        local.chain_recv_pending_eq[1].clone(),
        local.chain_recv_pending_eq[2].clone(),
        local.chain_recv_pending_eq[3].clone(),
        local.chain_recv_pending_eq[4].clone(),
        local.emit_prep_seed.clone(),
        local.merge_log_height.clone(),
    ));
    denominators.push(air.query_init_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.w_qbase.clone(),
        local.query_bits.clone(),
        local.r_rounds.clone(),
        local.cfr[0].clone(),
        local.cfr[1].clone(),
        local.cfr[2].clone(),
        local.cfr[3].clone(),
        local.cfr[4].clone(),
    ));
    denominators.push(air.round_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.chain_recv_round.clone(),
        local.chain_recv_tidx.clone(),
        local.chain_recv_claim[0].clone(),
        local.chain_recv_claim[1].clone(),
        local.chain_recv_claim[2].clone(),
        local.chain_recv_claim[3].clone(),
        local.chain_recv_claim[4].clone(),
        local.chain_recv_eq[0].clone(),
        local.chain_recv_eq[1].clone(),
        local.chain_recv_eq[2].clone(),
        local.chain_recv_eq[3].clone(),
        local.chain_recv_eq[4].clone(),
        local.chain_recv_pending_is_merge.clone(),
        local.chain_recv_pending_beta[0].clone(),
        local.chain_recv_pending_beta[1].clone(),
        local.chain_recv_pending_beta[2].clone(),
        local.chain_recv_pending_beta[3].clone(),
        local.chain_recv_pending_beta[4].clone(),
        local.chain_recv_pending_eq[0].clone(),
        local.chain_recv_pending_eq[1].clone(),
        local.chain_recv_pending_eq[2].clone(),
        local.chain_recv_pending_eq[3].clone(),
        local.chain_recv_pending_eq[4].clone(),
    ));
    denominators.push(air.round_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.chain_send_round.clone(),
        local.chain_send_tidx.clone(),
        local.chain_send_claim[0].clone(),
        local.chain_send_claim[1].clone(),
        local.chain_send_claim[2].clone(),
        local.chain_send_claim[3].clone(),
        local.chain_send_claim[4].clone(),
        local.chain_send_eq[0].clone(),
        local.chain_send_eq[1].clone(),
        local.chain_send_eq[2].clone(),
        local.chain_send_eq[3].clone(),
        local.chain_send_eq[4].clone(),
        local.chain_send_pending_is_merge.clone(),
        local.chain_send_pending_beta[0].clone(),
        local.chain_send_pending_beta[1].clone(),
        local.chain_send_pending_beta[2].clone(),
        local.chain_send_pending_beta[3].clone(),
        local.chain_send_pending_beta[4].clone(),
        local.chain_send_pending_eq[0].clone(),
        local.chain_send_pending_eq[1].clone(),
        local.chain_send_pending_eq[2].clone(),
        local.chain_send_pending_eq[3].clone(),
        local.chain_send_pending_eq[4].clone(),
    ));
    denominators.push(air.summary_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.r_rounds.clone(),
        local.c_chips.clone(),
        const_maybe::<AB>(air.num_public_values),
        local.summary_id_base.clone(),
    ));
    denominators.push(air.opening_point_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.opening_idx.clone(),
        local.opening_point.clone(),
    ));
    denominators.push(air.height_group_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.height_group_rank.clone(),
        local.height_group_log_height.clone(),
    ));
    denominators.push(air.group_claim_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.group_claim_log_height.clone(),
        local.group_claim[0].clone(),
        local.group_claim[1].clone(),
        local.group_claim[2].clone(),
        local.group_claim[3].clone(),
        local.group_claim[4].clone(),
    ));
    denominators.push(air.commitment_root_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.commit_id.clone(),
        local.commit_root.clone(),
    ));
    denominators.push(air.poseidon2_bus.denominator(
        builder,
        final_root_recv_state::<AB>(local),
        final_root_poseidon2_output::<AB>(local),
    ));
    denominators.push(air.final_root_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.opening_idx.clone(),
        final_root_recv_state_lane::<AB>(local, 0),
        final_root_recv_state_lane::<AB>(local, 1),
        final_root_recv_state_lane::<AB>(local, 2),
        final_root_recv_state_lane::<AB>(local, 3),
        final_root_recv_state_lane::<AB>(local, 4),
        final_root_recv_state_lane::<AB>(local, 5),
        final_root_recv_state_lane::<AB>(local, 6),
        final_root_recv_state_lane::<AB>(local, 7),
        final_root_recv_state_lane::<AB>(local, 8),
        final_root_recv_state_lane::<AB>(local, 9),
        final_root_recv_state_lane::<AB>(local, 10),
        final_root_recv_state_lane::<AB>(local, 11),
        final_root_recv_state_lane::<AB>(local, 12),
        final_root_recv_state_lane::<AB>(local, 13),
        final_root_recv_state_lane::<AB>(local, 14),
        final_root_recv_state_lane::<AB>(local, 15),
    ));
    denominators.push(air.final_root_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.height_group_rank.clone(),
        final_root_send_state_lane::<AB>(local, 0),
        final_root_send_state_lane::<AB>(local, 1),
        final_root_send_state_lane::<AB>(local, 2),
        final_root_send_state_lane::<AB>(local, 3),
        final_root_send_state_lane::<AB>(local, 4),
        final_root_send_state_lane::<AB>(local, 5),
        final_root_send_state_lane::<AB>(local, 6),
        final_root_send_state_lane::<AB>(local, 7),
        final_root_send_state_lane::<AB>(local, 8),
        final_root_send_state_lane::<AB>(local, 9),
        final_root_send_state_lane::<AB>(local, 10),
        final_root_send_state_lane::<AB>(local, 11),
        final_root_send_state_lane::<AB>(local, 12),
        final_root_send_state_lane::<AB>(local, 13),
        final_root_send_state_lane::<AB>(local, 14),
        final_root_send_state_lane::<AB>(local, 15),
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.pow_sample_high.clone(),
            max_bits: whir_round_pow_range_bits::<AB>(local),
        },
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: whir_round_pow_high_max::<AB>(local) - local.pow_sample_high.clone(),
            max_bits: whir_round_pow_range_bits::<AB>(local),
        },
    ));
    for i in 0..WHIR_ROUND_MAX_TRANSCRIPT_EVENTS {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            local.proof_idx.clone(),
            whir_round_event_tidx::<AB>(local, i),
            whir_round_event_is_sample::<AB>(local, i),
            local.event_value[i].clone(),
        ));
    }
    denominators
}

fn whir_batch_eval_denominators<AB: FullAirBuilder>(
    air: &WhirBatchEvalAir,
    builder: &AB,
    local: &WhirBatchEvalCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let mut denominators = vec![
        air.batch_dim_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.batch_id.clone(),
            local.batch_pos.clone(),
            local.chip_idx.clone(),
            local.static_chip_id.clone(),
            local.width.clone(),
            local.log_height.clone(),
        ),
        air.eval_chain_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.chain_recv_cursor.clone(),
            local.chain_recv_log_height.clone(),
            local.chain_recv_batch_id.clone(),
            local.chain_recv_batch_pos.clone(),
            local.chain_recv_value_idx.clone(),
            local.chain_recv_segment_element_count.clone(),
            local.alpha[0].clone(),
            local.alpha[1].clone(),
            local.alpha[2].clone(),
            local.alpha[3].clone(),
            local.alpha[4].clone(),
            local.pow_in[0].clone(),
            local.pow_in[1].clone(),
            local.pow_in[2].clone(),
            local.pow_in[3].clone(),
            local.pow_in[4].clone(),
            local.acc_in[0].clone(),
            local.acc_in[1].clone(),
            local.acc_in[2].clone(),
            local.acc_in[3].clone(),
            local.acc_in[4].clone(),
            local.group_base_in[0].clone(),
            local.group_base_in[1].clone(),
            local.group_base_in[2].clone(),
            local.group_base_in[3].clone(),
            local.group_base_in[4].clone(),
        ),
        air.eval_chain_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.chain_send_cursor.clone(),
            local.log_height.clone(),
            local.batch_id.clone(),
            local.batch_pos.clone(),
            local.value_idx.clone(),
            local.segment_element_count.clone(),
            local.alpha[0].clone(),
            local.alpha[1].clone(),
            local.alpha[2].clone(),
            local.alpha[3].clone(),
            local.alpha[4].clone(),
            local.pow_out[0].clone(),
            local.pow_out[1].clone(),
            local.pow_out[2].clone(),
            local.pow_out[3].clone(),
            local.pow_out[4].clone(),
            local.acc_out[0].clone(),
            local.acc_out[1].clone(),
            local.acc_out[2].clone(),
            local.acc_out[3].clone(),
            local.acc_out[4].clone(),
            local.group_base_out[0].clone(),
            local.group_base_out[1].clone(),
            local.group_base_out[2].clone(),
            local.group_base_out[3].clone(),
            local.group_base_out[4].clone(),
        ),
        air.group_claim_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.log_height.clone(),
            local.acc_out[0].clone() - local.group_base_in[0].clone(),
            local.acc_out[1].clone() - local.group_base_in[1].clone(),
            local.acc_out[2].clone() - local.group_base_in[2].clone(),
            local.acc_out[3].clone() - local.group_base_in[3].clone(),
            local.acc_out[4].clone() - local.group_base_in[4].clone(),
        ),
        air.leaf_pow_seed_bus.denominator(
            builder,
            local.proof_idx.clone(),
            // BatchEval walks TRACE heights; the bus key is the codeword height
            // (trace + baked blowup), matching the leaf streams' log_height column.
            local.log_height.clone() + const_maybe::<AB>(air.role_config.log_blowup),
            local.alpha[0].clone(),
            local.alpha[1].clone(),
            local.alpha[2].clone(),
            local.alpha[3].clone(),
            local.alpha[4].clone(),
            local.pow_in[0].clone(),
            local.pow_in[1].clone(),
            local.pow_in[2].clone(),
            local.pow_in[3].clone(),
            local.pow_in[4].clone(),
        ),
        air.opened_eval_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.batch_id.clone(),
            local.batch_pos.clone(),
            local.chip_idx.clone(),
            local.value_idx.clone(),
            local.value[0].clone(),
            local.value[1].clone(),
            local.value[2].clone(),
            local.value[3].clone(),
            local.value[4].clone(),
        ),
    ];
    for idx in 0..D_EF {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            local.proof_idx.clone(),
            local.alpha_tidx.clone() + const_maybe::<AB>(idx),
            AB::one_maybe(),
            local.alpha[idx].clone(),
        ));
    }
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.group_log_height_gap.clone(),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators
}

fn whir_query_fold_denominators<AB: FullAirBuilder>(
    air: &WhirQueryFoldAir,
    builder: &AB,
    local: &WhirQueryFoldCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let merge_log_height = local.query_bits.clone() - local.cursor.clone();
    let mut denominators =
        Vec::with_capacity(11 + WHIR_TWIDDLE_TABLES + WHIR_QUERY_PAIR_LEAF_BLOCKS);
    for table_id in 0..WHIR_TWIDDLE_TABLES {
        denominators.push(air.twiddle_bus.denominator(
            builder,
            const_maybe::<AB>(table_id),
            local.twiddle_bytes[table_id].clone(),
            local.twiddle_values[table_id].clone(),
        ));
    }
    denominators.push(air.sample_band_bus.denominator(
        builder,
        local.query_bits.clone(),
        local.query_sample_shift.clone(),
        local.query_sample_high_max.clone(),
        local.query_sample_high_bits.clone(),
    ));
    denominators.push(air.round_bcast_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.cursor.clone(),
        local.r_fold[0].clone(),
        local.r_fold[1].clone(),
        local.r_fold[2].clone(),
        local.r_fold[3].clone(),
        local.r_fold[4].clone(),
        local.is_merge.clone(),
        local.merge_beta[0].clone(),
        local.merge_beta[1].clone(),
        local.merge_beta[2].clone(),
        local.merge_beta[3].clone(),
        local.merge_beta[4].clone(),
        local.merge_eq[0].clone(),
        local.merge_eq[1].clone(),
        local.merge_eq[2].clone(),
        local.merge_eq[3].clone(),
        local.merge_eq[4].clone(),
        local.emit_prep_seed.clone(),
        merge_log_height.clone(),
    ));
    // The leaf sum is keyed by the fold-bound truncated index on this row
    // (idx = query_sample >> cursor = the merkle leaf index at merge_log_height).
    denominators.push(air.query_leaf_sum_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.idx.clone(),
        merge_log_height,
        local.leaf_sum[0].clone(),
        local.leaf_sum[1].clone(),
        local.leaf_sum[2].clone(),
        local.leaf_sum[3].clone(),
        local.leaf_sum[4].clone(),
    ));
    denominators.push(air.query_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.query_idx.clone(),
        local.cursor.clone(),
        local.query_bits.clone(),
        local.r_rounds.clone(),
        local.idx.clone(),
        local.idx_bit.clone(),
        local.x.clone(),
        local.acc.clone(),
        local.ipw.clone(),
        local.folded[0].clone(),
        local.folded[1].clone(),
        local.folded[2].clone(),
        local.folded[3].clone(),
        local.folded[4].clone(),
    ));
    denominators.push(air.query_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.query_idx.clone(),
        local.chain_send_cursor.clone(),
        local.query_bits.clone(),
        local.r_rounds.clone(),
        local.chain_send_idx.clone(),
        local.chain_send_idx_bit.clone(),
        local.chain_send_x.clone(),
        local.chain_send_acc.clone(),
        local.chain_send_ipw.clone(),
        local.chain_send_folded[0].clone(),
        local.chain_send_folded[1].clone(),
        local.chain_send_folded[2].clone(),
        local.chain_send_folded[3].clone(),
        local.chain_send_folded[4].clone(),
    ));
    denominators.push(air.query_init_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.w_qbase.clone(),
        local.query_bits.clone(),
        local.r_rounds.clone(),
        local.cfr[0].clone(),
        local.cfr[1].clone(),
        local.cfr[2].clone(),
        local.cfr[3].clone(),
        local.cfr[4].clone(),
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.query_sample_high.clone(),
            max_bits: const_maybe::<AB>(WHIR_PAIRED_RANGE_BITS),
        },
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.query_sample_high_max.clone() - local.query_sample_high.clone(),
            max_bits: const_maybe::<AB>(WHIR_PAIRED_RANGE_BITS),
        },
    ));
    // Pair-leaf identity: unit_key = (3 + cursor)*32 + (query_bits - 1 - cursor)
    // = query_bits + 31*cursor + 95 (affine). Pair position = chain_send_idx
    // (= idx >> 1); blocks 0/1.
    // Note: these constants must stay in sync with the record-side unit_key expression.
    let pair_unit_key = local.query_bits.clone() +
        local.cursor.clone() * const_maybe::<AB>(31) +
        const_maybe::<AB>(95);
    for block in 0..WHIR_QUERY_PAIR_LEAF_BLOCKS {
        denominators.push(air.merkle_leaf_block_bus.denominator(
            builder,
            local.proof_idx.clone(),
            const_maybe::<AB>(100) + local.cursor.clone(),
            pair_unit_key.clone(),
            local.chain_send_idx.clone(),
            const_maybe::<AB>(block),
            query_pair_leaf_mask::<AB>(block),
            query_pair_leaf_chunk::<AB>(local, block),
        ));
    }
    denominators.push(air.transcript_event_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.w_qbase.clone() + local.query_idx.clone(),
        AB::one_maybe(),
        local.query_sample_raw.clone(),
    ));
    denominators
}

fn constrain_round_claim_chain<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirRoundCols<AB::VarMaybeExt>,
    log_blowup: usize,
) {
    let non_pow = AB::one_maybe() - local.is_pow_batch.clone();
    let non_pow_stride = local.is_preamble.clone() * const_maybe::<AB>(8) +
        local.is_round.clone() *
            (const_maybe::<AB>(20) +
                local.round_has_oracle.clone() * const_maybe::<AB>(8) +
                local.is_merge.clone() * const_maybe::<AB>(5)) +
        local.is_final.clone() * const_maybe::<AB>(11);
    builder.assert_zero(non_pow.clone() * (local.tidx.clone() - local.chain_recv_tidx.clone()));
    builder.assert_eq(
        local.chain_send_tidx.clone(),
        local.is_pow_batch.clone() * (local.tidx.clone() + const_maybe::<AB>(3)) +
            non_pow.clone() * (local.chain_recv_tidx.clone() + non_pow_stride),
    );
    builder.assert_eq(
        local.chain_send_round.clone(),
        non_pow.clone() * (local.chain_recv_round.clone() + local.is_round.clone()),
    );
    builder.assert_zero(
        local.is_round.clone() * (local.round.clone() - local.chain_recv_round.clone()),
    );
    builder.assert_zero(
        local.is_round.clone() *
            (local.opening_idx.clone() + local.round.clone() + AB::one_maybe() -
                local.r_rounds.clone()),
    );
    builder.assert_zero(
        local.is_round.clone() *
            (local.merge_log_height.clone() + local.round.clone() -
                local.r_rounds.clone() -
                const_maybe::<AB>(log_blowup)),
    );
    builder.assert_zero(
        local.is_preamble.clone() * (local.commit_id.clone() - const_maybe::<AB>(100)),
    );
    builder.assert_zero(
        local.is_round.clone() *
            local.round_has_oracle.clone() *
            (local.commit_id.clone() - const_maybe::<AB>(100) - local.round.clone()),
    );
    builder.assert_zero(
        local.is_final.clone() * (local.chain_recv_round.clone() - local.r_rounds.clone()),
    );
    builder.assert_zero(local.is_preamble.clone() * local.chain_recv_round.clone());
    builder.assert_zero(
        local.is_final.clone() * (local.w_qbase.clone() - local.chain_send_tidx.clone()),
    );

    builder.assert_zero(
        local.is_preamble.clone() *
            (local.height_group_log_height.clone() - local.r_rounds.clone()),
    );
    builder.assert_zero(
        local.is_merge.clone() *
            (local.height_group_log_height.clone() + local.round.clone() + AB::one_maybe() -
                local.r_rounds.clone()),
    );
    builder.assert_zero(
        (local.is_preamble.clone() + local.is_merge.clone()) *
            (local.group_claim_log_height.clone() - local.height_group_log_height.clone()),
    );

    for limb in 0..D_EF {
        builder.assert_zero(
            local.is_round.clone() * (local.r_fold[limb].clone() - round_sample::<AB>(local, limb)),
        );
        builder.assert_zero(
            local.is_round.clone() *
                (round_coeff::<AB>(local, 0, limb) * const_maybe::<AB>(2) +
                    round_coeff::<AB>(local, 1, limb) +
                    round_coeff::<AB>(local, 2, limb) -
                    local.chain_recv_claim[limb].clone()),
        );
    }

    let r = ChallengeExtension(local.r_fold.clone());
    let c0 = round_coeffs::<AB>(local, 0);
    let c1 = round_coeffs::<AB>(local, 1);
    let c2 = round_coeffs::<AB>(local, 2);
    let claim_acc_expected = ChallengeExtension(c1) + r.clone() * ChallengeExtension(c2);
    let claim_folded_expected =
        ChallengeExtension(c0) + r.clone() * ChallengeExtension(local.claim_acc.clone());

    let one_ext = ChallengeExtension(one_ext::<AB>());
    let z = ChallengeExtension(local.opening_point.clone());
    let eq_factor_expected =
        z.clone() * r.clone() + (one_ext.clone() - z) * (one_ext.clone() - r.clone());
    let eq_folded_expected = ext_mul::<AB>(&local.chain_recv_eq, &local.eq_factor);

    for limb in 0..D_EF {
        builder.assert_zero(
            local.is_round.clone() *
                (local.claim_acc[limb].clone() - claim_acc_expected.0[limb].clone()),
        );
        builder.assert_zero(
            local.is_round.clone() *
                (local.claim_folded[limb].clone() - claim_folded_expected.0[limb].clone()),
        );
        builder.assert_zero(
            local.is_round.clone() *
                (local.eq_factor[limb].clone() - eq_factor_expected.0[limb].clone()),
        );
        builder.assert_zero(
            local.is_round.clone() *
                (local.eq_folded[limb].clone() - eq_folded_expected[limb].clone()),
        );
    }

    let merge_delta = ext_mul::<AB>(&round_merge_beta::<AB>(local), &local.group_claim);
    let cfr_times_eq = ext_mul::<AB>(&local.cfr, &local.chain_recv_eq);
    let passthrough = AB::one_maybe() -
        local.is_pow_batch.clone() -
        local.is_preamble.clone() -
        local.is_round.clone();
    for limb in 0..D_EF {
        let expected_claim = passthrough.clone() * local.chain_recv_claim[limb].clone() +
            local.is_preamble.clone() * local.group_claim[limb].clone() +
            local.is_round.clone() * local.claim_folded[limb].clone() +
            local.is_merge.clone() * merge_delta[limb].clone();
        builder.assert_eq(local.chain_send_claim[limb].clone(), expected_claim);

        let expected_eq = passthrough.clone() * local.chain_recv_eq[limb].clone() +
            local.is_preamble.clone() * one_ext_limb::<AB>(limb) +
            local.is_round.clone() * local.eq_folded[limb].clone() +
            local.is_merge.clone() * (one_ext_limb::<AB>(limb) - local.eq_folded[limb].clone());
        builder.assert_eq(local.chain_send_eq[limb].clone(), expected_eq);

        builder.assert_eq(
            local.chain_send_pending_beta[limb].clone(),
            local.is_merge.clone() * round_merge_sample::<AB>(local, limb),
        );
        builder.assert_eq(
            local.chain_send_pending_eq[limb].clone(),
            local.is_merge.clone() * local.eq_folded[limb].clone(),
        );
        builder.assert_zero(
            local.is_final.clone() *
                (cfr_times_eq[limb].clone() - local.chain_recv_claim[limb].clone()),
        );
    }
    builder.assert_eq(
        local.chain_send_pending_is_merge.clone(),
        local.is_preamble.clone() + local.is_merge.clone(),
    );
}

fn constrain_round_pow_samples<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirRoundCols<AB::VarMaybeExt>,
) {
    let pow_sample_mult = whir_round_pow_sample_mult::<AB>(local);
    let pow_sample = whir_round_pow_sample::<AB>(local);
    builder.assert_zero(
        pow_sample_mult.clone() *
            (pow_sample - local.pow_sample_high.clone() * whir_round_pow_shift::<AB>(local)),
    );
    builder.assert_zero((AB::one_maybe() - pow_sample_mult) * local.pow_sample_high.clone());
}

fn constrain_batch_eval_prefix_walk<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirBatchEvalCols<AB::VarMaybeExt>,
) {
    let pow_step = ext_mul::<AB>(&local.pow_in, &local.alpha);
    let acc_step = ext_mul::<AB>(&local.pow_in, &local.value);
    for limb in 0..D_EF {
        builder.assert_zero(
            local.is_value.clone() * (local.pow_out[limb].clone() - pow_step[limb].clone()),
        );
        builder.assert_zero(
            local.is_value.clone() *
                (local.acc_out[limb].clone() -
                    local.acc_in[limb].clone() -
                    acc_step[limb].clone()),
        );
        let expected_group_base_out = local.group_base_in[limb].clone() +
            local.is_group_end.clone() *
                (local.acc_out[limb].clone() - local.group_base_in[limb].clone());
        builder.assert_zero(
            local.is_value.clone() * (local.group_base_out[limb].clone() - expected_group_base_out),
        );
    }
}

fn constrain_batch_eval_segment_order<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirBatchEvalCols<AB::VarMaybeExt>,
) {
    let one = AB::one_maybe();
    let is_value = local.is_value.clone();
    let is_segment_start = local.is_segment_start.clone();
    let is_first_value = local.is_first_value.clone();
    let is_group_start = local.is_group_start.clone();
    let is_non_start_value = is_value.clone() * (one.clone() - is_segment_start.clone());

    builder.assert_zero(is_first_value.clone() * local.cursor.clone());
    builder.assert_zero(is_value.clone() * is_segment_start.clone() * local.value_idx.clone());
    builder.assert_zero(
        is_non_start_value.clone() *
            (local.log_height.clone() - local.chain_recv_log_height.clone()),
    );
    builder.assert_zero(
        is_non_start_value.clone() * (local.batch_id.clone() - local.chain_recv_batch_id.clone()),
    );
    builder.assert_zero(
        is_non_start_value.clone() * (local.batch_pos.clone() - local.chain_recv_batch_pos.clone()),
    );
    builder.assert_zero(
        is_non_start_value.clone() *
            (local.segment_element_count.clone() -
                local.chain_recv_segment_element_count.clone()),
    );
    builder.assert_zero(
        is_non_start_value *
            (local.value_idx.clone() - local.chain_recv_value_idx.clone() - one.clone()),
    );

    let recv_segment_end = local.chain_recv_value_idx.clone() + one.clone() -
        local.chain_recv_segment_element_count.clone();
    builder.assert_zero(local.is_start.clone() * recv_segment_end.clone());
    builder
        .assert_zero(is_segment_start.clone() * (one.clone() - is_first_value) * recv_segment_end);
    builder.assert_zero(
        local.is_segment_end.clone() *
            (local.value_idx.clone() + one.clone() - local.segment_element_count.clone()),
    );

    builder.assert_zero(
        is_segment_start.clone() *
            (one.clone() - is_group_start.clone()) *
            (local.chain_recv_log_height.clone() - local.log_height.clone()),
    );
    builder.assert_zero(
        is_segment_start.clone() *
            is_group_start.clone() *
            (local.group_log_height_gap.clone() - local.chain_recv_log_height.clone() +
                local.log_height.clone() +
                one.clone()),
    );
    builder
        .assert_zero((one.clone() - is_group_start.clone()) * local.group_log_height_gap.clone());

    builder.assert_zero(
        local.is_perm_batch.clone() *
            (local.batch_id.clone() - const_maybe::<AB>(WHIR_BATCH_PERMUTATION)),
    );
    builder.assert_zero(
        (one.clone() - local.is_perm_batch.clone()) *
            (local.segment_element_count.clone() - local.width.clone()),
    );
    builder.assert_zero(
        local.is_perm_batch.clone() *
            (const_maybe::<AB>(D_EF) * local.segment_element_count.clone() - local.width.clone()),
    );
}

fn constrain_query_chain_scaffold<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirQueryFoldReservedCols<AB::VarMaybeExt>,
    packed: &WhirQueryFoldPackedCols<AB::VarExt>,
) {
    let inv2 = AB::VarMaybeExt::from(AB::F::from_canonical_usize(2).inverse());
    builder.assert_zero(local.is_seed.clone() * local.chain_send_cursor.clone());
    builder.assert_zero(
        local.is_seed.clone() * (local.chain_send_idx.clone() - local.query_sample.clone()),
    );
    builder.assert_zero(
        local.is_seed.clone() *
            (local.query_sample_raw.clone() -
                local.query_sample.clone() -
                local.query_sample_shift.clone() * local.query_sample_high.clone()),
    );
    let high_gap = local.query_sample_high_max.clone() - local.query_sample_high.clone();
    let high_nonzero = high_gap.clone() * local.query_sample_high_gap_inv.clone();
    let high_inverse_error = high_nonzero.clone() - AB::one_maybe();
    builder.assert_zero(high_gap.clone() * high_inverse_error.clone());
    builder.assert_zero(local.query_sample_high_gap_inv.clone() * high_inverse_error);
    builder.assert_zero(local.query_sample.clone() * (AB::one_maybe() - high_nonzero));
    builder.assert_zero(local.is_seed.clone() * local.chain_send_acc.clone());
    builder.assert_zero(local.is_seed.clone() * (local.chain_send_ipw.clone() - inv2.clone()));
    builder.assert_zero_ext(packed.chain_send_folded.clone() * local.is_seed.clone());
    builder.assert_zero_ext((packed.folded.clone() - packed.cfr.clone()) * local.is_seed.clone());
    builder.assert_zero(
        local.is_seed.clone() *
            (local.idx.clone() -
                local.idx_bit.clone() -
                const_maybe::<AB>(2) * local.idx_tail_bit0.clone() -
                const_maybe::<AB>(4) * local.idx_tail_bit1.clone()),
    );
    builder.assert_zero(local.is_round.clone() * local.idx_tail_bit0.clone());
    builder.assert_zero(local.is_round.clone() * local.idx_tail_bit1.clone());

    builder.assert_zero(
        local.is_round.clone() *
            (local.chain_send_cursor.clone() - local.cursor.clone() - AB::one_maybe()),
    );
    builder.assert_zero(
        local.is_round.clone() *
            (local.idx.clone() -
                const_maybe::<AB>(2) * local.chain_send_idx.clone() -
                local.idx_bit.clone()),
    );
    builder.assert_zero(
        local.is_round.clone() *
            (local.chain_send_acc.clone() -
                local.acc.clone() -
                local.chain_send_idx_bit.clone() * local.ipw.clone()),
    );
    builder.assert_zero(
        local.is_round.clone() * (local.chain_send_ipw.clone() - local.ipw.clone() * inv2),
    );
}

fn constrain_query_twiddle_seed<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirQueryFoldReservedCols<AB::VarMaybeExt>,
) {
    let byte_exp = local.twiddle_bytes[0].clone() +
        const_maybe::<AB>(1 << 8) * local.twiddle_bytes[1].clone() +
        const_maybe::<AB>(1 << 16) * local.twiddle_bytes[2].clone();
    let inv2 = AB::VarMaybeExt::from(AB::F::from_canonical_usize(2).inverse());
    let tail_acc =
        local.ipw.clone() * (local.idx_tail_bit0.clone() + inv2 * local.idx_tail_bit1.clone());
    builder.assert_zero(
        local.is_seed.clone() *
            (byte_exp - const_maybe::<AB>(1 << 23) * (local.acc.clone() + tail_acc)),
    );
    builder.assert_zero(
        local.is_seed.clone() *
            (local.twiddle_product_01.clone() -
                local.twiddle_values[0].clone() * local.twiddle_values[1].clone()),
    );
    builder.assert_zero(
        local.is_seed.clone() *
            (local.chain_send_x.clone() -
                local.twiddle_product_01.clone() * local.twiddle_values[2].clone()),
    );
}

fn constrain_query_fold_arithmetic<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirQueryFoldReservedCols<AB::VarMaybeExt>,
    packed: &WhirQueryFoldPackedCols<AB::VarExt>,
) {
    let merge_eq_folded = packed.merge_eq.clone() * packed.folded.clone();
    let merge_beta_leaf = packed.merge_beta.clone() * packed.leaf_sum.clone();
    let selected =
        packed.f0.clone() + (packed.f1.clone() - packed.f0.clone()) * local.idx_bit.clone();
    let r_delta = packed.r_fold.clone() * (packed.f0.clone() - packed.f1.clone());

    builder.assert_zero_ext(packed.merge_beta.clone() * local.is_assign.clone());
    builder.assert_zero_ext(packed.merge_eq.clone() * local.is_assign.clone());
    builder.assert_zero_ext(
        (selected.clone() - packed.folded.clone()) *
            (local.is_round.clone() - local.is_merge.clone()),
    );
    builder.assert_zero_ext((selected.clone() - packed.leaf_sum.clone()) * local.is_assign.clone());
    builder.assert_zero_ext(
        (selected - merge_eq_folded - merge_beta_leaf) *
            (local.is_merge.clone() - local.is_assign.clone()),
    );
    builder.assert_zero_ext(
        packed.chain_send_folded.clone() * local.x.clone() * AB::VarMaybeExt::from(AB::F::two()) -
            (packed.f0.clone() + packed.f1.clone()) * local.x.clone() -
            r_delta,
    );
    builder.assert_zero(
        local.is_round.clone() *
            (local.chain_send_x.clone() *
                (AB::one_maybe() - const_maybe::<AB>(2) * local.chain_send_idx_bit.clone()) -
                local.x.clone() * local.x.clone()),
    );
}

fn query_pair_leaf_mask<AB: FullAirBuilder>(
    block: usize,
) -> [AB::VarMaybeExt; crate::config::DIGEST_SIZE] {
    core::array::from_fn(
        |idx| {
            if block == 0 || idx < 2 {
                AB::one_maybe()
            } else {
                AB::zero_maybe()
            }
        },
    )
}

fn query_pair_leaf_chunk<AB: FullAirBuilder>(
    local: &WhirQueryFoldCols<AB::VarMaybeExt>,
    block: usize,
) -> [AB::VarMaybeExt; crate::config::DIGEST_SIZE] {
    core::array::from_fn(|idx| match (block, idx) {
        (0, 0..=4) => local.f0[idx].clone(),
        (0, 5..=7) => local.f1[idx - 5].clone(),
        (1, 0..=1) => local.f1[idx + 3].clone(),
        _ => AB::zero_maybe(),
    })
}

fn round_coeffs<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    coeff_idx: usize,
) -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(|limb| round_coeff::<AB>(local, coeff_idx, limb))
}

fn round_coeff<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    coeff_idx: usize,
    limb: usize,
) -> AB::VarMaybeExt {
    debug_assert!(coeff_idx < 3);
    debug_assert!(limb < D_EF);
    local.event_value[8 + coeff_idx * D_EF + limb].clone()
}

fn round_sample<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    limb: usize,
) -> AB::VarMaybeExt {
    debug_assert!(limb < D_EF);
    local.event_value[23 + limb].clone()
}

fn round_merge_beta<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(|limb| round_merge_sample::<AB>(local, limb))
}

fn round_merge_sample<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    limb: usize,
) -> AB::VarMaybeExt {
    debug_assert!(limb < D_EF);
    local.event_value[28 + limb].clone()
}

fn one_ext<AB: FullAirBuilder>() -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(one_ext_limb::<AB>)
}

fn one_ext_limb<AB: FullAirBuilder>(limb: usize) -> AB::VarMaybeExt {
    if limb == 0 {
        AB::one_maybe()
    } else {
        AB::zero_maybe()
    }
}

fn ext_mul<AB: FullAirBuilder>(
    a: &[AB::VarMaybeExt; D_EF],
    b: &[AB::VarMaybeExt; D_EF],
) -> [AB::VarMaybeExt; D_EF] {
    let a_ext = ChallengeExtension(core::array::from_fn(|idx| a[idx].clone()));
    let b_ext = ChallengeExtension(core::array::from_fn(|idx| b[idx].clone()));
    (a_ext * b_ext).0
}

fn ext_scale<AB: FullAirBuilder>(
    a: &[AB::VarMaybeExt; D_EF],
    scalar: AB::VarMaybeExt,
) -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(|idx| a[idx].clone() * scalar.clone())
}

fn mask_ext<AB: FullAirBuilder>(
    value: &[AB::VarMaybeExt; D_EF],
    mask: AB::VarMaybeExt,
) -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(|idx| value[idx].clone() * mask.clone())
}

fn constrain_leaf_base_rlc<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirLeafStreamCols<AB::VarMaybeExt>,
) {
    // There is no per-query cycle-closing start row; every valid row is a data row
    // (all-zero padding satisfies the unconditional forms).
    let active = AB::one_maybe();
    let base_acc: [AB::VarMaybeExt; D_EF] = core::array::from_fn(|idx| {
        local.acc_in[idx].clone() * (AB::one_maybe() - local.is_unit_start.clone())
    });
    let mut contribution_sum: [AB::VarMaybeExt; D_EF] = core::array::from_fn(|_| AB::zero_maybe());

    for slot in 0..WHIR_LEAF_BASE_LIMBS_PER_ROW {
        let pow: [AB::VarMaybeExt; D_EF] =
            core::array::from_fn(|limb| local.slot_pows[slot][limb].clone());
        for limb in 0..D_EF {
            if slot == 0 {
                builder.assert_zero(
                    active.clone() *
                        (local.slot_pows[slot][limb].clone() - local.pow_in[limb].clone()),
                );
            }
        }
        let contribution = mask_ext::<AB>(
            &ext_scale::<AB>(&pow, local.values[slot].clone()),
            local.chunk_mask[slot].clone(),
        );
        for limb in 0..D_EF {
            contribution_sum[limb] = contribution_sum[limb].clone() + contribution[limb].clone();
        }
        let stepped = ext_mul::<AB>(&pow, &local.alpha);
        if slot + 1 < WHIR_LEAF_BASE_LIMBS_PER_ROW {
            for limb in 0..D_EF {
                builder.assert_zero(
                    active.clone() * (local.slot_pows[slot + 1][limb].clone() - pow[limb].clone()) -
                        local.chunk_mask[slot].clone() *
                            (stepped[limb].clone() - pow[limb].clone()),
                );
            }
        } else {
            for limb in 0..D_EF {
                builder.assert_zero(
                    active.clone() * (local.pow_out[limb].clone() - pow[limb].clone()) -
                        local.chunk_mask[slot].clone() *
                            (stepped[limb].clone() - pow[limb].clone()),
                );
            }
        }
    }

    for limb in 0..D_EF {
        builder.assert_zero(
            active.clone() * (local.acc_out[limb].clone() - base_acc[limb].clone()) -
                contribution_sum[limb].clone(),
        );
    }
}

fn constrain_leaf_base_unit_key_order<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirLeafStreamCols<AB::VarMaybeExt>,
) {
    constrain_leaf_unit_key_order(
        builder,
        local.is_valid.clone(),
        local.is_unit_start.clone(),
        local.is_unit_key_start.clone(),
        local.log_height.clone(),
        local.batch_id.clone(),
        local.chain_recv_log_height.clone(),
        local.chain_recv_batch_id.clone(),
        local.unit_key_gap.clone(),
    );
}

fn constrain_leaf_ext_rlc<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirLeafExtStreamReservedCols<AB::VarMaybeExt>,
    packed: &WhirLeafExtStreamPackedCols<AB::VarExt>,
) {
    let mut contribution_sum = AB::pack_ext_limbs(&[AB::zero_maybe()]);
    let alpha_step = packed.alpha.clone() - AB::pack_ext_limbs(&[AB::one_maybe()]);

    for elem_idx in 0..WHIR_LEAF_RLC_SLOTS {
        let pow = if elem_idx == 0 {
            packed.pow_in.clone()
        } else {
            packed.slot_pows[elem_idx - 1].clone()
        };
        let elem_mask = local.element_masks[elem_idx].clone();
        let masked_pow = pow.clone() * elem_mask;
        contribution_sum = contribution_sum + masked_pow.clone() * packed.values[elem_idx].clone();
        let next_pow = if elem_idx + 1 < WHIR_LEAF_RLC_SLOTS {
            packed.slot_pows[elem_idx].clone()
        } else {
            packed.pow_out.clone()
        };
        builder.assert_zero_ext(next_pow - pow - masked_pow * alpha_step.clone());
    }

    builder.assert_zero_ext(packed.acc_delta.clone() - contribution_sum);
}

#[allow(clippy::too_many_arguments)]
fn constrain_leaf_unit_key_order<AB: FullAirBuilder>(
    builder: &mut AB,
    is_valid: AB::VarMaybeExt,
    is_unit_start: AB::VarMaybeExt,
    is_unit_key_start: AB::VarMaybeExt,
    log_height: AB::VarMaybeExt,
    batch_id: AB::VarMaybeExt,
    chain_recv_log_height: AB::VarMaybeExt,
    chain_recv_batch_id: AB::VarMaybeExt,
    unit_key_gap: AB::VarMaybeExt,
) {
    // Order gadget: a group instance never crosses heights, so there is no
    // cross-height branch; within an instance the batch segments ascend strictly
    // (gap range-checked with mult = is_unit_key_start - is_unit_start,
    // matching the lookup). Instance-start rows have no chain recv (boundary mult),
    // so their chain_recv_* columns are dead and unconstrained.
    let one = AB::one_maybe();
    let is_same_key = is_valid.clone() * (one.clone() - is_unit_key_start.clone());
    let key_step = is_unit_key_start.clone() - is_unit_start;

    builder.assert_zero(is_same_key.clone() * (log_height.clone() - chain_recv_log_height.clone()));
    builder.assert_zero(is_same_key * (batch_id.clone() - chain_recv_batch_id.clone()));
    builder.assert_zero(key_step.clone() * (chain_recv_log_height - log_height));
    builder.assert_zero(key_step * (unit_key_gap - batch_id + chain_recv_batch_id + one));
}

fn whir_leaf_stream_denominators<AB: FullAirBuilder>(
    air: &WhirLeafStreamAir,
    builder: &AB,
    local: &WhirLeafStreamCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let mut denominators = Vec::with_capacity(6);
    denominators.push(air.leaf_pow_seed_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.log_height.clone(),
        local.alpha[0].clone(),
        local.alpha[1].clone(),
        local.alpha[2].clone(),
        local.alpha[3].clone(),
        local.alpha[4].clone(),
        local.pow_in[0].clone(),
        local.pow_in[1].clone(),
        local.pow_in[2].clone(),
        local.pow_in[3].clone(),
        local.pow_in[4].clone(),
    ));
    denominators.push(air.leaf_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.idx.clone(),
        local.chain_recv_cursor.clone(),
        local.chain_recv_log_height.clone(),
        local.chain_recv_batch_id.clone(),
        local.alpha[0].clone(),
        local.alpha[1].clone(),
        local.alpha[2].clone(),
        local.alpha[3].clone(),
        local.alpha[4].clone(),
        local.pow_in[0].clone(),
        local.pow_in[1].clone(),
        local.pow_in[2].clone(),
        local.pow_in[3].clone(),
        local.pow_in[4].clone(),
        local.acc_in[0].clone(),
        local.acc_in[1].clone(),
        local.acc_in[2].clone(),
        local.acc_in[3].clone(),
        local.acc_in[4].clone(),
    ));
    denominators.push(air.leaf_chain_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.idx.clone(),
        local.chain_send_cursor.clone(),
        local.log_height.clone(),
        local.batch_id.clone(),
        local.alpha[0].clone(),
        local.alpha[1].clone(),
        local.alpha[2].clone(),
        local.alpha[3].clone(),
        local.alpha[4].clone(),
        local.pow_out[0].clone(),
        local.pow_out[1].clone(),
        local.pow_out[2].clone(),
        local.pow_out[3].clone(),
        local.pow_out[4].clone(),
        local.acc_out[0].clone(),
        local.acc_out[1].clone(),
        local.acc_out[2].clone(),
        local.acc_out[3].clone(),
        local.acc_out[4].clone(),
    ));
    let mask = core::array::from_fn(|idx| local.chunk_mask[idx].clone());
    let chunk = core::array::from_fn(|idx| local.values[idx].clone());
    denominators.push(air.merkle_leaf_block_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.batch_id.clone(),
        local.unit_key.clone(),
        local.idx.clone(),
        local.block_idx.clone(),
        mask,
        chunk,
    ));
    denominators.push(air.query_leaf_sum_bus.denominator(
        builder,
        local.proof_idx.clone(),
        local.idx.clone(),
        local.log_height.clone(),
        local.acc_out[0].clone(),
        local.acc_out[1].clone(),
        local.acc_out[2].clone(),
        local.acc_out[3].clone(),
        local.acc_out[4].clone(),
    ));
    denominators.push(air.range_bus.denominator(
        builder,
        RangeCheckerBusMessage {
            value: local.unit_key_gap.clone(),
            max_bits: const_maybe::<AB>(8),
        },
    ));
    denominators
}

fn whir_leaf_ext_stream_precomputed<AB: FullAirBuilder>(
    air: &WhirLeafExtStreamAir,
    builder: &AB,
    local: &WhirLeafExtStreamCols<AB::VarMaybeExt>,
) -> WhirLeafExtStreamPrecomputedCols<AB::VarExt> {
    let batch_id = const_maybe::<AB>(WHIR_BATCH_PERMUTATION);
    let chain_recv_batch_id = batch_id.clone() - local.is_unit_key_start.clone();
    let unit_key = const_maybe::<AB>(whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, 0)) +
        local.log_height.clone();

    WhirLeafExtStreamPrecomputedCols {
        denominators: WhirLeafExtStreamDenominatorCols {
            leaf_chain_recv: air.leaf_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.idx.clone(),
                local.chain_recv_cursor.clone(),
                local.log_height.clone(),
                chain_recv_batch_id,
                local.alpha[0].clone(),
                local.alpha[1].clone(),
                local.alpha[2].clone(),
                local.alpha[3].clone(),
                local.alpha[4].clone(),
                local.pow_in[0].clone(),
                local.pow_in[1].clone(),
                local.pow_in[2].clone(),
                local.pow_in[3].clone(),
                local.pow_in[4].clone(),
                local.acc_in[0].clone(),
                local.acc_in[1].clone(),
                local.acc_in[2].clone(),
                local.acc_in[3].clone(),
                local.acc_in[4].clone(),
            ),
            leaf_chain_send: air.leaf_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.idx.clone(),
                local.chain_recv_cursor.clone() + AB::one_maybe(),
                local.log_height.clone(),
                batch_id,
                local.alpha[0].clone(),
                local.alpha[1].clone(),
                local.alpha[2].clone(),
                local.alpha[3].clone(),
                local.alpha[4].clone(),
                local.pow_out[0].clone(),
                local.pow_out[1].clone(),
                local.pow_out[2].clone(),
                local.pow_out[3].clone(),
                local.pow_out[4].clone(),
                local.acc_out[0].clone(),
                local.acc_out[1].clone(),
                local.acc_out[2].clone(),
                local.acc_out[3].clone(),
                local.acc_out[4].clone(),
            ),
            merkle_leaf_blocks: core::array::from_fn(|block| {
                let chunk = core::array::from_fn(|idx| {
                    local.values[block * WHIR_LEAF_BASE_LIMBS_PER_ROW + idx].clone()
                });
                air.merkle_leaf_block_bus.denominator_with_mask_bitset(
                    builder,
                    local.proof_idx.clone(),
                    const_maybe::<AB>(WHIR_BATCH_PERMUTATION),
                    unit_key.clone(),
                    local.idx.clone(),
                    local.block_idx.clone() + const_maybe::<AB>(block),
                    whir_leaf_ext_merkle_mask_bitset::<AB>(&local.element_masks, block),
                    chunk,
                )
            }),
            query_leaf_sum: air.query_leaf_sum_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.idx.clone(),
                local.log_height.clone(),
                local.acc_out[0].clone(),
                local.acc_out[1].clone(),
                local.acc_out[2].clone(),
                local.acc_out[3].clone(),
                local.acc_out[4].clone(),
            ),
        },
        packed: WhirLeafExtStreamPackedCols {
            alpha: AB::pack_ext_limbs(&local.alpha),
            pow_in: AB::pack_ext_limbs(&local.pow_in),
            slot_pows: core::array::from_fn(|slot| AB::pack_ext_limbs(&local.slot_pows[slot])),
            pow_out: AB::pack_ext_limbs(&local.pow_out),
            acc_delta: AB::pack_ext_limbs(&core::array::from_fn::<AB::VarMaybeExt, D_EF, _>(
                |limb| local.acc_out[limb].clone() - local.acc_in[limb].clone(),
            )),
            values: core::array::from_fn(|slot| {
                AB::pack_ext_limbs(&local.values[slot * D_EF..(slot + 1) * D_EF])
            }),
        },
    }
}

const WHIR_LEAF_EXT_MERKLE_MASK_WEIGHTS: [[usize; WHIR_LEAF_RLC_SLOTS]; WHIR_LEAF_BLOCKS_PER_ROW] = [
    [31, 224, 0, 0, 0, 0, 0, 0],
    [0, 3, 124, 128, 0, 0, 0, 0],
    [0, 0, 0, 15, 240, 0, 0, 0],
    [0, 0, 0, 0, 1, 62, 192, 0],
    [0, 0, 0, 0, 0, 0, 7, 248],
];

fn whir_leaf_ext_merkle_mask_bitset<AB: FullAirBuilder>(
    element_masks: &[AB::VarMaybeExt; WHIR_LEAF_RLC_SLOTS],
    block: usize,
) -> AB::VarMaybeExt {
    WHIR_LEAF_EXT_MERKLE_MASK_WEIGHTS[block]
        .into_iter()
        .enumerate()
        .filter(|(_, weight)| *weight != 0)
        .fold(AB::zero_maybe(), |bitset, (element, weight)| {
            bitset + element_masks[element].clone() * const_maybe::<AB>(weight)
        })
}

fn final_root_codeword_limb<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    idx: usize,
) -> AB::VarMaybeExt {
    local.cfr[idx % D_EF].clone()
}

fn final_root_log_blowup_flag<AB: FullAirBuilder>(
    log_blowup: usize,
    target: usize,
) -> AB::VarMaybeExt {
    let lb = const_maybe::<AB>(log_blowup);
    match target {
        1 => {
            let inv2 = AB::VarMaybeExt::from(AB::F::from_canonical_usize(2).inverse());
            (lb.clone() - const_maybe::<AB>(2)) * (lb - const_maybe::<AB>(3)) * inv2
        }
        2 => {
            let term = (lb.clone() - const_maybe::<AB>(1)) * (lb - const_maybe::<AB>(3));
            AB::zero_maybe() - term
        }
        3 => {
            let inv2 = AB::VarMaybeExt::from(AB::F::from_canonical_usize(2).inverse());
            (lb.clone() - const_maybe::<AB>(1)) * (lb - const_maybe::<AB>(2)) * inv2
        }
        _ => panic!("unsupported WHIR log_blowup selector"),
    }
}

fn final_root_poseidon2_mult<AB: FullAirBuilder>(
    log_blowup: usize,
    perm_idx: usize,
) -> AB::VarMaybeExt {
    debug_assert!(perm_idx < WHIR_FINAL_ROOT_POSEIDON2_PERMS);
    match perm_idx {
        0 | 1 => {
            final_root_log_blowup_flag::<AB>(log_blowup, 1) +
                const_maybe::<AB>(2) * final_root_log_blowup_flag::<AB>(log_blowup, 2) +
                const_maybe::<AB>(4) * final_root_log_blowup_flag::<AB>(log_blowup, 3)
        }
        2 => {
            final_root_log_blowup_flag::<AB>(log_blowup, 2) +
                const_maybe::<AB>(2) * final_root_log_blowup_flag::<AB>(log_blowup, 3)
        }
        3 => final_root_log_blowup_flag::<AB>(log_blowup, 3),
        _ => AB::zero_maybe(),
    }
}

fn final_root_recv_state<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; POSEIDON2_WIDTH] {
    core::array::from_fn(|lane| final_root_recv_state_lane::<AB>(local, lane))
}

fn final_root_poseidon2_output<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; POSEIDON2_WIDTH] {
    core::array::from_fn(|lane| final_root_poseidon2_output_lane::<AB>(local, lane))
}

fn final_root_recv_state_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    lane: usize,
) -> AB::VarMaybeExt {
    debug_assert!(lane < POSEIDON2_WIDTH);
    if lane < WHIR_FINAL_ROOT_DIGEST_LANES {
        local.event_value[lane].clone()
    } else if lane < 13 {
        local.r_fold[lane - WHIR_FINAL_ROOT_DIGEST_LANES].clone()
    } else {
        local.claim_acc[lane - 13].clone()
    }
}

fn final_root_poseidon2_output_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    lane: usize,
) -> AB::VarMaybeExt {
    debug_assert!(lane < POSEIDON2_WIDTH);
    if lane < D_EF {
        local.claim_folded[lane].clone()
    } else if lane < 2 * D_EF {
        local.eq_factor[lane - D_EF].clone()
    } else if lane < 3 * D_EF {
        local.eq_folded[lane - 2 * D_EF].clone()
    } else {
        local.event_value[WHIR_ROUND_MAX_TRANSCRIPT_EVENTS - 1].clone()
    }
}

fn final_root_send_state_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    lane: usize,
) -> AB::VarMaybeExt {
    debug_assert!(lane < POSEIDON2_WIDTH);
    local.event_value[16 + lane].clone()
}

fn final_root_seed_state_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    lane: usize,
) -> AB::VarMaybeExt {
    debug_assert!(lane < POSEIDON2_WIDTH);
    if lane < WHIR_FINAL_ROOT_DIGEST_LANES {
        final_root_codeword_limb::<AB>(local, lane)
    } else {
        AB::zero_maybe()
    }
}

fn final_root_duplicated_digest_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    lane: usize,
) -> AB::VarMaybeExt {
    final_root_poseidon2_output_lane::<AB>(local, lane % WHIR_FINAL_ROOT_DIGEST_LANES)
}

fn final_root_step_send_state_lane<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    log_blowup: usize,
    step: usize,
    lane: usize,
) -> AB::VarMaybeExt {
    let lb1 = final_root_log_blowup_flag::<AB>(log_blowup, 1);
    let lb2 = final_root_log_blowup_flag::<AB>(log_blowup, 2);
    let lb3 = final_root_log_blowup_flag::<AB>(log_blowup, 3);
    let input = final_root_recv_state_lane::<AB>(local, lane);
    let output = final_root_poseidon2_output_lane::<AB>(local, lane);
    let duplicated = final_root_duplicated_digest_lane::<AB>(local, lane);
    match step {
        0 => {
            if lane < 2 {
                final_root_codeword_limb::<AB>(local, WHIR_FINAL_ROOT_DIGEST_LANES + lane)
            } else {
                output
            }
        }
        1 => lb1 * output + (lb2 + lb3) * duplicated,
        2 => lb1 * input + lb2 * output + lb3 * duplicated,
        3 => (lb1 + lb2) * input + lb3 * output,
        _ => AB::zero_maybe(),
    }
}

fn constrain_round_final_root_poseidon2<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &WhirRoundCols<AB::VarMaybeExt>,
    log_blowup: usize,
) {
    let step_flag_sum = local
        .final_root_perm_step_flags
        .iter()
        .cloned()
        .fold(AB::zero_maybe(), |acc, flag| acc + flag);
    builder.assert_eq(step_flag_sum, local.is_final_perm.clone());

    let mut expected_poseidon_mult = AB::zero_maybe();
    for step in 0..WHIR_FINAL_ROOT_POSEIDON2_PERMS {
        expected_poseidon_mult = expected_poseidon_mult +
            local.final_root_perm_step_flags[step].clone() *
                final_root_poseidon2_mult::<AB>(log_blowup, step);
        builder.assert_zero(
            local.final_root_perm_step_flags[step].clone() *
                (local.opening_idx.clone() - const_maybe::<AB>(step)),
        );
        builder.assert_zero(
            local.final_root_perm_step_flags[step].clone() *
                (local.height_group_rank.clone() - const_maybe::<AB>(step + 1)),
        );
    }
    builder.assert_eq(local.final_root_poseidon2_recv_mult.clone(), expected_poseidon_mult);
    builder.assert_zero(
        local.is_final.clone() *
            (local.opening_idx.clone() - const_maybe::<AB>(WHIR_FINAL_ROOT_POSEIDON2_PERMS)),
    );
    builder.assert_zero(local.is_final.clone() * local.height_group_rank.clone());

    for lane in 0..POSEIDON2_WIDTH {
        builder.assert_zero(
            local.is_final.clone() *
                (final_root_send_state_lane::<AB>(local, lane) -
                    final_root_seed_state_lane::<AB>(local, lane)),
        );
        for step in 0..WHIR_FINAL_ROOT_POSEIDON2_PERMS {
            builder.assert_zero(
                local.final_root_perm_step_flags[step].clone() *
                    (final_root_send_state_lane::<AB>(local, lane) -
                        final_root_step_send_state_lane::<AB>(local, log_blowup, step, lane)),
            );
        }
    }
}

fn whir_round_event_tidx<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    idx: usize,
) -> AB::VarMaybeExt {
    let mut offset = const_maybe::<AB>(idx);
    if idx >= 8 {
        offset = offset - local.is_round.clone() * const_maybe::<AB>(8) +
            local.round_has_oracle.clone() * const_maybe::<AB>(8);
    }
    local.tidx.clone() + offset
}

fn whir_round_event_is_sample<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    idx: usize,
) -> AB::VarMaybeExt {
    let pow_sample = if idx == 2 { local.is_pow_batch.clone() } else { AB::zero_maybe() };
    let round_sample = if (23..WHIR_ROUND_MAX_TRANSCRIPT_EVENTS).contains(&idx) {
        local.is_round.clone()
    } else {
        AB::zero_maybe()
    };
    let final_sample = if idx == 10 { local.is_final.clone() } else { AB::zero_maybe() };
    pow_sample + round_sample + final_sample
}

fn whir_round_chain_mult<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_valid.clone() - local.is_final_perm.clone()
}

fn whir_final_root_chain_mult<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_final.clone() + local.is_final_perm.clone()
}

fn whir_round_pow_sample_mult<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_pow_batch.clone() + local.is_final.clone()
}

fn whir_round_pow_sample<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_pow_batch.clone() * local.event_value[2].clone() +
        local.is_final.clone() * local.event_value[10].clone()
}

fn whir_round_pow_shift<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_pow_batch.clone() * const_maybe::<AB>(WHIR_BATCHING_POW_SHIFT) +
        local.is_final.clone() * const_maybe::<AB>(WHIR_QUERY_POW_SHIFT)
}

fn whir_round_pow_range_bits<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    whir_round_pow_sample_mult::<AB>(local) * const_maybe::<AB>(WHIR_PAIRED_RANGE_BITS)
}

fn whir_round_pow_high_max<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_pow_batch.clone() * const_maybe::<AB>(WHIR_BATCHING_POW_HIGH_MAX) +
        local.is_final.clone() * const_maybe::<AB>(WHIR_QUERY_POW_HIGH_MAX)
}

fn whir_round_event_mult<AB: FullAirBuilder>(
    local: &WhirRoundCols<AB::VarMaybeExt>,
    idx: usize,
) -> AB::VarMaybeExt {
    let pow_mult = if idx < 3 { local.is_pow_batch.clone() } else { AB::zero_maybe() };
    let preamble_mult = if idx < 8 { local.is_preamble.clone() } else { AB::zero_maybe() };
    let round_mult = if idx < 8 {
        local.round_has_oracle.clone()
    } else if idx < 28 {
        local.is_round.clone()
    } else {
        local.is_merge.clone()
    };
    let final_mult = if idx < 11 { local.is_final.clone() } else { AB::zero_maybe() };
    pow_mult + preamble_mult + round_mult + final_mult
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn assert_flag_implies<AB: FullAirBuilder>(
    builder: &mut AB,
    flag: AB::VarMaybeExt,
    condition: AB::VarMaybeExt,
) {
    builder.assert_zero(flag * (AB::one_maybe() - condition));
}

fn assert_prefix_mask<AB: FullAirBuilder, const N: usize>(
    builder: &mut AB,
    mask: &[AB::VarMaybeExt; N],
) {
    for idx in 1..N {
        builder.assert_zero(mask[idx].clone() * (AB::one_maybe() - mask[idx - 1].clone()));
    }
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::D_EF,
        symbolic_expr_fixed_dt::RecursionFixedSymbolicChip,
        symbolic_ir_dt::RecursionPolyAirChipIr,
        whir_dt::columns::{
            NUM_WHIR_LEAF_EXT_STREAM_DENOMINATOR_COLS, NUM_WHIR_LEAF_EXT_STREAM_PACKED_COLS,
            NUM_WHIR_LEAF_EXT_STREAM_PRECOMPUTED_COLS, NUM_WHIR_LEAF_EXT_STREAM_RESERVED_COLS,
            NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS, NUM_WHIR_QUERY_FOLD_RESERVED_COLS,
        },
    };
    use polyair::Chip;

    #[test]
    fn leaf_ext_stream_layout_is_exact() {
        assert_eq!(NUM_WHIR_LEAF_EXT_STREAM_COLS, 116);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, proof_idx), 0);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, is_unit_end), 1);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, idx), 2);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, serve_cnt), 3);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, chain_recv_cursor), 4);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, log_height), 5);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, is_unit_key_start), 6);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, alpha), 7);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, pow_in), 12);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, acc_in), 17);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, slot_pows), 22);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, pow_out), 57);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, acc_out), 62);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, values), 67);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, element_masks), 107);
        assert_eq!(core::mem::offset_of!(WhirLeafExtStreamCols<u8>, block_idx), 115);
        assert_eq!(
            whir_leaf_ext_reserved_main_indices().as_slice(),
            &[1, 3, 6, 107, 108, 109, 110, 111, 112, 113, 114]
        );
        assert_eq!(NUM_WHIR_LEAF_EXT_STREAM_RESERVED_COLS, 11);
        assert_eq!(NUM_WHIR_LEAF_EXT_STREAM_DENOMINATOR_COLS, 8);
        assert_eq!(NUM_WHIR_LEAF_EXT_STREAM_PACKED_COLS, 19);
        assert_eq!(NUM_WHIR_LEAF_EXT_STREAM_PRECOMPUTED_COLS, 27);
    }

    #[test]
    fn query_fold_layout_is_exact() {
        assert_eq!(NUM_WHIR_QUERY_FOLD_COLS, 84);
        assert_eq!(NUM_WHIR_QUERY_FOLD_RESERVED_COLS, 33);
        assert_eq!(NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS, 14);
        assert_eq!(NUM_WHIR_QUERY_FOLD_PACKED_COLS, 9);
        assert_eq!(NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS, 23);
        assert_eq!(
            whir_query_fold_reserved_main_indices(),
            vec![
                1, 2, 4, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 37, 38, 39, 40, 41, 42,
                53, 54, 55, 66, 77, 78, 79, 80, 81, 82, 83,
            ]
        );

        let query = Chip::<WhirQueryFoldAir, F, D_EF>::new(WhirQueryFoldAir::default());
        assert_eq!(query.width(), NUM_WHIR_QUERY_FOLD_COLS);
        assert_eq!(query.reserved_poly().len(), NUM_WHIR_QUERY_FOLD_RESERVED_COLS);
        assert_eq!(query.num_precompute(), NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS);
        assert_eq!(query.perm_width(), 7);
        assert_eq!(query.reserved_poly().len() + query.num_precompute() + query.perm_width(), 63);
        assert_eq!(query.num_lookup(), 14);
        assert_eq!(query.symbolic_builder.gate.len(), 42);
        assert_eq!(query.num_alpha, 50);
        assert!(query.degree <= 3);
        let query_fixed =
            RecursionFixedSymbolicChip::from_polyair_chip(0, &query).expect("fixed WhirQueryFold");
        let query_ir = RecursionPolyAirChipIr::compile(&query_fixed).expect("WhirQueryFold IR");
        let query_roots = query_ir.gate_roots.len() + 2 * query_ir.lookup_multiplicity_roots.len();
        let query_folds = query_ir.gate_roots.len() +
            query_ir.lookup_multiplicity_roots.len().div_ceil(query_ir.logup_batch_size) +
            1;
        assert_eq!(query.width(), 84);
        assert_eq!(query.reserved_poly().len(), 33);
        assert_eq!(query.num_precompute(), 23);
        assert_eq!(query.perm_width(), 7);
        assert_eq!(query_ir.node_table.len(), 430);
        assert_eq!(query_ir.gate_roots.len(), 42);
        assert_eq!(query_ir.lookup_multiplicity_roots.len(), 14);
        assert_eq!(query_roots, 70);
        assert_eq!(query_folds, 50);
        assert_eq!(query_roots.next_power_of_two(), 128);
        assert_eq!(query_folds.next_power_of_two(), 64);
    }

    #[test]
    fn symbolic_analysis() {
        let twiddle = Chip::<WhirTwiddleTableAir, F, D_EF>::new(WhirTwiddleTableAir::default());
        assert_eq!(twiddle.num_lookup(), 3);
        assert!(twiddle.degree <= 3);

        let sample_band = Chip::<WhirSampleBandAir, F, D_EF>::new(WhirSampleBandAir::default());
        assert_eq!(sample_band.num_lookup(), 1);
        assert!(sample_band.degree <= 3);

        let round = Chip::<WhirRoundAir, F, D_EF>::new(WhirRoundAir::default());
        assert_eq!(round.num_lookup(), 47);
        assert!(round.required_max_beta_power() >= 25);
        assert!(round.degree <= 3);

        let batch = Chip::<WhirBatchEvalAir, F, D_EF>::new(WhirBatchEvalAir::default());
        assert_eq!(batch.num_lookup(), 12);
        assert!(batch.degree <= 3);

        let query = Chip::<WhirQueryFoldAir, F, D_EF>::new(WhirQueryFoldAir::default());
        assert_eq!(query.num_lookup(), 14);
        assert!(query.required_max_beta_power() >= 20);
        assert_eq!(query.symbolic_builder.gate.len(), 42);
        assert_eq!(query.num_alpha, 50);
        assert!(query.degree <= 3);

        let leaf = Chip::<WhirLeafStreamAir, F, D_EF>::new(WhirLeafStreamAir::default());
        assert_eq!(leaf.num_lookup(), 6);
        assert!(leaf.degree <= 3);

        let leaf_ext = Chip::<WhirLeafExtStreamAir, F, D_EF>::new(WhirLeafExtStreamAir::default());
        assert_eq!(leaf_ext.num_lookup(), 8);
        assert_eq!(leaf_ext.width(), 116);
        assert_eq!(leaf_ext.reserved_poly().len(), 11);
        assert_eq!(leaf_ext.num_precompute(), 27);
        assert_eq!(leaf_ext.perm_width(), 4);
        assert_eq!(leaf_ext.symbolic_builder.gate.len(), 20);
        assert_eq!(leaf_ext.num_alpha, 25);
        assert_eq!(leaf_ext.required_max_beta_power(), 22);
        assert!(leaf_ext.degree <= 3);

        let fixed = RecursionFixedSymbolicChip::from_polyair_chip(0, &leaf_ext)
            .expect("fixed WhirLeafExtStream");
        let ir = RecursionPolyAirChipIr::compile(&fixed).expect("WhirLeafExtStream IR");
        let roots = ir.gate_roots.len() + 2 * ir.lookup_multiplicity_roots.len();
        let folds = ir.gate_roots.len() +
            ir.lookup_multiplicity_roots.len().div_ceil(ir.logup_batch_size) +
            1;
        assert_eq!(ir.node_table.len(), 461);
        assert_eq!(roots, 36);
        assert_eq!(folds, 25);
    }

    #[test]
    fn leaf_ext_direct_merkle_mask_bitsets_match_flat_ext5_masks() {
        for element_count in 0..=WHIR_LEAF_RLC_SLOTS {
            let element_masks = core::array::from_fn::<usize, WHIR_LEAF_RLC_SLOTS, _>(|idx| {
                usize::from(idx < element_count)
            });
            for block in 0..WHIR_LEAF_BLOCKS_PER_ROW {
                let direct = WHIR_LEAF_EXT_MERKLE_MASK_WEIGHTS[block]
                    .into_iter()
                    .zip(element_masks)
                    .map(|(weight, mask)| weight * mask)
                    .sum::<usize>();
                let flat = (0..WHIR_LEAF_BASE_LIMBS_PER_ROW)
                    .map(|limb| {
                        let flat_limb = block * WHIR_LEAF_BASE_LIMBS_PER_ROW + limb;
                        usize::from(flat_limb / D_EF < element_count) << limb
                    })
                    .sum::<usize>();
                assert_eq!(direct, flat, "element_count={element_count}, block={block}");
            }
        }
    }
}
