use p3_field::PrimeField64;

use crate::interaction::{
    validate_recursion_interaction_budget, RecursionInteractionBudget,
    RecursionInteractionBudgetError, RecursionInteractionIdx, RecursionInteractionIndexSpace,
};

pub const GLOBAL_RECURSION_INTERACTION_IDX_START: usize = 0;
pub const PER_PROOF_RECURSION_INTERACTION_IDX_START: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionInteractionSchema {
    pub name: &'static str,
    pub interaction_idx: RecursionInteractionIdx,
    pub index_space: RecursionInteractionIndexSpace,
    pub payload_arity: usize,
}

impl RecursionInteractionSchema {
    pub const fn global(name: &'static str, interaction_idx: usize, payload_arity: usize) -> Self {
        assert!(interaction_idx < PER_PROOF_RECURSION_INTERACTION_IDX_START);
        Self {
            name,
            interaction_idx: RecursionInteractionIdx(interaction_idx),
            index_space: RecursionInteractionIndexSpace::Global,
            payload_arity,
        }
    }

    pub const fn per_proof(
        name: &'static str,
        interaction_idx: usize,
        payload_arity: usize,
    ) -> Self {
        assert!(interaction_idx >= PER_PROOF_RECURSION_INTERACTION_IDX_START);
        Self {
            name,
            interaction_idx: RecursionInteractionIdx(interaction_idx),
            index_space: RecursionInteractionIndexSpace::PerProof,
            payload_arity,
        }
    }

    pub const fn proof_idx_arity(&self) -> usize {
        match self.index_space {
            RecursionInteractionIndexSpace::Global => 0,
            RecursionInteractionIndexSpace::PerProof => 1,
        }
    }

    pub const fn denominator_value_count(&self) -> usize {
        1 + self.proof_idx_arity() + self.payload_arity
    }

    pub const fn required_max_beta_power_floor(&self) -> usize {
        let values_width = self.denominator_value_count();
        if values_width > 13 {
            values_width
        } else {
            13
        }
    }
}

pub const POSEIDON2_PERMUTE_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("Poseidon2Permute", 0, 32);
pub const RANGE_CHECKER_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("RangeChecker", 1, 2);
// Bus id 2 (PowerChecker) is retired.
// Note: never reuse retired bus ids.
pub const NATIVE_CHIP_METADATA_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("NativeChipMetadata", 3, 9);
pub const MERKLE_DIGEST_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("MerkleDigestChain", 1000, 11);
// The 1001 state chain is keyed by `(unit_key, idx)`. The 1004 leaf payload also carries
// `commit_id`, binding each block to the transcript-authenticated commitment tree.
pub const MERKLE_SPONGE_STATE_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("MerkleSpongeStateChain", 1001, 19);
pub const MERKLE_COMMITMENT_ROOT_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("MerkleCommitmentRoot", 1002, 9);
// Bus id 1003 (MerkleQuerySeed) is retired; the query-to-leaf-index binding rides the 1025/1004
// keys. Note: never reuse retired bus ids.
pub const MERKLE_LEAF_BLOCK_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("MerkleLeafBlock", 1004, 13);
pub const TRANSCRIPT_SPONGE_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("TranscriptSpongeChain", 1006, 18);
pub const TRANSCRIPT_EVENT_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("TranscriptEvent", 1007, 3);
pub const PROOF_SHAPE_CHIP_META_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeChipMeta", 1008, 5);
pub const PROOF_SHAPE_BATCH_DIM_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeBatchDim", 1009, 6);
pub const PROOF_SHAPE_VALUES_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeValues", 1010, 3);
pub const PROOF_SHAPE_HEIGHT_GROUP_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeHeightGroup", 1011, 2);
pub const PROOF_SHAPE_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeChain", 1012, 7);
pub const PROOF_SHAPE_HEIGHT_MEMBER_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeHeightMember", 1013, 1);
pub const PROOF_SHAPE_HEIGHT_RANK_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeHeightRank", 1014, 2);
pub const BATCH_OPENING_POINT_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("BatchOpeningPoint", 1017, 6);
pub const BATCH_SUMCHECK_CLAIM_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("BatchSumcheckClaimChain", 1018, 8);
pub const SUMCHECK_OUT_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("SumcheckOut", 1019, 7);
pub const PROOF_SHAPE_SUMMARY_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeSummary", 1022, 4);
pub const WHIR_TWIDDLE_POW_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("WhirTwiddlePow", 4, 3);
pub const WHIR_SAMPLE_BAND_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("WhirSampleBand", 11, 4);
pub const WHIR_ROUND_BCAST_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirRoundBcast", 1023, 19);
pub const WHIR_GROUP_CLAIM_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirGroupClaim", 1024, 6);
pub const WHIR_QUERY_LEAF_SUM_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirQueryLeafSum", 1025, 7);
pub const WHIR_QUERY_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirQueryChain", 1026, 14);
pub const WHIR_EVAL_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirEvalChain", 1027, 26);
pub const WHIR_LEAF_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirLeafChain", 1028, 19);
// Bus id 1029 (WhirAlphaBcast) is retired; alpha rides the WhirLeafPowSeed payload (1044).
// Note: never reuse retired bus ids.
pub const WHIR_QUERY_INIT_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirQueryInit", 1030, 8);
pub const WHIR_OPENED_EVAL_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirOpenedEval", 1031, 9);
pub const WHIR_ROUND_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirRoundChain", 1032, 23);
pub const CONSTRAINT_PROGRAM_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("ConstraintProgram", 6, 9);
pub const CONSTRAINT_ROOT_TABLE_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("ConstraintRootTable", 7, 5);
pub const STATEMENT_SCALAR_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("StatementScalarChain", 8, 12);
pub const STATEMENT_DIGEST_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("StatementDigestChain", 9, 10);
pub const STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("StatementGlobalIntervalChain", 15, 13);
pub const STATEMENT_CONFIG_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("StatementConfig", 12, 9);
pub const CONSTRAINT_HEIGHT_INVERSE_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::global("ConstraintHeightInverse", 13, 2);
pub const CONSTRAINT_NODE_VALUE_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ConstraintNodeValue", 1033, 8);
pub const CONSTRAINT_CHALLENGE_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ConstraintChallenge", 1034, 8);
pub const CONSTRAINT_FOLD_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ConstraintFoldChain", 1035, 21);
pub const CONSTRAINT_EQ_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ConstraintEqChain", 1036, 16);
pub const STATEMENT_VK_DIGEST_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("StatementVkDigest", 1038, 9);
pub const STATEMENT_HASH_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("StatementHashChain", 1039, 18);
pub const STATEMENT_CHILD_FACTS_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("StatementChildFacts", 1040, 3);
pub const BETA_LADDER_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("BetaLadderChain", 1042, 11);
pub const WHIR_FINAL_ROOT_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirFinalRootChain", 1043, 17);
// Per-height-group alpha-power seeds for the deduped leaf streams, published from
// WhirBatchEval's group-start rows.
pub const WHIR_LEAF_POW_SEED_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("WhirLeafPowSeed", 1044, 11);

// Sponge state-window buses — reserved but unused.
// Note: ids 1045-1047 stay reserved; never reuse them.
pub const SPONGE_ABSORB_WINDOW_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("SpongeAbsorbWindow", 1045, 9);
pub const SPONGE_SQUEEZE_WINDOW_LO_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("SpongeSqueezeWindowLo", 1046, 9);
pub const SPONGE_SQUEEZE_WINDOW_HI_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("SpongeSqueezeWindowHi", 1047, 9);
pub const CONSTRAINT_FOLD_PLAN_CHAIN_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ConstraintFoldPlanChain", 1048, 3);
// Bus id 1049 (ConstraintFoldPlan row equivalence) is retired after the canonical
// schedule was fused into ConstraintFoldAir. Note: never reuse retired bus ids.
/// Binder-aligned public-value rows used by the Core boundary checker.  The payload is
/// `(shape_idx_base, packed[0..5], packed[5..8])`; the latter two logical fields are
/// extension elements rather than eight independent base-field lookup coordinates.
pub const PROOF_SHAPE_GLOBAL_PACKED_SCHEMA: RecursionInteractionSchema =
    RecursionInteractionSchema::per_proof("ProofShapeGlobalPacked", 1050, 3);

pub const NATIVE_RECURSION_SCHEMAS: &[RecursionInteractionSchema] = &[
    POSEIDON2_PERMUTE_SCHEMA,
    RANGE_CHECKER_SCHEMA,
    NATIVE_CHIP_METADATA_SCHEMA,
    MERKLE_DIGEST_CHAIN_SCHEMA,
    MERKLE_SPONGE_STATE_CHAIN_SCHEMA,
    MERKLE_COMMITMENT_ROOT_SCHEMA,
    MERKLE_LEAF_BLOCK_SCHEMA,
    TRANSCRIPT_SPONGE_CHAIN_SCHEMA,
    TRANSCRIPT_EVENT_SCHEMA,
    PROOF_SHAPE_CHIP_META_SCHEMA,
    PROOF_SHAPE_BATCH_DIM_SCHEMA,
    PROOF_SHAPE_VALUES_SCHEMA,
    PROOF_SHAPE_HEIGHT_GROUP_SCHEMA,
    PROOF_SHAPE_CHAIN_SCHEMA,
    PROOF_SHAPE_HEIGHT_MEMBER_SCHEMA,
    PROOF_SHAPE_HEIGHT_RANK_SCHEMA,
    BATCH_OPENING_POINT_SCHEMA,
    BATCH_SUMCHECK_CLAIM_CHAIN_SCHEMA,
    SUMCHECK_OUT_SCHEMA,
    PROOF_SHAPE_SUMMARY_SCHEMA,
    WHIR_TWIDDLE_POW_SCHEMA,
    WHIR_SAMPLE_BAND_SCHEMA,
    WHIR_ROUND_BCAST_SCHEMA,
    WHIR_GROUP_CLAIM_SCHEMA,
    WHIR_QUERY_LEAF_SUM_SCHEMA,
    WHIR_QUERY_CHAIN_SCHEMA,
    WHIR_EVAL_CHAIN_SCHEMA,
    WHIR_LEAF_CHAIN_SCHEMA,
    WHIR_QUERY_INIT_SCHEMA,
    WHIR_OPENED_EVAL_SCHEMA,
    WHIR_ROUND_CHAIN_SCHEMA,
    CONSTRAINT_PROGRAM_SCHEMA,
    CONSTRAINT_ROOT_TABLE_SCHEMA,
    STATEMENT_SCALAR_CHAIN_SCHEMA,
    STATEMENT_DIGEST_CHAIN_SCHEMA,
    STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA,
    STATEMENT_CONFIG_SCHEMA,
    CONSTRAINT_HEIGHT_INVERSE_SCHEMA,
    CONSTRAINT_NODE_VALUE_SCHEMA,
    CONSTRAINT_CHALLENGE_SCHEMA,
    CONSTRAINT_FOLD_CHAIN_SCHEMA,
    CONSTRAINT_EQ_CHAIN_SCHEMA,
    STATEMENT_VK_DIGEST_SCHEMA,
    STATEMENT_HASH_CHAIN_SCHEMA,
    STATEMENT_CHILD_FACTS_SCHEMA,
    BETA_LADDER_CHAIN_SCHEMA,
    WHIR_FINAL_ROOT_CHAIN_SCHEMA,
    WHIR_LEAF_POW_SEED_SCHEMA,
    SPONGE_ABSORB_WINDOW_SCHEMA,
    SPONGE_SQUEEZE_WINDOW_LO_SCHEMA,
    SPONGE_SQUEEZE_WINDOW_HI_SCHEMA,
    CONSTRAINT_FOLD_PLAN_CHAIN_SCHEMA,
    PROOF_SHAPE_GLOBAL_PACKED_SCHEMA,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionRegisteredInteractionBudget {
    pub schema: RecursionInteractionSchema,
    pub num_sends: usize,
    pub num_receives: usize,
    pub log_height: usize,
}

impl RecursionRegisteredInteractionBudget {
    pub const fn new(
        schema: RecursionInteractionSchema,
        num_sends: usize,
        num_receives: usize,
        log_height: usize,
    ) -> Self {
        Self { schema, num_sends, num_receives, log_height }
    }

    pub const fn budget(&self) -> RecursionInteractionBudget {
        RecursionInteractionBudget::new(self.num_sends, self.num_receives, self.log_height)
    }
}

pub fn validate_recursion_interaction_registry() {
    for (i, left) in NATIVE_RECURSION_SCHEMAS.iter().enumerate() {
        for right in &NATIVE_RECURSION_SCHEMAS[i + 1..] {
            assert_ne!(
                left.interaction_idx, right.interaction_idx,
                "duplicate native recursion interaction index"
            );
        }
        match left.index_space {
            RecursionInteractionIndexSpace::Global => {
                assert!(left.interaction_idx.0 >= GLOBAL_RECURSION_INTERACTION_IDX_START);
                assert!(left.interaction_idx.0 < PER_PROOF_RECURSION_INTERACTION_IDX_START);
            }
            RecursionInteractionIndexSpace::PerProof => {
                assert!(left.interaction_idx.0 >= PER_PROOF_RECURSION_INTERACTION_IDX_START);
            }
        }
    }
}

pub fn validate_registered_recursion_interaction_budget<F>(
    chips: impl IntoIterator<Item = RecursionRegisteredInteractionBudget>,
) -> Result<u64, RecursionInteractionBudgetError>
where
    F: PrimeField64,
{
    validate_recursion_interaction_registry();
    validate_recursion_interaction_budget::<F>(chips.into_iter().map(|chip| chip.budget()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_part_a_registry_slots_are_exact() {
        validate_recursion_interaction_registry();
        let expected = [
            (STATEMENT_SCALAR_CHAIN_SCHEMA, 8, RecursionInteractionIndexSpace::Global, 12),
            (STATEMENT_DIGEST_CHAIN_SCHEMA, 9, RecursionInteractionIndexSpace::Global, 10),
            (
                STATEMENT_GLOBAL_INTERVAL_CHAIN_SCHEMA,
                15,
                RecursionInteractionIndexSpace::Global,
                13,
            ),
            (STATEMENT_CONFIG_SCHEMA, 12, RecursionInteractionIndexSpace::Global, 9),
            (STATEMENT_VK_DIGEST_SCHEMA, 1038, RecursionInteractionIndexSpace::PerProof, 9),
            (STATEMENT_HASH_CHAIN_SCHEMA, 1039, RecursionInteractionIndexSpace::PerProof, 18),
            (STATEMENT_CHILD_FACTS_SCHEMA, 1040, RecursionInteractionIndexSpace::PerProof, 3),
        ];
        for (schema, idx, space, arity) in expected {
            assert_eq!(schema.interaction_idx.0, idx);
            assert_eq!(schema.index_space, space);
            assert_eq!(schema.payload_arity, arity);
        }
    }

    #[test]
    fn merkle_leaf_block_registry_slot_and_beta_floor_are_exact() {
        validate_recursion_interaction_registry();
        assert_eq!(MERKLE_LEAF_BLOCK_SCHEMA.interaction_idx.0, 1004);
        assert_eq!(MERKLE_LEAF_BLOCK_SCHEMA.index_space, RecursionInteractionIndexSpace::PerProof);
        assert_eq!(MERKLE_LEAF_BLOCK_SCHEMA.payload_arity, 13);
        assert_eq!(MERKLE_LEAF_BLOCK_SCHEMA.required_max_beta_power_floor(), 15);
    }

    #[test]
    fn packed_global_registry_slot_and_beta_floor_are_exact() {
        validate_recursion_interaction_registry();
        assert_eq!(PROOF_SHAPE_GLOBAL_PACKED_SCHEMA.interaction_idx.0, 1050);
        assert_eq!(
            PROOF_SHAPE_GLOBAL_PACKED_SCHEMA.index_space,
            RecursionInteractionIndexSpace::PerProof
        );
        assert_eq!(PROOF_SHAPE_GLOBAL_PACKED_SCHEMA.payload_arity, 3);
        // interaction id + proof id + three payload extension elements.
        assert_eq!(PROOF_SHAPE_GLOBAL_PACKED_SCHEMA.required_max_beta_power_floor(), 13);
    }
}
