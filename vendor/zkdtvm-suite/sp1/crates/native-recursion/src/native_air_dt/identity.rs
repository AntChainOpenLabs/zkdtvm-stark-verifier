use crate::{
    machine_dt::{NativeRecursionAssemblyError, NativeRecursionAssemblyResult},
    symbolic_expr_fixed_dt::RecursionChildRole,
    system_dt::RecursionStatementRole,
};

/// Persistent authority for the native AIR family/layer wire-name registry.
pub const NATIVE_AIR_REGISTRY_VERSION: u32 =
    dt_stark::global_d11::GLOBAL146_NATIVE_AIR_REGISTRY_VERSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NativeRecursionLayer {
    L1Lift,
    L2Reduce,
    L3Reduce,
    L4Root,
}

impl NativeRecursionLayer {
    pub const ALL: [Self; 4] = [Self::L1Lift, Self::L2Reduce, Self::L3Reduce, Self::L4Root];

    pub fn from_roles(
        child_role: RecursionChildRole,
        statement_role: RecursionStatementRole,
    ) -> NativeRecursionAssemblyResult<Self> {
        match (child_role, statement_role) {
            (RecursionChildRole::Core, RecursionStatementRole::Lift) => Ok(Self::L1Lift),
            (RecursionChildRole::Compress, RecursionStatementRole::ReduceL2) => Ok(Self::L2Reduce),
            (RecursionChildRole::Compress, RecursionStatementRole::ReduceL3) => Ok(Self::L3Reduce),
            (RecursionChildRole::Shrink, RecursionStatementRole::RootShrink) => Ok(Self::L4Root),
            _ => Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                "invalid native recursion role pair: child={child_role:?} statement={statement_role:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NativeAirFamily {
    TranscriptSponge,
    MerklePath,
    Poseidon2Permute,
    ProofHeightSet,
    WhirTwiddleTable,
    WhirSampleBand,
    WhirQueryFold,
    WhirLeafStream,
    WhirLeafExtStream,
    Range8,
    Range21,

    ProofShapeBinder,
    BatchTranscriptInputs,
    BatchSumcheck,
    WhirRound,
    WhirBatchEval,
    ConstraintTerminal,
    ConstraintBoundary,
    Statement,
    StatementHash,

    NativeChipMetadata,
    ConstraintProgramTable,
    ConstraintRootTable,
    ConstraintDagEval,
    ConstraintFold,
    ConstraintBetaLadder,
    ConstraintChallenge,
    StatementConfig,
}

impl NativeAirFamily {
    pub const ALL: [Self; 28] = [
        Self::TranscriptSponge,
        Self::MerklePath,
        Self::Poseidon2Permute,
        Self::ProofHeightSet,
        Self::WhirTwiddleTable,
        Self::WhirSampleBand,
        Self::WhirQueryFold,
        Self::WhirLeafStream,
        Self::WhirLeafExtStream,
        Self::Range8,
        Self::Range21,
        Self::ProofShapeBinder,
        Self::BatchTranscriptInputs,
        Self::BatchSumcheck,
        Self::WhirRound,
        Self::WhirBatchEval,
        Self::ConstraintTerminal,
        Self::ConstraintBoundary,
        Self::Statement,
        Self::StatementHash,
        Self::NativeChipMetadata,
        Self::ConstraintProgramTable,
        Self::ConstraintRootTable,
        Self::ConstraintDagEval,
        Self::ConstraintFold,
        Self::ConstraintBetaLadder,
        Self::ConstraintChallenge,
        Self::StatementConfig,
    ];
}

pub const SHARED_AIR_FAMILIES: [NativeAirFamily; 11] = [
    NativeAirFamily::TranscriptSponge,
    NativeAirFamily::MerklePath,
    NativeAirFamily::Poseidon2Permute,
    NativeAirFamily::ProofHeightSet,
    NativeAirFamily::WhirTwiddleTable,
    NativeAirFamily::WhirSampleBand,
    NativeAirFamily::WhirQueryFold,
    NativeAirFamily::WhirLeafStream,
    NativeAirFamily::WhirLeafExtStream,
    NativeAirFamily::Range8,
    NativeAirFamily::Range21,
];

pub const SYMBOLIC_SENSITIVE_AIR_FAMILIES: [NativeAirFamily; 10] = [
    NativeAirFamily::ProofShapeBinder,
    NativeAirFamily::BatchTranscriptInputs,
    NativeAirFamily::BatchSumcheck,
    NativeAirFamily::WhirRound,
    NativeAirFamily::WhirBatchEval,
    NativeAirFamily::ConstraintTerminal,
    NativeAirFamily::ConstraintBoundary,
    NativeAirFamily::ConstraintChallenge,
    NativeAirFamily::Statement,
    NativeAirFamily::StatementHash,
];

pub const PROGRAM_SENSITIVE_AIR_FAMILIES: [NativeAirFamily; 7] = [
    NativeAirFamily::NativeChipMetadata,
    NativeAirFamily::ConstraintProgramTable,
    NativeAirFamily::ConstraintRootTable,
    NativeAirFamily::ConstraintDagEval,
    NativeAirFamily::ConstraintFold,
    NativeAirFamily::ConstraintBetaLadder,
    NativeAirFamily::StatementConfig,
];

pub const LAYER_AIR_FAMILIES: [NativeAirFamily; 17] = [
    NativeAirFamily::ProofShapeBinder,
    NativeAirFamily::BatchTranscriptInputs,
    NativeAirFamily::BatchSumcheck,
    NativeAirFamily::WhirRound,
    NativeAirFamily::WhirBatchEval,
    NativeAirFamily::ConstraintTerminal,
    NativeAirFamily::ConstraintBoundary,
    NativeAirFamily::Statement,
    NativeAirFamily::StatementHash,
    NativeAirFamily::NativeChipMetadata,
    NativeAirFamily::ConstraintProgramTable,
    NativeAirFamily::ConstraintRootTable,
    NativeAirFamily::ConstraintDagEval,
    NativeAirFamily::ConstraintFold,
    NativeAirFamily::ConstraintBetaLadder,
    NativeAirFamily::ConstraintChallenge,
    NativeAirFamily::StatementConfig,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeAirId {
    pub family: NativeAirFamily,
    pub layer: Option<NativeRecursionLayer>,
}

impl NativeAirId {
    /// The canonical wire name is a pure function of family and semantic layer.
    pub fn wire_name(self) -> &'static str {
        match (self.family, self.layer) {
            (NativeAirFamily::TranscriptSponge, None) => "NativeTranscriptSponge",
            (NativeAirFamily::MerklePath, None) => "NativeMerklePath",
            (NativeAirFamily::Poseidon2Permute, None) => "NativePoseidon2Permute",
            (NativeAirFamily::ProofHeightSet, None) => "NativeProofHeightSet",
            (NativeAirFamily::WhirTwiddleTable, None) => "WhirTwiddleTable",
            (NativeAirFamily::WhirSampleBand, None) => "WhirSampleBand",
            (NativeAirFamily::WhirQueryFold, None) => "WhirQueryFold",
            (NativeAirFamily::WhirLeafStream, None) => "WhirLeafStream",
            (NativeAirFamily::WhirLeafExtStream, None) => "WhirLeafExtStream",
            (NativeAirFamily::Range8, None) => "NativeRangeChecker8",
            (NativeAirFamily::Range21, None) => "NativeRangeChecker21",

            (NativeAirFamily::ProofShapeBinder, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ProofShapeBinder"
            }
            (NativeAirFamily::BatchTranscriptInputs, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1BatchTranscriptInputs"
            }
            (NativeAirFamily::BatchSumcheck, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1BatchSumcheck"
            }
            (NativeAirFamily::WhirRound, Some(NativeRecursionLayer::L1Lift)) => "NativeL1WhirRound",
            (NativeAirFamily::WhirBatchEval, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1WhirBatchEval"
            }
            (NativeAirFamily::ConstraintTerminal, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintTerminal"
            }
            (NativeAirFamily::ConstraintBoundary, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintBoundary"
            }
            (NativeAirFamily::Statement, Some(NativeRecursionLayer::L1Lift)) => "NativeL1Statement",
            (NativeAirFamily::StatementHash, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1StatementHash"
            }
            (NativeAirFamily::NativeChipMetadata, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1NativeChipMetadata"
            }
            (NativeAirFamily::ConstraintProgramTable, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintProgramTable"
            }
            (NativeAirFamily::ConstraintRootTable, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintRootTable"
            }
            (NativeAirFamily::ConstraintDagEval, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintDagEval"
            }
            (NativeAirFamily::ConstraintFold, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintFold"
            }
            (NativeAirFamily::ConstraintBetaLadder, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintBetaLadder"
            }
            (NativeAirFamily::ConstraintChallenge, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1ConstraintChallenge"
            }
            (NativeAirFamily::StatementConfig, Some(NativeRecursionLayer::L1Lift)) => {
                "NativeL1StatementConfig"
            }

            (NativeAirFamily::ProofShapeBinder, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ProofShapeBinder"
            }
            (NativeAirFamily::BatchTranscriptInputs, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2BatchTranscriptInputs"
            }
            (NativeAirFamily::BatchSumcheck, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2BatchSumcheck"
            }
            (NativeAirFamily::WhirRound, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2WhirRound"
            }
            (NativeAirFamily::WhirBatchEval, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2WhirBatchEval"
            }
            (NativeAirFamily::ConstraintTerminal, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintTerminal"
            }
            (NativeAirFamily::ConstraintBoundary, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintBoundary"
            }
            (NativeAirFamily::Statement, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2Statement"
            }
            (NativeAirFamily::StatementHash, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2StatementHash"
            }
            (NativeAirFamily::NativeChipMetadata, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2NativeChipMetadata"
            }
            (NativeAirFamily::ConstraintProgramTable, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintProgramTable"
            }
            (NativeAirFamily::ConstraintRootTable, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintRootTable"
            }
            (NativeAirFamily::ConstraintDagEval, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintDagEval"
            }
            (NativeAirFamily::ConstraintFold, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintFold"
            }
            (NativeAirFamily::ConstraintBetaLadder, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintBetaLadder"
            }
            (NativeAirFamily::ConstraintChallenge, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2ConstraintChallenge"
            }
            (NativeAirFamily::StatementConfig, Some(NativeRecursionLayer::L2Reduce)) => {
                "NativeL2StatementConfig"
            }

            (NativeAirFamily::ProofShapeBinder, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ProofShapeBinder"
            }
            (NativeAirFamily::BatchTranscriptInputs, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3BatchTranscriptInputs"
            }
            (NativeAirFamily::BatchSumcheck, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3BatchSumcheck"
            }
            (NativeAirFamily::WhirRound, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3WhirRound"
            }
            (NativeAirFamily::WhirBatchEval, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3WhirBatchEval"
            }
            (NativeAirFamily::ConstraintTerminal, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintTerminal"
            }
            (NativeAirFamily::ConstraintBoundary, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintBoundary"
            }
            (NativeAirFamily::Statement, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3Statement"
            }
            (NativeAirFamily::StatementHash, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3StatementHash"
            }
            (NativeAirFamily::NativeChipMetadata, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3NativeChipMetadata"
            }
            (NativeAirFamily::ConstraintProgramTable, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintProgramTable"
            }
            (NativeAirFamily::ConstraintRootTable, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintRootTable"
            }
            (NativeAirFamily::ConstraintDagEval, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintDagEval"
            }
            (NativeAirFamily::ConstraintFold, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintFold"
            }
            (NativeAirFamily::ConstraintBetaLadder, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintBetaLadder"
            }
            (NativeAirFamily::ConstraintChallenge, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3ConstraintChallenge"
            }
            (NativeAirFamily::StatementConfig, Some(NativeRecursionLayer::L3Reduce)) => {
                "NativeL3StatementConfig"
            }

            (NativeAirFamily::ProofShapeBinder, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ProofShapeBinder"
            }
            (NativeAirFamily::BatchTranscriptInputs, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4BatchTranscriptInputs"
            }
            (NativeAirFamily::BatchSumcheck, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4BatchSumcheck"
            }
            (NativeAirFamily::WhirRound, Some(NativeRecursionLayer::L4Root)) => "NativeL4WhirRound",
            (NativeAirFamily::WhirBatchEval, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4WhirBatchEval"
            }
            (NativeAirFamily::ConstraintTerminal, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintTerminal"
            }
            (NativeAirFamily::ConstraintBoundary, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintBoundary"
            }
            (NativeAirFamily::Statement, Some(NativeRecursionLayer::L4Root)) => "NativeL4Statement",
            (NativeAirFamily::StatementHash, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4StatementHash"
            }
            (NativeAirFamily::NativeChipMetadata, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4NativeChipMetadata"
            }
            (NativeAirFamily::ConstraintProgramTable, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintProgramTable"
            }
            (NativeAirFamily::ConstraintRootTable, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintRootTable"
            }
            (NativeAirFamily::ConstraintDagEval, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintDagEval"
            }
            (NativeAirFamily::ConstraintFold, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintFold"
            }
            (NativeAirFamily::ConstraintBetaLadder, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintBetaLadder"
            }
            (NativeAirFamily::ConstraintChallenge, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4ConstraintChallenge"
            }
            (NativeAirFamily::StatementConfig, Some(NativeRecursionLayer::L4Root)) => {
                "NativeL4StatementConfig"
            }

            (family, layer) => {
                panic!("invalid native AIR identity for wire naming: family={family:?} layer={layer:?}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeProofConfigClass {
    Compress,
    Shrink,
    RootShrink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeChildClass {
    CoreShard,
    Lift,
    L2,
    L3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeFinalReplayLayout {
    SingleBase0,
    DualBase0Base128,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn native_layer_classification_is_exact_11_10_7() {
        let shared = SHARED_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
        let symbolic = SYMBOLIC_SENSITIVE_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
        let program = PROGRAM_SENSITIVE_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
        let layer = LAYER_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
        let all = NativeAirFamily::ALL.into_iter().collect::<BTreeSet<_>>();

        assert_eq!(shared.len(), 11);
        assert_eq!(symbolic.len(), 10);
        assert_eq!(program.len(), 7);
        assert!(shared.is_disjoint(&symbolic));
        assert!(shared.is_disjoint(&program));
        assert!(symbolic.is_disjoint(&program));
        assert_eq!(symbolic.union(&program).copied().collect::<BTreeSet<_>>(), layer);
        assert_eq!(shared.union(&layer).copied().collect::<BTreeSet<_>>(), all);
        assert_eq!(all.len(), NativeAirFamily::ALL.len());
    }

    #[test]
    fn persistent_registry_keeps_exactly_28_families() {
        assert_eq!(NATIVE_AIR_REGISTRY_VERSION, 16);
        assert_eq!(NativeAirFamily::ALL.len(), 28);
        assert!(
            NativeAirFamily::ALL.iter().all(|family| !format!("{family:?}").contains("FoldPlan")),
            "schema 1048 is a fused-Fold chain, not a temporary AIR family"
        );
    }
}
