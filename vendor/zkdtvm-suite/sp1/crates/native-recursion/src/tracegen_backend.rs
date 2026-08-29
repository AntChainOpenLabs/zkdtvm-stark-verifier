//! Backend-neutral ownership contract between semantic preparation and trace generation.

use std::sync::Arc;

use crate::{
    native_air_dt::{NativeAirFamily, NativeRecursionLayer},
    system_dt::{
        FinalizedRecord, ProviderInputLayout, RecursionNativeChipMetadataRequest,
        RecursionPoseidon2Request, RecursionPowerRequest, RecursionRangeRequest, RecursionRecord,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FamilyAuthorityRepresentation {
    CompactEvents,
    ExactUnpaddedRows,
    CaseDerivation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FamilyAuthorityDescriptor {
    pub family: NativeAirFamily,
    pub representation: FamilyAuthorityRepresentation,
}

/// Frozen single-owner choice for all 28 AIR families.
pub(crate) const FAMILY_AUTHORITIES: [FamilyAuthorityDescriptor; 28] = [
    compact(NativeAirFamily::TranscriptSponge),
    compact(NativeAirFamily::MerklePath),
    compact(NativeAirFamily::Poseidon2Permute),
    compact(NativeAirFamily::ProofHeightSet),
    compact(NativeAirFamily::WhirTwiddleTable),
    compact(NativeAirFamily::WhirSampleBand),
    compact(NativeAirFamily::WhirQueryFold),
    compact(NativeAirFamily::WhirLeafStream),
    compact(NativeAirFamily::WhirLeafExtStream),
    compact(NativeAirFamily::Range8),
    compact(NativeAirFamily::Range21),
    compact(NativeAirFamily::ProofShapeBinder),
    compact(NativeAirFamily::BatchTranscriptInputs),
    compact(NativeAirFamily::BatchSumcheck),
    compact(NativeAirFamily::WhirRound),
    compact(NativeAirFamily::WhirBatchEval),
    derived(NativeAirFamily::ConstraintTerminal),
    derived(NativeAirFamily::ConstraintBoundary),
    exact(NativeAirFamily::Statement),
    exact(NativeAirFamily::StatementHash),
    compact(NativeAirFamily::NativeChipMetadata),
    exact(NativeAirFamily::ConstraintProgramTable),
    exact(NativeAirFamily::ConstraintRootTable),
    derived(NativeAirFamily::ConstraintDagEval),
    derived(NativeAirFamily::ConstraintFold),
    derived(NativeAirFamily::ConstraintBetaLadder),
    derived(NativeAirFamily::ConstraintChallenge),
    exact(NativeAirFamily::StatementConfig),
];

const fn compact(family: NativeAirFamily) -> FamilyAuthorityDescriptor {
    FamilyAuthorityDescriptor {
        family,
        representation: FamilyAuthorityRepresentation::CompactEvents,
    }
}

const fn exact(family: NativeAirFamily) -> FamilyAuthorityDescriptor {
    FamilyAuthorityDescriptor {
        family,
        representation: FamilyAuthorityRepresentation::ExactUnpaddedRows,
    }
}

const fn derived(family: NativeAirFamily) -> FamilyAuthorityDescriptor {
    FamilyAuthorityDescriptor {
        family,
        representation: FamilyAuthorityRepresentation::CaseDerivation,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PreparedCounts {
    pub proof_count: usize,
    pub family_count: usize,
    pub matrix_count_upper_bound: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SafeUpperBoundPlan {
    pub descriptor_bytes: usize,
    pub matrix_count_upper_bound: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SemanticSeal {
    generation: u64,
    program_authority: u64,
    layer: NativeRecursionLayer,
    proof_count: usize,
}

/// Opaque, process-local authority carried from semantic admission to a
/// device-resident trace bundle.
///
/// There is intentionally no public constructor and none of the sealed
/// identity fields are exposed individually. Backends may retain and compare
/// the handle, but only the canonical native-recursion preparation path can
/// mint one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TracegenAuthorityHandle(SemanticSeal);

impl TracegenAuthorityHandle {
    pub(crate) const fn from_semantic_seal(seal: SemanticSeal) -> Self {
        Self(seal)
    }

    /// Check public handoff metadata without exposing the sealed program
    /// identity or proof-count authority.
    pub fn matches(self, generation: u64, layer: NativeRecursionLayer) -> bool {
        self.0.generation == generation && self.0.layer == layer
    }

    /// Bind one canonical family slot to the layer sealed by admission.
    pub fn air_id(self, family: NativeAirFamily) -> crate::native_air_dt::NativeAirId {
        let layer = if crate::native_air_dt::SHARED_AIR_FAMILIES.contains(&family) {
            None
        } else {
            Some(self.0.layer)
        };
        crate::native_air_dt::NativeAirId { family, layer }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderInputSeal {
    generation: u64,
    pub segment_count: usize,
    pub entry_count: usize,
    pub retained_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidationSeal(SemanticSeal);

/// Move-only semantic record. It deliberately has no `Clone` implementation.
pub(crate) struct PreparedRecord {
    record: FinalizedRecord,
    counts: PreparedCounts,
    bounds: SafeUpperBoundPlan,
    validation: ValidationSeal,
    semantic_seal: SemanticSeal,
}

impl PreparedRecord {
    pub(crate) fn seal(
        record: FinalizedRecord,
        layer: NativeRecursionLayer,
    ) -> Result<Self, String> {
        let proof_count = record.record().proof_records.len();
        let descriptor_bytes = proof_count
            .checked_mul(core::mem::size_of::<usize>() * 2)
            .ok_or_else(|| "descriptor byte upper bound overflow".to_string())?;
        let counts = PreparedCounts {
            proof_count,
            family_count: FAMILY_AUTHORITIES.len(),
            matrix_count_upper_bound: FAMILY_AUTHORITIES.len(),
        };
        let bounds = SafeUpperBoundPlan {
            descriptor_bytes,
            matrix_count_upper_bound: FAMILY_AUTHORITIES.len(),
        };
        let semantic_seal = SemanticSeal {
            generation: record.generation(),
            program_authority: record.program_authority_identity(),
            layer,
            proof_count,
        };
        Ok(Self {
            record,
            counts,
            bounds,
            validation: ValidationSeal(semantic_seal),
            semantic_seal,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.validation.0 != self.semantic_seal {
            return Err("validation seal does not match semantic seal".to_string());
        }
        if self.record.generation() != self.semantic_seal.generation ||
            self.record.program_authority_identity() != self.semantic_seal.program_authority ||
            self.record.record().proof_records.len() != self.semantic_seal.proof_count
        {
            return Err("prepared record authority changed after semantic seal".to_string());
        }
        Ok(())
    }

    pub(crate) fn record(&self) -> &FinalizedRecord {
        &self.record
    }

    pub(crate) fn counts(&self) -> PreparedCounts {
        self.counts
    }

    pub(crate) fn bounds(&self) -> SafeUpperBoundPlan {
        self.bounds
    }

    pub(crate) fn semantic_seal(&self) -> SemanticSeal {
        self.semantic_seal
    }

    fn into_workspace_parts(
        self,
    ) -> (RecursionRecord, PreparedCounts, SafeUpperBoundPlan, SemanticSeal) {
        (self.record.into_tracegen_record(), self.counts, self.bounds, self.semantic_seal)
    }
}

/// Backend-neutral handoff. Move-only by construction.
pub(crate) struct TracegenInput {
    prepared: PreparedRecord,
    semantic_seal: SemanticSeal,
}

/// Small control-plane admission for the trace generation boundary.
pub(crate) struct TracegenAdmission;

impl TracegenAdmission {
    pub(crate) fn admit(input: &TracegenInput) -> Result<(), String> {
        input.validate()
    }
}

impl TracegenInput {
    pub(crate) fn new(prepared: PreparedRecord) -> Result<Self, String> {
        prepared.validate()?;
        let semantic_seal = prepared.semantic_seal();
        Ok(Self { prepared, semantic_seal })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.prepared.validate()?;
        if self.semantic_seal != self.prepared.semantic_seal() {
            return Err("TracegenInput semantic seal mismatch".to_string());
        }
        Ok(())
    }

    pub(crate) fn record(&self) -> &FinalizedRecord {
        self.prepared.record()
    }

    /// Consume the immutable semantic handoff and create the sole mutable derivation owner.
    pub(crate) fn into_workspace(self) -> Result<TracegenWorkspace, String> {
        self.validate()?;
        let (record, counts, bounds, semantic_seal) = self.prepared.into_workspace_parts();
        let workspace = TracegenWorkspace { record, counts, bounds, semantic_seal };
        workspace.validate()?;
        workspace.record.profile.set_structural_counters([
            ("sealed_semantic_mutation_paths", 0),
            ("tracegen_workspace_derivation_owner_count", 1),
            ("production_dynamic_row_cache_entries", 0),
        ]);
        Ok(workspace)
    }
}

/// Mutable, request-local derivation owner created by consuming one [`TracegenInput`].
///
/// The sealed input no longer exists after this transition, and this type is deliberately neither
/// `Clone` nor serializable. Errors drop the complete workspace, so no partial trace bundle can be
/// published.
pub(crate) struct TracegenWorkspace {
    record: RecursionRecord,
    counts: PreparedCounts,
    bounds: SafeUpperBoundPlan,
    semantic_seal: SemanticSeal,
}

impl TracegenWorkspace {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.record.proof_records.len() != self.semantic_seal.proof_count ||
            self.counts.proof_count != self.semantic_seal.proof_count ||
            self.counts.family_count != FAMILY_AUTHORITIES.len() ||
            self.counts.matrix_count_upper_bound != self.bounds.matrix_count_upper_bound ||
            self.bounds.matrix_count_upper_bound != FAMILY_AUTHORITIES.len()
        {
            return Err("tracegen workspace semantic seal mismatch".to_string());
        }
        let expected_descriptor_bytes = self
            .semantic_seal
            .proof_count
            .checked_mul(core::mem::size_of::<usize>() * 2)
            .ok_or_else(|| "tracegen workspace descriptor byte bound overflow".to_string())?;
        if self.bounds.descriptor_bytes != expected_descriptor_bytes {
            return Err("tracegen workspace descriptor bound changed after seal".to_string());
        }
        Ok(())
    }

    pub(crate) fn record(&self) -> &RecursionRecord {
        &self.record
    }

    pub(crate) fn record_mut(&mut self) -> &mut RecursionRecord {
        &mut self.record
    }

    pub(crate) fn into_record(self) -> RecursionRecord {
        self.record
    }

    pub(crate) fn generation(&self) -> u64 {
        self.semantic_seal.generation
    }

    pub(crate) fn authority_handle(&self) -> TracegenAuthorityHandle {
        TracegenAuthorityHandle::from_semantic_seal(self.semantic_seal)
    }

    /// Move the source-captured sponge blocks into their one post-seal owner without cloning rows
    /// or replaying the transcript. This runs after all semantic workspace mutations and before
    /// exact admission/matrix population.
    pub(crate) fn install_transcript_owner(&mut self) -> Result<(), String> {
        if self.record.tracegen_artifacts.transcript_sponge.get().is_some() {
            return Err("transcript workspace owner was installed more than once".to_string());
        }
        let block_count = self
            .record
            .proof_records
            .iter()
            .map(|proof| proof.transcript.sponge_blocks.len())
            .sum();
        let mut transcript_blocks = Vec::with_capacity(block_count);
        for proof in &mut self.record.proof_records {
            if proof.transcript.sponge_blocks.is_empty() {
                return Err(format!(
                    "sealed proof {} lost its source-captured transcript blocks",
                    proof.proof_idx
                ));
            }
            transcript_blocks.append(&mut proof.transcript.sponge_blocks);
        }
        self.record
            .tracegen_artifacts
            .transcript_sponge
            .set(Arc::from(transcript_blocks))
            .map_err(|_| "transcript workspace owner was installed more than once".to_string())?;
        self.record.profile.set_structural_counters([
            ("transcript_source_exact_rows_retained_after_workspace_transition", 0),
            ("transcript_workspace_exact_rows", u64::try_from(block_count).unwrap_or(u64::MAX)),
        ]);
        Ok(())
    }

    /// Seal provider descriptors after source publication and before the single canonical
    /// reduction. This is O(family-count) and never inspects, hashes, sorts, or copies payloads.
    pub(crate) fn seal_provider_inputs(&self) -> Result<ProviderInputSeal, String> {
        self.validate()?;
        let families = provider_families(self.record.provider_input_layout());
        let segment_count = checked_sum(
            families.iter().map(|family| family.segment_count),
            "provider segment count",
        )?;
        let entry_count =
            checked_sum(families.iter().map(|family| family.entry_count), "provider entry count")?;
        let retained_bytes = families
            .iter()
            .zip([
                core::mem::size_of::<RecursionNativeChipMetadataRequest>(),
                core::mem::size_of::<RecursionPoseidon2Request>(),
                core::mem::size_of::<RecursionRangeRequest>(),
                core::mem::size_of::<RecursionPowerRequest>(),
            ])
            .try_fold(0usize, |total, (family, entry_size)| {
                family
                    .entry_count
                    .checked_mul(entry_size)
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| "provider retained-byte count overflow".to_string())
            })?;
        Ok(ProviderInputSeal {
            generation: self.generation(),
            segment_count,
            entry_count,
            retained_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReducedBufferLeaseId {
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TracegenReductionSummary {
    pub raw_entry_count: u64,
    pub exact_unique_count: u64,
    pub duplicate_count: u64,
    pub exact_multiplicity_sum: u64,
    pub overflow: bool,
    pub reduce_passes: u8,
    pub input_seal: ProviderInputSeal,
    pub reduced_buffer_lease: ReducedBufferLeaseId,
}

impl TracegenReductionSummary {
    pub(crate) fn new(
        workspace: &TracegenWorkspace,
        input_seal: ProviderInputSeal,
        raw_entry_count: u64,
        exact_unique_count: u64,
        duplicate_count: u64,
        exact_multiplicity_sum: u64,
        reduce_passes: u8,
    ) -> Self {
        Self {
            raw_entry_count,
            exact_unique_count,
            duplicate_count,
            exact_multiplicity_sum,
            overflow: false,
            reduce_passes,
            input_seal,
            reduced_buffer_lease: ReducedBufferLeaseId { generation: workspace.generation() },
        }
    }

    pub(crate) fn validate(
        &self,
        workspace: &TracegenWorkspace,
        expected_seal: ProviderInputSeal,
    ) -> Result<(), String> {
        self.validate_control_binding(expected_seal, workspace.generation())?;
        self.validate_arithmetic()?;
        if u64::try_from(self.input_seal.entry_count).ok() != Some(self.raw_entry_count) {
            return Err("provider reduction summary input-count mismatch".to_string());
        }
        let reduced_families = provider_families(workspace.record().provider_input_layout());
        let reduced_unique = reduced_families.iter().try_fold(0u64, |total, family| {
            total.checked_add(u64::try_from(family.entry_count).ok()?)
        });
        if reduced_unique != Some(self.exact_unique_count) {
            return Err(
                "provider reduction summary reduced-buffer unique count mismatch".to_string()
            );
        }
        let reduced_multiplicity = [
            workspace.record().native_chip_metadata.total_count(),
            workspace.record().poseidon2.total_count(),
            workspace.record().range.total_count(),
            workspace.record().pow.total_count(),
        ]
        .into_iter()
        .try_fold(0u64, |total, count| total.checked_add(count));
        if reduced_multiplicity != Some(self.exact_multiplicity_sum) {
            return Err(
                "provider reduction summary reduced-buffer multiplicity mismatch".to_string()
            );
        }
        Ok(())
    }

    fn validate_control_binding(
        &self,
        expected_seal: ProviderInputSeal,
        expected_generation: u64,
    ) -> Result<(), String> {
        if self.input_seal != expected_seal ||
            expected_seal.generation != expected_generation ||
            self.reduced_buffer_lease.generation != expected_generation
        {
            return Err("provider reduction summary input seal/buffer lease mismatch".to_string());
        }
        Ok(())
    }

    fn validate_arithmetic(&self) -> Result<(), String> {
        if self.overflow || self.reduce_passes != 1 {
            return Err("provider reduction summary overflow/pass-count mismatch".to_string());
        }
        if self.raw_entry_count < self.exact_unique_count ||
            self.raw_entry_count - self.exact_unique_count != self.duplicate_count
        {
            return Err(
                "provider reduction summary raw/unique/duplicate counts mismatch".to_string()
            );
        }
        if self.exact_multiplicity_sum < self.raw_entry_count {
            return Err(
                "provider reduction summary multiplicity sum is below raw entries".to_string()
            );
        }
        Ok(())
    }
}

fn provider_families(layout: ProviderInputLayout) -> [crate::system_dt::ProviderSegmentSummary; 4] {
    layout.families
}

fn checked_sum(
    values: impl IntoIterator<Item = usize>,
    description: &'static str,
) -> Result<usize, String> {
    values.into_iter().try_fold(0usize, |total, value| {
        total.checked_add(value).ok_or_else(|| format!("{description} overflow"))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn every_air_family_has_one_frozen_authority_representation() {
        let actual = FAMILY_AUTHORITIES.iter().map(|entry| entry.family).collect::<BTreeSet<_>>();
        let expected = NativeAirFamily::ALL.into_iter().collect::<BTreeSet<_>>();
        assert_eq!(FAMILY_AUTHORITIES.len(), NativeAirFamily::ALL.len());
        assert_eq!(actual, expected);
    }

    fn valid_reduction_summary() -> TracegenReductionSummary {
        let seal = ProviderInputSeal {
            generation: 7,
            segment_count: 8,
            entry_count: 10,
            retained_bytes: 160,
        };
        TracegenReductionSummary {
            raw_entry_count: 10,
            exact_unique_count: 6,
            duplicate_count: 4,
            exact_multiplicity_sum: 17,
            overflow: false,
            reduce_passes: 1,
            input_seal: seal,
            reduced_buffer_lease: ReducedBufferLeaseId { generation: 7 },
        }
    }

    #[test]
    fn reduction_summary_arithmetic_fails_closed() {
        let summary = valid_reduction_summary();
        assert!(summary.validate_arithmetic().is_ok());

        let mut corrupt = summary;
        corrupt.duplicate_count += 1;
        assert!(corrupt.validate_arithmetic().is_err());

        let mut corrupt = summary;
        corrupt.exact_multiplicity_sum = corrupt.raw_entry_count - 1;
        assert!(corrupt.validate_arithmetic().is_err());

        let mut corrupt = summary;
        corrupt.reduce_passes = 2;
        assert!(corrupt.validate_arithmetic().is_err());

        let mut corrupt = summary;
        corrupt.overflow = true;
        assert!(corrupt.validate_arithmetic().is_err());
    }

    #[test]
    fn reduction_summary_seals_and_buffer_lease_fail_closed() {
        let summary = valid_reduction_summary();
        assert!(summary.validate_control_binding(summary.input_seal, 7).is_ok());

        let mut wrong_seal = summary.input_seal;
        wrong_seal.entry_count += 1;
        assert!(summary.validate_control_binding(wrong_seal, 7).is_err());
        assert!(summary.validate_control_binding(summary.input_seal, 8).is_err());
    }
}
