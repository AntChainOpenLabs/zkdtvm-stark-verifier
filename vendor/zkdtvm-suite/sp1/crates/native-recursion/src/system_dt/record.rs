use core::marker::PhantomData;
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use dt_stark::{air::MachineProgram, sumcheck::proof::SCShardOpenedValues, MachineRecord};
use hashbrown::HashMap;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::Dimensions;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

use super::{spec_fold::WhirSpecFoldShape, spec_sponge::SpecSpongeBlock};

use crate::{
    config::{ChildMlPcsOpeningProof, DIGEST_SIZE, EF, F, POSEIDON2_WIDTH},
    constraint_replay_dt::trace::ConstraintCaseArtifact,
    native_air_dt::{NativeAirFamily, NativeFinalReplayLayout, NativeRecursionLayer},
    primitives_dt::pow::PowerCheckerCounts,
    statement_boundary_air_dt::StatementBoundaryRow,
    statement_dt::{
        NativeRecursionPublicValues, SpecStatement, SpecStatementError,
        NATIVE_RECURSION_NUM_PV_ELTS,
    },
    statement_hash_air_dt::{StatementDigestMode, StatementHashRow},
    symbolic_expr_fixed_dt::RecursionChildRole,
    symbolic_ir_dt::{
        validate_constraint_program_dto, RecursionPolyAirVerifierProgram,
        RecursionPolyAirVerifierProgramDto,
    },
    transcript_dt::poseidon2::{
        RecursionPoseidon2Memo, RecursionPoseidon2Output, RecursionPoseidon2TracegenCache,
    },
    whir_dt::columns::WHIR_FINAL_ROOT_POSEIDON2_PERMS,
};
use crate::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderSegmentSummary {
    pub segment_count: usize,
    pub entry_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderInputLayout {
    pub families: [ProviderSegmentSummary; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderReductionStats {
    pub raw_entries: usize,
    pub unique_entries: usize,
    pub duplicate_entries: usize,
    pub passes: u8,
}

/// Read-only, canonical provider input snapshot for differential/oracle
/// tooling. Segment order and entry order are preserved exactly. This is not
/// used by production trace generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecursionProviderOracleSnapshot {
    pub metadata_segments: Vec<Vec<RecursionNativeChipMetadataRequest>>,
    pub poseidon2_segments: Vec<Vec<RecursionPoseidon2Request>>,
    pub range_segments: Vec<Vec<RecursionRangeRequest>>,
    pub power_segments: Vec<Vec<RecursionPowerRequest>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RecursionProfileCounter {
    pub count: u64,
    pub ms: u128,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RecursionPoseidon2MemoCounter {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RecursionRecordProfileSnapshot {
    pub record_splits: BTreeMap<String, RecursionProfileCounter>,
    pub poseidon2_memo: BTreeMap<String, RecursionPoseidon2MemoCounter>,
    pub structural_counters: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default)]
pub struct RecursionRecordProfile {
    inner: Arc<Mutex<RecursionRecordProfileSnapshot>>,
    prepare_starts: Arc<Mutex<Option<(Instant, Instant)>>>,
}

impl PartialEq for RecursionRecordProfile {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for RecursionRecordProfile {}

impl RecursionRecordProfile {
    pub fn mark_prepare_started(&self, started: Instant) {
        let mut starts = self.prepare_starts.lock().expect("record prepare-window lock");
        match starts.as_mut() {
            Some((earliest, latest)) => {
                *earliest = (*earliest).min(started);
                *latest = (*latest).max(started);
            }
            None => *starts = Some((started, started)),
        }
    }

    /// Returns `(first_proof_ready, last_proof_ready)` elapsed times to the
    /// supplied TracegenInput seal instant.
    pub fn prepare_elapsed_to(&self, sealed: Instant) -> Option<(u128, u128)> {
        self.prepare_starts.lock().expect("record prepare-window lock").map(|(earliest, latest)| {
            (
                sealed.saturating_duration_since(earliest).as_millis(),
                sealed.saturating_duration_since(latest).as_millis(),
            )
        })
    }

    pub fn add_record_split(&self, label: impl Into<String>, ms: u128) {
        add_profile_counter(&mut self.inner.lock().expect("record profile lock"), label.into(), ms);
    }

    pub fn add_poseidon2_memo_delta(&self, label: impl Into<String>, hits: u64, misses: u64) {
        if hits == 0 && misses == 0 {
            return;
        }
        let mut snapshot = self.inner.lock().expect("record profile lock");
        let entry = snapshot.poseidon2_memo.entry(label.into()).or_default();
        entry.hits = entry.hits.checked_add(hits).expect("poseidon2 memo hits overflow");
        entry.misses = entry.misses.checked_add(misses).expect("poseidon2 memo misses overflow");
    }

    pub fn add_structural_counter(&self, label: impl Into<String>, delta: u64) {
        let mut snapshot = self.inner.lock().expect("record profile lock");
        let value = snapshot.structural_counters.entry(label.into()).or_default();
        *value = value.checked_add(delta).expect("structural counter overflow");
    }

    /// Publish one proof's diagnostic package and seal its ready-to-return timing boundary under
    /// one profile lock. Labels are static so the hot record path does not format per-proof timer
    /// names. The elapsed sample is taken only after the semantic descriptor and all ordinary
    /// synchronous diagnostics have been published; only the self-observing timer fields written
    /// from that sample necessarily follow it.
    pub fn publish_proof_batch_and_seal(
        &self,
        prepare_started: Instant,
        semantic_prepare_us: u64,
        record_splits: &[(&'static str, u128)],
        poseidon2_memo: &[(&'static str, u64, u64)],
        structural_counters: &[(&'static str, u64)],
    ) {
        let diagnostic_started = Instant::now();
        let mut snapshot = self.inner.lock().expect("record profile lock");
        for &(label, ms) in record_splits {
            add_profile_counter(&mut snapshot, label.to_string(), ms);
        }
        for &(label, hits, misses) in poseidon2_memo {
            if hits == 0 && misses == 0 {
                continue;
            }
            let entry = snapshot.poseidon2_memo.entry(label.to_string()).or_default();
            entry.hits = entry.hits.checked_add(hits).expect("poseidon2 memo hits overflow");
            entry.misses =
                entry.misses.checked_add(misses).expect("poseidon2 memo misses overflow");
        }
        for &(label, value) in structural_counters {
            snapshot.structural_counters.insert(label.to_string(), value);
        }

        let diagnostic_publish_us =
            u64::try_from(diagnostic_started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let prepare_us = u64::try_from(prepare_started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let prepare_ms = u128::from(prepare_us) / 1_000;
        for label in [
            "proof_ready_to_event_segment_ready",
            "child_proof_ready_to_prepared_segment_sealed",
            "prepare_work_sum_all_proofs",
            "total_prepare_work_sum",
        ] {
            add_profile_counter(&mut snapshot, label.to_string(), prepare_ms);
        }
        for (label, value) in [
            (
                "child_proof_ready_to_prepared_segment_sealed_max_ms",
                u64::try_from(prepare_ms).unwrap_or(u64::MAX),
            ),
            ("preflight.semantic_us", semantic_prepare_us),
            ("preflight.semantic_max_us", semantic_prepare_us),
            ("preflight.diagnostic_profile_publish_us", diagnostic_publish_us),
            ("preflight.diagnostic_profile_publish_max_us", diagnostic_publish_us),
            ("preflight.total_us", prepare_us),
            ("preflight.total_max_us", prepare_us),
        ] {
            snapshot.structural_counters.insert(label.to_string(), value);
        }
        // These counters are incremented at the actual publication/lock event. Each proof owns a
        // separate profile until ordered node merge, where the ordinary merge path sums them.
        for label in ["profile_batch_publications", "profile_lock_acquisitions"] {
            let value = snapshot.structural_counters.entry(label.to_string()).or_default();
            *value = value.checked_add(1).expect("profile publication counter overflow");
        }
    }

    /// Publish the ordinary TracegenInput admission diagnostics, then sample the node's true
    /// ready-to-sealed boundary while holding the same profile locks. This includes construction,
    /// validation, admission, and synchronous diagnostic publication. Only fields that report the
    /// sampled duration itself are written after the sample.
    pub fn publish_tracegen_input_batch_and_seal(
        &self,
        admit_started: Instant,
        record_splits: &[(&'static str, u128)],
        structural_counters: &[(&'static str, u64)],
    ) -> u128 {
        // Keep the same lock order as `merge_from`: prepare window, then profile map.
        let starts = self.prepare_starts.lock().expect("record prepare-window lock");
        let mut snapshot = self.inner.lock().expect("record profile lock");
        for &(label, ms) in record_splits {
            add_profile_counter(&mut snapshot, label.to_string(), ms);
        }
        for &(label, value) in structural_counters {
            snapshot.structural_counters.insert(label.to_string(), value);
        }
        let admissions = snapshot
            .structural_counters
            .entry("tracegen_input_admission_events".to_string())
            .or_default();
        *admissions = admissions.checked_add(1).expect("TracegenInput admission counter overflow");

        let sealed = Instant::now();
        let admit_ms = sealed.saturating_duration_since(admit_started).as_millis();
        for label in ["tracegen_input_seal", "node_admission_and_tracegen_input_seal"] {
            add_profile_counter(&mut snapshot, label.to_string(), admit_ms);
        }
        if let Some((earliest, latest)) = *starts {
            let total_prepare_ms = sealed.saturating_duration_since(earliest).as_millis();
            let last_ready_ms = sealed.saturating_duration_since(latest).as_millis();
            add_profile_counter(&mut snapshot, "total_t_prepare".to_string(), total_prepare_ms);
            for label in [
                "last_required_proof_ready_to_tracegen_input_sealed",
                "last_required_child_ready_to_tracegen_input_sealed",
            ] {
                add_profile_counter(&mut snapshot, label.to_string(), last_ready_ms);
            }
            snapshot.structural_counters.insert(
                "last_required_child_ready_to_tracegen_input_sealed_max_ms".to_string(),
                u64::try_from(last_ready_ms).unwrap_or(u64::MAX),
            );
        }
        admit_ms
    }

    pub fn merge_from(&self, other: &Self) {
        let other_prepare = *other.prepare_starts.lock().expect("record prepare-window lock");
        if let Some((other_earliest, other_latest)) = other_prepare {
            let mut starts = self.prepare_starts.lock().expect("record prepare-window lock");
            match starts.as_mut() {
                Some((earliest, latest)) => {
                    *earliest = (*earliest).min(other_earliest);
                    *latest = (*latest).max(other_latest);
                }
                None => *starts = Some((other_earliest, other_latest)),
            }
        }
        let other = other.snapshot();
        let mut this = self.inner.lock().expect("record profile lock");
        for (label, counter) in other.record_splits {
            let entry = this.record_splits.entry(label).or_default();
            entry.count += counter.count;
            entry.ms += counter.ms;
        }
        for (label, counter) in other.poseidon2_memo {
            let entry = this.poseidon2_memo.entry(label).or_default();
            entry.hits += counter.hits;
            entry.misses += counter.misses;
        }
        for (label, value) in other.structural_counters {
            let is_maximum = label.ends_with("_max_ms") || label.ends_with("_max_us");
            let entry = this.structural_counters.entry(label).or_default();
            if is_maximum {
                *entry = (*entry).max(value);
            } else {
                *entry = entry.checked_add(value).expect("structural counter overflow");
            }
        }
    }

    pub fn set_structural_counter(&self, label: impl Into<String>, value: u64) {
        self.inner
            .lock()
            .expect("record profile lock")
            .structural_counters
            .insert(label.into(), value);
    }

    pub fn set_structural_counters<I, S>(&self, counters: I)
    where
        I: IntoIterator<Item = (S, u64)>,
        S: Into<String>,
    {
        let mut profile = self.inner.lock().expect("record profile lock");
        profile
            .structural_counters
            .extend(counters.into_iter().map(|(label, value)| (label.into(), value)));
    }

    pub fn structural_counter(&self, label: &str) -> Option<u64> {
        self.inner.lock().expect("record profile lock").structural_counters.get(label).copied()
    }

    pub fn snapshot(&self) -> RecursionRecordProfileSnapshot {
        self.inner.lock().expect("record profile lock").clone()
    }
}

fn add_profile_counter(snapshot: &mut RecursionRecordProfileSnapshot, label: String, ms: u128) {
    let entry = snapshot.record_splits.entry(label).or_default();
    entry.count += 1;
    entry.ms += ms;
}

static NEXT_FINALIZED_RECORD_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_finalized_record_generation() -> RecordGeneration {
    let generation = NEXT_FINALIZED_RECORD_GENERATION.fetch_add(1, Ordering::Relaxed);
    RecordGeneration(
        NonZeroU64::new(generation).expect("finalized recursion record generation overflow"),
    )
}

/// Concrete, request-local artifacts installed only by the tracegen workspace.
///
/// This is deliberately not a generic cache: each slot has one semantic meaning and one owner.
/// Slots are never cleared or invalidated. A semantic mutation after any slot is installed fails
/// immediately, while cloning or deserializing a raw record starts with an empty artifact owner.
#[derive(Debug, Default)]
pub struct TracegenWorkspaceArtifacts {
    pub(crate) transcript_sponge: OnceLock<Arc<[SpecSpongeBlock]>>,
    pub(crate) statement: OnceLock<(u64, Arc<[StatementBoundaryRow]>)>,
    pub(crate) statement_hash: OnceLock<(StatementDigestMode, Arc<[StatementHashRow]>)>,
    pub(crate) constraint_case: OnceLock<(u64, Arc<ConstraintCaseArtifact>)>,
}

impl Clone for TracegenWorkspaceArtifacts {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for TracegenWorkspaceArtifacts {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for TracegenWorkspaceArtifacts {}

impl TracegenWorkspaceArtifacts {
    pub(crate) fn initialized_entries(&self) -> usize {
        usize::from(self.transcript_sponge.get().is_some()) +
            usize::from(self.statement.get().is_some()) +
            usize::from(self.statement_hash.get().is_some()) +
            usize::from(self.constraint_case.get().is_some())
    }

    fn assert_uninitialized(&self) {
        assert_eq!(
            self.initialized_entries(),
            0,
            "semantic recursion record mutated after tracegen artifact installation"
        );
    }
}

/// Process-local provenance allocated exactly once after the last semantic record mutation.
/// It is neither serialized nor derived from record contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RecordGeneration(NonZeroU64);

/// Mutable orchestration state. This wrapper is intentionally non-serde and non-`Clone`: the raw
/// [`RecursionRecord`] remains the persistence DTO, while native proving must consume this value
/// through the one canonical finalization path.
#[derive(Debug)]
pub struct BuildingRecord {
    record: RecursionRecord,
}

impl BuildingRecord {
    pub(crate) fn from_record(record: RecursionRecord) -> Self {
        assert!(
            !record.provider_requests_finalized,
            "provider-finalized records cannot re-enter the building state"
        );
        Self { record }
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

    pub(crate) fn append(&mut self, other: &mut Self) {
        assert!(
            !self.record.provider_requests_finalized && !other.record.provider_requests_finalized,
            "provider-finalized records cannot be merged"
        );
        self.record.append(&mut other.record);
    }

    pub fn set_statement_vk_root(&mut self, vk_root: [F; DIGEST_SIZE]) {
        self.record.mark_semantic_mutation();
        self.record.statement_vk_root = vk_root;
        self.record.statement_public_values = None;
    }

    pub fn set_statement_is_complete(&mut self, is_complete: bool) {
        self.record.mark_semantic_mutation();
        self.record.statement_is_complete = is_complete;
        self.record.statement_public_values = None;
    }
}

/// Immutable semantic proof input. Provider finalization intentionally occurs
/// later, after tracegen has expanded compact sources.
#[derive(Debug)]
pub struct FinalizedRecord {
    record: RecursionRecord,
    generation: RecordGeneration,
    program_authority: FinalizedProgramAuthority,
}

impl FinalizedRecord {
    pub(crate) fn from_record(
        record: RecursionRecord,
        program: &RecursionNativeProgram<F>,
        _seal: crate::machine_dt::FinalizationSeal,
    ) -> Self {
        Self {
            record,
            generation: next_finalized_record_generation(),
            program_authority: FinalizedProgramAuthority::from_program(program),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.0.get()
    }

    pub fn program_authority_identity(&self) -> u64 {
        self.program_authority.constraint_program.authority_identity()
    }

    pub fn record(&self) -> &RecursionRecord {
        &self.record
    }

    pub fn into_tracegen_record(self) -> RecursionRecord {
        self.record
    }

    pub fn matches_program(&self, program: &RecursionNativeProgram<F>) -> bool {
        self.program_authority == FinalizedProgramAuthority::from_program(program)
    }
}

/// Exact, compact authority for the program used by canonical finalization. The constraint
/// program's request-owned identity pins its full IR without cloning it; the remaining fields pin
/// the statement layer, whose small configuration is independent of that IR.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalizedProgramAuthority {
    constraint_program: RecursionPolyAirVerifierProgram,
    role: RecursionChildRole,
    statement_role: RecursionStatementRole,
    num_child_public_values: usize,
    child_contains_global_bus: bool,
    statement_config: Vec<StatementConfigRow>,
}

impl FinalizedProgramAuthority {
    fn from_program(program: &RecursionNativeProgram<F>) -> Self {
        Self {
            constraint_program: program.constraint_program.clone(),
            role: program.role,
            statement_role: program.statement_role,
            num_child_public_values: program.num_child_public_values,
            child_contains_global_bus: program.child_contains_global_bus,
            statement_config: program.statement_config.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionRecord {
    #[serde(skip)]
    pub(crate) tracegen_artifacts: TracegenWorkspaceArtifacts,
    #[serde(skip)]
    pub profile: RecursionRecordProfile,
    /// Request-local host computation cache. It is deliberately absent from the persistence DTO:
    /// clones and deserialized records rebuild it from their own computation graph.
    #[serde(skip)]
    pub(crate) poseidon2_memo: RecursionPoseidon2Memo,
    /// Complete permutation columns created only after `TracegenInput` is
    /// admitted. This cache is empty during preparation and is drained into
    /// the Poseidon2 provider matrix during trace generation.
    #[serde(skip)]
    pub(crate) poseidon2_tracegen: RecursionPoseidon2TracegenCache,
    /// True when every dynamic provider consumer has registered its request at the source.
    /// Deserialized or manually assembled records must be finalized explicitly before proving;
    /// there is no compatibility preparation scan.
    #[serde(skip)]
    pub(crate) provider_requests_finalized: bool,
    #[serde(skip)]
    pub(crate) provider_reduce_passes: u8,
    #[serde(default)]
    pub statement_is_complete: bool,
    /// ReduceL2 threaded self-vk input: the node's `vk_root[8]` PV input. Zero for
    /// every other statement role; the honest L2 value is the digest of the node's own vk.
    #[serde(default)]
    pub statement_vk_root: [F; DIGEST_SIZE],
    #[serde(default)]
    pub statement_public_values: Option<NativeRecursionPublicValues<F>>,
    pub proof_records: Vec<RecursionProofRecord>,
    pub native_chip_metadata: RecursionNativeChipMetadataPool,
    pub poseidon2: RecursionPoseidon2Pool,
    pub range: RecursionRangePool,
    // `pow.counts.range` serves only PowerCheckerAir's internal log range proof.
    // External range consumers must route to `range`, not duplicate the same request here.
    pub pow: RecursionPowerPool,
}

impl RecursionRecord {
    pub fn poseidon2_memo_snapshot(
        &self,
    ) -> crate::transcript_dt::poseidon2::RecursionPoseidon2MemoSnapshot {
        self.poseidon2_memo.snapshot()
    }

    /// Clone only when an explicit oracle/debug path needs exact provider
    /// manifests. Production paths use the owned pools directly.
    pub fn provider_oracle_snapshot(&self) -> RecursionProviderOracleSnapshot {
        RecursionProviderOracleSnapshot {
            metadata_segments: oracle_segments(
                &self.native_chip_metadata.segments,
                &self.native_chip_metadata.requests,
            ),
            poseidon2_segments: oracle_segments(&self.poseidon2.segments, &self.poseidon2.requests),
            range_segments: oracle_segments(&self.range.segments, &self.range.requests),
            power_segments: oracle_segments(&self.pow.segments, &self.pow.requests),
        }
    }

    pub(crate) fn provider_input_layout(&self) -> ProviderInputLayout {
        ProviderInputLayout {
            families: [
                self.native_chip_metadata.segment_summary(),
                self.poseidon2.segment_summary(),
                self.range.segment_summary(),
                self.pow.segment_summary(),
            ],
        }
    }

    pub(crate) fn reduce_provider_inputs(&mut self) -> Result<ProviderReductionStats, String> {
        if self.provider_reduce_passes != 0 {
            return Err("provider reduction executed more than once".to_string());
        }
        let raw_entries = self.native_chip_metadata.unique_count() +
            self.poseidon2.unique_count() +
            self.range.unique_count() +
            self.pow.unique_count();
        self.native_chip_metadata.reduce()?;
        self.poseidon2.reduce()?;
        self.range.reduce()?;
        self.pow.reduce()?;
        self.provider_reduce_passes = 1;
        let unique_entries = self.native_chip_metadata.unique_count() +
            self.poseidon2.unique_count() +
            self.range.unique_count() +
            self.pow.unique_count();
        Ok(ProviderReductionStats {
            raw_entries,
            unique_entries,
            duplicate_entries: raw_entries.saturating_sub(unique_entries),
            passes: self.provider_reduce_passes,
        })
    }

    pub fn proof_record_mut(&mut self, proof_idx: usize) -> &mut RecursionProofRecord {
        self.mark_semantic_mutation();
        self.statement_public_values = None;
        if let Some(pos) =
            self.proof_records.iter().position(|record| record.proof_idx == proof_idx)
        {
            return &mut self.proof_records[pos];
        }
        self.proof_records
            .push(RecursionProofRecord { proof_idx, ..RecursionProofRecord::default() });
        self.proof_records.last_mut().expect("inserted proof record")
    }

    pub(crate) fn mark_semantic_mutation(&mut self) {
        self.tracegen_artifacts.assert_uninitialized();
        self.provider_requests_finalized = false;
    }

    pub(crate) fn mark_provider_requests_finalized(&mut self) {
        self.provider_requests_finalized = true;
    }

    pub fn refresh_statement_public_values(
        &mut self,
        program: &RecursionNativeProgram<F>,
    ) -> Result<(), SpecStatementError> {
        self.mark_semantic_mutation();
        let statement = SpecStatement::from_record(self, program)?;
        self.statement_public_values = Some(statement.public_values);
        Ok(())
    }
}

fn oracle_segments<T: Clone>(segments: &[Vec<T>], requests: &[T]) -> Vec<Vec<T>> {
    let mut snapshot = segments.to_vec();
    if !requests.is_empty() {
        snapshot.push(requests.to_vec());
    }
    snapshot
}

impl MachineRecord for RecursionRecord {
    type Config = ();

    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert(
            "native_chip_metadata_unique_requests".to_string(),
            self.native_chip_metadata.unique_count(),
        );
        stats.insert(
            "native_chip_metadata_total_requests".to_string(),
            self.native_chip_metadata.total_count_usize(),
        );
        stats.insert("poseidon2_unique_requests".to_string(), self.poseidon2.unique_count());
        stats.insert("poseidon2_total_requests".to_string(), self.poseidon2.total_count_usize());
        stats.insert("range_unique_requests".to_string(), self.range.unique_count());
        stats.insert("range_total_requests".to_string(), self.range.total_count_usize());
        stats.insert("power_unique_requests".to_string(), self.pow.unique_count());
        stats.insert("power_total_requests".to_string(), self.pow.total_count_usize());
        stats
    }

    fn append(&mut self, other: &mut Self) {
        assert_eq!(self.provider_reduce_passes, 0, "cannot append after provider reduction");
        assert_eq!(other.provider_reduce_passes, 0, "cannot append a reduced provider record");
        let appended_proofs = !other.proof_records.is_empty();
        let appended_metadata = other.native_chip_metadata.unique_count() != 0;
        let appended_poseidon2 = other.poseidon2.unique_count() != 0;
        let appended_range = other.range.unique_count() != 0;
        let appended_pow = other.pow.unique_count() != 0;
        let changed = appended_proofs ||
            appended_metadata ||
            appended_poseidon2 ||
            appended_range ||
            appended_pow;
        if changed {
            self.mark_semantic_mutation();
            other.mark_semantic_mutation();
        }
        self.proof_records.append(&mut other.proof_records);
        if appended_proofs {
            self.statement_public_values = None;
        }
        // Per-proof records come only from the recording run, one per distinct proof_idx;
        // deps-phase chips must never emit them. `append` concatenates (it does not merge by
        // proof_idx), so a duplicate proof_idx here means that invariant was broken — fail loud.
        debug_assert!(
            {
                let mut seen = BTreeSet::new();
                self.proof_records.iter().all(|record| seen.insert(record.proof_idx))
            },
            "duplicate proof_idx after RecursionRecord::append; proof records must be recording-run-only"
        );
        self.native_chip_metadata.append(&mut other.native_chip_metadata);
        self.poseidon2_memo.append(&mut other.poseidon2_memo);
        self.poseidon2.append(&mut other.poseidon2);
        self.range.append(&mut other.range);
        self.pow.append(&mut other.pow);
        self.profile.merge_from(&other.profile);
    }

    fn public_values<AF: AbstractField>(&self) -> Vec<AF> {
        debug_assert_eq!(
            NATIVE_RECURSION_NUM_PV_ELTS, 159,
            "Global-146 native recursion public-value schema changed"
        );
        let Some(public_values) = &self.statement_public_values else {
            return vec![AF::zero(); NATIVE_RECURSION_NUM_PV_ELTS];
        };
        public_values
            .as_array()
            .into_iter()
            .map(|value| AF::from_canonical_u32(value.as_canonical_u32()))
            .collect()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionProofRecord {
    pub proof_idx: usize,
    pub transcript: RecursionTranscriptRecord,
    pub merkle_path: RecursionMerklePathRecord,
    pub constraints: RecursionConstraintRecord,
    pub proof_shape: RecursionProofShapeRecord,
    pub batch_constraint: RecursionBatchConstraintRecord,
    /// Backend-neutral proof material consumed exactly once by tracegen. It
    /// intentionally contains no verifier result and no complete AIR row family.
    #[serde(default)]
    pub whir_source: Option<RecursionWhirTracegenSource>,
    pub whir: RecursionWhirRecord,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RecursionWhirTracegenSource {
    pub shape: WhirSpecFoldShape,
    pub opening_proof: Arc<ChildMlPcsOpeningProof>,
    pub opened_values: Arc<SCShardOpenedValues<F, EF>>,
    pub dimensions: Vec<Vec<Dimensions>>,
    pub input_roots: Vec<[F; DIGEST_SIZE]>,
    pub publish_opened_eval: bool,
    /// Exact external-consumer multiplicities, frozen from the child program
    /// before tracegen. Rows consume this authority at construction;
    /// no post-row publication patch is needed.
    #[serde(default)]
    pub opened_eval_publications: Vec<RecursionWhirOpenedEvalPublication>,
}

impl core::fmt::Debug for RecursionWhirTracegenSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RecursionWhirTracegenSource")
            .field("shape", &self.shape)
            .field("dimension_batches", &self.dimensions.len())
            .field("input_roots", &self.input_roots.len())
            .field("publish_opened_eval", &self.publish_opened_eval)
            .field("opened_eval_publications", &self.opened_eval_publications)
            .finish_non_exhaustive()
    }
}

impl PartialEq for RecursionWhirTracegenSource {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.opening_proof, &other.opening_proof) &&
            Arc::ptr_eq(&self.opened_values, &other.opened_values) &&
            self.shape == other.shape &&
            self.dimensions == other.dimensions &&
            self.input_roots == other.input_roots &&
            self.publish_opened_eval == other.publish_opened_eval &&
            self.opened_eval_publications == other.opened_eval_publications
    }
}

impl Eq for RecursionWhirTracegenSource {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirOpenedEvalPublication {
    pub batch_id: usize,
    pub batch_pos: usize,
    pub chip_idx: usize,
    pub value_idx: usize,
    pub multiplicity: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirRecord {
    pub role_config_mults: [u32; 3],
    pub twiddle_mults: Vec<[u32; 3]>,
    pub round_rows: Vec<RecursionWhirRoundRow>,
    pub batch_eval_rows: Vec<RecursionWhirBatchEvalRow>,
    pub query_fold_rows: Vec<RecursionWhirQueryFoldRow>,
    pub leaf_stream_rows: Vec<RecursionWhirLeafStreamRow>,
    pub leaf_ext_stream_rows: Vec<RecursionWhirLeafExtStreamTraceRow>,
}

impl RecursionWhirRecord {
    pub fn is_empty(&self) -> bool {
        self.role_config_mults.iter().all(|&mult| mult == 0) &&
            self.twiddle_mults.iter().all(|row| row.iter().all(|&mult| mult == 0)) &&
            self.round_rows.is_empty() &&
            self.batch_eval_rows.is_empty() &&
            self.query_fold_rows.is_empty() &&
            self.leaf_stream_rows.is_empty() &&
            self.leaf_ext_stream_rows.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirRoundRow {
    pub proof_idx: usize,
    pub is_pow_batch: bool,
    pub is_preamble: bool,
    pub is_round: bool,
    pub is_final: bool,
    pub round: usize,
    pub tidx: usize,
    pub role_id: usize,
    pub num_queries: usize,
    pub batching_bits: usize,
    pub query_bits: usize,
    pub log_blowup: usize,
    pub r_rounds: usize,
    pub c_chips: usize,
    pub num_public_values: usize,
    pub w_qbase: usize,
    pub opening_idx: usize,
    pub opening_point: [F; 5],
    pub height_group_rank: usize,
    pub height_group_log_height: usize,
    pub group_claim_log_height: usize,
    pub group_claim: [F; 5],
    pub commit_id: usize,
    pub commit_root: [F; 8],
    pub event_value: [F; 32],
    pub event_value_last: F,
    pub pow_sample_high: usize,
    pub round_has_oracle: bool,
    pub chain_recv_round: usize,
    pub chain_recv_tidx: usize,
    pub chain_recv_claim: [F; 5],
    pub chain_recv_eq: [F; 5],
    pub chain_recv_pending_is_merge: bool,
    pub chain_recv_pending_beta: [F; 5],
    pub chain_recv_pending_eq: [F; 5],
    pub chain_send_round: usize,
    pub chain_send_tidx: usize,
    pub chain_send_claim: [F; 5],
    pub chain_send_eq: [F; 5],
    pub chain_send_pending_is_merge: bool,
    pub chain_send_pending_beta: [F; 5],
    pub chain_send_pending_eq: [F; 5],
    pub r_fold: [F; 5],
    pub is_merge: bool,
    pub emit_prep_seed: bool,
    pub merge_log_height: usize,
    pub cfr: [F; 5],
    pub claim_acc: [F; 5],
    pub claim_folded: [F; 5],
    pub eq_factor: [F; 5],
    pub eq_folded: [F; 5],
    pub final_root_poseidon2_inputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub final_root_poseidon2_outputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub bcast_mult: u32,
    pub query_init_mult: u32,
    pub chain_recv_mult: u32,
    pub chain_send_mult: u32,
    pub role_config_recv_mult: u32,
    pub summary_recv_mult: u32,
    #[serde(default)]
    pub summary_id_base: usize,
    pub opening_point_recv_mult: u32,
    pub height_group_recv_mult: u32,
    pub group_claim_recv_mult: u32,
    pub commitment_root_send_mult: u32,
    pub final_root_poseidon2_recv_mults: [u32; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    #[serde(default)]
    pub is_final_perm: bool,
    #[serde(default)]
    pub final_root_perm_step_flags: [bool; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    #[serde(default)]
    pub final_root_poseidon2_input: [F; POSEIDON2_WIDTH],
    #[serde(default)]
    pub final_root_poseidon2_output: [F; POSEIDON2_WIDTH],
    #[serde(default)]
    pub final_root_poseidon2_recv_mult: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirBatchEvalRow {
    pub proof_idx: usize,
    pub is_start: bool,
    pub is_group_end: bool,
    pub role_id: usize,
    pub role_num_queries: usize,
    pub role_batching_bits: usize,
    pub role_log_blowup: usize,
    pub cursor: usize,
    pub chain_recv_cursor: usize,
    pub chain_send_cursor: usize,
    pub chain_recv_log_height: usize,
    pub chain_recv_batch_id: usize,
    pub chain_recv_batch_pos: usize,
    pub chain_recv_value_idx: usize,
    pub chain_recv_segment_element_count: usize,
    pub alpha_tidx: usize,
    pub alpha: [F; 5],
    pub pow_in: [F; 5],
    pub acc_in: [F; 5],
    pub group_base_in: [F; 5],
    pub pow_out: [F; 5],
    pub acc_out: [F; 5],
    pub group_base_out: [F; 5],
    pub value: [F; 5],
    pub log_height: usize,
    pub batch_id: usize,
    pub batch_pos: usize,
    pub chip_idx: usize,
    pub static_chip_id: usize,
    pub width: usize,
    pub value_idx: usize,
    pub segment_element_count: usize,
    pub is_value: bool,
    pub is_segment_start: bool,
    pub is_segment_end: bool,
    pub is_first_value: bool,
    pub is_group_start: bool,
    pub is_perm_batch: bool,
    pub group_log_height_gap: usize,
    pub batch_dim_recv_mult: u32,
    pub role_config_recv_mult: u32,
    pub group_claim_send_mult: u32,
    pub opened_eval_send_mult: u32,
    /// 1044 pow-seed publication count (deduped leaf group instances
    /// at this height); nonzero only on group-start rows.
    pub pow_seed_cnt: u32,
    pub chain_recv_mult: u32,
    pub chain_send_mult: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirQueryFoldRow {
    pub proof_idx: usize,
    pub is_seed: bool,
    pub is_round: bool,
    pub query_idx: usize,
    pub cursor: usize,
    pub w_qbase: usize,
    pub query_bits: usize,
    pub r_rounds: usize,
    pub query_sample: F,
    pub query_sample_raw: F,
    pub query_sample_high: usize,
    pub query_sample_shift: usize,
    pub query_sample_high_max: usize,
    pub query_sample_high_bits: usize,
    pub query_sample_high_gap_inv: F,
    pub idx: F,
    pub idx_bit: bool,
    pub idx_tail_bit0: bool,
    pub idx_tail_bit1: bool,
    pub x: F,
    pub acc: F,
    pub ipw: F,
    pub folded: [F; 5],
    pub f0: [F; 5],
    pub f1: [F; 5],
    pub chain_send_cursor: usize,
    pub chain_send_idx: F,
    pub chain_send_idx_bit: bool,
    pub chain_send_x: F,
    pub chain_send_acc: F,
    pub chain_send_ipw: F,
    pub chain_send_folded: [F; 5],
    pub r_fold: [F; 5],
    pub is_merge: bool,
    pub is_assign: bool,
    pub merge_beta: [F; 5],
    pub merge_eq: [F; 5],
    pub emit_prep_seed: bool,
    pub cfr: [F; 5],
    pub leaf_sum: [F; 5],
    pub twiddle_bytes: [u8; 3],
    pub twiddle_values: [F; 3],
    pub twiddle_product_01: F,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirLeafStreamRow {
    pub proof_idx: usize,
    pub is_unit_start: bool,
    pub is_unit_end: bool,
    /// The group instance's truncated leaf index (fold-bound on the
    /// consumer side).
    pub idx: usize,
    /// 1025 send count = number of consuming query merges.
    pub serve_cnt: usize,
    pub cursor: usize,
    pub chain_recv_cursor: usize,
    pub chain_send_cursor: usize,
    pub log_height: usize,
    pub batch_id: usize,
    pub chain_recv_log_height: usize,
    pub chain_recv_batch_id: usize,
    pub is_unit_key_start: bool,
    pub unit_key_gap: usize,
    pub alpha: [F; 5],
    pub pow_in: [F; 5],
    pub acc_in: [F; 5],
    pub slot_pows: [[F; 5]; 8],
    pub pow_out: [F; 5],
    pub acc_out: [F; 5],
    pub values: [F; 8],
    pub chunk_mask: [bool; 8],
    pub unit_key: usize,
    pub block_idx: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirLeafExtStreamRow {
    pub proof_idx: usize,
    pub is_unit_start: bool,
    pub is_unit_end: bool,
    /// The group instance's truncated leaf index (fold-bound on the
    /// consumer side).
    pub idx: usize,
    /// 1025 send count = number of consuming query merges.
    pub serve_cnt: usize,
    pub cursor: usize,
    pub chain_recv_cursor: usize,
    pub chain_send_cursor: usize,
    pub log_height: usize,
    pub batch_id: usize,
    pub chain_recv_log_height: usize,
    pub chain_recv_batch_id: usize,
    pub is_unit_key_start: bool,
    pub unit_key_gap: usize,
    pub alpha: [F; 5],
    pub pow_in: [F; 5],
    pub acc_in: [F; 5],
    pub slot_pows: [[F; 5]; 8],
    pub pow_out: [F; 5],
    pub acc_out: [F; 5],
    pub value_blocks: [[F; 8]; 5],
    pub chunk_masks: [[bool; 8]; 5],
    pub unit_key: usize,
    pub block_idx: usize,
}

/// Compact retained form of [`RecursionWhirLeafExtStreamRow`].
///
/// Construction and Merkle materialization use the full semantic row. Once
/// those readers finish, the per-proof record retains only the fields read by
/// the committed writer and residual/report reconstruction. `proof_idx` is
/// owned by the parent proof record; the other omitted mirrors are exact
/// products of the retained fields and the fixed permutation-batch contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionWhirLeafExtStreamTraceRow {
    pub is_unit_end: bool,
    pub is_unit_key_start: bool,
    pub element_masks: [bool; 8],
    pub idx: usize,
    pub serve_cnt: usize,
    pub chain_recv_cursor: usize,
    pub log_height: usize,
    pub block_idx: usize,
    pub alpha: [F; 5],
    pub pow_in: [F; 5],
    pub acc_in: [F; 5],
    /// Powers for extension elements 1 through 7. Element zero is exactly `pow_in`.
    pub slot_pows: [[F; 5]; 7],
    pub pow_out: [F; 5],
    pub acc_out: [F; 5],
    pub value_blocks: [[F; 8]; 5],
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionBatchConstraintRecord {
    pub num_public_values: usize,
    pub num_rounds: usize,
    pub c_chips: usize,
    pub cum_sums: Vec<RecursionBatchCumSumRecord>,
    pub perm_alpha: [F; 5],
    pub perm_beta: [F; 5],
    pub alpha: [F; 5],
    pub eq_challenges: Vec<[F; 5]>,
    pub rounds: Vec<RecursionSumcheckRoundRecord>,
    pub last_claim: [F; 5],
    pub publish_opening_point: bool,
    pub publish_terminal_outputs: bool,
}

impl RecursionBatchConstraintRecord {
    pub fn is_empty(&self) -> bool {
        self.rounds.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionBatchCumSumRecord {
    pub chip_idx: usize,
    pub lcs: [F; 5],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionSumcheckRoundRecord {
    pub round_idx: usize,
    /// Production is specialized to the degree-four / five-evaluation child
    /// sumcheck contract.
    pub evals: [[F; 5]; crate::batch_constraint_dt::BATCH_SUMCHECK_EVALS],
    pub challenge: [F; 5],
    pub claim_in: [F; 5],
    pub claim_out: [F; 5],
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionTranscriptRecord {
    pub events: Vec<RecursionTranscriptEvent>,
    pub bits_events: Vec<RecursionTranscriptBitsEvent>,
    /// Finalized source rows for transcript trace generation and Poseidon2 registration.
    /// Native finalization rejects proof records whose source rows were not captured.
    #[serde(default)]
    pub sponge_blocks: Vec<crate::system_dt::SpecSpongeBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionTranscriptEvent {
    pub tidx: usize,
    pub kind: RecursionTranscriptEventKind,
    pub value: F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionTranscriptEventKind {
    Observe,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionTranscriptBitsEvent {
    pub sample_tidx: usize,
    pub bits: usize,
    pub value: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionMerklePathRecord {
    rows: Vec<RecursionMerklePathRow>,
}

impl RecursionMerklePathRecord {
    pub fn push_row(&mut self, row: RecursionMerklePathRow) {
        self.rows.push(row);
    }

    pub(crate) fn install_rows(&mut self, rows: Vec<RecursionMerklePathRow>) {
        assert!(self.rows.is_empty(), "Merkle rows installed into a non-empty record");
        self.rows = rows;
    }

    pub fn rows(&self) -> &[RecursionMerklePathRow] {
        &self.rows
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionMerklePathOp {
    LeafAbsorb,
    PathCompress,
    InjectCompress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionMerklePathRow {
    pub proof_idx: usize,
    pub op: RecursionMerklePathOp,
    /// Routing tag: `slot*32 + codeword_log_height`.
    /// Note: bounds are asserted at record build time, not constrained in the AIR.
    pub unit_key: usize,
    pub commit_id: usize,
    pub level: usize,
    pub next_level: usize,
    pub block_idx: usize,
    pub cur_idx: usize,
    pub next_idx: usize,
    pub idx_bit: bool,
    pub left_idx: usize,
    pub left_cnt: usize,
    pub right_cnt: usize,
    pub root_cnt: usize,
    pub absorb_cnt: usize,
    pub is_last: bool,
    pub is_leaf_first: bool,
    pub is_leaf_last: bool,
    pub in_digest: [F; 8],
    pub sibling: [F; 8],
    pub prev_state: [F; POSEIDON2_WIDTH],
    pub chunk: [F; 8],
    pub chunk_mask: [bool; 8],
    pub input: [F; POSEIDON2_WIDTH],
    pub output: [F; POSEIDON2_WIDTH],
}

impl RecursionMerklePathRow {
    #[allow(clippy::too_many_arguments)]
    pub fn leaf_absorb(
        proof_idx: usize,
        unit_key: usize,
        commit_id: usize,
        block_idx: usize,
        cur_idx: usize,
        absorb_cnt: usize,
        is_last: bool,
        is_leaf_first: bool,
        is_leaf_last: bool,
        prev_state: [F; POSEIDON2_WIDTH],
        chunk: [F; 8],
        chunk_mask: [bool; 8],
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Self {
        Self::leaf_absorb_at_level(
            proof_idx,
            unit_key,
            commit_id,
            0,
            block_idx,
            cur_idx,
            absorb_cnt,
            is_last,
            is_leaf_first,
            is_leaf_last,
            prev_state,
            chunk,
            chunk_mask,
            poseidon2_output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn leaf_absorb_at_level(
        proof_idx: usize,
        unit_key: usize,
        commit_id: usize,
        digest_level: usize,
        block_idx: usize,
        cur_idx: usize,
        absorb_cnt: usize,
        is_last: bool,
        is_leaf_first: bool,
        is_leaf_last: bool,
        prev_state: [F; POSEIDON2_WIDTH],
        chunk: [F; 8],
        chunk_mask: [bool; 8],
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Self {
        let mut input = prev_state;
        for i in 0..8 {
            if chunk_mask[i] {
                input[i] = chunk[i];
            }
        }
        let output = poseidon2_output.permute_output(input);
        Self {
            proof_idx,
            op: RecursionMerklePathOp::LeafAbsorb,
            unit_key,
            commit_id,
            level: digest_level,
            next_level: digest_level,
            block_idx,
            cur_idx,
            next_idx: cur_idx,
            idx_bit: false,
            left_idx: 0,
            left_cnt: 0,
            right_cnt: 0,
            root_cnt: 0,
            absorb_cnt,
            is_last,
            is_leaf_first,
            is_leaf_last,
            in_digest: [F::zero(); 8],
            sibling: [F::zero(); 8],
            prev_state,
            chunk,
            chunk_mask,
            input,
            output,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn path_compress(
        proof_idx: usize,
        commit_id: usize,
        level: usize,
        cur_idx: usize,
        in_digest: [F; 8],
        sibling: [F; 8],
        is_last: bool,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Self {
        let idx_bit = cur_idx & 1 == 1;
        let next_idx = cur_idx >> 1;
        let left_idx = next_idx * 2;
        let mut input = [F::zero(); POSEIDON2_WIDTH];
        if idx_bit {
            input[..8].copy_from_slice(&sibling);
            input[8..].copy_from_slice(&in_digest);
        } else {
            input[..8].copy_from_slice(&in_digest);
            input[8..].copy_from_slice(&sibling);
        }
        let output = poseidon2_output.permute_output(input);
        Self {
            proof_idx,
            op: RecursionMerklePathOp::PathCompress,
            unit_key: 0,
            commit_id,
            level,
            next_level: level + 1,
            block_idx: 0,
            cur_idx,
            next_idx,
            idx_bit,
            left_idx,
            left_cnt: usize::from(!idx_bit),
            right_cnt: usize::from(idx_bit),
            root_cnt: usize::from(is_last),
            absorb_cnt: 0,
            is_last,
            is_leaf_first: false,
            is_leaf_last: false,
            in_digest,
            sibling,
            prev_state: [F::zero(); POSEIDON2_WIDTH],
            chunk: [F::zero(); 8],
            chunk_mask: [false; 8],
            input,
            output,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inject_compress(
        proof_idx: usize,
        commit_id: usize,
        level: usize,
        cur_idx: usize,
        in_digest: [F; 8],
        injected_digest: [F; 8],
        is_last: bool,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Self {
        let mut input = [F::zero(); POSEIDON2_WIDTH];
        input[..8].copy_from_slice(&in_digest);
        input[8..].copy_from_slice(&injected_digest);
        let output = poseidon2_output.permute_output(input);
        Self {
            proof_idx,
            op: RecursionMerklePathOp::InjectCompress,
            unit_key: 0,
            commit_id,
            level,
            next_level: level + 1,
            block_idx: 0,
            cur_idx,
            next_idx: cur_idx,
            idx_bit: false,
            left_idx: cur_idx,
            left_cnt: 1,
            right_cnt: 1,
            root_cnt: usize::from(is_last),
            absorb_cnt: 0,
            is_last,
            is_leaf_first: false,
            is_leaf_last: false,
            in_digest,
            sibling: injected_digest,
            prev_state: [F::zero(); POSEIDON2_WIDTH],
            chunk: [F::zero(); 8],
            chunk_mask: [false; 8],
            input,
            output,
        }
    }

    pub fn is_leaf_absorb(&self) -> bool {
        matches!(self.op, RecursionMerklePathOp::LeafAbsorb)
    }

    pub fn is_node(&self) -> bool {
        !self.is_leaf_absorb()
    }
}

pub type RecursionMerklePathEvent = RecursionMerklePathRow;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionConstraintRecord {
    pub events: Vec<RecursionConstraintEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionConstraintEvent {
    pub step_idx: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionProofShapeRecord {
    pub role_id: usize,
    pub num_public_values: usize,
    pub vk_commit: [F; 8],
    pub vk_meta: Vec<F>,
    #[serde(default)]
    pub vk_meta_send_mults: Vec<u32>,
    pub public_values: Vec<F>,
    #[serde(default)]
    pub public_value_send_mults: Vec<u32>,
    pub main_commit: [F; 8],
    pub permutation_commit: [F; 8],
    pub chips: Vec<RecursionProofShapeChip>,
    /// External provider buses are witness-multiplied. Keep them off until their consumers are
    /// registered, while internal/transcript/range buses stay active.
    pub publish_external: bool,
    /// Narrow provider switch for WHIR recursion inputs. This publishes only the proof-shape
    /// artifacts WHIR consumes: commitment roots, batch dimensions, and height groups.
    pub publish_whir_inputs: bool,
    /// Future terminal consumer for ProofShapeSummary. Kept separate from `publish_external`
    /// because 1008/1010 can be live before the terminal's 1022 consumer lands.
    #[serde(default)]
    pub publish_terminal_summary: bool,
}

impl RecursionProofShapeRecord {
    pub fn is_empty(&self) -> bool {
        self.chips.is_empty() && self.public_values.is_empty()
    }

    pub fn chip_count(&self) -> usize {
        self.chips.len()
    }

    pub fn e1_tidx_base(&self) -> usize {
        crate::batch_constraint_dt::columns::batch_seed_prefix_limbs_for_role_id(self.role_id) +
            self.num_public_values
    }

    pub fn e5_tidx_base(&self) -> usize {
        crate::batch_constraint_dt::record::BatchTranscriptLayout::new(
            self.num_public_values,
            self.chips.len(),
            0,
            self.role_id == 0,
        )
        .e3_tidx() +
            2 * crate::config::D_EF
    }

    pub fn distinct_log_heights_desc(&self) -> Vec<usize> {
        let mut heights = self.chips.iter().map(|chip| chip.log_height).collect::<Vec<_>>();
        heights.sort_unstable_by(|left, right| right.cmp(left));
        heights.dedup();
        heights
    }

    /// The replay segment base of this child's static chip ids (stride-128
    /// aligned). Homogeneous per proof — enforced in-circuit by the binder band gates.
    pub fn segment_id_base(&self) -> usize {
        self.chips.iter().map(|chip| chip.static_chip_id).min().unwrap_or(0) & !127
    }

    pub fn static_chip_ids_by_chip_idx(&self) -> Option<Vec<usize>> {
        let mut ids = vec![None; self.chips.len()];
        for chip in &self.chips {
            if chip.chip_idx >= ids.len() || ids[chip.chip_idx].is_some() {
                return None;
            }
            ids[chip.chip_idx] = Some(chip.static_chip_id);
        }
        ids.into_iter().collect()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionProofShapeChip {
    pub chip_idx: usize,
    pub static_chip_id: usize,
    pub stable_air_id: u32,
    pub log_height: usize,
    pub prep_width: usize,
    pub main_width: usize,
    pub perm_width: usize,
    pub constraint_count: usize,
    /// Static number of `air.eval()` roots, authenticated through
    /// `NativeChipMetadata`.
    pub gate_count: usize,
}

impl RecursionProofShapeChip {
    pub fn has_prep(&self) -> bool {
        self.prep_width != 0
    }

    pub fn metadata_request(&self, role_id: usize) -> RecursionNativeChipMetadataRequest {
        RecursionNativeChipMetadataRequest {
            role_id,
            chip_id: self.static_chip_id,
            stable_air_id: self.stable_air_id,
            prep_width: self.prep_width,
            main_width: self.main_width,
            perm_width: self.perm_width,
            constraint_count: self.constraint_count,
            gate_count: self.gate_count,
            count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecursionNativeChipMetadataPool {
    #[serde(default)]
    segments: Vec<Vec<RecursionNativeChipMetadataRequest>>,
    requests: Vec<RecursionNativeChipMetadataRequest>,
    #[serde(skip)]
    reduce_complete: bool,
}

impl Default for RecursionNativeChipMetadataPool {
    fn default() -> Self {
        Self { segments: Vec::new(), requests: Vec::new(), reduce_complete: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionNativeChipMetadataRequest {
    pub role_id: usize,
    pub chip_id: usize,
    pub stable_air_id: u32,
    pub prep_width: usize,
    pub main_width: usize,
    pub perm_width: usize,
    pub constraint_count: usize,
    pub gate_count: usize,
    pub count: u32,
}

impl RecursionNativeChipMetadataRequest {
    pub fn key(&self) -> (usize, usize, u32, usize, usize, usize, usize, usize) {
        (
            self.role_id,
            self.chip_id,
            self.stable_air_id,
            self.prep_width,
            self.main_width,
            self.perm_width,
            self.constraint_count,
            self.gate_count,
        )
    }
}

impl RecursionNativeChipMetadataPool {
    fn segment_summary(&self) -> ProviderSegmentSummary {
        ProviderSegmentSummary {
            segment_count: self.segments.len() + usize::from(!self.requests.is_empty()),
            entry_count: self.unique_count(),
        }
    }

    pub fn record_metadata(&mut self, request: RecursionNativeChipMetadataRequest) {
        self.record_metadata_count(request, request.count);
    }

    pub fn record_metadata_count(
        &mut self,
        mut request: RecursionNativeChipMetadataRequest,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        assert!(!self.reduce_complete, "metadata recorded after provider reduction");
        request.count = count;
        self.requests.push(request);
    }

    pub fn count_for(&self, row: RecursionNativeChipMetadataRequest) -> u32 {
        self.requests()
            .filter(|request| request.key() == row.key())
            .try_fold(0u32, |count, request| count.checked_add(request.count))
            .expect("metadata request count overflow across segments")
    }

    pub fn requests(&self) -> impl Iterator<Item = &RecursionNativeChipMetadataRequest> {
        self.segments.iter().flatten().chain(self.requests.iter())
    }

    pub fn append(&mut self, other: &mut Self) {
        assert!(!self.reduce_complete && !other.reduce_complete);
        if !self.requests.is_empty() {
            self.segments.push(core::mem::take(&mut self.requests));
        }
        self.segments.append(&mut other.segments);
        if !other.requests.is_empty() {
            self.segments.push(core::mem::take(&mut other.requests));
        }
    }

    pub fn unique_count(&self) -> usize {
        self.segments.iter().map(Vec::len).sum::<usize>() + self.requests.len()
    }

    pub fn total_count(&self) -> u64 {
        self.requests().map(|request| u64::from(request.count)).sum()
    }

    pub fn total_count_usize(&self) -> usize {
        usize::try_from(self.total_count()).expect("metadata request count exceeds usize")
    }

    fn reduce(&mut self) -> Result<(), String> {
        if self.reduce_complete {
            return Err("metadata provider reducer executed more than once".to_string());
        }
        let mut reduced = Vec::<RecursionNativeChipMetadataRequest>::new();
        let mut index =
            BTreeMap::<(usize, usize, u32, usize, usize, usize, usize, usize), usize>::new();
        for request in self.requests() {
            let key = request.key();
            if let Some(&idx) = index.get(&key) {
                reduced[idx].count =
                    reduced[idx].count.checked_add(request.count).ok_or_else(|| {
                        "metadata multiplicity overflow in provider reduction".to_string()
                    })?;
            } else {
                index.insert(key, reduced.len());
                reduced.push(*request);
            }
        }
        self.segments.clear();
        self.requests = reduced;
        self.reduce_complete = true;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RecursionNativeChipMetadataPool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            segments: Vec<Vec<RecursionNativeChipMetadataRequest>>,
            requests: Vec<RecursionNativeChipMetadataRequest>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let mut pool = Self::default();
        pool.segments = wire.segments;
        for request in wire.requests {
            pool.record_metadata(request);
        }
        Ok(pool)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionStatementRole {
    Lift,
    ReduceL2,
    ReduceL3,
    RootShrink,
}

/// One accepted-child vk class of the `StatementConfig` preprocessed table (global bus 12,
/// arity 9: `[class_id, digest[8]]`). Content is per-(statement_role, layer) setup data:
/// the reduce (L2) machine bakes one `BAKED_LIFT` row; the L3 machine adds `BAKED_L2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatementConfigRow {
    pub class_id: usize,
    pub digest: [F; DIGEST_SIZE],
}

#[derive(Debug, Clone, Serialize)]
#[serde(bound(serialize = ""))]
pub struct RecursionNativeProgram<Fld> {
    pub role: RecursionChildRole,
    pub statement_role: RecursionStatementRole,
    pub num_child_public_values: usize,
    pub child_contains_global_bus: bool,
    pub native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
    pub constraint_program: RecursionPolyAirVerifierProgram,
    pub statement_config: Vec<StatementConfigRow>,
    #[serde(skip)]
    _marker: PhantomData<Fld>,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = ""))]
struct RecursionNativeProgramWire {
    role: RecursionChildRole,
    statement_role: RecursionStatementRole,
    num_child_public_values: usize,
    child_contains_global_bus: bool,
    native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
    constraint_program: RecursionPolyAirVerifierProgramDto,
    statement_config: Vec<StatementConfigRow>,
}

impl<'de, Fld> Deserialize<'de> for RecursionNativeProgram<Fld> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RecursionNativeProgramWire::deserialize(deserializer)?;
        Self::try_from_wire(wire, false).map_err(D::Error::custom)
    }
}

impl<Fld> Default for RecursionNativeProgram<Fld> {
    fn default() -> Self {
        let constraint_program = RecursionPolyAirVerifierProgram::try_new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            [F::zero(); DIGEST_SIZE],
            Vec::new(),
            0,
        )
        .expect("empty default constraint program");
        Self {
            role: RecursionChildRole::Core,
            statement_role: RecursionStatementRole::Lift,
            num_child_public_values: 0,
            child_contains_global_bus: false,
            native_chip_metadata: Vec::new(),
            constraint_program,
            statement_config: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<Fld> RecursionNativeProgram<Fld> {
    fn try_from_wire(
        wire: RecursionNativeProgramWire,
        allow_l2_bootstrap_layout: bool,
    ) -> Result<Self, String> {
        validate_native_program_wire(&wire, allow_l2_bootstrap_layout)?;
        let constraint_program =
            RecursionPolyAirVerifierProgram::try_from_dto(wire.constraint_program)
                .map_err(|err| format!("invalid constraint program: {err:?}"))?;
        Ok(Self {
            role: wire.role,
            statement_role: wire.statement_role,
            num_child_public_values: wire.num_child_public_values,
            child_contains_global_bus: wire.child_contains_global_bus,
            native_chip_metadata: wire.native_chip_metadata,
            constraint_program,
            statement_config: wire.statement_config,
            _marker: PhantomData,
        })
    }

    pub(crate) fn try_from_constraint_dto(
        role: RecursionChildRole,
        statement_role: RecursionStatementRole,
        num_child_public_values: usize,
        child_contains_global_bus: bool,
        native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
        constraint_program: RecursionPolyAirVerifierProgramDto,
        statement_config: Vec<StatementConfigRow>,
        allow_l2_bootstrap_layout: bool,
    ) -> Result<Self, String> {
        Self::try_from_wire(
            RecursionNativeProgramWire {
                role,
                statement_role,
                num_child_public_values,
                child_contains_global_bus,
                native_chip_metadata,
                constraint_program,
                statement_config,
            },
            allow_l2_bootstrap_layout,
        )
    }

    pub fn layer(&self) -> crate::machine_dt::NativeRecursionAssemblyResult<NativeRecursionLayer> {
        NativeRecursionLayer::from_roles(self.role, self.statement_role)
    }

    pub fn new_core(
        num_child_public_values: usize,
        child_contains_global_bus: bool,
        native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
        constraint_program: RecursionPolyAirVerifierProgram,
    ) -> Self {
        Self {
            role: RecursionChildRole::Core,
            statement_role: RecursionStatementRole::Lift,
            num_child_public_values,
            child_contains_global_bus,
            native_chip_metadata,
            constraint_program,
            statement_config: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn new_compress(
        num_child_public_values: usize,
        child_contains_global_bus: bool,
        native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
        constraint_program: RecursionPolyAirVerifierProgram,
    ) -> Self {
        Self {
            role: RecursionChildRole::Compress,
            statement_role: RecursionStatementRole::ReduceL2,
            num_child_public_values,
            child_contains_global_bus,
            native_chip_metadata,
            constraint_program,
            statement_config: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn new_with_roles(
        role: RecursionChildRole,
        statement_role: RecursionStatementRole,
        num_child_public_values: usize,
        child_contains_global_bus: bool,
        native_chip_metadata: Vec<RecursionNativeChipMetadataRequest>,
        constraint_program: RecursionPolyAirVerifierProgram,
        statement_config: Vec<StatementConfigRow>,
    ) -> Self {
        Self {
            role,
            statement_role,
            num_child_public_values,
            child_contains_global_bus,
            native_chip_metadata,
            constraint_program,
            statement_config,
            _marker: PhantomData,
        }
    }
}

fn validate_native_program_wire(
    wire: &RecursionNativeProgramWire,
    allow_l2_bootstrap_layout: bool,
) -> Result<(), String> {
    validate_constraint_program_dto(&wire.constraint_program)?;
    let layer = NativeRecursionLayer::from_roles(wire.role, wire.statement_role)
        .map_err(|err| format!("invalid native recursion role/layer: {err}"))?;
    let params = layer.params();
    if wire.role != params.child_role ||
        wire.statement_role != params.statement_role ||
        wire.num_child_public_values != params.num_child_public_values ||
        wire.child_contains_global_bus != params.child_contains_global_bus ||
        wire.constraint_program.role != params.child_role
    {
        return Err(format!("native program fields do not match layer {layer:?}"));
    }
    crate::native_air_dt::validate_statement_config(wire.statement_role, &wire.statement_config)
        .map_err(|err| format!("invalid statement configuration: {err}"))?;

    let chips = &wire.constraint_program.chips;
    if chips.is_empty() {
        return Err("native program constraint universe is empty".to_string());
    }
    let actual_bases = chips
        .iter()
        .map(|chip| chip.static_chip_id & !127)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let expected_bases: &[usize] = match params.final_replay_layout {
        NativeFinalReplayLayout::SingleBase0 => &[0],
        NativeFinalReplayLayout::DualBase0Base128
            if allow_l2_bootstrap_layout &&
                layer == NativeRecursionLayer::L2Reduce &&
                actual_bases.as_slice() == [0] =>
        {
            &[0]
        }
        NativeFinalReplayLayout::DualBase0Base128 => &[0, 128],
    };
    if actual_bases != expected_bases {
        return Err(format!(
            "native program replay segments {actual_bases:?} do not match layer {layer:?}"
        ));
    }
    for &base in expected_bases {
        let ids = chips
            .iter()
            .filter(|chip| chip.static_chip_id & !127 == base)
            .map(|chip| chip.static_chip_id)
            .collect::<Vec<_>>();
        let expected_len = if layer == NativeRecursionLayer::L1Lift {
            ids.len()
        } else {
            NativeAirFamily::ALL.len()
        };
        let expected_end = base
            .checked_add(expected_len)
            .ok_or_else(|| "native replay segment row count overflow".to_string())?;
        if ids != (base..expected_end).collect::<Vec<_>>() {
            return Err(format!("native replay segment {base} is partial or malformed"));
        }
    }

    if wire.native_chip_metadata.len() != chips.len() {
        return Err("native metadata/program chip counts differ".to_string());
    }
    let expected_role_id = match wire.role {
        RecursionChildRole::Core => 0,
        RecursionChildRole::Compress => 1,
        RecursionChildRole::Shrink => 2,
    };
    for (metadata, chip) in wire.native_chip_metadata.iter().zip(chips) {
        let perm_width = chip
            .lookup_multiplicity_roots
            .len()
            .div_ceil(chip.logup_batch_size)
            .checked_mul(crate::config::D_EF)
            .ok_or_else(|| "native metadata permutation width overflow".to_string())?;
        if metadata.role_id != expected_role_id ||
            metadata.chip_id != chip.static_chip_id ||
            metadata.prep_width != chip.widths.preprocessed ||
            metadata.main_width != chip.widths.main ||
            metadata.perm_width != perm_width ||
            metadata.constraint_count != chip.num_constraints_from_builder ||
            metadata.gate_count != chip.gate_roots.len()
        {
            return Err(format!(
                "native metadata does not match constraint chip {}",
                chip.static_chip_id
            ));
        }
    }
    Ok(())
}

impl<Fld: Field> MachineProgram<Fld> for RecursionNativeProgram<Fld> {
    fn pc_start(&self) -> Fld {
        Fld::zero()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecursionPoseidon2Pool {
    #[serde(default)]
    segments: Vec<Vec<RecursionPoseidon2Request>>,
    requests: Vec<RecursionPoseidon2Request>,
    #[serde(skip)]
    reduce_complete: bool,
}

impl Default for RecursionPoseidon2Pool {
    fn default() -> Self {
        Self { segments: Vec::new(), requests: Vec::new(), reduce_complete: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPoseidon2Request {
    pub input: [F; POSEIDON2_WIDTH],
    pub count: u32,
}

impl RecursionPoseidon2Pool {
    fn segment_summary(&self) -> ProviderSegmentSummary {
        ProviderSegmentSummary {
            segment_count: self.segments.len() + usize::from(!self.requests.is_empty()),
            entry_count: self.unique_count(),
        }
    }

    pub fn record_poseidon2(&mut self, input: [F; POSEIDON2_WIDTH]) {
        self.record_poseidon2_count(input, 1);
    }

    pub(crate) fn record_poseidon2_batch<I>(&mut self, inputs: I)
    where
        I: IntoIterator<Item = [F; POSEIDON2_WIDTH]>,
        I::IntoIter: ExactSizeIterator,
    {
        let inputs = inputs.into_iter();
        let additional = inputs.len();
        if additional == 0 {
            return;
        }
        assert!(!self.reduce_complete, "Poseidon2 recorded after provider reduction");
        self.requests.reserve(additional);
        self.requests.extend(inputs.map(|input| RecursionPoseidon2Request { input, count: 1 }));
    }

    pub fn record_poseidon2_count(&mut self, input: [F; POSEIDON2_WIDTH], count: u32) {
        if count == 0 {
            return;
        }
        assert!(!self.reduce_complete, "Poseidon2 recorded after provider reduction");
        self.requests.push(RecursionPoseidon2Request { input, count });
    }

    pub fn record_request(&mut self, request: RecursionPoseidon2Request) {
        self.record_poseidon2_count(request.input, request.count);
    }

    pub fn requests(&self) -> impl Iterator<Item = &RecursionPoseidon2Request> {
        self.segments.iter().flatten().chain(self.requests.iter())
    }

    pub fn append(&mut self, other: &mut Self) {
        assert!(!self.reduce_complete && !other.reduce_complete);
        if !self.requests.is_empty() {
            self.segments.push(core::mem::take(&mut self.requests));
        }
        self.segments.append(&mut other.segments);
        if !other.requests.is_empty() {
            self.segments.push(core::mem::take(&mut other.requests));
        }
    }

    pub fn unique_count(&self) -> usize {
        self.segments.iter().map(Vec::len).sum::<usize>() + self.requests.len()
    }

    pub fn total_count(&self) -> u64 {
        self.requests().map(|request| u64::from(request.count)).sum()
    }

    pub fn total_count_usize(&self) -> usize {
        usize::try_from(self.total_count()).expect("Poseidon2 request count exceeds usize")
    }

    pub fn clear(&mut self) {
        self.segments.clear();
        self.requests.clear();
        self.reduce_complete = false;
    }

    fn reduce(&mut self) -> Result<(), String> {
        if self.reduce_complete {
            return Err("Poseidon2 provider reducer executed more than once".to_string());
        }
        let mut reduced = Vec::<RecursionPoseidon2Request>::new();
        let mut index = HashMap::<[u32; POSEIDON2_WIDTH], usize>::new();
        for request in self.requests() {
            let key = canonical_poseidon2_input(request.input);
            if let Some(&idx) = index.get(&key) {
                if reduced[idx].input != request.input {
                    return Err("equal Poseidon2 key has unequal authenticated input".to_string());
                }
                reduced[idx].count =
                    reduced[idx].count.checked_add(request.count).ok_or_else(|| {
                        "Poseidon2 multiplicity overflow in provider reduction".to_string()
                    })?;
            } else {
                index.insert(key, reduced.len());
                reduced.push(*request);
            }
        }
        self.segments.clear();
        self.requests = reduced;
        self.reduce_complete = true;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RecursionPoseidon2Pool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            segments: Vec<Vec<RecursionPoseidon2Request>>,
            requests: Vec<RecursionPoseidon2Request>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let mut pool = Self::default();
        pool.segments = wire.segments;
        for request in wire.requests {
            pool.record_request(request);
        }
        Ok(pool)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecursionRangePool {
    #[serde(default)]
    segments: Vec<Vec<RecursionRangeRequest>>,
    requests: Vec<RecursionRangeRequest>,
    #[serde(skip)]
    reduce_complete: bool,
}

impl Default for RecursionRangePool {
    fn default() -> Self {
        Self { segments: Vec::new(), requests: Vec::new(), reduce_complete: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionRangeRequest {
    pub value: usize,
    pub max_bits: usize,
    pub count: u32,
}

impl RecursionRangePool {
    fn segment_summary(&self) -> ProviderSegmentSummary {
        ProviderSegmentSummary {
            segment_count: self.segments.len() + usize::from(!self.requests.is_empty()),
            entry_count: self.unique_count(),
        }
    }

    pub fn record_range(&mut self, value: usize, max_bits: usize) {
        self.record_range_count(value, max_bits, 1);
    }

    pub fn record_range_count(&mut self, value: usize, max_bits: usize, count: u32) {
        if count == 0 {
            return;
        }
        assert!(!self.reduce_complete, "range recorded after provider reduction");
        assert!(max_bits > 0, "range request needs at least one bit");
        assert!(
            value < (1usize.checked_shl(max_bits as u32).expect("range bit width exceeds usize")),
            "range request value exceeds configured bit width"
        );

        self.requests.push(RecursionRangeRequest { value, max_bits, count });
    }

    pub fn requests(&self) -> impl Iterator<Item = &RecursionRangeRequest> {
        self.segments.iter().flatten().chain(self.requests.iter())
    }

    pub fn requests_for_bits(
        &self,
        max_bits: usize,
    ) -> impl Iterator<Item = RecursionRangeRequest> + '_ {
        self.requests().copied().filter(move |request| request.max_bits == max_bits)
    }

    pub fn append(&mut self, other: &mut Self) {
        assert!(!self.reduce_complete && !other.reduce_complete);
        if !self.requests.is_empty() {
            self.segments.push(core::mem::take(&mut self.requests));
        }
        self.segments.append(&mut other.segments);
        if !other.requests.is_empty() {
            self.segments.push(core::mem::take(&mut other.requests));
        }
    }

    pub fn unique_count(&self) -> usize {
        self.segments.iter().map(Vec::len).sum::<usize>() + self.requests.len()
    }

    pub fn total_count(&self) -> u64 {
        self.requests().map(|request| u64::from(request.count)).sum()
    }

    pub fn total_count_usize(&self) -> usize {
        usize::try_from(self.total_count()).expect("range request count exceeds usize")
    }

    fn reduce(&mut self) -> Result<(), String> {
        if self.reduce_complete {
            return Err("range provider reducer executed more than once".to_string());
        }
        let mut reduced = Vec::<RecursionRangeRequest>::new();
        let mut index = BTreeMap::<(usize, usize), usize>::new();
        for request in self.requests() {
            let key = (request.value, request.max_bits);
            if let Some(&idx) = index.get(&key) {
                reduced[idx].count =
                    reduced[idx].count.checked_add(request.count).ok_or_else(|| {
                        "range multiplicity overflow in provider reduction".to_string()
                    })?;
            } else {
                index.insert(key, reduced.len());
                reduced.push(*request);
            }
        }
        self.segments.clear();
        self.requests = reduced;
        self.reduce_complete = true;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RecursionRangePool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            segments: Vec<Vec<RecursionRangeRequest>>,
            requests: Vec<RecursionRangeRequest>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let mut pool = Self::default();
        pool.segments = wire.segments;
        for request in wire.requests {
            pool.record_range_count(request.value, request.max_bits, request.count);
        }
        Ok(pool)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecursionPowerPool {
    #[serde(default)]
    segments: Vec<Vec<RecursionPowerRequest>>,
    requests: Vec<RecursionPowerRequest>,
    #[serde(skip)]
    reduce_complete: bool,
}

impl Default for RecursionPowerPool {
    fn default() -> Self {
        Self { segments: Vec::new(), requests: Vec::new(), reduce_complete: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursionPowerRequest {
    pub base: usize,
    pub log_bits: usize,
    pub log: usize,
    pub pow: F,
    pub counts: PowerCheckerCounts,
}

impl RecursionPowerPool {
    fn segment_summary(&self) -> ProviderSegmentSummary {
        ProviderSegmentSummary {
            segment_count: self.segments.len() + usize::from(!self.requests.is_empty()),
            entry_count: self.unique_count(),
        }
    }

    pub fn record_pow<const BASE: usize, const LOG_BITS: usize>(&mut self, log: usize) -> F {
        self.record_pow_count::<BASE, LOG_BITS>(log, 1)
    }

    pub fn record_pow_count<const BASE: usize, const LOG_BITS: usize>(
        &mut self,
        log: usize,
        count: u32,
    ) -> F {
        self.record_count_dynamic(BASE, LOG_BITS, log, count, |counts| &mut counts.pow)
    }

    pub fn record_range_count<const BASE: usize, const LOG_BITS: usize>(
        &mut self,
        log: usize,
        count: u32,
    ) -> F {
        self.record_count_dynamic(BASE, LOG_BITS, log, count, |counts| &mut counts.range)
    }

    pub fn requests(&self) -> impl Iterator<Item = &RecursionPowerRequest> {
        self.segments.iter().flatten().chain(self.requests.iter())
    }

    pub fn requests_for<const BASE: usize, const LOG_BITS: usize>(
        &self,
    ) -> impl Iterator<Item = RecursionPowerRequest> + '_ {
        self.requests()
            .copied()
            .filter(|request| request.base == BASE && request.log_bits == LOG_BITS)
    }

    pub fn append(&mut self, other: &mut Self) {
        assert!(!self.reduce_complete && !other.reduce_complete);
        if !self.requests.is_empty() {
            self.segments.push(core::mem::take(&mut self.requests));
        }
        self.segments.append(&mut other.segments);
        if !other.requests.is_empty() {
            self.segments.push(core::mem::take(&mut other.requests));
        }
    }

    pub fn record_power_request(&mut self, request: RecursionPowerRequest) {
        if request.counts.pow != 0 {
            let pow = self.record_pow_count_dynamic(
                request.base,
                request.log_bits,
                request.log,
                request.counts.pow,
            );
            assert_eq!(pow, request.pow, "power record output mismatch");
        }
        if request.counts.range != 0 {
            let pow = self.record_range_count_dynamic(
                request.base,
                request.log_bits,
                request.log,
                request.counts.range,
            );
            assert_eq!(pow, request.pow, "power record output mismatch");
        }
    }

    fn record_pow_count_dynamic(
        &mut self,
        base: usize,
        log_bits: usize,
        log: usize,
        count: u32,
    ) -> F {
        self.record_count_dynamic(base, log_bits, log, count, |counts| &mut counts.pow)
    }

    fn record_range_count_dynamic(
        &mut self,
        base: usize,
        log_bits: usize,
        log: usize,
        count: u32,
    ) -> F {
        self.record_count_dynamic(base, log_bits, log, count, |counts| &mut counts.range)
    }

    fn record_count_dynamic(
        &mut self,
        base: usize,
        log_bits: usize,
        log: usize,
        count: u32,
        select_count: impl FnOnce(&mut PowerCheckerCounts) -> &mut u32,
    ) -> F {
        assert!(base > 1, "power request base must be greater than one");
        assert!(log_bits > 0, "power request needs at least one log bit");
        let max_log =
            1usize.checked_shl(log_bits as u32).expect("power request log bit width exceeds usize");
        assert!(log < max_log, "power request log exceeds configured bit width");

        let pow = power_value(base, log);
        if count == 0 {
            return pow;
        }
        assert!(!self.reduce_complete, "power recorded after provider reduction");

        let mut counts = PowerCheckerCounts::default();
        *select_count(&mut counts) = count;
        self.requests.push(RecursionPowerRequest { base, log_bits, log, pow, counts });
        pow
    }

    pub fn unique_count(&self) -> usize {
        self.segments.iter().map(Vec::len).sum::<usize>() + self.requests.len()
    }

    pub fn total_count(&self) -> u64 {
        self.requests()
            .map(|request| u64::from(request.counts.pow) + u64::from(request.counts.range))
            .sum()
    }

    pub fn total_count_usize(&self) -> usize {
        usize::try_from(self.total_count()).expect("power request count exceeds usize")
    }

    fn reduce(&mut self) -> Result<(), String> {
        if self.reduce_complete {
            return Err("power provider reducer executed more than once".to_string());
        }
        let mut reduced = Vec::<RecursionPowerRequest>::new();
        let mut index = BTreeMap::<(usize, usize, usize), usize>::new();
        for request in self.requests() {
            let key = (request.base, request.log_bits, request.log);
            if let Some(&idx) = index.get(&key) {
                if reduced[idx].pow != request.pow {
                    return Err("equal power key has unequal authenticated output".to_string());
                }
                reduced[idx].counts.pow =
                    reduced[idx].counts.pow.checked_add(request.counts.pow).ok_or_else(|| {
                        "power multiplicity overflow in provider reduction".to_string()
                    })?;
                reduced[idx].counts.range =
                    reduced[idx].counts.range.checked_add(request.counts.range).ok_or_else(
                        || "power range multiplicity overflow in provider reduction".to_string(),
                    )?;
            } else {
                index.insert(key, reduced.len());
                reduced.push(*request);
            }
        }
        self.segments.clear();
        self.requests = reduced;
        self.reduce_complete = true;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RecursionPowerPool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            segments: Vec<Vec<RecursionPowerRequest>>,
            requests: Vec<RecursionPowerRequest>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let mut pool = Self::default();
        pool.segments = wire.segments;
        for request in wire.requests {
            pool.record_power_request(request);
        }
        Ok(pool)
    }
}

fn canonical_poseidon2_input(input: [F; POSEIDON2_WIDTH]) -> [u32; POSEIDON2_WIDTH] {
    input.map(|x| x.as_canonical_u32())
}

fn power_value(base: usize, log: usize) -> F {
    let mut result = F::one();
    let base = F::from_canonical_usize(base);
    for _ in 0..log {
        result *= base;
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn poseidon_input(value: usize) -> [F; POSEIDON2_WIDTH] {
        [F::from_canonical_usize(value); POSEIDON2_WIDTH]
    }

    #[test]
    fn provider_inputs_remain_raw_across_owner_segments_until_the_single_reduction() {
        let input = poseidon_input(7);
        let mut left = RecursionRecord::default();
        left.poseidon2.record_poseidon2_count(input, 1);
        left.poseidon2.record_poseidon2_count(input, 2);
        let mut right = RecursionRecord::default();
        right.poseidon2.record_poseidon2_count(input, 4);

        MachineRecord::append(&mut left, &mut right);

        let rows = left.poseidon2.requests().copied().collect::<Vec<_>>();
        assert_eq!(rows.len(), 3, "provider preparation preserves raw publications");
        assert_eq!(rows.iter().map(|row| row.count).collect::<Vec<_>>(), [1, 2, 4]);
        assert_eq!(left.poseidon2.total_count(), 7);
        assert_eq!(
            left.provider_input_layout().families[1],
            ProviderSegmentSummary { segment_count: 2, entry_count: 3 }
        );
    }

    #[test]
    fn provider_publishes_raw_then_reduces_exactly_once() {
        let input = poseidon_input(11);
        let mut record = RecursionRecord::default();
        record.poseidon2.record_poseidon2_count(input, 2);
        record.poseidon2.record_poseidon2_count(input, 5);
        assert_eq!(record.poseidon2.unique_count(), 2, "provider preparation stays raw");

        let stats = record.reduce_provider_inputs().expect("one provider pass");
        assert_eq!(stats.passes, 1);
        assert_eq!(stats.raw_entries, 2);
        assert_eq!(stats.unique_entries, 1);
        assert_eq!(stats.duplicate_entries, 1);
        assert_eq!(record.poseidon2.requests().next().expect("reduced row").count, 7);
        assert!(record.reduce_provider_inputs().is_err());
    }

    #[test]
    fn poseidon2_batch_publication_matches_scalar_order_and_counts() {
        let inputs = [poseidon_input(3), poseidon_input(5), poseidon_input(3)];
        let mut scalar = RecursionPoseidon2Pool::default();
        for input in inputs {
            scalar.record_poseidon2(input);
        }
        let mut batch = RecursionPoseidon2Pool::default();
        batch.record_poseidon2_batch(inputs);
        assert_eq!(
            batch.requests().copied().collect::<Vec<_>>(),
            scalar.requests().copied().collect::<Vec<_>>()
        );
        assert_eq!(batch.total_count(), scalar.total_count());
    }

    #[test]
    fn provider_reduction_rejects_multiplicity_overflow() {
        let input = poseidon_input(13);
        let mut record = RecursionRecord::default();
        record.poseidon2.record_poseidon2_count(input, u32::MAX);
        record.poseidon2.record_poseidon2_count(input, 1);
        let err = record.reduce_provider_inputs().expect_err("overflow must fail closed");
        assert!(err.contains("multiplicity overflow"));
    }

    #[test]
    fn provider_reduction_rejects_equal_key_with_unequal_authenticated_payload() {
        let mut record = RecursionRecord::default();
        record.pow.requests.push(RecursionPowerRequest {
            base: 2,
            log_bits: 8,
            log: 3,
            pow: F::from_canonical_usize(8),
            counts: PowerCheckerCounts { pow: 1, range: 0 },
        });
        record.pow.requests.push(RecursionPowerRequest {
            base: 2,
            log_bits: 8,
            log: 3,
            pow: F::from_canonical_usize(9),
            counts: PowerCheckerCounts { pow: 1, range: 0 },
        });
        let err = record.reduce_provider_inputs().expect_err("payload mismatch must fail closed");
        assert!(err.contains("unequal authenticated output"));
    }

    #[test]
    fn concrete_workspace_artifacts_are_single_flight_and_not_cloned() {
        let builds = AtomicUsize::new(0);
        let artifacts = TracegenWorkspaceArtifacts::default();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let rows = artifacts.transcript_sponge.get_or_init(|| {
                        builds.fetch_add(1, Ordering::Relaxed);
                        Arc::from(Vec::<SpecSpongeBlock>::new())
                    });
                    assert!(rows.is_empty());
                });
            }
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(artifacts.initialized_entries(), 1);

        let cloned = artifacts.clone();
        let rows = cloned.transcript_sponge.get_or_init(|| {
            builds.fetch_add(1, Ordering::Relaxed);
            Arc::from(Vec::<SpecSpongeBlock>::new())
        });
        assert!(rows.is_empty());
        assert_eq!(builds.load(Ordering::Relaxed), 2);
        assert_eq!(cloned.initialized_entries(), 1);
    }

    #[test]
    fn profile_merge_takes_max_for_millisecond_and_microsecond_maxima() {
        let aggregate = RecursionRecordProfile::default();
        aggregate.set_structural_counters([
            ("phase_max_ms", 11),
            ("phase_max_us", 11_000),
            ("phase_total_us", 11_000),
        ]);
        let child = RecursionRecordProfile::default();
        child.set_structural_counters([
            ("phase_max_ms", 7),
            ("phase_max_us", 7_000),
            ("phase_total_us", 7_000),
        ]);

        aggregate.merge_from(&child);

        let counters = aggregate.snapshot().structural_counters;
        assert_eq!(counters["phase_max_ms"], 11);
        assert_eq!(counters["phase_max_us"], 11_000);
        assert_eq!(counters["phase_total_us"], 18_000);
    }

    #[test]
    fn proof_and_tracegen_seals_publish_real_batched_events_at_the_timing_boundary() {
        let profile = RecursionRecordProfile::default();
        let proof_ready = Instant::now();
        profile.mark_prepare_started(proof_ready);
        profile.publish_proof_batch_and_seal(
            proof_ready,
            7,
            &[("semantic", 0)],
            &[("transcript", 2, 3)],
            &[
                ("per_proof_vk_full_validation_calls", 0),
                ("per_proof_metadata_full_rebuilds", 0),
                ("per_proof_metadata_name_sorts", 0),
                ("per_proof_machine_layout_second_passes", 0),
            ],
        );
        let proof_snapshot = profile.snapshot();
        assert_eq!(proof_snapshot.structural_counters["profile_batch_publications"], 1);
        assert_eq!(proof_snapshot.structural_counters["profile_lock_acquisitions"], 1);
        assert_eq!(proof_snapshot.structural_counters["preflight.semantic_us"], 7);
        assert_eq!(proof_snapshot.poseidon2_memo["transcript"].hits, 2);
        assert_eq!(proof_snapshot.poseidon2_memo["transcript"].misses, 3);
        assert_eq!(
            proof_snapshot.record_splits["child_proof_ready_to_prepared_segment_sealed"].count,
            1
        );

        let admit_started = Instant::now();
        let admit_ms = profile.publish_tracegen_input_batch_and_seal(
            admit_started,
            &[("descriptor", 0)],
            &[("tracegen_input_descriptor_bytes", 16)],
        );
        let seal_snapshot = profile.snapshot();
        assert_eq!(seal_snapshot.structural_counters["tracegen_input_admission_events"], 1);
        assert_eq!(seal_snapshot.structural_counters["tracegen_input_descriptor_bytes"], 16);
        assert_eq!(seal_snapshot.record_splits["tracegen_input_seal"].ms, admit_ms);
        assert_eq!(
            seal_snapshot.record_splits["last_required_child_ready_to_tracegen_input_sealed"].count,
            1
        );
    }

    #[test]
    fn frozen_program_clone_shares_authority_and_decode_refreezes() {
        let program = RecursionNativeProgram::<F>::default().constraint_program;
        let cloned = program.clone();
        assert!(program.shares_authority_with(&cloned));
        assert!(Arc::ptr_eq(&program.constraint_static_plan(), &cloned.constraint_static_plan()));

        let encoded = bincode::serialize(&program).expect("serialize program");
        let decoded: RecursionPolyAirVerifierProgram =
            bincode::deserialize(&encoded).expect("deserialize program");
        assert!(!program.shares_authority_with(&decoded));
        assert!(!Arc::ptr_eq(&program.constraint_static_plan(), &decoded.constraint_static_plan()));
    }

    #[test]
    fn finalized_program_authority_pins_statement_layer() {
        let mut l2 = RecursionNativeProgram::<F>::default();
        l2.statement_role = RecursionStatementRole::ReduceL2;
        l2.statement_config =
            vec![StatementConfigRow { class_id: 0, digest: [F::one(); DIGEST_SIZE] }];

        let authority = FinalizedProgramAuthority::from_program(&l2);
        assert_eq!(authority, FinalizedProgramAuthority::from_program(&l2.clone()));

        let mut l3 = l2.clone();
        l3.statement_role = RecursionStatementRole::ReduceL3;
        assert_ne!(authority, FinalizedProgramAuthority::from_program(&l3));

        let mut different_config = l2.clone();
        different_config.statement_config[0].class_id = 1;
        assert_ne!(authority, FinalizedProgramAuthority::from_program(&different_config));

        let mut different_ir = l2.clone();
        let mut different_ir_dto = different_ir.constraint_program.to_dto();
        different_ir_dto.artifact_digest[0] = F::one();
        different_ir.constraint_program =
            RecursionPolyAirVerifierProgram::try_from_dto(different_ir_dto)
                .expect("modified frozen program");
        assert_ne!(authority, FinalizedProgramAuthority::from_program(&different_ir));
    }

    #[test]
    fn proof_shape_static_ids_are_indexed_by_chip_idx() {
        let shape = RecursionProofShapeRecord {
            chips: vec![
                RecursionProofShapeChip { chip_idx: 2, static_chip_id: 11, ..Default::default() },
                RecursionProofShapeChip { chip_idx: 0, static_chip_id: 7, ..Default::default() },
                RecursionProofShapeChip { chip_idx: 1, static_chip_id: 9, ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(shape.static_chip_ids_by_chip_idx(), Some(vec![7, 9, 11]));
    }

    #[test]
    fn proof_shape_static_ids_reject_non_contiguous_chip_idx() {
        let missing = RecursionProofShapeRecord {
            chips: vec![
                RecursionProofShapeChip { chip_idx: 0, static_chip_id: 7, ..Default::default() },
                RecursionProofShapeChip { chip_idx: 2, static_chip_id: 11, ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(missing.static_chip_ids_by_chip_idx(), None);

        let duplicate = RecursionProofShapeRecord {
            chips: vec![
                RecursionProofShapeChip { chip_idx: 0, static_chip_id: 7, ..Default::default() },
                RecursionProofShapeChip { chip_idx: 0, static_chip_id: 9, ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(duplicate.static_chip_ids_by_chip_idx(), None);
    }
}
