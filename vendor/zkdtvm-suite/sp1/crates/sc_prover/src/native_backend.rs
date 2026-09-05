//! The native recursion ladder as the `compress` backend: SDK glue over
//! `native_recursion::compress_dt` — type bridging, a fail-closed config check
//! at init, and a bounded layer-parallel scheduler.
//! Note: no AIR/machine content lives here; changing the machine re-keys every
//! verifying key (refresh the pinned vk digest below when that happens).

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{sync_channel, TrySendError},
        Arc, Condvar, Mutex,
    },
};

use crate::Instant;

use dt_core_machine::reduce::DTReduceProof;
use dt_stark::{
    koalabear_poseidon2::{whir_config, StageJsonConfig, WhirJsonConfig},
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::SCStarkVerifyingKey,
        proof::{SCMachineProof, SCShardProof},
    },
    DTProverOpts,
};
use native_recursion::{
    compress_dt::{
        build_ladder_with_provider, root_vk_digest, verifying_keys_equal, NativeCompressNodeStat,
        NativeLadderContext, NativeRecursionRequest, NativeReduceChild,
    },
    machine_dt::{CpuNativeProver, NativeProverProvider},
    prelude::{BuildingRecord, F, SC},
    statement_dt::core_vk_statement_digest,
    verifier_dt::{NativeRootVerifierArtifactV1, NATIVE_ROOT_VK_FROZEN_DIGEST_V1},
};
use p3_field::PrimeField32;
use rayon::prelude::*;
use serde::Serialize;

use crate::{tree_plan, CoreSC, DTPublicValues, DTRecursionProverError, DTVerifyingKey, RootSC};

/// Maximum node arity the scheduler may use — the machine's keyed capacity,
/// re-exported from the crate that owns the reduce programs.
pub use native_recursion::compress_dt::NATIVE_MAX_NODE_ARITY;

/// root_shrink stack height
pub const NATIVE_ROOT_SHRINK_STACK_LOG_HEIGHT: usize = 18;

/// The vk_L4 statement digest for the current native recursion machine (H = 18).
/// vk_L4 transitively pins the whole ladder —
/// its program bakes the L3 digest, which bakes L2 and lift. A mismatch here means
/// the machine, a program builder, or the PCS config drifted from the freeze.
///
/// Re-pinned 2026-07-09 for the SHA256 root_shrink switch: the L4 stage now
/// commits its PCS with the `RootSC` byte-digest Merkle (32-byte SHA256
/// commitments), so vk_L4's commit — and this digest, now computed via
/// `root_vk_digest` (byte-lifted commit into the same Poseidon2 sponge) —
/// re-keyed. Lift/L2/L3 stay on the Poseidon2 `SC` and did not re-key.
///
/// Re-pinned 2026-07-18 for the explicit L1/L2/L3/L4 AIR wire identities and
/// their corresponding alphabetical static-ID maps.
///
/// Re-pinned 2026-07-20 after all paired WHIR upper-bound checks were consolidated
/// onto Range21 and the retired Range7/9/10/11/12 AIRs were removed (registry v2).
///
/// Re-pinned 2026-07-29 after the accepted remaining-native-AIR epoch reached
/// program schema 3, registry 5, and cache schema 17. Two fresh repository-
/// external cache builds produced this same root digest.
pub const VK_L4_FROZEN_DIGEST: [u32; 8] = NATIVE_ROOT_VK_FROZEN_DIGEST_V1;

fn runtime(err: impl std::fmt::Display) -> DTRecursionProverError {
    DTRecursionProverError::RuntimeError(err.to_string())
}

fn validate_frozen_l4_digest(
    vk: &SCStarkVerifyingKey<RootSC>,
) -> Result<(), DTRecursionProverError> {
    let actual = root_vk_digest(vk).map(|limb| limb.as_canonical_u32());
    if actual != VK_L4_FROZEN_DIGEST {
        return Err(runtime(format!(
            "vk_L4 digest mismatch: actual={actual:?} expected={VK_L4_FROZEN_DIGEST:?}"
        )));
    }
    Ok(())
}

fn stable_hash_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

// ───────────────────────── fail-closed config authority ─────────────────────────

/// The env overrides that can silently repoint WHIR parameters. The SDK product route
/// rejects ALL of them: diagnostic runs belong to the harness bins (which drive the
/// ladder library directly), never to `DTProver::compress`.
fn rejected_env_overrides() -> Vec<String> {
    std::env::vars()
        .map(|(key, _)| key)
        .filter(|key| {
            key.starts_with("WHIR_") || key == "FRI_QUERIES" || key == "DT_USE_PATH_PRUNING"
        })
        .collect()
}

/// Strict re-read of the suite JSON: resolving the same
/// ancestor walk the runtime loader uses, but treating a missing file, a parse
/// failure, or an absent mode-relevant field as a hard error instead of a silent
/// fallback to compiled-in defaults.
struct ConfigAuthority {
    config: WhirJsonConfig,
    hash: String,
    contents: String,
}

fn read_config_authority() -> Result<ConfigAuthority, String> {
    let name = "whir_config_koalabear_ext5.json";
    let mut dir = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    let path = loop {
        let candidate = dir.join(name);
        if candidate.is_file() {
            break candidate;
        }
        if !dir.pop() {
            return Err(format!(
                "config authority {name} not found in any ancestor of the working directory; \
                 the native backend refuses to run on compiled-in defaults"
            ));
        }
    };
    let contents =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let config =
        serde_json::from_str(&contents).map_err(|e| format!("parse {}: {e}", path.display()))?;
    Ok(ConfigAuthority { config, hash: stable_hash_hex(contents.as_bytes()), contents })
}

fn require<T: Copy>(stage: &str, field: &str, value: Option<T>) -> Result<T, String> {
    value.ok_or_else(|| format!("config authority: {stage}.{field} is absent (fail-closed)"))
}

fn assert_stage_eq(
    stage: &str,
    authority: &StageJsonConfig,
    live: &StageJsonConfig,
) -> Result<(), String> {
    macro_rules! check {
        ($field:ident) => {
            if authority.$field != live.$field {
                return Err(format!(
                    "config authority mismatch at {}.{}: JSON {:?} vs active {:?}",
                    stage,
                    stringify!($field),
                    authority.$field,
                    live.$field
                ));
            }
        };
    }
    check!(log_blowup);
    check!(num_queries);
    check!(grinding_bits_query);
    check!(grinding_bits_batching);
    check!(grinding_bits_folding);
    check!(log_final_poly_len);
    check!(num_committed_groups);
    check!(round_query_counts);
    check!(stack_log_height);
    check!(stacking);
    check!(path_pruning);
    Ok(())
}

/// The init-time config gate. Ordering: env rejection first (an override invalidates
/// everything else), then the strict re-read, then field-for-field equality with the
/// live singleton the active SC configs were built from, then the signed pins.
fn assert_config_authority() -> Result<String, String> {
    let overrides = rejected_env_overrides();
    if !overrides.is_empty() {
        return Err(format!(
            "the native backend rejects WHIR env overrides on the product route \
             (fail-closed): {} is set; unset it or run diagnostics through the \
             harness bins",
            overrides.join(", ")
        ));
    }

    let authority_file = read_config_authority()?;
    let authority_hash = authority_file.hash;
    let authority = authority_file.config;

    // Mode-relevant fields must be PRESENT in the authority file (never defaulted).
    for stage in ["core", "compress", "shrink"] {
        let cfg = authority.stage(stage);
        require(stage, "log_blowup", cfg.log_blowup)?;
        require(stage, "num_queries", cfg.num_queries)?;
        require(stage, "grinding_bits_query", cfg.grinding_bits_query)?;
        require(stage, "grinding_bits_batching", cfg.grinding_bits_batching)?;
        let stacking = require(stage, "stacking", cfg.stacking)?;
        require(stage, "path_pruning", cfg.path_pruning)?;
        if stacking {
            return Err(format!(
                "config authority: {stage}.stacking is true; the frozen native machines \
                 expect the non-stacking path below root_shrink"
            ));
        }
    }
    let root = authority.stage("root_shrink");
    require("root_shrink", "log_blowup", root.log_blowup)?;
    require("root_shrink", "grinding_bits_query", root.grinding_bits_query)?;
    require("root_shrink", "grinding_bits_batching", root.grinding_bits_batching)?;
    require("root_shrink", "grinding_bits_folding", root.grinding_bits_folding)?;
    require("root_shrink", "log_final_poly_len", root.log_final_poly_len)?;
    let groups = require("root_shrink", "num_committed_groups", root.num_committed_groups)?;
    let queries = root
        .round_query_counts
        .as_ref()
        .ok_or("config authority: root_shrink.round_query_counts is absent (fail-closed)")?;
    if queries.len() != groups {
        return Err(format!(
            "config authority: root_shrink.round_query_counts has {} entries, \
             num_committed_groups is {groups}",
            queries.len()
        ));
    }
    let stack = require("root_shrink", "stack_log_height", root.stack_log_height)?;
    if stack != NATIVE_ROOT_SHRINK_STACK_LOG_HEIGHT {
        return Err(format!(
            "config authority: root_shrink.stack_log_height is {stack}, the signed pin \
             is {NATIVE_ROOT_SHRINK_STACK_LOG_HEIGHT} (H is user-signed; changing it is \
             a freeze reopen, not a config edit)"
        ));
    }
    if !require("root_shrink", "stacking", root.stacking)? {
        return Err("config authority: root_shrink.stacking must be true".into());
    }
    require("root_shrink", "path_pruning", root.path_pruning)?;
    if authority.num_skip_rounds.is_none() ||
        authority.chip_log_height_threshold.is_none() ||
        authority.use_algebraic_decomp.is_none()
    {
        return Err(
            "config authority: num_skip_rounds / chip_log_height_threshold / use_algebraic_decomp absent"
                .into(),
        );
    }

    // The live singleton is what every SC constructor consumed; it must be the same
    // content (catches an init-time silent default or a cwd-dependent divergence).
    let live = whir_config();
    if live.num_skip_rounds != authority.num_skip_rounds ||
        live.chip_log_height_threshold != authority.chip_log_height_threshold ||
        live.use_algebraic_decomp != authority.use_algebraic_decomp
    {
        return Err(
            "config authority mismatch at num_skip_rounds/chip_log_height_threshold/use_algebraic_decomp"
                .into(),
        );
    }
    for stage in ["core", "compress", "shrink", "root_shrink"] {
        assert_stage_eq(stage, authority.stage(stage), live.stage(stage))?;
    }

    // Live-object probe: the active root_shrink SC must report the signed stack pin.
    // The L4 machine consumes the SHA256-hashed RootSC since the hash switch.
    let live_hint = RootSC::default().mlpcs_stack_log_height_hint();
    if live_hint != Some(NATIVE_ROOT_SHRINK_STACK_LOG_HEIGHT) {
        return Err(format!(
            "active root_shrink config reports stack_log_height {live_hint:?}, \
             the signed pin is {NATIVE_ROOT_SHRINK_STACK_LOG_HEIGHT}"
        ));
    }
    Ok(authority_hash)
}

fn native_ladder_cache_dir_from_override(path: Option<std::ffi::OsString>) -> PathBuf {
    if let Some(path) = path.filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }
    dirs::cache_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("zkdtvm-suite")
        .join("native-recursion")
}

fn native_ladder_cache_dir() -> PathBuf {
    native_ladder_cache_dir_from_override(std::env::var_os("DT_NATIVE_RECURSION_CACHE_DIR"))
}

// ───────────────────────── the bounded layer-parallel scheduler ─────────────────────────

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from).max(1)
}

/// Resolve the one production worker-count input that affects TreePlan
/// topology. Qualification harnesses use this function too, so an explicit
/// production override cannot silently select a different tree.
pub fn tree_plan_worker_hint() -> Result<usize, DTRecursionProverError> {
    let default_workers = available_parallelism().min(2);
    let workers = match std::env::var("DT_NATIVE_RECURSION_EARLY_LIFT_WORKERS") {
        Ok(value) => value.parse::<usize>().ok().filter(|value| *value > 0).ok_or_else(|| {
            runtime(format!(
                "DT_NATIVE_RECURSION_EARLY_LIFT_WORKERS must be a positive integer, got {value:?}"
            ))
        })?,
        Err(std::env::VarError::NotPresent) => default_workers,
        Err(err) => {
            return Err(runtime(format!("read DT_NATIVE_RECURSION_EARLY_LIFT_WORKERS: {err}")));
        }
    };
    if workers > available_parallelism() {
        return Err(runtime(format!(
            "DT_NATIVE_RECURSION_EARLY_LIFT_WORKERS={workers} exceeds available parallelism {}",
            available_parallelism()
        )));
    }
    Ok(workers)
}

/// One scheduler decision, recorded per layer for the run report.
#[derive(Debug, Clone, Serialize)]
pub struct NativeLayerDecision {
    pub layer: String,
    pub nodes: usize,
    pub arities: Vec<usize>,
    pub wall_ms: u128,
    pub queue_wait_ms_max: u128,
    pub node_run_ms_max: u128,
    pub join_tail_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeSchedulerEvent {
    pub layer: String,
    pub node_index: usize,
    pub arity: usize,
    pub worker_index: Option<usize>,
    pub queue_wait_ms: u128,
    pub start_ms: u128,
    pub end_ms: u128,
    pub run_ms: u128,
    /// Width of the shared process-global Rayon pool, never a node-local budget.
    pub shared_rayon_pool_threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativePhaseTiming {
    pub phase: String,
    pub start_ms: u128,
    pub end_ms: u128,
    pub wall_ms: u128,
}

/// Pure planner inputs and selected output recorded for deterministic ledger
/// replay. The requested policy may select V1 when its strict guard does not
/// find an improvement.
#[derive(Debug, Clone, Serialize)]
pub struct NativeTreePolicyTelemetry {
    pub requested_version: u32,
    pub selected_version: u32,
    pub arity_cap: u8,
    pub worker_hint: u8,
    pub lift_bands: Vec<u8>,
    pub l2_bands: Vec<u8>,
    pub l3_bands: Vec<u8>,
    pub lift_spans: Vec<(usize, usize)>,
}

/// The scheduler/timing report `compress_native` leaves behind for diagnostics.
#[derive(Debug, Clone, Serialize, Default)]
pub struct NativeCompressReport {
    pub shard_count: usize,
    pub core_child_record_count: usize,
    /// Sum of per-child elapsed recording durations. This is concurrent work accounting, not
    /// measured CPU time and not an additive wall-time component.
    pub core_child_record_work_sum_ms: u128,
    pub core_child_record_max_ms: u128,
    /// Elapsed from the final core shard until the whole streamed pipeline drains. This includes
    /// any remaining child recording and early lift proving.
    pub core_pipeline_tail_after_core_ms: u128,
    pub core_pipeline_wall_ms: u128,
    pub core_recorder_workers: usize,
    pub core_proof_queue_capacity: usize,
    /// Explicit raw/saved-core normalization before the canonical handoff. Zero for streamed core.
    pub raw_core_normalize_wall_ms: u128,
    /// Sum of the non-overlapping per-bin recording wall intervals in raw/saved normalization.
    /// Early-lift work running concurrently with those intervals is excluded.
    pub raw_core_record_wall_ms: u128,
    /// Elapsed from the first completed record bin until every early lift (including an optional
    /// direct-L3 prerecord) drains. This interval intentionally overlaps later record bins.
    pub raw_core_lift_pipeline_wall_ms: u128,
    pub preproved_lift_nodes: usize,
    pub precompress_lift_timings: Vec<NativePrecompressLiftTiming>,
    pub precompress_l3_record_count: usize,
    pub precompress_l3_record_work_sum_ms: u128,
    pub precompress_l3_record_max_ms: u128,
    pub speculative_l3_records_discarded: usize,
    /// S3: producer count-ticket arrival telemetry (streamed route only).
    pub count_ticket: Option<NativeCountTicketTelemetry>,
    pub tree_policy: Option<NativeTreePolicyTelemetry>,
    pub decisions: Vec<NativeLayerDecision>,
    pub node_stats: Vec<NativeCompressNodeStat>,
    pub scheduler_events: Vec<NativeSchedulerEvent>,
    pub native_phases: Vec<NativePhaseTiming>,
    /// Peak concurrent nodes inside bounded canonical parallel layers (currently L2). This is
    /// zero when the direct-L3 route schedules no parallel layer; it is not a process-wide
    /// thread count and does not include precompress recording/lifts.
    pub max_in_flight: usize,
    pub peak_rss_kb: Option<u64>,
    /// Wall time measured from entry to `compress_native`. For a streamed handoff, core recording
    /// and preproved lifts happened before this boundary and are intentionally excluded.
    pub compress_native_wall_ms: u128,
}

/// One-shot gate the streamed pipeline resolves at count-ticket arrival: to the
/// immutable full plan on a ticket, to an error when the producer ends without one (so
/// the pipeline fails cleanly instead of waiting forever). First resolution wins.
pub(crate) struct PlanGate {
    slot: Mutex<Option<Result<Arc<tree_plan::TreePlan>, String>>>,
    ready: Condvar,
}

impl PlanGate {
    pub(crate) fn new() -> Self {
        Self { slot: Mutex::new(None), ready: Condvar::new() }
    }

    pub(crate) fn resolve(&self, value: Result<Arc<tree_plan::TreePlan>, String>) {
        let mut slot = self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(value);
            self.ready.notify_all();
        }
    }

    pub(crate) fn try_get(&self) -> Option<Result<Arc<tree_plan::TreePlan>, String>> {
        self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    pub(crate) fn wait(&self) -> Result<Arc<tree_plan::TreePlan>, String> {
        let mut slot = self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while slot.is_none() {
            slot = self.ready.wait(slot).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        slot.clone().expect("resolved")
    }
}

/// Build the immutable native-recursion tree with the same cap, band table,
/// and policy authority used by the production compression pipeline.
///
/// This is public so fixed-fixture qualification harnesses can exercise an
/// actual production-planned node instead of reconstructing the policy.
pub fn build_tree_plan(
    shard_count: usize,
    worker_hint: usize,
) -> Result<Arc<tree_plan::TreePlan>, DTRecursionProverError> {
    let core_count = u32::try_from(shard_count)
        .ok()
        .and_then(std::num::NonZeroU32::new)
        .ok_or_else(|| runtime(format!("shard count {shard_count} outside the planner domain")))?;
    let worker_hint = u8::try_from(worker_hint)
        .ok()
        .and_then(std::num::NonZeroU8::new)
        .ok_or_else(|| runtime(format!("worker hint {worker_hint} outside 1..=255")))?;
    let arity_cap = NATIVE_MAX_NODE_ARITY as u8;
    let bands = tree_plan::ArityBandTable::current_native(arity_cap)
        .map_err(|err| runtime(format!("native arity bands: {err}")))?;
    let plan = tree_plan::plan_band_aware_even_v1(core_count, arity_cap, worker_hint, bands)
        .map_err(|err| runtime(format!("tree plan for {shard_count} shards: {err}")))?;
    Ok(Arc::new(plan))
}

fn lift_nodes(
    plan: &tree_plan::TreePlan,
) -> Result<Vec<&tree_plan::NodePlan>, DTRecursionProverError> {
    let nodes = plan
        .layers
        .first()
        .map(|layer| {
            layer
                .actions
                .iter()
                .filter_map(|action| match action {
                    tree_plan::NodeAction::Reduce(node) => Some(node),
                    tree_plan::NodeAction::Carry(_) => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if nodes.is_empty() {
        return Err(runtime("TreePlan produced no Lift layer"));
    }
    Ok(nodes)
}

pub(crate) fn lift_spans(
    plan: &tree_plan::TreePlan,
) -> Result<Vec<(usize, usize)>, DTRecursionProverError> {
    lift_nodes(plan).map(|nodes| {
        nodes.into_iter().map(|node| (node.span.start as usize, node.span.end as usize)).collect()
    })
}

pub(crate) fn shard_routes(
    plan: &tree_plan::TreePlan,
) -> Result<Vec<(usize, usize)>, DTRecursionProverError> {
    let spans = lift_spans(plan)?;
    let mut routes = Vec::with_capacity(plan.core_count as usize);
    for (node_index, &(span_start, span_end)) in spans.iter().enumerate() {
        for shard in span_start..span_end {
            debug_assert_eq!(shard, routes.len());
            routes.push((node_index, shard - span_start));
        }
    }
    if routes.len() != plan.core_count as usize {
        return Err(runtime("TreePlan Lift spans do not cover the shard range"));
    }
    Ok(routes)
}

pub(crate) fn lift_is_l3_child(
    plan: &tree_plan::TreePlan,
    node_index: usize,
) -> Result<bool, DTRecursionProverError> {
    let node = *lift_nodes(plan)?
        .get(node_index)
        .ok_or_else(|| runtime(format!("Lift node {node_index} missing from TreePlan")))?;
    Ok(node.output.is_some_and(|output| output.parent == plan.l3.id))
}

pub(crate) fn lift_l3_slot(
    plan: &tree_plan::TreePlan,
    node_index: usize,
) -> Result<Option<usize>, DTRecursionProverError> {
    let node = *lift_nodes(plan)?
        .get(node_index)
        .ok_or_else(|| runtime(format!("Lift node {node_index} missing from TreePlan")))?;
    Ok(node
        .output
        .filter(|output| output.parent == plan.l3.id)
        .map(|output| output.parent_slot as usize))
}

fn tree_policy_telemetry(
    plan: &tree_plan::TreePlan,
    worker_hint: usize,
) -> Result<NativeTreePolicyTelemetry, DTRecursionProverError> {
    let bands = tree_plan::ArityBandTable::current_native(plan.arity_cap)
        .map_err(|err| runtime(format!("native arity bands: {err}")))?;
    Ok(NativeTreePolicyTelemetry {
        requested_version: tree_plan::TREE_POLICY_BAND_AWARE_EVEN_V1,
        selected_version: plan.version,
        arity_cap: plan.arity_cap,
        worker_hint: u8::try_from(worker_hint)
            .map_err(|_| runtime("TreePlan worker hint exceeds u8"))?,
        lift_bands: bands.lift,
        l2_bands: bands.l2,
        l3_bands: bands.l3,
        lift_spans: lift_spans(plan)?,
    })
}

fn elapsed_ms_since(start: Instant) -> u128 {
    start.elapsed().as_millis()
}

fn record_native_phase(
    phases: &mut Vec<NativePhaseTiming>,
    compress_start: Instant,
    phase: &str,
    phase_start: Instant,
) {
    let end_ms = elapsed_ms_since(compress_start);
    let wall_ms = phase_start.elapsed().as_millis();
    phases.push(NativePhaseTiming {
        phase: phase.to_string(),
        start_ms: end_ms.saturating_sub(wall_ms),
        end_ms,
        wall_ms,
    });
}

fn scheduler_decision(
    layer: &str,
    nodes: usize,
    arities: Vec<usize>,
    wall_ms: u128,
    events: &[NativeSchedulerEvent],
) -> NativeLayerDecision {
    let queue_wait_ms_max = events.iter().map(|event| event.queue_wait_ms).max().unwrap_or(0);
    let node_run_ms_max = events.iter().map(|event| event.run_ms).max().unwrap_or(0);
    let join_tail_ms = events.iter().map(|event| event.end_ms).max().map_or(wall_ms, |last_end| {
        events
            .iter()
            .map(|event| event.start_ms.saturating_sub(event.queue_wait_ms))
            .min()
            .map_or(0, |layer_start| layer_start + wall_ms)
            .saturating_sub(last_end)
    });

    NativeLayerDecision {
        layer: layer.into(),
        nodes,
        arities,
        wall_ms,
        queue_wait_ms_max,
        node_run_ms_max,
        join_tail_ms,
    }
}

/// Linux VmHWM (peak resident set) in kB; `None` where /proc is absent (macOS dev).
fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1).and_then(|value| value.parse().ok()))
}

struct PreRecordedL3Child {
    record: BuildingRecord,
    start_ms: u128,
    end_ms: u128,
}

impl PreRecordedL3Child {
    fn wall_ms(&self) -> u128 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A bounded worker pool over one layer's nodes: at most `max_in_flight` nodes are
/// recorded+proved concurrently, outputs land in input order.
/// Note: the replay segments require outputs in shard order; the input-order
/// guarantee is load-bearing, not cosmetic.
fn run_layer<T, F>(
    layer: &str,
    jobs: usize,
    arities: &[usize],
    max_in_flight: usize,
    in_flight_peak: &AtomicUsize,
    compress_start: Instant,
    layer_start: Instant,
    job: F,
) -> Result<
    (Vec<(T, Vec<NativeCompressNodeStat>)>, Vec<NativeSchedulerEvent>),
    DTRecursionProverError,
>
where
    T: Send,
    F: Fn(usize, &mut Vec<NativeCompressNodeStat>) -> Result<T, DTRecursionProverError>
        + Send
        + Sync,
{
    let indices = (0..jobs).collect::<Vec<_>>();
    run_owned_layer_with_completion(
        layer,
        indices,
        arities,
        max_in_flight,
        in_flight_peak,
        compress_start,
        layer_start,
        |_queue_index, job_index, stats| job(job_index, stats),
        |_idx, _value| Ok(()),
    )
}

fn run_owned_layer_with_completion<J, T, F, C>(
    layer: &str,
    jobs: Vec<J>,
    arities: &[usize],
    max_in_flight: usize,
    in_flight_peak: &AtomicUsize,
    compress_start: Instant,
    layer_start: Instant,
    job: F,
    on_complete: C,
) -> Result<
    (Vec<(T, Vec<NativeCompressNodeStat>)>, Vec<NativeSchedulerEvent>),
    DTRecursionProverError,
>
where
    J: Send,
    T: Send,
    F: Fn(usize, J, &mut Vec<NativeCompressNodeStat>) -> Result<T, DTRecursionProverError>
        + Send
        + Sync,
    C: Fn(usize, &T) -> Result<(), DTRecursionProverError> + Send + Sync,
{
    let job_count = jobs.len();
    let token_count = max_in_flight.max(1).min(job_count.max(1));
    let tokens = (Mutex::new(token_count), Condvar::new());
    let in_flight = AtomicUsize::new(0);
    let events = Mutex::new(Vec::with_capacity(job_count));
    let results = jobs
        .into_par_iter()
        .enumerate()
        .map(|(idx, payload)| {
            let mut available = tokens.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while *available == 0 {
                available =
                    tokens.1.wait(available).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            *available -= 1;
            drop(available);

            let node_start = Instant::now();
            let queue_wait_ms = node_start.duration_since(layer_start).as_millis();
            let start_ms = node_start.duration_since(compress_start).as_millis();
            let live = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            in_flight_peak.fetch_max(live, Ordering::SeqCst);
            let mut stats = Vec::new();
            let outcome = job(idx, payload, &mut stats);
            in_flight.fetch_sub(1, Ordering::SeqCst);
            let end_ms = elapsed_ms_since(compress_start);
            events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(
                NativeSchedulerEvent {
                    layer: layer.to_string(),
                    node_index: idx,
                    arity: arities.get(idx).copied().unwrap_or(0),
                    worker_index: rayon::current_thread_index(),
                    queue_wait_ms,
                    start_ms,
                    end_ms,
                    run_ms: node_start.elapsed().as_millis(),
                    shared_rayon_pool_threads: Some(rayon::current_num_threads()),
                },
            );

            let mut available = tokens.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *available += 1;
            tokens.1.notify_one();
            drop(available);
            outcome.and_then(|value| on_complete(idx, &value).map(|()| (value, stats)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut events = events.into_inner().unwrap();
    events.sort_by_key(|event| event.node_index);
    Ok((results, events))
}

fn store_indexed_pipeline_result<T>(
    stage: &str,
    index: usize,
    value: T,
    slots: &mut [Option<T>],
    received: &mut usize,
) -> Result<(), DTRecursionProverError> {
    let slot_count = slots.len();
    let slot = slots.get_mut(index).ok_or_else(|| {
        runtime(format!(
            "{stage} returned out-of-range index {index}; expected fewer than {slot_count}"
        ))
    })?;
    if slot.is_some() {
        return Err(runtime(format!("{stage} returned duplicate index {index}")));
    }
    *slot = Some(value);
    *received += 1;
    Ok(())
}

// ───────────────────────── the backend state ─────────────────────────

/// The lazily-built native backend: the four ladder layers (programs, provers,
/// pk/vk) plus the recording machines, all inside [`NativeLadderContext`].
/// Note: setup determinism is asserted at build, and the config check runs
/// BEFORE the ladder is derived (fail-closed init).
pub struct NativeRecursionBackend<P: NativeProverProvider = CpuNativeProver> {
    ladder: NativeLadderContext<P>,
    last_report: Mutex<Option<NativeCompressReport>>,
}

/// Count-ticket arrival telemetry (S3, `docs/sol-final-op.md` V3 §4): the
/// producer announces the exact final shard count on a dedicated control
/// channel at its finalization event; the consumer stamps when it arrived
/// and how many proofs the recorders had already pulled by then.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct NativeCountTicketTelemetry {
    pub count: u32,
    pub ready_ms: u128,
    pub proofs_received_at_ready: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct CorePrerecordSummary {
    children: Option<usize>,
    child_record_work_sum_ms: u128,
    max_child_ms: u128,
    pipeline_tail_after_core_ms: u128,
    pipeline_wall_ms: u128,
    recorder_workers: usize,
    proof_queue_capacity: usize,
    early_lift_workers: usize,
    raw_core_normalize_wall_ms: u128,
    raw_core_record_wall_ms: u128,
    raw_core_lift_pipeline_wall_ms: u128,
    l3_record_count: usize,
    l3_record_work_sum_ms: u128,
    l3_record_max_ms: u128,
    speculative_l3_records_discarded: usize,
    count_ticket: Option<NativeCountTicketTelemetry>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct L3PrerecordSummary {
    record_count: usize,
    record_work_sum_ms: u128,
    record_max_ms: u128,
    discarded: usize,
}

impl L3PrerecordSummary {
    fn observe(&mut self, record: &PreRecordedL3Child, discarded: bool) {
        let wall_ms = record.wall_ms();
        self.record_count += 1;
        self.record_work_sum_ms += wall_ms;
        self.record_max_ms = self.record_max_ms.max(wall_ms);
        self.discarded += usize::from(discarded);
    }
}

/// Bounded CPU handoff policy used while core shards are still being produced. Carrying this
/// policy into the handoff keeps producer scheduling out of canonical compression.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NativePipelineOptions {
    pub recorder_workers: usize,
    pub proof_queue_capacity: usize,
    pub early_lift_workers: usize,
    pub early_lift_queue_capacity: usize,
}

pub(crate) struct CorePrerecordEntry {
    shard_idx: usize,
    proof_idx: usize,
    record: BuildingRecord,
    wall_ms: u128,
}

impl CorePrerecordEntry {
    pub(crate) const fn wall_ms(&self) -> u128 {
        self.wall_ms
    }
}

fn store_raw_core_record_result(
    first_shard: usize,
    shard_idx: usize,
    result: Result<CorePrerecordEntry, DTRecursionProverError>,
    slots: &mut [Option<Result<CorePrerecordEntry, DTRecursionProverError>>],
    received: &mut usize,
) -> Result<(), DTRecursionProverError> {
    let local_idx = shard_idx.checked_sub(first_shard).ok_or_else(|| {
        runtime(format!(
            "raw core recorder returned shard {shard_idx} before current bin {first_shard}"
        ))
    })?;
    store_indexed_pipeline_result("raw core recorder", local_idx, result, slots, received)
}

#[derive(Debug, Clone, Serialize)]
pub struct NativePrecompressLiftTiming {
    pub node_index: usize,
    pub first_shard: usize,
    pub arity: usize,
    pub ready_ms: u128,
    pub start_ms: u128,
    pub end_ms: u128,
    pub queue_wait_ms: u128,
    pub wall_ms: u128,
    /// Width of the one process-global Rayon pool observed by this worker.
    pub shared_rayon_pool_threads: usize,
}

/// A lift completed before the SDK enters `compress_native`. This request-owned value is moved
/// exactly once and is never cloned or serialized.
pub(crate) struct PreprovedLift {
    pub(crate) node_index: usize,
    first_shard: usize,
    arity: usize,
    proof: Option<SCMachineProof<SC>>,
    l3_child_record: Option<PreRecordedL3Child>,
    stats: Vec<NativeCompressNodeStat>,
    timing: NativePrecompressLiftTiming,
}

/// Transient streamed result. The L3 child record is present exactly when the plan says the
/// lift's parent is L3 (the count is known before the first record; nothing is speculative).
pub(crate) struct EarlyLiftResult {
    lift: PreprovedLift,
    l3_child_record: Option<Result<PreRecordedL3Child, DTRecursionProverError>>,
}

impl EarlyLiftResult {
    fn finish(
        mut self,
        l3_child: bool,
        summary: &mut L3PrerecordSummary,
    ) -> Result<PreprovedLift, DTRecursionProverError> {
        match (l3_child, self.l3_child_record.take()) {
            (true, Some(record)) => {
                let record = record?;
                summary.observe(&record, false);
                self.lift.l3_child_record = Some(record);
            }
            (true, None) => {
                return Err(runtime(format!(
                    "missing L3 child record for direct lift {}",
                    self.lift.node_index
                )));
            }
            (false, Some(_)) => {
                return Err(runtime(format!(
                    "lift {} pre-recorded an L3 child under an L2 plan",
                    self.lift.node_index
                )));
            }
            (false, None) => {}
        }
        Ok(self.lift)
    }
}

/// Request-owned canonical native batch. It is never serialized and never stored in
/// backend-global state.
pub(crate) struct NativeCorePrerecordBatch {
    plan: Arc<tree_plan::TreePlan>,
    request: NativeRecursionRequest,
    shard_count: usize,
    preproved_lifts: Vec<PreprovedLift>,
    summary: CorePrerecordSummary,
    core_vk_digest: [F; dt_stark::DIGEST_SIZE],
}

/// Opaque request-owned handoff for a core run whose shards have already been authenticated by
/// preproved lifts. It retains only public values and the canonical native batch; core shards and
/// stdin are released by the streaming producer. It is intentionally neither `Clone` nor serde.
pub struct NativeCoreHandoff {
    public_values: DTPublicValues,
    batch: NativeCorePrerecordBatch,
}

impl NativeCorePrerecordBatch {
    fn new_streamed(
        plan: Arc<tree_plan::TreePlan>,
        request: NativeRecursionRequest,
        shard_count: usize,
        early_lifts: Vec<EarlyLiftResult>,
        child_wall_ms: Vec<u128>,
        tail_ms: u128,
        pipeline_wall_ms: u128,
        core_vk: &SCStarkVerifyingKey<CoreSC>,
        options: NativePipelineOptions,
        mut l3_summary: L3PrerecordSummary,
        count_ticket: Option<NativeCountTicketTelemetry>,
    ) -> Result<Self, DTRecursionProverError> {
        // S3 progress-contract assertion: an announced count that disagrees
        // with the drained stream is a hard integration error, never a
        // planner input to reconcile.
        if let Some(ticket) = count_ticket {
            if ticket.count as usize != shard_count {
                return Err(runtime(format!(
                    "core count ticket announced {} shards, the stream delivered {shard_count}",
                    ticket.count
                )));
            }
        }
        let preproved_lifts = early_lifts
            .into_iter()
            .enumerate()
            .map(|(node_index, lift)| {
                lift.finish(lift_is_l3_child(&plan, node_index)?, &mut l3_summary)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let summary = CorePrerecordSummary {
            children: Some(child_wall_ms.len()),
            child_record_work_sum_ms: child_wall_ms.iter().sum(),
            max_child_ms: child_wall_ms.iter().copied().max().unwrap_or(0),
            pipeline_tail_after_core_ms: tail_ms,
            pipeline_wall_ms,
            recorder_workers: options.recorder_workers,
            proof_queue_capacity: options.proof_queue_capacity,
            early_lift_workers: options.early_lift_workers,
            raw_core_normalize_wall_ms: 0,
            raw_core_record_wall_ms: 0,
            raw_core_lift_pipeline_wall_ms: 0,
            l3_record_count: l3_summary.record_count,
            l3_record_work_sum_ms: l3_summary.record_work_sum_ms,
            l3_record_max_ms: l3_summary.record_max_ms,
            speculative_l3_records_discarded: l3_summary.discarded,
            count_ticket,
        };
        Ok(Self {
            plan,
            request,
            shard_count,
            preproved_lifts,
            summary,
            core_vk_digest: core_vk_statement_digest(
                &core_vk.commit,
                core_vk.pc_start,
                &core_vk.program_boundary,
                &core_vk.global146_identity,
            ),
        })
    }

    fn new_normalized_raw(
        plan: Arc<tree_plan::TreePlan>,
        request: NativeRecursionRequest,
        shard_count: usize,
        preproved_lifts: Vec<PreprovedLift>,
        child_wall_ms: &[u128],
        normalize_wall_ms: u128,
        record_wall_ms: u128,
        lift_pipeline_wall_ms: u128,
        options: NativePipelineOptions,
        core_vk: &SCStarkVerifyingKey<CoreSC>,
        l3_summary: L3PrerecordSummary,
    ) -> Self {
        Self {
            plan,
            request,
            shard_count,
            preproved_lifts,
            summary: CorePrerecordSummary {
                children: Some(child_wall_ms.len()),
                child_record_work_sum_ms: child_wall_ms.iter().sum(),
                max_child_ms: child_wall_ms.iter().copied().max().unwrap_or(0),
                recorder_workers: options.recorder_workers,
                // Raw shards are already owned in memory; no core-producer proof queue exists.
                proof_queue_capacity: 0,
                early_lift_workers: options.early_lift_workers,
                raw_core_normalize_wall_ms: normalize_wall_ms,
                raw_core_record_wall_ms: record_wall_ms,
                raw_core_lift_pipeline_wall_ms: lift_pipeline_wall_ms,
                l3_record_count: l3_summary.record_count,
                l3_record_work_sum_ms: l3_summary.record_work_sum_ms,
                l3_record_max_ms: l3_summary.record_max_ms,
                speculative_l3_records_discarded: l3_summary.discarded,
                ..Default::default()
            },
            core_vk_digest: core_vk_statement_digest(
                &core_vk.commit,
                core_vk.pc_start,
                &core_vk.program_boundary,
                &core_vk.global146_identity,
            ),
        }
    }
}

impl NativeCoreHandoff {
    pub(crate) fn new(public_values: DTPublicValues, batch: NativeCorePrerecordBatch) -> Self {
        Self { public_values, batch }
    }

    pub fn public_values(&self) -> &DTPublicValues {
        &self.public_values
    }

    pub(crate) fn into_batch(self) -> NativeCorePrerecordBatch {
        self.batch
    }
}

/// Build a native recursion backend with a custom prover provider (e.g. GPU). Skips disk cache.
pub fn new_native_backend_with_provider<P: NativeProverProvider>(
    core_config: &CoreSC,
) -> Result<NativeRecursionBackend<P>, DTRecursionProverError> {
    if core_config.whir_stage_name() != "core" {
        return Err(runtime(format!(
            "native recursion requires a core-stage proof config, got {:?}",
            core_config.whir_stage_name()
        )));
    }
    let _config_hash = assert_config_authority().map_err(runtime)?;
    let ladder =
        build_ladder_with_provider::<P>().map_err(|err| runtime(format!("ladder build: {err}")))?;
    // This uncached provider path does not pass through the disk-cache admission gate.
    validate_frozen_l4_digest(ladder.root_vk())?;
    Ok(NativeRecursionBackend { ladder, last_report: Mutex::new(None) })
}

impl NativeRecursionBackend {
    /// Builds the backend: config-authority gate first, then the four-layer ladder
    /// derivation (~0.3 s per setup, once per process).
    pub fn new(core_config: &CoreSC) -> Result<Self, DTRecursionProverError> {
        if core_config.whir_stage_name() != "core" {
            return Err(runtime(format!(
                "native recursion requires a core-stage proof config, got {:?}",
                core_config.whir_stage_name()
            )));
        }
        let config_hash = assert_config_authority().map_err(runtime)?;
        let cache_dir = native_ladder_cache_dir();
        let ladder = NativeLadderContext::build_with_disk_cache(
            &cache_dir,
            &config_hash,
            VK_L4_FROZEN_DIGEST,
        )
        .map_err(|err| runtime(format!("ladder build: {err}")))?;
        Ok(Self { ladder, last_report: Mutex::new(None) })
    }
}

impl<P: NativeProverProvider> NativeRecursionBackend<P> {
    /// Exports the minimal, application-bound trust root consumed by the browser verifier.
    ///
    /// The returned value contains no proving key or MMCS prover data. Callers must serialize it
    /// into a release artifact and compile those exact bytes into the WASM verifier; it is not a
    /// runtime input protocol.
    pub fn root_verifier_artifact(
        &self,
        core_vk: &SCStarkVerifyingKey<CoreSC>,
    ) -> Result<NativeRootVerifierArtifactV1, DTRecursionProverError> {
        // Re-run the product authority gate at export time so an environment change after backend
        // construction cannot produce an artifact for a different configuration.
        validate_frozen_l4_digest(self.ladder.root_vk())?;
        let authority_hash = assert_config_authority().map_err(runtime)?;
        let authority = read_config_authority().map_err(runtime)?;
        if authority.hash != authority_hash {
            return Err(runtime(
                "WHIR config authority changed while exporting the browser verifier artifact",
            ));
        }
        let expected_core_statement_digest = core_vk_statement_digest(
            &core_vk.commit,
            core_vk.pc_start,
            &core_vk.program_boundary,
            &core_vk.global146_identity,
        );
        Ok(NativeRootVerifierArtifactV1::new(
            authority.contents,
            self.ladder.l4_program.clone(),
            self.ladder.l4_vk.clone(),
            expected_core_statement_digest,
            core_vk.program_boundary.clone(),
        ))
    }

    pub(crate) fn pipeline_options(&self) -> Result<NativePipelineOptions, DTRecursionProverError> {
        fn positive_env(name: &str, default: usize) -> Result<usize, DTRecursionProverError> {
            match std::env::var(name) {
                Ok(value) => {
                    value.parse::<usize>().ok().filter(|value| *value > 0).ok_or_else(|| {
                        runtime(format!("{name} must be a positive integer, got {value:?}"))
                    })
                }
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(err) => Err(runtime(format!("read {name}: {err}"))),
            }
        }

        let default_workers = available_parallelism().min(NATIVE_MAX_NODE_ARITY);
        // GPU mode: force serial(多线程 CUDA context 死锁). Env var 可覆盖。
        let force_serial = std::env::var("DT_NATIVE_SERIAL").map_or(false, |v| v == "1");
        if force_serial {
            return Ok(NativePipelineOptions {
                recorder_workers: 1,
                proof_queue_capacity: 1,
                early_lift_workers: 1,
                early_lift_queue_capacity: 1,
            });
        }
        let recorder_workers = positive_env("DT_NATIVE_RECURSION_RECORD_WORKERS", default_workers)?;
        if recorder_workers > available_parallelism() {
            return Err(runtime(format!(
                "DT_NATIVE_RECURSION_RECORD_WORKERS={recorder_workers} exceeds available parallelism {}",
                available_parallelism()
            )));
        }
        let proof_queue_capacity =
            positive_env("DT_NATIVE_RECURSION_PROOF_QUEUE_CAPACITY", recorder_workers)?;
        let early_lift_workers = tree_plan_worker_hint()?;
        let early_lift_queue_capacity =
            positive_env("DT_NATIVE_RECURSION_EARLY_LIFT_QUEUE_CAPACITY", early_lift_workers)?;
        Ok(NativePipelineOptions {
            recorder_workers,
            proof_queue_capacity,
            early_lift_workers,
            early_lift_queue_capacity,
        })
    }

    /// `proof_idx` is the shard's dense slot within its lift node — the caller
    /// owns the shard-to-node routing from the immutable TreePlan.
    pub(crate) fn build_core_prerecord(
        &self,
        request: &NativeRecursionRequest,
        vk: &SCStarkVerifyingKey<CoreSC>,
        shard: SCShardProof<CoreSC>,
        global_shard_idx: usize,
        proof_idx: usize,
    ) -> Result<CorePrerecordEntry, DTRecursionProverError> {
        let start = Instant::now();
        let record = self
            .ladder
            .record_core_child_record(request, vk, shard, proof_idx)
            .map_err(|err| runtime(format!("core child {global_shard_idx}: {err}")))?;
        Ok(CorePrerecordEntry {
            shard_idx: global_shard_idx,
            proof_idx,
            record,
            wall_ms: start.elapsed().as_millis(),
        })
    }

    pub(crate) fn finish_core_prerecords(
        &self,
        plan: Arc<tree_plan::TreePlan>,
        request: NativeRecursionRequest,
        shard_count: usize,
        early_lifts: Vec<EarlyLiftResult>,
        child_wall_ms: Vec<u128>,
        tail_ms: u128,
        pipeline_wall_ms: u128,
        core_vk: &SCStarkVerifyingKey<CoreSC>,
        options: NativePipelineOptions,
        l3_summary: L3PrerecordSummary,
        count_ticket: Option<NativeCountTicketTelemetry>,
    ) -> Result<NativeCorePrerecordBatch, DTRecursionProverError> {
        NativeCorePrerecordBatch::new_streamed(
            plan,
            request,
            shard_count,
            early_lifts,
            child_wall_ms,
            tail_ms,
            pipeline_wall_ms,
            core_vk,
            options,
            l3_summary,
            count_ticket,
        )
    }

    /// `first_shard` is the node's span start from the immutable TreePlan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_early_lift_bin(
        &self,
        request: &NativeRecursionRequest,
        node_index: usize,
        first_shard: usize,
        entries: Vec<CorePrerecordEntry>,
        pipeline_start: Instant,
        ready_ms: u128,
        l3_parent_slot: Option<usize>,
    ) -> Result<EarlyLiftResult, DTRecursionProverError> {
        if entries.is_empty() || entries.len() > NATIVE_MAX_NODE_ARITY {
            return Err(runtime(format!(
                "early lift node {node_index} has invalid arity {}",
                entries.len()
            )));
        }
        for (proof_idx, entry) in entries.iter().enumerate() {
            let expected_shard = first_shard + proof_idx;
            if entry.shard_idx != expected_shard || entry.proof_idx != proof_idx {
                return Err(runtime(format!(
                    "early lift node {node_index} child mismatch: got shard/proof {}/{}, expected {expected_shard}/{proof_idx}",
                    entry.shard_idx, entry.proof_idx
                )));
            }
        }

        let arity = entries.len();
        let records = entries.into_iter().map(|entry| entry.record).collect();
        let node_start = Instant::now();
        let start_ms = node_start.duration_since(pipeline_start).as_millis();
        let mut stats = Vec::new();
        let proof = self
            .ladder
            .prove_lift_from_child_records(request, records, &mut stats)
            .map_err(|err| runtime(format!("early lift node {node_index}: {err}")))?;
        if stats.len() != 1 {
            return Err(runtime(format!(
                "early lift node {node_index} emitted {} node stats, expected one",
                stats.len()
            )));
        }
        stats[0].lift_bin_ready_ms = Some(ready_ms);
        stats[0].lift_worker_started_ms = Some(start_ms);
        let proof_end_ms = elapsed_ms_since(pipeline_start);
        let (proof, l3_child_record) = if let Some(l3_parent_slot) = l3_parent_slot {
            let record_start_ms = elapsed_ms_since(pipeline_start);
            let record = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.ladder.record_l3_lift_child_record(request, l3_parent_slot, proof)
            })) {
                Ok(result) => result
                    .map(|record| PreRecordedL3Child {
                        record,
                        start_ms: record_start_ms,
                        end_ms: elapsed_ms_since(pipeline_start),
                    })
                    .map_err(|err| runtime(format!("L3 lift child prerecord {node_index}: {err}"))),
                Err(_) => {
                    Err(runtime(format!("L3 lift child prerecord panicked at node {node_index}")))
                }
            };
            (None, Some(record))
        } else {
            (Some(proof), None)
        };
        Ok(EarlyLiftResult {
            lift: PreprovedLift {
                node_index,
                first_shard,
                arity,
                proof,
                l3_child_record: None,
                stats,
                timing: NativePrecompressLiftTiming {
                    node_index,
                    first_shard,
                    arity,
                    ready_ms,
                    start_ms,
                    end_ms: proof_end_ms,
                    queue_wait_ms: start_ms.saturating_sub(ready_ms),
                    wall_ms: proof_end_ms.saturating_sub(start_ms),
                    shared_rayon_pool_threads: rayon::current_num_threads(),
                },
            },
            l3_child_record,
        })
    }

    /// The most recent compress run's scheduler/timing report.
    pub fn last_report(&self) -> Option<NativeCompressReport> {
        self.last_report.lock().unwrap().clone()
    }

    /// Is this proof a native-ladder proof? The presented vk being content-identical
    /// to the pinned vk_L4 (canonical order-independent encoding — its hash-map
    /// fields serialize in per-instance random order, so raw byte equality would
    /// randomly reject deserialized wire vks) is the discriminator: the vk commits
    /// to the machine, so equality means the native verifier is authoritative and
    /// no DSL fallthrough may run; inequality means the proof is not native at all.
    pub fn is_native_proof(
        &self,
        proof: &DTReduceProof<RootSC>,
    ) -> Result<bool, DTRecursionProverError> {
        let phase = Instant::now();
        let is_native = verifying_keys_equal(self.ladder.root_vk(), &proof.vk);
        pcs::whir::profile::add_ms("verify.is_native_vk_cmp_us", phase.elapsed().as_micros());
        Ok(is_native)
    }

    /// The full native verification: root-shrink machine verify plus the definition
    /// checks (root-form digest recomputation, dt_vk thread == the core vk's
    /// statement digest, vk_root == 0, is_complete == 1), via the ladder library's
    /// external check. Callers must have established `is_native_proof`.
    pub fn verify_native(
        &self,
        proof: &DTReduceProof<RootSC>,
        vk: &DTVerifyingKey,
    ) -> Result<(), DTRecursionProverError> {
        self.ladder.external_check(proof, &vk.vk).map_err(runtime)
    }

    /// Normalize an explicit raw/saved core proof into the same owned handoff produced by the
    /// streamed core route. This adapter moves shards directly into lift proving; it never
    /// serializes, clones, or converts proof configurations.
    pub(crate) fn normalize_core_shards(
        &self,
        core_vk: &SCStarkVerifyingKey<CoreSC>,
        shard_proofs: Vec<SCShardProof<CoreSC>>,
    ) -> Result<NativeCorePrerecordBatch, DTRecursionProverError> {
        let shard_count = shard_proofs.len();
        if shard_count == 0 {
            return Err(runtime("compress requires at least one core shard"));
        }

        let request = NativeRecursionRequest::new().map_err(runtime)?;
        let pipeline = self.pipeline_options()?;
        let normalize_start = Instant::now();
        // Raw shards arrive with the count known, so the one immutable plan is
        // installed before the first record and retained in the handoff.
        let plan = build_tree_plan(shard_count, pipeline.early_lift_workers)?;
        let lift_spans = lift_spans(&plan)?;
        let shard_routes = shard_routes(&plan)?;
        let lift_count = lift_spans.len();
        let lift_l3_slots = (0..lift_count)
            .map(|node_index| lift_l3_slot(&plan, node_index))
            .collect::<Result<Vec<_>, _>>()?;

        // Consume the raw proof vector exactly once. At most one arity-sized raw bin is handed to
        // the recorder pool at a time. Once that bin is recorded, its owned proof buffers are
        // already gone and its lift is dispatched before recording begins on the next bin.
        let pipeline_output = std::thread::scope(|scope| {
            let (record_job_tx, record_job_rx) =
                sync_channel::<(usize, SCShardProof<CoreSC>)>(pipeline.proof_queue_capacity);
            let record_job_rx = Arc::new(Mutex::new(record_job_rx));
            let (record_result_tx, record_result_rx) = sync_channel::<(
                usize,
                Result<CorePrerecordEntry, DTRecursionProverError>,
            )>(pipeline.proof_queue_capacity);
            let mut recorder_handles = Vec::with_capacity(pipeline.recorder_workers);
            for _ in 0..pipeline.recorder_workers {
                let record_job_rx = Arc::clone(&record_job_rx);
                let record_result_tx = record_result_tx.clone();
                let request = &request;
                let shard_routes = &shard_routes;
                recorder_handles.push(scope.spawn(move || loop {
                    let job = record_job_rx
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .recv();
                    let Ok((shard_idx, shard)) = job else {
                        break;
                    };
                    let recorded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let (_node_index, proof_idx) = shard_routes[shard_idx];
                        self.build_core_prerecord(request, core_vk, shard, shard_idx, proof_idx)
                    }))
                    .unwrap_or_else(|_| {
                        Err(runtime(format!("raw core recorder panicked at shard {shard_idx}")))
                    });
                    // The BuildingRecord now owns the moved tracegen material.
                    if record_result_tx.send((shard_idx, recorded)).is_err() {
                        break;
                    }
                }));
            }
            drop(record_job_rx);
            drop(record_result_tx);

            let (lift_job_tx, lift_job_rx) = sync_channel::<(usize, Vec<CorePrerecordEntry>, u128)>(
                pipeline.early_lift_queue_capacity,
            );
            let lift_job_rx = Arc::new(Mutex::new(lift_job_rx));
            let (lift_result_tx, lift_result_rx) = sync_channel::<(
                usize,
                Result<EarlyLiftResult, DTRecursionProverError>,
            )>(pipeline.early_lift_workers);
            let mut lift_handles = Vec::with_capacity(pipeline.early_lift_workers);
            for _ in 0..pipeline.early_lift_workers {
                let lift_job_rx = Arc::clone(&lift_job_rx);
                let lift_result_tx = lift_result_tx.clone();
                let request = &request;
                let lift_spans = &lift_spans;
                let lift_l3_slots = &lift_l3_slots;
                lift_handles.push(scope.spawn(move || loop {
                    let job =
                        lift_job_rx.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).recv();
                    let Ok((node_index, entries, ready_ms)) = job else {
                        break;
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.prove_early_lift_bin(
                            request,
                            node_index,
                            lift_spans[node_index].0,
                            entries,
                            normalize_start,
                            ready_ms,
                            lift_l3_slots[node_index],
                        )
                    }))
                    .unwrap_or_else(|_| {
                        Err(runtime(format!("raw early lift worker panicked at node {node_index}")))
                    });
                    if lift_result_tx.send((node_index, result)).is_err() {
                        break;
                    }
                }));
            }
            drop(lift_job_rx);
            drop(lift_result_tx);

            let mut lift_slots = (0..lift_count)
                .map(|_| None)
                .collect::<Vec<Option<Result<EarlyLiftResult, DTRecursionProverError>>>>();
            let mut received_lifts = 0usize;
            let mut dispatched_lifts = 0usize;
            let mut child_wall_ms = Vec::with_capacity(shard_count);
            let mut record_wall_ms = 0u128;
            let mut first_lift_ready_ms = None;
            let mut coordinator_error = None;
            let mut recorder_error = None;
            let mut indexed_shards = shard_proofs.into_iter().enumerate();

            loop {
                let Some(&(span_start, span_end)) = lift_spans.get(dispatched_lifts) else {
                    if indexed_shards.next().is_some() {
                        coordinator_error =
                            Some(runtime("raw shards remain after the last planned lift span"));
                    }
                    break;
                };
                let raw_bin =
                    indexed_shards.by_ref().take(span_end - span_start).collect::<Vec<_>>();
                if raw_bin.is_empty() {
                    break;
                }
                let expected_first_shard = span_start;
                let bin_len = raw_bin.len();
                let record_bin_start = Instant::now();
                let mut record_slots = (0..bin_len)
                    .map(|_| None)
                    .collect::<Vec<Option<Result<CorePrerecordEntry, DTRecursionProverError>>>>();
                let mut sent_records = 0usize;
                let mut received_records = 0usize;
                for mut job in raw_bin {
                    let expected_shard = expected_first_shard + sent_records;
                    if job.0 != expected_shard {
                        coordinator_error = Some(runtime(format!(
                            "raw core recorder input mismatch: got shard {}, expected {expected_shard}",
                            job.0
                        )));
                        break;
                    }
                    loop {
                        match record_job_tx.try_send(job) {
                            Ok(()) => {
                                sent_records += 1;
                                break;
                            }
                            Err(TrySendError::Full(returned)) => {
                                job = returned;
                                match record_result_rx.recv() {
                                    Ok((shard_idx, result)) => {
                                        if let Err(err) = store_raw_core_record_result(
                                            expected_first_shard,
                                            shard_idx,
                                            result,
                                            &mut record_slots,
                                            &mut received_records,
                                        ) {
                                            coordinator_error.get_or_insert(err);
                                        }
                                    }
                                    Err(_) => {
                                        coordinator_error.get_or_insert_with(|| {
                                            runtime(
                                                "raw core recorder result channel closed while dispatch was backpressured",
                                            )
                                        });
                                        break;
                                    }
                                }
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                coordinator_error.get_or_insert_with(|| {
                                    runtime("raw core recorder job channel closed during dispatch")
                                });
                                break;
                            }
                        }
                    }
                    if coordinator_error.is_some() {
                        break;
                    }
                }
                while coordinator_error.is_none() && received_records < sent_records {
                    match record_result_rx.recv() {
                        Ok((shard_idx, result)) => {
                            if let Err(err) = store_raw_core_record_result(
                                expected_first_shard,
                                shard_idx,
                                result,
                                &mut record_slots,
                                &mut received_records,
                            ) {
                                coordinator_error.get_or_insert(err);
                            }
                        }
                        Err(_) => {
                            coordinator_error.get_or_insert_with(|| {
                                runtime("raw core recorder result channel closed before bin drain")
                            });
                        }
                    }
                }
                record_wall_ms += record_bin_start.elapsed().as_millis();
                if coordinator_error.is_some() {
                    break;
                }
                if sent_records != bin_len || received_records != bin_len {
                    coordinator_error = Some(runtime(format!(
                        "raw core recorder bin count mismatch: sent={sent_records}, received={received_records}, expected={bin_len}"
                    )));
                    break;
                }

                let mut entries = Vec::with_capacity(bin_len);
                for (local_idx, slot) in record_slots.into_iter().enumerate() {
                    let expected_shard = expected_first_shard + local_idx;
                    match slot {
                        Some(Ok(entry)) if entry.shard_idx == expected_shard => {
                            child_wall_ms.push(entry.wall_ms());
                            entries.push(entry);
                        }
                        Some(Ok(entry)) => {
                            recorder_error = Some(runtime(format!(
                                "raw core recorder output mismatch at shard {expected_shard}: got {}",
                                entry.shard_idx
                            )));
                            break;
                        }
                        Some(Err(err)) => {
                            recorder_error.get_or_insert(err);
                        }
                        None => {
                            coordinator_error = Some(runtime(format!(
                                "raw core recorder result missing at shard {expected_shard}"
                            )));
                            break;
                        }
                    }
                }
                if coordinator_error.is_some() || recorder_error.is_some() {
                    break;
                }

                let ready_ms = elapsed_ms_since(normalize_start);
                first_lift_ready_ms.get_or_insert(ready_ms);
                let mut job = (dispatched_lifts, entries, ready_ms);
                loop {
                    match lift_job_tx.try_send(job) {
                        Ok(()) => {
                            dispatched_lifts += 1;
                            break;
                        }
                        Err(TrySendError::Full(returned)) => {
                            job = returned;
                            match lift_result_rx.recv() {
                                Ok((node_index, result)) => {
                                    if let Err(err) = store_indexed_pipeline_result(
                                        "raw early lift",
                                        node_index,
                                        result,
                                        &mut lift_slots,
                                        &mut received_lifts,
                                    ) {
                                        coordinator_error.get_or_insert(err);
                                    }
                                }
                                Err(_) => {
                                    coordinator_error.get_or_insert_with(|| {
                                        runtime(
                                            "raw early lift result channel closed while dispatch was backpressured",
                                        )
                                    });
                                    break;
                                }
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            coordinator_error.get_or_insert_with(|| {
                                runtime("raw early lift job channel closed during dispatch")
                            });
                            break;
                        }
                    }
                }
                if coordinator_error.is_some() {
                    break;
                }
            }

            // On an early recorder/coordinator failure, release every unconsumed proof before
            // draining already-dispatched recorder and lift work.
            drop(indexed_shards);
            drop(record_job_tx);
            for _ in record_result_rx {}
            for handle in recorder_handles {
                if handle.join().is_err() {
                    coordinator_error.get_or_insert_with(|| {
                        runtime("raw core recorder worker panicked outside child recording")
                    });
                }
            }

            drop(lift_job_tx);
            for (node_index, result) in lift_result_rx {
                if let Err(err) = store_indexed_pipeline_result(
                    "raw early lift",
                    node_index,
                    result,
                    &mut lift_slots,
                    &mut received_lifts,
                ) {
                    coordinator_error.get_or_insert(err);
                }
            }
            for handle in lift_handles {
                if handle.join().is_err() {
                    coordinator_error.get_or_insert_with(|| {
                        runtime("raw early lift worker panicked outside node proving")
                    });
                }
            }

            if received_lifts != dispatched_lifts {
                coordinator_error.get_or_insert_with(|| {
                    runtime(format!(
                        "raw early lift result count mismatch: received={received_lifts}, dispatched={dispatched_lifts}"
                    ))
                });
            }
            if coordinator_error.is_none() &&
                recorder_error.is_none() &&
                (dispatched_lifts != lift_count || child_wall_ms.len() != shard_count)
            {
                coordinator_error = Some(runtime(format!(
                    "raw normalization coverage mismatch: recorded={}, shards={shard_count}, lifts={dispatched_lifts}, expected_lifts={lift_count}",
                    child_wall_ms.len()
                )));
            }
            if let Some(err) = coordinator_error {
                return Err(err);
            }

            let mut ordered_lifts = Vec::with_capacity(dispatched_lifts);
            let mut first_lift_error = None;
            for (node_index, slot) in lift_slots.into_iter().take(dispatched_lifts).enumerate() {
                match slot {
                    Some(Ok(result)) => ordered_lifts.push(result),
                    Some(Err(err)) if first_lift_error.is_none() => first_lift_error = Some(err),
                    Some(Err(_)) => {}
                    None => {
                        return Err(runtime(format!(
                            "raw early lift result missing at node {node_index}"
                        )));
                    }
                }
            }
            // Lift nodes cover only bins that precede a failed recorder bin, so a lift error is
            // the deterministic first pipeline error when both stages fail.
            if let Some(err) = first_lift_error {
                return Err(err);
            }
            if let Some(err) = recorder_error {
                return Err(err);
            }

            let lift_pipeline_wall_ms = first_lift_ready_ms
                .map(|ready_ms| elapsed_ms_since(normalize_start).saturating_sub(ready_ms))
                .unwrap_or(0);
            Ok((ordered_lifts, child_wall_ms, record_wall_ms, lift_pipeline_wall_ms))
        })?;
        let (lift_results, child_wall_ms, record_wall_ms, lift_pipeline_wall_ms) = pipeline_output;

        let mut l3_summary = L3PrerecordSummary::default();
        let mut normalized = Vec::with_capacity(lift_results.len());
        for (node_index, result) in lift_results.into_iter().enumerate() {
            normalized.push(result.finish(lift_l3_slots[node_index].is_some(), &mut l3_summary)?);
        }

        Ok(NativeCorePrerecordBatch::new_normalized_raw(
            plan,
            request,
            shard_count,
            normalized,
            &child_wall_ms,
            normalize_start.elapsed().as_millis(),
            record_wall_ms,
            lift_pipeline_wall_ms,
            pipeline,
            core_vk,
            l3_summary,
        ))
    }

    /// The canonical native compress stage: normalized preproved lifts in, root-shrink
    /// `DTReduceProof<RootSC>` out — the identical wire type the DSL route emits.
    ///
    /// Tree policy: execute the immutable NodeAction DAG exactly. Carries retain
    /// proof ownership and reducers run at their planned Lift/L2/L3 arities.
    pub(crate) fn compress_native(
        &self,
        vk: &DTVerifyingKey,
        batch: NativeCorePrerecordBatch,
        opts: &DTProverOpts,
    ) -> Result<DTReduceProof<RootSC>, DTRecursionProverError> {
        let NativeCorePrerecordBatch {
            plan,
            request,
            shard_count,
            preproved_lifts,
            summary: prerecord,
            core_vk_digest,
        } = batch;
        if let Some(children) = prerecord.children {
            if children != shard_count {
                return Err(runtime(format!(
                    "native prerecord child count {} does not cover {} core shards",
                    children, shard_count
                )));
            }
        }
        let supplied_vk_digest = core_vk_statement_digest(
            &vk.vk.commit,
            vk.vk.pc_start,
            &vk.vk.program_boundary,
            &vk.vk.global146_identity,
        );
        if core_vk_digest != supplied_vk_digest {
            return Err(runtime(
                "native core handoff was produced for a different core verifying key",
            ));
        }
        let request = &request;

        let start = Instant::now();
        let mut native_phases = Vec::new();
        let phase_start = Instant::now();
        if shard_count == 0 {
            return Err(runtime("compress requires at least one core shard"));
        }
        record_native_phase(&mut native_phases, start, "prerecord_handoff_adopt", phase_start);

        let max_in_flight = opts.recursion_opts.shard_batch_size.max(1);
        let in_flight_peak = AtomicUsize::new(0);
        let mut report = NativeCompressReport {
            shard_count,
            core_child_record_count: prerecord.children.unwrap_or(0),
            core_child_record_work_sum_ms: prerecord.child_record_work_sum_ms,
            core_child_record_max_ms: prerecord.max_child_ms,
            core_pipeline_tail_after_core_ms: prerecord.pipeline_tail_after_core_ms,
            core_pipeline_wall_ms: prerecord.pipeline_wall_ms,
            core_recorder_workers: prerecord.recorder_workers,
            core_proof_queue_capacity: prerecord.proof_queue_capacity,
            raw_core_normalize_wall_ms: prerecord.raw_core_normalize_wall_ms,
            raw_core_record_wall_ms: prerecord.raw_core_record_wall_ms,
            raw_core_lift_pipeline_wall_ms: prerecord.raw_core_lift_pipeline_wall_ms,
            precompress_l3_record_count: prerecord.l3_record_count,
            precompress_l3_record_work_sum_ms: prerecord.l3_record_work_sum_ms,
            precompress_l3_record_max_ms: prerecord.l3_record_max_ms,
            speculative_l3_records_discarded: prerecord.speculative_l3_records_discarded,
            count_ticket: prerecord.count_ticket,
            native_phases,
            ..Default::default()
        };

        // Adopt the already-proved Lift layer into NodeId-keyed move-only
        // storage. The exact Arc<TreePlan> installed before recording is the
        // only topology authority used below.
        let layer_start = Instant::now();
        report.tree_policy = Some(tree_policy_telemetry(&plan, prerecord.early_lift_workers)?);
        let lift_plans = lift_nodes(&plan)?;
        let lift_count = lift_plans.len();
        if preproved_lifts.len() != lift_count {
            return Err(runtime(format!(
                "native Lift handoff has {} nodes, TreePlan has {lift_count}",
                preproved_lifts.len()
            )));
        }
        let mut proof_slots = BTreeMap::<tree_plan::NodeId, NativeReduceChild>::new();
        let mut l3_prerecords = BTreeMap::<tree_plan::NodeId, BuildingRecord>::new();
        for (node_index, lift) in preproved_lifts.into_iter().enumerate() {
            let node = lift_plans[node_index];
            let span_start = node.span.start as usize;
            let span_end = node.span.end as usize;
            if lift.node_index != node_index ||
                lift.first_shard != span_start ||
                lift.arity != span_end - span_start
            {
                return Err(runtime(format!(
                    "preproved Lift {node_index} is off NodeId {:?}: node={} first={} arity={}, expected first={span_start} arity={}",
                    node.id,
                    lift.node_index,
                    lift.first_shard,
                    lift.arity,
                    span_end - span_start
                )));
            }
            let l3_child = node.output.is_some_and(|output| output.parent == plan.l3.id);
            match (l3_child, lift.l3_child_record, lift.proof) {
                (true, Some(record), None) => {
                    if l3_prerecords.insert(node.id, record.record).is_some() {
                        return Err(runtime(format!("duplicate Lift prerecord at {:?}", node.id)));
                    }
                }
                (false, None, Some(proof)) => {
                    if proof_slots.insert(node.id, NativeReduceChild::Lift(proof)).is_some() {
                        return Err(runtime(format!("duplicate Lift proof at {:?}", node.id)));
                    }
                }
                (true, _, _) => {
                    return Err(runtime(format!(
                        "Lift {:?} must carry exactly one L3 prerecord and no proof",
                        node.id
                    )));
                }
                (false, _, _) => {
                    return Err(runtime(format!(
                        "Lift {:?} must carry exactly one owned proof and no L3 prerecord",
                        node.id
                    )));
                }
            }
            report.precompress_lift_timings.push(lift.timing);
            report.node_stats.extend(lift.stats);
        }
        report.preproved_lift_nodes = lift_count;
        record_native_phase(&mut report.native_phases, start, "lift_adopt", layer_start);

        // Execute every planned L2 round directly. Reduce consumes each child
        // with `remove`; Carry performs no proof/record/clone and leaves its
        // source installed for its path-compressed real parent.
        for layer in plan.layers.iter().skip(1) {
            let phase_start = Instant::now();
            let reduce_nodes = layer
                .actions
                .iter()
                .filter_map(|action| match action {
                    tree_plan::NodeAction::Reduce(node) => Some(node),
                    tree_plan::NodeAction::Carry(_) => None,
                })
                .collect::<Vec<_>>();
            let mut jobs = Vec::with_capacity(reduce_nodes.len());
            for node in &reduce_nodes {
                let mut children = Vec::with_capacity(node.children.len());
                for (slot, binding) in node.children.iter().enumerate() {
                    if binding.local_slot as usize != slot {
                        return Err(runtime(format!(
                            "TreePlan node {:?} has non-dense child slot {} at {slot}",
                            node.id, binding.local_slot
                        )));
                    }
                    let tree_plan::SourceNodeId::Node(source) = binding.source else {
                        return Err(runtime(format!(
                            "L2 node {:?} references a core shard",
                            node.id
                        )));
                    };
                    children.push(proof_slots.remove(&source).ok_or_else(|| {
                        runtime(format!(
                            "L2 node {:?} could not move child {:?} from its NodeId slot",
                            node.id, source
                        ))
                    })?);
                }
                jobs.push((node.id, children));
            }
            for action in &layer.actions {
                if let tree_plan::NodeAction::Carry(carry) = action {
                    if let tree_plan::SourceNodeId::Node(source) = carry.source {
                        if !proof_slots.contains_key(&source) &&
                            !l3_prerecords.contains_key(&source)
                        {
                            return Err(runtime(format!(
                                "Carry at depth {} lost source {:?}",
                                layer.depth, source
                            )));
                        }
                    }
                }
            }
            let arities = jobs.iter().map(|(_, children)| children.len()).collect::<Vec<_>>();
            let node_ids = jobs.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let job_count = jobs.len();
            let jobs = Mutex::new(
                jobs.into_iter().map(|(_, children)| Some(children)).collect::<Vec<_>>(),
            );
            record_native_phase(&mut report.native_phases, start, "l2_plan", phase_start);
            let layer_start = Instant::now();
            let label = format!("L2(depth={})", layer.depth);
            let (results, events) = run_layer(
                &label,
                job_count,
                &arities,
                max_in_flight,
                &in_flight_peak,
                start,
                layer_start,
                |job_index, stats| {
                    let children = jobs
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_mut(job_index)
                        .and_then(Option::take)
                        .ok_or_else(|| {
                            runtime(format!(
                                "TreePlan L2 node {:?} lost its move-only job",
                                node_ids[job_index]
                            ))
                        })?;
                    self.ladder
                        .prove_l2(request, children, stats)
                        .map_err(|err| runtime(format!("L2 node {:?}: {err}", node_ids[job_index])))
                },
            )?;
            let layer_wall_ms = layer_start.elapsed().as_millis();
            for ((proof, stats), node_id) in results.into_iter().zip(&node_ids) {
                report.node_stats.extend(stats);
                if proof_slots.insert(*node_id, NativeReduceChild::L2(proof)).is_some() {
                    return Err(runtime(format!("duplicate L2 result at {node_id:?}")));
                }
            }
            report.decisions.push(scheduler_decision(
                &label,
                job_count,
                arities,
                layer_wall_ms,
                &events,
            ));
            report.scheduler_events.extend(events);
        }

        // L3 may mix carried Lift prerecords and L2 proofs. Install every
        // child by its plan-provided dense local slot, moving it exactly once.
        let layer_start = Instant::now();
        let l3_start_ms = elapsed_ms_since(start);
        let l3_arity = plan.l3.children.len();
        let mut l3_child_records = Vec::with_capacity(l3_arity);
        for (slot, binding) in plan.l3.children.iter().enumerate() {
            if binding.local_slot as usize != slot {
                return Err(runtime(format!(
                    "L3 has non-dense child slot {} at {slot}",
                    binding.local_slot
                )));
            }
            let tree_plan::SourceNodeId::Node(source) = binding.source else {
                return Err(runtime("L3 references a core shard"));
            };
            if let Some(record) = l3_prerecords.remove(&source) {
                l3_child_records.push(record);
            } else {
                let child = proof_slots.remove(&source).ok_or_else(|| {
                    runtime(format!("L3 could not move planned child {source:?}"))
                })?;
                l3_child_records.push(
                    self.ladder
                        .record_l3_child_record(request, slot, child)
                        .map_err(|err| runtime(format!("record L3 child {source:?}: {err}")))?,
                );
            }
        }
        if !proof_slots.is_empty() || !l3_prerecords.is_empty() {
            return Err(runtime(format!(
                "TreePlan execution left {} proofs and {} prerecords unconsumed",
                proof_slots.len(),
                l3_prerecords.len()
            )));
        }
        let mut stats = Vec::new();
        let l3 = self
            .ladder
            .prove_l3_from_child_records(request, l3_arity, l3_child_records, &mut stats)
            .map_err(|err| runtime(format!("L3 node {:?}: {err}", plan.l3.id)))?;

        let l3_end_ms = elapsed_ms_since(start);
        let l3_event = NativeSchedulerEvent {
            layer: "L3".into(),
            node_index: 0,
            arity: l3_arity,
            worker_index: None,
            queue_wait_ms: 0,
            start_ms: l3_start_ms,
            end_ms: l3_end_ms,
            run_ms: layer_start.elapsed().as_millis(),
            shared_rayon_pool_threads: None,
        };
        let l3_events = [l3_event.clone()];
        report.decisions.push(scheduler_decision(
            "L3",
            1,
            vec![l3_arity],
            layer_start.elapsed().as_millis(),
            &l3_events,
        ));
        report.scheduler_events.push(l3_event);
        if let Some(l3_shard) = l3.shard_proofs.first() {
            d21_duplicate_census("l3", l3_shard);
        }
        let layer_start = Instant::now();
        let l4_start_ms = elapsed_ms_since(start);
        let l4 = self
            .ladder
            .prove_l4(request, l3, true, &mut stats)
            .map_err(|err| runtime(format!("L4 node: {err}")))?;
        let l4_end_ms = elapsed_ms_since(start);
        let l4_event = NativeSchedulerEvent {
            layer: "L4".into(),
            node_index: 0,
            arity: 1,
            worker_index: None,
            queue_wait_ms: 0,
            start_ms: l4_start_ms,
            end_ms: l4_end_ms,
            run_ms: layer_start.elapsed().as_millis(),
            shared_rayon_pool_threads: None,
        };
        let l4_events = [l4_event.clone()];
        report.decisions.push(scheduler_decision(
            "L4",
            1,
            vec![1],
            layer_start.elapsed().as_millis(),
            &l4_events,
        ));
        report.scheduler_events.push(l4_event);
        report.node_stats.extend(stats);

        let root_postprocess_start = Instant::now();
        let reduce_proof =
            l4.shard_proofs.into_iter().next().ok_or_else(|| runtime("empty L4 proof"))?;
        // Browser transport keeps the first preprocessed-input opening so the final proof is
        // self-contained. Native product verification is full-only and never reconstructs an
        // omitted batch from prover-side state.
        d21_duplicate_census_root("root", &reduce_proof);
        let reduce = DTReduceProof { vk: self.ladder.root_vk().clone(), proof: reduce_proof };
        record_native_phase(
            &mut report.native_phases,
            start,
            "root_postprocess",
            root_postprocess_start,
        );

        report.max_in_flight = in_flight_peak.load(Ordering::SeqCst);
        report.peak_rss_kb = peak_rss_kb();
        report.compress_native_wall_ms = start.elapsed().as_millis();
        for decision in &report.decisions {
            tracing::info!(
                "native scheduler: layer={} nodes={} arities={:?} wall_ms={}",
                decision.layer,
                decision.nodes,
                decision.arities,
                decision.wall_ms
            );
        }
        tracing::info!(
            "native compress: shards={} max_in_flight={} peak_rss_kb={:?} total_ms={}",
            shard_count,
            report.max_in_flight,
            report.peak_rss_kb,
            report.compress_native_wall_ms
        );
        // Stage ledger (opt-in, output-only): the full scheduler + shape report
        // as one JSON line.
        if let Some(dir) = crate::stage_ledger::ledger_dir() {
            if let Ok(value) = serde_json::to_value(&report) {
                crate::stage_ledger::append(&dir, "native-report.jsonl", &value);
            }
        }
        *self.last_report.lock().unwrap() = Some(report);
        Ok(reduce)
    }
}

/// Opt-in diagnostic (`DT_NATIVE_D21_CENSUS=1`): per tree, count exact-duplicate
/// FULL opened rows at distinct query slots and exact-zero rows, with
/// serialized-byte equivalents (bincode of the row values). Trees: input rounds
/// (`query_openings.per_query`, per matrix), per-round IOPP groups
/// (`round_iopp`), and legacy global IOPP commit-phase rounds (`iopp_queries`).
// The census reads the opening proof's fields structurally, which cannot be
// done through the `MlPcsOpeningProof<C>` associated-type projection in a
// generic fn — so a macro stamps one monomorphic copy per config (the l3
// census sees Poseidon2 proofs, the root census SHA256 proofs).
macro_rules! d21_duplicate_census_impl {
    ($fn_name:ident, $cfg:ty) => {
fn $fn_name(label: &str, proof: &SCShardProof<$cfg>) {
    if std::env::var("DT_NATIVE_D21_CENSUS").is_err() {
        return;
    }
    use std::collections::HashMap;
    let mut total_dup_bytes = 0usize;
    let mut report = |tree: String, rows: Vec<Vec<u8>>| {
        let queries = rows.len();
        let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut zero_rows = 0usize;
        let mut zero_bytes = 0usize;
        let mut dup_bytes = 0usize;
        for row in rows {
            // A row of all-zero field limbs bincodes to len prefix + zero
            // bytes; detect zeros on the value bytes (skip the 8-byte len).
            if row.len() > 8 && row[8..].iter().all(|&b| b == 0) {
                zero_rows += 1;
                zero_bytes += row.len();
            }
            *counts.entry(row).or_default() += 1;
        }
        let mut dup_rows = 0usize;
        for (row, count) in &counts {
            if *count > 1 {
                dup_rows += count - 1;
                dup_bytes += (count - 1) * row.len();
            }
        }
        total_dup_bytes += dup_bytes;
        println!(
            "native_d21_census label={label} tree={tree} queries={queries} dup_rows={dup_rows} \
dup_bytes={dup_bytes} zero_rows={zero_rows} zero_bytes={zero_bytes}"
        );
    };

    let per_query = &proof.opening_proof.query_openings.per_query;
    if let Some(first) = per_query.first() {
        for round in 0..first.len() {
            let mats = first[round].opened_values.len();
            for mat in 0..mats {
                let rows = per_query
                    .iter()
                    .map(|q| bincode::serialize(&q[round].opened_values[mat]).unwrap_or_default())
                    .collect();
                report(format!("input{round}/mat{mat}"), rows);
            }
        }
    }
    if let Some(round_iopp) = &proof.opening_proof.round_iopp {
        for (round_idx, round) in round_iopp.rounds.iter().enumerate() {
            let rows = round
                .query_proofs
                .iter()
                .map(|qp| bincode::serialize(&qp.current_opening.opened_values).unwrap_or_default())
                .collect();
            report(format!("iopp{round_idx}"), rows);
        }
    }
    if !proof.opening_proof.iopp_queries.is_empty() {
        let phases = proof.opening_proof.iopp_queries[0].commit_phase_openings.len();
        for phase in 0..phases {
            let rows = proof
                .opening_proof
                .iopp_queries
                .iter()
                .map(|q| {
                    let step = &q.commit_phase_openings[phase];
                    bincode::serialize(&(&step.sibling_value, &step.opened_values))
                        .unwrap_or_default()
                })
                .collect();
            report(format!("legacy_iopp{phase}"), rows);
        }
    }
    println!("native_d21_census label={label} total_dup_bytes={total_dup_bytes}");
}
    };
}

d21_duplicate_census_impl!(d21_duplicate_census, SC);
d21_duplicate_census_impl!(d21_duplicate_census_root, RootSC);

#[cfg(test)]
mod tests {
    use native_recursion::compress_dt::root_vk_digest;
    use p3_field::PrimeField32;

    use super::*;

    /// The lazily-built vk_L4 must equal the pinned digest constant.
    /// Note: the eprintln keeps the measured value in the test log so a legitimate
    /// re-pin can copy the new digest from evidence rather than from memory.
    #[test]
    fn vk_l4_digest_matches_the_frozen_pin() {
        let backend = NativeRecursionBackend::<CpuNativeProver>::new(&CoreSC::default())
            .expect("backend init");
        let vk = backend.ladder.root_vk();
        let digest = root_vk_digest(vk);
        let got: Vec<u32> = digest.iter().map(|limb| limb.as_canonical_u32()).collect();
        eprintln!("vk_L4 statement digest = {got:?}");
        assert_eq!(got.as_slice(), VK_L4_FROZEN_DIGEST.as_slice(), "vk_L4 drifted from the freeze");
    }

    #[test]
    fn tree_plan_inserts_l2_only_when_the_lift_frontier_requires_it() {
        assert_eq!(build_tree_plan(120, 3).expect("plan").layers.len(), 1);
        assert_eq!(build_tree_plan(121, 3).expect("plan").layers.len(), 1);
        assert!(build_tree_plan(122, 3).expect("plan").layers.len() > 1);
    }

    #[test]
    fn band_aware_routing_records_inputs_and_uses_safe_fallback() {
        let two = build_tree_plan(19, 2).expect("two-worker plan");
        assert_eq!(lift_spans(&two).expect("spans"), vec![(0, 10), (10, 19)]);
        let two_telemetry = tree_policy_telemetry(&two, 2).expect("telemetry");
        assert_eq!(two_telemetry.selected_version, tree_plan::TREE_POLICY_MIN_NODES_EVEN_V1);

        let three = build_tree_plan(19, 3).expect("three-worker plan");
        assert_eq!(lift_spans(&three).expect("spans"), vec![(0, 10), (10, 19)]);
        let telemetry = tree_policy_telemetry(&three, 3).expect("telemetry");
        assert_eq!(telemetry.selected_version, tree_plan::TREE_POLICY_MIN_NODES_EVEN_V1);
        assert_eq!(telemetry.worker_hint, 3);
        assert_eq!(telemetry.lift_bands, vec![9, 11]);
        assert_eq!(telemetry.l3_bands, vec![2, 11]);
    }

    #[test]
    fn empty_native_cache_dir_override_is_treated_as_unset() {
        assert_eq!(
            native_ladder_cache_dir_from_override(Some(std::ffi::OsString::new())),
            native_ladder_cache_dir_from_override(None)
        );
    }
}
