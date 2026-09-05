//! The fixed native compression ladder primitives consumed by the SDK backend.
//! Product tree orchestration lives in `dt-prover`; this module owns only the
//! frozen recursion machines and their typed prove/verify operations.
//! Note: no AIR content lives here; changing the machines re-keys every verifying key.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use bincode::Options;
use dt_core_machine::reduce::DTReduceProof;
use dt_stark::{
    air::MachineAir,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, SCStarkGenericConfig},
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
    },
};
use p3_field::{AbstractField, PrimeField32};
use p3_maybe_rayon::prelude::*;
use pcs::basefold::mlpcs::MlPCS;
use polyair::prover::SCMachineProver;
use serde::{Deserialize, Serialize};

use crate::{
    config::{RootSC, DIGEST_SIZE, D_EF, F, SC},
    machine_dt::{
        build_core_native_recursion_program, build_dual_segment_reduce_program,
        build_native_recursion_program, build_root_shrink_program, core_recording_machine,
        finalize_building_record, merge_child_proof_shard_records, native_metadata_from_machine,
        native_recording_machine, native_recording_machine_for_stage,
        native_recursion_prover_with_config, native_recursion_prover_with_config_and_provider,
        native_root_shrink_prover, native_root_shrink_prover_with_provider, native_shrink_prover,
        native_shrink_prover_with_provider, prove_recursion_with_metrics, record_core_proof_shard,
        record_native_proof_shard, record_native_proof_shard_in_segment,
        validate_l2_bootstrap_fixed_point, verify_recursion, verify_root_recursion_shard,
        CoreRecordingMachine, CpuNativeProver, NativeProverFor, NativeProverProvider,
        NativeRecordingMachine, NativeRecursionAssemblyError, NativeRecursionAssemblyResult,
        NativeRecursionProver, NativeRootProver, NATIVE_ROOT_SHRINK_DEGREE_FLOOR,
        NATIVE_SHRINK_DEGREE_FLOOR,
    },
    native_air_dt::{
        validate_final_replay_layout, validate_l2_bootstrap_layout, NATIVE_AIR_REGISTRY_VERSION,
    },
    statement_dt::{
        core_vk_statement_digest, native_vk_statement_digest, validate_native_root_global_interval,
        NATIVE_PV_DIGEST_START, NATIVE_PV_DT_VK_DIGEST_START, NATIVE_PV_GLOBAL_INTERVAL_END,
        NATIVE_PV_GLOBAL_INTERVAL_START, NATIVE_PV_IS_COMPLETE, NATIVE_PV_VK_ROOT_START,
        NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
        STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
    },
    statement_hash_air_dt::root_public_values_digest,
    symbolic_expr_adapter_dt::{RecursionPolyAirLeaf, RecursionPolyAirOp},
    symbolic_ir_dt::RecursionPolyAirDerivedRoot,
    system_dt::{
        BuildingRecord, CoreRecordingChallenger, FinalizedRecord, RecordingSC, RecordingStage,
        RecursionNativeProgram, RecursionRecord, RecursionRecordProfileSnapshot,
        RecursionStatementRole, ReplayCompatibleProofConfig, StatementConfigRow,
    },
    transcript_dt::poseidon2::{RecursionPoseidon2Memo, RecursionPoseidon2MemoSnapshot},
    verifier_dt::{require_full_root_input_opening, require_safe_root_polyair_shape},
};

const NODE_DIAGNOSTICS_ENV: &str = "DT_NATIVE_RECURSION_NODE_DIAGNOSTICS";
const POST_PROVE_VERIFY_ENV: &str = "DT_NATIVE_RECURSION_POST_PROVE_VERIFY";
const LADDER_CACHE_REBUILD_ENV: &str = "DT_NATIVE_RECURSION_CACHE_REBUILD";

fn node_diagnostics_enabled() -> bool {
    crate::debug_prints_enabled() ||
        std::env::var(NODE_DIAGNOSTICS_ENV)
            .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

fn post_prove_verify_enabled() -> bool {
    cfg!(test) ||
        std::env::var(POST_PROVE_VERIFY_ENV)
            .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

/// One end-to-end native-recursion request. Each recording proof gets a proof-local output-only
/// Poseidon2 memo; merge aggregates telemetry but deliberately discards the proof tables. The
/// request itself is neither cloneable nor serializable and never crosses the in-process ladder
/// boundary.
pub struct NativeRecursionRequest {
    poseidon2_memo: RecursionPoseidon2Memo,
}

impl NativeRecursionRequest {
    pub fn new() -> NativeRecursionAssemblyResult<Self> {
        tracing::info!(
            provider_canonicalization = "single-complete-key-domain",
            poseidon2_memo_scope = "proof-local",
            poseidon2_memo_scope_source = "audited-policy",
            "native recursion request initialized canonical provider semantics"
        );
        Ok(Self { poseidon2_memo: RecursionPoseidon2Memo::default() })
    }

    /// Create a child view over the request-owned output-only permutation memo.
    pub fn recording_seed(&self, mut seed: CoreRecordingChallenger) -> CoreRecordingChallenger {
        seed.record_mut().poseidon2_memo = self.poseidon2_memo.fork_isolated();
        seed
    }

    pub fn ensure_owns_records(
        &self,
        layer: &str,
        records: &[BuildingRecord],
    ) -> NativeRecursionAssemblyResult<()> {
        if let Some((idx, _)) = records.iter().enumerate().find(|(_, record)| {
            !self.poseidon2_memo.shares_request_with(&record.record().poseidon2_memo)
        }) {
            return Err(validation(format!(
                "{layer} child record {idx} belongs to a different native recursion request"
            )));
        }
        Ok(())
    }
}

/// GPU device-matrices telemetry attached only to production GPU recursion
/// nodes. CPU-native callers leave this absent instead of fabricating zeroes.
#[derive(Debug, Clone, Serialize)]
pub struct NativeDeviceMatricesStat {
    pub role: String,
    pub arity: usize,
    pub record_generation: u64,
    pub source_count: usize,
    pub proof_ready_to_trace_matrices_ready_ms: u128,
    pub compact_dto_ms: u128,
    pub prepare_ms: u128,
    pub pool_gate_ms: u128,
    pub tracegen_preparation_ms: u128,
    pub pass_a_ms: u128,
    pub pass_a_prepare_ms: u128,
    pub pass_a_canonicalize_ms: u128,
    pub pass_a_summary_barrier_ms: u128,
    pub pass_a_publish_reduce_ms: u128,
    pub exact_admission_ms: u128,
    pub pass_b_ms: u128,
    pub matrix_enqueue_and_sync_ms: u128,
    pub lease_handoff_ms: u128,
    pub compact_h2d_bytes: usize,
    pub summary_d2h_bytes: usize,
    pub prohibited_payload_h2d_bytes: usize,
    pub prohibited_payload_d2h_bytes: usize,
    pub raw_merkle_candidates: usize,
    pub unique_merkle_rows: u64,
    pub unique_poseidon_rows: u64,
    pub poseidon_multiplicity: u64,
    pub range8_rows: u64,
    pub range21_rows: u64,
    pub range_multiplicity: u64,
    pub ready_bundle_stored_bytes: usize,
    pub ready_bundle_total_bytes: usize,
    pub workspace_peak_bytes: usize,
    pub mempool_used_peak_bytes: usize,
    pub mempool_reserved_peak_bytes: usize,
    pub mempool_used_current_bytes: usize,
    pub mempool_reserved_current_bytes: usize,
    pub consumer_record_generation: u64,
    pub consumer_included_matrices: usize,
    pub consumer_stored_cells: usize,
    pub consumer_total_cells: usize,
    pub compatibility_d2h_ms: u128,
    pub compatibility_transpose_ms: u128,
    pub compatibility_receipt_check_ms: u128,
    pub compatibility_plan_check_ms: u128,
    pub compatibility_release_sync_ms: u128,
    pub compatibility_total_ms: u128,
    pub compatibility_d2h_bytes: usize,
}

/// Per-node stats row produced by every prove in the ladder: per-chip shape data,
/// pool aggregates, and the record/tracegen/prove timing split.
/// Orchestration-layer data only — no AIR content.
#[derive(Debug, Clone, Serialize)]
pub struct NativeCompressNodeStat {
    pub kind: String,
    pub record_generation: u64,
    pub device_matrices: Option<NativeDeviceMatricesStat>,
    /// Full wire size, populated only when `DT_NATIVE_RECURSION_NODE_DIAGNOSTICS=1`.
    pub proof_bytes: Option<usize>,
    pub diagnostics_enabled: bool,
    pub diagnostic_census_ms: u128,
    pub proof_size_ms: u128,
    /// Record building, finalization, and source registration time before trace generation.
    pub record_ms: u128,
    /// Total prove wall time. The stage fields below form a complete accounting; any framework
    /// overhead between their timers is reported explicitly as `prove_residual_ms`.
    pub prove_ms: u128,
    /// Host-side post-prove verification time. Absent on the normal product path; enable the
    /// diagnostic with `DT_NATIVE_RECURSION_POST_PROVE_VERIFY=1` (unit tests enable it
    /// implicitly).
    pub post_prove_verify_ms: Option<u128>,
    /// Lift-bin readiness relative to the core/lift pipeline start.
    /// `None` for nodes outside the early Lift pipeline.
    pub lift_bin_ready_ms: Option<u128>,
    /// Lift worker entry relative to the core/lift pipeline start.
    /// `None` for nodes outside the early Lift pipeline.
    pub lift_worker_started_ms: Option<u128>,
    pub prove_started_unix_ms: u128,
    pub prove_finished_unix_ms: u128,
    pub record_profile: RecursionRecordProfileSnapshot,
    pub poseidon2_memo: RecursionPoseidon2MemoSnapshot,
    /// Exact admission-plan chip heights in deterministic machine order.
    pub planned_chip_log_heights: Vec<(String, u8)>,
    /// Parallel row-count admission from already-owned workspace artifacts/events.
    pub row_count_admission_ms: u128,
    /// Ordered exact-plan fold inside admission.
    pub trace_plan_fold_ms: u128,
    pub tracegen_ms: u128,
    pub budget_ms: u128,
    pub pool_gate_ms: u128,
    pub commit_ms: u128,
    pub commit_profile: BTreeMap<String, u128>,
    pub open_ms: u128,
    pub open_profile: BTreeMap<String, u128>,
    pub prove_residual_ms: u128,
    pub tallest_log_height: usize,
    /// Per-chip shape rows (chip, padded height, stored height, width, perm width,
    /// interactions, constraints) from the prove metrics.
    pub chips: Vec<crate::machine_dt::RecursionTraceCost>,
    pub poseidon2_unique: usize,
    pub poseidon2_total: u64,
    pub merkle_rows: usize,
    /// Detailed row classification is absent when node diagnostics are disabled.
    pub merkle_leaf_rows: Option<usize>,
    pub merkle_node_rows: Option<usize>,
    pub merkle_union_census: Option<MerkleUnionCensus>,
    pub dag_node_mix_census: Option<DagNodeMixCensus>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MerkleUnionCensus {
    pub total_rows: usize,
    pub distinct_nodes: usize,
    pub duplicate_rows: usize,
    pub duplicate_bps: u32,
    pub trees: Vec<MerkleUnionTreeCensus>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MerkleUnionTreeCensus {
    pub proof_idx: usize,
    pub commit_id: usize,
    pub total_rows: usize,
    pub distinct_nodes: usize,
    pub duplicate_rows: usize,
    pub duplicate_bps: u32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DagNodeMixCensus {
    pub mul_nodes: usize,
    pub single_add_sub_consumer_muls: usize,
    pub single_add_sub_consumer_bps: u32,
    pub chips: Vec<DagNodeMixChipCensus>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DagNodeMixChipCensus {
    pub static_chip_id: usize,
    pub chip_name: String,
    pub mul_nodes: usize,
    pub single_add_sub_consumer_muls: usize,
    pub single_add_sub_consumer_bps: u32,
}

impl NativeCompressNodeStat {
    /// Total padded cells across chips (padded height × main width).
    pub fn padded_cells(&self) -> u64 {
        self.chips.iter().map(|c| c.height as u64 * c.width as u64).sum()
    }

    /// Total stored (actual) cells across chips.
    pub fn stored_cells(&self) -> u64 {
        self.chips.iter().map(|c| c.stored_height as u64 * c.width as u64).sum()
    }
}

fn ratio_bps(numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        0
    } else {
        ((numerator as u128 * 10_000) / denominator as u128) as u32
    }
}

fn unix_now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_millis())
}

fn merkle_union_census(record: &RecursionRecord) -> MerkleUnionCensus {
    let mut trees = BTreeMap::<(usize, usize), (usize, BTreeSet<(usize, usize, usize)>)>::new();
    for proof in &record.proof_records {
        for row in proof.merkle_path.rows() {
            let entry =
                trees.entry((row.proof_idx, row.commit_id)).or_insert_with(|| (0, BTreeSet::new()));
            entry.0 += 1;
            entry.1.insert((row.commit_id, row.level, row.cur_idx));
        }
    }

    let trees = trees
        .into_iter()
        .map(|((proof_idx, commit_id), (total_rows, nodes))| {
            let distinct_nodes = nodes.len();
            let duplicate_rows = total_rows.saturating_sub(distinct_nodes);
            MerkleUnionTreeCensus {
                proof_idx,
                commit_id,
                total_rows,
                distinct_nodes,
                duplicate_rows,
                duplicate_bps: ratio_bps(duplicate_rows, total_rows),
            }
        })
        .collect::<Vec<_>>();
    let total_rows: usize = trees.iter().map(|tree| tree.total_rows).sum();
    let distinct_nodes: usize = trees.iter().map(|tree| tree.distinct_nodes).sum();
    let duplicate_rows = total_rows.saturating_sub(distinct_nodes);
    MerkleUnionCensus {
        total_rows,
        distinct_nodes,
        duplicate_rows,
        duplicate_bps: ratio_bps(duplicate_rows, total_rows),
        trees,
    }
}

fn dag_node_mix_census(program: &RecursionNativeProgram<F>) -> DagNodeMixCensus {
    let mut chips = Vec::new();
    let mut total_mul_nodes = 0;
    let mut total_single_add_sub_consumer_muls = 0;
    for chip in &program.constraint_program.chips {
        let mut consumers = vec![0usize; chip.node_table.len()];
        let mut add_sub_consumers = vec![0usize; chip.node_table.len()];
        for node in &chip.node_table {
            match node.op {
                RecursionPolyAirOp::Add { lhs, rhs } | RecursionPolyAirOp::Sub { lhs, rhs } => {
                    bump_consumer(&mut consumers, lhs);
                    bump_consumer(&mut consumers, rhs);
                    bump_consumer(&mut add_sub_consumers, lhs);
                    bump_consumer(&mut add_sub_consumers, rhs);
                }
                RecursionPolyAirOp::Mul { lhs, rhs } => {
                    bump_consumer(&mut consumers, lhs);
                    bump_consumer(&mut consumers, rhs);
                }
                RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, .. } => {
                    bump_consumer(&mut consumers, lhs);
                    bump_consumer(&mut consumers, rhs);
                    bump_consumer(&mut consumers, addend);
                }
                RecursionPolyAirOp::Neg { input } => {
                    bump_consumer(&mut consumers, input);
                }
                RecursionPolyAirOp::Leaf(RecursionPolyAirLeaf::Precomputed { index }) => {
                    if let Some(root_node) = precompute_root_node_for_census(chip, index) {
                        bump_consumer(&mut consumers, root_node);
                    }
                }
                _ => {}
            }
        }
        let lookup_roots = chip.lookup_multiplicity_roots.len();
        for root in chip.derived_roots.iter().filter_map(|root| match root {
            RecursionPolyAirDerivedRoot::PrecomputeLc { index, root_node_id }
                if *index < lookup_roots =>
            {
                Some(*root_node_id)
            }
            _ => None,
        }) {
            bump_consumer(&mut consumers, root);
        }
        for root in &chip.lookup_multiplicity_roots {
            bump_consumer(&mut consumers, root.root_node_id);
        }
        for root in &chip.gate_roots {
            bump_consumer(&mut consumers, root.root_node_id);
        }

        let mut mul_nodes = 0;
        let mut single_add_sub_consumer_muls = 0;
        for node in &chip.node_table {
            if !matches!(node.op, RecursionPolyAirOp::Mul { .. }) {
                continue;
            }
            mul_nodes += 1;
            let idx = node.node_id as usize;
            if consumers.get(idx).copied().unwrap_or(0) == 1 &&
                add_sub_consumers.get(idx).copied().unwrap_or(0) == 1
            {
                single_add_sub_consumer_muls += 1;
            }
        }
        total_mul_nodes += mul_nodes;
        total_single_add_sub_consumer_muls += single_add_sub_consumer_muls;
        chips.push(DagNodeMixChipCensus {
            static_chip_id: chip.static_chip_id,
            chip_name: chip.chip_name.clone(),
            mul_nodes,
            single_add_sub_consumer_muls,
            single_add_sub_consumer_bps: ratio_bps(single_add_sub_consumer_muls, mul_nodes),
        });
    }
    DagNodeMixCensus {
        mul_nodes: total_mul_nodes,
        single_add_sub_consumer_muls: total_single_add_sub_consumer_muls,
        single_add_sub_consumer_bps: ratio_bps(total_single_add_sub_consumer_muls, total_mul_nodes),
        chips,
    }
}

fn precompute_root_node_for_census(
    chip: &crate::symbolic_ir_dt::RecursionPolyAirChipIr,
    index: usize,
) -> Option<u32> {
    chip.derived_roots.iter().find_map(|root| match root {
        RecursionPolyAirDerivedRoot::PrecomputeLc { index: root_index, root_node_id }
            if *root_index == index =>
        {
            Some(*root_node_id)
        }
        _ => None,
    })
}

fn bump_consumer(consumers: &mut [usize], node_id: u32) {
    if let Some(value) = consumers.get_mut(node_id as usize) {
        *value += 1;
    }
}

/// The keyed node-arity capacity shared by every reduce program (lift bins, L2, L3):
/// static-universe rows are sized for at most this many children per node, so exceeding
/// it is a re-key, not a scheduling choice. A capacity, not a target — planners pick
/// arities `<=` this.
pub const NATIVE_MAX_NODE_ARITY: usize = 11;

/// A reduce-node child in shard order: a bare lift (replay segment u1@0) or an L2 output
/// (replay segment u2@128). An L3 parent accepts both as baked classes; an L2 parent
/// accepts lifts as the baked class and L2 outputs through the threaded vk_root slot
/// (self-recursion).
#[derive(Serialize, Deserialize)]
pub enum NativeReduceChild {
    Lift(SCMachineProof<SC>),
    L2(SCMachineProof<SC>),
}

fn merge_l3_child_records(
    child_records: Vec<BuildingRecord>,
) -> NativeRecursionAssemblyResult<BuildingRecord> {
    merge_child_proof_shard_records(child_records)
}

#[allow(clippy::too_many_arguments)]
fn record_reduce_child(
    request: &NativeRecursionRequest,
    idx: usize,
    child: NativeReduceChild,
    lift_child_machine: &NativeRecordingMachine,
    l2_child_machine: &NativeRecordingMachine,
    parent_program: &RecursionNativeProgram<F>,
    lift_vk: &SCStarkVerifyingKey<SC>,
    l2_vk: &SCStarkVerifyingKey<SC>,
) -> NativeRecursionAssemblyResult<BuildingRecord> {
    match child {
        NativeReduceChild::Lift(lift) => record_reduce_lift_child(
            request,
            idx,
            lift,
            lift_child_machine,
            parent_program,
            lift_vk,
        ),
        NativeReduceChild::L2(mut l2) => {
            if l2.shard_proofs.len() != 1 {
                return Err(validation("reduce L2 child proof must contain one shard"));
            }
            let shard = l2.shard_proofs.pop().expect("length checked");
            let mut seed = request.recording_seed(l2_child_machine.config.mlchallenger());
            crate::machine_dt::observe_replay_vk(l2_vk, &mut seed);
            record_native_proof_shard_in_segment(
                l2_child_machine,
                l2_vk,
                shard,
                idx,
                seed,
                parent_program,
                crate::machine_dt::MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
            )
        }
    }
}

fn record_reduce_lift_child(
    request: &NativeRecursionRequest,
    idx: usize,
    mut lift: SCMachineProof<SC>,
    lift_child_machine: &NativeRecordingMachine,
    parent_program: &RecursionNativeProgram<F>,
    lift_vk: &SCStarkVerifyingKey<SC>,
) -> NativeRecursionAssemblyResult<BuildingRecord> {
    if lift.shard_proofs.len() != 1 {
        return Err(validation("reduce lift child proof must contain one shard"));
    }
    let shard = lift.shard_proofs.pop().expect("length checked");
    let mut seed = request.recording_seed(lift_child_machine.config.mlchallenger());
    crate::machine_dt::observe_replay_vk(lift_vk, &mut seed);
    record_native_proof_shard(lift_child_machine, lift_vk, shard, idx, seed, parent_program)
}

/// The four frozen layers, derived once (deterministic; setup keys asserted content-identical).
pub struct NativeLadderContext<P: NativeProverProvider = CpuNativeProver> {
    pub core_machine: CoreRecordingMachine,
    pub lift_program: RecursionNativeProgram<F>,
    pub lift_prover: NativeRecursionProver<P>,
    pub lift_pk: SCStarkProvingKey<SC>,
    pub lift_device_pk: OnceLock<<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey>,
    pub lift_vk: SCStarkVerifyingKey<SC>,
    pub lift_digest: [F; DIGEST_SIZE],
    pub lift_child_machine: NativeRecordingMachine,
    pub l2_program: RecursionNativeProgram<F>,
    pub l2_prover: NativeRecursionProver<P>,
    pub l2_pk: SCStarkProvingKey<SC>,
    pub l2_device_pk: OnceLock<<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey>,
    pub l2_vk: SCStarkVerifyingKey<SC>,
    pub l2_digest: [F; DIGEST_SIZE],
    pub l2_child_machine: NativeRecordingMachine,
    pub l3_program: RecursionNativeProgram<F>,
    pub l3_prover: NativeRecursionProver<P>,
    pub l3_pk: SCStarkProvingKey<SC>,
    pub l3_device_pk: OnceLock<<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey>,
    pub l3_vk: SCStarkVerifyingKey<SC>,
    pub l3_digest: [F; DIGEST_SIZE],
    pub l3_shrink_machine: NativeRecordingMachine,
    pub l4_program: RecursionNativeProgram<F>,
    pub l4_prover: NativeRootProver<P>,
    pub l4_pk: SCStarkProvingKey<RootSC>,
    pub l4_device_pk: OnceLock<<P::RootProver as SCMachineProver<RootSC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey>,
    pub l4_vk: SCStarkVerifyingKey<RootSC>,
}

// Cache schema for the current KoalaBear/ext5 ladder artifacts and static constraint plans.
const NATIVE_LADDER_CACHE_SCHEMA_VERSION: u32 =
    dt_stark::global_d11::GLOBAL146_NATIVE_LADDER_CACHE_SCHEMA_VERSION;
const NATIVE_LADDER_CACHE_MAX_BYTES: u64 = 1 << 30;
static NATIVE_LADDER_CACHE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Serialize, Deserialize)]
struct NativeLadderCachedArtifacts {
    lift_program: RecursionNativeProgram<F>,
    lift_pk: SCStarkProvingKey<SC>,
    lift_vk: SCStarkVerifyingKey<SC>,
    lift_digest: [F; DIGEST_SIZE],
    l2_program: RecursionNativeProgram<F>,
    l2_pk: SCStarkProvingKey<SC>,
    l2_vk: SCStarkVerifyingKey<SC>,
    l2_digest: [F; DIGEST_SIZE],
    l3_program: RecursionNativeProgram<F>,
    l3_pk: SCStarkProvingKey<SC>,
    l3_vk: SCStarkVerifyingKey<SC>,
    l3_digest: [F; DIGEST_SIZE],
    l4_program: RecursionNativeProgram<F>,
    l4_pk: SCStarkProvingKey<RootSC>,
    l4_vk: SCStarkVerifyingKey<RootSC>,
}

#[derive(Serialize, Deserialize)]
struct NativeLadderCacheFile {
    schema_version: u32,
    global146_identity: [u8; 32],
    registry_version: u32,
    package_version: String,
    setup_hash: String,
    config_hash: String,
    expected_l4_digest: [u32; DIGEST_SIZE],
    artifacts_hash: u64,
    artifacts_bytes: Vec<u8>,
}

enum NativeLadderCacheLoad {
    Missing,
    Valid(NativeLadderContext),
    /// The envelope was torn or its payload failed the envelope's integrity hash. These failures
    /// are safe to replace atomically. Metadata or typed-artifact failures are deliberately not
    /// represented here: they remain fatal semantic incompatibilities.
    RecoverableStorageCorruption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NewCachePublish {
    Published,
    DestinationExists,
}

/// One fully-written cache candidate. The name is unique across concurrent writers (and guarded
/// by `create_new` even across pid namespaces), lives beside the destination, and is removed on
/// every path that does not publish it.
struct NativeLadderCacheTemp {
    path: Option<PathBuf>,
    file: Option<std::fs::File>,
}

impl NativeLadderCacheTemp {
    fn create(cache_path: &Path) -> std::io::Result<Self> {
        let parent = cache_parent(cache_path);
        let file_name = cache_path.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ladder cache path has no file name",
            )
        })?;

        loop {
            let sequence = NATIVE_LADDER_CACHE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut temp_name = OsString::from(".");
            temp_name.push(file_name);
            temp_name.push(format!(".tmp.{}.{sequence}", std::process::id()));
            let path = parent.join(temp_name);
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok(Self { path: Some(path), file: Some(file) }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(err),
            }
        }
    }

    fn write_and_sync(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let file = self.file.as_mut().expect("cache candidate file is open");
        file.write_all(bytes)?;
        file.sync_all()?;
        self.file.take();
        Ok(())
    }

    /// Atomically installs a cache only if no destination exists. A hard link provides portable
    /// create-if-absent semantics without exposing a partially-written destination.
    fn publish_new(&mut self, cache_path: &Path) -> std::io::Result<NewCachePublish> {
        self.file.take();
        let temp_path = self.path.as_ref().expect("cache candidate path is live");
        match std::fs::hard_link(temp_path, cache_path) {
            Ok(()) => {
                if let Err(err) = self.remove_temp() {
                    tracing::warn!(
                        cache_path = %cache_path.display(),
                        error = %err,
                        "native ladder cache was published but its temporary file could not be removed"
                    );
                }
                if let Err(err) = sync_cache_parent(cache_path) {
                    tracing::warn!(
                        cache_path = %cache_path.display(),
                        error = %err,
                        "native ladder cache was published but its parent directory could not be synced"
                    );
                }
                Ok(NewCachePublish::Published)
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok(NewCachePublish::DestinationExists)
            }
            Err(err) => Err(err),
        }
    }

    /// Atomically replaces an existing cache after either a corruption probe or an explicit
    /// force-rebuild request. The candidate was fsynced and lives in the same directory.
    fn replace_existing(&mut self, cache_path: &Path) -> std::io::Result<()> {
        self.file.take();
        let temp_path = self.path.as_ref().expect("cache candidate path is live");
        std::fs::rename(temp_path, cache_path)?;
        self.path.take();
        if let Err(err) = sync_cache_parent(cache_path) {
            tracing::warn!(
                cache_path = %cache_path.display(),
                error = %err,
                "native ladder cache was replaced but its parent directory could not be synced"
            );
        }
        Ok(())
    }

    fn cleanup(&mut self) -> std::io::Result<()> {
        self.file.take();
        self.remove_temp()
    }

    fn remove_temp(&mut self) -> std::io::Result<()> {
        let Some(path) = self.path.take() else {
            return Ok(());
        };
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => {
                self.path = Some(path);
                Err(err)
            }
        }
    }
}

impl Drop for NativeLadderCacheTemp {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn cache_parent(cache_path: &Path) -> &Path {
    cache_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(unix)]
fn sync_cache_parent(cache_path: &Path) -> std::io::Result<()> {
    std::fs::File::open(cache_parent(cache_path))?.sync_all()
}

#[cfg(not(unix))]
fn sync_cache_parent(_cache_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn validation(err: impl std::fmt::Display) -> NativeRecursionAssemblyError {
    NativeRecursionAssemblyError::Validation(err.to_string())
}

fn cache_validation(
    cache_path: &Path,
    operation: &str,
    err: impl std::fmt::Display,
) -> NativeRecursionAssemblyError {
    validation(format!("{operation} at {}: {err}", cache_path.display()))
}

fn force_ladder_cache_rebuild() -> bool {
    std::env::var(LADDER_CACHE_REBUILD_ENV)
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Typed, order-independent verifying-key equality. Hash maps are compared by content; no
/// serialization or randomized iteration order participates in product verification.
pub fn verifying_keys_equal<C>(
    left: &SCStarkVerifyingKey<C>,
    right: &SCStarkVerifyingKey<C>,
) -> bool
where
    C: SCStarkGenericConfig,
    dt_stark::sumcheck::config::MlCom<C>: PartialEq,
{
    // Exhaustive destructuring is deliberate: adding a VK field must make this comparison fail to
    // compile instead of silently weakening the native-proof discriminator.
    let SCStarkVerifyingKey {
        commit: left_commit,
        pc_start: left_pc_start,
        program_boundary: left_program_boundary,
        owner_registry: left_owner_registry,
        global146_identity: left_global146_identity,
        chip_information: left_chip_information,
        chip_ordering: left_chip_ordering,
        constraints_map: left_constraints_map,
    } = left;
    let SCStarkVerifyingKey {
        commit: right_commit,
        pc_start: right_pc_start,
        program_boundary: right_program_boundary,
        owner_registry: right_owner_registry,
        global146_identity: right_global146_identity,
        chip_information: right_chip_information,
        chip_ordering: right_chip_ordering,
        constraints_map: right_constraints_map,
    } = right;

    left_commit == right_commit &&
        left_pc_start == right_pc_start &&
        left_program_boundary == right_program_boundary &&
        left_owner_registry == right_owner_registry &&
        left_global146_identity == right_global146_identity &&
        left_chip_information == right_chip_information &&
        left_chip_ordering == right_chip_ordering &&
        left_constraints_map == right_constraints_map
}

fn vk_digest(vk: &SCStarkVerifyingKey<SC>) -> [F; DIGEST_SIZE] {
    native_vk_statement_digest(&vk.commit, &vk.global146_identity)
}

/// The vk_L4 pin digest for the SHA256-hashed root config. The commit is a
/// 32-byte SHA256 digest, lifted one byte per field element (injective) into
/// the same Poseidon2 sponge as the field-commit form. Host-side pin only —
/// this digest is never observed in-circuit (the root vk is not recursively
/// verified).
pub fn root_vk_digest(vk: &SCStarkVerifyingKey<RootSC>) -> [F; DIGEST_SIZE] {
    dt_stark::global_d11::validate_global146_identity(&vk.global146_identity)
        .expect("root VK has the current Global146 identity");
    let commit_bytes: [u8; 32] = vk.commit.into();
    let mut input = Vec::with_capacity(32 + 1 + 14);
    input.extend(commit_bytes.iter().map(|&byte| F::from_canonical_u8(byte)));
    input.push(vk.pc_start);
    match vk.program_boundary {
        dt_stark::global_d11::ProgramImageBoundaryV1::Infinity => input.push(F::zero()),
        dt_stark::global_d11::ProgramImageBoundaryV1::Affine { x, y } => {
            input.push(F::one());
            input.extend(x.into_iter().map(F::from_canonical_u32));
            input.extend(y.into_iter().map(F::from_canonical_u32));
        }
    }
    input.extend(vk.owner_registry.digest.map(F::from_canonical_u8));
    input.extend(vk.global146_identity.map(F::from_canonical_u8));
    crate::statement_dt::poseidon2_hash_slice(&input)
}

fn digest_u32(digest: [F; DIGEST_SIZE]) -> [u32; DIGEST_SIZE] {
    digest.map(|limb| limb.as_canonical_u32())
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn root_input_opening_batch_count(proof: &SCShardProof<RootSC>) -> usize {
    proof
        .opening_proof
        .query_openings
        .pruned
        .as_ref()
        .map(|pruned| pruned.round_pruned.len())
        .unwrap_or_else(|| proof.opening_proof.query_openings.per_query.first().map_or(0, Vec::len))
}

fn ladder_cache_setup_hash() -> String {
    let setup = format!(
        "root_shrink_degree_floor={NATIVE_ROOT_SHRINK_DEGREE_FLOOR};shrink_degree_floor={NATIVE_SHRINK_DEGREE_FLOOR};"
    );
    format!("{:016x}", stable_hash64(setup.as_bytes()))
}

fn assert_deterministic_setup<C, PROV>(
    label: &str,
    prover: &PROV,
    program: &RecursionNativeProgram<F>,
    vk: &SCStarkVerifyingKey<C>,
) -> NativeRecursionAssemblyResult<()>
where
    C: crate::validate::NativeValidateConfig,
    PROV: SCMachineProver<C, crate::machine_dt::NativeRecursionAir, D_EF>,
    dt_stark::sumcheck::config::MlCom<C>: PartialEq,
{
    let (_, again) = prover.setup(program);
    if !verifying_keys_equal(vk, &again) {
        return Err(validation(format!("{label} setup is not deterministic")));
    }
    Ok(())
}

fn validate_expected_l4_digest(
    vk: &SCStarkVerifyingKey<RootSC>,
    expected: [u32; DIGEST_SIZE],
) -> NativeRecursionAssemblyResult<()> {
    let actual = digest_u32(root_vk_digest(vk));
    if actual != expected {
        return Err(validation(format!(
            "cached vk_L4 digest mismatch: actual={actual:?} expected={expected:?}"
        )));
    }
    Ok(())
}

fn assert_cached_pk_vk_match<C>(
    label: &str,
    pk: &SCStarkProvingKey<C>,
    vk: &SCStarkVerifyingKey<C>,
) -> NativeRecursionAssemblyResult<()>
where
    C: SCStarkGenericConfig,
    dt_stark::sumcheck::config::MlCom<C>: PartialEq,
{
    if pk.commit != vk.commit ||
        pk.pc_start != vk.pc_start ||
        pk.program_boundary != vk.program_boundary ||
        pk.owner_registry != vk.owner_registry ||
        pk.global146_identity != vk.global146_identity ||
        pk.chip_ordering != vk.chip_ordering ||
        pk.constraints_map != vk.constraints_map
    {
        return Err(validation(format!("{label} cached pk/vk public fields mismatch")));
    }
    Ok(())
}

fn validate_cached_layer(
    label: &str,
    pk: &SCStarkProvingKey<SC>,
    vk: &SCStarkVerifyingKey<SC>,
    digest: [F; DIGEST_SIZE],
) -> NativeRecursionAssemblyResult<()> {
    assert_cached_pk_vk_match(label, pk, vk)?;
    let actual = vk_digest(vk);
    if actual != digest {
        return Err(validation(format!("{label} cached digest does not match vk")));
    }
    Ok(())
}

fn validate_child_layout_machine<MachineSC, A>(
    label: &str,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
    machine: &polyair::SCStarkMachine<MachineSC, A, D_EF>,
) -> NativeRecursionAssemblyResult<()>
where
    MachineSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let layout = program
        .constraint_program
        .verified_child_layout(static_chip_id_offset)
        .ok_or_else(|| validation(format!("{label} missing verified child layout")))?;
    let metadata = native_metadata_from_machine(machine);
    layout
        .validate_machine_metadata(&metadata)
        .map_err(|err| validation(format!("{label} machine/program layout: {err:?}")))
}

fn validate_child_layout_vk<ChildSC>(
    label: &str,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
    vk: &SCStarkVerifyingKey<ChildSC>,
) -> NativeRecursionAssemblyResult<()>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
{
    program
        .constraint_program
        .verified_child_layout(static_chip_id_offset)
        .ok_or_else(|| validation(format!("{label} missing verified child layout")))?
        .validate_vk(vk)
        .map_err(|err| validation(format!("{label} vk/program layout: {err:?}")))
}

#[allow(clippy::too_many_arguments)]
fn validate_ladder_child_layout_authorities(
    core_machine: &CoreRecordingMachine,
    lift_program: &RecursionNativeProgram<F>,
    lift_vk: &SCStarkVerifyingKey<SC>,
    lift_child_machine: &NativeRecordingMachine,
    l2_program: &RecursionNativeProgram<F>,
    l2_vk: &SCStarkVerifyingKey<SC>,
    l2_child_machine: &NativeRecordingMachine,
    l3_program: &RecursionNativeProgram<F>,
    l3_vk: &SCStarkVerifyingKey<SC>,
    l3_shrink_machine: &NativeRecordingMachine,
    l4_program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    validate_child_layout_machine("core->lift", lift_program, 0, core_machine)?;

    for (label, program) in [("lift->L2", l2_program), ("lift->L3", l3_program)] {
        validate_child_layout_machine(label, program, 0, lift_child_machine)?;
        validate_child_layout_vk(label, program, 0, lift_vk)?;
    }
    for (label, program) in [("L2->L2", l2_program), ("L2->L3", l3_program)] {
        validate_child_layout_machine(
            label,
            program,
            crate::machine_dt::MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
            l2_child_machine,
        )?;
        validate_child_layout_vk(
            label,
            program,
            crate::machine_dt::MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
            l2_vk,
        )?;
    }
    validate_child_layout_machine("L3->L4", l4_program, 0, l3_shrink_machine)?;
    validate_child_layout_vk("L3->L4", l4_program, 0, l3_vk)?;
    Ok(())
}

impl NativeLadderCachedArtifacts {
    fn from_context(context: &NativeLadderContext) -> Self {
        Self {
            lift_program: context.lift_program.clone(),
            lift_pk: context.lift_pk.clone(),
            lift_vk: context.lift_vk.clone(),
            lift_digest: context.lift_digest,
            l2_program: context.l2_program.clone(),
            l2_pk: context.l2_pk.clone(),
            l2_vk: context.l2_vk.clone(),
            l2_digest: context.l2_digest,
            l3_program: context.l3_program.clone(),
            l3_pk: context.l3_pk.clone(),
            l3_vk: context.l3_vk.clone(),
            l3_digest: context.l3_digest,
            l4_program: context.l4_program.clone(),
            l4_pk: context.l4_pk.clone(),
            l4_vk: context.l4_vk.clone(),
        }
    }

    fn into_context(self) -> NativeRecursionAssemblyResult<NativeLadderContext> {
        let Self {
            lift_program,
            lift_pk,
            lift_vk,
            lift_digest,
            l2_program,
            l2_pk,
            l2_vk,
            l2_digest,
            l3_program,
            l3_pk,
            l3_vk,
            l3_digest,
            l4_program,
            l4_pk,
            l4_vk,
        } = self;

        validate_final_replay_layout(&lift_program)?;
        validate_final_replay_layout(&l2_program)?;
        validate_final_replay_layout(&l3_program)?;
        validate_final_replay_layout(&l4_program)?;
        validate_cached_layer("vk_lift", &lift_pk, &lift_vk, lift_digest)?;
        validate_cached_layer("vk_L2", &l2_pk, &l2_vk, l2_digest)?;
        validate_cached_layer("vk_L3", &l3_pk, &l3_vk, l3_digest)?;
        assert_cached_pk_vk_match("vk_L4", &l4_pk, &l4_vk)?;

        let core_machine = core_recording_machine();
        let lift_prover = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
            &lift_program,
            SC::compressed(),
        )?;
        let lift_child_machine = native_recording_machine(&lift_program)?;
        let l2_prover = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
            &l2_program,
            SC::compressed(),
        )?;
        let l2_child_machine = native_recording_machine(&l2_program)?;
        let l3_prover = native_shrink_prover_with_provider::<CpuNativeProver>(&l3_program)?;
        let l3_shrink_machine =
            native_recording_machine_for_stage(&l3_program, RecordingStage::Shrink)?;
        let l4_prover = native_root_shrink_prover_with_provider::<CpuNativeProver>(&l4_program)?;

        Ok(NativeLadderContext {
            core_machine,
            lift_program,
            lift_prover,
            lift_pk: lift_pk.clone(),
            lift_device_pk: OnceLock::from(lift_pk),
            lift_vk,
            lift_digest,
            lift_child_machine,
            l2_program,
            l2_prover,
            l2_pk: l2_pk.clone(),
            l2_device_pk: OnceLock::from(l2_pk),
            l2_vk,
            l2_digest,
            l2_child_machine,
            l3_program,
            l3_prover,
            l3_pk: l3_pk.clone(),
            l3_device_pk: OnceLock::from(l3_pk),
            l3_vk,
            l3_digest,
            l3_shrink_machine,
            l4_program,
            l4_prover,
            l4_pk: l4_pk.clone(),
            l4_device_pk: OnceLock::from(l4_pk),
            l4_vk,
        })
    }
}

fn ladder_cache_path(
    cache_dir: &Path,
    config_hash: &str,
    expected_l4_digest: [u32; DIGEST_SIZE],
) -> PathBuf {
    let vk_key =
        expected_l4_digest.iter().map(|limb| format!("{limb:08x}")).collect::<Vec<_>>().join("");
    let setup_hash = ladder_cache_setup_hash();
    cache_dir.join(format!(
        "native-ladder-v{}-registry{}-pkg{}-setup{}-cfg{}-vk{}.bin",
        NATIVE_LADDER_CACHE_SCHEMA_VERSION,
        NATIVE_AIR_REGISTRY_VERSION,
        env!("CARGO_PKG_VERSION"),
        setup_hash,
        config_hash,
        vk_key
    ))
}

/// Build a native ladder context with a custom prover provider (e.g. GPU). Skips disk cache.
pub fn build_ladder_with_provider<P: NativeProverProvider>(
) -> NativeRecursionAssemblyResult<NativeLadderContext<P>> {
    {
        use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::{
            compressed_fri_config, root_shrink_fri_config, shrink_fri_config,
        };
        if crate::debug_prints_enabled() {
            for (stage, cfg) in [
                ("compress", compressed_fri_config()),
                ("shrink", shrink_fri_config()),
                ("root_shrink", root_shrink_fri_config()),
            ] {
                println!(
                    "native_live_whir_config stage={} num_queries={} log_blowup={} log_final_poly_len={}",
                    stage, cfg.num_queries, cfg.log_blowup, cfg.log_final_poly_len,
                );
            }
        }
    }
    let core_machine = core_recording_machine();
    let lift_program = build_core_native_recursion_program(&core_machine)?;
    validate_final_replay_layout(&lift_program)?;
    let (lift_pk, lift_vk) = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
        &lift_program,
        SC::compressed(),
    )?
    .setup(&lift_program);
    let lift_prover =
        native_recursion_prover_with_config_and_provider::<P>(&lift_program, SC::compressed())?;
    let _cpu_a1 = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
        &lift_program,
        SC::compressed(),
    )?;
    assert_deterministic_setup("vk_lift", &_cpu_a1, &lift_program, &lift_vk)?;
    let lift_digest = vk_digest(&lift_vk);
    let lift_child_machine = native_recording_machine(&lift_program)?;

    let l2_config = vec![StatementConfigRow {
        class_id: STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        digest: lift_digest,
    }];
    let bootstrap_program = build_native_recursion_program(
        &lift_child_machine,
        RecursionStatementRole::ReduceL2,
        crate::symbolic_expr_fixed_dt::RecursionChildRole::Compress,
        NATIVE_RECURSION_NUM_PV_ELTS,
        false,
        l2_config.clone(),
    )?;
    validate_l2_bootstrap_layout(&bootstrap_program)?;
    let bootstrap_machine = native_recording_machine(&bootstrap_program)?;
    let l2_program = build_dual_segment_reduce_program(
        &lift_child_machine,
        &bootstrap_machine,
        RecursionStatementRole::ReduceL2,
        l2_config,
    )?;
    validate_final_replay_layout(&l2_program)?;
    let (l2_pk, l2_vk) = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
        &l2_program,
        SC::compressed(),
    )?
    .setup(&l2_program);
    let l2_prover =
        native_recursion_prover_with_config_and_provider::<P>(&l2_program, SC::compressed())?;
    let _cpu_a2 = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
        &l2_program,
        SC::compressed(),
    )?;
    assert_deterministic_setup("vk_L2", &_cpu_a2, &l2_program, &l2_vk)?;
    let l2_digest = vk_digest(&l2_vk);
    let l2_child_machine = native_recording_machine(&l2_program)?;
    validate_l2_bootstrap_fixed_point(&l2_child_machine, &l2_program)?;

    let l3_config = vec![
        StatementConfigRow { class_id: STATEMENT_CONFIG_CLASS_BAKED_LIFT, digest: lift_digest },
        StatementConfigRow { class_id: STATEMENT_CONFIG_CLASS_BAKED_L2, digest: l2_digest },
    ];
    let l3_program = build_dual_segment_reduce_program(
        &lift_child_machine,
        &l2_child_machine,
        RecursionStatementRole::ReduceL3,
        l3_config,
    )?;
    validate_final_replay_layout(&l3_program)?;
    let (l3_pk, l3_vk) =
        native_shrink_prover_with_provider::<CpuNativeProver>(&l3_program)?.setup(&l3_program);
    let l3_prover = native_shrink_prover_with_provider::<P>(&l3_program)?;
    let _cpu_a3 = native_shrink_prover_with_provider::<CpuNativeProver>(&l3_program)?;
    assert_deterministic_setup("vk_L3", &_cpu_a3, &l3_program, &l3_vk)?;
    let l3_digest = vk_digest(&l3_vk);
    let l3_shrink_machine =
        native_recording_machine_for_stage(&l3_program, RecordingStage::Shrink)?;

    let l4_config =
        vec![StatementConfigRow { class_id: STATEMENT_CONFIG_CLASS_BAKED_L3, digest: l3_digest }];
    let l4_program = build_root_shrink_program(&l3_shrink_machine, l4_config)?;
    validate_final_replay_layout(&l4_program)?;
    let (l4_pk, l4_vk) =
        native_root_shrink_prover_with_provider::<CpuNativeProver>(&l4_program)?.setup(&l4_program);
    let l4_prover = native_root_shrink_prover_with_provider::<P>(&l4_program)?;
    let _cpu_a4 = native_root_shrink_prover_with_provider::<CpuNativeProver>(&l4_program)?;
    assert_deterministic_setup("vk_L4", &_cpu_a4, &l4_program, &l4_vk)?;

    Ok(NativeLadderContext {
        core_machine,
        lift_program,
        lift_prover,
        lift_pk,
        lift_device_pk: OnceLock::new(),
        lift_vk,
        lift_digest,
        lift_child_machine,
        l2_program,
        l2_prover,
        l2_pk,
        l2_device_pk: OnceLock::new(),
        l2_vk,
        l2_digest,
        l2_child_machine,
        l3_program,
        l3_prover,
        l3_pk,
        l3_device_pk: OnceLock::new(),
        l3_vk,
        l3_digest,
        l3_shrink_machine,
        l4_program,
        l4_prover,
        l4_pk,
        l4_device_pk: OnceLock::new(),
        l4_vk,
    })
}

fn load_ladder_cache(
    cache_path: &Path,
    config_hash: &str,
    expected_l4_digest: [u32; DIGEST_SIZE],
) -> NativeRecursionAssemblyResult<NativeLadderCacheLoad> {
    let cold_started = Instant::now();
    let cache_file = match std::fs::File::open(cache_path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(NativeLadderCacheLoad::Missing);
        }
        Err(err) => return Err(cache_validation(cache_path, "open ladder cache", err)),
    };
    let cache_len = cache_file
        .metadata()
        .map_err(|err| cache_validation(cache_path, "stat opened ladder cache", err))?
        .len();
    if cache_len > NATIVE_LADDER_CACHE_MAX_BYTES {
        return Err(cache_validation(
            cache_path,
            "reject oversized ladder cache before allocation",
            format!("{cache_len} bytes exceeds {NATIVE_LADDER_CACHE_MAX_BYTES}"),
        ));
    }
    let initial_capacity = usize::try_from(cache_len).map_err(|_| {
        cache_validation(cache_path, "convert ladder cache length", "length does not fit usize")
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(initial_capacity).map_err(|_| {
        cache_validation(cache_path, "reserve ladder cache buffer", "allocation rejected")
    })?;
    let read_started = Instant::now();
    cache_file
        .take(NATIVE_LADDER_CACHE_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| cache_validation(cache_path, "read bounded ladder cache", err))?;
    let read_us = u64::try_from(read_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    if bytes.len() as u64 > NATIVE_LADDER_CACHE_MAX_BYTES {
        return Err(cache_validation(
            cache_path,
            "reject growing ladder cache before decode",
            format!("cache exceeded {NATIVE_LADDER_CACHE_MAX_BYTES} bytes while reading"),
        ));
    }
    let envelope_decode_started = Instant::now();
    let file: NativeLadderCacheFile = match bounded_cache_bincode().deserialize(&bytes) {
        Ok(file) => file,
        Err(_) => return Ok(NativeLadderCacheLoad::RecoverableStorageCorruption),
    };
    let envelope_decode_us =
        u64::try_from(envelope_decode_started.elapsed().as_micros()).unwrap_or(u64::MAX);

    let mut metadata_mismatches = Vec::new();
    if file.schema_version != NATIVE_LADDER_CACHE_SCHEMA_VERSION {
        metadata_mismatches.push("schema_version");
    }
    if file.global146_identity != dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY {
        metadata_mismatches.push("global146_identity");
    }
    if file.registry_version != NATIVE_AIR_REGISTRY_VERSION {
        metadata_mismatches.push("registry_version");
    }
    if file.package_version != env!("CARGO_PKG_VERSION") {
        metadata_mismatches.push("package_version");
    }
    if file.setup_hash != ladder_cache_setup_hash() {
        metadata_mismatches.push("setup_hash");
    }
    if file.config_hash != config_hash {
        metadata_mismatches.push("config_hash");
    }
    if file.expected_l4_digest != expected_l4_digest {
        metadata_mismatches.push("expected_l4_digest");
    }
    if !metadata_mismatches.is_empty() {
        return Err(cache_validation(
            cache_path,
            "ladder cache metadata mismatch",
            metadata_mismatches.join(", "),
        ));
    }
    if stable_hash64(&file.artifacts_bytes) != file.artifacts_hash {
        return Ok(NativeLadderCacheLoad::RecoverableStorageCorruption);
    }
    let artifact_decode_started = Instant::now();
    let artifacts: NativeLadderCachedArtifacts =
        bounded_cache_bincode().deserialize(&file.artifacts_bytes).map_err(|err| {
            cache_validation(
                cache_path,
                "decode integrity-checked cached native ladder artifacts",
                err,
            )
        })?;
    let artifact_decode_us =
        u64::try_from(artifact_decode_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let (static_plan_compile_us, static_plan_retained_bytes) = [
        &artifacts.lift_program,
        &artifacts.l2_program,
        &artifacts.l3_program,
        &artifacts.l4_program,
    ]
    .into_iter()
    .map(|program| program.constraint_program.constraint_static_plan_cold_metrics())
    .fold((0u64, 0u64), |(compile, retained), (next_compile, next_retained)| {
        (compile.saturating_add(next_compile), retained.saturating_add(next_retained))
    });
    let validation_started = Instant::now();
    let context = artifacts
        .into_context()
        .map_err(|err| cache_validation(cache_path, "validate cached native ladder", err))?;
    validate_expected_l4_digest(&context.l4_vk, expected_l4_digest)
        .map_err(|err| cache_validation(cache_path, "validate cached vk_L4 digest", err))?;
    let validation_us = u64::try_from(validation_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    tracing::info!(
        target: "native_recursion::cold_start",
        cache_bytes = bytes.len(),
        read_us,
        envelope_decode_us,
        artifact_decode_us,
        validation_us,
        static_plan_compile_us,
        static_plan_retained_bytes,
        cold_total_us = u64::try_from(cold_started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "native ladder cold load complete; excluded from steady-state proof metrics"
    );
    Ok(NativeLadderCacheLoad::Valid(context))
}

fn bounded_cache_bincode() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(NATIVE_LADDER_CACHE_MAX_BYTES)
}

fn publish_ladder_cache_bytes(
    cache_path: &Path,
    config_hash: &str,
    expected_l4_digest: [u32; DIGEST_SIZE],
    bytes: &[u8],
    force_replace: bool,
) -> NativeRecursionAssemblyResult<()> {
    std::fs::create_dir_all(cache_parent(cache_path))
        .map_err(|err| cache_validation(cache_path, "create ladder cache directory", err))?;
    let mut candidate = NativeLadderCacheTemp::create(cache_path)
        .map_err(|err| cache_validation(cache_path, "create ladder cache candidate", err))?;
    if let Err(err) = candidate.write_and_sync(bytes) {
        let cleanup = candidate.cleanup();
        return Err(cache_validation(
            cache_path,
            "write and sync ladder cache candidate",
            match cleanup {
                Ok(()) => err.to_string(),
                Err(cleanup_err) => format!("{err}; cleanup failed: {cleanup_err}"),
            },
        ));
    }

    if force_replace {
        return candidate
            .replace_existing(cache_path)
            .map_err(|err| cache_validation(cache_path, "replace ladder cache", err));
    }

    loop {
        match load_ladder_cache(cache_path, config_hash, expected_l4_digest) {
            Ok(NativeLadderCacheLoad::Valid(_)) => {
                candidate.cleanup().map_err(|err| {
                    cache_validation(cache_path, "clean unused ladder cache candidate", err)
                })?;
                return Ok(());
            }
            Ok(NativeLadderCacheLoad::Missing) => match candidate.publish_new(cache_path) {
                Ok(NewCachePublish::Published) => return Ok(()),
                Ok(NewCachePublish::DestinationExists) => continue,
                Err(err) => {
                    let cleanup = candidate.cleanup();
                    return Err(cache_validation(
                        cache_path,
                        "publish new ladder cache",
                        match cleanup {
                            Ok(()) => err.to_string(),
                            Err(cleanup_err) => format!("{err}; cleanup failed: {cleanup_err}"),
                        },
                    ));
                }
            },
            Ok(NativeLadderCacheLoad::RecoverableStorageCorruption) => {
                if let Err(err) = candidate.replace_existing(cache_path) {
                    let cleanup = candidate.cleanup();
                    return Err(cache_validation(
                        cache_path,
                        "replace corrupt ladder cache",
                        match cleanup {
                            Ok(()) => err.to_string(),
                            Err(cleanup_err) => format!("{err}; cleanup failed: {cleanup_err}"),
                        },
                    ));
                }
                return Ok(());
            }
            Err(err) => {
                let cleanup = candidate.cleanup();
                return Err(match cleanup {
                    Ok(()) => err,
                    Err(cleanup_err) => cache_validation(
                        cache_path,
                        "clean rejected ladder cache candidate",
                        format!("{err}; cleanup failed: {cleanup_err}"),
                    ),
                });
            }
        }
    }
}

fn write_ladder_cache(
    cache_path: &Path,
    config_hash: &str,
    expected_l4_digest: [u32; DIGEST_SIZE],
    context: NativeLadderContext,
    force_replace: bool,
) -> (NativeLadderContext, Option<NativeRecursionAssemblyError>) {
    let publish_result = (|| {
        let artifacts = NativeLadderCachedArtifacts::from_context(&context);
        let artifacts_bytes = bincode::serialize(&artifacts)
            .map_err(|err| cache_validation(cache_path, "encode native ladder artifacts", err))?;
        let artifacts_hash = stable_hash64(&artifacts_bytes);
        let file = NativeLadderCacheFile {
            schema_version: NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
            registry_version: NATIVE_AIR_REGISTRY_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            setup_hash: ladder_cache_setup_hash(),
            config_hash: config_hash.to_string(),
            expected_l4_digest,
            artifacts_hash,
            artifacts_bytes,
        };
        let bytes = bincode::serialize(&file)
            .map_err(|err| cache_validation(cache_path, "encode native ladder cache", err))?;
        publish_ladder_cache_bytes(
            cache_path,
            config_hash,
            expected_l4_digest,
            &bytes,
            force_replace,
        )
    })();
    (context, publish_result.err())
}

impl NativeLadderContext {
    pub fn build() -> NativeRecursionAssemblyResult<Self> {
        Self::build_uncached()
    }

    pub fn build_with_disk_cache(
        cache_dir: &Path,
        config_hash: &str,
        expected_l4_digest: [u32; DIGEST_SIZE],
    ) -> NativeRecursionAssemblyResult<Self> {
        let cache_path = ladder_cache_path(cache_dir, config_hash, expected_l4_digest);
        let force_rebuild = force_ladder_cache_rebuild();
        if !force_rebuild {
            match load_ladder_cache(&cache_path, config_hash, expected_l4_digest) {
                Ok(NativeLadderCacheLoad::Valid(context)) => return Ok(context),
                Ok(
                    NativeLadderCacheLoad::Missing |
                    NativeLadderCacheLoad::RecoverableStorageCorruption,
                ) => {}
                Err(err) => return Err(err),
            }
        }

        let context = Self::build_uncached()?;
        validate_expected_l4_digest(&context.l4_vk, expected_l4_digest)?;
        let (context, publish_error) = write_ladder_cache(
            &cache_path,
            config_hash,
            expected_l4_digest,
            context,
            force_rebuild,
        );
        if let Some(err) = publish_error {
            tracing::warn!(
                cache_path = %cache_path.display(),
                error = %err,
                "native ladder cache publish failed; continuing with the freshly built context"
            );
        }
        Ok(context)
    }

    fn build_uncached() -> NativeRecursionAssemblyResult<Self> {
        // Instrumentation: print the live per-stage WHIR config so size accounting
        // keys on the real parameters, not code defaults.
        {
            use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::{
                compressed_fri_config, root_shrink_fri_config, shrink_fri_config,
            };
            if crate::debug_prints_enabled() {
                for (stage, cfg) in [
                    ("compress", compressed_fri_config()),
                    ("shrink", shrink_fri_config()),
                    ("root_shrink", root_shrink_fri_config()),
                ] {
                    println!(
                        "native_live_whir_config stage={} num_queries={} log_blowup={} \
log_final_poly_len={}",
                        stage, cfg.num_queries, cfg.log_blowup, cfg.log_final_poly_len,
                    );
                }
            }
        }
        let core_machine = core_recording_machine();
        let lift_program = build_core_native_recursion_program(&core_machine)?;
        validate_final_replay_layout(&lift_program)?;
        let lift_prover = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
            &lift_program,
            SC::compressed(),
        )?;
        let (lift_pk, lift_vk) = lift_prover.setup(&lift_program);
        assert_deterministic_setup("vk_lift", &lift_prover, &lift_program, &lift_vk)?;
        let lift_digest = vk_digest(&lift_vk);
        let lift_child_machine = native_recording_machine(&lift_program)?;

        let l2_config = vec![StatementConfigRow {
            class_id: STATEMENT_CONFIG_CLASS_BAKED_LIFT,
            digest: lift_digest,
        }];
        let bootstrap_program = build_native_recursion_program(
            &lift_child_machine,
            RecursionStatementRole::ReduceL2,
            crate::symbolic_expr_fixed_dt::RecursionChildRole::Compress,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false,
            l2_config.clone(),
        )?;
        validate_l2_bootstrap_layout(&bootstrap_program)?;
        let bootstrap_machine = native_recording_machine(&bootstrap_program)?;
        let l2_program = build_dual_segment_reduce_program(
            &lift_child_machine,
            &bootstrap_machine,
            RecursionStatementRole::ReduceL2,
            l2_config,
        )?;
        validate_final_replay_layout(&l2_program)?;
        let l2_prover = native_recursion_prover_with_config_and_provider::<CpuNativeProver>(
            &l2_program,
            SC::compressed(),
        )?;
        let (l2_pk, l2_vk) = l2_prover.setup(&l2_program);
        assert_deterministic_setup("vk_L2", &l2_prover, &l2_program, &l2_vk)?;
        let l2_digest = vk_digest(&l2_vk);
        let l2_child_machine = native_recording_machine(&l2_program)?;
        validate_l2_bootstrap_fixed_point(&l2_child_machine, &l2_program)?;

        let l3_config = vec![
            StatementConfigRow { class_id: STATEMENT_CONFIG_CLASS_BAKED_LIFT, digest: lift_digest },
            StatementConfigRow { class_id: STATEMENT_CONFIG_CLASS_BAKED_L2, digest: l2_digest },
        ];
        let l3_program = build_dual_segment_reduce_program(
            &lift_child_machine,
            &l2_child_machine,
            RecursionStatementRole::ReduceL3,
            l3_config,
        )?;
        validate_final_replay_layout(&l3_program)?;
        let l3_prover = native_shrink_prover_with_provider::<CpuNativeProver>(&l3_program)?;
        let (l3_pk, l3_vk) = l3_prover.setup(&l3_program);
        assert_deterministic_setup("vk_L3", &l3_prover, &l3_program, &l3_vk)?;
        let l3_digest = vk_digest(&l3_vk);
        let l3_shrink_machine =
            native_recording_machine_for_stage(&l3_program, RecordingStage::Shrink)?;

        let l4_config = vec![StatementConfigRow {
            class_id: STATEMENT_CONFIG_CLASS_BAKED_L3,
            digest: l3_digest,
        }];
        let l4_program = build_root_shrink_program(&l3_shrink_machine, l4_config)?;
        validate_final_replay_layout(&l4_program)?;
        let l4_prover = native_root_shrink_prover_with_provider::<CpuNativeProver>(&l4_program)?;
        let (l4_pk, l4_vk) = l4_prover.setup(&l4_program);
        assert_deterministic_setup("vk_L4", &l4_prover, &l4_program, &l4_vk)?;

        validate_ladder_child_layout_authorities(
            &core_machine,
            &lift_program,
            &lift_vk,
            &lift_child_machine,
            &l2_program,
            &l2_vk,
            &l2_child_machine,
            &l3_program,
            &l3_vk,
            &l3_shrink_machine,
            &l4_program,
        )?;

        Ok(Self {
            core_machine,
            lift_program,
            lift_prover,
            lift_pk: lift_pk.clone(),
            lift_device_pk: OnceLock::from(lift_pk),
            lift_vk,
            lift_digest,
            lift_child_machine,
            l2_program,
            l2_prover,
            l2_pk: l2_pk.clone(),
            l2_device_pk: OnceLock::from(l2_pk),
            l2_vk,
            l2_digest,
            l2_child_machine,
            l3_program,
            l3_prover,
            l3_pk: l3_pk.clone(),
            l3_device_pk: OnceLock::from(l3_pk),
            l3_vk,
            l3_digest,
            l3_shrink_machine,
            l4_program,
            l4_prover,
            l4_pk: l4_pk.clone(),
            l4_device_pk: OnceLock::from(l4_pk),
            l4_vk,
        })
    }
}

impl<P: NativeProverProvider> NativeLadderContext<P> {
    /// The sole externally visible artifact of the frozen ladder setup.
    pub fn root_vk(&self) -> &SCStarkVerifyingKey<RootSC> {
        &self.l4_vk
    }

    /// Lazily setup and return the L2 device proving key.
    pub fn l2_device_pk(
        &self,
    ) -> &<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey{
        self.l2_device_pk.get_or_init(|| {
            let (pk, _) = self.l2_prover.setup(&self.l2_program);
            pk
        })
    }

    /// Lazily setup and return the L3 device proving key.
    pub fn l3_device_pk(
        &self,
    ) -> &<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey{
        self.l3_device_pk.get_or_init(|| {
            let (pk, _) = self.l3_prover.setup(&self.l3_program);
            pk
        })
    }

    /// Lazily setup and return the L4 device proving key.
    pub fn l4_device_pk(
        &self,
    ) -> &<P::RootProver as SCMachineProver<RootSC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey{
        self.l4_device_pk.get_or_init(|| {
            let (pk, _) = self.l4_prover.setup(&self.l4_program);
            pk
        })
    }

    /// Lazily setup and return the lift device proving key.
    pub fn lift_device_pk(
        &self,
    ) -> &<P::SCProver as SCMachineProver<SC, crate::machine_dt::NativeRecursionAir, D_EF>>::DeviceProvingKey{
        self.lift_device_pk.get_or_init(|| {
            let (pk, _) = self.lift_prover.setup(&self.lift_program);
            pk
        })
    }

    /// Removes the first stacked input-opening batch for explicit proof-size experiments.
    ///
    /// The resulting legacy proof is deliberately not accepted by [`Self::external_check`].
    /// Product callers must publish the full proof returned by [`Self::prove_l4`].
    pub fn elide_root_prep_input_opening(
        &self,
        proof: &mut SCShardProof<RootSC>,
    ) -> NativeRecursionAssemblyResult<bool> {
        let expected_batches = proof.dimensions.len();
        let opening_proof = &mut proof.opening_proof;
        if opening_proof.stack_log_height.is_none() || opening_proof.round_iopp.is_none() {
            return Ok(false);
        }
        let pruned = opening_proof
            .query_openings
            .pruned
            .as_mut()
            .ok_or_else(|| validation("root proof is missing pruned input openings"))?;
        let current_batches = pruned.round_pruned.len();
        if current_batches + 1 == expected_batches {
            return Ok(false);
        }
        if current_batches != expected_batches ||
            current_batches < 2 ||
            pruned.round_opened_values.len() != current_batches ||
            pruned.query_to_unique_slot.len() != current_batches
        {
            return Err(validation(format!(
                "unexpected root input-opening batches: pruned={} opened={} q2u={} expected={}",
                pruned.round_pruned.len(),
                pruned.round_opened_values.len(),
                pruned.query_to_unique_slot.len(),
                expected_batches
            )));
        }
        pruned.round_pruned.remove(0);
        pruned.round_opened_values.remove(0);
        pruned.query_to_unique_slot.remove(0);
        Ok(true)
    }

    /// Derives all four frozen layers (two-pass bootstrap for the canonical L2) and
    /// asserts per-layer setup determinism once.
    pub fn external_check<CoreProofSC>(
        &self,
        reduce: &DTReduceProof<RootSC>,
        core_vk: &SCStarkVerifyingKey<CoreProofSC>,
    ) -> NativeRecursionAssemblyResult<()>
    where
        CoreProofSC: SCStarkGenericConfig<Val = F>,
        CoreProofSC::Mlpcs: MlPCS<Commitment = MlCom<RecordingSC>>,
    {
        let phase = std::time::Instant::now();
        if !verifying_keys_equal(&self.l4_vk, &reduce.vk) {
            return Err(validation("presented root vk differs from the frozen vk_L4"));
        }
        pcs::whir::profile::add_ms("verify.vk_typed_cmp_us", phase.elapsed().as_micros());
        let public = &reduce.proof.public_values;
        let (global_start, global_end) = checked_native_root_public_interval(public)?;
        let phase = std::time::Instant::now();
        require_safe_root_polyair_shape(self.l4_prover.machine(), &self.l4_vk, reduce)?;
        pcs::whir::profile::add_ms(
            "verify.safe_root_shape_preflight_us",
            phase.elapsed().as_micros(),
        );
        let phase = std::time::Instant::now();
        require_full_root_input_opening(reduce)?;
        pcs::whir::profile::add_ms(
            "verify.full_input_opening_preflight_us",
            phase.elapsed().as_micros(),
        );
        let phase = std::time::Instant::now();
        verify_root_recursion_shard(self.l4_prover.machine(), &self.l4_vk, &reduce.proof)?;
        pcs::whir::profile::add_ms("verify.machine_verify_us", phase.elapsed().as_micros());

        // Instrumentation: log the per-component byte breakdown of the root proof
        // (bincode size of each field as-is).
        if crate::debug_prints_enabled() {
            let sz = |r: Result<Vec<u8>, _>| r.map(|b| b.len()).unwrap_or(0);
            let proof = &reduce.proof;
            println!(
                "native_root_proof_slice commitment={} opened_values={} opening_proof={} \
		sumcheck={} dims={} ordering={} public_values={} shard_total={} \
		reduce_wire_total={} input_opening_batches={} \
		full_input_opening=true",
                sz(bincode::serialize(&proof.commitment)),
                sz(bincode::serialize(&proof.opened_values)),
                sz(bincode::serialize(&proof.opening_proof)),
                sz(bincode::serialize(&proof.sumcheck_proof)),
                sz(bincode::serialize(&proof.dimensions)),
                sz(bincode::serialize(&proof.chip_ordering)),
                sz(bincode::serialize(&proof.public_values)),
                sz(bincode::serialize(proof)),
                sz(bincode::serialize(reduce)),
                root_input_opening_batch_count(proof),
            );
            let mut d_histogram = BTreeMap::<usize, usize>::new();
            for unipoly in &proof.sumcheck_proof.unipolys {
                *d_histogram.entry(unipoly.evals.len()).or_default() += 1;
            }
            println!("native_root_d_histogram eval_len_counts={d_histogram:?}");
        }

        let phase = std::time::Instant::now();
        let expected_dt_vk = core_vk_statement_digest(
            &core_vk.commit,
            core_vk.pc_start,
            &core_vk.program_boundary,
            &core_vk.global146_identity,
        );
        for idx in 0..DIGEST_SIZE {
            if public[NATIVE_PV_DT_VK_DIGEST_START + idx] != expected_dt_vk[idx] {
                return Err(validation("root dt_vk digest does not match the core vk"));
            }
        }
        validate_native_root_global_interval(&core_vk.program_boundary, global_start, global_end)
            .map_err(|error| validation(error.to_string()))?;
        pcs::whir::profile::add_ms("verify.pv_checks_us", phase.elapsed().as_micros());
        Ok(())
    }

    /// Record one core child as soon as its shard proof is available. The caller assigns the
    /// proof index within its future lift bin; final publication and statement aggregation remain
    /// deferred until the bin closes.
    pub fn record_core_child_record<CoreProofSC>(
        &self,
        request: &NativeRecursionRequest,
        core_vk: &SCStarkVerifyingKey<CoreProofSC>,
        shard: SCShardProof<CoreProofSC>,
        proof_idx: usize,
    ) -> NativeRecursionAssemblyResult<BuildingRecord>
    where
        CoreProofSC: ReplayCompatibleProofConfig,
        CoreProofSC::Mlpcs:
            MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
    {
        validate_final_replay_layout(&self.lift_program)?;
        let mut seed = request.recording_seed(self.core_machine.config.mlchallenger());
        crate::machine_dt::observe_replay_vk(core_vk, &mut seed);
        record_core_proof_shard(
            &self.core_machine,
            core_vk,
            shard,
            proof_idx,
            seed,
            &self.lift_program,
        )
    }

    /// Close a lift bin from child records produced by [`Self::record_core_child_record`].
    pub fn prove_lift_from_child_records(
        &self,
        request: &NativeRecursionRequest,
        child_records: Vec<BuildingRecord>,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
        validate_final_replay_layout(&self.lift_program)?;
        if child_records.is_empty() {
            return Err(validation("lift requires at least one pre-recorded core child"));
        }
        request.ensure_owns_records("lift", &child_records)?;
        let finalize_start = Instant::now();
        let arity = child_records.len();
        let record = merge_child_proof_shard_records(child_records)?;
        let record = finalize_building_record(record, &self.lift_program, "lift_prerecorded")?;
        self.prove_node_with_record_ms(
            &format!("lift(k={arity})"),
            &self.lift_prover,
            &self.lift_pk,
            self.lift_device_pk(),
            &self.lift_vk,
            &self.lift_program,
            record,
            finalize_start.elapsed().as_millis(),
            stats,
        )
    }

    /// One L2 node over children in shard order: lift children replay in segment u1@0
    /// as the baked class; L2 children replay in segment u2@128 and must match the
    /// node's threaded vk_root input (self-recursion). The threaded export slot
    /// carries vk_L2 either way.
    pub fn prove_l2(
        &self,
        request: &NativeRecursionRequest,
        children: Vec<NativeReduceChild>,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
        validate_final_replay_layout(&self.l2_program)?;
        let arity = children.len();
        if arity < 2 {
            return Err(validation(format!(
                "L2 requires at least two children, got {arity} \
                 (a lone child routes to its parent as a carry, not an L2 wrapper)"
            )));
        }
        if arity > NATIVE_MAX_NODE_ARITY {
            return Err(validation(format!(
                "L2 arity {arity} exceeds the keyed node capacity {NATIVE_MAX_NODE_ARITY}"
            )));
        }
        let record_start = Instant::now();
        let child_records = self.record_reduce_child_records_parallel(
            request,
            children,
            &self.l2_program,
            &self.lift_vk,
            &self.l2_vk,
        )?;
        let mut record = merge_child_proof_shard_records(child_records)?;
        record.set_statement_vk_root(self.l2_digest);
        let record = finalize_building_record(record, &self.l2_program, "L2")?;
        let kind = format!("L2(k={arity})");
        self.prove_node_with_record_ms(
            &kind,
            &self.l2_prover,
            &self.l2_pk,
            self.l2_device_pk(),
            &self.l2_vk,
            &self.l2_program,
            record,
            record_start.elapsed().as_millis(),
            stats,
        )
    }

    /// Gate-harness seam (V3 §13.3), not a public API: prove an L2 node whose
    /// threaded vk_root export is caller-chosen instead of the canonical
    /// `l2_digest`. A parent recording such a child must reject it — the wrong
    /// export can only be caught there (the L2 node itself cannot check the
    /// self-referential digest). Everything else the mutation harness needs
    /// travels through the owning APIs.
    #[doc(hidden)]
    pub fn harness_prove_l2_with_vk_root(
        &self,
        request: &NativeRecursionRequest,
        children: Vec<NativeReduceChild>,
        vk_root: [F; DIGEST_SIZE],
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
        validate_final_replay_layout(&self.l2_program)?;
        let arity = children.len();
        if !(2..=NATIVE_MAX_NODE_ARITY).contains(&arity) {
            return Err(validation(format!("harness L2 arity {arity} outside 2..=cap")));
        }
        let record_start = Instant::now();
        let child_records = self.record_reduce_child_records_parallel(
            request,
            children,
            &self.l2_program,
            &self.lift_vk,
            &self.l2_vk,
        )?;
        let mut record = merge_child_proof_shard_records(child_records)?;
        record.set_statement_vk_root(vk_root);
        let record = finalize_building_record(record, &self.l2_program, "L2_harness")?;
        self.prove_node_with_record_ms(
            &format!("L2_harness(k={arity})"),
            &self.l2_prover,
            &self.l2_pk,
            self.l2_device_pk(),
            &self.l2_vk,
            &self.l2_program,
            record,
            record_start.elapsed().as_millis(),
            stats,
        )
    }

    /// One L3 node at the shrink config over children in shard order.
    pub fn prove_l3(
        &self,
        request: &NativeRecursionRequest,
        children: Vec<NativeReduceChild>,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
        validate_final_replay_layout(&self.l3_program)?;
        let arity = children.len();
        if children.is_empty() {
            return Err(validation("L3 requires at least one child"));
        }
        if children.len() > NATIVE_MAX_NODE_ARITY {
            return Err(validation(format!(
                "L3 arity {} exceeds the keyed node capacity {NATIVE_MAX_NODE_ARITY}",
                children.len()
            )));
        }
        let record_start = Instant::now();
        let child_records = self.record_reduce_child_records_parallel(
            request,
            children,
            &self.l3_program,
            &self.lift_vk,
            &self.l2_vk,
        )?;
        let record = merge_l3_child_records(child_records)?;
        let record = finalize_building_record(record, &self.l3_program, "L3")?;
        let kind = format!("L3(k={arity})");
        self.prove_node_with_record_ms(
            &kind,
            &self.l3_prover,
            &self.l3_pk,
            self.l3_device_pk(),
            &self.l3_vk,
            &self.l3_program,
            record,
            record_start.elapsed().as_millis(),
            stats,
        )
    }

    /// Move a completed lift's only shard into its L3 tracegen source.
    pub fn record_l3_lift_child_record(
        &self,
        request: &NativeRecursionRequest,
        idx: usize,
        lift: SCMachineProof<SC>,
    ) -> NativeRecursionAssemblyResult<BuildingRecord> {
        validate_final_replay_layout(&self.l3_program)?;
        record_reduce_lift_child(
            request,
            idx,
            lift,
            &self.lift_child_machine,
            &self.l3_program,
            &self.lift_vk,
        )
    }

    /// Move one plan-selected Lift/L2 child into its exact L3 local slot.
    /// This is the mixed-frontier seam used by the TreePlan executor; carried
    /// proofs are not wrapped or cloned.
    pub fn record_l3_child_record(
        &self,
        request: &NativeRecursionRequest,
        idx: usize,
        child: NativeReduceChild,
    ) -> NativeRecursionAssemblyResult<BuildingRecord> {
        validate_final_replay_layout(&self.l3_program)?;
        record_reduce_child(
            request,
            idx,
            child,
            &self.lift_child_machine,
            &self.l2_child_machine,
            &self.l3_program,
            &self.lift_vk,
            &self.l2_vk,
        )
    }

    pub fn prove_l3_from_child_records(
        &self,
        request: &NativeRecursionRequest,
        arity: usize,
        child_records: Vec<BuildingRecord>,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
        validate_final_replay_layout(&self.l3_program)?;
        if child_records.is_empty() {
            return Err(validation("L3 requires at least one child"));
        }
        request.ensure_owns_records("L3", &child_records)?;
        let record_start = Instant::now();
        let record = merge_l3_child_records(child_records)?;
        let record = finalize_building_record(record, &self.l3_program, "L3_prerecorded")?;
        let kind = format!("L3(k={arity})");
        self.prove_node_with_record_ms(
            &kind,
            &self.l3_prover,
            &self.l3_pk,
            self.l3_device_pk(),
            &self.l3_vk,
            &self.l3_program,
            record,
            record_start.elapsed().as_millis(),
            stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_reduce_child_records_parallel(
        &self,
        request: &NativeRecursionRequest,
        children: Vec<NativeReduceChild>,
        parent_program: &RecursionNativeProgram<F>,
        lift_vk: &SCStarkVerifyingKey<SC>,
        l2_vk: &SCStarkVerifyingKey<SC>,
    ) -> NativeRecursionAssemblyResult<Vec<BuildingRecord>> {
        // Use the same process-global executor as all other CPU-parallel recursion work. This
        // keeps reduce recording within one CPU budget instead of spawning an arity-wide
        // OS-thread pool beside the prover's Rayon pool.
        let results = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lift_child_machine = &self.lift_child_machine;
            let l2_child_machine = &self.l2_child_machine;
            children
                .into_par_iter()
                .enumerate()
                .map(|(idx, child)| {
                    record_reduce_child(
                        request,
                        idx,
                        child,
                        lift_child_machine,
                        l2_child_machine,
                        parent_program,
                        lift_vk,
                        l2_vk,
                    )
                })
                .collect::<Vec<NativeRecursionAssemblyResult<_>>>()
        }))
        .map_err(|_| validation("shared reduce child recorder panicked"))?;
        results.into_iter().collect()
    }

    /// The L4 (root_shrink) node over one L3 shrink proof; `complete` marks the tree
    /// top (the in-circuit assert_complete set fires on it).
    pub fn prove_l4(
        &self,
        request: &NativeRecursionRequest,
        mut l3: SCMachineProof<SC>,
        complete: bool,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<RootSC>> {
        validate_final_replay_layout(&self.l4_program)?;
        let record_start = Instant::now();
        let mut seed = request.recording_seed(self.l3_shrink_machine.config.mlchallenger());
        crate::machine_dt::observe_replay_vk(&self.l3_vk, &mut seed);
        if l3.shard_proofs.len() != 1 {
            return Err(validation("L3 child proof must contain one shard"));
        }
        let l3_shard = l3.shard_proofs.pop().expect("length checked");
        let mut record = record_native_proof_shard(
            &self.l3_shrink_machine,
            &self.l3_vk,
            l3_shard,
            0,
            seed,
            &self.l4_program,
        )?;
        if complete {
            record.set_statement_is_complete(true);
        }
        let record = finalize_building_record(record, &self.l4_program, "L4")?;
        self.prove_node_with_record_ms(
            "L4(root)",
            &self.l4_prover,
            &self.l4_pk,
            self.l4_device_pk(),
            &self.l4_vk,
            &self.l4_program,
            record,
            record_start.elapsed().as_millis(),
            stats,
        )
    }

    fn prove_node_with_record_ms<C, PROV>(
        &self,
        kind: &str,
        prover: &PROV,
        pk: &SCStarkProvingKey<C>,
        device_pk: &PROV::DeviceProvingKey,
        vk: &SCStarkVerifyingKey<C>,
        program: &RecursionNativeProgram<F>,
        record: FinalizedRecord,
        record_ms: u128,
        stats: &mut Vec<NativeCompressNodeStat>,
    ) -> NativeRecursionAssemblyResult<SCMachineProof<C>>
    where
        C: crate::native_air_dt::NativeLayerProofConfig,
        PROV: SCMachineProver<C, crate::machine_dt::NativeRecursionAir, D_EF>,
        dt_stark::sumcheck::config::MlCom<C>: Send + Sync + Serialize,
        dt_stark::sumcheck::config::MlPcsProverData<C>:
            Send + Sync + Serialize + serde::de::DeserializeOwned,
        dt_stark::sumcheck::config::MlPcsOpeningProof<C>: Serialize,
        C::MlChallenger: Clone,
    {
        let raw = record.record();
        let poseidon2_unique = raw.poseidon2.unique_count();
        let poseidon2_total = raw.poseidon2.total_count();
        let diagnostics_enabled = node_diagnostics_enabled();
        let diagnostic_start = Instant::now();
        let (diagnostic_merkle_rows, merkle_leaf_rows, merkle_node_rows) = if diagnostics_enabled {
            let rows: usize =
                raw.proof_records.iter().map(|proof| proof.merkle_path.row_count()).sum();
            let leaf_rows = raw
                .proof_records
                .iter()
                .flat_map(|proof| proof.merkle_path.rows())
                .filter(|row| row.is_leaf_absorb())
                .count();
            (rows, Some(leaf_rows), Some(rows.saturating_sub(leaf_rows)))
        } else {
            (0, None, None)
        };
        let merkle_union_census = diagnostics_enabled.then(|| merkle_union_census(raw));
        let dag_node_mix_census = diagnostics_enabled.then(|| dag_node_mix_census(program));
        let diagnostic_census_ms = diagnostic_start.elapsed().as_millis();
        let prove_started_unix_ms = unix_now_ms();
        let start = Instant::now();
        let prove_result = prove_recursion_with_metrics(prover, pk, device_pk, record, program);
        let prove_ms = start.elapsed().as_millis();
        let prove_finished_unix_ms = unix_now_ms();
        let (proof, metrics) =
            prove_result.map_err(|err| validation(format!("prove {kind}: {err}")))?;
        let post_prove_verify_ms = if post_prove_verify_enabled() {
            let verify_start = Instant::now();
            verify_recursion(prover, vk, &proof)
                .map_err(|err| validation(format!("verify {kind}: {err}")))?;
            Some(verify_start.elapsed().as_millis())
        } else {
            None
        };
        let crate::machine_dt::ProveRecursionMetrics { timings, trace_costs } = metrics;
        let tallest = trace_costs
            .iter()
            .map(|cost| cost.height.next_power_of_two().trailing_zeros() as usize)
            .max()
            .unwrap_or(0);
        let merkle_rows = if diagnostics_enabled {
            diagnostic_merkle_rows
        } else {
            trace_costs
                .iter()
                .find(|cost| cost.chip == "NativeMerklePath")
                .map_or(0, |cost| cost.stored_height)
        };
        let proof_size_start = Instant::now();
        let proof_bytes = if diagnostics_enabled {
            Some(bincode::serialize(&proof).map_err(validation)?.len())
        } else {
            None
        };
        let proof_size_ms = proof_size_start.elapsed().as_millis();
        let accounted_prove_ms = timings
            .tracegen_ms
            .saturating_add(timings.budget_ms)
            .saturating_add(timings.pool_gate_ms)
            .saturating_add(timings.commit_ms)
            .saturating_add(timings.open_ms);
        stats.push(NativeCompressNodeStat {
            kind: kind.to_string(),
            record_generation: timings.record_generation,
            device_matrices: None,
            proof_bytes,
            diagnostics_enabled,
            diagnostic_census_ms,
            proof_size_ms,
            record_ms,
            prove_ms,
            post_prove_verify_ms,
            lift_bin_ready_ms: None,
            lift_worker_started_ms: None,
            prove_started_unix_ms,
            prove_finished_unix_ms,
            record_profile: timings.record_profile,
            poseidon2_memo: timings.poseidon2_memo,
            planned_chip_log_heights: timings.planned_chip_log_heights,
            row_count_admission_ms: timings.row_count_admission_ms,
            trace_plan_fold_ms: timings.trace_plan_fold_ms,
            tracegen_ms: timings.tracegen_ms,
            budget_ms: timings.budget_ms,
            pool_gate_ms: timings.pool_gate_ms,
            commit_ms: timings.commit_ms,
            commit_profile: timings.commit_profile,
            open_ms: timings.open_ms,
            open_profile: timings.open_profile,
            prove_residual_ms: prove_ms.saturating_sub(accounted_prove_ms),
            tallest_log_height: tallest,
            chips: trace_costs,
            poseidon2_unique,
            poseidon2_total,
            merkle_rows,
            merkle_leaf_rows,
            merkle_node_rows,
            merkle_union_census,
            dag_node_mix_census,
        });
        Ok(proof)
    }
}

pub(crate) fn checked_native_root_public_interval(
    public: &[F],
) -> NativeRecursionAssemblyResult<([[F; 11]; 3], [[F; 11]; 3])> {
    if public.len() != NATIVE_RECURSION_NUM_PV_ELTS {
        return Err(validation(format!(
            "root proof carries {} public values, expected {}",
            public.len(),
            NATIVE_RECURSION_NUM_PV_ELTS
        )));
    }
    for idx in 0..DIGEST_SIZE {
        if public[NATIVE_PV_VK_ROOT_START + idx] != F::zero() {
            return Err(validation("root proof exports a non-zero vk_root"));
        }
    }
    if public[NATIVE_PV_IS_COMPLETE] != F::one() {
        return Err(validation("root proof is not complete"));
    }
    let expected_digest = root_public_values_digest(public);
    for idx in 0..DIGEST_SIZE {
        if public[NATIVE_PV_DIGEST_START + idx] != expected_digest[idx] {
            return Err(validation("root PV digest slot != host root_public_values_digest"));
        }
    }
    Ok((
        core::array::from_fn(|coordinate| {
            core::array::from_fn(|limb| {
                public[NATIVE_PV_GLOBAL_INTERVAL_START + coordinate * 11 + limb]
            })
        }),
        core::array::from_fn(|coordinate| {
            core::array::from_fn(|limb| {
                public[NATIVE_PV_GLOBAL_INTERVAL_END + coordinate * 11 + limb]
            })
        }),
    ))
}

/// Sanity guard used by callers displaying PVs: canonical u32 read.
pub fn pv_u32(value: F) -> u32 {
    value.as_canonical_u32()
}

#[cfg(test)]
mod tests {
    use bincode::Options as _;
    use std::sync::{Arc, Barrier};

    use dt_stark::septic_digest::SepticDigest;
    use p3_field::{AbstractField, PrimeField32};

    use crate::{
        batch_constraint_dt::columns::{
            NUM_BATCH_SUMCHECK_COLS, NUM_BATCH_SUMCHECK_PACKED_COLS,
            NUM_BATCH_SUMCHECK_RESERVED_COLS,
        },
        config::{DIGEST_SIZE, F},
        constraint_replay_dt::columns::{
            NUM_CONSTRAINT_BOUNDARY_COLS, NUM_CONSTRAINT_BOUNDARY_PRECOMPUTED_COLS,
            NUM_CONSTRAINT_BOUNDARY_RESERVED_COLS, NUM_CONSTRAINT_FOLD_COLS,
            NUM_CONSTRAINT_FOLD_PRECOMPUTED_COLS, NUM_CONSTRAINT_FOLD_RESERVED_COLS,
            NUM_CONSTRAINT_TERMINAL_NARROW_COLS, NUM_CONSTRAINT_TERMINAL_PRECOMPUTED_NARROW_COLS,
            NUM_CONSTRAINT_TERMINAL_RESERVED_NARROW_COLS,
        },
        proof_shape_dt::NUM_PROOF_SHAPE_BINDER_COLS,
        whir_dt::{
            NUM_WHIR_QUERY_FOLD_COLS, NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS,
            NUM_WHIR_QUERY_FOLD_RESERVED_COLS,
        },
    };

    #[test]
    fn native_root_public_boundary_rejects_truncation_vk_root_and_incomplete() {
        let mut values = crate::statement_dt::NativeRecursionPublicValues::<F>::default();
        values.global_interval_start = [
            [F::zero(); 11],
            {
                let mut y = [F::zero(); 11];
                y[0] = F::one();
                y
            },
            [F::zero(); 11],
        ];
        values.global_interval_end = values.global_interval_start;
        values.is_complete = F::one();
        let mut public = values.as_array();
        let digest = super::root_public_values_digest(&public);
        public[super::NATIVE_PV_DIGEST_START..super::NATIVE_PV_DIGEST_START + DIGEST_SIZE]
            .copy_from_slice(&digest);
        super::checked_native_root_public_interval(&public).unwrap();

        assert!(super::checked_native_root_public_interval(&public[..public.len() - 1]).is_err());
        let mut bad_vk_root = public;
        bad_vk_root[super::NATIVE_PV_VK_ROOT_START] = F::one();
        assert!(super::checked_native_root_public_interval(&bad_vk_root).is_err());
        let mut incomplete = public;
        incomplete[super::NATIVE_PV_IS_COMPLETE] = F::zero();
        assert!(super::checked_native_root_public_interval(&incomplete).is_err());
    }

    #[test]
    fn optimized_native_air_layouts_are_cache_schema_visible() {
        assert_eq!(super::NATIVE_LADDER_CACHE_SCHEMA_VERSION, 25);
        assert_eq!(
            (
                NUM_CONSTRAINT_FOLD_COLS,
                NUM_CONSTRAINT_FOLD_RESERVED_COLS,
                NUM_CONSTRAINT_FOLD_PRECOMPUTED_COLS,
            ),
            (80, 8, 34)
        );
        assert_eq!(
            (
                NUM_WHIR_QUERY_FOLD_COLS,
                NUM_WHIR_QUERY_FOLD_RESERVED_COLS,
                NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS,
            ),
            (84, 33, 23)
        );
        assert_eq!(NUM_PROOF_SHAPE_BINDER_COLS, 83);
        assert_eq!(
            (
                NUM_BATCH_SUMCHECK_COLS,
                NUM_BATCH_SUMCHECK_RESERVED_COLS,
                NUM_BATCH_SUMCHECK_PACKED_COLS,
            ),
            (62, 3, 10)
        );
        assert_eq!(
            (
                NUM_CONSTRAINT_BOUNDARY_COLS,
                NUM_CONSTRAINT_BOUNDARY_RESERVED_COLS,
                NUM_CONSTRAINT_BOUNDARY_PRECOMPUTED_COLS,
            ),
            (167, 80, 41)
        );
        assert_eq!(
            (
                NUM_CONSTRAINT_TERMINAL_NARROW_COLS,
                NUM_CONSTRAINT_TERMINAL_RESERVED_NARROW_COLS,
                NUM_CONSTRAINT_TERMINAL_PRECOMPUTED_NARROW_COLS,
            ),
            (94, 6, 25)
        );
    }

    struct CacheTestDir(std::path::PathBuf);

    impl CacheTestDir {
        fn new(label: &str) -> Self {
            let sequence = super::NATIVE_LADDER_CACHE_TEMP_SEQUENCE
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("native-ladder-cache-{label}-{}-{sequence}", std::process::id()));
            std::fs::create_dir(&path).expect("create cache test directory");
            Self(path)
        }

        fn join(&self, name: &str) -> std::path::PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for CacheTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn unit_tests_enable_post_prove_verification() {
        assert!(super::post_prove_verify_enabled());
    }

    /// Explicit regeneration path for the SDK's frozen L4 digest. This remains ignored because
    /// it performs the full uncached L1 -> L4 setup.
    #[test]
    #[ignore]
    fn print_uncached_l4_digest_for_repin() {
        let ladder = super::NativeLadderContext::build().expect("build uncached native ladder");
        let digest = super::root_vk_digest(ladder.root_vk()).map(|limb| limb.as_canonical_u32());
        eprintln!("vk_L4 statement digest = {digest:?}");
    }

    /// Checks the constant-pair relations the two-constant septic chain relies on.
    /// Note: ±Z0 share an x-coordinate (negation flips y only), so START ≠ ±Z0 is
    /// the single x-inequality that must hold.
    #[test]
    fn septic_constant_pair_relations() {
        let start = SepticDigest::<F>::starting_digest_for_field().0;
        let z0 = SepticDigest::<F>::zero_for_field().0;
        assert_ne!(start.x, z0.x, "START x-collides with ±Z0");
        let start_plus_z0 = start.add_incomplete(z0);
        assert_ne!(start_plus_z0.x, z0.x, "x(START + Z0) == x(Z0)");
        let z0_minus_start = z0.add_incomplete(start.neg());
        assert_ne!(start.x, z0_minus_start.x, "x(START) == x(Z0 − START)");
        // The de-offset constant itself must not collide with the seed either.
        assert_ne!(z0_minus_start.x, start.x, "de-offset constant x-collides with START");
    }

    #[test]
    fn concurrent_cache_publish_has_one_complete_winner_and_no_temp_leaks() {
        const WRITERS: usize = 8;

        let dir = CacheTestDir::new("concurrent-publish");
        let cache_path = Arc::new(dir.join("ladder.bin"));
        let barrier = Arc::new(Barrier::new(WRITERS));
        let writers = (0..WRITERS)
            .map(|writer| {
                let cache_path = Arc::clone(&cache_path);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let payload = format!("complete-cache-candidate-{writer}").into_bytes();
                    let mut candidate = super::NativeLadderCacheTemp::create(&cache_path)
                        .expect("create candidate");
                    candidate.write_and_sync(&payload).expect("write candidate");
                    barrier.wait();
                    let outcome = candidate.publish_new(&cache_path).expect("publish candidate");
                    if outcome == super::NewCachePublish::DestinationExists {
                        candidate.cleanup().expect("clean losing candidate");
                    }
                    (outcome, payload)
                })
            })
            .collect::<Vec<_>>();
        let results = writers
            .into_iter()
            .map(|writer| writer.join().expect("join writer"))
            .collect::<Vec<_>>();

        let winners = results
            .iter()
            .filter(|(outcome, _)| *outcome == super::NewCachePublish::Published)
            .collect::<Vec<_>>();
        assert_eq!(winners.len(), 1);
        assert_eq!(std::fs::read(&*cache_path).expect("read winner"), winners[0].1);
        assert!(std::fs::read_dir(&dir.0).expect("read cache directory").all(|entry| !entry
            .expect("read cache entry")
            .file_name()
            .to_string_lossy()
            .contains(".tmp.")));
    }

    #[test]
    fn corrupt_cache_replacement_is_atomic_and_removes_its_candidate() {
        let dir = CacheTestDir::new("corrupt-replacement");
        let cache_path = dir.join("ladder.bin");
        std::fs::write(&cache_path, b"torn-envelope").expect("write corrupt cache");

        assert!(matches!(
            super::load_ladder_cache(&cache_path, "config", [0; DIGEST_SIZE])
                .expect("probe corrupt cache"),
            super::NativeLadderCacheLoad::RecoverableStorageCorruption
        ));

        let replacement = b"complete-replacement";
        let mut candidate =
            super::NativeLadderCacheTemp::create(&cache_path).expect("create candidate");
        candidate.write_and_sync(replacement).expect("write candidate");
        candidate.replace_existing(&cache_path).expect("replace corrupt cache");

        assert_eq!(std::fs::read(&cache_path).expect("read replacement"), replacement);
        assert!(std::fs::read_dir(&dir.0).expect("read cache directory").all(|entry| !entry
            .expect("read cache entry")
            .file_name()
            .to_string_lossy()
            .contains(".tmp.")));
    }

    #[test]
    fn force_publish_replaces_an_existing_cache_atomically() {
        let dir = CacheTestDir::new("force-replace");
        let cache_path = dir.join("ladder.bin");
        std::fs::write(&cache_path, b"old-valid-cache").expect("write existing cache");

        super::publish_ladder_cache_bytes(
            &cache_path,
            "config",
            [0; DIGEST_SIZE],
            b"rebuilt-cache",
            true,
        )
        .expect("force-replace cache");

        assert_eq!(std::fs::read(&cache_path).expect("read replacement"), b"rebuilt-cache");
        assert!(std::fs::read_dir(&dir.0).expect("read cache directory").all(|entry| !entry
            .expect("read cache entry")
            .file_name()
            .to_string_lossy()
            .contains(".tmp.")));
    }

    #[test]
    fn cache_publish_failure_is_path_bearing_and_non_destructive() {
        let dir = CacheTestDir::new("publish-failure");
        let parent_file = dir.join("not-a-directory");
        std::fs::write(&parent_file, b"block cache directory creation")
            .expect("write blocking parent file");
        let cache_path = parent_file.join("ladder.bin");

        let err = super::publish_ladder_cache_bytes(
            &cache_path,
            "config",
            [0; DIGEST_SIZE],
            b"fresh-context-remains-usable",
            true,
        )
        .expect_err("cache publication through a file must fail");

        assert!(err.to_string().contains(&cache_path.display().to_string()));
        assert_eq!(
            std::fs::read(&parent_file).expect("blocking file remains intact"),
            b"block cache directory creation"
        );
    }

    #[test]
    fn cache_read_and_integrity_checked_decode_errors_include_the_keyed_path() {
        let dir = CacheTestDir::new("path-errors");
        let read_error_path = dir.join("cache-is-a-directory");
        std::fs::create_dir(&read_error_path).expect("create directory at cache path");
        let read_err = match super::load_ladder_cache(&read_error_path, "config", [0; DIGEST_SIZE])
        {
            Err(err) => err,
            Ok(_) => panic!("reading a directory as a cache must fail"),
        };
        assert!(read_err.to_string().contains(&read_error_path.display().to_string()));

        let decode_error_path = dir.join("decode-error.bin");
        let artifacts_bytes = b"not-native-ladder-artifacts".to_vec();
        let file = super::NativeLadderCacheFile {
            schema_version: super::NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
            registry_version: crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            setup_hash: super::ladder_cache_setup_hash(),
            config_hash: "config".to_string(),
            expected_l4_digest: [0; DIGEST_SIZE],
            artifacts_hash: super::stable_hash64(&artifacts_bytes),
            artifacts_bytes,
        };
        std::fs::write(
            &decode_error_path,
            bincode::serialize(&file).expect("serialize decode-error envelope"),
        )
        .expect("write decode-error envelope");
        let decode_err =
            match super::load_ladder_cache(&decode_error_path, "config", [0; DIGEST_SIZE]) {
                Err(err) => err,
                Ok(_) => panic!("integrity-checked invalid artifacts must fail"),
            };
        assert!(decode_err.to_string().contains(&decode_error_path.display().to_string()));
    }

    #[test]
    fn native_layer_keyed_cache_metadata_mismatch_is_not_repaired_as_storage_corruption() {
        let dir = CacheTestDir::new("semantic-mismatch");
        let cache_path = dir.join("ladder.bin");
        let expected_l4_digest = [0; DIGEST_SIZE];
        let file = super::NativeLadderCacheFile {
            schema_version: super::NATIVE_LADDER_CACHE_SCHEMA_VERSION + 1,
            global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
            registry_version: crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            setup_hash: super::ladder_cache_setup_hash(),
            config_hash: "config".to_string(),
            expected_l4_digest,
            artifacts_hash: super::stable_hash64(&[]),
            artifacts_bytes: Vec::new(),
        };
        std::fs::write(&cache_path, bincode::serialize(&file).expect("serialize envelope"))
            .expect("write envelope");

        let err = match super::load_ladder_cache(&cache_path, "config", expected_l4_digest) {
            Err(err) => err,
            Ok(_) => panic!("semantic metadata mismatch must fail closed"),
        };
        assert!(err.to_string().contains("schema_version"));
        assert!(err.to_string().contains(&cache_path.display().to_string()));
        assert_eq!(
            std::fs::read(&cache_path).expect("read rejected cache"),
            bincode::serialize(&file).expect("serialize envelope")
        );
    }

    #[test]
    fn native_layer_cache_path_and_envelope_round_trip_include_registry_authority() {
        let dir = CacheTestDir::new("registry-authority");
        let expected_l4_digest = [7; DIGEST_SIZE];
        let path = super::ladder_cache_path(&dir.0, "config", expected_l4_digest);
        let file_name = path.file_name().expect("cache file name").to_string_lossy();
        assert!(file_name.contains(&format!(
            "-v{}-registry{}-",
            super::NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION
        )));

        let file = super::NativeLadderCacheFile {
            schema_version: super::NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
            registry_version: crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            setup_hash: super::ladder_cache_setup_hash(),
            config_hash: "config".to_string(),
            expected_l4_digest,
            artifacts_hash: super::stable_hash64(b"round-trip"),
            artifacts_bytes: b"round-trip".to_vec(),
        };
        let encoded = bincode::serialize(&file).expect("serialize current envelope");
        let decoded: super::NativeLadderCacheFile =
            bincode::deserialize(&encoded).expect("deserialize current envelope");
        assert_eq!(decoded.schema_version, super::NATIVE_LADDER_CACHE_SCHEMA_VERSION);
        assert_eq!(decoded.registry_version, crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION);
        assert_eq!(decoded.config_hash, "config");
        assert_eq!(decoded.expected_l4_digest, expected_l4_digest);
        assert_eq!(decoded.artifacts_hash, super::stable_hash64(&decoded.artifacts_bytes));
    }

    #[test]
    fn malformed_cache_length_is_bounded_without_allocation_or_panic() {
        let claimed_len = usize::MAX.to_le_bytes();
        let result = std::panic::catch_unwind(|| {
            super::bounded_cache_bincode().deserialize::<Vec<u8>>(&claimed_len)
        });
        assert!(result.is_ok(), "bounded cache decoder panicked");
        assert!(result.expect("checked above").is_err());
    }

    #[test]
    fn oversized_cache_file_is_rejected_before_reading_or_decoding() {
        let dir = CacheTestDir::new("oversized");
        let cache_path = dir.join("ladder.bin");
        std::fs::File::create(&cache_path)
            .expect("create sparse cache")
            .set_len(super::NATIVE_LADDER_CACHE_MAX_BYTES + 1)
            .expect("size sparse cache");

        let result = std::panic::catch_unwind(|| {
            super::load_ladder_cache(&cache_path, "config", [0; DIGEST_SIZE])
        });
        assert!(result.is_ok(), "oversized cache load panicked");
        let message = match result.expect("checked above") {
            Err(err) => err.to_string(),
            Ok(_) => panic!("oversized cache must fail closed"),
        };
        assert!(message.contains("reject oversized ladder cache"), "{message}");
    }

    #[test]
    fn native_layer_cache_rejects_stale_schema_and_registry_versions() {
        fn rejected_field(label: &str, schema_version: u32, registry_version: u32) -> String {
            let dir = CacheTestDir::new(label);
            let cache_path = dir.join("ladder.bin");
            let expected_l4_digest = [0; DIGEST_SIZE];
            let file = super::NativeLadderCacheFile {
                schema_version,
                global146_identity: dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY,
                registry_version,
                package_version: env!("CARGO_PKG_VERSION").to_string(),
                setup_hash: super::ladder_cache_setup_hash(),
                config_hash: "config".to_string(),
                expected_l4_digest,
                artifacts_hash: super::stable_hash64(&[]),
                artifacts_bytes: Vec::new(),
            };
            std::fs::write(&cache_path, bincode::serialize(&file).expect("serialize envelope"))
                .expect("write envelope");
            match super::load_ladder_cache(&cache_path, "config", expected_l4_digest) {
                Err(err) => {
                    let message = err.to_string();
                    assert!(message.contains(&cache_path.display().to_string()));
                    message
                }
                Ok(_) => panic!("obsolete cache authority must fail closed"),
            }
        }

        assert_eq!(super::NATIVE_LADDER_CACHE_SCHEMA_VERSION - 1, 24);
        assert_eq!(crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION - 1, 12);
        assert!(rejected_field(
            "prior-schema",
            super::NATIVE_LADDER_CACHE_SCHEMA_VERSION - 1,
            crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION
        )
        .contains("schema_version"));
        assert!(rejected_field(
            "old-registry",
            super::NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION - 1
        )
        .contains("registry_version"));
        assert!(rejected_field(
            "pre-epoch-schema",
            4,
            crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION
        )
        .contains("schema_version"));

        assert_eq!(crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION, 11);
        assert_eq!(super::NATIVE_LADDER_CACHE_SCHEMA_VERSION, 25);
        assert_eq!(crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION, 13);
    }

    #[test]
    fn native_layer_cache_rejects_wrong_global146_identity() {
        let dir = CacheTestDir::new("wrong-global146-identity");
        let cache_path = dir.join("ladder.bin");
        let expected_l4_digest = [0; DIGEST_SIZE];
        let mut wrong_identity = dt_stark::global_d11::GLOBAL146_COMPOSITE_IDENTITY;
        wrong_identity[0] ^= 1;
        let file = super::NativeLadderCacheFile {
            schema_version: super::NATIVE_LADDER_CACHE_SCHEMA_VERSION,
            global146_identity: wrong_identity,
            registry_version: crate::native_air_dt::NATIVE_AIR_REGISTRY_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            setup_hash: super::ladder_cache_setup_hash(),
            config_hash: "config".to_string(),
            expected_l4_digest,
            artifacts_hash: super::stable_hash64(&[]),
            artifacts_bytes: Vec::new(),
        };
        std::fs::write(&cache_path, bincode::serialize(&file).expect("serialize envelope"))
            .expect("write envelope");
        let err = match super::load_ladder_cache(&cache_path, "config", expected_l4_digest) {
            Err(err) => err,
            Ok(_) => panic!("wrong Global146 identity must fail closed"),
        };
        assert!(err.to_string().contains("global146_identity"));
    }
}
