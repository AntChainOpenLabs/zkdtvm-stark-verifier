use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use crate::{
    batch_constraint_dt::record::BatchTranscriptLayout,
    child_views::{NativeChildRole, NativeChildViews},
    config::{ChildMlPcsOpeningProof, DIGEST_SIZE, D_EF, EF, F},
    proof_shape_dt::{
        PROOF_SHAPE_COMMIT_MAIN, PROOF_SHAPE_COMMIT_PERMUTATION, PROOF_SHAPE_COMMIT_VK,
    },
    system_dt::{
        spec_fold::limbs_to_ext, RecursionMerklePathOp, RecursionMerklePathRow,
        RecursionProofRecord, RecursionRecord, RecursionTranscriptBitsEvent,
        RecursionTranscriptEvent, RecursionTranscriptEventKind, RecursionWhirLeafExtStreamRow,
        RecursionWhirLeafExtStreamTraceRow, RecursionWhirLeafStreamRow, RecursionWhirQueryFoldRow,
        RecursionWhirRecord, RecursionWhirTracegenSource, WhirBatchRlc, WhirOpenedMatrices,
        WhirQueryRoundControl, WhirRoundReplayInput, WhirSpecFoldError, WhirSpecFoldSeed,
        WhirSpecFoldShape, WHIR_BATCH_MAIN, WHIR_BATCH_PERMUTATION, WHIR_BATCH_PREPROCESSED,
    },
    transcript_dt::poseidon2::RecursionPoseidon2Output,
    whir_dt::{
        columns::{
            whir_unit_key, WHIR_BATCHING_POW_HIGH_MAX, WHIR_INPUT_MAIN_PATH_SLOT,
            WHIR_INPUT_PERMUTATION_PATH_SLOT, WHIR_INPUT_PREPROCESSED_PATH_SLOT,
            WHIR_IOPP_ORACLE_PATH_SLOT_BASE, WHIR_PAIRED_RANGE_BITS, WHIR_QUERY_POW_HIGH_MAX,
            WHIR_ROLE_COMPRESS, WHIR_ROLE_CORE, WHIR_ROLE_SHRINK, WHIR_TWIDDLE_ROWS,
            WHIR_TWIDDLE_TABLES,
        },
        trace::whir_role_config,
    },
    Instant,
};
use dt_stark::sumcheck::config::{MlCom, SCStarkGenericConfig};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use p3_maybe_rayon::prelude::*;
use pcs::basefold::mlpcs::MlPCS;

#[cfg(test)]
use crate::transcript_dt::poseidon2::RecursionPoseidon2Memo;

pub(crate) struct WhirTracegenMaterialHeader {
    shape: WhirSpecFoldShape,
    input_roots: Vec<[F; DIGEST_SIZE]>,
    publish_opened_eval: bool,
}

/// Prepare the small header while the child layout is borrowed. The large
/// proof buffers are attached later by move, after the borrow has ended.
pub(crate) fn prepare_whir_tracegen_materials<ChildSC>(
    record: &RecursionRecord,
    proof_idx: usize,
    views: &NativeChildViews<'_, ChildSC>,
    publish_opened_eval: bool,
) -> Result<WhirTracegenMaterialHeader, WhirRecordError>
where
    ChildSC: SCStarkGenericConfig<Val = F, Challenge = EF, MlChallenge = EF>,
    <ChildSC as SCStarkGenericConfig>::Mlpcs: MlPCS<BatchProof = ChildMlPcsOpeningProof>,
    MlCom<ChildSC>: AsRef<[F; DIGEST_SIZE]>,
{
    reject_unsupported_modes(views.proof.opening_proof())?;
    let shape = preflight_child_whir(record, proof_idx, views)?;
    let commitment = views.proof.commitment();
    let mut input_roots = vec![*views.vk.vk().commit.as_ref(), *commitment.main_commit.as_ref()];
    if let Some(permutation) = commitment.permutation_commit.as_ref() {
        input_roots.push(*permutation.as_ref());
    }
    let proof_record = proof_record_by_idx(record, proof_idx)?;
    if proof_record.whir_source.is_some() {
        return Err(WhirRecordError::IoppQueryCountMismatch { expected: 0, actual: 1 });
    }
    Ok(WhirTracegenMaterialHeader { shape: shape.into(), input_roots, publish_opened_eval })
}

/// Move the large child material into its single-owner tracegen source.
pub(crate) fn attach_whir_tracegen_materials(
    record: &mut RecursionRecord,
    proof_idx: usize,
    header: WhirTracegenMaterialHeader,
    opening_proof: ChildMlPcsOpeningProof,
    opened_values: dt_stark::sumcheck::proof::SCShardOpenedValues<F, EF>,
    dimensions: Vec<Vec<p3_matrix::Dimensions>>,
) -> Result<(), WhirRecordError> {
    let proof_record = record.proof_record_mut(proof_idx);
    if proof_record.whir_source.is_some() {
        return Err(WhirRecordError::IoppQueryCountMismatch { expected: 0, actual: 1 });
    }
    proof_record.whir_source = Some(RecursionWhirTracegenSource {
        shape: header.shape,
        opening_proof: Arc::new(opening_proof),
        opened_values: Arc::new(opened_values),
        dimensions,
        input_roots: header.input_roots,
        publish_opened_eval: header.publish_opened_eval,
        opened_eval_publications: Vec::new(),
    });
    Ok(())
}

/// Move-only work item at the compact-source tracegen boundary.
///
/// This deliberately does not implement `Clone` or serialization: either the CPU fallback or a
/// device backend owns and consumes the source, never both.
#[derive(Debug)]
pub struct OwnedWhirTracegenSource {
    proof_idx: usize,
    source: RecursionWhirTracegenSource,
}

impl OwnedWhirTracegenSource {
    pub const fn proof_idx(&self) -> usize {
        self.proof_idx
    }

    pub const fn source(&self) -> &RecursionWhirTracegenSource {
        &self.source
    }

    pub fn into_parts(self) -> (usize, RecursionWhirTracegenSource) {
        (self.proof_idx, self.source)
    }
}

/// Single-owner batch handed from semantic preparation to a tracegen backend.
#[derive(Debug)]
pub struct WhirTracegenSourceBatch {
    record_proof_count: usize,
    sources: Vec<OwnedWhirTracegenSource>,
}

impl WhirTracegenSourceBatch {
    pub const fn record_proof_count(&self) -> usize {
        self.record_proof_count
    }

    pub fn sources(&self) -> &[OwnedWhirTracegenSource] {
        &self.sources
    }

    pub fn into_sources(self) -> Vec<OwnedWhirTracegenSource> {
        self.sources
    }
}

/// One canonical leaf chain consumed by the production device candidate path.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactMerkleLeafDescriptor {
    pub proof_idx: u64,
    pub unit_key: u64,
    pub commit_id: u64,
    pub level: u64,
    pub cur_idx: u64,
    pub block_offset: u64,
    pub block_count: u64,
    pub absorb_count: u64,
    pub iopp_pair: u64,
}

/// A typed eight-element IOPP-pair leaf absorb block. Input-Merkle leaves
/// instead use [`CompactMerkleLeafBlockRef`] to borrow the WHIR leaf value
/// arena, so only IOPP pairs retain an independent value payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactMerkleLeafBlock {
    pub block_idx: u64,
    pub chunk: [u32; 8],
    pub chunk_mask: u64,
}

pub const COMPACT_MERKLE_LEAF_SOURCE_BASE: u32 = 0;
pub const COMPACT_MERKLE_LEAF_SOURCE_EXT_BASE: u32 = 1;
pub const COMPACT_MERKLE_LEAF_SOURCE_EXT_COUNT: u32 = D_EF as u32;
pub const COMPACT_MERKLE_LEAF_SOURCE_IOPP: u32 =
    COMPACT_MERKLE_LEAF_SOURCE_EXT_BASE + COMPACT_MERKLE_LEAF_SOURCE_EXT_COUNT;

/// Eight-byte block view into the one leaf authority arena.
///
/// Base views refer to `CompactWhirLeafBaseInput`; extension views encode the
/// flattened eight-limb subblock in `source_kind`; IOPP views refer to the
/// compact independent `CompactMerkleLeafBlock` arena.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactMerkleLeafBlockRef {
    pub source_idx: u32,
    pub source_kind: u32,
}

/// One raw proof path. Its output range is deterministic and disjoint from
/// every other path, so the device never assigns row identity by atomics.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactMerklePathDescriptor {
    pub proof_idx: u64,
    pub commit_id: u64,
    pub start_leaf: u64,
    pub cur_idx: u64,
    pub step_offset: u64,
    pub step_count: u64,
    pub output_offset: u64,
    pub output_count: u64,
}

/// One path-compress level and its optional mixed-height leaf injection.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactMerklePathStep {
    pub sibling: [u32; 8],
    /// `u64::MAX` means that this level has no injection.
    pub injected_leaf: u64,
}

/// Canonical per-proof ranges for compact WHIR producer rows.
///
/// Merkle rows are produced on device, but their final proof-local ranges are
/// fixed here so Pass B can scatter the selected groups in the same
/// proof/leaf/node order as the CPU oracle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirProofRowsDescriptor {
    pub proof_idx: u64,
    pub merkle_row_offset: u64,
    pub merkle_row_count: u64,
    pub whir_round_offset: u64,
    pub whir_round_count: u64,
    pub whir_batch_eval_offset: u64,
    pub whir_batch_eval_count: u64,
    pub whir_query_fold_offset: u64,
    pub whir_query_fold_count: u64,
    pub whir_leaf_stream_offset: u64,
    pub whir_leaf_stream_count: u64,
    pub whir_leaf_ext_stream_offset: u64,
    pub whir_leaf_ext_stream_count: u64,
}

/// One deduplicated WHIR leaf-height instance. A single device thread walks
/// its ordered row references and derives the alpha-power/accumulator chain.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirLeafGroupDescriptor {
    pub proof_idx: u32,
    pub idx: u32,
    pub log_height: u32,
    pub serve_cnt: u32,
    pub row_ref_offset: u32,
    pub row_ref_count: u32,
    pub alpha: [u32; D_EF],
    pub start_pow: [u32; D_EF],
}

pub const COMPACT_WHIR_LEAF_ROW_BASE: u32 = 0;
pub const COMPACT_WHIR_LEAF_ROW_EXT: u32 = 1;

/// Ordered reference from a height-group chain to one base or extension row.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirLeafRowRef {
    pub kind: u32,
    pub row_idx: u32,
}

/// Compact base-field leaf row input. Prefix powers and accumulators are
/// intentionally absent: the device expansion derives them from the group.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirLeafBaseInput {
    pub batch_id: u32,
    pub block_idx: u32,
    pub value_count: u32,
    pub output_idx: u32,
    pub values: [u32; 8],
}

/// Compact extension-field leaf row input. Values remain source material;
/// all row witnesses are derived after upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirLeafExtInput {
    pub batch_id: u32,
    pub block_idx: u32,
    pub value_count: u32,
    pub output_idx: u32,
    pub values: [[u32; D_EF]; 8],
}

/// One proof-local opened-evaluation chain. A single device thread expands
/// its ordered segments and derives all alpha powers and accumulators.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirBatchProofDescriptor {
    pub proof_idx: u32,
    pub alpha_tidx: u32,
    pub segment_offset: u32,
    pub segment_count: u32,
    pub value_offset: u32,
    pub value_count: u32,
    pub output_offset: u32,
    pub output_count: u32,
    pub alpha: [u32; D_EF],
}

/// Structural metadata for one opened matrix in the canonical height-group
/// order. Values live in the adjacent compact value arena.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirBatchSegment {
    pub log_height: u32,
    pub batch_id: u32,
    pub batch_pos: u32,
    pub chip_idx: u32,
    pub static_chip_id: u32,
    pub width: u32,
    pub value_offset: u32,
    pub value_count: u32,
    pub pow_seed_cnt: u32,
}

/// One source opened evaluation plus its independently captured publication
/// multiplicity. No chain witness is carried across H2D.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirBatchValue {
    pub value: [u32; D_EF],
    pub opened_eval_send_mult: u32,
}

/// One proof-level WHIR round replay. Transcript samples and static shape
/// remain CPU authority; every expanded round-chain witness is derived after
/// upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirRoundProofDescriptor {
    pub proof_idx: u32,
    pub output_offset: u32,
    pub output_count: u32,
    pub input_offset: u32,
    pub input_count: u32,
    pub group_offset: u32,
    pub group_count: u32,
    pub num_rounds: u32,
    pub num_queries: u32,
    pub log_blowup: u32,
    pub w0_tidx: u32,
    /// `u32::MAX` denotes that no round emits the preparation seed.
    pub prep_seed_round: u32,
    pub batching_pow_events: [u32; 3],
    pub query_pow_events: [u32; 3],
    pub final_oracle: [u32; 8],
}

/// One height-group claim used by the round merge chain.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirRoundGroup {
    pub log_height: u32,
    pub rank: u32,
    pub claim: [u32; D_EF],
}

/// Raw authority for one sumcheck round. Accumulators, equality folds, chain
/// links, CFR, and final-root Poseidon states are deliberately absent.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirRoundInput {
    pub sumcheck_coeffs: [[u32; D_EF]; 3],
    pub r_fold: [u32; D_EF],
    pub opening_point: [u32; D_EF],
    pub merge_beta: [u32; D_EF],
    pub iopp_oracle: [u32; 8],
}

/// One proof-level QueryFold round control. Query-independent replay fields
/// are uploaded once and shared by every query in the proof.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirQueryControl {
    pub round_row_idx: u32,
    pub is_merge: u32,
    pub r_fold: [u32; D_EF],
    pub merge_beta: [u32; D_EF],
    pub merge_eq: [u32; D_EF],
}

/// One query chain. The device derives twiddles, affine chain state, selected
/// pairs, and folded values from the raw transcript sample and round inputs.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirQueryDescriptor {
    pub proof_idx: u32,
    pub query_idx: u32,
    pub query_sample_raw: u32,
    pub query_bits: u32,
    pub control_offset: u32,
    pub control_count: u32,
    pub input_offset: u32,
    pub input_count: u32,
    pub output_offset: u32,
    pub output_count: u32,
    pub final_round_row_idx: u32,
}

/// One IOPP sibling plus the deduplicated leaf-group accumulator selected by
/// the matching round control. `u32::MAX` denotes a non-merge round.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactWhirQueryRoundInput {
    pub sibling: [u32; D_EF],
    pub leaf_group_idx: u32,
    /// Two device-generated `CompactMerkleLeafBlock`s start here. `u32::MAX`
    /// means that another duplicate query round owns the canonical IOPP leaf.
    pub iopp_block_offset: u32,
}

/// Additional Poseidon publication produced by WHIR outside the Merkle union.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactPoseidonCandidate {
    pub input: [u32; 16],
    pub count: u64,
}

/// Typed publication for the one complete-key range-provider domain.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactRangeCandidate {
    pub value: u32,
    pub max_bits: u32,
    pub count: u64,
}

/// Opt-in CPU telemetry for compact DTO construction.
///
/// Timings use microseconds so the telemetry can locate sub-millisecond work. The switch is read
/// once at the builder boundary and disabled production runs do not call `Instant::now` inside
/// proof/query loops.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactDtoTelemetry {
    pub total_us: u64,
    pub proof_local_build_wall_us: u64,
    pub directory_build_us: u64,
    pub proof_setup_round_replay_us: u64,
    pub leaf_group_key_collection_us: u64,
    pub leaf_value_packing_us: u64,
    pub input_path_descriptor_us: u64,
    pub query_fold_iopp_pair_us: u64,
    pub proof_finalize_us: u64,
    pub leaf_group_count: usize,
    pub leaf_base_row_count: usize,
    pub leaf_ext_row_count: usize,
    pub leaf_block_count: usize,
    pub path_descriptor_count: usize,
    pub path_step_count: usize,
    pub node_occurrence_count: usize,
    pub retained_arena_allocation_count: usize,
    pub retained_arena_bytes: usize,
    pub configured_source_workers: usize,
    pub active_source_workers: usize,
    pub proof_blob_count: usize,
}

/// One proof-owned compact source blob.
///
/// Every offset is relative to this blob. The payload is never flattened into
/// a second host arena; [`ProofArenaDirectory`] supplies its final device
/// bases.
#[derive(Debug)]
pub struct ProofCompactBlob {
    proof_idx: usize,
    generation: u64,
    program_authority: u64,
    leaf_descriptors: Vec<CompactMerkleLeafDescriptor>,
    leaf_block_refs: Vec<CompactMerkleLeafBlockRef>,
    iopp_leaf_block_count: usize,
    path_descriptors: Vec<CompactMerklePathDescriptor>,
    path_steps: Vec<CompactMerklePathStep>,
    ordinary_path_occurrences: usize,
    node_candidate_occurrences: usize,
    proof_rows: Vec<CompactWhirProofRowsDescriptor>,
    batch_proof_descriptors: Vec<CompactWhirBatchProofDescriptor>,
    batch_segments: Vec<CompactWhirBatchSegment>,
    batch_values: Vec<CompactWhirBatchValue>,
    round_proof_descriptors: Vec<CompactWhirRoundProofDescriptor>,
    round_groups: Vec<CompactWhirRoundGroup>,
    round_inputs: Vec<CompactWhirRoundInput>,
    query_controls: Vec<CompactWhirQueryControl>,
    query_descriptors: Vec<CompactWhirQueryDescriptor>,
    query_round_inputs: Vec<CompactWhirQueryRoundInput>,
    leaf_group_descriptors: Vec<CompactWhirLeafGroupDescriptor>,
    leaf_row_refs: Vec<CompactWhirLeafRowRef>,
    leaf_base_inputs: Vec<CompactWhirLeafBaseInput>,
    leaf_ext_inputs: Vec<CompactWhirLeafExtInput>,
    poseidon_candidates: Vec<CompactPoseidonCandidate>,
    range_candidates: Vec<CompactRangeCandidate>,
    telemetry: Option<CompactDtoTelemetry>,
}

impl ProofCompactBlob {
    pub const fn proof_idx(&self) -> usize {
        self.proof_idx
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn program_authority(&self) -> u64 {
        self.program_authority
    }

    pub fn leaf_descriptors(&self) -> &[CompactMerkleLeafDescriptor] {
        &self.leaf_descriptors
    }

    pub fn leaf_block_refs(&self) -> &[CompactMerkleLeafBlockRef] {
        &self.leaf_block_refs
    }

    pub const fn iopp_leaf_block_count(&self) -> usize {
        self.iopp_leaf_block_count
    }

    pub fn path_descriptors(&self) -> &[CompactMerklePathDescriptor] {
        &self.path_descriptors
    }

    pub fn path_steps(&self) -> &[CompactMerklePathStep] {
        &self.path_steps
    }

    pub const fn ordinary_path_occurrences(&self) -> usize {
        self.ordinary_path_occurrences
    }

    pub const fn node_candidate_occurrences(&self) -> usize {
        self.node_candidate_occurrences
    }

    pub fn proof_rows(&self) -> &[CompactWhirProofRowsDescriptor] {
        &self.proof_rows
    }

    pub fn batch_proof_descriptors(&self) -> &[CompactWhirBatchProofDescriptor] {
        &self.batch_proof_descriptors
    }

    pub fn batch_segments(&self) -> &[CompactWhirBatchSegment] {
        &self.batch_segments
    }

    pub fn batch_values(&self) -> &[CompactWhirBatchValue] {
        &self.batch_values
    }

    pub fn round_proof_descriptors(&self) -> &[CompactWhirRoundProofDescriptor] {
        &self.round_proof_descriptors
    }

    pub fn round_groups(&self) -> &[CompactWhirRoundGroup] {
        &self.round_groups
    }

    pub fn round_inputs(&self) -> &[CompactWhirRoundInput] {
        &self.round_inputs
    }

    pub fn query_controls(&self) -> &[CompactWhirQueryControl] {
        &self.query_controls
    }

    pub fn query_descriptors(&self) -> &[CompactWhirQueryDescriptor] {
        &self.query_descriptors
    }

    pub fn query_round_inputs(&self) -> &[CompactWhirQueryRoundInput] {
        &self.query_round_inputs
    }

    pub fn leaf_group_descriptors(&self) -> &[CompactWhirLeafGroupDescriptor] {
        &self.leaf_group_descriptors
    }

    pub fn leaf_row_refs(&self) -> &[CompactWhirLeafRowRef] {
        &self.leaf_row_refs
    }

    pub fn leaf_base_inputs(&self) -> &[CompactWhirLeafBaseInput] {
        &self.leaf_base_inputs
    }

    pub fn leaf_ext_inputs(&self) -> &[CompactWhirLeafExtInput] {
        &self.leaf_ext_inputs
    }

    pub fn poseidon_candidates(&self) -> &[CompactPoseidonCandidate] {
        &self.poseidon_candidates
    }

    pub fn range_candidates(&self) -> &[CompactRangeCandidate] {
        &self.range_candidates
    }

    pub const fn telemetry(&self) -> Option<&CompactDtoTelemetry> {
        self.telemetry.as_ref()
    }

    pub fn candidate_capacity(&self) -> Option<usize> {
        self.leaf_block_refs.len().checked_add(self.node_candidate_occurrences)
    }

    /// Bytes in the proof-local structural metadata slab.
    ///
    /// Large value/sibling arenas are deliberately excluded and retain their
    /// single proof owner. The GPU endpoint packs only these small POD slices
    /// into pinned staging before one direct H2D.
    pub fn packed_metadata_bytes(&self) -> Option<usize> {
        let mut bytes = 0usize;
        macro_rules! add_slice {
            ($slice:expr, $ty:ty) => {
                bytes =
                    bytes.checked_add($slice.len().checked_mul(core::mem::size_of::<$ty>())?)?;
            };
        }
        add_slice!(self.leaf_descriptors, CompactMerkleLeafDescriptor);
        add_slice!(self.leaf_block_refs, CompactMerkleLeafBlockRef);
        add_slice!(self.path_descriptors, CompactMerklePathDescriptor);
        add_slice!(self.batch_proof_descriptors, CompactWhirBatchProofDescriptor);
        add_slice!(self.batch_segments, CompactWhirBatchSegment);
        add_slice!(self.round_proof_descriptors, CompactWhirRoundProofDescriptor);
        add_slice!(self.round_groups, CompactWhirRoundGroup);
        add_slice!(self.query_controls, CompactWhirQueryControl);
        add_slice!(self.query_descriptors, CompactWhirQueryDescriptor);
        add_slice!(self.leaf_group_descriptors, CompactWhirLeafGroupDescriptor);
        add_slice!(self.leaf_row_refs, CompactWhirLeafRowRef);
        Some(bytes)
    }

    pub fn h2d_bytes(&self) -> Option<usize> {
        self.leaf_descriptors
            .len()
            .checked_mul(core::mem::size_of::<CompactMerkleLeafDescriptor>())
            .and_then(|bytes| {
                self.leaf_block_refs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactMerkleLeafBlockRef>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.path_descriptors
                    .len()
                    .checked_mul(core::mem::size_of::<CompactMerklePathDescriptor>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.path_steps
                    .len()
                    .checked_mul(core::mem::size_of::<CompactMerklePathStep>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.leaf_group_descriptors
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirLeafGroupDescriptor>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.leaf_row_refs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirLeafRowRef>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.leaf_base_inputs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirLeafBaseInput>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.leaf_ext_inputs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirLeafExtInput>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.batch_proof_descriptors
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirBatchProofDescriptor>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.batch_segments
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirBatchSegment>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.batch_values
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirBatchValue>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.round_proof_descriptors
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirRoundProofDescriptor>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.round_groups
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirRoundGroup>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.round_inputs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirRoundInput>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.query_controls
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirQueryControl>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.query_descriptors
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirQueryDescriptor>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.query_round_inputs
                    .len()
                    .checked_mul(core::mem::size_of::<CompactWhirQueryRoundInput>())
                    .and_then(|next| bytes.checked_add(next))
            })
    }

    /// Bytes in the typed compact producer DTO. This excludes the Merkle
    /// preimage/path arena, whose transfer is reported by [`Self::h2d_bytes`].
    pub fn producer_h2d_bytes(&self) -> Option<usize> {
        self.proof_rows
            .len()
            .checked_mul(core::mem::size_of::<CompactWhirProofRowsDescriptor>())
            .and_then(|bytes| {
                self.poseidon_candidates
                    .len()
                    .checked_mul(core::mem::size_of::<CompactPoseidonCandidate>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.range_candidates
                    .len()
                    .checked_mul(core::mem::size_of::<CompactRangeCandidate>())
                    .and_then(|next| bytes.checked_add(next))
            })
    }
}

/// Canonical final-device ranges for one proof blob.
///
/// The table is built once in proof-index order. CUDA rebases the relative POD
/// metadata after direct H2D; CPU never mutates or copies a proof payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProofArenaDirectory {
    pub proof_idx: u64,
    pub metadata_slab_base: u64,
    pub metadata_slab_count: u64,
    pub leaf_descriptor_base: u64,
    pub leaf_descriptor_count: u64,
    pub leaf_block_ref_base: u64,
    pub leaf_block_ref_count: u64,
    pub iopp_leaf_block_base: u64,
    pub iopp_leaf_block_count: u64,
    pub path_descriptor_base: u64,
    pub path_descriptor_count: u64,
    pub path_step_base: u64,
    pub path_step_count: u64,
    pub node_candidate_base: u64,
    pub node_candidate_count: u64,
    pub batch_proof_base: u64,
    pub batch_proof_count: u64,
    pub batch_segment_base: u64,
    pub batch_segment_count: u64,
    pub batch_value_base: u64,
    pub batch_value_count: u64,
    pub batch_output_base: u64,
    pub batch_output_count: u64,
    pub round_proof_base: u64,
    pub round_proof_count: u64,
    pub round_group_base: u64,
    pub round_group_count: u64,
    pub round_input_base: u64,
    pub round_input_count: u64,
    pub round_output_base: u64,
    pub round_output_count: u64,
    pub query_control_base: u64,
    pub query_control_count: u64,
    pub query_descriptor_base: u64,
    pub query_descriptor_count: u64,
    pub query_input_base: u64,
    pub query_input_count: u64,
    pub query_output_base: u64,
    pub query_output_count: u64,
    pub leaf_group_base: u64,
    pub leaf_group_count: u64,
    pub leaf_row_ref_base: u64,
    pub leaf_row_ref_count: u64,
    pub leaf_base_input_base: u64,
    pub leaf_base_input_count: u64,
    pub leaf_ext_input_base: u64,
    pub leaf_ext_input_count: u64,
    pub poseidon_candidate_base: u64,
    pub poseidon_candidate_count: u64,
    pub range_candidate_base: u64,
    pub range_candidate_count: u64,
    pub merkle_row_base: u64,
    pub merkle_row_count: u64,
}

/// Node-level collection of immutable proof blobs and their one canonical
/// device directory.
#[derive(Debug)]
pub struct CompactMerkleCandidateBatch {
    generation: u64,
    program_authority: u64,
    record_proof_count: usize,
    blobs: Vec<ProofCompactBlob>,
    directory: Vec<ProofArenaDirectory>,
    proof_rows: Vec<CompactWhirProofRowsDescriptor>,
    telemetry: Option<CompactDtoTelemetry>,
}

impl CompactMerkleCandidateBatch {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn program_authority(&self) -> u64 {
        self.program_authority
    }

    pub const fn record_proof_count(&self) -> usize {
        self.record_proof_count
    }

    pub fn blobs(&self) -> &[ProofCompactBlob] {
        &self.blobs
    }

    pub fn directory(&self) -> &[ProofArenaDirectory] {
        &self.directory
    }

    pub fn proof_rows(&self) -> &[CompactWhirProofRowsDescriptor] {
        &self.proof_rows
    }

    pub fn packed_metadata_bytes(&self) -> Option<usize> {
        let last = self.directory.last()?;
        last.metadata_slab_base
            .checked_add(last.metadata_slab_count)
            .and_then(|bytes| usize::try_from(bytes).ok())
    }

    pub const fn telemetry(&self) -> Option<&CompactDtoTelemetry> {
        self.telemetry.as_ref()
    }

    pub fn ordinary_path_occurrences(&self) -> usize {
        self.blobs.iter().map(ProofCompactBlob::ordinary_path_occurrences).sum()
    }

    pub fn node_candidate_occurrences(&self) -> usize {
        self.blobs.iter().map(ProofCompactBlob::node_candidate_occurrences).sum()
    }

    pub fn candidate_capacity(&self) -> Option<usize> {
        self.blobs
            .iter()
            .try_fold(0usize, |total, blob| total.checked_add(blob.candidate_capacity()?))
    }

    pub fn h2d_bytes(&self) -> Option<usize> {
        self.blobs
            .iter()
            .try_fold(0usize, |total, blob| total.checked_add(blob.h2d_bytes()?))
            .and_then(|bytes| {
                self.directory
                    .len()
                    .checked_mul(core::mem::size_of::<ProofArenaDirectory>())
                    .and_then(|next| bytes.checked_add(next))
            })
    }

    pub fn producer_h2d_bytes(&self) -> Option<usize> {
        self.blobs
            .iter()
            .try_fold(0usize, |total, blob| total.checked_add(blob.producer_h2d_bytes()?))
    }

    /// Install the exact proof-local Merkle counts returned by device Pass A.
    ///
    /// Only the O(proof_count) directory/row metadata changes; proof payload
    /// blobs remain immutable and are never rebased on the CPU.
    pub fn admit_device_merkle_counts(
        &mut self,
        counts: &[(usize, usize, usize)],
    ) -> Result<(), String> {
        if counts.len() != self.directory.len() {
            return Err(format!(
                "device Merkle summary has {} proofs, expected {}",
                counts.len(),
                self.directory.len()
            ));
        }
        let mut row_base = 0u64;
        for (((directory, rows), blob), &(proof_idx, leaf_count, node_count)) in
            self.directory.iter_mut().zip(&mut self.proof_rows).zip(&self.blobs).zip(counts)
        {
            if proof_idx != blob.proof_idx ||
                directory.proof_idx != proof_idx as u64 ||
                rows.proof_idx != proof_idx as u64
            {
                return Err(format!("device Merkle proof order diverges at proof {proof_idx}"));
            }
            if leaf_count != blob.leaf_block_refs.len() {
                return Err(format!(
                    "device proof {proof_idx} leaf count {leaf_count} differs from canonical {}",
                    blob.leaf_block_refs.len()
                ));
            }
            if node_count > blob.node_candidate_occurrences {
                return Err(format!(
                    "device proof {proof_idx} node count {node_count} exceeds raw {}",
                    blob.node_candidate_occurrences
                ));
            }
            let row_count = leaf_count
                .checked_add(node_count)
                .ok_or_else(|| format!("proof {proof_idx} Merkle count overflow"))?;
            let row_count = u64::try_from(row_count)
                .map_err(|_| format!("proof {proof_idx} Merkle count exceeds u64"))?;
            directory.merkle_row_base = row_base;
            directory.merkle_row_count = row_count;
            rows.merkle_row_offset = row_base;
            rows.merkle_row_count = row_count;
            row_base = row_base
                .checked_add(row_count)
                .ok_or_else(|| "Merkle row prefix overflow".to_string())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CandidateLeafChain {
    proof_idx: usize,
    unit_key: usize,
    commit_id: usize,
    level: usize,
    cur_idx: usize,
    absorb_count: usize,
    iopp_pair: bool,
    block_refs: Vec<CompactMerkleLeafBlockRef>,
    iopp_input_idx: Option<usize>,
}

fn canonical_chunk(values: [F; 8]) -> [u32; 8] {
    values.map(|value| value.as_canonical_u32())
}

#[derive(Debug, Clone, Copy)]
struct CompactLeafGroupAuthority {
    representative_query_idx: usize,
    serve_cnt: usize,
    descriptor_idx: u32,
}

enum CompactLeafPackState {
    Base { batch_id: usize, block_idx: usize, values: [u32; 8], value_count: usize },
    Ext { batch_id: usize, block_idx: usize, values: [[u32; D_EF]; 8], value_count: usize },
}

fn flush_compact_leaf_pack_state(
    state: Option<CompactLeafPackState>,
    leaf_row_refs: &mut Vec<CompactWhirLeafRowRef>,
    leaf_base_inputs: &mut Vec<CompactWhirLeafBaseInput>,
    leaf_ext_inputs: &mut Vec<CompactWhirLeafExtInput>,
) -> Result<(), WhirRecordError> {
    match state {
        Some(CompactLeafPackState::Base { batch_id, block_idx, values, value_count }) => {
            let row_idx = leaf_base_inputs.len();
            leaf_base_inputs.push(CompactWhirLeafBaseInput {
                batch_id: u32::try_from(batch_id).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf base batch id exceeds u32".to_string(),
                    }
                })?,
                block_idx: u32::try_from(block_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf base block index exceeds u32".to_string(),
                    }
                })?,
                value_count: u32::try_from(value_count).expect("leaf base chunk has at most 8"),
                output_idx: u32::try_from(row_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf base output index exceeds u32".to_string(),
                    }
                })?,
                values,
            });
            leaf_row_refs.push(CompactWhirLeafRowRef {
                kind: COMPACT_WHIR_LEAF_ROW_BASE,
                row_idx: u32::try_from(row_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf base row-reference index exceeds u32".to_string(),
                    }
                })?,
            });
        }
        Some(CompactLeafPackState::Ext { batch_id, block_idx, values, value_count }) => {
            let row_idx = leaf_ext_inputs.len();
            leaf_ext_inputs.push(CompactWhirLeafExtInput {
                batch_id: u32::try_from(batch_id).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf extension batch id exceeds u32".to_string(),
                    }
                })?,
                block_idx: u32::try_from(block_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf extension block index exceeds u32".to_string(),
                    }
                })?,
                value_count: u32::try_from(value_count)
                    .expect("leaf extension chunk has at most 8"),
                output_idx: u32::try_from(row_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf extension output index exceeds u32".to_string(),
                    }
                })?,
                values,
            });
            leaf_row_refs.push(CompactWhirLeafRowRef {
                kind: COMPACT_WHIR_LEAF_ROW_EXT,
                row_idx: u32::try_from(row_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "leaf extension row-reference index exceeds u32".to_string(),
                    }
                })?,
            });
        }
        None => {}
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_compact_leaf_group<B>(
    batch_rlc: &WhirBatchRlc,
    proof_idx: usize,
    codeword_log_height: usize,
    idx: usize,
    serve_cnt: usize,
    leaf_openings: &[B],
    log_blowup: usize,
    start_pow: EF,
    leaf_group_descriptors: &mut Vec<CompactWhirLeafGroupDescriptor>,
    leaf_row_refs: &mut Vec<CompactWhirLeafRowRef>,
    leaf_base_inputs: &mut Vec<CompactWhirLeafBaseInput>,
    leaf_ext_inputs: &mut Vec<CompactWhirLeafExtInput>,
    range_candidates: &mut Vec<CompactRangeCandidate>,
) -> Result<u32, WhirRecordError>
where
    B: AsRef<[Vec<F>]>,
{
    let group_log_height = codeword_log_height.checked_sub(log_blowup).ok_or_else(|| {
        WhirRecordError::CompactSourceSerialization {
            message: "leaf codeword height is below log blowup".to_string(),
        }
    })?;
    if !batch_rlc.groups.iter().any(|group| group.log_height == group_log_height) {
        return Err(WhirRecordError::SpecFoldSeed(WhirSpecFoldError::MissingQueryLeafSum {
            log_height: codeword_log_height,
        }));
    }

    let descriptor_idx = u32::try_from(leaf_group_descriptors.len()).map_err(|_| {
        WhirRecordError::CompactSourceSerialization {
            message: "leaf group output index exceeds u32".to_string(),
        }
    })?;
    let row_ref_offset = leaf_row_refs.len();
    let mut state = None;
    let mut previous_batch = None;
    for segment in
        batch_rlc.segments.iter().filter(|segment| segment.log_height == group_log_height)
    {
        let batch = leaf_openings.get(segment.batch_id).ok_or_else(|| {
            WhirRecordError::SpecFoldSeed(WhirSpecFoldError::QueryOpeningBatchCountMismatch {
                expected_min: segment.batch_id + 1,
                actual: leaf_openings.len(),
            })
        })?;
        let matrices = batch.as_ref();
        let row_values = matrices.get(segment.batch_pos).ok_or_else(|| {
            WhirRecordError::SpecFoldSeed(WhirSpecFoldError::QueryOpeningMatrixCountMismatch {
                batch_id: segment.batch_id,
                expected_min: segment.batch_pos + 1,
                actual: matrices.len(),
            })
        })?;
        let segment_steps =
            &batch_rlc.steps[segment.first_cursor..segment.first_cursor + segment.element_count];
        if previous_batch != Some(segment.batch_id) {
            let state_has_values = match &state {
                Some(CompactLeafPackState::Base { value_count, .. }) |
                Some(CompactLeafPackState::Ext { value_count, .. }) => *value_count != 0,
                None => false,
            };
            if state_has_values {
                flush_compact_leaf_pack_state(
                    state.take(),
                    leaf_row_refs,
                    leaf_base_inputs,
                    leaf_ext_inputs,
                )?;
            } else {
                state.take();
            }
            if let Some(previous) = previous_batch {
                let gap = segment.batch_id.checked_sub(previous + 1).ok_or_else(|| {
                    WhirRecordError::CompactSourceSerialization {
                        message: format!(
                            "proof {proof_idx} compact leaf batch order is not increasing"
                        ),
                    }
                })?;
                if compact_leaf_transition_publishes_range(previous, segment.batch_id) {
                    range_candidates.push(compact_range_candidate(gap, 8, 1)?);
                }
            }
            previous_batch = Some(segment.batch_id);
            state = Some(if segment.batch_id == WHIR_BATCH_PERMUTATION {
                CompactLeafPackState::Ext {
                    batch_id: segment.batch_id,
                    block_idx: 0,
                    values: [[0u32; D_EF]; 8],
                    value_count: 0,
                }
            } else {
                CompactLeafPackState::Base {
                    batch_id: segment.batch_id,
                    block_idx: 0,
                    values: [0u32; 8],
                    value_count: 0,
                }
            });
        }
        for step in segment_steps {
            if segment.batch_id == WHIR_BATCH_PERMUTATION {
                let start = step.value_idx.checked_mul(D_EF).ok_or_else(|| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "extension leaf value offset overflow".to_string(),
                    }
                })?;
                let end = start + D_EF;
                let value: [F; D_EF] = row_values
                    .get(start..end)
                    .ok_or_else(|| {
                        WhirRecordError::SpecFoldSeed(
                            WhirSpecFoldError::QueryOpeningWidthMismatch {
                                batch_id: segment.batch_id,
                                batch_pos: segment.batch_pos,
                                expected_width: end,
                                actual_width: row_values.len(),
                            },
                        )
                    })?
                    .try_into()
                    .expect("checked extension leaf slice has the active degree");
                let CompactLeafPackState::Ext { batch_id, block_idx, values, value_count } =
                    state.as_mut().expect("leaf pack state exists")
                else {
                    unreachable!("permutation segment uses extension leaf state")
                };
                values[*value_count] = value.map(|limb| limb.as_canonical_u32());
                *value_count += 1;
                if *value_count == 8 {
                    let next_block_idx = block_idx.saturating_add(D_EF);
                    let next_batch_id = *batch_id;
                    flush_compact_leaf_pack_state(
                        state.take(),
                        leaf_row_refs,
                        leaf_base_inputs,
                        leaf_ext_inputs,
                    )?;
                    state = Some(CompactLeafPackState::Ext {
                        batch_id: next_batch_id,
                        block_idx: next_block_idx,
                        values: [[0u32; D_EF]; 8],
                        value_count: 0,
                    });
                }
            } else {
                let value = *row_values.get(step.value_idx).ok_or_else(|| {
                    WhirRecordError::SpecFoldSeed(WhirSpecFoldError::QueryOpeningWidthMismatch {
                        batch_id: segment.batch_id,
                        batch_pos: segment.batch_pos,
                        expected_width: step.value_idx + 1,
                        actual_width: row_values.len(),
                    })
                })?;
                let CompactLeafPackState::Base { batch_id, block_idx, values, value_count } =
                    state.as_mut().expect("leaf pack state exists")
                else {
                    unreachable!("base segment uses base leaf state")
                };
                values[*value_count] = value.as_canonical_u32();
                *value_count += 1;
                if *value_count == 8 {
                    let next_block_idx = block_idx.saturating_add(1);
                    let next_batch_id = *batch_id;
                    flush_compact_leaf_pack_state(
                        state.take(),
                        leaf_row_refs,
                        leaf_base_inputs,
                        leaf_ext_inputs,
                    )?;
                    state = Some(CompactLeafPackState::Base {
                        batch_id: next_batch_id,
                        block_idx: next_block_idx,
                        values: [0u32; 8],
                        value_count: 0,
                    });
                }
            }
        }
    }
    let keep_state = match &state {
        Some(CompactLeafPackState::Base { value_count, .. }) |
        Some(CompactLeafPackState::Ext { value_count, .. }) => *value_count != 0,
        None => false,
    };
    if keep_state {
        flush_compact_leaf_pack_state(
            state.take(),
            leaf_row_refs,
            leaf_base_inputs,
            leaf_ext_inputs,
        )?;
    }
    if leaf_row_refs.len() == row_ref_offset {
        flush_compact_leaf_pack_state(
            Some(CompactLeafPackState::Base {
                batch_id: 0,
                block_idx: 0,
                values: [0u32; 8],
                value_count: 0,
            }),
            leaf_row_refs,
            leaf_base_inputs,
            leaf_ext_inputs,
        )?;
    }
    leaf_group_descriptors.push(CompactWhirLeafGroupDescriptor {
        proof_idx: u32::try_from(proof_idx).map_err(|_| {
            WhirRecordError::CompactSourceSerialization {
                message: "leaf group proof index exceeds u32".to_string(),
            }
        })?,
        idx: u32::try_from(idx).map_err(|_| WhirRecordError::CompactSourceSerialization {
            message: "leaf group index exceeds u32".to_string(),
        })?,
        log_height: u32::try_from(codeword_log_height).map_err(|_| {
            WhirRecordError::CompactSourceSerialization {
                message: "leaf group height exceeds u32".to_string(),
            }
        })?,
        serve_cnt: u32::try_from(serve_cnt)
            .map_err(|_| WhirRecordError::MultiplicityOverflow { proof_idx })?,
        row_ref_offset: u32::try_from(row_ref_offset).map_err(|_| {
            WhirRecordError::CompactSourceSerialization {
                message: "leaf row-reference offset exceeds u32".to_string(),
            }
        })?,
        row_ref_count: u32::try_from(leaf_row_refs.len() - row_ref_offset).map_err(|_| {
            WhirRecordError::CompactSourceSerialization {
                message: "leaf row-reference count exceeds u32".to_string(),
            }
        })?,
        alpha: batch_rlc.alpha.map(|value| value.as_canonical_u32()),
        start_pow: ef_limbs(&start_pow).map(|value| value.as_canonical_u32()),
    });
    Ok(descriptor_idx)
}

fn compact_leaf_transition_publishes_range(previous_batch: usize, current_batch: usize) -> bool {
    previous_batch < WHIR_BATCH_PERMUTATION && current_batch < WHIR_BATCH_PERMUTATION
}

fn compact_input_leaf_candidate_block_refs(
    unit_key: usize,
    descriptor: &CompactWhirLeafGroupDescriptor,
    leaf_row_refs: &[CompactWhirLeafRowRef],
    leaf_base_inputs: &[CompactWhirLeafBaseInput],
    leaf_ext_inputs: &[CompactWhirLeafExtInput],
) -> Result<Vec<CompactMerkleLeafBlockRef>, WhirRecordError> {
    let batch_id =
        u32::try_from(unit_key / 32).map_err(|_| WhirRecordError::CompactSourceSerialization {
            message: "input leaf batch id exceeds u32".to_string(),
        })?;
    let row_start = descriptor.row_ref_offset as usize;
    let row_end = row_start.checked_add(descriptor.row_ref_count as usize).ok_or_else(|| {
        WhirRecordError::CompactSourceSerialization {
            message: "input leaf row-reference range overflow".to_string(),
        }
    })?;
    let refs = leaf_row_refs.get(row_start..row_end).ok_or_else(|| {
        WhirRecordError::CompactSourceSerialization {
            message: "input leaf row-reference range is out of bounds".to_string(),
        }
    })?;
    let mut block_refs = Vec::new();
    for row_ref in refs {
        if row_ref.kind == COMPACT_WHIR_LEAF_ROW_BASE {
            let row = leaf_base_inputs.get(row_ref.row_idx as usize).ok_or_else(|| {
                WhirRecordError::CompactSourceSerialization {
                    message: "input base leaf row index is out of bounds".to_string(),
                }
            })?;
            if row.batch_id == batch_id && row.value_count != 0 {
                block_refs.push(CompactMerkleLeafBlockRef {
                    source_idx: row_ref.row_idx,
                    source_kind: COMPACT_MERKLE_LEAF_SOURCE_BASE,
                });
            }
        } else if row_ref.kind == COMPACT_WHIR_LEAF_ROW_EXT {
            let row = leaf_ext_inputs.get(row_ref.row_idx as usize).ok_or_else(|| {
                WhirRecordError::CompactSourceSerialization {
                    message: "input extension leaf row index is out of bounds".to_string(),
                }
            })?;
            if row.batch_id == batch_id {
                let limb_count = (row.value_count as usize).checked_mul(D_EF).ok_or_else(|| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "input extension leaf limb count overflow".to_string(),
                    }
                })?;
                for block in 0..limb_count.div_ceil(8) {
                    block_refs.push(CompactMerkleLeafBlockRef {
                        source_idx: row_ref.row_idx,
                        source_kind: COMPACT_MERKLE_LEAF_SOURCE_EXT_BASE +
                            u32::try_from(block).expect("extension leaf has at most D_EF blocks"),
                    });
                }
            }
        } else {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!("input leaf row kind {} is invalid", row_ref.kind),
            });
        }
    }
    Ok(block_refs)
}

fn compact_range_candidate(
    value: usize,
    max_bits: usize,
    count: u64,
) -> Result<CompactRangeCandidate, WhirRecordError> {
    Ok(CompactRangeCandidate {
        value: u32::try_from(value).map_err(|_| WhirRecordError::CompactSourceSerialization {
            message: "range candidate value exceeds u32".to_string(),
        })?,
        max_bits: u32::try_from(max_bits).map_err(|_| {
            WhirRecordError::CompactSourceSerialization {
                message: "range candidate bit width exceeds u32".to_string(),
            }
        })?,
        count,
    })
}

fn insert_candidate_leaf(
    leaves: &mut Vec<CandidateLeafChain>,
    leaf_indices: &mut BTreeMap<(usize, usize, usize, usize), usize>,
    mut leaf: CandidateLeafChain,
) -> Result<usize, WhirRecordError> {
    let key = (leaf.proof_idx, leaf.commit_id, leaf.level, leaf.cur_idx);
    match leaf_indices.entry(key) {
        Entry::Vacant(entry) => {
            let index = leaves.len();
            entry.insert(index);
            leaves.push(leaf);
            Ok(index)
        }
        Entry::Occupied(entry) => {
            let index = *entry.get();
            let existing = &mut leaves[index];
            if existing.unit_key != leaf.unit_key ||
                existing.commit_id != leaf.commit_id ||
                existing.level != leaf.level ||
                existing.cur_idx != leaf.cur_idx ||
                existing.iopp_pair != leaf.iopp_pair ||
                (!leaf.iopp_pair && existing.block_refs != leaf.block_refs)
            {
                return Err(WhirRecordError::MerkleUnionNodeMismatch {
                    proof_idx: leaf.proof_idx,
                    commit_id: leaf.commit_id,
                    level: leaf.level,
                    idx: leaf.cur_idx,
                });
            }
            if leaf.iopp_pair {
                existing.absorb_count = existing
                    .absorb_count
                    .checked_add(leaf.absorb_count)
                    .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx: leaf.proof_idx })?;
            }
            leaf.block_refs.clear();
            Ok(index)
        }
    }
}

const COMPACT_DTO_TELEMETRY_ENV: &str = "DT_NATIVE_RECURSION_COMPACT_DTO_TELEMETRY";

fn compact_dto_telemetry_enabled() -> bool {
    crate::env_var(COMPACT_DTO_TELEMETRY_ENV).is_ok_and(|value| {
        matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

fn compact_dto_telemetry_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

fn compact_dto_elapsed_us(started: Option<Instant>) -> u64 {
    started
        .map(|started| u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Build the typed leaf/path arena directly from borrowed compact sources.
///
/// Transcript sampling and structural checks remain CPU authority. Poseidon
/// leaf chaining, path outputs, canonical union, and provider reduction are
/// deliberately absent from this builder and are performed by the device
/// shadow.
fn compact_merkle_candidate_batch_for_proof(
    record: &RecursionRecord,
    generation: u64,
    program_authority: u64,
    proof_idx_filter: usize,
) -> Result<ProofCompactBlob, WhirRecordError> {
    let telemetry_enabled = compact_dto_telemetry_enabled();
    let total_started = compact_dto_telemetry_start(telemetry_enabled);
    let mut telemetry = CompactDtoTelemetry::default();
    let mut leaves = Vec::<CandidateLeafChain>::new();
    let mut leaf_indices = BTreeMap::<(usize, usize, usize, usize), usize>::new();
    let mut path_descriptors = Vec::new();
    let mut path_steps = Vec::new();
    let mut ordinary_path_occurrences = 0usize;
    let mut node_candidate_occurrences = 0usize;
    let mut proof_rows = Vec::new();
    let mut batch_proof_descriptors = Vec::new();
    let mut batch_segments = Vec::new();
    let mut batch_values = Vec::new();
    let mut next_batch_output_offset = 0usize;
    let mut round_proof_descriptors = Vec::new();
    let mut round_groups = Vec::new();
    let mut round_inputs = Vec::new();
    let mut next_round_output_offset = 0usize;
    let mut compact_query_controls = Vec::new();
    let mut compact_query_descriptors = Vec::new();
    let mut compact_query_round_inputs = Vec::new();
    let mut next_query_output_offset = 0usize;
    let mut leaf_group_descriptors = Vec::new();
    let mut leaf_row_refs = Vec::new();
    let mut leaf_base_inputs = Vec::new();
    let mut leaf_ext_inputs = Vec::new();
    let mut poseidon_candidates = Vec::new();
    let mut range_candidates = Vec::new();

    for proof_record in &record.proof_records {
        let Some(source) = proof_record.whir_source.as_ref() else {
            continue;
        };
        if proof_record.proof_idx != proof_idx_filter {
            continue;
        }
        let proof_setup_started = compact_dto_telemetry_start(telemetry_enabled);
        let proof_idx = proof_record.proof_idx;
        let shape = source.shape;
        reject_unsupported_modes(&source.opening_proof)?;
        let seed = WhirSpecFoldSeed::from_batch(proof_idx, shape, &proof_record.batch_constraint)
            .map_err(WhirRecordError::SpecFoldSeed)?;
        let opened_matrices =
            WhirOpenedMatrices::from_child_openings(&source.dimensions, &source.opened_values)
                .map_err(WhirRecordError::SpecFoldSeed)?;
        opened_matrices.assert_prep_first_height_is_max().map_err(WhirRecordError::SpecFoldSeed)?;
        let alpha = read_whir_alpha(&proof_record.transcript.events, shape.w0_tidx)?;
        let batch_rlc = WhirBatchRlc::from_opened_matrices(&opened_matrices, alpha);
        let static_chip_ids = proof_record
            .proof_shape
            .static_chip_ids_by_chip_idx()
            .ok_or(WhirRecordError::InvalidProofShapeChipIndex)?;
        if static_chip_ids.len() != shape.c_chips {
            return Err(WhirRecordError::InvalidProofShapeChipIndex);
        }
        let opened_eval_mults = source
            .opened_eval_publications
            .iter()
            .map(|publication| {
                (
                    (
                        publication.batch_id,
                        publication.batch_pos,
                        publication.chip_idx,
                        publication.value_idx,
                    ),
                    publication.multiplicity,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let prep_max_log_height = opened_matrices
            .matrices
            .iter()
            .filter(|matrix| matrix.batch_id == WHIR_BATCH_PREPROCESSED)
            .map(|matrix| matrix.log_height)
            .max();
        let round_replay = build_round_replay_input(
            seed.clone(),
            proof_record.proof_shape.segment_id_base(),
            &batch_rlc,
            &source.opening_proof,
            &proof_record.transcript.events,
            prep_max_log_height,
        )?;
        let round_authority = round_replay
            .compact_round_authority(&record.poseidon2_memo)
            .map_err(WhirRecordError::SpecFoldSeed)?;
        let query_controls = &round_authority.controls;
        let w_qbase = round_authority.w_qbase;
        let round_output_offset = next_round_output_offset;
        let round_group_offset = round_groups.len();
        for (rank, group) in round_replay.group_claims.iter().enumerate() {
            round_groups.push(CompactWhirRoundGroup {
                log_height: u32::try_from(group.log_height).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "round group height exceeds u32".to_string(),
                    }
                })?,
                rank: u32::try_from(rank).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "round group rank exceeds u32".to_string(),
                    }
                })?,
                claim: group.claim.map(|limb| limb.as_canonical_u32()),
            });
        }
        let round_input_offset = round_inputs.len();
        for round in 0..shape.num_rounds {
            let opening_idx = shape.num_rounds - round - 1;
            let merge_height = opening_idx;
            let merge_beta = round_replay
                .merge_betas_by_height
                .get(&merge_height)
                .copied()
                .unwrap_or([F::zero(); D_EF]);
            round_inputs.push(CompactWhirRoundInput {
                sumcheck_coeffs: round_replay.sumcheck_coeffs[round]
                    .map(|coeff| coeff.map(|limb| limb.as_canonical_u32())),
                r_fold: round_replay.r_folds[round].map(|limb| limb.as_canonical_u32()),
                opening_point: round_replay.seed.opening_point[opening_idx]
                    .map(|limb| limb.as_canonical_u32()),
                merge_beta: merge_beta.map(|limb| limb.as_canonical_u32()),
                iopp_oracle: round_replay.iopp_oracles[round].map(|limb| limb.as_canonical_u32()),
            });
        }
        round_proof_descriptors.push(CompactWhirRoundProofDescriptor {
            proof_idx: u32::try_from(proof_idx).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round proof index exceeds u32".to_string(),
                }
            })?,
            output_offset: u32::try_from(round_output_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round output offset exceeds u32".to_string(),
                }
            })?,
            output_count: u32::try_from(round_authority.output_count).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round output count exceeds u32".to_string(),
                }
            })?,
            input_offset: u32::try_from(round_input_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round input offset exceeds u32".to_string(),
                }
            })?,
            input_count: u32::try_from(round_inputs.len() - round_input_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round input count exceeds u32".to_string(),
                }
            })?,
            group_offset: u32::try_from(round_group_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round group offset exceeds u32".to_string(),
                }
            })?,
            group_count: u32::try_from(round_groups.len() - round_group_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round group count exceeds u32".to_string(),
                }
            })?,
            num_rounds: u32::try_from(shape.num_rounds).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round count exceeds u32".to_string(),
                }
            })?,
            num_queries: u32::try_from(shape.num_queries).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round query count exceeds u32".to_string(),
                }
            })?,
            log_blowup: u32::try_from(shape.log_blowup).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round blowup exceeds u32".to_string(),
                }
            })?,
            w0_tidx: u32::try_from(shape.w0_tidx).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "round transcript base exceeds u32".to_string(),
                }
            })?,
            prep_seed_round: round_replay
                .prep_seed_round
                .map(u32::try_from)
                .transpose()
                .map_err(|_| WhirRecordError::CompactSourceSerialization {
                    message: "round preparation-seed index exceeds u32".to_string(),
                })?
                .unwrap_or(u32::MAX),
            batching_pow_events: round_replay
                .batching_pow_events
                .map(|event| event.as_canonical_u32()),
            query_pow_events: round_replay.query_pow_events.map(|event| event.as_canonical_u32()),
            final_oracle: round_replay.iopp_oracles[shape.num_rounds]
                .map(|limb| limb.as_canonical_u32()),
        });
        let compact_query_control_offset = compact_query_controls.len();
        for (round, control) in query_controls.iter().enumerate() {
            compact_query_controls.push(CompactWhirQueryControl {
                round_row_idx: u32::try_from(round_output_offset + round + 2).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query control round-row index exceeds u32".to_string(),
                    }
                })?,
                is_merge: u32::from(control.is_merge),
                r_fold: control.r_fold.map(|limb| limb.as_canonical_u32()),
                merge_beta: control.merge_beta.map(|limb| limb.as_canonical_u32()),
                merge_eq: control.merge_eq.map(|limb| limb.as_canonical_u32()),
            });
        }
        let final_round_row_idx = round_output_offset + round_authority.final_row_idx;
        let query_output_offset = next_query_output_offset;
        let compact_query_input_offset = compact_query_round_inputs.len();
        let compact_query_descriptor_offset = compact_query_descriptors.len();
        let query_samples = read_whir_query_samples(
            &proof_record.transcript.events,
            &proof_record.transcript.bits_events,
            w_qbase,
            shape.num_queries,
            shape.query_bits,
        )?;
        let group_start_pows = batch_rlc.group_start_pows(shape.log_blowup);
        let mut leaf_groups = BTreeMap::<(usize, usize), CompactLeafGroupAuthority>::new();
        let query_rows_per_query = shape.num_rounds.checked_add(1).ok_or_else(|| {
            WhirRecordError::CompactSourceSerialization {
                message: "WHIR query-fold rows-per-query overflow".to_string(),
            }
        })?;
        let mut query_fold_row_count = 0usize;
        telemetry.proof_setup_round_replay_us = telemetry
            .proof_setup_round_replay_us
            .saturating_add(compact_dto_elapsed_us(proof_setup_started));

        let leaf_group_started = compact_dto_telemetry_start(telemetry_enabled);
        for (query_idx, (_, query_sample)) in query_samples.iter().copied().enumerate() {
            for &codeword_height in group_start_pows.keys() {
                let trunc_idx = query_sample >> (shape.query_bits - codeword_height);
                match leaf_groups.entry((codeword_height, trunc_idx)) {
                    Entry::Vacant(slot) => {
                        slot.insert(CompactLeafGroupAuthority {
                            representative_query_idx: query_idx,
                            serve_cnt: 1,
                            descriptor_idx: u32::MAX,
                        });
                    }
                    Entry::Occupied(mut slot) => {
                        slot.get_mut().serve_cnt = slot
                            .get()
                            .serve_cnt
                            .checked_add(1)
                            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
                    }
                }
            }
        }
        telemetry.leaf_group_key_collection_us = telemetry
            .leaf_group_key_collection_us
            .saturating_add(compact_dto_elapsed_us(leaf_group_started));

        let leaf_value_packing_started = compact_dto_telemetry_start(telemetry_enabled);
        let leaf_base_offset = leaf_base_inputs.len();
        let leaf_ext_offset = leaf_ext_inputs.len();
        let mut base_capacity = 0usize;
        let mut ext_capacity = 0usize;
        for &(codeword_height, _) in leaf_groups.keys() {
            let group_height = codeword_height - shape.log_blowup;
            for segment in
                batch_rlc.segments.iter().filter(|segment| segment.log_height == group_height)
            {
                if segment.batch_id == WHIR_BATCH_PERMUTATION {
                    ext_capacity = ext_capacity.saturating_add(segment.element_count.div_ceil(8));
                } else {
                    base_capacity = base_capacity.saturating_add(segment.element_count.div_ceil(8));
                }
            }
        }
        leaf_group_descriptors.reserve(leaf_groups.len());
        leaf_base_inputs.reserve(base_capacity);
        leaf_ext_inputs.reserve(ext_capacity);
        leaf_row_refs.reserve(base_capacity.saturating_add(ext_capacity));
        let mut instances_per_height = BTreeMap::<usize, u32>::new();
        let mut packing_leaf_openings = Vec::<&[Vec<F>]>::with_capacity(source.input_roots.len());
        for (&(codeword_height, trunc_idx), authority) in &mut leaf_groups {
            *instances_per_height.entry(codeword_height).or_default() += 1;
            packing_leaf_openings.clear();
            packing_leaf_openings.extend(
                source.opening_proof.query_openings.per_query[authority.representative_query_idx]
                    .iter()
                    .map(|opening| opening.opened_values.as_slice()),
            );
            authority.descriptor_idx = append_compact_leaf_group(
                &batch_rlc,
                proof_idx,
                codeword_height,
                trunc_idx,
                authority.serve_cnt,
                &packing_leaf_openings,
                shape.log_blowup,
                *group_start_pows.get(&codeword_height).ok_or(WhirRecordError::SpecFoldSeed(
                    WhirSpecFoldError::MissingQueryLeafSum { log_height: codeword_height },
                ))?,
                &mut leaf_group_descriptors,
                &mut leaf_row_refs,
                &mut leaf_base_inputs,
                &mut leaf_ext_inputs,
                &mut range_candidates,
            )?;
        }
        telemetry.leaf_value_packing_us = telemetry
            .leaf_value_packing_us
            .saturating_add(compact_dto_elapsed_us(leaf_value_packing_started));

        for (query_idx, (query_sample_raw, query_sample)) in
            query_samples.iter().copied().enumerate()
        {
            let input_path_started = compact_dto_telemetry_start(telemetry_enabled);
            let openings = source.opening_proof.query_openings.per_query.get(query_idx).ok_or(
                WhirRecordError::InputQueryCountMismatch {
                    expected: query_idx + 1,
                    actual: source.opening_proof.query_openings.per_query.len(),
                },
            )?;
            if openings.len() != source.input_roots.len() {
                return Err(WhirRecordError::InputMerkleOpeningCountMismatch {
                    query_idx,
                    expected: source.input_roots.len(),
                    actual: openings.len(),
                });
            }
            for batch_id in 0..source.input_roots.len() {
                let segments = batch_rlc
                    .segments
                    .iter()
                    .filter(|segment| segment.batch_id == batch_id)
                    .collect::<Vec<_>>();
                if segments.is_empty() {
                    continue;
                }
                let max_depth = segments
                    .iter()
                    .map(|segment| segment.log_height + shape.log_blowup)
                    .max()
                    .expect("non-empty compact source segments");
                let opening_proof = &openings[batch_id].opening_proof;
                if opening_proof.len() != max_depth {
                    return Err(WhirRecordError::InputMerkleProofLengthMismatch {
                        query_idx,
                        batch_id,
                        expected: max_depth,
                        actual: opening_proof.len(),
                    });
                }
                let heights = segments
                    .iter()
                    .map(|segment| segment.log_height + shape.log_blowup)
                    .collect::<BTreeSet<_>>();
                let leaf_levels = input_leaf_chain_levels(max_depth, &heights);
                let seed_idx = query_sample >> shape.query_bits.saturating_sub(max_depth);
                let commit_id = input_commit_id(batch_id);
                let mut leaf_by_height = BTreeMap::new();
                for &height in heights.iter().rev() {
                    let source_level = max_depth - height;
                    let level = *leaf_levels.get(&height).expect("candidate leaf level exists");
                    let idx = seed_idx >> source_level;
                    let unit_key = whir_unit_key(input_path_slot(batch_id), height);
                    let authority = leaf_groups
                        .get(&(height, idx))
                        .expect("candidate leaf group exists for every touched height");
                    let descriptor = leaf_group_descriptors
                        .get(authority.descriptor_idx as usize)
                        .expect("candidate leaf group descriptor exists");
                    let block_refs = compact_input_leaf_candidate_block_refs(
                        unit_key,
                        descriptor,
                        &leaf_row_refs,
                        &leaf_base_inputs,
                        &leaf_ext_inputs,
                    )?;
                    let leaf_index = insert_candidate_leaf(
                        &mut leaves,
                        &mut leaf_indices,
                        CandidateLeafChain {
                            proof_idx,
                            unit_key,
                            commit_id,
                            level,
                            cur_idx: idx,
                            absorb_count: 1,
                            iopp_pair: false,
                            block_refs,
                            iopp_input_idx: None,
                        },
                    )?;
                    leaf_by_height.insert(height, leaf_index);
                }
                let start_leaf = *leaf_by_height.get(&max_depth).ok_or(
                    WhirRecordError::InputMerkleMissingTallestLeaf {
                        query_idx,
                        batch_id,
                        log_height: max_depth,
                    },
                )?;
                let step_offset = path_steps.len();
                let mut output_count = 0usize;
                for (source_level, sibling) in opening_proof.iter().copied().enumerate() {
                    let next_height = max_depth - source_level - 1;
                    let injected_leaf =
                        leaf_by_height.get(&next_height).copied().unwrap_or(usize::MAX);
                    path_steps.push(CompactMerklePathStep {
                        sibling: canonical_chunk(sibling),
                        injected_leaf: if injected_leaf == usize::MAX {
                            u64::MAX
                        } else {
                            injected_leaf as u64
                        },
                    });
                    ordinary_path_occurrences += 1;
                    output_count += 1 + usize::from(injected_leaf != usize::MAX);
                }
                path_descriptors.push(CompactMerklePathDescriptor {
                    proof_idx: proof_idx as u64,
                    commit_id: commit_id as u64,
                    start_leaf: start_leaf as u64,
                    cur_idx: seed_idx as u64,
                    step_offset: step_offset as u64,
                    step_count: max_depth as u64,
                    output_offset: node_candidate_occurrences as u64,
                    output_count: output_count as u64,
                });
                node_candidate_occurrences += output_count;
            }
            telemetry.input_path_descriptor_us = telemetry
                .input_path_descriptor_us
                .saturating_add(compact_dto_elapsed_us(input_path_started));

            let query_fold_started = compact_dto_telemetry_start(telemetry_enabled);
            let (query_sample_high, query_sample_high_max, query_sample_high_bits) =
                crate::system_dt::spec_fold::compact_query_sample_band(
                    shape,
                    query_sample_raw,
                    query_sample,
                )
                .map_err(WhirRecordError::SpecFoldSeed)?;
            let query_input_offset = compact_query_round_inputs.len();
            for (round, sibling) in
                iopp_query_siblings(&source.opening_proof, query_idx, shape.num_rounds)?.enumerate()
            {
                let control = query_controls.get(round).ok_or_else(|| {
                    WhirRecordError::CompactSourceSerialization {
                        message: format!(
                            "proof {proof_idx} query {query_idx} is missing control {round}"
                        ),
                    }
                })?;
                let leaf_group_idx = if control.is_merge {
                    let height = shape.query_bits.checked_sub(round).ok_or_else(|| {
                        WhirRecordError::CompactSourceSerialization {
                            message: format!(
                                "proof {proof_idx} query merge cursor {round} exceeds query bits {}",
                                shape.query_bits
                            ),
                        }
                    })?;
                    let shift = shape.query_bits.checked_sub(height).ok_or_else(|| {
                        WhirRecordError::CompactSourceSerialization {
                            message: format!(
                                "proof {proof_idx} query merge height {height} exceeds query bits {}",
                                shape.query_bits
                            ),
                        }
                    })?;
                    leaf_groups
                        .get(&(height, query_sample >> shift))
                        .ok_or_else(|| WhirRecordError::CompactSourceSerialization {
                            message: format!(
                                "proof {proof_idx} compact query references missing leaf group ({height}, {})",
                                query_sample >> shift
                            ),
                        })?
                        .descriptor_idx
                } else {
                    u32::MAX
                };
                compact_query_round_inputs.push(CompactWhirQueryRoundInput {
                    sibling: sibling.map(|limb| limb.as_canonical_u32()),
                    leaf_group_idx,
                    iopp_block_offset: u32::MAX,
                });
            }
            let query_row_output_offset = query_output_offset + query_fold_row_count;
            let query_output_count = query_controls.len().checked_add(1).ok_or_else(|| {
                WhirRecordError::CompactSourceSerialization {
                    message: "query output count overflow".to_string(),
                }
            })?;
            let provider_bits = query_sample_range_provider_bits(query_sample_high_bits)?;
            range_candidates.push(compact_range_candidate(query_sample_high, provider_bits, 1)?);
            range_candidates.push(compact_range_candidate(
                query_sample_high_max - query_sample_high,
                provider_bits,
                1,
            )?);
            compact_query_descriptors.push(CompactWhirQueryDescriptor {
                proof_idx: u32::try_from(proof_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query proof index exceeds u32".to_string(),
                    }
                })?,
                query_idx: u32::try_from(query_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query index exceeds u32".to_string(),
                    }
                })?,
                query_sample_raw: query_sample_raw.as_canonical_u32(),
                query_bits: u32::try_from(shape.query_bits).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query bit count exceeds u32".to_string(),
                    }
                })?,
                control_offset: u32::try_from(compact_query_control_offset).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query control offset exceeds u32".to_string(),
                    }
                })?,
                control_count: u32::try_from(query_controls.len()).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query control count exceeds u32".to_string(),
                    }
                })?,
                input_offset: u32::try_from(query_input_offset).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query input offset exceeds u32".to_string(),
                    }
                })?,
                input_count: u32::try_from(query_controls.len()).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query input count exceeds u32".to_string(),
                    }
                })?,
                output_offset: u32::try_from(query_row_output_offset).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query output offset exceeds u32".to_string(),
                    }
                })?,
                output_count: u32::try_from(query_output_count).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query output count exceeds u32".to_string(),
                    }
                })?,
                final_round_row_idx: u32::try_from(final_round_row_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "query final-round row index exceeds u32".to_string(),
                    }
                })?,
            });
            for round_idx in 0..query_controls.len() {
                let depth = shape.query_bits - 1 - round_idx;
                let opening = &source.opening_proof.iopp_queries[query_idx].commit_phase_openings
                    [round_idx]
                    .opening_proof;
                if opening.len() != depth {
                    return Err(WhirRecordError::IoppMerkleProofLengthMismatch {
                        query_idx,
                        round_idx,
                        expected: depth,
                        actual: opening.len(),
                    });
                }
                let unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE + round_idx, depth);
                let commit_id = 100 + round_idx;
                let cur_idx = query_sample >> (round_idx + 1);
                let leaf_index = insert_candidate_leaf(
                    &mut leaves,
                    &mut leaf_indices,
                    CandidateLeafChain {
                        proof_idx,
                        unit_key,
                        commit_id,
                        level: 0,
                        cur_idx,
                        absorb_count: 1,
                        iopp_pair: true,
                        block_refs: Vec::new(),
                        iopp_input_idx: Some(query_input_offset + round_idx),
                    },
                )?;
                let step_offset = path_steps.len();
                path_steps.extend(opening.iter().copied().map(|sibling| CompactMerklePathStep {
                    sibling: canonical_chunk(sibling),
                    injected_leaf: u64::MAX,
                }));
                path_descriptors.push(CompactMerklePathDescriptor {
                    proof_idx: proof_idx as u64,
                    commit_id: commit_id as u64,
                    start_leaf: leaf_index as u64,
                    cur_idx: cur_idx as u64,
                    step_offset: step_offset as u64,
                    step_count: depth as u64,
                    output_offset: node_candidate_occurrences as u64,
                    output_count: depth as u64,
                });
                ordinary_path_occurrences += depth;
                node_candidate_occurrences += depth;
            }
            query_fold_row_count = query_fold_row_count
                .checked_add(query_output_count)
                .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
            telemetry.query_fold_iopp_pair_us = telemetry
                .query_fold_iopp_pair_us
                .saturating_add(compact_dto_elapsed_us(query_fold_started));
        }

        let batch_segment_offset = batch_segments.len();
        let batch_value_offset = batch_values.len();
        let mut next_local_value = 0usize;
        let mut batch_output_count = usize::from(!batch_rlc.steps.is_empty());
        let mut previous_group_log_height =
            batch_rlc.steps.first().map_or(0, |step| step.log_height + 1);
        for segment in &batch_rlc.segments {
            if segment.first_cursor != next_local_value {
                return Err(WhirRecordError::CompactSourceSerialization {
                    message: format!(
                        "proof {proof_idx} batch segment cursor {} is not contiguous at {next_local_value}",
                        segment.first_cursor
                    ),
                });
            }
            let end = next_local_value.checked_add(segment.element_count).ok_or(
                WhirRecordError::CompactSourceSerialization {
                    message: format!("proof {proof_idx} batch value range overflow"),
                },
            )?;
            let segment_steps = batch_rlc.steps.get(next_local_value..end).ok_or_else(|| {
                WhirRecordError::CompactSourceSerialization {
                    message: format!("proof {proof_idx} batch segment value range is invalid"),
                }
            })?;
            let segment_value_offset = batch_values.len();
            for step in segment_steps {
                let opened_eval_send_mult = opened_eval_mults
                    .get(&(step.batch_id, step.batch_pos, step.chip_idx, step.value_idx))
                    .copied()
                    .unwrap_or(0);
                batch_values.push(CompactWhirBatchValue {
                    value: step.value.map(|limb| limb.as_canonical_u32()),
                    opened_eval_send_mult,
                });
            }
            let pow_seed_cnt = instances_per_height
                .get(&(segment.log_height + shape.log_blowup))
                .copied()
                .unwrap_or(0);
            if segment.element_count != 0 && segment.log_height != previous_group_log_height {
                let group_log_height_gap = previous_group_log_height
                    .checked_sub(segment.log_height + 1)
                    .ok_or_else(|| WhirRecordError::CompactSourceSerialization {
                        message: format!(
                            "proof {proof_idx} batch group heights are not strictly descending"
                        ),
                    })?;
                range_candidates.push(compact_range_candidate(group_log_height_gap, 8, 1)?);
            }
            batch_segments.push(CompactWhirBatchSegment {
                log_height: u32::try_from(segment.log_height).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment log height exceeds u32".to_string(),
                    }
                })?,
                batch_id: u32::try_from(segment.batch_id).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment id exceeds u32".to_string(),
                    }
                })?,
                batch_pos: u32::try_from(segment.batch_pos).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment position exceeds u32".to_string(),
                    }
                })?,
                chip_idx: u32::try_from(segment.chip_idx).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment chip index exceeds u32".to_string(),
                    }
                })?,
                static_chip_id: u32::try_from(
                    *static_chip_ids
                        .get(segment.chip_idx)
                        .ok_or(WhirRecordError::InvalidProofShapeChipIndex)?,
                )
                .map_err(|_| WhirRecordError::CompactSourceSerialization {
                    message: "batch segment static chip id exceeds u32".to_string(),
                })?,
                width: u32::try_from(segment.width).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment width exceeds u32".to_string(),
                    }
                })?,
                value_offset: u32::try_from(segment_value_offset).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment value offset exceeds u32".to_string(),
                    }
                })?,
                value_count: u32::try_from(segment.element_count).map_err(|_| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "batch segment value count exceeds u32".to_string(),
                    }
                })?,
                pow_seed_cnt,
            });
            batch_output_count = batch_output_count
                .checked_add(segment.element_count.max(1))
                .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
            next_local_value = end;
            if segment.element_count != 0 {
                previous_group_log_height = segment.log_height;
            }
        }
        if next_local_value != batch_rlc.steps.len() ||
            batch_values.len() - batch_value_offset != batch_rlc.steps.len()
        {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!(
                    "proof {proof_idx} compact batch partition/count differs from canonical rows"
                ),
            });
        }
        batch_proof_descriptors.push(CompactWhirBatchProofDescriptor {
            proof_idx: u32::try_from(proof_idx).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch proof index exceeds u32".to_string(),
                }
            })?,
            alpha_tidx: u32::try_from(shape.w0_tidx).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch alpha transcript index exceeds u32".to_string(),
                }
            })?,
            segment_offset: u32::try_from(batch_segment_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch segment offset exceeds u32".to_string(),
                }
            })?,
            segment_count: u32::try_from(batch_segments.len() - batch_segment_offset).map_err(
                |_| WhirRecordError::CompactSourceSerialization {
                    message: "batch segment count exceeds u32".to_string(),
                },
            )?,
            value_offset: u32::try_from(batch_value_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch value offset exceeds u32".to_string(),
                }
            })?,
            value_count: u32::try_from(batch_values.len() - batch_value_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch value count exceeds u32".to_string(),
                }
            })?,
            output_offset: u32::try_from(next_batch_output_offset).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch output offset exceeds u32".to_string(),
                }
            })?,
            output_count: u32::try_from(batch_output_count).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "batch output count exceeds u32".to_string(),
                }
            })?,
            alpha: alpha.map(|limb| limb.as_canonical_u32()),
        });
        let batch_output_offset = next_batch_output_offset;
        next_batch_output_offset = next_batch_output_offset
            .checked_add(batch_output_count)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;

        range_candidates.push(compact_range_candidate(
            round_authority.batching_pow_sample_high,
            WHIR_PAIRED_RANGE_BITS,
            1,
        )?);
        range_candidates.push(compact_range_candidate(
            WHIR_BATCHING_POW_HIGH_MAX - round_authority.batching_pow_sample_high,
            WHIR_PAIRED_RANGE_BITS,
            1,
        )?);
        range_candidates.push(compact_range_candidate(
            round_authority.query_pow_sample_high,
            WHIR_PAIRED_RANGE_BITS,
            1,
        )?);
        range_candidates.push(compact_range_candidate(
            WHIR_QUERY_POW_HIGH_MAX - round_authority.query_pow_sample_high,
            WHIR_PAIRED_RANGE_BITS,
            1,
        )?);
        for (input, count) in
            round_authority.final_root.inputs.iter().zip(round_authority.final_root.recv_mults)
        {
            if count != 0 {
                poseidon_candidates.push(CompactPoseidonCandidate {
                    input: input.map(|value| value.as_canonical_u32()),
                    count: u64::from(count),
                });
            }
        }
        let expected_query_input_count = shape
            .num_queries
            .checked_mul(shape.num_rounds)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
        let expected_query_output_count = shape
            .num_queries
            .checked_mul(query_rows_per_query)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
        if compact_query_descriptors.len() - compact_query_descriptor_offset != shape.num_queries ||
            compact_query_round_inputs.len() - compact_query_input_offset !=
                expected_query_input_count ||
            query_fold_row_count != expected_query_output_count
        {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!(
                    "proof {proof_idx} compact query partitions differ from canonical rows"
                ),
            });
        }
        next_query_output_offset = next_query_output_offset
            .checked_add(query_fold_row_count)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
        let descriptor = CompactWhirProofRowsDescriptor {
            proof_idx: proof_idx as u64,
            whir_round_offset: round_output_offset as u64,
            whir_round_count: round_authority.output_count as u64,
            whir_batch_eval_offset: batch_output_offset as u64,
            whir_batch_eval_count: batch_output_count as u64,
            whir_query_fold_offset: query_output_offset as u64,
            whir_query_fold_count: query_fold_row_count as u64,
            whir_leaf_stream_offset: leaf_base_offset as u64,
            whir_leaf_stream_count: (leaf_base_inputs.len() - leaf_base_offset) as u64,
            whir_leaf_ext_stream_offset: leaf_ext_offset as u64,
            whir_leaf_ext_stream_count: (leaf_ext_inputs.len() - leaf_ext_offset) as u64,
            ..CompactWhirProofRowsDescriptor::default()
        };
        proof_rows.push(descriptor);
        next_round_output_offset = next_round_output_offset
            .checked_add(round_authority.output_count)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
    }

    let final_flatten_started = compact_dto_telemetry_start(telemetry_enabled);
    let mut leaf_descriptors = Vec::with_capacity(leaves.len());
    let mut leaf_block_refs = Vec::new();
    let mut iopp_leaf_block_count = 0usize;
    let mut unique_leaf_rows_by_proof = BTreeMap::<usize, usize>::new();
    let mut canonical_leaf_remap = vec![usize::MAX; leaves.len()];
    let canonical_leaf_order = leaf_indices.values().copied().collect::<Vec<_>>();
    for (canonical_idx, &source_idx) in canonical_leaf_order.iter().enumerate() {
        canonical_leaf_remap[source_idx] = canonical_idx;
    }
    for descriptor in &mut path_descriptors {
        descriptor.start_leaf = *canonical_leaf_remap
            .get(descriptor.start_leaf as usize)
            .ok_or_else(|| WhirRecordError::CompactSourceSerialization {
                message: "path start leaf is outside the canonical leaf arena".to_string(),
            })? as u64;
    }
    for step in &mut path_steps {
        if step.injected_leaf != u64::MAX {
            step.injected_leaf =
                *canonical_leaf_remap.get(step.injected_leaf as usize).ok_or_else(|| {
                    WhirRecordError::CompactSourceSerialization {
                        message: "injected leaf is outside the canonical leaf arena".to_string(),
                    }
                })? as u64;
        }
    }
    for source_idx in canonical_leaf_order {
        let leaf = leaves.get_mut(source_idx).ok_or_else(|| {
            WhirRecordError::CompactSourceSerialization {
                message: "canonical leaf order references a missing leaf".to_string(),
            }
        })?;
        let block_offset = leaf_block_refs.len();
        let block_count = if leaf.iopp_pair { 2 } else { leaf.block_refs.len() };
        *unique_leaf_rows_by_proof.entry(leaf.proof_idx).or_default() += block_count;
        if leaf.iopp_pair {
            let source_idx = u32::try_from(iopp_leaf_block_count).map_err(|_| {
                WhirRecordError::CompactSourceSerialization {
                    message: "IOPP leaf block index exceeds u32".to_string(),
                }
            })?;
            let input_idx =
                leaf.iopp_input_idx.ok_or_else(|| WhirRecordError::CompactSourceSerialization {
                    message: "IOPP leaf is missing its representative query input".to_string(),
                })?;
            compact_query_round_inputs[input_idx].iopp_block_offset = source_idx;
            for block in 0..2u32 {
                leaf_block_refs.push(CompactMerkleLeafBlockRef {
                    source_idx: source_idx + block,
                    source_kind: COMPACT_MERKLE_LEAF_SOURCE_IOPP,
                });
            }
            iopp_leaf_block_count += 2;
        } else {
            leaf_block_refs.append(&mut leaf.block_refs);
        }
        leaf_descriptors.push(CompactMerkleLeafDescriptor {
            proof_idx: leaf.proof_idx as u64,
            unit_key: leaf.unit_key as u64,
            commit_id: leaf.commit_id as u64,
            level: leaf.level as u64,
            cur_idx: leaf.cur_idx as u64,
            block_offset: block_offset as u64,
            block_count: block_count as u64,
            absorb_count: leaf.absorb_count as u64,
            iopp_pair: u64::from(leaf.iopp_pair),
        });
    }
    let mut merkle_row_offset = 0usize;
    for descriptor in &mut proof_rows {
        let proof_idx = descriptor.proof_idx as usize;
        let leaf_count = unique_leaf_rows_by_proof.get(&proof_idx).copied().unwrap_or(0);
        // Internal-node counts are admitted from the O(proof_count) GPU
        // summary after proof-local canonicalization.
        let row_count = leaf_count;
        descriptor.merkle_row_offset = merkle_row_offset as u64;
        descriptor.merkle_row_count = row_count as u64;
        merkle_row_offset = merkle_row_offset
            .checked_add(row_count)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
    }
    telemetry.proof_finalize_us = compact_dto_elapsed_us(final_flatten_started);
    telemetry.total_us = compact_dto_elapsed_us(total_started);
    telemetry.leaf_group_count = leaf_group_descriptors.len();
    telemetry.leaf_base_row_count = leaf_base_inputs.len();
    telemetry.leaf_ext_row_count = leaf_ext_inputs.len();
    telemetry.leaf_block_count = leaf_block_refs.len();
    telemetry.path_descriptor_count = path_descriptors.len();
    telemetry.path_step_count = path_steps.len();
    telemetry.node_occurrence_count = node_candidate_occurrences;
    macro_rules! retained_arena {
        ($arena:expr, $element:ty) => {
            if $arena.capacity() != 0 {
                telemetry.retained_arena_allocation_count =
                    telemetry.retained_arena_allocation_count.saturating_add(1);
                telemetry.retained_arena_bytes = telemetry.retained_arena_bytes.saturating_add(
                    $arena.capacity().saturating_mul(core::mem::size_of::<$element>()),
                );
            }
        };
    }
    if telemetry_enabled {
        retained_arena!(leaf_descriptors, CompactMerkleLeafDescriptor);
        retained_arena!(leaf_block_refs, CompactMerkleLeafBlockRef);
        retained_arena!(path_descriptors, CompactMerklePathDescriptor);
        retained_arena!(path_steps, CompactMerklePathStep);
        retained_arena!(proof_rows, CompactWhirProofRowsDescriptor);
        retained_arena!(batch_proof_descriptors, CompactWhirBatchProofDescriptor);
        retained_arena!(batch_segments, CompactWhirBatchSegment);
        retained_arena!(batch_values, CompactWhirBatchValue);
        retained_arena!(round_proof_descriptors, CompactWhirRoundProofDescriptor);
        retained_arena!(round_groups, CompactWhirRoundGroup);
        retained_arena!(round_inputs, CompactWhirRoundInput);
        retained_arena!(compact_query_controls, CompactWhirQueryControl);
        retained_arena!(compact_query_descriptors, CompactWhirQueryDescriptor);
        retained_arena!(compact_query_round_inputs, CompactWhirQueryRoundInput);
        retained_arena!(leaf_group_descriptors, CompactWhirLeafGroupDescriptor);
        retained_arena!(leaf_row_refs, CompactWhirLeafRowRef);
        retained_arena!(leaf_base_inputs, CompactWhirLeafBaseInput);
        retained_arena!(leaf_ext_inputs, CompactWhirLeafExtInput);
        retained_arena!(poseidon_candidates, CompactPoseidonCandidate);
        retained_arena!(range_candidates, CompactRangeCandidate);
    }

    Ok(ProofCompactBlob {
        proof_idx: proof_idx_filter,
        generation,
        program_authority,
        leaf_descriptors,
        leaf_block_refs,
        iopp_leaf_block_count,
        path_descriptors,
        path_steps,
        ordinary_path_occurrences,
        node_candidate_occurrences,
        proof_rows,
        batch_proof_descriptors,
        batch_segments,
        batch_values,
        round_proof_descriptors,
        round_groups,
        round_inputs,
        query_controls: compact_query_controls,
        query_descriptors: compact_query_descriptors,
        query_round_inputs: compact_query_round_inputs,
        leaf_group_descriptors,
        leaf_row_refs,
        leaf_base_inputs,
        leaf_ext_inputs,
        poseidon_candidates,
        range_candidates,
        telemetry: telemetry_enabled.then_some(telemetry),
    })
}

fn accumulate_compact_dto_telemetry(total: &mut CompactDtoTelemetry, proof: CompactDtoTelemetry) {
    macro_rules! add_u64 {
        ($field:ident) => {
            total.$field = total.$field.saturating_add(proof.$field);
        };
    }
    macro_rules! add_usize {
        ($field:ident) => {
            total.$field = total.$field.saturating_add(proof.$field);
        };
    }
    add_u64!(proof_setup_round_replay_us);
    add_u64!(leaf_group_key_collection_us);
    add_u64!(leaf_value_packing_us);
    add_u64!(input_path_descriptor_us);
    add_u64!(query_fold_iopp_pair_us);
    add_u64!(proof_finalize_us);
    add_usize!(leaf_group_count);
    add_usize!(leaf_base_row_count);
    add_usize!(leaf_ext_row_count);
    add_usize!(leaf_block_count);
    add_usize!(path_descriptor_count);
    add_usize!(path_step_count);
    add_usize!(node_occurrence_count);
    add_usize!(retained_arena_allocation_count);
    add_usize!(retained_arena_bytes);
}

fn build_proof_compact_blob(
    record: &RecursionRecord,
    generation: u64,
    program_authority: u64,
    proof_idx: usize,
) -> Result<ProofCompactBlob, WhirRecordError> {
    compact_merkle_candidate_batch_for_proof(record, generation, program_authority, proof_idx)
}

fn build_proof_compact_blobs(
    record: &RecursionRecord,
    generation: u64,
    program_authority: u64,
    proof_indices: &[usize],
) -> Result<Vec<ProofCompactBlob>, WhirRecordError> {
    proof_indices
        .par_iter()
        .copied()
        .map(|proof_idx| build_proof_compact_blob(record, generation, program_authority, proof_idx))
        .collect()
}

fn checked_prefix(
    base: u64,
    count: usize,
    proof_idx: usize,
    field: &'static str,
) -> Result<u64, WhirRecordError> {
    base.checked_add(u64::try_from(count).map_err(|_| {
        WhirRecordError::CompactSourceSerialization {
            message: format!("proof {proof_idx} {field} count exceeds u64"),
        }
    })?)
    .ok_or_else(|| WhirRecordError::CompactSourceSerialization {
        message: format!("proof {proof_idx} {field} prefix overflows u64"),
    })
}

fn assemble_proof_compact_blobs(
    record_proof_count: usize,
    generation: u64,
    program_authority: u64,
    mut blobs: Vec<ProofCompactBlob>,
    configured_source_workers: usize,
    active_source_workers: usize,
    proof_local_build_wall_us: u64,
    telemetry_enabled: bool,
    total_started: Option<Instant>,
) -> Result<CompactMerkleCandidateBatch, WhirRecordError> {
    let directory_started = compact_dto_telemetry_start(telemetry_enabled);
    let mut directory = Vec::with_capacity(blobs.len());
    let mut proof_rows = Vec::with_capacity(blobs.len());
    let mut telemetry = CompactDtoTelemetry::default();
    let mut previous_proof_idx = None;

    let mut metadata_slab_base = 0u64;
    let mut leaf_descriptor_base = 0u64;
    let mut leaf_block_ref_base = 0u64;
    let mut iopp_leaf_block_base = 0u64;
    let mut path_descriptor_base = 0u64;
    let mut path_step_base = 0u64;
    let mut node_candidate_base = 0u64;
    let mut batch_proof_base = 0u64;
    let mut batch_segment_base = 0u64;
    let mut batch_value_base = 0u64;
    let mut batch_output_base = 0u64;
    let mut round_proof_base = 0u64;
    let mut round_group_base = 0u64;
    let mut round_input_base = 0u64;
    let mut round_output_base = 0u64;
    let mut query_control_base = 0u64;
    let mut query_descriptor_base = 0u64;
    let mut query_input_base = 0u64;
    let mut query_output_base = 0u64;
    let mut leaf_group_base = 0u64;
    let mut leaf_row_ref_base = 0u64;
    let mut leaf_base_input_base = 0u64;
    let mut leaf_ext_input_base = 0u64;
    let mut poseidon_candidate_base = 0u64;
    let mut range_candidate_base = 0u64;
    let mut merkle_row_base = 0u64;

    for blob in &mut blobs {
        let proof_idx = blob.proof_idx;
        if previous_proof_idx.is_some_and(|previous| previous >= proof_idx) {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: "proof compact blobs are not in strict canonical proof order".to_string(),
            });
        }
        if blob.generation != generation || blob.program_authority != program_authority {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!("proof {proof_idx} compact blob authority mismatch"),
            });
        }
        let [local_rows] = blob.proof_rows.as_slice() else {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!(
                    "proof compact blob {proof_idx} does not contain exactly one row descriptor"
                ),
            });
        };
        if local_rows.proof_idx != proof_idx as u64 {
            return Err(WhirRecordError::CompactSourceSerialization {
                message: format!("proof compact blob {proof_idx} row authority mismatch"),
            });
        }
        if let Some(proof_telemetry) = blob.telemetry.take() {
            accumulate_compact_dto_telemetry(&mut telemetry, proof_telemetry);
        }

        let batch_output_count =
            blob.batch_proof_descriptors.iter().try_fold(0usize, |count, descriptor| {
                count
                    .checked_add(descriptor.output_count as usize)
                    .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })
            })?;
        let round_output_count = usize::try_from(local_rows.whir_round_count)
            .map_err(|_| WhirRecordError::MultiplicityOverflow { proof_idx })?;
        let query_output_count =
            blob.query_descriptors.iter().try_fold(0usize, |count, descriptor| {
                count
                    .checked_add(descriptor.output_count as usize)
                    .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })
            })?;
        let metadata_slab_count = blob.packed_metadata_bytes().ok_or_else(|| {
            WhirRecordError::CompactSourceSerialization {
                message: format!("proof {proof_idx} metadata slab size overflows usize"),
            }
        })?;

        let entry = ProofArenaDirectory {
            proof_idx: proof_idx as u64,
            metadata_slab_base,
            metadata_slab_count: metadata_slab_count as u64,
            leaf_descriptor_base,
            leaf_descriptor_count: blob.leaf_descriptors.len() as u64,
            leaf_block_ref_base,
            leaf_block_ref_count: blob.leaf_block_refs.len() as u64,
            iopp_leaf_block_base,
            iopp_leaf_block_count: blob.iopp_leaf_block_count as u64,
            path_descriptor_base,
            path_descriptor_count: blob.path_descriptors.len() as u64,
            path_step_base,
            path_step_count: blob.path_steps.len() as u64,
            node_candidate_base,
            node_candidate_count: blob.node_candidate_occurrences as u64,
            batch_proof_base,
            batch_proof_count: blob.batch_proof_descriptors.len() as u64,
            batch_segment_base,
            batch_segment_count: blob.batch_segments.len() as u64,
            batch_value_base,
            batch_value_count: blob.batch_values.len() as u64,
            batch_output_base,
            batch_output_count: batch_output_count as u64,
            round_proof_base,
            round_proof_count: blob.round_proof_descriptors.len() as u64,
            round_group_base,
            round_group_count: blob.round_groups.len() as u64,
            round_input_base,
            round_input_count: blob.round_inputs.len() as u64,
            round_output_base,
            round_output_count: round_output_count as u64,
            query_control_base,
            query_control_count: blob.query_controls.len() as u64,
            query_descriptor_base,
            query_descriptor_count: blob.query_descriptors.len() as u64,
            query_input_base,
            query_input_count: blob.query_round_inputs.len() as u64,
            query_output_base,
            query_output_count: query_output_count as u64,
            leaf_group_base,
            leaf_group_count: blob.leaf_group_descriptors.len() as u64,
            leaf_row_ref_base,
            leaf_row_ref_count: blob.leaf_row_refs.len() as u64,
            leaf_base_input_base,
            leaf_base_input_count: blob.leaf_base_inputs.len() as u64,
            leaf_ext_input_base,
            leaf_ext_input_count: blob.leaf_ext_inputs.len() as u64,
            poseidon_candidate_base,
            poseidon_candidate_count: blob.poseidon_candidates.len() as u64,
            range_candidate_base,
            range_candidate_count: blob.range_candidates.len() as u64,
            merkle_row_base,
            merkle_row_count: local_rows.merkle_row_count,
        };

        proof_rows.push(CompactWhirProofRowsDescriptor {
            proof_idx: proof_idx as u64,
            merkle_row_offset: merkle_row_base,
            merkle_row_count: local_rows.merkle_row_count,
            whir_round_offset: round_output_base,
            whir_round_count: local_rows.whir_round_count,
            whir_batch_eval_offset: batch_output_base,
            whir_batch_eval_count: local_rows.whir_batch_eval_count,
            whir_query_fold_offset: query_output_base,
            whir_query_fold_count: local_rows.whir_query_fold_count,
            whir_leaf_stream_offset: leaf_base_input_base,
            whir_leaf_stream_count: local_rows.whir_leaf_stream_count,
            whir_leaf_ext_stream_offset: leaf_ext_input_base,
            whir_leaf_ext_stream_count: local_rows.whir_leaf_ext_stream_count,
        });
        directory.push(entry);

        metadata_slab_base =
            checked_prefix(metadata_slab_base, metadata_slab_count, proof_idx, "metadata slab")?;
        leaf_descriptor_base = checked_prefix(
            leaf_descriptor_base,
            blob.leaf_descriptors.len(),
            proof_idx,
            "leaf descriptor",
        )?;
        leaf_block_ref_base = checked_prefix(
            leaf_block_ref_base,
            blob.leaf_block_refs.len(),
            proof_idx,
            "leaf block reference",
        )?;
        iopp_leaf_block_base = checked_prefix(
            iopp_leaf_block_base,
            blob.iopp_leaf_block_count,
            proof_idx,
            "IOPP leaf block",
        )?;
        path_descriptor_base = checked_prefix(
            path_descriptor_base,
            blob.path_descriptors.len(),
            proof_idx,
            "path descriptor",
        )?;
        path_step_base =
            checked_prefix(path_step_base, blob.path_steps.len(), proof_idx, "path step")?;
        node_candidate_base = checked_prefix(
            node_candidate_base,
            blob.node_candidate_occurrences,
            proof_idx,
            "node candidate",
        )?;
        batch_proof_base = checked_prefix(
            batch_proof_base,
            blob.batch_proof_descriptors.len(),
            proof_idx,
            "batch proof",
        )?;
        batch_segment_base = checked_prefix(
            batch_segment_base,
            blob.batch_segments.len(),
            proof_idx,
            "batch segment",
        )?;
        batch_value_base =
            checked_prefix(batch_value_base, blob.batch_values.len(), proof_idx, "batch value")?;
        batch_output_base =
            checked_prefix(batch_output_base, batch_output_count, proof_idx, "batch output")?;
        round_proof_base = checked_prefix(
            round_proof_base,
            blob.round_proof_descriptors.len(),
            proof_idx,
            "round proof",
        )?;
        round_group_base =
            checked_prefix(round_group_base, blob.round_groups.len(), proof_idx, "round group")?;
        round_input_base =
            checked_prefix(round_input_base, blob.round_inputs.len(), proof_idx, "round input")?;
        round_output_base =
            checked_prefix(round_output_base, round_output_count, proof_idx, "round output")?;
        query_control_base = checked_prefix(
            query_control_base,
            blob.query_controls.len(),
            proof_idx,
            "query control",
        )?;
        query_descriptor_base = checked_prefix(
            query_descriptor_base,
            blob.query_descriptors.len(),
            proof_idx,
            "query descriptor",
        )?;
        query_input_base = checked_prefix(
            query_input_base,
            blob.query_round_inputs.len(),
            proof_idx,
            "query input",
        )?;
        query_output_base =
            checked_prefix(query_output_base, query_output_count, proof_idx, "query output")?;
        leaf_group_base = checked_prefix(
            leaf_group_base,
            blob.leaf_group_descriptors.len(),
            proof_idx,
            "leaf group",
        )?;
        leaf_row_ref_base = checked_prefix(
            leaf_row_ref_base,
            blob.leaf_row_refs.len(),
            proof_idx,
            "leaf row reference",
        )?;
        leaf_base_input_base = checked_prefix(
            leaf_base_input_base,
            blob.leaf_base_inputs.len(),
            proof_idx,
            "leaf base input",
        )?;
        leaf_ext_input_base = checked_prefix(
            leaf_ext_input_base,
            blob.leaf_ext_inputs.len(),
            proof_idx,
            "leaf extension input",
        )?;
        poseidon_candidate_base = checked_prefix(
            poseidon_candidate_base,
            blob.poseidon_candidates.len(),
            proof_idx,
            "Poseidon candidate",
        )?;
        range_candidate_base = checked_prefix(
            range_candidate_base,
            blob.range_candidates.len(),
            proof_idx,
            "range candidate",
        )?;
        merkle_row_base = checked_prefix(
            merkle_row_base,
            usize::try_from(local_rows.merkle_row_count)
                .map_err(|_| WhirRecordError::MultiplicityOverflow { proof_idx })?,
            proof_idx,
            "Merkle row",
        )?;
        previous_proof_idx = Some(proof_idx);
    }

    let directory_build_us = compact_dto_elapsed_us(directory_started);
    if telemetry_enabled {
        telemetry.total_us = compact_dto_elapsed_us(total_started);
        telemetry.proof_local_build_wall_us = proof_local_build_wall_us;
        telemetry.directory_build_us = directory_build_us;
        telemetry.configured_source_workers = configured_source_workers;
        telemetry.active_source_workers = active_source_workers;
        telemetry.proof_blob_count = blobs.len();
    }
    Ok(CompactMerkleCandidateBatch {
        generation,
        program_authority,
        record_proof_count,
        blobs,
        directory,
        proof_rows,
        telemetry: telemetry_enabled.then_some(telemetry),
    })
}

/// Build immutable proof-local blobs and one small canonical device directory.
pub fn compact_merkle_candidate_batch(
    record: &RecursionRecord,
    generation: u64,
    program_authority: u64,
) -> Result<CompactMerkleCandidateBatch, WhirRecordError> {
    let telemetry_enabled = compact_dto_telemetry_enabled();
    let total_started = compact_dto_telemetry_start(telemetry_enabled);
    let mut proof_indices = record
        .proof_records
        .iter()
        .filter(|proof| proof.whir_source.is_some())
        .map(|proof| proof.proof_idx)
        .collect::<Vec<_>>();
    proof_indices.sort_unstable();
    if proof_indices.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(WhirRecordError::CompactSourceSerialization {
            message: "compact WHIR sources contain duplicate proof indices".to_string(),
        });
    }
    let configured_source_workers = current_num_threads();
    let active_source_workers = configured_source_workers.min(proof_indices.len()).max(1);
    let proof_local_build_started = compact_dto_telemetry_start(telemetry_enabled);
    let blobs = build_proof_compact_blobs(record, generation, program_authority, &proof_indices)?;
    let proof_local_build_wall_us = compact_dto_elapsed_us(proof_local_build_started);
    assemble_proof_compact_blobs(
        record.proof_records.len(),
        generation,
        program_authority,
        blobs,
        configured_source_workers,
        active_source_workers,
        proof_local_build_wall_us,
        telemetry_enabled,
        total_started,
    )
}

/// Atomically detach every compact WHIR source from its semantic record.
///
/// All target slots are checked before the first `take`, so an invalid record is not left in a
/// partially consumed state. The returned order is the canonical proof-record order.
pub fn take_whir_tracegen_sources(
    record: &mut RecursionRecord,
) -> Result<WhirTracegenSourceBatch, WhirRecordError> {
    let source_count =
        record.proof_records.iter().filter(|proof| proof.whir_source.is_some()).count();
    for proof in &record.proof_records {
        if proof.whir_source.is_none() {
            continue;
        }
        if !proof.whir.is_empty() || proof.merkle_path.row_count() != 0 {
            return Err(WhirRecordError::TracegenSourceAlreadyMaterialized {
                proof_idx: proof.proof_idx,
            });
        }
    }

    let mut sources = Vec::with_capacity(source_count);
    for proof in &mut record.proof_records {
        if let Some(source) = proof.whir_source.take() {
            sources.push(OwnedWhirTracegenSource { proof_idx: proof.proof_idx, source });
        }
    }

    Ok(WhirTracegenSourceBatch { record_proof_count: record.proof_records.len(), sources })
}

/// Consume a compact-source batch with the current CPU backend.
pub fn materialize_whir_tracegen_source_batch(
    record: &mut RecursionRecord,
    batch: WhirTracegenSourceBatch,
) -> Result<(), WhirRecordError> {
    if batch.record_proof_count != record.proof_records.len() {
        return Err(WhirRecordError::TracegenBatchProofCountMismatch {
            expected: batch.record_proof_count,
            actual: record.proof_records.len(),
        });
    }
    for source in batch.into_sources() {
        let (proof_idx, source) = source.into_parts();
        materialize_whir_tracegen_source(record, proof_idx, source)?;
    }
    Ok(())
}

/// Expand every proof-backed source exactly once under tracegen ownership.
/// The generated rows carry invalid identities through to the AIR instead of
/// turning them into host-side proof-verification errors.
pub fn materialize_whir_tracegen_sources(
    record: &mut RecursionRecord,
) -> Result<(), WhirRecordError> {
    let batch = take_whir_tracegen_sources(record)?;
    materialize_whir_tracegen_source_batch(record, batch)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirRecordError {
    MissingProofRecord {
        proof_idx: usize,
    },
    TracegenSourceAlreadyMaterialized {
        proof_idx: usize,
    },
    TracegenBatchProofCountMismatch {
        expected: usize,
        actual: usize,
    },
    CompactSourceSerialization {
        message: String,
    },
    MultiplicityOverflow {
        proof_idx: usize,
    },
    MissingRecordedBatchConstraint,
    UnsupportedRole {
        role: NativeChildRole,
    },
    UnsupportedLogBlowup {
        log_blowup: usize,
    },
    UnsupportedStacking,
    UnsupportedPathPruning,
    UnsupportedOodValues,
    UnsupportedFinalPolynomial,
    UnsupportedRoundIopp,
    UnsupportedStackingReduction,
    UnsupportedQuerySampleRange {
        required_bits: usize,
        provider_bits: usize,
    },
    PowWitnessShapeMismatch {
        phase: &'static str,
        expected: usize,
        actual: usize,
    },
    SumcheckRoundCountMismatch {
        expected: usize,
        actual: usize,
    },
    SumcheckRoundDegreeMismatch {
        round_idx: usize,
        expected: usize,
        actual: usize,
    },
    IoppOracleCountMismatch {
        expected: usize,
        actual: usize,
    },
    IoppQueryCountMismatch {
        expected: usize,
        actual: usize,
    },
    InputQueryCountMismatch {
        expected: usize,
        actual: usize,
    },
    BatchConstraintShapeMismatch {
        expected_rounds: usize,
        actual_rounds: usize,
        expected_chips: usize,
        actual_chips: usize,
        expected_public_values: usize,
        actual_public_values: usize,
    },
    RoleConfigMismatch {
        role_id: usize,
        expected_num_queries: usize,
        actual_num_queries: usize,
        expected_batching_bits: usize,
        actual_batching_bits: usize,
        expected_log_blowup: usize,
        actual_log_blowup: usize,
    },
    InvalidProofShapeChipIndex,
    MissingTranscriptEvent {
        tidx: usize,
    },
    TranscriptEventTidxMismatch {
        expected: usize,
        actual: usize,
    },
    TranscriptEventKindMismatch {
        tidx: usize,
        expected: RecursionTranscriptEventKind,
        actual: RecursionTranscriptEventKind,
    },
    TranscriptEventValueMismatch {
        tidx: usize,
    },
    MissingTranscriptBitsEvent {
        sample_tidx: usize,
    },
    DuplicateTranscriptBitsEvent {
        sample_tidx: usize,
    },
    TranscriptBitsEventBitsMismatch {
        sample_tidx: usize,
        expected: usize,
        actual: usize,
    },
    IoppQueryOpeningCountMismatch {
        query_idx: usize,
        expected: usize,
        actual: usize,
    },
    IoppQueryRowCountMismatch {
        query_idx: usize,
        expected: usize,
        actual: usize,
    },
    IoppMerkleProofLengthMismatch {
        query_idx: usize,
        round_idx: usize,
        expected: usize,
        actual: usize,
    },
    InputMerkleOpeningCountMismatch {
        query_idx: usize,
        expected: usize,
        actual: usize,
    },
    InputMerkleProofLengthMismatch {
        query_idx: usize,
        batch_id: usize,
        expected: usize,
        actual: usize,
    },
    InputMerkleMissingTallestLeaf {
        query_idx: usize,
        batch_id: usize,
        log_height: usize,
    },
    MerkleUnionNodeMismatch {
        proof_idx: usize,
        commit_id: usize,
        level: usize,
        idx: usize,
    },
    UnsupportedIoppOpenedValues {
        query_idx: usize,
        round_idx: usize,
        actual: usize,
    },
    SpecFoldSeed(WhirSpecFoldError),
}

fn query_sample_range_provider_bits(required_bits: usize) -> Result<usize, WhirRecordError> {
    if required_bits > WHIR_PAIRED_RANGE_BITS {
        return Err(WhirRecordError::UnsupportedQuerySampleRange {
            required_bits,
            provider_bits: WHIR_PAIRED_RANGE_BITS,
        });
    }
    Ok(WHIR_PAIRED_RANGE_BITS)
}

/// Regular proof-material expansion. This is trace generation, not a verifier:
/// it computes witness rows without accepting or rejecting the proof on host.
fn materialize_whir_tracegen_source(
    record: &mut RecursionRecord,
    proof_idx: usize,
    source: RecursionWhirTracegenSource,
) -> Result<(), WhirRecordError> {
    let profile = record.profile.clone();
    let phase_prefix = format!("tracegen.child[{proof_idx}].whir");
    let preflight_start = Instant::now();
    let shape = source.shape;
    reject_unsupported_modes(&source.opening_proof)?;
    let proof_record = proof_record_by_idx(record, proof_idx)?;
    let batch = &proof_record.batch_constraint;
    let seed = WhirSpecFoldSeed::from_batch(proof_idx, shape, batch)
        .map_err(WhirRecordError::SpecFoldSeed)?;
    let opened_matrices =
        WhirOpenedMatrices::from_child_openings(&source.dimensions, &source.opened_values)
            .map_err(WhirRecordError::SpecFoldSeed)?;
    opened_matrices.assert_prep_first_height_is_max().map_err(WhirRecordError::SpecFoldSeed)?;
    let alpha = read_whir_alpha(&proof_record.transcript.events, shape.w0_tidx)?;
    let batch_rlc = WhirBatchRlc::from_opened_matrices(&opened_matrices, alpha);
    let static_chip_ids = proof_record
        .proof_shape
        .static_chip_ids_by_chip_idx()
        .ok_or(WhirRecordError::InvalidProofShapeChipIndex)?;
    if static_chip_ids.len() != shape.c_chips {
        return Err(WhirRecordError::InvalidProofShapeChipIndex);
    }
    profile.add_record_split(
        format!("{phase_prefix}.preflight_seed"),
        preflight_start.elapsed().as_millis(),
    );
    let batch_eval_start = Instant::now();
    let mut batch_eval_rows = batch_rlc.batch_eval_rows(
        proof_idx,
        shape.w0_tidx,
        shape.role_id,
        shape.num_queries,
        shape.batching_bits,
        shape.log_blowup,
        &static_chip_ids,
        source.publish_opened_eval,
    );
    let opened_eval_mults = source
        .opened_eval_publications
        .iter()
        .map(|publication| {
            (
                (
                    publication.batch_id,
                    publication.batch_pos,
                    publication.chip_idx,
                    publication.value_idx,
                ),
                publication.multiplicity,
            )
        })
        .collect::<BTreeMap<_, _>>();
    for row in batch_eval_rows.iter_mut().filter(|row| row.is_value) {
        row.opened_eval_send_mult = opened_eval_mults
            .get(&(row.batch_id, row.batch_pos, row.chip_idx, row.value_idx))
            .copied()
            .unwrap_or(0);
    }
    profile.add_record_split(
        format!("{phase_prefix}.batch_eval_rows"),
        batch_eval_start.elapsed().as_millis(),
    );
    let round_rows_start = Instant::now();
    let prep_max_log_height = opened_matrices
        .matrices
        .iter()
        .filter(|matrix| matrix.batch_id == WHIR_BATCH_PREPROCESSED)
        .map(|matrix| matrix.log_height)
        .max();
    let round_rows = build_round_replay_input(
        seed.clone(),
        proof_record.proof_shape.segment_id_base(),
        &batch_rlc,
        &source.opening_proof,
        &proof_record.transcript.events,
        prep_max_log_height,
    )?
    .round_rows(&record.poseidon2_memo)
    .map_err(WhirRecordError::SpecFoldSeed)?;
    profile.add_record_split(
        format!("{phase_prefix}.round_replay_rows"),
        round_rows_start.elapsed().as_millis(),
    );
    let query_setup_start = Instant::now();
    let query_controls = WhirQueryRoundControl::from_round_rows(seed.shape, &round_rows)
        .map_err(WhirRecordError::SpecFoldSeed)?;
    let w_qbase = round_rows
        .iter()
        .find(|row| row.is_final)
        .map(|row| row.w_qbase)
        .expect("round replay always emits a final row");
    let query_samples = read_whir_query_samples(
        &proof_record.transcript.events,
        &proof_record.transcript.bits_events,
        w_qbase,
        shape.num_queries,
        shape.query_bits,
    )?;
    let input_roots = &source.input_roots;
    profile.add_record_split(
        format!("{phase_prefix}.query_setup"),
        query_setup_start.elapsed().as_millis(),
    );
    // Leaf-stream group instances dedup by (codeword height, truncated index).
    let group_start_pows = batch_rlc.group_start_pows(shape.log_blowup);
    let query_fold_row_capacity = shape
        .num_queries
        .checked_mul(shape.num_rounds + 1)
        .expect("WHIR query-fold row capacity overflow");
    let mut query_fold_rows = Vec::with_capacity(query_fold_row_capacity);
    let mut walk_memo = MerkleWalkMemo::default();
    let mut leaf_groups: BTreeMap<(usize, usize), LeafGroupInstance> = BTreeMap::new();
    let mut stream_instance_count = 0usize;
    let mut leaf_base_row_count = 0usize;
    let mut leaf_ext_row_count = 0usize;
    let mut leaf_dedup_elapsed = Duration::ZERO;
    let mut merkle_walk_elapsed = Duration::ZERO;
    let mut query_fold_elapsed = Duration::ZERO;
    for (query_idx, (query_sample_raw, query_sample)) in query_samples.into_iter().enumerate() {
        let leaf_dedup_start = Instant::now();
        let leaf_openings = query_leaf_openings(&source.opening_proof, query_idx);
        let query_leaf_sums = batch_rlc
            .query_leaf_sums(&leaf_openings, shape.log_blowup)
            .map_err(WhirRecordError::SpecFoldSeed)?;
        for (cursor, control) in query_controls.iter().enumerate() {
            let log_height = shape.query_bits - cursor;
            if control.is_merge && !query_leaf_sums.contains_key(&log_height) {
                return Err(WhirRecordError::SpecFoldSeed(WhirSpecFoldError::MissingQueryLeafSum {
                    log_height,
                }));
            }
        }
        for (&codeword_height, &start_pow) in &group_start_pows {
            let trunc_idx = query_sample >> (shape.query_bits - codeword_height);
            stream_instance_count += 1;
            match leaf_groups.entry((codeword_height, trunc_idx)) {
                Entry::Vacant(slot) => {
                    let (base, ext) = batch_rlc
                        .leaf_group_stream_rows(
                            proof_idx,
                            codeword_height,
                            trunc_idx,
                            &leaf_openings,
                            shape.log_blowup,
                            start_pow,
                        )
                        .map_err(WhirRecordError::SpecFoldSeed)?;
                    leaf_base_row_count = leaf_base_row_count
                        .checked_add(base.len())
                        .expect("WHIR base leaf row count overflow");
                    leaf_ext_row_count = leaf_ext_row_count
                        .checked_add(ext.len())
                        .expect("WHIR extension leaf row count overflow");
                    slot.insert(LeafGroupInstance { base, ext, serve_cnt: 1 });
                }
                Entry::Occupied(mut slot) => {
                    slot.get_mut().serve_cnt += 1;
                }
            }
        }
        leaf_dedup_elapsed += leaf_dedup_start.elapsed();
        let merkle_start = Instant::now();
        build_input_batch_merkle_rows_for_query(
            proof_idx,
            query_idx,
            query_sample,
            &batch_rlc,
            &leaf_groups,
            &source.opening_proof,
            input_roots,
            shape.query_bits,
            shape.log_blowup,
            &record.poseidon2_memo,
            &mut walk_memo,
        )?;
        merkle_walk_elapsed += merkle_start.elapsed();
        let query_fold_start = Instant::now();
        let siblings = iopp_query_siblings(&source.opening_proof, query_idx, shape.num_rounds)?;
        let query_rows = crate::system_dt::spec_fold::query_fold_rows_from_sibling_values(
            &seed,
            query_idx,
            w_qbase,
            query_sample_raw,
            query_sample,
            &query_controls,
            siblings,
            &query_leaf_sums,
        )
        .map_err(WhirRecordError::SpecFoldSeed)?;
        query_fold_elapsed += query_fold_start.elapsed();
        let iopp_merkle_start = Instant::now();
        build_iopp_merkle_rows_for_query(
            proof_idx,
            query_idx,
            &query_rows,
            &source.opening_proof,
            shape.num_rounds,
            &record.poseidon2_memo,
            &mut walk_memo,
        )?;
        merkle_walk_elapsed += iopp_merkle_start.elapsed();
        query_fold_rows.extend(query_rows);
    }
    profile
        .add_record_split(format!("{phase_prefix}.leaf_dedup_map"), leaf_dedup_elapsed.as_millis());
    profile.add_record_split(
        format!("{phase_prefix}.merkle_leaf_walk"),
        merkle_walk_elapsed.as_millis(),
    );
    profile.add_record_split(
        format!("{phase_prefix}.query_fold_rows"),
        query_fold_elapsed.as_millis(),
    );

    // Flatten the deduped instances; stamp serve_cnt on each instance's unit-end row
    // and count instances per height for the 1044 publication.
    let flatten_start = Instant::now();
    let stream_distinct_count = leaf_groups.len();
    let mut instances_per_height = BTreeMap::<usize, u32>::new();
    let mut leaf_stream_rows = Vec::with_capacity(leaf_base_row_count);
    let mut leaf_ext_stream_rows = Vec::with_capacity(leaf_ext_row_count);
    for ((codeword_height, _idx), mut instance) in core::mem::take(&mut leaf_groups) {
        *instances_per_height.entry(codeword_height).or_default() += 1;
        let serve = instance.serve_cnt;
        let mut stamped = false;
        for row in instance.base.iter_mut() {
            if row.is_unit_end {
                row.serve_cnt = serve;
                stamped = true;
            }
        }
        if !stamped {
            for row in instance.ext.iter_mut() {
                if row.is_unit_end {
                    row.serve_cnt = serve;
                    stamped = true;
                }
            }
        }
        debug_assert!(stamped, "leaf group instance must have a unit-end row");
        leaf_stream_rows.extend(instance.base);
        leaf_ext_stream_rows
            .extend(instance.ext.into_iter().map(RecursionWhirLeafExtStreamTraceRow::from));
    }
    profile.add_record_split(
        format!("{phase_prefix}.leaf_group_flatten"),
        flatten_start.elapsed().as_millis(),
    );
    if crate::debug_prints_enabled() {
        println!(
            "native_stage1b_stream_split proof={proof_idx} instances={stream_instance_count} distinct={stream_distinct_count}"
        );
    }

    // Alignment assert: WhirBatchEval's group-start pow must equal the leaf-side
    // schedule seed, and its 1044 publication count must equal the deduped instance
    // count at that height.
    for row in batch_eval_rows.iter_mut().filter(|row| row.is_group_start) {
        let codeword_height = row.log_height + shape.log_blowup;
        let expected = group_start_pows.get(&codeword_height).copied().ok_or(
            WhirRecordError::SpecFoldSeed(WhirSpecFoldError::MissingQueryLeafSum {
                log_height: codeword_height,
            }),
        )?;
        assert_eq!(
            limbs_to_ext(row.pow_in),
            expected,
            "stage-1b F3: BatchEval group-start pow diverges from the leaf schedule at h={codeword_height}"
        );
        row.pow_seed_cnt = instances_per_height.get(&codeword_height).copied().unwrap_or(0);
    }

    let twiddle_mults = twiddle_mults_from_query_rows(&query_fold_rows);
    let range_start = Instant::now();
    for row in &round_rows {
        if row.is_pow_batch {
            let high = row.pow_sample_high;
            record.range.record_range_count(high, WHIR_PAIRED_RANGE_BITS, 1);
            record.range.record_range_count(
                WHIR_BATCHING_POW_HIGH_MAX - high,
                WHIR_PAIRED_RANGE_BITS,
                1,
            );
        }
        if row.is_final {
            let high = row.pow_sample_high;
            record.range.record_range_count(high, WHIR_PAIRED_RANGE_BITS, 1);
            record.range.record_range_count(
                WHIR_QUERY_POW_HIGH_MAX - high,
                WHIR_PAIRED_RANGE_BITS,
                1,
            );
        }
    }
    for row in query_fold_rows.iter().filter(|row| row.is_seed) {
        let provider_bits = query_sample_range_provider_bits(row.query_sample_high_bits)?;
        let high = row.query_sample_high;
        record.range.record_range_count(high, provider_bits, 1);
        record.range.record_range_count(row.query_sample_high_max - high, provider_bits, 1);
    }
    for row in batch_eval_rows.iter().filter(|row| row.is_group_start) {
        record.range.record_range_count(row.group_log_height_gap, 8, 1);
    }
    // The gap range recv fires only on intra-instance batch transitions
    // (AIR mult = is_unit_key_start - is_unit_start); instance-start rows are exempt.
    for row in leaf_stream_rows.iter().filter(|row| row.is_unit_key_start && !row.is_unit_start) {
        record.range.record_range_count(row.unit_key_gap, 8, 1);
    }
    {
        profile.add_record_split(
            format!("{phase_prefix}.range_pool_bookkeeping"),
            range_start.elapsed().as_millis(),
        );
        let final_start = Instant::now();
        let union_start = Instant::now();
        let merkle = walk_memo.into_materialized(proof_idx);
        profile.add_record_split(
            format!("{phase_prefix}.merkle_union_materialize"),
            union_start.elapsed().as_millis(),
        );
        let provider_start = Instant::now();
        record.poseidon2.record_poseidon2_batch(merkle.poseidon2_inputs);
        for row in &round_rows {
            let mult = row.final_root_poseidon2_recv_mult;
            if row.is_final_perm && mult != 0 {
                record.poseidon2.record_poseidon2_count(row.final_root_poseidon2_input, mult);
            }
        }
        profile.add_record_split(
            format!("{phase_prefix}.poseidon2_provider_publish"),
            provider_start.elapsed().as_millis(),
        );
        let row_install_start = Instant::now();
        // Tracegen is materializing rows from an already-finalized semantic
        // source.  Do not route this through `proof_record_mut`: that helper
        // correctly clears statement public values for record-time semantic
        // mutations, but row expansion does not change the statement.
        let proof_record = record
            .proof_records
            .iter_mut()
            .find(|proof| proof.proof_idx == proof_idx)
            .ok_or(WhirRecordError::MissingProofRecord { proof_idx })?;
        proof_record.whir = RecursionWhirRecord {
            role_config_mults: [0; 3],
            twiddle_mults,
            round_rows,
            batch_eval_rows,
            query_fold_rows,
            leaf_stream_rows,
            leaf_ext_stream_rows,
        };
        proof_record.merkle_path.install_rows(merkle.rows);
        profile.add_record_split(
            format!("{phase_prefix}.row_install"),
            row_install_start.elapsed().as_millis(),
        );
        profile.add_record_split(
            format!("{phase_prefix}.final_union_gather"),
            final_start.elapsed().as_millis(),
        );
    }
    let source_drop_start = Instant::now();
    drop(source);
    profile.add_record_split(
        format!("{phase_prefix}.source_payload_drop"),
        source_drop_start.elapsed().as_millis(),
    );
    Ok(())
}

fn proof_record_by_idx(
    record: &RecursionRecord,
    proof_idx: usize,
) -> Result<&RecursionProofRecord, WhirRecordError> {
    record
        .proof_records
        .iter()
        .find(|proof| proof.proof_idx == proof_idx)
        .ok_or(WhirRecordError::MissingProofRecord { proof_idx })
}

fn preflight_child_whir<ChildSC>(
    record: &RecursionRecord,
    proof_idx: usize,
    views: &NativeChildViews<'_, ChildSC>,
) -> Result<WhirRecordShape, WhirRecordError>
where
    ChildSC: SCStarkGenericConfig<Val = F, Challenge = EF, MlChallenge = EF>,
    <ChildSC as SCStarkGenericConfig>::Mlpcs: MlPCS<BatchProof = ChildMlPcsOpeningProof>,
{
    let role_id = role_id(views.layout.role())?;
    let proof = views.proof.opening_proof();
    reject_unsupported_modes(proof)?;

    let verifier_log_height = views.proof.verifier_round_log_height().map_err(|_| {
        WhirRecordError::SumcheckRoundCountMismatch {
            expected: 0,
            actual: proof.sumcheck_transcript.uni_polys.len(),
        }
    })?;
    let round_shape = views.verifier_config.round_shape(verifier_log_height).map_err(|_| {
        WhirRecordError::SumcheckRoundCountMismatch {
            expected: verifier_log_height,
            actual: proof.sumcheck_transcript.uni_polys.len(),
        }
    })?;
    let num_rounds = round_shape.num_rounds;
    let c_chips = views.proof.chip_count();
    let num_public_values = views.layout.num_observed_public_values();
    let log_blowup = views.verifier_config.whir.log_blowup;

    let batch = &proof_record_by_idx(record, proof_idx)?.batch_constraint;
    if batch.num_rounds == 0 && batch.c_chips == 0 && batch.eq_challenges.is_empty() {
        return Err(WhirRecordError::MissingRecordedBatchConstraint);
    }
    if batch.num_rounds != num_rounds ||
        batch.c_chips != c_chips ||
        batch.num_public_values != num_public_values
    {
        return Err(WhirRecordError::BatchConstraintShapeMismatch {
            expected_rounds: num_rounds,
            actual_rounds: batch.num_rounds,
            expected_chips: c_chips,
            actual_chips: batch.c_chips,
            expected_public_values: num_public_values,
            actual_public_values: batch.num_public_values,
        });
    }

    validate_role_config(
        role_id,
        views.verifier_config.whir.num_queries,
        views.verifier_config.whir.grinding_bits_batching,
        log_blowup,
    )?;
    validate_sumcheck_shape(proof, num_rounds)?;
    validate_global_query_shape(proof, views.verifier_config.whir.num_queries)?;

    let layout = BatchTranscriptLayout::new(
        num_public_values,
        c_chips,
        num_rounds,
        views.layout.contains_global_bus(),
    );
    Ok(WhirRecordShape {
        role_id,
        num_rounds,
        c_chips,
        num_public_values,
        num_queries: views.verifier_config.whir.num_queries,
        batching_bits: views.verifier_config.whir.grinding_bits_batching,
        query_bits: num_rounds + log_blowup,
        log_blowup,
        w0_tidx: layout.e9_tidx(num_rounds),
    })
}

fn reject_unsupported_modes(proof: &ChildMlPcsOpeningProof) -> Result<(), WhirRecordError> {
    if proof.stack_log_height.is_some() {
        return Err(WhirRecordError::UnsupportedStacking);
    }
    if proof.iopp_pruned.is_some() || proof.query_openings.pruned.is_some() {
        return Err(WhirRecordError::UnsupportedPathPruning);
    }
    if !proof.ood_values.is_empty() {
        return Err(WhirRecordError::UnsupportedOodValues);
    }
    if !proof.final_poly.is_empty() {
        return Err(WhirRecordError::UnsupportedFinalPolynomial);
    }
    if proof.round_iopp.is_some() {
        return Err(WhirRecordError::UnsupportedRoundIopp);
    }
    if proof.stacking_reduction.is_some() {
        return Err(WhirRecordError::UnsupportedStackingReduction);
    }
    validate_pow_witness("batching", &proof.grinding_batching_witness)?;
    validate_pow_witness("query", &proof.grinding_query_witness)?;
    Ok(())
}

fn validate_pow_witness(phase: &'static str, witness: &[F]) -> Result<(), WhirRecordError> {
    if witness.len() != 2 {
        return Err(WhirRecordError::PowWitnessShapeMismatch {
            phase,
            expected: 2,
            actual: witness.len(),
        });
    }
    Ok(())
}

fn validate_sumcheck_shape(
    proof: &ChildMlPcsOpeningProof,
    num_rounds: usize,
) -> Result<(), WhirRecordError> {
    let unipolys = &proof.sumcheck_transcript.uni_polys;
    if unipolys.len() != num_rounds {
        return Err(WhirRecordError::SumcheckRoundCountMismatch {
            expected: num_rounds,
            actual: unipolys.len(),
        });
    }
    for (round_idx, unipoly) in unipolys.iter().enumerate() {
        if unipoly.coeffs.len() != 3 {
            return Err(WhirRecordError::SumcheckRoundDegreeMismatch {
                round_idx,
                expected: 3,
                actual: unipoly.coeffs.len(),
            });
        }
    }
    Ok(())
}

fn validate_global_query_shape(
    proof: &ChildMlPcsOpeningProof,
    num_queries: usize,
) -> Result<(), WhirRecordError> {
    if proof.iopp_queries.len() != num_queries {
        return Err(WhirRecordError::IoppQueryCountMismatch {
            expected: num_queries,
            actual: proof.iopp_queries.len(),
        });
    }
    if proof.query_openings.per_query.len() != num_queries {
        return Err(WhirRecordError::InputQueryCountMismatch {
            expected: num_queries,
            actual: proof.query_openings.per_query.len(),
        });
    }
    let expected_oracles = proof.sumcheck_transcript.uni_polys.len() + 1;
    if proof.iopp_oracles.len() != expected_oracles {
        return Err(WhirRecordError::IoppOracleCountMismatch {
            expected: expected_oracles,
            actual: proof.iopp_oracles.len(),
        });
    }
    Ok(())
}

fn validate_role_config(
    role_id: usize,
    num_queries: usize,
    batching_bits: usize,
    log_blowup: usize,
) -> Result<(), WhirRecordError> {
    let expected = whir_role_config(role_id);
    if expected.num_queries != num_queries ||
        expected.batching_bits != batching_bits ||
        expected.log_blowup != log_blowup
    {
        return Err(WhirRecordError::RoleConfigMismatch {
            role_id,
            expected_num_queries: expected.num_queries,
            actual_num_queries: num_queries,
            expected_batching_bits: expected.batching_bits,
            actual_batching_bits: batching_bits,
            expected_log_blowup: expected.log_blowup,
            actual_log_blowup: log_blowup,
        });
    }
    Ok(())
}

fn role_id(role: NativeChildRole) -> Result<usize, WhirRecordError> {
    match role {
        NativeChildRole::Core => Ok(WHIR_ROLE_CORE),
        NativeChildRole::Compress => Ok(WHIR_ROLE_COMPRESS),
        NativeChildRole::Shrink => Ok(WHIR_ROLE_SHRINK),
    }
}

fn read_whir_alpha(
    events: &[RecursionTranscriptEvent],
    w0_tidx: usize,
) -> Result<[F; D_EF], WhirRecordError> {
    let mut values = [F::zero(); D_EF];
    for (idx, value) in values.iter_mut().enumerate() {
        *value =
            expect_transcript_event(events, w0_tidx + idx, RecursionTranscriptEventKind::Sample)?;
    }
    Ok(values)
}

fn expect_transcript_event(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    kind: RecursionTranscriptEventKind,
) -> Result<F, WhirRecordError> {
    let event = events.get(tidx).ok_or(WhirRecordError::MissingTranscriptEvent { tidx })?;
    if event.tidx != tidx {
        return Err(WhirRecordError::TranscriptEventTidxMismatch {
            expected: tidx,
            actual: event.tidx,
        });
    }
    if event.kind != kind {
        return Err(WhirRecordError::TranscriptEventKindMismatch {
            tidx,
            expected: kind,
            actual: event.kind,
        });
    }
    Ok(event.value)
}

fn read_whir_query_samples(
    events: &[RecursionTranscriptEvent],
    bits_events: &[RecursionTranscriptBitsEvent],
    w_qbase: usize,
    num_queries: usize,
    query_bits: usize,
) -> Result<Vec<(F, usize)>, WhirRecordError> {
    let mut bits_by_tidx = BTreeMap::<usize, RecursionTranscriptBitsEvent>::new();
    for event in bits_events {
        if bits_by_tidx.insert(event.sample_tidx, *event).is_some() {
            return Err(WhirRecordError::DuplicateTranscriptBitsEvent {
                sample_tidx: event.sample_tidx,
            });
        }
    }

    (0..num_queries)
        .map(|query_idx| {
            let sample_tidx = w_qbase + query_idx;
            let raw =
                expect_transcript_event(events, sample_tidx, RecursionTranscriptEventKind::Sample)?;
            let bits = bits_by_tidx
                .get(&sample_tidx)
                .copied()
                .ok_or(WhirRecordError::MissingTranscriptBitsEvent { sample_tidx })?;
            if bits.bits != query_bits {
                return Err(WhirRecordError::TranscriptBitsEventBitsMismatch {
                    sample_tidx,
                    expected: query_bits,
                    actual: bits.bits,
                });
            }
            Ok((raw, bits.value))
        })
        .collect()
}

fn query_leaf_openings(proof: &ChildMlPcsOpeningProof, query_idx: usize) -> Vec<&[Vec<F>]> {
    proof.query_openings.per_query[query_idx]
        .iter()
        .map(|opening| opening.opened_values.as_slice())
        .collect()
}

fn iopp_query_siblings<'a>(
    proof: &'a ChildMlPcsOpeningProof,
    query_idx: usize,
    num_rounds: usize,
) -> Result<impl ExactSizeIterator<Item = [F; D_EF]> + 'a, WhirRecordError> {
    let query = &proof.iopp_queries[query_idx];
    let expected = num_rounds + 1;
    if query.commit_phase_openings.len() != expected {
        return Err(WhirRecordError::IoppQueryOpeningCountMismatch {
            query_idx,
            expected,
            actual: query.commit_phase_openings.len(),
        });
    }
    for (round_idx, opening) in query.commit_phase_openings.iter().take(num_rounds).enumerate() {
        if !opening.opened_values.is_empty() {
            return Err(WhirRecordError::UnsupportedIoppOpenedValues {
                query_idx,
                round_idx,
                actual: opening.opened_values.len(),
            });
        }
    }
    Ok(query
        .commit_phase_openings
        .iter()
        .take(num_rounds)
        .map(|opening| ef_limbs(&opening.sibling_value)))
}

fn build_iopp_merkle_rows_for_query(
    proof_idx: usize,
    query_idx: usize,
    query_rows: &[RecursionWhirQueryFoldRow],
    proof: &ChildMlPcsOpeningProof,
    num_rounds: usize,
    poseidon2_output: &impl RecursionPoseidon2Output,
    memo: &mut MerkleWalkMemo,
) -> Result<(), WhirRecordError> {
    let query = &proof.iopp_queries[query_idx];
    let expected_openings = num_rounds + 1;
    if query.commit_phase_openings.len() != expected_openings {
        return Err(WhirRecordError::IoppQueryOpeningCountMismatch {
            query_idx,
            expected: expected_openings,
            actual: query.commit_phase_openings.len(),
        });
    }

    let actual_rounds = query_rows.iter().filter(|row| row.is_round).count();
    if actual_rounds != num_rounds {
        return Err(WhirRecordError::IoppQueryRowCountMismatch {
            query_idx,
            expected: num_rounds,
            actual: actual_rounds,
        });
    }

    for row in query_rows.iter().filter(|row| row.is_round) {
        let opening = &query.commit_phase_openings[row.cursor];
        build_iopp_merkle_rows_for_round(
            proof_idx,
            query_idx,
            row,
            &opening.opening_proof,
            poseidon2_output,
            memo,
        )?;
    }
    Ok(())
}

fn build_iopp_merkle_rows_for_round(
    proof_idx: usize,
    query_idx: usize,
    row: &RecursionWhirQueryFoldRow,
    opening_proof: &[[F; 8]],
    poseidon2_output: &impl RecursionPoseidon2Output,
    memo: &mut MerkleWalkMemo,
) -> Result<(), WhirRecordError> {
    let round_idx = row.cursor;
    let depth = row.query_bits - 1 - round_idx;
    if opening_proof.len() != depth {
        return Err(WhirRecordError::IoppMerkleProofLengthMismatch {
            query_idx,
            round_idx,
            expected: depth,
            actual: opening_proof.len(),
        });
    }

    // Pair-leaf identity: slot = 3 + cursor, h = the round tree's path depth;
    // position = idx >> 1.
    // Note: must match the QueryFold AIR's affine unit_key expression.
    let unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE + round_idx, depth);
    let commit_id = 100 + round_idx;
    let cur_idx = row.chain_send_idx.as_canonical_u32() as usize;
    let (chunk0, mask0) = iopp_pair_leaf_block(row, 0);
    let (chunk1, mask1) = iopp_pair_leaf_block(row, 1);

    let leaf_key = (commit_id, cur_idx);
    let mut digest = if let Some(&out) = memo.iopp_leaves.get(&leaf_key) {
        // IOPP pair leaves are re-sent by every consuming QueryFold row —
        // accumulate the per-visit absorb count instead of re-building.
        memo.bump_iopp_absorb(proof_idx, commit_id, cur_idx, 0)?;
        memo.bump_iopp_absorb(proof_idx, commit_id, cur_idx, 1)?;
        out
    } else {
        let first = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            commit_id,
            0,
            cur_idx,
            1,
            false,
            true,
            false,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            chunk0,
            mask0,
            poseidon2_output,
        );
        let second = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            commit_id,
            1,
            cur_idx,
            1,
            false,
            false,
            true,
            first.output,
            chunk1,
            mask1,
            poseidon2_output,
        );
        let out = output_digest(&second);
        memo.iopp_leaves.insert(leaf_key, out);
        memo.insert_leaf(proof_idx, first)?;
        memo.insert_leaf(proof_idx, second)?;
        out
    };
    let mut idx = cur_idx;
    for (level, sibling) in opening_proof.iter().copied().enumerate() {
        let is_last = level + 1 == depth;
        let node_key = (commit_id, level, idx);
        if let Some(&(out, next_idx, node_is_last)) = memo.walk_nodes.get(&node_key) {
            memo.observe_cached_root(commit_id, level, next_idx, node_is_last);
            digest = out;
            idx = next_idx;
        } else {
            let compress = RecursionMerklePathRow::path_compress(
                proof_idx,
                commit_id,
                level,
                idx,
                digest,
                sibling,
                is_last,
                poseidon2_output,
            );
            digest = output_digest(&compress);
            memo.walk_nodes.insert(node_key, (digest, compress.next_idx, is_last));
            idx = compress.next_idx;
            memo.insert_node(proof_idx, compress)?;
        }
    }

    // The computed root is published by the Merkle rows. Its equality with
    // the transcript commitment is an AIR relation, not a host-verifier gate.
    let _ = digest;
    Ok(())
}

fn iopp_pair_leaf_block(row: &RecursionWhirQueryFoldRow, block: usize) -> ([F; 8], [bool; 8]) {
    iopp_pair_leaf_block_from_pair(row.f0, row.f1, block)
}

fn iopp_pair_leaf_block_from_pair(
    f0: [F; D_EF],
    f1: [F; D_EF],
    block: usize,
) -> ([F; 8], [bool; 8]) {
    let chunk = core::array::from_fn(|idx| match (block, idx) {
        (0, 0..=4) => f0[idx],
        (0, 5..=7) => f1[idx - 5],
        (1, 0..=1) => f1[idx + 3],
        _ => F::zero(),
    });
    let mask = core::array::from_fn(|idx| block == 0 || idx < 2);
    (chunk, mask)
}

fn output_digest(row: &RecursionMerklePathRow) -> [F; 8] {
    core::array::from_fn(|idx| row.output[idx])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MerkleNodeKey {
    commit_id: usize,
    level: usize,
    idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MerkleLeafKey {
    commit_id: usize,
    level: usize,
    idx: usize,
    block_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MerkleEdgeKey {
    commit_id: usize,
    level: usize,
    idx: usize,
    digest: [u32; 8],
}

/// Authoritative compact Merkle event builder. It owns each distinct event
/// from first insertion onward, so no candidate vector or second union exists.
#[derive(Default)]
struct MerkleWalkMemo {
    /// Pre-canonical walk key → (output digest, next index, is root).
    walk_nodes: std::collections::HashMap<(usize, usize, usize), ([F; 8], usize, bool)>,
    input_leaves: std::collections::HashMap<(usize, usize, usize), [F; 8]>,
    iopp_leaves: std::collections::HashMap<(usize, usize), [F; 8]>,
    leaves: BTreeMap<MerkleLeafKey, RecursionMerklePathRow>,
    nodes: BTreeMap<MerkleNodeKey, RecursionMerklePathRow>,
    root_counts: BTreeMap<MerkleNodeKey, usize>,
    leaf_visits: usize,
    poseidon2_inputs: Vec<[F; crate::config::POSEIDON2_WIDTH]>,
}

impl MerkleWalkMemo {
    fn insert_leaf(
        &mut self,
        proof_idx: usize,
        row: RecursionMerklePathRow,
    ) -> Result<(), WhirRecordError> {
        debug_assert!(row.is_leaf_absorb());
        self.leaf_visits += 1;
        let key = MerkleLeafKey {
            commit_id: row.commit_id,
            level: row.level,
            idx: row.cur_idx,
            block_idx: row.block_idx,
        };
        match self.leaves.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.poseidon2_inputs.push(row.input);
                entry.insert(row);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = entry.get();
                if existing.unit_key != row.unit_key ||
                    existing.input != row.input ||
                    existing.output != row.output ||
                    existing.chunk_mask != row.chunk_mask ||
                    existing.is_leaf_first != row.is_leaf_first ||
                    existing.is_leaf_last != row.is_leaf_last
                {
                    return Err(WhirRecordError::MerkleUnionNodeMismatch {
                        proof_idx,
                        commit_id: key.commit_id,
                        level: key.level,
                        idx: key.idx,
                    });
                }
            }
        }
        Ok(())
    }

    fn bump_iopp_absorb(
        &mut self,
        proof_idx: usize,
        commit_id: usize,
        idx: usize,
        block_idx: usize,
    ) -> Result<(), WhirRecordError> {
        self.leaf_visits += 1;
        let key = MerkleLeafKey { commit_id, level: 0, idx, block_idx };
        let row = self.leaves.get_mut(&key).ok_or(WhirRecordError::MerkleUnionNodeMismatch {
            proof_idx,
            commit_id,
            level: 0,
            idx,
        })?;
        row.absorb_cnt = row
            .absorb_cnt
            .checked_add(1)
            .ok_or(WhirRecordError::MultiplicityOverflow { proof_idx })?;
        Ok(())
    }

    fn insert_node(
        &mut self,
        proof_idx: usize,
        row: RecursionMerklePathRow,
    ) -> Result<(), WhirRecordError> {
        debug_assert!(row.is_node());
        let mut node = row;
        let is_inject = matches!(node.op, RecursionMerklePathOp::InjectCompress);
        node.cur_idx = node.next_idx;
        node.left_idx = if is_inject { row.cur_idx } else { node.cur_idx * 2 };
        node.left_cnt = 0;
        node.right_cnt = 0;
        node.root_cnt = 0;
        let key = MerkleNodeKey { commit_id: node.commit_id, level: node.level, idx: node.cur_idx };
        if node.is_last {
            *self.root_counts.entry(key).or_default() += 1;
        }
        match self.nodes.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.poseidon2_inputs.push(node.input);
                entry.insert(node);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = entry.get();
                if existing.op != node.op ||
                    existing.left_idx != node.left_idx ||
                    existing.is_last != node.is_last ||
                    existing.input != node.input ||
                    existing.output != node.output
                {
                    return Err(WhirRecordError::MerkleUnionNodeMismatch {
                        proof_idx,
                        commit_id: key.commit_id,
                        level: key.level,
                        idx: key.idx,
                    });
                }
            }
        }
        Ok(())
    }

    fn observe_cached_root(&mut self, commit_id: usize, level: usize, idx: usize, is_last: bool) {
        if is_last {
            *self.root_counts.entry(MerkleNodeKey { commit_id, level, idx }).or_default() += 1;
        }
    }

    fn into_materialized(mut self, proof_idx: usize) -> MaterializedMerkleEvents {
        let mut producer_counts = BTreeMap::<MerkleEdgeKey, usize>::new();
        for leaf in self.leaves.values() {
            if leaf.is_leaf_last && !leaf.is_last {
                *producer_counts
                    .entry(edge_key(leaf.commit_id, leaf.level, leaf.cur_idx, output_digest(leaf)))
                    .or_default() += 1;
            }
        }
        for node in self.nodes.values() {
            if !node.is_last {
                *producer_counts
                    .entry(edge_key(
                        node.commit_id,
                        node.level + 1,
                        node.cur_idx,
                        output_digest(node),
                    ))
                    .or_default() += 1;
            }
        }
        for (key, node) in self.nodes.iter_mut() {
            let left_digest = digest_from_input(node, 0);
            let right_digest = digest_from_input(node, DIGEST_SIZE);
            let is_inject = matches!(node.op, RecursionMerklePathOp::InjectCompress);
            let right_idx = node.left_idx + 1 - usize::from(is_inject);
            node.left_cnt = producer_counts
                .get(&edge_key(node.commit_id, node.level, node.left_idx, left_digest))
                .copied()
                .unwrap_or(0);
            node.right_cnt = producer_counts
                .get(&edge_key(node.commit_id, node.level, right_idx, right_digest))
                .copied()
                .unwrap_or(0);
            node.root_cnt =
                if node.is_last { self.root_counts.get(key).copied().unwrap_or(0) } else { 0 };
        }
        let mut rows = self.leaves.into_values().collect::<Vec<_>>();
        let leaf_distinct = rows.len();
        let nodes = self.nodes.into_values().collect::<Vec<_>>();
        if crate::debug_prints_enabled() {
            println!(
                "native_stage1b_merkle_split proof_idx={proof_idx} leaf_instances={} leaf_distinct={leaf_distinct} node_rows={} output_rows={} removed_rows={}",
                self.leaf_visits,
                nodes.len(),
                leaf_distinct + nodes.len(),
                self.leaf_visits.saturating_sub(leaf_distinct)
            );
        }
        rows.extend(nodes);
        MaterializedMerkleEvents { rows, poseidon2_inputs: self.poseidon2_inputs }
    }
}

struct MaterializedMerkleEvents {
    rows: Vec<RecursionMerklePathRow>,
    poseidon2_inputs: Vec<[F; crate::config::POSEIDON2_WIDTH]>,
}

fn digest_from_input(row: &RecursionMerklePathRow, offset: usize) -> [F; 8] {
    core::array::from_fn(|idx| row.input[offset + idx])
}

fn edge_key(commit_id: usize, level: usize, idx: usize, digest: [F; 8]) -> MerkleEdgeKey {
    MerkleEdgeKey { commit_id, level, idx, digest: digest.map(|value| value.as_canonical_u32()) }
}

#[derive(Debug, Clone)]
struct InputLeafDigest {
    digest: [F; 8],
    rows: Vec<RecursionMerklePathRow>,
}

#[allow(clippy::too_many_arguments)]
/// One deduped leaf-stream height-group instance and its 1025 publication count.
struct LeafGroupInstance {
    base: Vec<RecursionWhirLeafStreamRow>,
    ext: Vec<RecursionWhirLeafExtStreamRow>,
    serve_cnt: usize,
}

fn build_input_batch_merkle_rows_for_query(
    proof_idx: usize,
    query_idx: usize,
    query_sample: usize,
    batch_rlc: &WhirBatchRlc,
    leaf_groups: &BTreeMap<(usize, usize), LeafGroupInstance>,
    proof: &ChildMlPcsOpeningProof,
    roots: &[[F; 8]],
    query_bits: usize,
    log_blowup: usize,
    poseidon2_output: &impl RecursionPoseidon2Output,
    memo: &mut MerkleWalkMemo,
) -> Result<(), WhirRecordError> {
    let openings = proof.query_openings.per_query.get(query_idx).ok_or(
        WhirRecordError::InputQueryCountMismatch {
            expected: query_idx + 1,
            actual: proof.query_openings.per_query.len(),
        },
    )?;
    if openings.len() != roots.len() {
        return Err(WhirRecordError::InputMerkleOpeningCountMismatch {
            query_idx,
            expected: roots.len(),
            actual: openings.len(),
        });
    }

    for batch_id in 0..roots.len() {
        if !batch_rlc.segments.iter().any(|segment| segment.batch_id == batch_id) {
            continue;
        }
        build_input_batch_merkle_rows_for_batch(
            proof_idx,
            query_idx,
            query_sample,
            batch_id,
            &batch_rlc.segments,
            leaf_groups,
            &openings[batch_id].opening_proof,
            query_bits,
            log_blowup,
            poseidon2_output,
            memo,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_input_batch_merkle_rows_for_batch(
    proof_idx: usize,
    query_idx: usize,
    query_sample: usize,
    batch_id: usize,
    segments: &[crate::system_dt::WhirBatchRlcSegment],
    leaf_groups: &BTreeMap<(usize, usize), LeafGroupInstance>,
    opening_proof: &[[F; 8]],
    query_bits: usize,
    log_blowup: usize,
    poseidon2_output: &impl RecursionPoseidon2Output,
    memo: &mut MerkleWalkMemo,
) -> Result<(), WhirRecordError> {
    let max_depth = segments
        .iter()
        .filter(|segment| segment.batch_id == batch_id)
        .map(|segment| segment.log_height + log_blowup)
        .max()
        .expect("caller passes non-empty segments");
    if opening_proof.len() != max_depth {
        return Err(WhirRecordError::InputMerkleProofLengthMismatch {
            query_idx,
            batch_id,
            expected: max_depth,
            actual: opening_proof.len(),
        });
    }

    let heights = segments
        .iter()
        .filter(|segment| segment.batch_id == batch_id)
        .map(|segment| segment.log_height + log_blowup)
        .collect::<BTreeSet<_>>();
    let leaf_levels = input_leaf_chain_levels(max_depth, &heights);
    let seed_idx = query_sample >> query_bits.saturating_sub(max_depth);
    let commit_id = input_commit_id(batch_id);

    let mut leaf_digests = BTreeMap::<usize, [F; 8]>::new();
    for height in heights.iter().rev() {
        let source_level = max_depth - *height;
        let chain_level = *leaf_levels.get(height).expect("height level was computed");
        let trunc_idx = seed_idx >> source_level;
        let leaf_key = (commit_id, chain_level, trunc_idx);
        let digest = if let Some(digest) = memo.input_leaves.get(&leaf_key) {
            *digest
        } else {
            let unit_key = whir_unit_key(input_path_slot(batch_id), *height);
            let instance = leaf_groups.get(&(*height, trunc_idx)).expect(
                "leaf group instance was built for every (height, index) the query touches",
            );
            let leaf = build_input_leaf_digest(
                proof_idx,
                unit_key,
                commit_id,
                chain_level,
                trunc_idx,
                &instance.base,
                &instance.ext,
                poseidon2_output,
            );
            for row in leaf.rows {
                memo.insert_leaf(proof_idx, row)?;
            }
            memo.input_leaves.insert(leaf_key, leaf.digest);
            leaf.digest
        };
        leaf_digests.insert(*height, digest);
    }

    let mut digest =
        *leaf_digests.get(&max_depth).ok_or(WhirRecordError::InputMerkleMissingTallestLeaf {
            query_idx,
            batch_id,
            log_height: max_depth,
        })?;
    let mut idx = seed_idx;
    let mut chain_level = 0usize;
    for (source_level, sibling) in opening_proof.iter().copied().enumerate() {
        let next_height = max_depth - source_level - 1;
        let has_inject = leaf_digests.contains_key(&next_height);
        let is_last_path = source_level + 1 == max_depth && !has_inject;
        let node_key = (commit_id, chain_level, idx);
        if let Some(&(out, next_idx, is_last)) = memo.walk_nodes.get(&node_key) {
            memo.observe_cached_root(commit_id, chain_level, next_idx, is_last);
            digest = out;
            idx = next_idx;
        } else {
            let path = RecursionMerklePathRow::path_compress(
                proof_idx,
                commit_id,
                chain_level,
                idx,
                digest,
                sibling,
                is_last_path,
                poseidon2_output,
            );
            digest = output_digest(&path);
            memo.walk_nodes.insert(node_key, (digest, path.next_idx, is_last_path));
            idx = path.next_idx;
            memo.insert_node(proof_idx, path)?;
        }
        chain_level += 1;

        if let Some(injected) = leaf_digests.get(&next_height).copied() {
            let is_last_inject = source_level + 1 == max_depth;
            let node_key = (commit_id, chain_level, idx);
            if let Some(&(out, next_idx, is_last)) = memo.walk_nodes.get(&node_key) {
                memo.observe_cached_root(commit_id, chain_level, next_idx, is_last);
                digest = out;
                idx = next_idx;
            } else {
                let inject = RecursionMerklePathRow::inject_compress(
                    proof_idx,
                    commit_id,
                    chain_level,
                    idx,
                    digest,
                    injected,
                    is_last_inject,
                    poseidon2_output,
                );
                digest = output_digest(&inject);
                memo.walk_nodes.insert(node_key, (digest, inject.next_idx, is_last_inject));
                idx = inject.next_idx;
                memo.insert_node(proof_idx, inject)?;
            }
            chain_level += 1;
        }
    }

    // Keep the calculated root in the row chain; proof-shape/WHIR buses bind
    // it to the claimed root inside the recursive proof.
    let _ = digest;
    Ok(())
}

fn input_leaf_chain_levels(max_depth: usize, heights: &BTreeSet<usize>) -> BTreeMap<usize, usize> {
    let mut levels = BTreeMap::new();
    let mut chain_level = 0usize;
    if heights.contains(&max_depth) {
        levels.insert(max_depth, 0);
    }
    for source_level in 0..max_depth {
        chain_level += 1;
        let next_height = max_depth - source_level - 1;
        if heights.contains(&next_height) {
            levels.insert(next_height, chain_level);
            chain_level += 1;
        }
    }
    levels
}

#[allow(clippy::too_many_arguments)]
fn build_input_leaf_digest(
    proof_idx: usize,
    unit_key: usize,
    commit_id: usize,
    digest_level: usize,
    cur_idx: usize,
    base_rows: &[RecursionWhirLeafStreamRow],
    ext_rows: &[RecursionWhirLeafExtStreamRow],
    poseidon2_output: &impl RecursionPoseidon2Output,
) -> InputLeafDigest {
    let matching_base_rows = base_rows.iter().filter(|row| {
        row.proof_idx == proof_idx &&
            row.unit_key == unit_key &&
            row.idx == cur_idx &&
            row.chunk_mask[0]
    });
    let matching_ext_rows = ext_rows
        .iter()
        .filter(|row| row.proof_idx == proof_idx && row.unit_key == unit_key && row.idx == cur_idx);
    let block_count = matching_base_rows.clone().count() +
        matching_ext_rows
            .clone()
            .map(|row| row.chunk_masks.iter().filter(|mask| mask[0]).count())
            .sum::<usize>();

    let mut rows = Vec::with_capacity(block_count);
    let mut prev_state = [F::zero(); crate::config::POSEIDON2_WIDTH];
    let mut emitted = 0usize;
    let mut prior_block_idx = None;
    let mut push_block = |block_idx, chunk, mask| {
        debug_assert!(prior_block_idx.is_none_or(|prior| prior < block_idx));
        prior_block_idx = Some(block_idx);
        let is_first = emitted == 0;
        emitted += 1;
        let is_last = emitted == block_count;
        let row = RecursionMerklePathRow::leaf_absorb_at_level(
            proof_idx,
            unit_key,
            commit_id,
            digest_level,
            block_idx,
            cur_idx,
            1,
            false,
            is_first,
            is_last,
            prev_state,
            chunk,
            mask,
            poseidon2_output,
        );
        prev_state = row.output;
        rows.push(row);
    };
    for row in matching_base_rows {
        push_block(row.block_idx, row.values, row.chunk_mask);
    }
    for row in matching_ext_rows {
        for (block, mask) in row.chunk_masks.iter().copied().enumerate() {
            if mask[0] {
                push_block(row.block_idx + block, row.value_blocks[block], mask);
            }
        }
    }
    debug_assert_eq!(emitted, block_count);
    let digest = rows.last().map(output_digest).unwrap_or([F::zero(); 8]);
    InputLeafDigest { digest, rows }
}

fn input_path_slot(batch_id: usize) -> usize {
    match batch_id {
        WHIR_BATCH_PREPROCESSED => WHIR_INPUT_PREPROCESSED_PATH_SLOT,
        WHIR_BATCH_MAIN => WHIR_INPUT_MAIN_PATH_SLOT,
        WHIR_BATCH_PERMUTATION => WHIR_INPUT_PERMUTATION_PATH_SLOT,
        _ => panic!("unsupported WHIR input batch id {batch_id}"),
    }
}

fn input_commit_id(batch_id: usize) -> usize {
    match batch_id {
        WHIR_BATCH_PREPROCESSED => PROOF_SHAPE_COMMIT_VK,
        WHIR_BATCH_MAIN => PROOF_SHAPE_COMMIT_MAIN,
        WHIR_BATCH_PERMUTATION => PROOF_SHAPE_COMMIT_PERMUTATION,
        _ => panic!("unsupported WHIR input batch id {batch_id}"),
    }
}

fn twiddle_mults_from_query_rows(query_rows: &[RecursionWhirQueryFoldRow]) -> Vec<[u32; 3]> {
    let mut mults = vec![[0u32; WHIR_TWIDDLE_TABLES]; WHIR_TWIDDLE_ROWS];
    for row in query_rows.iter().filter(|row| row.is_seed) {
        for (table_id, byte) in row.twiddle_bytes.iter().copied().enumerate() {
            let byte = usize::from(byte);
            mults[byte][table_id] =
                mults[byte][table_id].checked_add(1).expect("WHIR twiddle mult overflow");
        }
    }
    mults
}

/// Expand the proof-backed round material under tracegen ownership.
fn build_round_replay_input(
    seed: WhirSpecFoldSeed,
    summary_id_base: usize,
    batch_rlc: &WhirBatchRlc,
    proof: &ChildMlPcsOpeningProof,
    events: &[RecursionTranscriptEvent],
    prep_max_log_height: Option<usize>,
) -> Result<WhirRoundReplayInput, WhirRecordError> {
    let shape = seed.shape;
    let mut tidx = shape.w0_tidx + D_EF;
    let batching_pow_events = [
        expect_transcript_event(events, tidx, RecursionTranscriptEventKind::Observe)?,
        expect_transcript_event(events, tidx + 1, RecursionTranscriptEventKind::Observe)?,
        expect_transcript_event(events, tidx + 2, RecursionTranscriptEventKind::Sample)?,
    ];
    tidx += 3;

    let iopp_oracles =
        proof.iopp_oracles.iter().map(|commitment| *commitment.as_ref()).collect::<Vec<_>>();
    expect_digest(events, tidx, iopp_oracles[0])?;
    tidx += DIGEST_SIZE;

    let group_by_height =
        batch_rlc.groups.iter().map(|group| (group.log_height, ())).collect::<BTreeMap<_, _>>();
    let mut sumcheck_coeffs = Vec::with_capacity(shape.num_rounds);
    let mut r_folds = Vec::with_capacity(shape.num_rounds);
    let mut merge_betas_by_height = BTreeMap::new();
    for round in 0..shape.num_rounds {
        if round > 0 {
            expect_digest(events, tidx, iopp_oracles[round])?;
            tidx += DIGEST_SIZE;
        }
        let mut coeffs = [[F::zero(); D_EF]; 3];
        for (coeff_idx, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = ef_limbs(&proof.sumcheck_transcript.uni_polys[round].coeffs[coeff_idx]);
            expect_ext_events(events, tidx, RecursionTranscriptEventKind::Observe, *coeff)?;
            tidx += D_EF;
        }
        sumcheck_coeffs.push(coeffs);
        r_folds.push(read_ext_events(events, tidx, RecursionTranscriptEventKind::Sample)?);
        tidx += D_EF;
        let merge_height = shape.num_rounds - round - 1;
        if group_by_height.contains_key(&merge_height) {
            let beta = read_ext_events(events, tidx, RecursionTranscriptEventKind::Sample)?;
            merge_betas_by_height.insert(merge_height, beta);
            tidx += D_EF;
        }
    }

    expect_digest(events, tidx, iopp_oracles[shape.num_rounds])?;
    let query_pow_events = [
        expect_transcript_event(events, tidx + DIGEST_SIZE, RecursionTranscriptEventKind::Observe)?,
        expect_transcript_event(
            events,
            tidx + DIGEST_SIZE + 1,
            RecursionTranscriptEventKind::Observe,
        )?,
        expect_transcript_event(
            events,
            tidx + DIGEST_SIZE + 2,
            RecursionTranscriptEventKind::Sample,
        )?,
    ];
    let prep_seed_round = prep_max_log_height
        .and_then(|height| shape.num_rounds.checked_sub(height))
        .filter(|&round| round < shape.num_rounds);
    Ok(WhirRoundReplayInput {
        seed,
        summary_id_base,
        group_claims: batch_rlc.groups.clone(),
        sumcheck_coeffs,
        r_folds,
        merge_betas_by_height,
        iopp_oracles,
        batching_pow_events,
        query_pow_events,
        prep_seed_round,
    })
}

fn read_ext_events(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    kind: RecursionTranscriptEventKind,
) -> Result<[F; D_EF], WhirRecordError> {
    let mut values = [F::zero(); D_EF];
    for (idx, value) in values.iter_mut().enumerate() {
        *value = expect_transcript_event(events, tidx + idx, kind)?;
    }
    Ok(values)
}

fn expect_ext_events(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    kind: RecursionTranscriptEventKind,
    expected: [F; D_EF],
) -> Result<(), WhirRecordError> {
    for (idx, expected_value) in expected.iter().copied().enumerate() {
        expect_transcript_value(events, tidx + idx, kind, expected_value)?;
    }
    Ok(())
}

fn expect_digest(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    expected: [F; 8],
) -> Result<(), WhirRecordError> {
    for (idx, expected_value) in expected.iter().copied().enumerate() {
        expect_transcript_value(
            events,
            tidx + idx,
            RecursionTranscriptEventKind::Observe,
            expected_value,
        )?;
    }
    Ok(())
}

fn expect_transcript_value(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    kind: RecursionTranscriptEventKind,
    expected: F,
) -> Result<(), WhirRecordError> {
    let actual = expect_transcript_event(events, tidx, kind)?;
    if actual != expected {
        return Err(WhirRecordError::TranscriptEventValueMismatch { tidx });
    }
    Ok(())
}

fn ef_limbs(value: &EF) -> [F; D_EF] {
    value.as_base_slice().try_into().expect("active WHIR extension degree must equal D_EF")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhirRecordShape {
    role_id: usize,
    num_rounds: usize,
    c_chips: usize,
    num_public_values: usize,
    num_queries: usize,
    batching_bits: usize,
    query_bits: usize,
    log_blowup: usize,
    w0_tidx: usize,
}

impl From<WhirRecordShape> for WhirSpecFoldShape {
    fn from(shape: WhirRecordShape) -> Self {
        Self {
            role_id: shape.role_id,
            num_rounds: shape.num_rounds,
            c_chips: shape.c_chips,
            num_public_values: shape.num_public_values,
            num_queries: shape.num_queries,
            batching_bits: shape.batching_bits,
            query_bits: shape.query_bits,
            log_blowup: shape.log_blowup,
            w0_tidx: shape.w0_tidx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_leaf_range_publication_is_base_only() {
        assert!(compact_leaf_transition_publishes_range(WHIR_BATCH_PREPROCESSED, WHIR_BATCH_MAIN));
        assert!(!compact_leaf_transition_publishes_range(WHIR_BATCH_MAIN, WHIR_BATCH_PERMUTATION));
    }

    #[test]
    fn query_sample_ranges_use_one_shape_independent_provider() {
        assert_eq!(query_sample_range_provider_bits(9), Ok(WHIR_PAIRED_RANGE_BITS));
        assert_eq!(query_sample_range_provider_bits(14), Ok(WHIR_PAIRED_RANGE_BITS));
        assert_eq!(
            query_sample_range_provider_bits(WHIR_PAIRED_RANGE_BITS + 1),
            Err(WhirRecordError::UnsupportedQuerySampleRange {
                required_bits: WHIR_PAIRED_RANGE_BITS + 1,
                provider_bits: WHIR_PAIRED_RANGE_BITS,
            })
        );
    }

    #[test]
    fn query_sample_reader_pairs_raw_transcript_with_recorded_low_bits() {
        let mut events = dense_sample_events(12);
        events[10].value = F::from_canonical_usize((7 << 4) + 3);
        events[11].value = F::from_canonical_usize((2 << 4) + 9);
        let bits_events = vec![
            RecursionTranscriptBitsEvent { sample_tidx: 10, bits: 4, value: 3 },
            RecursionTranscriptBitsEvent { sample_tidx: 11, bits: 4, value: 9 },
        ];

        let samples =
            read_whir_query_samples(&events, &bits_events, 10, 2, 4).expect("samples are valid");
        assert_eq!(
            samples,
            vec![(F::from_canonical_usize(115), 3), (F::from_canonical_usize(41), 9)]
        );
    }

    #[test]
    fn query_sample_reader_rejects_missing_or_wrong_bits_events() {
        let mut events = dense_sample_events(11);
        events[10].value = F::from_canonical_usize(3);

        assert_eq!(
            read_whir_query_samples(&events, &[], 10, 1, 4),
            Err(WhirRecordError::MissingTranscriptBitsEvent { sample_tidx: 10 })
        );
        assert_eq!(
            read_whir_query_samples(
                &events,
                &[RecursionTranscriptBitsEvent { sample_tidx: 10, bits: 5, value: 3 }],
                10,
                1,
                4,
            ),
            Err(WhirRecordError::TranscriptBitsEventBitsMismatch {
                sample_tidx: 10,
                expected: 4,
                actual: 5,
            })
        );
        assert_eq!(
            read_whir_query_samples(
                &events,
                &[
                    RecursionTranscriptBitsEvent { sample_tidx: 10, bits: 4, value: 3 },
                    RecursionTranscriptBitsEvent { sample_tidx: 10, bits: 4, value: 3 },
                ],
                10,
                1,
                4,
            ),
            Err(WhirRecordError::DuplicateTranscriptBitsEvent { sample_tidx: 10 })
        );
    }

    #[test]
    fn duplicate_merkle_coordinate_trusted_op_conflict_is_control_fatal() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 3;
        let commit_id = 17;
        let level = 2;
        let path = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            level,
            4,
            digest(10),
            digest(20),
            false,
            &poseidon2_memo,
        );
        // Inject keeps its input index while a regular path shifts once. These two rows therefore
        // normalize to the same output coordinate (commit=17, level=2, idx=2) with conflicting
        // trusted builder-derived operations/routing.
        let inject = RecursionMerklePathRow::inject_compress(
            proof_idx,
            commit_id,
            level,
            2,
            digest(10),
            digest(20),
            false,
            &poseidon2_memo,
        );
        let mut memo = MerkleWalkMemo::default();
        memo.insert_node(proof_idx, path).expect("first coordinate is admitted");
        assert_eq!(
            memo.insert_node(proof_idx, inject),
            Err(WhirRecordError::MerkleUnionNodeMismatch { proof_idx, commit_id, level, idx: 2 })
        );
    }

    #[test]
    fn duplicate_merkle_coordinate_sibling_conflict_is_representation_fatal_baseline() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 5;
        let commit_id = 23;
        let level = 1;
        let first = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            level,
            6,
            digest(30),
            digest(40),
            false,
            &poseidon2_memo,
        );
        let conflicting_sibling = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            level,
            6,
            digest(30),
            digest(41),
            false,
            &poseidon2_memo,
        );
        let mut memo = MerkleWalkMemo::default();
        memo.insert_node(proof_idx, first).expect("first coordinate is admitted");
        assert_eq!(
            memo.insert_node(proof_idx, conflicting_sibling),
            Err(WhirRecordError::MerkleUnionNodeMismatch { proof_idx, commit_id, level, idx: 3 })
        );
    }

    #[test]
    fn duplicate_merkle_leaf_preimage_conflict_is_representation_fatal_baseline() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 7;
        let commit_id = 29;
        let level = 3;
        let mut chunk = digest(50);
        let first = RecursionMerklePathRow::leaf_absorb_at_level(
            proof_idx,
            99,
            commit_id,
            level,
            0,
            11,
            1,
            false,
            true,
            true,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            chunk,
            [true; 8],
            &poseidon2_memo,
        );
        chunk[0] += F::one();
        let conflicting_preimage = RecursionMerklePathRow::leaf_absorb_at_level(
            proof_idx,
            99,
            commit_id,
            level,
            0,
            11,
            1,
            false,
            true,
            true,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            chunk,
            [true; 8],
            &poseidon2_memo,
        );
        let mut memo = MerkleWalkMemo::default();
        memo.insert_leaf(proof_idx, first).expect("first coordinate is admitted");
        assert_eq!(
            memo.insert_leaf(proof_idx, conflicting_preimage),
            Err(WhirRecordError::MerkleUnionNodeMismatch { proof_idx, commit_id, level, idx: 11 })
        );
    }

    #[test]
    fn merkle_dedup_is_commit_complete_and_inverses_use_post_dedup_demand() {
        use core::borrow::Borrow;

        use crate::transcript_dt::merkle_path::{columns::MerklePathCols, trace::trace_row};

        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 11;
        let first_commit = 37;
        let second_commit = 38;

        let leaf = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            73,
            first_commit,
            0,
            9,
            1,
            false,
            true,
            true,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            digest(60),
            [true; DIGEST_SIZE],
            &poseidon2_memo,
        );
        let first_root = RecursionMerklePathRow::path_compress(
            proof_idx,
            first_commit,
            0,
            9,
            output_digest(&leaf),
            digest(70),
            true,
            &poseidon2_memo,
        );
        let second_root = RecursionMerklePathRow::path_compress(
            proof_idx,
            second_commit,
            0,
            9,
            output_digest(&leaf),
            digest(70),
            true,
            &poseidon2_memo,
        );

        let mut memo = MerkleWalkMemo::default();
        memo.insert_leaf(proof_idx, leaf).expect("first leaf is admitted");
        memo.bump_iopp_absorb(proof_idx, first_commit, 9, 0).expect("first duplicate leaf demand");
        memo.bump_iopp_absorb(proof_idx, first_commit, 9, 0).expect("second duplicate leaf demand");
        memo.insert_node(proof_idx, first_root).expect("first root is admitted");
        memo.insert_node(proof_idx, first_root).expect("duplicate root is deduplicated");
        memo.insert_node(proof_idx, second_root)
            .expect("same coordinate under another commitment stays distinct");

        let materialized = memo.into_materialized(proof_idx);
        assert_eq!(
            materialized.rows.iter().filter(|row| row.is_leaf_absorb()).count(),
            1,
            "proof-local leaf structure is materialized once"
        );
        assert_eq!(
            materialized.rows.iter().filter(|row| row.is_node()).count(),
            2,
            "commit_id is part of the structural dedup key"
        );

        let materialized_leaf =
            materialized.rows.iter().find(|row| row.is_leaf_absorb()).expect("materialized leaf");
        assert_eq!(materialized_leaf.absorb_cnt, 3);
        let leaf_values = trace_row(materialized_leaf);
        let leaf_cols: &MerklePathCols<F> = leaf_values.as_slice().borrow();
        assert_eq!(leaf_cols.absorb_cnt * leaf_cols.left_idx, F::one());

        let first_materialized_root = materialized
            .rows
            .iter()
            .find(|row| row.is_node() && row.commit_id == first_commit)
            .expect("first materialized root");
        assert_eq!(first_materialized_root.root_cnt, 2);
        let first_root_values = trace_row(first_materialized_root);
        let first_root_cols: &MerklePathCols<F> = first_root_values.as_slice().borrow();
        assert_eq!(first_root_cols.root_cnt * first_root_cols.block_idx, F::one());

        let second_materialized_root = materialized
            .rows
            .iter()
            .find(|row| row.is_node() && row.commit_id == second_commit)
            .expect("second materialized root");
        assert_eq!(second_materialized_root.root_cnt, 1);
        let second_root_values = trace_row(second_materialized_root);
        let second_root_cols: &MerklePathCols<F> = second_root_values.as_slice().borrow();
        assert_eq!(second_root_cols.root_cnt * second_root_cols.block_idx, F::one());
    }

    #[test]
    fn iopp_merkle_rows_follow_c_pair_leaf_blocks_and_defer_root_binding() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 3;
        let query_idx = 7;
        let round_idx = 4;
        let depth = 2;
        let cur_idx = 3;
        // Pair-leaf identity: depth = query_bits - 1 - cursor; position = chain_send_idx.
        let query_bits = round_idx + 1 + depth;
        let unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE + round_idx, depth);
        let sibling0 = digest(10);
        let sibling1 = digest(20);
        let opening_proof = [sibling0, sibling1];
        let row = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx,
            cursor: round_idx,
            query_bits,
            chain_send_idx: F::from_canonical_usize(cur_idx),
            f0: ext_limbs_for_test(100),
            f1: ext_limbs_for_test(200),
            ..RecursionWhirQueryFoldRow::default()
        };

        let (chunk0, mask0) = iopp_pair_leaf_block(&row, 0);
        let (chunk1, mask1) = iopp_pair_leaf_block(&row, 1);
        assert_eq!(chunk0[..5], row.f0);
        assert_eq!(chunk0[5..], row.f1[..3]);
        assert_eq!(chunk1[..2], row.f1[3..]);
        assert_eq!(mask0, [true; 8]);
        assert_eq!(mask1, [true, true, false, false, false, false, false, false]);

        let leaf0 = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            100 + round_idx,
            0,
            cur_idx,
            1,
            false,
            true,
            false,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            chunk0,
            mask0,
            &poseidon2_memo,
        );
        let leaf1 = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            100 + round_idx,
            1,
            cur_idx,
            1,
            false,
            false,
            true,
            leaf0.output,
            chunk1,
            mask1,
            &poseidon2_memo,
        );
        let path0 = RecursionMerklePathRow::path_compress(
            proof_idx,
            100 + round_idx,
            0,
            cur_idx,
            output_digest(&leaf1),
            sibling0,
            false,
            &poseidon2_memo,
        );
        let path1 = RecursionMerklePathRow::path_compress(
            proof_idx,
            100 + round_idx,
            1,
            path0.next_idx,
            output_digest(&path0),
            sibling1,
            true,
            &poseidon2_memo,
        );
        let expected_root = output_digest(&path1);

        let mut memo = MerkleWalkMemo::default();
        build_iopp_merkle_rows_for_round(
            proof_idx,
            query_idx,
            &row,
            &opening_proof,
            &poseidon2_memo,
            &mut memo,
        )
        .expect("synthetic IOPP path is internally consistent");
        let rows = memo.into_materialized(proof_idx).rows;
        assert_eq!(&rows[..2], &[leaf0, leaf1]);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[2].input, path0.input);
        assert_eq!(rows[2].output, path0.output);
        assert_eq!(rows[3].input, path1.input);
        assert_eq!(rows[3].output, path1.output);
        assert_eq!(rows[3].root_cnt, 1);

        assert_eq!(
            build_iopp_merkle_rows_for_round(
                proof_idx,
                query_idx,
                &row,
                &opening_proof[..1],
                &poseidon2_memo,
                &mut MerkleWalkMemo::default(),
            ),
            Err(WhirRecordError::IoppMerkleProofLengthMismatch {
                query_idx,
                round_idx,
                expected: depth,
                actual: 1,
            })
        );

        assert_ne!(expected_root, digest(99));
        let mut mismatched_claim_memo = MerkleWalkMemo::default();
        build_iopp_merkle_rows_for_round(
            proof_idx,
            query_idx,
            &row,
            &opening_proof,
            &poseidon2_memo,
            &mut mismatched_claim_memo,
        )
        .expect("root binding is deferred to AIR");
        let mismatched_claim_rows = mismatched_claim_memo.into_materialized(proof_idx).rows;
        assert_eq!(output_digest(mismatched_claim_rows.last().expect("root row")), expected_root);
    }

    #[test]
    fn input_batch_merkle_rows_replay_mixed_height_injection_and_defer_root_binding() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 2;
        let query_idx = 5;
        let batch_id = WHIR_BATCH_MAIN;
        let query_bits = 4;
        let query_sample = 13;
        let seed_idx = query_sample >> 1;
        let max_depth = 3;
        let sibling0 = digest(10);
        let sibling1 = digest(20);
        let sibling2 = digest(30);
        let opening_proof = [sibling0, sibling1, sibling2];
        let tallest_unit_key = whir_unit_key(WHIR_INPUT_MAIN_PATH_SLOT, 3);
        let injected_unit_key = whir_unit_key(WHIR_INPUT_MAIN_PATH_SLOT, 2);
        let tallest_chunk = digest(100);
        let injected_chunk = digest(200);
        let full_mask = [true; 8];

        let tallest_row = RecursionWhirLeafStreamRow {
            proof_idx,
            idx: seed_idx,
            log_height: 3,
            values: tallest_chunk,
            chunk_mask: full_mask,
            unit_key: tallest_unit_key,
            block_idx: 0,
            ..RecursionWhirLeafStreamRow::default()
        };
        let injected_row = RecursionWhirLeafStreamRow {
            proof_idx,
            idx: seed_idx >> 1,
            log_height: 2,
            values: injected_chunk,
            chunk_mask: full_mask,
            unit_key: injected_unit_key,
            block_idx: 0,
            ..RecursionWhirLeafStreamRow::default()
        };
        let mut leaf_groups = BTreeMap::new();
        leaf_groups.insert(
            (3usize, seed_idx),
            LeafGroupInstance { base: vec![tallest_row], ext: Vec::new(), serve_cnt: 1 },
        );
        leaf_groups.insert(
            (2usize, seed_idx >> 1),
            LeafGroupInstance { base: vec![injected_row], ext: Vec::new(), serve_cnt: 1 },
        );
        let segments = [
            crate::system_dt::WhirBatchRlcSegment {
                log_height: 2,
                batch_id,
                batch_pos: 0,
                chip_idx: 0,
                width: 8,
                first_cursor: 0,
                element_count: 8,
            },
            crate::system_dt::WhirBatchRlcSegment {
                log_height: 1,
                batch_id,
                batch_pos: 1,
                chip_idx: 1,
                width: 8,
                first_cursor: 8,
                element_count: 8,
            },
        ];
        let tallest_leaf = RecursionMerklePathRow::leaf_absorb_at_level(
            proof_idx,
            tallest_unit_key,
            PROOF_SHAPE_COMMIT_MAIN,
            0,
            0,
            seed_idx,
            1,
            false,
            true,
            true,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            tallest_chunk,
            full_mask,
            &poseidon2_memo,
        );
        let path0 = RecursionMerklePathRow::path_compress(
            proof_idx,
            PROOF_SHAPE_COMMIT_MAIN,
            0,
            seed_idx,
            output_digest(&tallest_leaf),
            sibling0,
            false,
            &poseidon2_memo,
        );
        let injected_leaf = RecursionMerklePathRow::leaf_absorb_at_level(
            proof_idx,
            injected_unit_key,
            PROOF_SHAPE_COMMIT_MAIN,
            1,
            0,
            seed_idx >> 1,
            1,
            false,
            true,
            true,
            [F::zero(); crate::config::POSEIDON2_WIDTH],
            injected_chunk,
            full_mask,
            &poseidon2_memo,
        );
        let inject = RecursionMerklePathRow::inject_compress(
            proof_idx,
            PROOF_SHAPE_COMMIT_MAIN,
            1,
            path0.next_idx,
            output_digest(&path0),
            output_digest(&injected_leaf),
            false,
            &poseidon2_memo,
        );
        let path1 = RecursionMerklePathRow::path_compress(
            proof_idx,
            PROOF_SHAPE_COMMIT_MAIN,
            2,
            inject.next_idx,
            output_digest(&inject),
            sibling1,
            false,
            &poseidon2_memo,
        );
        let path2 = RecursionMerklePathRow::path_compress(
            proof_idx,
            PROOF_SHAPE_COMMIT_MAIN,
            3,
            path1.next_idx,
            output_digest(&path1),
            sibling2,
            true,
            &poseidon2_memo,
        );
        let expected_root = output_digest(&path2);

        let mut memo = MerkleWalkMemo::default();
        build_input_batch_merkle_rows_for_batch(
            proof_idx,
            query_idx,
            query_sample,
            batch_id,
            &segments,
            &leaf_groups,
            &opening_proof,
            query_bits,
            1,
            &poseidon2_memo,
            &mut memo,
        )
        .expect("mixed-height input path is internally consistent");
        let rows = memo.into_materialized(proof_idx).rows;
        assert_eq!(&rows[..2], &[tallest_leaf, injected_leaf]);
        assert_eq!(rows.len(), 6);
        for (actual, expected) in rows[2..].iter().zip([path0, inject, path1, path2]) {
            assert_eq!(actual.op, expected.op);
            assert_eq!(actual.input, expected.input);
            assert_eq!(actual.output, expected.output);
        }
        assert_eq!(rows[5].root_cnt, 1);
        assert_ne!(expected_root, digest(99));
        let mut mismatched_claim_memo = MerkleWalkMemo::default();
        build_input_batch_merkle_rows_for_batch(
            proof_idx,
            query_idx,
            query_sample,
            batch_id,
            &segments,
            &leaf_groups,
            &opening_proof,
            query_bits,
            1,
            &poseidon2_memo,
            &mut mismatched_claim_memo,
        )
        .expect("root binding is deferred to AIR");
        let mismatched_claim_rows = mismatched_claim_memo.into_materialized(proof_idx).rows;
        assert_eq!(output_digest(mismatched_claim_rows.last().expect("root row")), expected_root);
        assert_eq!(
            build_input_batch_merkle_rows_for_batch(
                proof_idx,
                query_idx,
                query_sample,
                batch_id,
                &segments,
                &leaf_groups,
                &opening_proof[..2],
                query_bits,
                1,
                &poseidon2_memo,
                &mut MerkleWalkMemo::default(),
            ),
            Err(WhirRecordError::InputMerkleProofLengthMismatch {
                query_idx,
                batch_id,
                expected: max_depth,
                actual: 2,
            })
        );
    }

    fn dense_sample_events(len: usize) -> Vec<RecursionTranscriptEvent> {
        (0..len)
            .map(|tidx| RecursionTranscriptEvent {
                tidx,
                kind: RecursionTranscriptEventKind::Sample,
                value: F::zero(),
            })
            .collect()
    }

    fn ext_limbs_for_test(seed: usize) -> [F; D_EF] {
        core::array::from_fn(|idx| F::from_canonical_usize(seed + idx))
    }

    fn digest(seed: usize) -> [F; 8] {
        core::array::from_fn(|idx| F::from_canonical_usize(seed + idx))
    }
}
