use native_recursion_derive::AlignedBorrow;

use crate::config::D_EF;

pub const WHIR_ROLE_CORE: usize = 0;
pub const WHIR_ROLE_COMPRESS: usize = 1;
pub const WHIR_ROLE_SHRINK: usize = 2;
pub const WHIR_ROLE_COUNT: usize = 3;

pub const WHIR_TWIDDLE_TABLES: usize = 3;
pub const WHIR_TWIDDLE_ROWS: usize = 256;
pub const WHIR_SAMPLE_BAND_ROWS: usize = crate::child_views::KOALABEAR_MAX_TRACE_LOG_HEIGHT;
pub const WHIR_ROUND_MAX_TRANSCRIPT_EVENTS: usize = 33;
pub const WHIR_LEAF_BASE_LIMBS_PER_ROW: usize = 8;
pub const WHIR_LEAF_EXT_LIMBS_PER_ROW: usize = 40;
pub const WHIR_LEAF_BLOCKS_PER_ROW: usize = 5;
pub const WHIR_LEAF_RLC_SLOTS: usize = 8;
pub const WHIR_LEAF_RLC_STORED_POWS: usize = WHIR_LEAF_RLC_SLOTS - 1;
pub const WHIR_QUERY_SEED_SLOTS: usize = 4;
pub const WHIR_QUERY_PAIR_LEAF_BLOCKS: usize = 2;
pub const WHIR_CORE_QUERY_SAMPLE_BITS: usize = 22;
pub const WHIR_CORE_LOG_BLOWUP: usize = 1;
pub const WHIR_CORE_NUM_ROUNDS: usize = WHIR_CORE_QUERY_SAMPLE_BITS - WHIR_CORE_LOG_BLOWUP;
pub const WHIR_CORE_QUERY_SAMPLE_HIGH_BITS: usize = 9;
pub const WHIR_CORE_QUERY_SAMPLE_SHIFT: usize = 1 << WHIR_CORE_QUERY_SAMPLE_BITS;
pub const WHIR_CORE_QUERY_SAMPLE_HIGH_MAX: usize = 508;
/// Provider capacity for paired `(high, high_max - high)` upper-bound checks.
///
/// Exact bounds remain authenticated by their source AIR; both non-negative terms use the one
/// fixed wide provider so mathematical widths do not create AIR identities.
pub const WHIR_PAIRED_RANGE_BITS: usize = 21;
// KoalaBear modulus is 2^31 - 2^24 + 1; PoW high maxima are (p - 1) / 2^bits.
pub const WHIR_BATCHING_POW_BITS: usize = 10;
pub const WHIR_BATCHING_POW_SHIFT: usize = 1 << WHIR_BATCHING_POW_BITS;
pub const WHIR_BATCHING_POW_HIGH_MAX: usize = 2_080_768;
pub const WHIR_QUERY_POW_BITS: usize = 20;
pub const WHIR_QUERY_POW_SHIFT: usize = 1 << WHIR_QUERY_POW_BITS;
pub const WHIR_QUERY_POW_HIGH_MAX: usize = 2_032;
pub const WHIR_PATH_SLOTS_PER_QUERY: usize = 32;
pub const WHIR_INPUT_PREPROCESSED_PATH_SLOT: usize = 0;
pub const WHIR_INPUT_MAIN_PATH_SLOT: usize = 1;
pub const WHIR_INPUT_PERMUTATION_PATH_SLOT: usize = 2;
pub const WHIR_IOPP_ORACLE_PATH_SLOT_BASE: usize = 3;
// Leaf identity: `unit_key = slot·32 + codeword_log_height` — a routing tag shared
// by the leaf-stream 1004/1001 payloads and the merkle leaf rows.
// Note: record time asserts the bounds slot < 28 and log_height < 32.
pub const WHIR_UNIT_KEY_SLOT_STRIDE: usize = 32;
pub const WHIR_UNIT_KEY_MAX_SLOT: usize = 28; // 3 batches + IOPP rounds (up to ~24)

pub const fn whir_unit_key(slot: usize, codeword_log_height: usize) -> usize {
    assert!(slot < WHIR_UNIT_KEY_MAX_SLOT);
    assert!(codeword_log_height < WHIR_UNIT_KEY_SLOT_STRIDE);
    slot * WHIR_UNIT_KEY_SLOT_STRIDE + codeword_log_height
}
pub const WHIR_FINAL_ROOT_POSEIDON2_PERMS: usize = 4;
pub const WHIR_FINAL_ROOT_DIGEST_LANES: usize = 8;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirTwiddlePreprocessedCols<T> {
    pub byte: T,
    pub values: [T; WHIR_TWIDDLE_TABLES],
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirTwiddleCols<T> {
    pub mults: [T; WHIR_TWIDDLE_TABLES],
}

pub const NUM_WHIR_TWIDDLE_PREPROCESSED_COLS: usize = WhirTwiddlePreprocessedCols::<u8>::width();
pub const NUM_WHIR_TWIDDLE_COLS: usize = WhirTwiddleCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirSampleBandPreprocessedCols<T> {
    pub query_bits: T,
    pub shift: T,
    pub high_max: T,
    pub high_bits: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirSampleBandCols<T> {
    pub mult: T,
}

pub const NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS: usize =
    WhirSampleBandPreprocessedCols::<u8>::width();
pub const NUM_WHIR_SAMPLE_BAND_COLS: usize = WhirSampleBandCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirRoundCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_pow_batch: T,
    pub is_preamble: T,
    pub is_round: T,
    pub is_final: T,
    pub is_final_perm: T,
    pub final_root_perm_step_flags: [T; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub round: T,
    pub tidx: T,
    pub query_bits: T,
    pub r_rounds: T,
    pub c_chips: T,
    pub w_qbase: T,
    pub opening_idx: T,
    pub opening_point: [T; D_EF],
    pub height_group_rank: T,
    pub height_group_log_height: T,
    pub group_claim_log_height: T,
    pub group_claim: [T; D_EF],
    pub commit_id: T,
    pub commit_root: [T; 8],
    pub event_value: [T; WHIR_ROUND_MAX_TRANSCRIPT_EVENTS],
    pub pow_sample_high: T,
    pub round_has_oracle: T,

    pub chain_recv_round: T,
    pub chain_recv_tidx: T,
    pub chain_recv_claim: [T; D_EF],
    pub chain_recv_eq: [T; D_EF],
    pub chain_recv_pending_is_merge: T,
    pub chain_recv_pending_beta: [T; D_EF],
    pub chain_recv_pending_eq: [T; D_EF],

    pub chain_send_round: T,
    pub chain_send_tidx: T,
    pub chain_send_claim: [T; D_EF],
    pub chain_send_eq: [T; D_EF],
    pub chain_send_pending_is_merge: T,
    pub chain_send_pending_beta: [T; D_EF],
    pub chain_send_pending_eq: [T; D_EF],

    pub r_fold: [T; D_EF],
    pub is_merge: T,
    pub emit_prep_seed: T,
    pub merge_log_height: T,
    pub cfr: [T; D_EF],
    pub claim_acc: [T; D_EF],
    pub claim_folded: [T; D_EF],
    pub eq_factor: [T; D_EF],
    pub eq_folded: [T; D_EF],

    pub bcast_mult: T,
    pub query_init_mult: T,
    pub summary_id_base: T,
    pub commitment_root_send_mult: T,
    pub final_root_poseidon2_recv_mult: T,
}

pub const NUM_WHIR_ROUND_COLS: usize = WhirRoundCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirBatchEvalCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_start: T,
    pub is_group_end: T,
    pub cursor: T,
    pub chain_recv_cursor: T,
    pub chain_send_cursor: T,
    pub chain_recv_log_height: T,
    pub chain_recv_batch_id: T,
    pub chain_recv_batch_pos: T,
    pub chain_recv_value_idx: T,
    pub chain_recv_segment_element_count: T,
    pub alpha_tidx: T,
    pub alpha: [T; D_EF],
    pub pow_in: [T; D_EF],
    pub acc_in: [T; D_EF],
    pub group_base_in: [T; D_EF],
    pub pow_out: [T; D_EF],
    pub acc_out: [T; D_EF],
    pub group_base_out: [T; D_EF],
    pub value: [T; D_EF],
    pub log_height: T,
    pub batch_id: T,
    pub batch_pos: T,
    pub chip_idx: T,
    pub static_chip_id: T,
    pub width: T,
    pub value_idx: T,
    pub segment_element_count: T,
    pub is_value: T,
    pub is_segment_start: T,
    pub is_segment_end: T,
    pub is_first_value: T,
    pub is_group_start: T,
    pub is_perm_batch: T,
    pub group_log_height_gap: T,
    pub batch_dim_recv_mult: T,
    pub opened_eval_send_mult: T,
    /// Bus 1044 pow-seed publication count (one recv per deduped leaf group
    /// instance at this height).
    pub pow_seed_cnt: T,
}

pub const NUM_WHIR_BATCH_EVAL_COLS: usize = WhirBatchEvalCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirQueryFoldCols<T> {
    pub proof_idx: T,
    pub is_seed: T,
    pub is_round: T,
    pub query_idx: T,
    pub cursor: T,
    pub w_qbase: T,
    pub query_bits: T,
    pub r_rounds: T,
    pub query_sample: T,
    pub query_sample_raw: T,
    pub query_sample_high: T,
    pub query_sample_shift: T,
    pub query_sample_high_max: T,
    pub query_sample_high_bits: T,
    pub query_sample_high_gap_inv: T,
    pub idx: T,
    pub idx_bit: T,
    pub idx_tail_bit0: T,
    pub idx_tail_bit1: T,
    pub x: T,
    pub acc: T,
    pub ipw: T,
    pub folded: [T; D_EF],
    pub f0: [T; D_EF],
    pub f1: [T; D_EF],
    pub chain_send_cursor: T,
    pub chain_send_idx: T,
    pub chain_send_idx_bit: T,
    pub chain_send_x: T,
    pub chain_send_acc: T,
    pub chain_send_ipw: T,
    pub chain_send_folded: [T; D_EF],
    pub r_fold: [T; D_EF],
    pub is_merge: T,
    pub is_assign: T,
    /// Inverse of `cursor` on nonzero-cursor merge rows, zero otherwise.
    /// Together with `is_assign`, this makes first-round assignment exact.
    pub merge_cursor_inv: T,
    pub merge_beta: [T; D_EF],
    pub merge_eq: [T; D_EF],
    pub emit_prep_seed: T,
    pub cfr: [T; D_EF],
    pub leaf_sum: [T; D_EF],
    pub twiddle_bytes: [T; WHIR_TWIDDLE_TABLES],
    pub twiddle_values: [T; WHIR_TWIDDLE_TABLES],
    pub twiddle_product_01: T,
}

pub const NUM_WHIR_QUERY_FOLD_COLS: usize = WhirQueryFoldCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirQueryFoldReservedCols<T> {
    pub is_seed: T,
    pub is_round: T,
    pub cursor: T,
    pub query_sample: T,
    pub query_sample_raw: T,
    pub query_sample_high: T,
    pub query_sample_shift: T,
    pub query_sample_high_max: T,
    pub query_sample_high_gap_inv: T,
    pub idx: T,
    pub idx_bit: T,
    pub idx_tail_bit0: T,
    pub idx_tail_bit1: T,
    pub x: T,
    pub acc: T,
    pub ipw: T,
    pub chain_send_cursor: T,
    pub chain_send_idx: T,
    pub chain_send_idx_bit: T,
    pub chain_send_x: T,
    pub chain_send_acc: T,
    pub chain_send_ipw: T,
    pub is_merge: T,
    pub is_assign: T,
    pub merge_cursor_inv: T,
    pub emit_prep_seed: T,
    pub twiddle_bytes: [T; WHIR_TWIDDLE_TABLES],
    pub twiddle_values: [T; WHIR_TWIDDLE_TABLES],
    pub twiddle_product_01: T,
}

pub const NUM_WHIR_QUERY_FOLD_RESERVED_COLS: usize = WhirQueryFoldReservedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirQueryFoldPackedCols<T> {
    pub folded: T,
    pub f0: T,
    pub f1: T,
    pub chain_send_folded: T,
    pub r_fold: T,
    pub merge_beta: T,
    pub merge_eq: T,
    pub cfr: T,
    pub leaf_sum: T,
}

pub const NUM_WHIR_QUERY_FOLD_PACKED_COLS: usize = WhirQueryFoldPackedCols::<u8>::width();
pub const NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS: usize =
    9 + WHIR_TWIDDLE_TABLES + WHIR_QUERY_PAIR_LEAF_BLOCKS;
pub const NUM_WHIR_QUERY_FOLD_PRECOMPUTED_COLS: usize =
    NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS + NUM_WHIR_QUERY_FOLD_PACKED_COLS;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafStreamCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_unit_start: T,
    pub is_unit_end: T,
    /// The group instance's truncated leaf index.
    pub idx: T,
    /// Bus 1025 publication count (consuming-query merges).
    pub serve_cnt: T,
    pub chain_recv_cursor: T,
    pub chain_send_cursor: T,
    pub log_height: T,
    pub batch_id: T,
    pub chain_recv_log_height: T,
    pub chain_recv_batch_id: T,
    pub is_unit_key_start: T,
    pub unit_key_gap: T,
    pub alpha: [T; D_EF],
    pub pow_in: [T; D_EF],
    pub acc_in: [T; D_EF],
    pub slot_pows: [[T; D_EF]; WHIR_LEAF_RLC_SLOTS],
    pub pow_out: [T; D_EF],
    pub acc_out: [T; D_EF],
    pub values: [T; WHIR_LEAF_BASE_LIMBS_PER_ROW],
    pub chunk_mask: [T; WHIR_LEAF_BASE_LIMBS_PER_ROW],
    pub unit_key: T,
    pub block_idx: T,
}

pub const NUM_WHIR_LEAF_STREAM_COLS: usize = WhirLeafStreamCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafExtStreamCols<T> {
    pub proof_idx: T,
    pub is_unit_end: T,
    /// The group instance's truncated leaf index.
    pub idx: T,
    /// Bus 1025 publication count (consuming-query merges).
    pub serve_cnt: T,
    pub chain_recv_cursor: T,
    pub log_height: T,
    pub is_unit_key_start: T,
    pub alpha: [T; D_EF],
    pub pow_in: [T; D_EF],
    pub acc_in: [T; D_EF],
    /// Slot powers for extension elements 1 through 7. Element zero reuses `pow_in`.
    pub slot_pows: [[T; D_EF]; WHIR_LEAF_RLC_STORED_POWS],
    pub pow_out: [T; D_EF],
    pub acc_out: [T; D_EF],
    pub values: [T; WHIR_LEAF_EXT_LIMBS_PER_ROW],
    /// One selector per ext5 element. The 40 limb masks and five Merkle block
    /// masks are exact repetitions of these selectors in flat-limb order.
    pub element_masks: [T; WHIR_LEAF_RLC_SLOTS],
    pub block_idx: T,
}

pub const NUM_WHIR_LEAF_EXT_STREAM_COLS: usize = WhirLeafExtStreamCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafExtStreamReservedCols<T> {
    pub is_unit_end: T,
    pub serve_cnt: T,
    pub is_unit_key_start: T,
    pub element_masks: [T; WHIR_LEAF_RLC_SLOTS],
}

pub const NUM_WHIR_LEAF_EXT_STREAM_RESERVED_COLS: usize =
    WhirLeafExtStreamReservedCols::<u8>::width();

/// Lookup denominators retain the row/product-closed positional order.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafExtStreamDenominatorCols<T> {
    pub leaf_chain_recv: T,
    pub leaf_chain_send: T,
    pub merkle_leaf_blocks: [T; WHIR_LEAF_BLOCKS_PER_ROW],
    pub query_leaf_sum: T,
}

pub const NUM_WHIR_LEAF_EXT_STREAM_DENOMINATOR_COLS: usize =
    WhirLeafExtStreamDenominatorCols::<u8>::width();

/// Exact degree-one ext5 packs consumed after the denominator prefix.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafExtStreamPackedCols<T> {
    pub alpha: T,
    pub pow_in: T,
    pub slot_pows: [T; WHIR_LEAF_RLC_STORED_POWS],
    pub pow_out: T,
    pub acc_delta: T,
    pub values: [T; WHIR_LEAF_RLC_SLOTS],
}

pub const NUM_WHIR_LEAF_EXT_STREAM_PACKED_COLS: usize = WhirLeafExtStreamPackedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirLeafExtStreamPrecomputedCols<T> {
    pub denominators: WhirLeafExtStreamDenominatorCols<T>,
    pub packed: WhirLeafExtStreamPackedCols<T>,
}

pub const NUM_WHIR_LEAF_EXT_STREAM_PRECOMPUTED_COLS: usize =
    WhirLeafExtStreamPrecomputedCols::<u8>::width();
