use native_recursion_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeChipMetadataPreprocessedCols<T> {
    pub role_id: T,
    pub chip_id: T,
    pub stable_air_id_lo: T,
    pub stable_air_id_hi: T,
    pub prep_width: T,
    pub main_width: T,
    pub perm_width: T,
    /// Pre-provisioned for batch_constraint alpha-shift derivation.
    pub constraint_count: T,
    /// Static number of actual `air.eval()` gate roots.
    pub gate_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeChipMetadataCols<T> {
    pub mult: T,
}

pub const NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS: usize =
    NativeChipMetadataPreprocessedCols::<u8>::width();
pub const NUM_NATIVE_CHIP_METADATA_COLS: usize = NativeChipMetadataCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ProofShapeBinderCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_vk_commit: T,
    pub is_vk_meta: T,
    pub is_public_values: T,
    pub is_e1: T,
    pub is_active_shape_header: T,
    pub is_chip: T,
    pub is_e5: T,

    pub tidx_base: T,
    pub event_values: [T; 8],
    pub event_recv_mask: [T; 8],

    pub shape_idx_base: T,
    pub shape_value_send_mask: [T; 8],
    pub shape_value_send_mults: [T; 8],

    pub commit_id: T,
    pub whir_role_config_recv_mult: T,

    pub role_id: T,
    pub chip_idx: T,
    pub static_chip_id: T,
    pub stable_air_id_lo: T,
    pub stable_air_id_hi: T,
    /// Segment bit of this chip row's static id (id in [128*b, 128*b + 128), forced by
    /// the dual range8 band); threaded transitively along the 1012 chain via the
    /// prev-side twin below, so the whole proof stays in ONE replay segment.
    pub seg_bit: T,
    /// bit7 of prev_static_chip_id (chain-bound), same band forcing; also serves the E5
    /// row (whose prev state is the LAST chip) to bind the 1022 id_base payload.
    pub prev_seg_bit: T,
    pub log_height: T,
    pub prep_width: T,
    pub main_width: T,
    pub perm_width: T,
    pub constraint_count: T,
    pub gate_count: T,
    pub has_prep: T,
    pub prep_width_inv: T,

    pub prev_chip_idx: T,
    pub prev_log_height: T,
    pub prev_static_chip_id: T,
    pub prev_tidx_acc: T,
    pub prev_prep_matrix_idx: T,
    pub prev_first_log_height: T,
    pub prev_shape_chip_count: T,
    pub prev_chip_idx_inv: T,
    pub is_first_chip: T,
    pub chain_send_chip_idx: T,
    pub chain_send_log_height: T,
    pub chain_send_static_chip_id: T,
    pub chain_send_tidx_acc: T,
    pub chain_send_prep_matrix_idx: T,
    pub chain_send_first_log_height: T,
    pub chain_send_shape_chip_count: T,
    pub is_group_start: T,
    pub range_val: T,

    /// Exact row-count publication for the canonical Fold plan rows of this
    /// chip. It is constrained to `is_chip * (gates + batches + 1)`.
    pub chip_meta_send_mult: T,
    pub batch_dim_prep_send_mult: T,
    pub batch_dim_perm_send_mult: T,
    pub summary_send_mult: T,
    /// Exact E5 source multiplicity: one Fold receiver, one Challenge receiver
    /// per present chip, and one fused BatchTranscriptInputs receiver (`C + 2`).
    pub fold_plan_source_mult: T,
}

pub const NUM_PROOF_SHAPE_BINDER_COLS: usize = ProofShapeBinderCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ProofHeightSetCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_first: T,
    pub is_last: T,
    pub height_cursor: T,
    pub member_count: T,
    pub member_count_inv: T,
    pub present: T,
    pub rank: T,
    pub height_group_send_mult: T,
}

pub const NUM_PROOF_HEIGHT_SET_COLS: usize = ProofHeightSetCols::<u8>::width();
