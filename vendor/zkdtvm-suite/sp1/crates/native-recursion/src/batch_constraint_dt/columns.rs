use native_recursion_derive::AlignedBorrow;

use crate::config::D_EF;

pub const BATCH_SUMCHECK_EVALS: usize = 5;
pub const BATCH_ROUND_EVENT_LIMBS: usize = BATCH_SUMCHECK_EVALS * D_EF + D_EF;

pub const fn batch_sumcheck_width() -> usize {
    NUM_BATCH_SUMCHECK_COLS
}
pub const BATCH_VK_TAG_VERSION_LIMBS: usize = 2;
pub const BATCH_VK_TAG_V1: u32 = 0x3156_4b47;
pub const BATCH_VK_VERSION_V1: u32 = 1;
pub const BATCH_COMMITMENT_LIMBS: usize = 8;
pub const BATCH_ACTIVE_SHAPE_HEADER_LIMBS: usize = 3;
pub const BATCH_ACTIVE_SHAPE_ENTRY_LIMBS: usize = 5;

pub const fn batch_active_shape_limbs(c_chips: usize) -> usize {
    BATCH_ACTIVE_SHAPE_HEADER_LIMBS + BATCH_ACTIVE_SHAPE_ENTRY_LIMBS * c_chips
}
/// GKV1 tag/version plus the canonical Core VK transcript metadata
/// (`commit[8] || pc_start || kind || x[11] || y[11]`).
pub const BATCH_CORE_SEED_PREFIX_LIMBS: usize =
    BATCH_VK_TAG_VERSION_LIMBS + BATCH_COMMITMENT_LIMBS + 1 + 23;
/// GKV1 tag/version plus the native VK commitment. Native programs have an
/// empty Global-owner registry, so pc and seed are not observed or transported.
pub const BATCH_NATIVE_SEED_PREFIX_LIMBS: usize =
    BATCH_VK_TAG_VERSION_LIMBS + BATCH_COMMITMENT_LIMBS;

pub const fn batch_seed_prefix_limbs(contains_global_bus: bool) -> usize {
    if contains_global_bus {
        BATCH_CORE_SEED_PREFIX_LIMBS
    } else {
        BATCH_NATIVE_SEED_PREFIX_LIMBS
    }
}

pub const fn batch_seed_prefix_limbs_for_role_id(role_id: usize) -> usize {
    batch_seed_prefix_limbs(role_id == 0)
}
pub const BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS: usize = 18;
pub const BATCH_INTERP_MATRIX: [[(i64, u32); BATCH_SUMCHECK_EVALS]; BATCH_SUMCHECK_EVALS] = [
    [(1, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
    [(-25, 12), (4, 1), (-3, 1), (4, 3), (-1, 4)],
    [(35, 24), (-13, 3), (19, 4), (-7, 3), (11, 24)],
    [(-5, 12), (3, 2), (-2, 1), (7, 6), (-1, 4)],
    [(1, 24), (-1, 6), (1, 4), (-1, 6), (1, 24)],
];

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchTranscriptInputsCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub c_chips: T,
    /// `perm_alpha || perm_beta || alpha`, in transcript order.
    pub event_values: [T; 3 * D_EF],
}

pub const NUM_BATCH_TRANSCRIPT_INPUTS_COLS: usize = BatchTranscriptInputsCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchSumcheckCols<T> {
    pub proof_idx: T,
    pub is_seed: T,
    pub is_round: T,

    pub round_idx: T,
    pub r_rounds: T,
    pub c_chips: T,
    pub summary_id_base: T,

    /// Monomial coefficients c1..c4. c0 is derived from
    /// claim_in = 2*c0 + c1 + c2 + c3 + c4.
    pub coefficients: [[T; D_EF]; BATCH_SUMCHECK_EVALS - 1],
    pub challenge: [T; D_EF],
    pub eq_challenge: [T; D_EF],
    pub claim_in: [T; D_EF],
    pub acc_3: [T; D_EF],
    pub acc_2: [T; D_EF],
    pub acc_1: [T; D_EF],
    pub claim_out: [T; D_EF],
}

pub const NUM_BATCH_SUMCHECK_COLS: usize = BatchSumcheckCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchSumcheckReservedCols<T> {
    pub is_seed: T,
    pub is_round: T,
    pub round_idx: T,
}

pub const NUM_BATCH_SUMCHECK_RESERVED_COLS: usize = BatchSumcheckReservedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchSumcheckPackedCols<T> {
    pub coefficients: [T; BATCH_SUMCHECK_EVALS - 1],
    pub challenge: T,
    pub claim_in: T,
    pub acc_3: T,
    pub acc_2: T,
    pub acc_1: T,
    pub claim_out: T,
}

pub const NUM_BATCH_SUMCHECK_PACKED_COLS: usize = BatchSumcheckPackedCols::<u8>::width();
