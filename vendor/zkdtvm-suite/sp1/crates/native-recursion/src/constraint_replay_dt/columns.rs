use native_recursion_derive::AlignedBorrow;

use crate::config::D_EF;

// Current core role max_required_beta_power is 104 across the 56-chip universe.
pub const CONSTRAINT_MAX_BETA_POWERS: usize = 105;
pub const CONSTRAINT_GLOBAL_CHAIN_BLOCKS: usize = 7;
pub const CONSTRAINT_CHAIN_LIMBS: usize = CONSTRAINT_GLOBAL_CHAIN_BLOCKS;
pub const CONSTRAINT_LEAF_KIND_COUNT: usize = 10;

pub const CONSTRAINT_LEAF_PREPROCESSED: usize = 0;
pub const CONSTRAINT_LEAF_MAIN: usize = 1;
pub const CONSTRAINT_LEAF_PUBLIC: usize = 2;
pub const CONSTRAINT_LEAF_PERM_ALPHA: usize = 3;
pub const CONSTRAINT_LEAF_BETA_POWER: usize = 4;
pub const CONSTRAINT_LEAF_BETA_SEPTIX: usize = 5;
pub const CONSTRAINT_LEAF_PRECOMPUTED: usize = 6;
pub const CONSTRAINT_LEAF_RESERVED_POLY: usize = 7;
pub const CONSTRAINT_LEAF_IS_FIRST_ROW: usize = 8;
pub const CONSTRAINT_LEAF_IS_LAST_ROW: usize = 9;

pub const CONSTRAINT_CHALLENGE_PERM_ALPHA: usize = 0;
pub const CONSTRAINT_CHALLENGE_BETA_POWER: usize = 2;
pub const CONSTRAINT_CHALLENGE_BETA_SEPTIX: usize = 3;
pub const CONSTRAINT_CHALLENGE_IS_FIRST: usize = 4;
pub const CONSTRAINT_CHALLENGE_IS_LAST: usize = 5;
pub const CONSTRAINT_CHALLENGE_LCS: usize = 6;
pub const CONSTRAINT_CHALLENGE_STATE_LCS: usize = 7;

pub const CONSTRAINT_OP_LEAF: usize = 0;
pub const CONSTRAINT_OP_CONST: usize = 1;
pub const CONSTRAINT_OP_ADD: usize = 2;
pub const CONSTRAINT_OP_SUB: usize = 3;
pub const CONSTRAINT_OP_MUL: usize = 4;
pub const CONSTRAINT_OP_FUSED: usize = 5;

pub const CONSTRAINT_ROOT_GATE: usize = 0;
pub const CONSTRAINT_ROOT_MULTIPLICITY: usize = 1;
pub const CONSTRAINT_ROOT_PRECOMPUTE_DENOM: usize = 2;
pub const CONSTRAINT_ROOT_HEIGHT_INVERSE: usize = 3;
/// Static-id sentinel outside the authenticated child-chip range 0..=255.
pub const CONSTRAINT_HEIGHT_TABLE_STATIC_ID: usize = 256;
pub const CONSTRAINT_HEIGHT_TABLE_ROWS: usize = 25;
/// Every shipped KoalaBear/ext5 child AIR uses PolyAir lookup batches of two.
/// A tail batch leaves slot 1 inactive (denominator 1 / multiplicity 0).
pub const CONSTRAINT_FOLD_BATCH_SIZE: usize = 2;
pub const CONSTRAINT_FOLD_ROOT_SLOTS: usize = 2 * CONSTRAINT_FOLD_BATCH_SIZE;
pub const CONSTRAINT_TERMINAL_LCS_LIMBS: usize = D_EF;
pub const CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT: usize = 77;
/// Six state/address fields precede the Binder-aligned row beginning at PV 48.
pub const CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT: usize = 6;
/// Core PV rows with `shape_idx_base = 48, 56, ..., 112`.
pub const CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS: usize = 9;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintProgramPreprocessedCols<T> {
    pub static_chip_id: T,
    pub node_idx: T,
    pub op_code: T,
    pub lhs_idx: T,
    pub rhs_idx: T,
    pub third_idx: T,
    pub aux: T,
    pub leaf_kind: T,
    pub fanout: T,
}

pub const NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS: usize =
    ConstraintProgramPreprocessedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintProgramCols<T> {
    pub mult: T,
}

pub const NUM_CONSTRAINT_PROGRAM_COLS: usize = ConstraintProgramCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintRootTablePreprocessedCols<T> {
    pub static_chip_id: T,
    pub root_kind: T,
    pub root_ord: T,
    pub node_idx: T,
    pub sign: T,
}

pub const NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS: usize =
    ConstraintRootTablePreprocessedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintRootTableCols<T> {
    pub root_mult: T,
    pub height_mult: T,
}

pub const NUM_CONSTRAINT_ROOT_TABLE_COLS: usize = ConstraintRootTableCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintDagEvalCols<T> {
    pub proof_idx: T,
    pub chip_idx: T,
    pub static_chip_id: T,
    pub node_idx: T,

    pub is_const: T,
    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_fused: T,
    pub lhs_idx: T,
    pub rhs_idx: T,
    pub third_idx: T,
    pub aux: T,
    pub fanout: T,

    pub leaf_flags: [T; CONSTRAINT_LEAF_KIND_COUNT],

    pub value: [T; D_EF],
    pub lhs_value: [T; D_EF],
    pub rhs_value: [T; D_EF],
    pub third_value: [T; D_EF],

    pub opened_batch_pos: T,
}

pub const NUM_CONSTRAINT_DAG_EVAL_COLS: usize = ConstraintDagEvalCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintDagEvalReservedCols<T> {
    pub chip_idx: T,
    pub static_chip_id: T,

    pub is_const: T,
    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_fused: T,
    pub lhs_idx: T,
    pub rhs_idx: T,
    pub aux: T,
    pub fanout: T,

    pub leaf_flags: [T; CONSTRAINT_LEAF_KIND_COUNT],

    pub value_0: T,
    pub opened_batch_pos: T,
}

pub const NUM_CONSTRAINT_DAG_EVAL_RESERVED_COLS: usize =
    ConstraintDagEvalReservedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintFoldCols<T> {
    pub proof_idx: T,
    pub is_skip: T,
    pub is_gate: T,
    pub is_batch: T,

    pub cursor: T,
    /// Canonical PlanChain state. The active chip index is
    /// `remaining_chips - 1`.
    pub remaining_chips: T,
    pub local_ord: T,
    pub chain_send_local_ord: T,
    pub static_chip_id: T,
    pub log_height: T,
    pub gate_count: T,
    /// Authenticated number of batch-two rows in this chip.
    pub batch_count: T,
    /// Gate ordinal on gate rows, `2 * batch_ord` on batch rows, and the
    /// authenticated log height on skip rows.
    pub root_ord: T,

    pub alpha: [T; D_EF],
    pub acc_in: [T; D_EF],
    pub acc_out: [T; D_EF],
    pub pacc_in: [T; D_EF],
    pub pacc_out: [T; D_EF],
    pub perm_sum_in: [T; D_EF],
    pub perm_sum_out: [T; D_EF],

    pub root_nodes: [T; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub multiplicity_signs: [T; CONSTRAINT_FOLD_BATCH_SIZE],
    pub root_values: [[T; D_EF]; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub batch_has_second: T,

    /// Authenticated permutation evaluation on batch rows and authenticated
    /// local cumulative sum on skip rows.
    pub perm_value: [T; D_EF],
}

pub const NUM_CONSTRAINT_FOLD_COLS: usize = ConstraintFoldCols::<u8>::width();

/// Base-field values retained after round one. Denominator-only proof/chip/root
/// metadata remains committed but is not carried into the non-first evaluator.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintFoldReservedCols<T> {
    pub is_skip: T,
    pub is_gate: T,
    pub is_batch: T,
    /// `root_nodes[0]`: node id on gate/batch rows, authenticated
    /// `2^(-log_height)` on skip rows.
    pub height_inverse: T,
    pub batch_count: T,
    pub multiplicity_signs: [T; CONSTRAINT_FOLD_BATCH_SIZE],
    pub batch_has_second: T,
}

pub const NUM_CONSTRAINT_FOLD_RESERVED_COLS: usize = ConstraintFoldReservedCols::<u8>::width();

/// Lookup denominators occupy the required positional prefix of the
/// precomputed trace.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintFoldDenominatorCols<T> {
    pub root_table: [T; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub node_value: [T; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub permutation: T,
    pub lcs: T,
    pub height_inverse: T,
    pub chip_meta: T,
    pub plan_chain_recv: T,
    pub plan_chain_send: T,
    pub fold_chain_recv: T,
    pub fold_chain_send: T,
}

pub const NUM_CONSTRAINT_FOLD_DENOMINATOR_COLS: usize =
    ConstraintFoldDenominatorCols::<u8>::width();

/// Exact degree-one ext5 packs consumed by the AIR after the denominator
/// prefix. No nonlinear residual is packed here.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintFoldPackedCols<T> {
    pub alpha: T,
    pub acc_in: T,
    pub acc_out: T,
    pub pacc_in: T,
    pub pacc_out: T,
    /// `perm_sum_out - perm_sum_in`, packed once as a linear ext5 value.
    pub perm_delta: T,
    /// Reset witness used only to force the skip-row output state to literal
    /// zero. Keeping this one extra linear pack is cheaper than five reserved
    /// limbs and is required for per-chip, rather than merely global, reset.
    pub perm_sum_out: T,
    pub root_values: [T; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub perm_value: T,
    /// Canonical schedule relations formerly owned by ConstraintFoldPlanAir.
    pub gate_position: T,
    pub batch_position: T,
    pub skip_position: T,
    pub skip_height: T,
    pub non_skip_successor: T,
    pub skip_successor: T,
}

pub const NUM_CONSTRAINT_FOLD_PACKED_COLS: usize = ConstraintFoldPackedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintFoldPrecomputedCols<T> {
    pub denominators: ConstraintFoldDenominatorCols<T>,
    pub packed: ConstraintFoldPackedCols<T>,
}

pub const NUM_CONSTRAINT_FOLD_PRECOMPUTED_COLS: usize =
    ConstraintFoldPrecomputedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBetaLadderCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub is_seed: T,
    pub is_last: T,
    pub power_idx: T,

    pub beta: [T; D_EF],
    // On the seed row this backs the alpha relay; on non-seed rows it backs the
    // previous ladder power. The alias keeps the new ladder at the intended 24 cols.
    pub prev_power_or_alpha: [T; D_EF],
    pub power: [T; D_EF],

    pub serve_mult: T,
    pub challenges_recv_mult: T,
    pub alpha_serve_mult: T,
    pub septix_serve_mult: T,
}

pub const NUM_CONSTRAINT_BETA_LADDER_COLS: usize = ConstraintBetaLadderCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintChallengeCols<T> {
    pub proof_idx: T,
    pub is_valid: T,

    pub chip_idx: T,
    pub static_chip_id: T,
    pub main_width: T,
    pub log_height: T,
    pub c_chips: T,

    pub lcs_limbs: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],

    pub selector_eq_acc: [T; D_EF],
    pub selector_first: [T; D_EF],
    pub selector_last: [T; D_EF],
    pub selector_first_send_mult: T,
    pub selector_last_send_mult: T,
}

pub const NUM_CONSTRAINT_CHALLENGE_COLS: usize = ConstraintChallengeCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintChallengeReservedCols<T> {
    pub is_valid: T,
    pub selector_first_send_mult: T,
    pub selector_last_send_mult: T,
}

pub const NUM_CONSTRAINT_CHALLENGE_RESERVED_COLS: usize =
    ConstraintChallengeReservedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintChallengeDenominatorCols<T> {
    pub lcs_events: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub eq_chain: T,
    pub lcs: T,
    pub is_first: T,
    pub is_last: T,
    pub batch_dim_main: T,
    pub fold_plan_source: T,
}

pub const NUM_CONSTRAINT_CHALLENGE_DENOMINATOR_COLS: usize =
    ConstraintChallengeDenominatorCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintChallengePrecomputedCols<T> {
    pub denominators: ConstraintChallengeDenominatorCols<T>,
}

pub const NUM_CONSTRAINT_CHALLENGE_PRECOMPUTED_COLS: usize =
    ConstraintChallengePrecomputedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalCols<T> {
    pub proof_idx: T,
    pub is_seed: T,
    pub is_eq_step: T,
    pub is_lcs_step: T,
    pub is_final: T,

    pub num_rounds: T,
    pub c_chips: T,
    pub round_idx: T,
    pub opening_idx: T,
    pub chip_idx: T,

    pub opening_point: [T; D_EF],
    pub eq_challenge: [T; D_EF],
    pub eq_factor: [T; D_EF],
    pub eq_in: [T; D_EF],
    pub eq_out: [T; D_EF],
    pub first_prefix_in: [T; D_EF],
    pub first_prefix_out: [T; D_EF],
    pub last_prefix_in: [T; D_EF],
    pub last_prefix_out: [T; D_EF],

    pub fold_cursor: T,
    pub alpha: [T; D_EF],
    pub main_eval: [T; D_EF],
    pub perm_eval: [T; D_EF],
    pub last_claim: [T; D_EF],

    pub lcs: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_in: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_out: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub public_values: [T; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub state_chain_send_mult: T,
    pub perm_alpha: [T; D_EF],
    pub beta_powers: [[T; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    pub state_clock_changed: T,
    pub state_clock_delta_inverse: T,
    pub state_transition_recv_inverse: [T; D_EF],
    pub state_transition_send_inverse: [T; D_EF],
    pub init_address_recv_inverse: [T; D_EF],
    pub init_address_send_inverse: [T; D_EF],
    pub finalize_address_recv_inverse: [T; D_EF],
    pub finalize_address_send_inverse: [T; D_EF],
    pub global_chain_source_inverse: [T; D_EF],
    pub global_chain_sink_inverse: [T; D_EF],

    pub summary_id_base: T,
    pub eq_chain_send_mult: T,
}

pub const NUM_CONSTRAINT_TERMINAL_COLS: usize = ConstraintTerminalCols::<u8>::width();

/// The cgb=false (Reduce-role) Terminal commits only the live sumcheck replay
/// and local cumulative-sum thread. Its trace writer fills this layout
/// directly; the wide curve and state-imbalance witnesses are never built.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalColsNarrow<T> {
    pub proof_idx: T,
    pub is_seed: T,
    pub is_eq_step: T,
    pub is_lcs_step: T,
    pub is_final: T,
    pub num_rounds: T,
    pub c_chips: T,
    pub round_idx: T,
    pub opening_idx: T,
    pub chip_idx: T,
    pub opening_point: [T; D_EF],
    pub eq_challenge: [T; D_EF],
    pub eq_factor: [T; D_EF],
    pub eq_in: [T; D_EF],
    pub eq_out: [T; D_EF],
    pub first_prefix_in: [T; D_EF],
    pub first_prefix_out: [T; D_EF],
    pub last_prefix_in: [T; D_EF],
    pub last_prefix_out: [T; D_EF],
    pub fold_cursor: T,
    pub alpha: [T; D_EF],
    pub main_eval: [T; D_EF],
    pub perm_eval: [T; D_EF],
    pub last_claim: [T; D_EF],
    pub lcs: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_in: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_out: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_chain_send_mult: T,
    pub summary_id_base: T,
    pub eq_chain_send_mult: T,
}

pub const NUM_CONSTRAINT_TERMINAL_NARROW_COLS: usize = ConstraintTerminalColsNarrow::<u8>::width();

/// Compact Core Terminal row retained for constraint evaluation and lookup multiplicities.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalReservedWideCols<T> {
    pub is_seed: T,
    pub is_eq_step: T,
    pub is_lcs_step: T,
    pub is_final: T,
    pub public_values: [T; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub state_chain_send_mult: T,
    pub state_clock_changed: T,
    pub state_clock_delta_inverse: T,
    pub eq_chain_send_mult: T,
}

pub const NUM_CONSTRAINT_TERMINAL_RESERVED_WIDE_COLS: usize =
    ConstraintTerminalReservedWideCols::<u8>::width();

/// Compact Reduce-role Terminal row retained for constraint evaluation and lookup multiplicities.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalReservedNarrowCols<T> {
    pub is_seed: T,
    pub is_eq_step: T,
    pub is_lcs_step: T,
    pub is_final: T,
    pub state_chain_send_mult: T,
    pub eq_chain_send_mult: T,
}

pub const NUM_CONSTRAINT_TERMINAL_RESERVED_NARROW_COLS: usize =
    ConstraintTerminalReservedNarrowCols::<u8>::width();

/// Degree-one extension packs retained after the unchanged Terminal lookup
/// denominator prefix. The order is shared by wide and narrow Terminal.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalPackedCommonCols<T> {
    pub opening_point: T,
    pub eq_challenge: T,
    pub eq_factor: T,
    pub eq_in: T,
    pub eq_out: T,
    pub first_prefix_in: T,
    pub first_prefix_out: T,
    pub last_prefix_in: T,
    pub last_prefix_out: T,
    pub main_eval: T,
    pub last_claim_minus_perm_eval: T,
    pub lcs: T,
    pub state_lcs_in: T,
    pub state_lcs_out: T,
}

pub const NUM_CONSTRAINT_TERMINAL_PACKED_COMMON_COLS: usize =
    ConstraintTerminalPackedCommonCols::<u8>::width();

/// Wide-only degree-one packs. Packing alpha/beta operands rather than the
/// derived fingerprints keeps every retained expression linear.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalPackedStateCols<T> {
    pub perm_alpha: T,
    pub beta_powers: [T; CONSTRAINT_CHAIN_LIMBS],
    pub state_transition_recv_inverse: T,
    pub state_transition_send_inverse: T,
    pub init_address_recv_inverse: T,
    pub init_address_send_inverse: T,
    pub finalize_address_recv_inverse: T,
    pub finalize_address_send_inverse: T,
    pub global_chain_source_inverse: T,
    pub global_chain_sink_inverse: T,
}

pub const NUM_CONSTRAINT_TERMINAL_PACKED_STATE_COLS: usize =
    ConstraintTerminalPackedStateCols::<u8>::width();

/// Denominators common to the wide and narrow Terminal layouts.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalOuterDenominators<T> {
    pub summary: T,
    pub opening_point: T,
    pub eq: T,
    pub fold_plan_chain: T,
    pub last_claim: T,
    pub fold_chain: T,
    pub eq_chain_recv: T,
    pub eq_chain_send: T,
}

/// Per-chip local-sum denominator shared by both Terminal layouts.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalLcsDenominator<T> {
    pub lcs: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalDenominatorsWideCols<T> {
    pub outer: ConstraintTerminalOuterDenominators<T>,
    pub public_values: [T; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub perm_alpha: T,
    pub beta_powers: [T; CONSTRAINT_CHAIN_LIMBS],
    pub lcs: ConstraintTerminalLcsDenominator<T>,
    pub state_lcs_in: T,
    pub state_lcs_out: T,
}

pub const NUM_CONSTRAINT_TERMINAL_DENOMINATORS_WIDE_COLS: usize =
    ConstraintTerminalDenominatorsWideCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalDenominatorsNarrowCols<T> {
    pub outer: ConstraintTerminalOuterDenominators<T>,
    pub lcs: ConstraintTerminalLcsDenominator<T>,
    pub state_lcs_in: T,
    pub state_lcs_out: T,
}

pub const NUM_CONSTRAINT_TERMINAL_DENOMINATORS_NARROW_COLS: usize =
    ConstraintTerminalDenominatorsNarrowCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalPrecomputedWideCols<T> {
    pub denominators: ConstraintTerminalDenominatorsWideCols<T>,
    pub common: ConstraintTerminalPackedCommonCols<T>,
    pub state: ConstraintTerminalPackedStateCols<T>,
}

pub const NUM_CONSTRAINT_TERMINAL_PRECOMPUTED_WIDE_COLS: usize =
    ConstraintTerminalPrecomputedWideCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintTerminalPrecomputedNarrowCols<T> {
    pub denominators: ConstraintTerminalDenominatorsNarrowCols<T>,
    pub common: ConstraintTerminalPackedCommonCols<T>,
}

pub const NUM_CONSTRAINT_TERMINAL_PRECOMPUTED_NARROW_COLS: usize =
    ConstraintTerminalPrecomputedNarrowCols::<u8>::width();

/// One Core-child row that owns the state/global boundary equation formerly
/// repeated across every row of the wide ConstraintTerminal trace.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBoundaryCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub c_chips: T,
    pub state_lcs: [T; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub public_values: [T; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub perm_alpha: [T; D_EF],
    pub beta_powers: [[T; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    pub state_clock_changed: T,
    pub state_clock_delta_inverse: T,
    pub state_transition_recv_inverse: [T; D_EF],
    pub state_transition_send_inverse: [T; D_EF],
    pub init_address_recv_inverse: [T; D_EF],
    pub init_address_send_inverse: [T; D_EF],
    pub finalize_address_recv_inverse: [T; D_EF],
    pub finalize_address_send_inverse: [T; D_EF],
    pub global_chain_source_inverse: [T; D_EF],
    pub global_chain_sink_inverse: [T; D_EF],
}

pub const NUM_CONSTRAINT_BOUNDARY_COLS: usize = ConstraintBoundaryCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBoundaryReservedCols<T> {
    pub is_valid: T,
    pub public_values: [T; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub state_clock_changed: T,
    pub state_clock_delta_inverse: T,
}

pub const NUM_CONSTRAINT_BOUNDARY_RESERVED_COLS: usize =
    ConstraintBoundaryReservedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBoundaryDenominatorCols<T> {
    pub public_values: [T; CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT],
    pub global_packed: [T; CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS],
    pub perm_alpha: T,
    pub beta_powers: [T; CONSTRAINT_CHAIN_LIMBS],
    pub state_lcs: T,
}

pub const NUM_CONSTRAINT_BOUNDARY_DENOMINATOR_COLS: usize =
    ConstraintBoundaryDenominatorCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBoundaryPackedCols<T> {
    pub state_lcs: T,
    pub state: ConstraintTerminalPackedStateCols<T>,
}

pub const NUM_CONSTRAINT_BOUNDARY_PACKED_COLS: usize =
    ConstraintBoundaryPackedCols::<u8>::width();

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintBoundaryPrecomputedCols<T> {
    pub denominators: ConstraintBoundaryDenominatorCols<T>,
    pub packed: ConstraintBoundaryPackedCols<T>,
}

pub const NUM_CONSTRAINT_BOUNDARY_PRECOMPUTED_COLS: usize =
    ConstraintBoundaryPrecomputedCols::<u8>::width();
