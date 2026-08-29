use core::fmt::{Debug, Display, Formatter};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriConfig, PrunedFriQueryProof, QueryProof};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use serde::{Deserialize, Serialize};

use crate::whir::sumcheck::SumcheckInstanceProof;
use crate::whir::whir_helpers::StackedBatchLayout;

fn path_pruning_from_env() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("DT_USE_PATH_PRUNING").unwrap_or_default() == "1"
    }
}

pub(crate) type CoefficientsByHeight<EF> = BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>>;
pub(crate) type DimGroupsByLogHeight<'a, EF> = BTreeMap<usize, Vec<(&'a DimAndNo, &'a Vec<EF>)>>;
pub(crate) type MatrixGroupsByLogHeight<'a, F, EF> =
    BTreeMap<usize, Vec<(&'a CompressedMatrix<F>, &'a Vec<EF>)>>;
pub(crate) type SharedMmcsProverData<F, InputMmcs> =
    Arc<<InputMmcs as Mmcs<F>>::ProverData<RowMajorMatrix<F>>>;
pub(crate) type StackedProverDataParts<F, InputMmcs> =
    (SharedMmcsProverData<F, InputMmcs>, StackedCommitmentData<F>);

/// Compact arithmetic facts emitted by the successful WHIR verifier pass.
/// These values are semantic sources, not AIR rows or padded matrices.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerificationTrace<EF, InputMerkleTrace = (), IoppMerkleTrace = ()> {
    /// Stacked verification remains supported by the ordinary verifier, but is
    /// intentionally outside native-recursion's first-pass capture contract.
    pub stacked: bool,
    pub alpha: Option<EF>,
    pub batch_steps: Vec<WhirVerifiedBatchStep<EF>>,
    pub groups: Vec<WhirVerifiedGroup<EF>>,
    pub rounds: Vec<WhirVerifiedRound<EF>>,
    pub combined_eq_sum: Option<EF>,
    pub combined_f_r: Option<EF>,
    pub queries: Vec<WhirVerifiedQuery<EF, InputMerkleTrace, IoppMerkleTrace>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedBatchStep<EF> {
    pub log_height: usize,
    pub batch_idx: usize,
    pub matrix_idx: usize,
    pub value_idx: usize,
    pub value: EF,
    pub coefficient: EF,
    pub coefficient_out: EF,
    pub accumulator_in: EF,
    pub accumulator_out: EF,
    pub group_accumulator_in: EF,
    pub group_accumulator_out: EF,
    pub is_group_start: bool,
    pub is_group_end: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedGroup<EF> {
    pub log_height: usize,
    pub claim: EF,
    pub first_step: usize,
    pub step_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedRound<EF> {
    pub round: usize,
    pub claim_in: EF,
    pub coefficients: Vec<EF>,
    pub r_fold: EF,
    pub claim_acc: EF,
    pub claim_folded: EF,
    pub eq_in: EF,
    pub eq_factor: EF,
    pub eq_folded: EF,
    pub merge_height: Option<usize>,
    pub merge_beta: Option<EF>,
    pub branch_claim: Option<EF>,
    pub claim_out: EF,
    pub eq_out: EF,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedQuery<EF, InputMerkleTrace = (), IoppMerkleTrace = ()> {
    pub query_idx: usize,
    pub query_point: usize,
    pub leaf_steps: Vec<WhirVerifiedLeafStep<EF>>,
    pub leaf_sums_by_log_height: BTreeMap<usize, EF>,
    pub fold_steps: Vec<WhirVerifiedQueryFoldStep<EF>>,
    pub final_value: EF,
    /// Exact MMCS operations captured by the successful input-opening verification, in batch
    /// order. These are compact input/output facts, not Merkle AIR rows.
    pub input_merkle: Vec<InputMerkleTrace>,
    /// Exact MMCS operations captured by the successful committed-IOPP verification, in round
    /// order.
    pub iopp_merkle: Vec<IoppMerkleTrace>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedLeafStep<EF> {
    pub log_height: usize,
    pub batch_idx: usize,
    pub matrix_idx: usize,
    pub value_idx: usize,
    pub value: EF,
    pub coefficient: EF,
    pub coefficient_out: EF,
    pub accumulator_in: EF,
    pub accumulator_out: EF,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirVerifiedQueryFoldStep<EF> {
    pub round: usize,
    pub query_point_in: usize,
    pub query_point_out: usize,
    pub pair: [EF; 2],
    pub generator: EF,
    pub folding_challenge: EF,
    pub folded_before_merge: EF,
    pub folded_value_in: EF,
    pub folded_value_out: EF,
    pub eq_before_merge: EF,
    pub eq_in: EF,
    pub eq_out: EF,
    pub merged_leaf: Option<(usize, EF, Option<EF>)>,
}

mod arc_serde {
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    pub(crate) fn serialize<T, S>(value: &Arc<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: serde::Serializer,
    {
        value.as_ref().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, T, D>(deserializer: D) -> Result<Arc<T>, D::Error>
    where
        T: Deserialize<'de>,
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Arc::new)
    }
}

mod option_arc_serde {
    use std::sync::Arc;

    use serde::{Deserialize, Serialize};

    pub(crate) fn serialize<T, S>(value: &Option<Arc<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: serde::Serializer,
    {
        value.as_deref().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, T, D>(deserializer: D) -> Result<Option<Arc<T>>, D::Error>
    where
        T: Deserialize<'de>,
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| value.map(Arc::new))
    }
}

#[derive(Debug)]
pub enum WhirError<CommitMmcsErr, InputError> {
    CommitPhaseMmcsError(CommitMmcsErr),
    CommitmentCheckFailed,
    SumcheckPhaseError,
    FinalPolyMismatch,
    CannotFindPowWitness,
    InvalidPowWitness,
    InvalidInputError,
    FriFinalStepMisMatch,
    _PhantomInputError(PhantomData<InputError>),
}

/// Batch opening proof for the WHIR PCS.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct WhirProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    /// `Some(log_height)` when the proof opens commit-local stacked batches.
    /// `None` keeps the legacy unstacked opening path.
    #[serde(default)]
    pub stack_log_height: Option<usize>,
    pub sumcheck_transcript: SumcheckInstanceProof<F>,
    pub iopp_oracles: Vec<M::Commitment>,
    /// Out-of-domain evaluations sampled after each non-final WHIR round.
    /// Empty on the legacy global-query path.
    #[serde(default)]
    pub ood_values: Vec<F>,
    pub iopp_queries: Vec<QueryProof<F, M>>,
    /// Optional per-round IOPP query payload.
    ///
    /// When `Some`, each committed IOPP group has its own query phase and
    /// query count. This is the first WHIR-style structural step: query
    /// payloads are grouped by committed round instead of by one global FRI
    /// path. In the stacked path this also enables OOD sampling and gamma
    /// accumulation between WHIR rounds.
    #[serde(default)]
    pub round_iopp: Option<WhirRoundQueryProof<F, M, Witness>>,
    /// Input PCS batch openings. Type-erased: callers fix this via the
    /// `InputProof` type parameter (typically [`WhirInputProof`]).
    pub query_openings: InputProof,
    /// Proof-of-work witness for the batching phase `[nonce, witness]`.
    pub grinding_batching_witness: Vec<Witness>,
    /// Proof-of-work witness for the query phase `[nonce, witness]`.
    pub grinding_query_witness: Vec<Witness>,
    /// Final polynomial coefficients for FRI early stopping.
    /// Empty when `log_final_poly_len = 0`.
    pub final_poly: Vec<F>,
    /// B-Stage 1' path-pruning: standard arity-2 IOPP queries with shared
    /// per-round merkle paths. When `Some`, `iopp_queries` is empty and
    /// the verifier walks `verify_challenges_pruned`. Backward-compat:
    /// `#[serde(default)]` so old proofs (pre B-Stage 1') deserialize fine.
    #[serde(default)]
    pub iopp_pruned: Option<PrunedFriQueryProof<F, M>>,
    /// Stacking reduction proof. Present when the stacked path uses the
    /// fresh-y eval-unification sumcheck (soundness fix: avoids reusing
    /// sumcheck random challenges z as selector coefficients).
    ///
    /// Contains the batched eval-unification sumcheck transcript and the
    /// per-stacked-column evaluations F_c(u) at the unified random point u
    /// produced by that sumcheck.
    #[serde(default)]
    pub stacking_reduction: Option<StackingReductionProof<F>>,
}

/// Per-round IOPP query proof for WHIR-style openings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>"
))]
pub struct WhirRoundQueryProof<F: Field, M: Mmcs<F>, Witness> {
    /// `rounds[r]` contains all query openings for committed IOPP group `r`.
    /// Empty when `pruned.is_some()`.
    pub rounds: Vec<WhirIoppRound<F, M>>,
    /// Path-pruned per-round IOPP openings. Each committed group keeps its
    /// own query set, unique opened rows, and merged Merkle proof.
    #[serde(default = "default_none_whir_round_pruned")]
    pub pruned: Option<WhirRoundPrunedQueryProof<F, M>>,
    /// One query-phase proof-of-work witness per committed IOPP group.
    /// Each entry has shape `[nonce, witness]`.
    pub query_witnesses: Vec<Vec<Witness>>,
    /// Per-round folding proof-of-work witnesses. `folding_witnesses[r]` is a
    /// flattened list `[nonce_0, witness_0, nonce_1, witness_1, ...]` with one
    /// `[nonce, witness]` pair after each fold inside committed IOPP group `r`.
    /// Empty (len 0) when `grinding_bits_folding == 0`.
    #[serde(default)]
    pub folding_witnesses: Vec<Vec<Witness>>,
}

fn default_none_whir_round_pruned<F: Field, M: Mmcs<F>>() -> Option<WhirRoundPrunedQueryProof<F, M>>
{
    None
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WhirIoppRound<F: Field, M: Mmcs<F>> {
    pub query_proofs: Vec<WhirIoppRoundQuery<F, M>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WhirRoundPrunedQueryProof<F: Field, M: Mmcs<F>> {
    /// `rounds[r]` authenticates the unique opened rows for committed IOPP
    /// group `r`.
    pub rounds: Vec<WhirPrunedIoppRound<F, M>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WhirPrunedIoppRound<F: Field, M: Mmcs<F>> {
    /// One merged Merkle proof for the sorted+deduped row indices of this
    /// committed group.
    pub pruned_proof: M::PrunedProof,
    /// Unique opened rows in the same sorted+deduped order as
    /// `pruned_proof`. Shape: `[unique_slot][matrix_idx=0][row_values]`.
    pub opened_rows: Vec<Vec<Vec<F>>>,
    /// Query index -> unique row slot in `opened_rows`.
    pub query_to_unique_slot: Vec<u32>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WhirIoppRoundQuery<F: Field, M: Mmcs<F>> {
    /// Opening against the current committed IOPP group. Unlike legacy
    /// arity-2 FRI paths, this always carries the full opened row in
    /// `opened_values`, because per-round queries may not inherit the
    /// current value from a previous round.
    pub current_opening: CommitPhaseProofStep<F, M>,
    /// Opening against the next committed IOPP group at the folded index.
    /// `None` for the last committed group, where the folded value is checked
    /// against the final polynomial/constant.
    pub next_opening: Option<CommitPhaseProofStep<F, M>>,
}

/// Stacking reduction proof: a single batched sumcheck that safely binds the
/// multiple original opening claims sharing one stacked column.
///
/// The verifier samples `λ` after absorbing the original `opened_values`, then
/// the prover proves the identity
///
/// ```text
/// T = Σ_{x ∈ {0,1}^L} Σ_c F_c(x) · Q_c(x)
/// ```
///
/// where `F_c` is stacked column `c`, `Q_c` is the coefficient polynomial the
/// verifier reconstructs from the stacking layout / original opening point /
/// `λ` powers (each source contributes `λ^i · selector_eq(slot) ·
/// eq(x_prefix; z_prefix)`), and `T = Σ_i λ^i · original_claim_i`.
///
/// The sumcheck yields a random point `u`; both parties compute `q_c = Q_c(u)`
/// independently, so the proof only needs to carry the sumcheck transcript.
/// The WHIR opening then proves `combined_evals = Σ_c q_c · F_c` at point `u`
/// with running claim equal to the sumcheck final claim.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct StackingReductionProof<F: Field> {
    /// Stacking reduction sumcheck transcript (L rounds, degree 2).
    pub sumcheck: SumcheckInstanceProof<F>,
}

/// Input-opening payload for [`WhirProof`].
///
/// Carries either standard per-query openings or per-round path-pruned
/// openings. It is paired with [`WhirProof::iopp_pruned`]: pruned input
/// openings are present exactly when pruned IOPP openings are present.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct WhirInputProof<F: Field, InputMmcs: Mmcs<F>> {
    /// Standard per-query openings: `per_query[q][r]` holds the merkle
    /// proof for query `q` against round `r`'s input MMCS. Empty when
    /// `pruned.is_some()`.
    pub per_query: Vec<Vec<BatchOpening<F, InputMmcs>>>,
    /// Path-pruned openings for the PCS opening phase. When `Some`, the
    /// verifier walks per-round `verify_batch_pruned` over the BFS-merged
    /// proofs and ignores `per_query`.
    #[serde(default = "default_none_pruned_whir_input")]
    pub pruned: Option<PrunedQueryOpenings<F, InputMmcs>>,
}

/// Default helper for [`WhirInputProof::pruned`] so pre-D6 proofs
/// deserialize as `None`.
fn default_none_pruned_whir_input<F: Field, InputMmcs: Mmcs<F>>(
) -> Option<PrunedQueryOpenings<F, InputMmcs>> {
    None
}

impl<F: Field, InputMmcs: Mmcs<F>> WhirInputProof<F, InputMmcs> {
    /// Wrap a plain per-query openings vector (env=0 / pre-D6 path).
    pub fn from_per_query(per_query: Vec<Vec<BatchOpening<F, InputMmcs>>>) -> Self {
        Self {
            per_query,
            pruned: None,
        }
    }
}

/// Path-pruned PCS openings with one merged Merkle proof per committed input batch.
///
/// Mirrors [`PrunedFriQueryProof`] for the PCS opening phase
/// (R batches x Q queries -> R [`Mmcs::PrunedProof`] values).
///
/// Note: `M` is the **input** MMCS (`InputMmcs`) and `F` is the **base**
/// field (input/leaf field), not the extension field. Each round still has
/// the same `Q` queries but a different `max_log_height` (per-batch).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrunedQueryOpenings<F: Field, M: Mmcs<F>> {
    /// Per-round pruned proof. `round_pruned[r]` carries Q query paths merged
    /// by BFS schedule for round r's input MMCS. Length = num_rounds.
    pub round_pruned: Vec<<M as Mmcs<F>>::PrunedProof>,
    /// Per-round per-query opened values. Mirrors the old layout
    /// `query_openings[q][r].opened_values` but transposed:
    /// `round_opened_values[r][q]: Vec<Vec<F>>` (one inner Vec per matrix
    /// in batch r). Required because `verify_batch_pruned` consumes leaf
    /// values separately from the pruned merkle proof.
    pub round_opened_values: Vec<Vec<Vec<Vec<F>>>>,
    /// Per-round query->unique-slot hint. `query_to_unique_slot[r][q]` is the
    /// slot index in the BFS-merged unique-leaves array that query q falls
    /// into for round r. Used by the circuit-side BFS layer-walk gadget.
    pub query_to_unique_slot: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirRoundQueryConfig {
    pub num_queries: usize,
    pub grinding_bits_query: usize,
    pub grinding_bits_folding: usize,
}

impl WhirRoundQueryConfig {
    pub const fn new(num_queries: usize, grinding_bits_query: usize) -> Self {
        Self {
            num_queries,
            grinding_bits_query,
            grinding_bits_folding: 0,
        }
    }

    pub const fn with_folding_pow(
        num_queries: usize,
        grinding_bits_query: usize,
        grinding_bits_folding: usize,
    ) -> Self {
        Self {
            num_queries,
            grinding_bits_query,
            grinding_bits_folding,
        }
    }
}

/// A single committed IOPP round, counted down from `num_vars`.
#[derive(Debug, Clone, Copy)]
pub struct CommitGroup {
    pub start_round: usize,
    pub log_folding: usize,
}

/// A committed WHIR round with its Reed-Solomon domain parameters.
///
/// `start_round`/`log_folding` describe how many multilinear variables are
/// folded by sumcheck. `codeword_log` describes the committed RS domain for
/// this round. In reduced-rate WHIR the polynomial loses `log_folding`
/// variables, while the RS domain only loses one bit per committed round.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WhirRoundSchedule {
    pub(crate) start_round: usize,
    pub(crate) log_folding: usize,
    pub(crate) codeword_log: usize,
    pub(crate) log_blowup: usize,
}

impl WhirRoundSchedule {
    pub(crate) const fn poly_log_after(self) -> usize {
        self.start_round - self.log_folding
    }

    pub(crate) const fn row_log_height(self) -> usize {
        self.codeword_log - self.log_folding
    }
}

/// Generates a uniform cross-round folding schedule for `num_groups` committed
/// IOPP groups that together cover `active_rounds` sumcheck variables.
///
/// Each group gets `active_rounds / num_groups` rounds, with the first
/// `active_rounds % num_groups` groups receiving one extra round.
///
/// # Examples
///
/// ```text
/// compute_uniform_log_foldings(14, 4) => [4, 4, 3, 3]   // 14 = 4+4+3+3
/// compute_uniform_log_foldings(16, 3) => [6, 5, 5]       // 16 = 6+5+5
/// compute_uniform_log_foldings(12, 3) => [4, 4, 4]       // 12 = 4+4+4
/// ```
pub fn compute_uniform_log_foldings(active_rounds: usize, num_groups: usize) -> Vec<usize> {
    if num_groups == 0 || active_rounds == 0 {
        return Vec::new();
    }
    let base = active_rounds / num_groups;
    let remainder = active_rounds % num_groups;
    (0..num_groups)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

/// Commit one round for each fold before early stopping.
///
/// `k = log_final_poly_len` means the committed rounds are `num_vars..k+1`.
/// When `k = 0`, the final constant commitment is handled separately to
/// preserve the original proof layout.
pub fn compute_commit_schedule(num_vars: usize, k: usize) -> Vec<CommitGroup> {
    compute_commit_schedule_with_log_foldings(num_vars, k, &[])
}

pub fn compute_commit_schedule_with_log_foldings(
    num_vars: usize,
    k: usize,
    log_foldings: &[usize],
) -> Vec<CommitGroup> {
    let active_rounds = num_vars.saturating_sub(k);
    let mut remaining = active_rounds;
    let mut start_round = num_vars;
    let mut groups = Vec::new();

    for &requested in log_foldings {
        if remaining == 0 {
            break;
        }
        if requested == 0 {
            continue;
        }
        let log_folding = requested.min(remaining);
        groups.push(CommitGroup {
            start_round,
            log_folding,
        });
        start_round -= log_folding;
        remaining -= log_folding;
    }

    while remaining > 0 {
        groups.push(CommitGroup {
            start_round,
            log_folding: 1,
        });
        start_round -= 1;
        remaining -= 1;
    }
    groups
}

pub(crate) fn compute_reduced_rate_commit_schedule(
    num_vars: usize,
    k: usize,
    initial_log_blowup: usize,
    log_foldings: &[usize],
) -> Option<Vec<WhirRoundSchedule>> {
    let initial_codeword_log = num_vars.checked_add(initial_log_blowup)?;
    compute_commit_schedule_with_log_foldings(num_vars, k, log_foldings)
        .into_iter()
        .enumerate()
        .map(|(round_idx, group)| {
            let codeword_log = initial_codeword_log.checked_sub(round_idx)?;
            if group.start_round > codeword_log
                || group.log_folding == 0
                || group.log_folding > codeword_log
            {
                return None;
            }
            Some(WhirRoundSchedule {
                start_round: group.start_round,
                log_folding: group.log_folding,
                codeword_log,
                log_blowup: codeword_log - group.start_round,
            })
        })
        .collect()
}

pub(crate) fn compute_constant_rate_commit_schedule(
    num_vars: usize,
    k: usize,
    log_blowup: usize,
    log_foldings: &[usize],
) -> Option<Vec<WhirRoundSchedule>> {
    compute_commit_schedule_with_log_foldings(num_vars, k, log_foldings)
        .into_iter()
        .map(|group| {
            let codeword_log = group.start_round.checked_add(log_blowup)?;
            Some(WhirRoundSchedule {
                start_round: group.start_round,
                log_folding: group.log_folding,
                codeword_log,
                log_blowup,
            })
        })
        .collect()
}

/// A matrix's dimensions paired with its global index across all batches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DimAndNo {
    /// The matrix dimensions (height, width).
    pub dim: Dimensions,
    /// The global index of this matrix (flattened across all batches).
    pub num: usize,
}

impl Debug for DimAndNo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "dim: {:?}, No: {}", self.dim, self.num)
    }
}

impl Display for DimAndNo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "dim: {:?}, No: {}", self.dim, self.num)
    }
}

#[derive(Debug)]
pub struct WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger> {
    pub(crate) mmcs: InputMmcs,
    pub config: WhirConfig<FriMmcs>,
    _phantom: PhantomData<(F, EF, Challenger)>,
}

/// Complete parameter set for the WHIR multilinear PCS.
///
/// `fri.log_final_poly_len` is the early-stop parameter: the IOPP stops once
/// the current polynomial has `2^log_final_poly_len` evaluations, while
/// sumcheck continues to bind the remaining variables.
///
/// `fri.cross_round_log_foldings` is the sparse IOPP commit schedule in the
/// stacked WHIR path.
#[derive(Debug)]
pub struct WhirConfig<FriMmcs> {
    pub fri: FriConfig<FriMmcs>,
    pub path_pruning: bool,
    pub reduced_rate: bool,
    /// Optional WHIR per-round query schedule.
    ///
    /// When absent, WHIR uses the legacy single query phase controlled by
    /// `fri.num_queries` and `fri.grinding_bits_query`. When present, each
    /// committed IOPP group gets an independent query phase. If the vector is
    /// shorter than the number of committed groups, the remaining groups fall
    /// back to the legacy query parameters; extra entries are ignored.
    pub round_queries: Option<Vec<WhirRoundQueryConfig>>,
}

impl<FriMmcs> WhirConfig<FriMmcs> {
    pub const fn new(fri: FriConfig<FriMmcs>) -> Self {
        Self {
            fri,
            path_pruning: false,
            reduced_rate: true,
            round_queries: None,
        }
    }

    pub fn with_cross_round_log_foldings(mut self, log_foldings: Vec<usize>) -> Self {
        assert!(
            log_foldings.iter().all(|&log_folding| log_folding > 0),
            "cross-round log foldings must be positive"
        );
        self.fri.cross_round_log_foldings = log_foldings;
        self
    }

    pub fn with_path_pruning(mut self, enabled: bool) -> Self {
        self.path_pruning = enabled;
        self
    }

    pub fn with_reduced_rate(mut self, enabled: bool) -> Self {
        self.reduced_rate = enabled;
        self
    }

    pub fn with_path_pruning_from_env(self) -> Self {
        self.with_path_pruning(path_pruning_from_env())
    }

    pub fn with_log_final_poly_len(mut self, log_final_poly_len: usize) -> Self {
        self.fri.log_final_poly_len = log_final_poly_len;
        self
    }

    pub fn with_round_query_counts(mut self, num_queries: Vec<usize>) -> Self {
        assert!(
            num_queries.iter().all(|&num_queries| num_queries > 0),
            "per-round query counts must be positive"
        );
        let folding_bits = self.fri.grinding_bits_folding;
        self.round_queries = Some(
            num_queries
                .into_iter()
                .map(|num_queries| {
                    WhirRoundQueryConfig::with_folding_pow(
                        num_queries,
                        self.fri.grinding_bits_query,
                        folding_bits,
                    )
                })
                .collect(),
        );
        self
    }

    pub fn with_round_queries(mut self, round_queries: Vec<WhirRoundQueryConfig>) -> Self {
        assert!(
            round_queries.iter().all(|config| config.num_queries > 0),
            "per-round query counts must be positive"
        );
        self.round_queries = if round_queries.is_empty() {
            None
        } else {
            Some(round_queries)
        };
        self
    }

    pub(crate) fn round_query_configs(
        &self,
        num_rounds: usize,
    ) -> Option<Vec<WhirRoundQueryConfig>> {
        let configured = self.round_queries.as_ref()?;
        Some(
            (0..num_rounds)
                .map(|idx| {
                    configured.get(idx).copied().unwrap_or_else(|| {
                        WhirRoundQueryConfig::with_folding_pow(
                            self.fri.num_queries,
                            self.fri.grinding_bits_query,
                            self.fri.grinding_bits_folding,
                        )
                    })
                })
                .collect(),
        )
    }

    pub const fn fri(&self) -> &FriConfig<FriMmcs> {
        &self.fri
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "InputMmcs::ProverData<RowMajorMatrix<F>>: Serialize",
    deserialize = "InputMmcs::ProverData<RowMajorMatrix<F>>: serde::de::DeserializeOwned"
))]
pub struct WhirPcsProverData<F: Field, InputMmcs: Mmcs<F>> {
    #[serde(with = "arc_serde")]
    pub(crate) mmcs_prover_data: SharedMmcsProverData<F, InputMmcs>,
    pub(crate) stacked: Option<StackedCommitmentData<F>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct StackedCommitmentData<F: Field> {
    pub(crate) layout: StackedBatchLayout,
    #[serde(default, with = "option_arc_serde")]
    pub(crate) cached_evaluations: Option<Arc<RowMajorMatrix<F>>>,
}

impl<F: Field, InputMmcs: Mmcs<F>> WhirPcsProverData<F, InputMmcs> {
    pub(crate) fn unstacked(mmcs_prover_data: InputMmcs::ProverData<RowMajorMatrix<F>>) -> Self {
        Self {
            mmcs_prover_data: Arc::new(mmcs_prover_data),
            stacked: None,
        }
    }

    pub(crate) fn stacked(
        mmcs_prover_data: InputMmcs::ProverData<RowMajorMatrix<F>>,
        layout: StackedBatchLayout,
        cached_evaluations: Option<RowMajorMatrix<F>>,
    ) -> Self {
        Self {
            mmcs_prover_data: Arc::new(mmcs_prover_data),
            stacked: Some(StackedCommitmentData {
                layout,
                cached_evaluations: cached_evaluations.map(Arc::new),
            }),
        }
    }

    pub(crate) fn into_unstacked_mmcs(self) -> Result<SharedMmcsProverData<F, InputMmcs>, ()> {
        if self.stacked.is_some() {
            return Err(());
        }
        Ok(self.mmcs_prover_data)
    }

    pub(crate) fn into_stacked(self) -> Result<StackedProverDataParts<F, InputMmcs>, ()> {
        let Self {
            mmcs_prover_data,
            stacked,
        } = self;
        stacked.map(|stacked| (mmcs_prover_data, stacked)).ok_or(())
    }

    pub(crate) fn stacked_log_height(&self) -> Option<usize> {
        self.stacked
            .as_ref()
            .map(|stacked| stacked.layout.log_height)
    }
}

impl<Val, InputMmcs, FriMmcs, EF, Challenger> WhirPcs<Val, InputMmcs, FriMmcs, EF, Challenger> {
    pub fn new(mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self::from_config(mmcs, WhirConfig::new(fri).with_path_pruning_from_env())
    }

    pub const fn from_config(mmcs: InputMmcs, config: WhirConfig<FriMmcs>) -> Self {
        Self {
            mmcs,
            config,
            _phantom: PhantomData,
        }
    }

    pub fn with_cross_round_log_foldings(mut self, log_foldings: Vec<usize>) -> Self {
        self.config = self.config.with_cross_round_log_foldings(log_foldings);
        self
    }

    pub fn with_path_pruning(mut self, enabled: bool) -> Self {
        self.config.path_pruning = enabled;
        self
    }

    pub fn with_reduced_rate(mut self, enabled: bool) -> Self {
        self.config.reduced_rate = enabled;
        self
    }

    pub fn with_path_pruning_from_env(self) -> Self {
        self.with_path_pruning(path_pruning_from_env())
    }

    pub fn with_round_query_counts(mut self, num_queries: Vec<usize>) -> Self {
        self.config = self.config.with_round_query_counts(num_queries);
        self
    }

    pub fn with_round_queries(mut self, round_queries: Vec<WhirRoundQueryConfig>) -> Self {
        self.config = self.config.with_round_queries(round_queries);
        self
    }

    pub fn cross_round_log_foldings(&self) -> Vec<usize> {
        self.config.fri.cross_round_log_foldings.clone()
    }

    /// Blowup used by both host and device main-commit backends.
    pub const fn device_commit_log_blowup(&self) -> usize {
        self.config.fri.log_blowup
    }

    fn effective_log_foldings(&self, num_vars: usize, k: usize) -> Vec<usize> {
        if let Some(n) = self.config.fri.num_committed_groups {
            let active = num_vars.saturating_sub(k);
            compute_uniform_log_foldings(active, n)
        } else {
            self.config.fri.cross_round_log_foldings.clone()
        }
    }

    pub(crate) fn commit_schedule(&self, num_vars: usize, k: usize) -> Vec<CommitGroup> {
        let log_foldings = self.effective_log_foldings(num_vars, k);
        compute_commit_schedule_with_log_foldings(num_vars, k, &log_foldings)
    }

    pub(crate) fn whir_round_schedule(
        &self,
        num_vars: usize,
        k: usize,
    ) -> Option<Vec<WhirRoundSchedule>> {
        let log_foldings = self.effective_log_foldings(num_vars, k);
        if self.config.reduced_rate {
            compute_reduced_rate_commit_schedule(
                num_vars,
                k,
                self.config.fri.log_blowup,
                &log_foldings,
            )
        } else {
            compute_constant_rate_commit_schedule(
                num_vars,
                k,
                self.config.fri.log_blowup,
                &log_foldings,
            )
        }
    }

    pub(crate) fn cross_round_enabled(&self, num_vars: usize, k: usize) -> bool {
        self.commit_schedule(num_vars, k)
            .iter()
            .any(|group| group.log_folding > 1)
    }
}
