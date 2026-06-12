use crate::basefold::mlpcs::MlPCS;
use crate::basefold::sumcheck::SumcheckInstanceProof;
use crate::utils::eqpoly::EqPolynomial;
use crate::utils::math::{compute_dotproduct, compute_dotproduct_mix};
use crate::utils::mlpoly::MultilinearPolynomial;
use core::fmt::{Debug, Display, Formatter};
use itertools::izip;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::dft_eval::EvalsDft;
use p3_field::{ExtensionField, Field, Powers, TwoAdicField};
use p3_fri::prover::{answer_queries_pruned, answer_query};
use p3_fri::{BatchOpening, FriConfig, PrunedFriQueryProof, QueryProof};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::marker::PhantomData;

#[derive(Debug)]
pub enum BaseFoldError<CommitMmcsErr, InputError> {
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

/// Batch opening proof for the Basefold/WHIR PCS.
///
/// `out_of_domain_responses` is `Some(...)` for WHIR (out-of-domain sampling enabled)
/// and `None` for basefold (no out-of-domain sampling).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct BasefoldProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    pub sumcheck_transcript: SumcheckInstanceProof<F>,
    pub iopp_oracles: Vec<M::Commitment>,
    pub iopp_queries: Vec<QueryProof<F, M>>,
    /// Input PCS batch openings. Type-erased: callers fix this via the
    /// `InputProof` type parameter (typically [`BasefoldInputProof`]).
    pub query_openings: InputProof,
    /// Proof-of-work witness for the batching phase `[nonce, witness]`.
    pub grinding_batching_witness: Vec<Witness>,
    /// Proof-of-work witness for the query phase `[nonce, witness]`.
    pub grinding_query_witness: Vec<Witness>,
    /// `Some(responses)` for WHIR, `None` for basefold.
    pub out_of_domain_responses: Option<Vec<F>>,
    /// Final polynomial coefficients for FRI early stopping.
    /// Empty when `log_final_poly_len = 0`.
    pub final_poly: Vec<F>,
    /// B-Stage 1' path-pruning: standard arity-2 IOPP queries with shared
    /// per-round merkle paths. When `Some`, `iopp_queries` is empty and
    /// the verifier walks `verify_challenges_pruned`. Backward-compat:
    /// `#[serde(default)]` so old proofs (pre B-Stage 1') deserialize fine.
    #[serde(default)]
    pub iopp_pruned: Option<PrunedFriQueryProof<F, M>>,
    /// Cross-round IOPP queries. When non-empty, `iopp_queries` is empty and the
    /// verifier walks [`CrossRoundQueryProof`] group-by-group, folding each
    /// group's opened row locally. Mutually exclusive with `iopp_pruned`.
    /// Backward-compat: `#[serde(default)]` so pre-cross-round proofs deserialize
    /// as an empty vec.
    #[serde(default = "Vec::new")]
    pub iopp_cross_round: Vec<CrossRoundQueryProof<F, M>>,
    /// Group-wise path-pruned cross-round IOPP openings. When `Some`, cross-round
    /// (fewer/wider commit groups) and path-pruning (BFS-merged Merkle paths per
    /// group) are BOTH active: `iopp_queries`/`iopp_cross_round` are empty and
    /// `iopp_pruned` is `None`. Backward-compat: `#[serde(default)]`.
    #[serde(default)]
    pub iopp_cross_round_pruned: Option<CrossRoundPrunedOpenings<F, M>>,
}

/// D6 wrapper for the `InputProof` type parameter of [`BasefoldProof`].
///
/// Carries either the standard per-query openings (`per_query`, env=0) or
/// the per-round path-pruned openings (`pruned`, env=1). Strongly bound to
/// [`BasefoldProof::iopp_pruned`]: both `Some` together (env=1) or both
/// `None` together (env=0). Backward-compat: `pruned` defaults to `None`
/// for proofs serialized before D6.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BasefoldInputProof<F: Field, InputMmcs: Mmcs<F>> {
    /// Standard per-query openings: `per_query[q][r]` holds the merkle
    /// proof for query `q` against round `r`'s input MMCS. Empty when
    /// `pruned.is_some()`.
    pub per_query: Vec<Vec<BatchOpening<F, InputMmcs>>>,
    /// Path-pruned openings for the PCS opening phase. When `Some`, the
    /// verifier walks per-round `verify_batch_pruned` over the BFS-merged
    /// proofs and ignores `per_query`.
    #[serde(default = "default_none_pruned_input")]
    pub pruned: Option<PrunedQueryOpenings<F, InputMmcs>>,
}

/// Default helper for [`BasefoldInputProof::pruned`] so pre-D6 proofs
/// deserialize as `None`.
fn default_none_pruned_input<F: Field, InputMmcs: Mmcs<F>>(
) -> Option<PrunedQueryOpenings<F, InputMmcs>> {
    None
}

impl<F: Field, InputMmcs: Mmcs<F>> BasefoldInputProof<F, InputMmcs> {
    /// Wrap a plain per-query openings vector (env=0 / pre-D6 path).
    pub fn from_per_query(per_query: Vec<Vec<BatchOpening<F, InputMmcs>>>) -> Self {
        Self {
            per_query,
            pruned: None,
        }
    }
}

/// D6 PCS opening pruned proof: per-round BFS-merged merkle paths for the
/// input MMCS round openings. Mirrors [`PrunedFriQueryProof`] but for the
/// PCS opening phase (R rounds × Q queries → R [`Mmcs::PrunedProof`]s).
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
    /// Per-round query→unique-slot hint. `query_to_unique_slot[r][q]` is the
    /// slot index in the BFS-merged unique-leaves array that query q falls
    /// into for round r. Used by the circuit-side BFS layer-walk gadget.
    pub query_to_unique_slot: Vec<Vec<u32>>,
}

/// A single IOPP commitment group.
///
/// In the legacy (per-round) layout each group covers exactly one folding
/// round (`log_folding == 1`) and `start_log_height` is the log-height of the
/// codeword committed at that round (counted down from `num_vars`).
///
/// With cross-round commitment enabled, a group may span `log_folding > 1`
/// consecutive folding rounds: a single Merkle tree of row width `2^log_folding`
/// is committed, and the `log_folding` folds are performed locally by the
/// verifier from that one opened row. See [`compute_commit_schedule_cross_round`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitGroup {
    /// Log-height of the codeword committed at the start of this group, i.e.
    /// the height *before* this group's first fold. (Legacy name: `start_round`.)
    pub start_log_height: usize,
    /// Number of consecutive folding rounds merged into this group's single
    /// commitment. `1` reproduces the legacy per-round behaviour.
    pub log_folding: usize,
}

/// Commit one round for each fold before early stopping (legacy per-round layout).
///
/// `k = log_final_poly_len` means the committed rounds are `num_vars..k+1`.
/// When `k = 0`, the final constant commitment is handled separately to
/// preserve the original proof layout.
pub fn compute_commit_schedule(num_vars: usize, k: usize) -> Vec<CommitGroup> {
    let active_rounds = num_vars.saturating_sub(k);
    (0..active_rounds)
        .map(|i| CommitGroup {
            start_log_height: num_vars - i,
            log_folding: 1,
        })
        .collect()
}

/// Build the IOPP commitment schedule, optionally merging consecutive folding
/// rounds that have no matrix-merge boundary between them (cross-round).
///
/// `present_heights` is the set of log-heights at which input matrices are
/// injected (i.e. the keys of `matrices_by_log_height`). `num_vars` is the max
/// log-height and `k = log_final_poly_len.min(min_log_height)` is the
/// early-stop boundary (folding stops at height `k`).
///
/// When `use_cross_round == false` this reproduces [`compute_commit_schedule`]
/// exactly (every group has `log_folding == 1`).
///
/// When `use_cross_round == true`, starting from `num_vars` we greedily extend
/// each group downward until the next lower height that either (a) has a matrix
/// injected (a merge boundary — constraint 3) or (b) is the early-stop boundary
/// `k` (constraint 4). The group's `log_folding` is the number of rounds spanned.
pub fn compute_commit_schedule_cross_round(
    present_heights: &std::collections::BTreeSet<usize>,
    num_vars: usize,
    k: usize,
    use_cross_round: bool,
) -> Vec<CommitGroup> {
    compute_commit_schedule_cross_round_capped(
        present_heights,
        num_vars,
        k,
        use_cross_round,
        cross_round_max_log_folding(),
    )
}

/// Maximum `log_folding` per cross-round group.
///
/// Merging consecutive folding rounds trades Merkle path digests (fewer trees,
/// shallower paths) for wider opened rows (`2^log_folding` values per query).
/// Beyond a moderate width the row data dominates and proof size grows, so we
/// cap the merge depth. `0` means "unbounded" (greedy, merge whole gaps).
///
/// Default cap is `4` (empirically near the proof-size minimum for the shrink
/// layer's `{4,17,18,19}` height distribution). Override with
/// `DT_CROSS_ROUND_MAX_FOLD`.
pub fn cross_round_max_log_folding() -> usize {
    std::env::var("DT_CROSS_ROUND_MAX_FOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

/// Schedule builder with an explicit `max_log_folding` cap (`0` = unbounded).
///
/// A gap wider than the cap is split into several cap-sized groups, each a
/// separate commitment. Group boundaries introduced purely by the cap (i.e. at
/// heights with no matrix injected) are handled by the verifier's
/// "continuation" check (`row[local] == previous group's folded value`); they
/// are NOT merge boundaries.
pub fn compute_commit_schedule_cross_round_capped(
    present_heights: &std::collections::BTreeSet<usize>,
    num_vars: usize,
    k: usize,
    use_cross_round: bool,
    max_log_folding: usize,
) -> Vec<CommitGroup> {
    if !use_cross_round {
        return compute_commit_schedule(num_vars, k);
    }

    let mut groups = Vec::new();
    let mut h = num_vars;
    while h > k {
        // Largest present height strictly below `h` (a merge boundary), or 0.
        let next_lower_present = present_heights
            .iter()
            .filter(|&&p| p < h)
            .max()
            .copied()
            .unwrap_or(0);
        // Stop at the merge boundary but never cross the early-stop boundary.
        let mut stop = next_lower_present.max(k);
        let mut log_folding = h - stop;
        // Cap the merge depth: a wider gap is split into cap-sized chunks.
        if max_log_folding > 0 && log_folding > max_log_folding {
            log_folding = max_log_folding;
            stop = h - max_log_folding;
        }
        debug_assert!(log_folding >= 1);
        groups.push(CommitGroup {
            start_log_height: h,
            log_folding,
        });
        h = stop;
    }
    groups
}

/// Cross-round IOPP opening for a single commit group under a single query.
///
/// Unlike p3-fri's `CommitPhaseProofStep` (which stores a single `sibling_value`
/// and hard-codes row width 2), this carries the **entire** opened row of
/// `2^log_folding` codeword values. The verifier folds these locally across the
/// group's `log_folding` rounds. When `log_folding == 1` this degenerates to a
/// 2-element row (equivalent information to `[query_value, sibling_value]`).
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CrossRoundProofStep<F: Field, M: Mmcs<F>> {
    /// The `2^log_folding` codeword values of the opened row, in natural order
    /// (`codeword[base..base + 2^log_folding]`, `base = (query >> lf) << lf`).
    pub row_values: Vec<F>,
    /// Merkle opening proof for this row against the group's commitment.
    pub opening_proof: M::Proof,
}

/// Cross-round IOPP query proof: one [`CrossRoundProofStep`] per commit group
/// (not per folding round). Replaces p3-fri's per-round `QueryProof` when
/// cross-round commitment is enabled.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CrossRoundQueryProof<F: Field, M: Mmcs<F>> {
    pub group_openings: Vec<CrossRoundProofStep<F, M>>,
}

/// Group-wise path-pruned cross-round IOPP openings.
///
/// Combines cross-round (consecutive folding rounds merged into one wide commit
/// group) with path-pruning (BFS-merged Merkle paths shared across the N queries
/// within each group). Mirrors [`PrunedQueryOpenings`] but for the IOPP commit
/// groups instead of the input PCS batches, and with rows of width `2^log_folding`.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CrossRoundPrunedOpenings<F: Field, M: Mmcs<F>> {
    /// Per-group BFS-merged Merkle proof (length == number of commit groups).
    pub group_pruned: Vec<<M as Mmcs<F>>::PrunedProof>,
    /// Per-group unique opened rows in sorted-index order:
    /// `group_opened_rows[g][unique_slot]` is the `2^log_folding` values of that
    /// group's row. Required because `verify_batch_pruned` consumes leaf values
    /// separately from the merged proof.
    pub group_opened_rows: Vec<Vec<Vec<F>>>,
    /// Per-group query→unique-slot map: `query_to_unique_slot[g][q]` is the index
    /// into `group_opened_rows[g]` for query `q` (replicates the sort+dedup that
    /// `open_batch_pruned` does internally).
    pub query_to_unique_slot: Vec<Vec<u32>>,
}

pub(crate) fn fold_codeword<EF: TwoAdicField>(codeword: &[EF], beta: EF) -> Vec<EF> {
    let n = codeword.len();
    debug_assert!(n >= 2 && n.is_power_of_two());
    let half = n / 2;
    let log_n = log2_strict_usize(n);
    let g_inv = EF::two_adic_generator(log_n).inverse();
    let one_half = EF::two().inverse();
    let half_beta = beta * one_half;

    (0..half)
        .map(|i| {
            let power = g_inv.exp_u64(reverse_bits_len(i, half.trailing_zeros() as usize) as u64)
                * half_beta;
            let r0 = codeword[2 * i];
            let r1 = codeword[2 * i + 1];
            (one_half + power) * r0 + (one_half - power) * r1
        })
        .collect()
}

/// Locally fold one opened row of `2^k` codeword values across `k` rounds
/// (cross-round), using the same Lagrange interpolation as the per-round
/// verifier (see `verify_iopp_query_basefold`) but with **global** generator
/// powers so the result is identical to folding the full codeword `k` times and
/// reading position `pair_index_group`.
///
/// - `row`: the `2^k` contiguous codeword values `codeword[base..base+2^k]`
///   (`base = pair_index_group << k`), in natural order.
/// - `pair_index_group`: the row index within the group's width-`2^k` matrix
///   (`= query_point >> k`).
/// - `log_codeword_len`: log2 of the group's codeword length *before* folding
///   (`= start_log_height + log_blowup`).
/// - `challenges[t]`: the folding challenge for local step `t` (global round
///   `rounds_done + t`).
///
/// The whole `row` is folded each step (`2^k → 2^{k-1} → … → 1`); folding only
/// the query's own pair would be wrong because the next step's sibling must
/// itself be the fold of its own pair. The generator power for buffer pair `p`
/// at step `t` uses the global index `(pair_index_group << (k-1-t)) | p`.
pub(crate) fn fold_row_cross_round<EF: TwoAdicField>(
    row: &[EF],
    pair_index_group: usize,
    log_codeword_len: usize,
    challenges: &[EF],
) -> EF {
    let k = challenges.len();
    debug_assert_eq!(row.len(), 1 << k);
    debug_assert!(log_codeword_len >= k);
    let mut buf = row.to_vec();
    for (t, &r) in challenges.iter().enumerate() {
        let half = buf.len() / 2;
        let lfh = log_codeword_len - 1 - t; // log_folded_height at this step
        let mut next = Vec::with_capacity(half);
        for p in 0..half {
            let e0 = buf[2 * p];
            let e1 = buf[2 * p + 1];
            let global_pair = (pair_index_group << (k - 1 - t)) | p;
            let generator = EF::two_adic_generator(lfh + 1)
                .exp_u64(reverse_bits_len(global_pair, lfh) as u64);
            let slope = (e1 - e0) / (-generator - generator);
            let intercept = e0 - slope * generator;
            next.push(intercept + slope * r);
        }
        buf = next;
    }
    buf[0]
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
pub struct BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger> {
    mmcs: InputMmcs,
    pub(crate) fri: FriConfig<FriMmcs>,
    /// When true, the prover generates path-pruned opening proofs (BFS-merged
    /// Merkle paths for both input PCS batches and IOPP rounds) instead of
    /// per-query independent proofs. This reduces proof size significantly but
    /// makes the recursion program data-dependent (variable Hint count), which
    /// prevents program caching. Set this only on provers whose output feeds
    /// the final layers (penultimate/root shrink) where caching is not needed.
    pub use_path_pruning: bool,
    /// When true, the prover merges consecutive folding rounds with no matrix
    /// merge boundary between them into a single Merkle commitment of row width
    /// `2^log_folding` (cross-round). Reduces the number of IOPP commitments at
    /// the cost of wider opened rows. First version is incompatible with
    /// `use_path_pruning` (both set => error). Default `false` reproduces the
    /// legacy per-round layout exactly.
    pub use_cross_round: bool,
    _phantom: PhantomData<(F, EF, Challenger)>,
}

/// Index structure for mapping a flat global matrix index back to its `(batch_idx, mat_idx)` pair.
///
/// Given `matrices_size: Vec<Vec<Dimensions>>`, each inner `Vec<Dimensions>` corresponds to one
/// batch. This struct builds a prefix-sum array so that a global index (across all batches) can
/// be efficiently mapped to the batch and matrix it belongs to via binary search.
struct MatricesSizeIndex {
    /// Prefix sum array: `prefix_sums[i]` is the total number of matrices in the first `i` batches.
    prefix_sums: Vec<usize>,
}

impl MatricesSizeIndex {
    /// Create index structure from matrices_size
    fn new(matrices_size: &Vec<Vec<Dimensions>>) -> Self {
        let mut prefix_sums = Vec::with_capacity(matrices_size.len() + 1);
        prefix_sums.push(0);

        let mut sum = 0;
        for vec in matrices_size {
            sum += vec.len();
            prefix_sums.push(sum);
        }

        Self { prefix_sums }
    }

    /// Find the position (i, j) corresponding to the global index
    /// - i: index of the outer Vec
    /// - j: index of the inner Vec
    fn find_position(&self, index: usize) -> (usize, usize) {
        // Use binary search to determine the outer index i
        let i = match self.prefix_sums.binary_search(&index) {
            Ok(exact) => exact,
            Err(insert_pos) => insert_pos - 1,
        };

        // Calculate the inner index j
        let j = index - self.prefix_sums[i];

        (i, j)
    }
}

impl<Val, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<Val, InputMmcs, FriMmcs, EF, Challenger> {
    pub const fn new(mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri,
            use_path_pruning: false,
            use_cross_round: false,
            _phantom: PhantomData,
        }
    }

    /// Create a new `BaseFoldPcs` with path-pruning enabled.
    /// Path-pruned proofs use BFS-merged Merkle paths to reduce proof size,
    /// but prevent recursion program caching due to data-dependent Hint counts.
    pub const fn new_with_path_pruning(mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri,
            use_path_pruning: true,
            use_cross_round: false,
            _phantom: PhantomData,
        }
    }

    /// Create a new `BaseFoldPcs` with cross-round commitment enabled.
    /// Consecutive folding rounds with no matrix merge boundary are merged into
    /// a single wide Merkle commitment. Incompatible with path-pruning (the
    /// first version errors if both are set).
    pub const fn new_with_cross_round(mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri,
            use_path_pruning: false,
            use_cross_round: true,
            _phantom: PhantomData,
        }
    }
}

// =====================================================================
// Shared commit implementation (used by both WHIR and Basefold)
// =====================================================================
impl<F, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    /// Commit multiple compressed matrices with different dimensions.
    ///
    /// Each CompressedMatrix stores only non-padding rows; padding rows are
    /// decompressed before DFT encoding.
    ///
    /// Returns: (merkle_root, merkle_tree)
    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::commit")]
    pub fn commit_impl(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
    ) -> (
        InputMmcs::Commitment,
        InputMmcs::ProverData<RowMajorMatrix<F>>,
    ) {
        assert!(self.fri.log_blowup > 0, "log_blowup must be greater than 0");

        let repeat_times = 1 << self.fri.log_blowup;
        let codewords: Vec<RowMajorMatrix<F>> = {
            let _span = tracing::debug_span!("decompress_and_dft").entered();
            evaluations
                .into_par_iter()
                .map(|compressed| {
                    let dft = EvalsDft::<F>::default();
                    dft.dft_batch_by_evals(compressed.decompress_and_repeat(repeat_times))
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect()
        };

        let commitment = {
            let _span = tracing::debug_span!("merkle_tree_commit").entered();
            self.mmcs.commit(codewords)
        };

        commitment
    }
}

// =====================================================================
// WHIR algorithm (default, when `basefold` feature is NOT enabled)
// =====================================================================
#[cfg(not(feature = "basefold"))]
impl<F, InputMmcs, FriMmcs, EF, Challenger> MlPCS
    for BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    type Field = F;
    type ExtensionField = EF;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<F>>;
    type Commitment = InputMmcs::Commitment;
    type BatchProof = BasefoldProof<EF, FriMmcs, F, BasefoldInputProof<F, InputMmcs>>;
    type Challenger = Challenger;
    type Error = BaseFoldError<FriMmcs::Error, InputMmcs::Error>;

    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::commit")]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.commit_impl(evaluations)
    }

    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::open")]
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<Self::Field>>>,
        prover_data_batch: Vec<Self::ProverData>,
        opening_point: &[Self::ExtensionField],
        opened_values: &Vec<Vec<Vec<Self::ExtensionField>>>,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::BatchProof, Self::Error> {
        // --- Input validation ---
        self.validate_open_inputs(&polynomials_batch, &prover_data_batch, opened_values)?;

        let num_vars = opening_point.len();

        // Max log_height per batch (needed for query point shifting)
        let max_log_height_per_batch: Vec<usize> = polynomials_batch
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|m| log2_strict_usize(m.height()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        // --- Flatten and group by height ---
        let polynomials: Vec<CompressedMatrix<Self::Field>> =
            polynomials_batch.into_iter().flatten().collect();
        let flat_opened_values: Vec<&Vec<Self::ExtensionField>> =
            opened_values.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height = self.group_by_log_height(&polynomials, &flat_opened_values)?;
        let max_log_height = *matrices_by_log_height.keys().last().unwrap_or(&0);
        debug_assert_eq!(max_log_height, num_vars);

        // --- Phase 1: Linear combination with a shared alpha stream across height groups ---
        let alpha: EF = challenger.sample_ext_element();
        let mut powers_of_alpha = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut f_polys_by_height: BTreeMap<usize, MultilinearPolynomial<EF>> = BTreeMap::new();

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (matrices, values): (Vec<_>, Vec<_>) = group.iter().cloned().unzip();

            let coefficients: Vec<Vec<EF>> = values
                .iter()
                .map(|vals| {
                    vals.iter()
                        .map(|_| powers_of_alpha.next().unwrap())
                        .collect()
                })
                .collect();

            let mut combined_evals = vec![EF::zero(); 1 << log_height];
            MultilinearPolynomial::random_linear_combine_columns_compressed(
                matrices,
                &coefficients,
                &mut combined_evals,
            );

            f_polys_by_height.insert(log_height, MultilinearPolynomial::new(combined_evals));
        }

        // --- Batching proof of work ---
        let grinding_batching_data =
            self.find_pow_witness(challenger, self.fri.grinding_bits_batching)?;

        // --- Phase 2: Iterative WHIR + sumcheck folding ---
        let merge_function = |x: &[EF]| x.iter().copied().product::<EF>();
        let dft = EvalsDft::<F>::default();

        let highest_f = f_polys_by_height
            .remove(&max_log_height)
            .expect("at least one matrix is required");
        let eq_polynomial = EqPolynomial::new(opening_point.to_vec()).to_ml();
        let mut current_polys = vec![highest_f, eq_polynomial];

        // Compute the initial claim: Σ F[i] * EQ[i] for the highest height group
        let mut running_claim: EF =
            compute_dotproduct(&current_polys[0].evals, &current_polys[1].evals);

        // Pre-compute branch claims for each height group that will be merged later
        let mut branch_claims: BTreeMap<usize, EF> = BTreeMap::new();
        for (&log_height, f_poly) in f_polys_by_height.iter() {
            let branch_eq = EqPolynomial::new(opening_point[..log_height].to_vec()).to_ml();
            let branch_claim: EF = compute_dotproduct(&f_poly.evals, &branch_eq.evals);
            branch_claims.insert(log_height, branch_claim);
        }

        let mut sumcheck_polys = Vec::new();
        let mut iopp_commitments = Vec::new();
        let mut iopp_prover_data = Vec::new();
        let mut out_of_domain_responses = Vec::new();

        for round in (0..=num_vars).rev() {
            // Step 1: Commit the current F polynomial as a Reed-Solomon codeword
            let codeword = self.encode_to_codeword(&current_polys[0].evals, &dft);
            let (root, tree) = self
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, 2));

            iopp_commitments.push(root.clone());
            iopp_prover_data.push(tree);
            challenger.observe(root);

            if round == 0 {
                break;
            }

            // Step 2: WHIR round — out-of-domain challenge, updates running_claim
            let (_, gamma, ood_response) =
                SumcheckInstanceProof::sumcheck_prove_whir_round(&mut current_polys, challenger)
                    .map_err(|_| Self::Error::SumcheckPhaseError)?;
            out_of_domain_responses.push(ood_response);
            running_claim += gamma * ood_response;

            // Step 3: One round of normal sumcheck with the current claim
            // sumcheck_prove_normal_round internally samples the folding challenge r_fold
            // and returns it in r_vec; running_claim updates to g(r_fold).
            let (sc_proof, r_vec, _) = SumcheckInstanceProof::sumcheck_prove_normal_round(
                &running_claim,
                1,
                &mut current_polys,
                &merge_function,
                2,
                challenger,
            )
            .map_err(|_| Self::Error::SumcheckPhaseError)?;
            running_claim = sc_proof.uni_polys[0].evaluate(&r_vec[0]);
            sumcheck_polys.push(sc_proof.uni_polys[0].clone());

            // Step 4: Merge with a smaller height group if one exists at this level
            if let Some(branch_f) = f_polys_by_height.remove(&(round - 1)) {
                debug_assert_eq!(branch_f.len(), current_polys[0].len());

                let branch_eq = EqPolynomial::new(opening_point[..(round - 1)].to_vec()).to_ml();

                let branch_claim = branch_claims
                    .remove(&(round - 1))
                    .expect("branch claim must exist for this height group");

                let branch_instance = vec![branch_f, branch_eq];

                // sumcheck_prove_merge_two_instances internally samples r_merge
                // and returns it; running_claim updates to merge_g(r_merge).
                let (merge_uni_poly, r_merge, merged_polys) =
                    SumcheckInstanceProof::sumcheck_prove_merge_two_instances(
                        &running_claim,
                        &branch_claim,
                        &current_polys,
                        &branch_instance,
                        challenger,
                    )
                    .map_err(|_| Self::Error::SumcheckPhaseError)?;

                running_claim = merge_uni_poly.evaluate(&r_merge);
                sumcheck_polys.push(merge_uni_poly);
                current_polys = merged_polys;
            }
        }

        // --- Phase 3: Query proof of work ---
        let grinding_query_data =
            self.find_pow_witness(challenger, self.fri.grinding_bits_query)?;

        // --- Phase 4: IOPP query generation ---
        let query_points: Vec<usize> = (0..self.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.fri.log_blowup))
            .collect();

        // [B-Stage 1' B6-5] Path-pruning swaps the IOPP query answer path
        // from per-query `answer_query` to a single batched
        // `answer_queries_pruned`. Controlled by PCS config field.
        let use_path_pruning = self.use_path_pruning;

        // Input batch open (always per-query path).
        let query_openings: Vec<Vec<BatchOpening<F, InputMmcs>>> = query_points
            .iter()
            .map(|&point| {
                prover_data_batch
                    .iter()
                    .enumerate()
                    .map(|(batch_idx, prover_data)| {
                        let shifted_point =
                            point >> (max_log_height - max_log_height_per_batch[batch_idx]);
                        let (values, proof) = self.mmcs.open_batch(shifted_point, prover_data);
                        BatchOpening {
                            opened_values: values,
                            opening_proof: proof,
                        }
                    })
                    .collect()
            })
            .collect();

        // IOPP: env-gated split.
        let (iopp_queries, iopp_pruned) = if use_path_pruning {
            let pruned = answer_queries_pruned(&self.fri, &iopp_prover_data, &query_points);
            (Vec::new(), Some(pruned))
        } else {
            let queries = query_points
                .iter()
                .map(|&point| answer_query(&self.fri, &iopp_prover_data, point))
                .collect::<Vec<_>>();
            (queries, None)
        };

        Ok(BasefoldProof {
            sumcheck_transcript: SumcheckInstanceProof {
                uni_polys: sumcheck_polys,
            },
            iopp_oracles: iopp_commitments,
            iopp_queries,
            query_openings: BasefoldInputProof::from_per_query(query_openings),
            grinding_batching_witness: grinding_batching_data,
            grinding_query_witness: grinding_query_data,
            out_of_domain_responses: Some(out_of_domain_responses),
            final_poly: vec![],
            iopp_pruned,
            iopp_cross_round: Vec::new(),
            iopp_cross_round_pruned: None,
        })
    }

    /// Verify the batch opening proof.
    ///
    /// Reconstructs the verifier's view of the sumcheck protocol (WHIR + folding + merging),
    /// then checks the final codeword commitment and IOPP queries.
    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::verify")]
    fn verify(
        &self,
        commitment_batch: Vec<Self::Commitment>,
        matrices_size_batch: &Vec<Vec<Dimensions>>,
        opening_point: &[Self::ExtensionField],
        opened_values_batch: &Vec<Vec<Vec<Self::ExtensionField>>>,
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        // --- Input validation ---
        self.validate_verify_inputs(&commitment_batch, matrices_size_batch, opened_values_batch)?;

        let Self::BatchProof {
            sumcheck_transcript,
            iopp_oracles,
            iopp_queries,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            out_of_domain_responses,
            final_poly: _,
            iopp_pruned: _,
            iopp_cross_round: _,
            iopp_cross_round_pruned: _,
        } = proof;
        // [D6] WHIR path does NOT support PCS opening path-pruning. Reject any
        // proof that claims `pruned` is populated to avoid silently dropping
        // a malicious prover's pruned data without verification.
        if query_openings.pruned.is_some() {
            return Err(Self::Error::InvalidInputError);
        }
        let query_openings = &query_openings.per_query;

        let out_of_domain_responses = out_of_domain_responses
            .as_ref()
            .ok_or(Self::Error::InvalidInputError)?;

        if grinding_batching_witness.len() != 2 || grinding_query_witness.len() != 2 {
            return Err(Self::Error::InvalidInputError);
        }

        let num_vars = opening_point.len();

        // --- Flatten and group by log_height ---
        let flat_dims: Vec<DimAndNo> = matrices_size_batch
            .iter()
            .flat_map(|batch| batch.iter())
            .enumerate()
            .map(|(idx, dim)| DimAndNo {
                dim: dim.clone(),
                num: idx,
            })
            .collect();
        let flat_opened_values: Vec<&Vec<EF>> =
            opened_values_batch.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height =
            Self::group_dims_by_log_height(&flat_dims, &flat_opened_values);
        let log_max_height = matrices_by_log_height.keys().max().cloned().unwrap_or(0);
        debug_assert_eq!(log_max_height, num_vars);

        let size_index = MatricesSizeIndex::new(matrices_size_batch);

        // --- Phase 1: Compute claimed sums and query coefficients per height group ---
        // Alpha powers are shared across all groups, so no extra group challenge is needed.
        let alpha: EF = challenger.sample_ext_element();
        let mut alpha_powers = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut claims_by_height: BTreeMap<usize, EF> = BTreeMap::new();
        let mut coefficients_by_height: BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>> =
            BTreeMap::new();

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (dims, values): (Vec<&DimAndNo>, Vec<&Vec<EF>>) =
                group.iter().map(|(d, v)| (*d, *v)).unzip();

            let coeffs: Vec<Vec<EF>> = values
                .iter()
                .map(|vals| vals.iter().map(|_| alpha_powers.next().unwrap()).collect())
                .collect();

            let claimed_sum: EF = values
                .iter()
                .zip(coeffs.iter())
                .flat_map(|(vals, cs)| vals.iter().zip(cs.iter()).map(|(v, c)| *v * *c))
                .sum::<EF>();
            claims_by_height.insert(log_height, claimed_sum);

            let indexed_coeffs: Vec<((usize, usize), Vec<EF>)> = coeffs
                .iter()
                .enumerate()
                .map(|(i, cs)| (size_index.find_position(dims[i].num), cs.clone()))
                .collect();
            coefficients_by_height.insert(log_height, indexed_coeffs);
        }

        // --- Verify batching proof of work ---
        challenger.observe(grinding_batching_witness[0]);
        if !challenger.check_witness(
            self.fri.grinding_bits_batching,
            grinding_batching_witness[1],
        ) {
            return Err(Self::Error::InvalidPowWitness);
        }

        // --- Phase 2: Verify sumcheck rounds (WHIR + fold + merge) ---
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut current_claim = claims_by_height
            .remove(&num_vars)
            .ok_or(Self::Error::SumcheckPhaseError)?;

        challenger.observe(iopp_oracles[0].clone());

        let mut folding_challenges: Vec<EF> = Vec::with_capacity(num_vars);
        let mut merging_challenges: Vec<EF> = Vec::new();
        let mut whir_alphas: Vec<EF> = Vec::with_capacity(num_vars);
        let mut whir_gammas: Vec<EF> = Vec::with_capacity(num_vars);
        // (loop_iteration_index, r_merge) — records when merges occurred
        let mut merge_rounds: Vec<(usize, EF)> = Vec::new();

        let mut ood_idx = 0;
        for round in (0..=num_vars).rev() {
            if round < num_vars {
                challenger.observe(iopp_oracles[num_vars - round].clone());
            }
            if round == 0 {
                break;
            }

            // WHIR out-of-domain challenge
            let whir_alpha: EF = challenger.sample_ext_element();
            let gamma: EF = challenger.sample_ext_element();
            whir_alphas.push(whir_alpha);
            whir_gammas.push(gamma);

            challenger.observe_ext_element(out_of_domain_responses[ood_idx]);
            current_claim += gamma * out_of_domain_responses[ood_idx];
            ood_idx += 1;

            // Normal sumcheck round: verify g(0) + g(1) = claim
            let uni_poly = poly_iter.next().ok_or(Self::Error::SumcheckPhaseError)?;
            if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != current_claim {
                return Err(Self::Error::SumcheckPhaseError);
            }
            uni_poly
                .coeffs
                .iter()
                .for_each(|c| challenger.observe_ext_element(*c));
            let r_fold: EF = challenger.sample_ext_element();
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate(&r_fold);

            // Merge round (if a shorter height group enters at this level)
            let loop_idx = num_vars - round;
            if let Some(branch_claim) = claims_by_height.remove(&(round - 1)) {
                let merge_poly = poly_iter.next().ok_or(Self::Error::SumcheckPhaseError)?;
                let merge_sum = merge_poly.eval_at_zero() + merge_poly.eval_at_one();
                if merge_sum != current_claim + branch_claim {
                    return Err(Self::Error::SumcheckPhaseError);
                }
                merge_poly
                    .coeffs
                    .iter()
                    .for_each(|c| challenger.observe_ext_element(*c));
                let r_merge: EF = challenger.sample_ext_element();
                merge_rounds.push((loop_idx, r_merge));
                merging_challenges.push(r_merge);
                current_claim = merge_poly.evaluate(&r_merge);
            }
        }

        // --- Phase 3: Reconstruct combined EQ sum ---
        // With little-endian folding, fc[k] binds x[n-1-k].
        // For eq(p; x), x[i] = fc[n-1-i], so we use reversed folding challenges.
        let merge_at_iter: BTreeMap<usize, usize> = merge_rounds
            .iter()
            .enumerate()
            .map(|(idx, &(loop_idx, _))| (loop_idx, idx))
            .collect();

        let fc_rev: Vec<EF> = folding_challenges.iter().rev().cloned().collect();

        let mut combined_eq_sum = EqPolynomial::new(opening_point.to_vec()).evaluate(&fc_rev);

        for iteration in 0..num_vars {
            let remaining = num_vars - iteration;
            let alpha_sq_powers: Vec<EF> =
                std::iter::successors(Some(whir_alphas[iteration]), |prev| Some(*prev * *prev))
                    .take(remaining)
                    .collect();
            // WHIR eq has `remaining` vars bound to fc_rev[0..remaining]
            combined_eq_sum += whir_gammas[iteration]
                * EqPolynomial::new(alpha_sq_powers).evaluate(&fc_rev[..remaining]);

            if let Some(&merge_idx) = merge_at_iter.get(&iteration) {
                let r_merge = merge_rounds[merge_idx].1;
                // Branch eq has (remaining-1) vars bound to fc_rev[0..remaining-1]
                let branch_eq =
                    EqPolynomial::new(opening_point[..(num_vars - iteration - 1)].to_vec())
                        .evaluate(&fc_rev[..remaining - 1]);
                combined_eq_sum += r_merge * (branch_eq - combined_eq_sum);
            }
        }

        // --- Phase 4: Final codeword commitment check ---
        let combined_f_r: EF = current_claim / combined_eq_sum;

        let expected_codeword = vec![combined_f_r; 1 << self.fri.log_blowup];
        let (expected_commitment, _) = self
            .fri
            .mmcs
            .commit_matrix(RowMajorMatrix::new(expected_codeword, 2));

        let last_oracle = iopp_oracles
            .last()
            .ok_or(Self::Error::CommitmentCheckFailed)?;
        let last_bytes =
            bincode::serialize(last_oracle).map_err(|_| Self::Error::CommitmentCheckFailed)?;
        let expected_bytes = bincode::serialize(&expected_commitment)
            .map_err(|_| Self::Error::CommitmentCheckFailed)?;
        if last_bytes != expected_bytes {
            return Err(Self::Error::CommitmentCheckFailed);
        }

        // --- Phase 5: Query proof of work ---
        challenger.observe(grinding_query_witness[0]);
        if !challenger.check_witness(self.fri.grinding_bits_query, grinding_query_witness[1]) {
            return Err(Self::Error::InvalidPowWitness);
        }

        // --- Phase 6: IOPP query verification ---
        let query_points: Vec<usize> = (0..self.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.fri.log_blowup))
            .collect();

        let all_queries_valid = iopp_queries
            .par_iter()
            .zip(query_openings.par_iter())
            .enumerate()
            .all(|(i, (query, leaf_opening))| {
                self.verify_query_p3_batch(
                    &commitment_batch,
                    iopp_oracles.as_slice(),
                    query_points[i],
                    matrices_size_batch,
                    query,
                    leaf_opening,
                    &coefficients_by_height,
                    &folding_challenges,
                    &merging_challenges,
                    &combined_f_r,
                )
                .is_ok()
            });

        if !all_queries_valid {
            return Err(BaseFoldError::FriFinalStepMisMatch);
        }

        Ok(())
    }
}

// =====================================================================
// Basefold algorithm (when `basefold` feature IS enabled)
// =====================================================================
#[cfg(feature = "basefold")]
impl<F, InputMmcs, FriMmcs, EF, Challenger> MlPCS
    for BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    type Field = F;
    type ExtensionField = EF;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<F>>;
    type Commitment = InputMmcs::Commitment;
    type BatchProof = BasefoldProof<EF, FriMmcs, F, BasefoldInputProof<F, InputMmcs>>;
    type Challenger = Challenger;
    type Error = BaseFoldError<FriMmcs::Error, InputMmcs::Error>;

    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::commit")]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.commit_impl(evaluations)
    }

    /// Basefold open: generates a batch opening proof using little-endian folding
    /// without WHIR out-of-domain sampling.
    ///
    /// At merge points, the EQ prefix matches due to little-endian folding,
    /// so we use a simple random linear combination instead of a full merge sumcheck round.
    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::open")]
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<F>>>,
        prover_data_batch: Vec<Self::ProverData>,
        opening_point: &[EF],
        opened_values: &Vec<Vec<Vec<EF>>>,
        challenger: &mut Challenger,
    ) -> Result<Self::BatchProof, Self::Error> {
        self.validate_open_inputs(&polynomials_batch, &prover_data_batch, opened_values)?;

        let num_vars = opening_point.len();

        let max_log_height_per_batch: Vec<usize> = polynomials_batch
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|m| log2_strict_usize(m.height()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        let polynomials: Vec<CompressedMatrix<F>> =
            polynomials_batch.into_iter().flatten().collect();
        let flat_opened_values: Vec<&Vec<EF>> =
            opened_values.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height = self.group_by_log_height(&polynomials, &flat_opened_values)?;
        let max_log_height = *matrices_by_log_height.keys().last().unwrap_or(&0);
        debug_assert_eq!(max_log_height, num_vars);

        // --- Phase 1: Linear combination per height group ---
        let alpha: EF = challenger.sample_ext_element();
        let mut powers_of_alpha = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut f_polys_by_height: BTreeMap<usize, MultilinearPolynomial<EF>> = BTreeMap::new();

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (matrices, values): (Vec<_>, Vec<_>) = group.iter().cloned().unzip();

            let coefficients: Vec<Vec<EF>> = values
                .iter()
                .map(|vals| {
                    vals.iter()
                        .map(|_| powers_of_alpha.next().unwrap())
                        .collect()
                })
                .collect();

            let mut combined_evals = vec![EF::zero(); 1 << log_height];
            MultilinearPolynomial::random_linear_combine_columns_compressed(
                matrices,
                &coefficients,
                &mut combined_evals,
            );

            f_polys_by_height.insert(log_height, MultilinearPolynomial::new(combined_evals));
        }

        // --- Batching proof of work ---
        let grinding_batching_data =
            self.find_pow_witness(challenger, self.fri.grinding_bits_batching)?;

        // --- Phase 2: Basefold sumcheck folding (no WHIR, simplified merge) ---
        let merge_function = |x: &[EF]| x.iter().copied().product::<EF>();
        let dft = EvalsDft::<F>::default();

        let highest_f = f_polys_by_height
            .remove(&max_log_height)
            .expect("at least one matrix is required");
        let eq_polynomial = EqPolynomial::new(opening_point.to_vec()).to_ml();
        let mut current_polys = vec![highest_f, eq_polynomial];

        let mut running_claim: EF =
            compute_dotproduct(&current_polys[0].evals, &current_polys[1].evals);

        let mut branch_claims: BTreeMap<usize, EF> = BTreeMap::new();
        for (&log_height, f_poly) in f_polys_by_height.iter() {
            let branch_eq = EqPolynomial::new(opening_point[..log_height].to_vec()).to_ml();
            let branch_claim: EF = compute_dotproduct(&f_poly.evals, &branch_eq.evals);
            branch_claims.insert(log_height, branch_claim);
        }

        let mut sumcheck_polys = Vec::new();
        let mut iopp_commitments = Vec::new();
        let mut iopp_prover_data = Vec::new();
        let mut eq_factor = EF::one();
        let min_log_height = matrices_by_log_height
            .keys()
            .min()
            .cloned()
            .unwrap_or(num_vars);
        let k = self.fri.log_final_poly_len.min(min_log_height);
        // [cross-round] Build the (possibly merged) commitment schedule. The
        // legacy per-round layout is `log_folding == 1` for every group; with
        // cross-round enabled, consecutive folds with no matrix merge boundary
        // share one wide commitment. `present_heights` are the input matrix
        // log-heights (= keys of `matrices_by_log_height`).
        let present_heights: std::collections::BTreeSet<usize> =
            matrices_by_log_height.keys().copied().collect();
        let commit_schedule = compute_commit_schedule_cross_round(
            &present_heights,
            num_vars,
            k,
            self.use_cross_round,
        );
        let mut final_poly_evals: Vec<EF> = Vec::new();

        for group in commit_schedule.iter() {
            // Commit the current codeword ONCE for the whole group, with row
            // width `2^log_folding`. Legacy (log_folding == 1) commits width 2.
            let row_width = 1usize << group.log_folding;
            let codeword = self.encode_to_codeword(&current_polys[0].evals, &dft);
            let (root, tree) = self
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, row_width));

            iopp_commitments.push(root.clone());
            iopp_prover_data.push(tree);
            challenger.observe(root);

            // Perform `log_folding` consecutive sumcheck/folding rounds. The
            // sumcheck protocol itself is unchanged (one fold per height
            // variable); only the IOPP commitment is shared across them.
            for t in 0..group.log_folding {
                let round = group.start_log_height - t;

                // Normal sumcheck round (little-endian folding, no WHIR)
                let (sc_proof, r_vec, _) = SumcheckInstanceProof::sumcheck_prove_normal_round(
                    &running_claim,
                    1,
                    &mut current_polys,
                    &merge_function,
                    2,
                    challenger,
                )
                .map_err(|_| BaseFoldError::SumcheckPhaseError)?;
                running_claim = sc_proof.uni_polys[0].evaluate(&r_vec[0]);
                sumcheck_polys.push(sc_proof.uni_polys[0].clone());

                // Accumulate eq factor: eq(p[round-1]; r_fold)
                let r_fold = r_vec[0];
                let p_i = opening_point[round - 1];
                eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);

                // Basefold merge: by schedule construction this fires only at a
                // group boundary (the last inner round), never mid-group, since
                // no present height lies strictly inside a group.
                if let Some(branch_f) = f_polys_by_height.remove(&(round - 1)) {
                    debug_assert_eq!(branch_f.len(), current_polys[0].len());
                    debug_assert_eq!(
                        t,
                        group.log_folding - 1,
                        "merge must only occur at a cross-round group boundary"
                    );

                    let branch_claim = branch_claims
                        .remove(&(round - 1))
                        .expect("branch claim must exist for this height group");

                    // Sample merge coefficient
                    let merge_beta: EF = challenger.sample_ext_element();

                    // F_new = eq_factor * F + merge_beta * G
                    // EQ_new = eq(p[0..round-1]; cube) = branch_eq
                    let branch_eq =
                        EqPolynomial::new(opening_point[..(round - 1)].to_vec()).to_ml();

                    current_polys[0]
                        .evals
                        .par_iter_mut()
                        .zip(branch_f.evals.par_iter())
                        .for_each(|(f_val, g_val)| {
                            *f_val = eq_factor * *f_val + merge_beta * *g_val;
                        });

                    current_polys[1] = branch_eq;

                    running_claim = running_claim + merge_beta * branch_claim;

                    eq_factor = EF::one();
                }
            }
        }

        if k > 0 {
            final_poly_evals = current_polys[0].evals.clone();
            for coeff in &final_poly_evals {
                challenger.observe_ext_element(*coeff);
            }

            for round in (1..=k).rev() {
                let (sc_proof, r_vec, _) = SumcheckInstanceProof::sumcheck_prove_normal_round(
                    &running_claim,
                    1,
                    &mut current_polys,
                    &merge_function,
                    2,
                    challenger,
                )
                .map_err(|_| BaseFoldError::SumcheckPhaseError)?;
                running_claim = sc_proof.uni_polys[0].evaluate(&r_vec[0]);
                sumcheck_polys.push(sc_proof.uni_polys[0].clone());

                let r_fold = r_vec[0];
                let p_i = opening_point[round - 1];
                eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);
            }
        } else {
            let codeword = self.encode_to_codeword(&current_polys[0].evals, &dft);
            let (root, tree) = self
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, 2));
            iopp_commitments.push(root.clone());
            iopp_prover_data.push(tree);
            challenger.observe(root);
        }

        // --- Phase 3: Query proof of work ---
        let grinding_query_data =
            self.find_pow_witness(challenger, self.fri.grinding_bits_query)?;

        // --- Phase 4: IOPP query generation ---
        let query_points: Vec<usize> = (0..self.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.fri.log_blowup))
            .collect();

        // [B-Stage 1' B6-5] Path-pruning swaps the IOPP query answer path
        // from per-query `answer_query` to a single batched
        // `answer_queries_pruned`. [D6] also swaps the input PCS batch open
        // path from per-query `open_batch` to per-round `open_batch_pruned`.
        // Controlled by PCS config field.
        let use_path_pruning = self.use_path_pruning;

        // [D6] Input batch open: env-gated per-query OR per-round-pruned.
        let query_openings_bundle: BasefoldInputProof<F, InputMmcs> = if use_path_pruning {
            // Per-round pruned path: for each round (== batch), call
            // `open_batch_pruned` once with all Q shifted query indices.
            let num_batches = prover_data_batch.len();
            let mut round_pruned = Vec::with_capacity(num_batches);
            let mut round_opened_values: Vec<Vec<Vec<Vec<F>>>> = Vec::with_capacity(num_batches);
            let mut q2u: Vec<Vec<u32>> = Vec::with_capacity(num_batches);

            for (batch_idx, prover_data) in prover_data_batch.iter().enumerate() {
                let shift = max_log_height - max_log_height_per_batch[batch_idx];
                let shifted_per_query: Vec<usize> =
                    query_points.iter().map(|&p| p >> shift).collect();

                // open_batch_pruned internally sort+dedup; returns unique
                // openings in sorted order + the pruned merkle proof.
                let (uniq_opened, pruned_proof) =
                    self.mmcs.open_batch_pruned(&shifted_per_query, prover_data);

                // Compute query→unique-slot hint by replicating the same
                // sort+dedup the trait impl does internally.
                let mut sorted_dedup: Vec<usize> = shifted_per_query.clone();
                sorted_dedup.sort_unstable();
                sorted_dedup.dedup();
                let q2u_round: Vec<u32> = shifted_per_query
                    .iter()
                    .map(|&q| sorted_dedup.binary_search(&q).unwrap() as u32)
                    .collect();

                round_pruned.push(pruned_proof);
                round_opened_values.push(uniq_opened);
                q2u.push(q2u_round);
            }

            // [SS-1] per_query is empty in pruned mode: the circuit
            // extracts opened values from `pruned.round_opened_values`
            // via `query_to_unique_slot`, saving ~N×batch×auth_path bytes.
            BasefoldInputProof {
                per_query: Vec::new(),
                pruned: Some(PrunedQueryOpenings {
                    round_pruned,
                    round_opened_values,
                    query_to_unique_slot: q2u,
                }),
            }
        } else {
            // Standard per-query path.
            let qo: Vec<Vec<BatchOpening<F, InputMmcs>>> = query_points
                .iter()
                .map(|&point| {
                    prover_data_batch
                        .iter()
                        .enumerate()
                        .map(|(batch_idx, prover_data)| {
                            let shifted_point =
                                point >> (max_log_height - max_log_height_per_batch[batch_idx]);
                            let (values, proof) = self.mmcs.open_batch(shifted_point, prover_data);
                            BatchOpening {
                                opened_values: values,
                                opening_proof: proof,
                            }
                        })
                        .collect()
                })
                .collect();
            BasefoldInputProof::from_per_query(qo)
        };

        // IOPP: gated split between both(cross-round+pruning), path-pruning,
        // cross-round, and standard.
        #[allow(clippy::type_complexity)]
        let (iopp_queries, iopp_pruned, iopp_cross_round, iopp_cross_round_pruned) =
            if use_path_pruning && self.use_cross_round {
                // [both] Group-wise path-pruning over cross-round wide rows.
                // For each commit group, open ALL queries' rows in one
                // `open_batch_pruned` (BFS-merged Merkle path), de-duplicating
                // shared rows. The query index shifts right by `log_folding`
                // after each group (same walk as plain cross-round).
                let num_groups = commit_schedule.len();
                let mut group_pruned = Vec::with_capacity(num_groups);
                let mut group_opened_rows: Vec<Vec<Vec<EF>>> = Vec::with_capacity(num_groups);
                let mut q2u: Vec<Vec<u32>> = Vec::with_capacity(num_groups);

                // Track each query's running point as it shifts per group.
                let mut query_points_shifted: Vec<usize> = query_points.clone();
                for (g, group) in commit_schedule.iter().enumerate() {
                    let lf = group.log_folding;
                    let row_idx: Vec<usize> =
                        query_points_shifted.iter().map(|&p| p >> lf).collect();

                    let (uniq_rows, pruned_proof) =
                        self.fri.mmcs.open_batch_pruned(&row_idx, &iopp_prover_data[g]);

                    // Replicate open_batch_pruned's internal sort+dedup to map
                    // each query to its unique-row slot.
                    let mut sorted_dedup: Vec<usize> = row_idx.clone();
                    sorted_dedup.sort_unstable();
                    sorted_dedup.dedup();
                    let q2u_g: Vec<u32> = row_idx
                        .iter()
                        .map(|&r| sorted_dedup.binary_search(&r).unwrap() as u32)
                        .collect();

                    // uniq_rows[slot] is Vec<Vec<EF>> (one inner Vec per matrix);
                    // each group commits a single matrix, so take [0].
                    let rows_g: Vec<Vec<EF>> =
                        uniq_rows.into_iter().map(|mut per_mat| per_mat.pop().unwrap()).collect();

                    group_pruned.push(pruned_proof);
                    group_opened_rows.push(rows_g);
                    q2u.push(q2u_g);

                    // Advance every query's point for the next group.
                    for p in query_points_shifted.iter_mut() {
                        *p >>= lf;
                    }
                }

                (
                    Vec::new(),
                    None,
                    Vec::new(),
                    Some(CrossRoundPrunedOpenings {
                        group_pruned,
                        group_opened_rows,
                        query_to_unique_slot: q2u,
                    }),
                )
            } else if use_path_pruning {
                let pruned = answer_queries_pruned(&self.fri, &iopp_prover_data, &query_points);
                (Vec::new(), Some(pruned), Vec::new(), None)
            } else if self.use_cross_round {
                // [cross-round] One opened row per commit group per query. Each group
                // committed a width-`2^log_folding` matrix; open the row the query
                // falls into and store all `2^log_folding` values. The query index
                // is shifted right by `log_folding` after each group.
                let cross = query_points
                    .iter()
                    .map(|&point| {
                        let mut query_point = point;
                        let mut group_openings = Vec::with_capacity(commit_schedule.len());
                        for (g, group) in commit_schedule.iter().enumerate() {
                            let lf = group.log_folding;
                            let pair_index_group = query_point >> lf;
                            let (mut rows, opening_proof) =
                                self.fri.mmcs.open_batch(pair_index_group, &iopp_prover_data[g]);
                            debug_assert_eq!(rows.len(), 1);
                            let row_values = rows.pop().unwrap();
                            debug_assert_eq!(row_values.len(), 1usize << lf);
                            group_openings.push(CrossRoundProofStep {
                                row_values,
                                opening_proof,
                            });
                            query_point >>= lf;
                        }
                        CrossRoundQueryProof { group_openings }
                    })
                    .collect::<Vec<_>>();
                (Vec::new(), None, cross, None)
            } else {
                let queries = query_points
                    .iter()
                    .map(|&point| answer_query(&self.fri, &iopp_prover_data, point))
                    .collect::<Vec<_>>();
                (queries, None, Vec::new(), None)
            };

        Ok(BasefoldProof {
            sumcheck_transcript: SumcheckInstanceProof {
                uni_polys: sumcheck_polys,
            },
            iopp_oracles: iopp_commitments,
            iopp_queries,
            query_openings: query_openings_bundle,
            grinding_batching_witness: grinding_batching_data,
            grinding_query_witness: grinding_query_data,
            out_of_domain_responses: None,
            final_poly: final_poly_evals,
            iopp_pruned,
            iopp_cross_round,
            iopp_cross_round_pruned,
        })
    }

    /// Verify the batch opening proof (basefold variant).
    ///
    /// No WHIR out-of-domain sampling. Merges are deterministic (no merge sumcheck polynomials).
    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::verify")]
    fn verify(
        &self,
        commitment_batch: Vec<Self::Commitment>,
        matrices_size_batch: &Vec<Vec<Dimensions>>,
        opening_point: &[Self::ExtensionField],
        opened_values_batch: &Vec<Vec<Vec<Self::ExtensionField>>>,
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        self.validate_verify_inputs(&commitment_batch, matrices_size_batch, opened_values_batch)?;

        let BasefoldProof {
            sumcheck_transcript,
            iopp_oracles,
            iopp_queries,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            out_of_domain_responses: _,
            final_poly,
            iopp_pruned,
            iopp_cross_round,
            iopp_cross_round_pruned,
        } = proof;
        // [D6] Split bundle into per-query openings and optional pruned variant.
        let BasefoldInputProof {
            per_query: query_openings,
            pruned: query_openings_pruned,
        } = query_openings;

        // [D6] Strong binding rule: pruned PCS input openings are valid together
        // with EITHER pruned IOPP queries (`iopp_pruned`) OR group-wise pruned
        // cross-round IOPP (`iopp_cross_round_pruned`). Without pruned input,
        // neither IOPP-pruned variant may be present.
        let iopp_is_pruned = iopp_pruned.is_some() || iopp_cross_round_pruned.is_some();
        match (iopp_is_pruned, query_openings_pruned.is_some()) {
            (true, true) | (false, false) => {}
            _ => return Err(BaseFoldError::InvalidInputError),
        }

        // IOPP mode detection + strong binding.
        //   both     = cross-round + path-pruning  (iopp_cross_round_pruned)
        //   cross    = cross-round only            (iopp_cross_round non-empty)
        //   pruned   = path-pruning only           (iopp_pruned)
        //   standard = neither
        let use_both = iopp_cross_round_pruned.is_some();
        let use_cross_round = !iopp_cross_round.is_empty();
        // Exactly one IOPP mode may be active.
        let active_modes = [use_both, use_cross_round, iopp_pruned.is_some()]
            .iter()
            .filter(|&&x| x)
            .count();
        if active_modes > 1 {
            return Err(BaseFoldError::InvalidInputError);
        }
        // Verifier config must match the prover's mode.
        if use_both {
            // both: input-batch openings must also be pruned (set together).
            if !self.use_cross_round || !self.use_path_pruning {
                return Err(BaseFoldError::InvalidInputError);
            }
            if !query_openings_pruned.is_some() {
                return Err(BaseFoldError::InvalidInputError);
            }
        } else {
            if use_cross_round && (iopp_pruned.is_some() || query_openings_pruned.is_some()) {
                return Err(BaseFoldError::InvalidInputError);
            }
            if use_cross_round != self.use_cross_round {
                return Err(BaseFoldError::InvalidInputError);
            }
        }

        if grinding_batching_witness.len() != 2 || grinding_query_witness.len() != 2 {
            return Err(BaseFoldError::InvalidInputError);
        }

        let num_vars = opening_point.len();

        // --- Flatten and group by log_height ---
        let flat_dims: Vec<DimAndNo> = matrices_size_batch
            .iter()
            .flat_map(|batch| batch.iter())
            .enumerate()
            .map(|(idx, dim)| DimAndNo {
                dim: dim.clone(),
                num: idx,
            })
            .collect();
        let flat_opened_values: Vec<&Vec<EF>> =
            opened_values_batch.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height =
            Self::group_dims_by_log_height(&flat_dims, &flat_opened_values);
        let log_max_height = matrices_by_log_height.keys().max().cloned().unwrap_or(0);
        debug_assert_eq!(log_max_height, num_vars);

        let size_index = MatricesSizeIndex::new(matrices_size_batch);

        // --- Phase 1: Compute claimed sums and query coefficients per height group ---
        let alpha: EF = challenger.sample_ext_element();
        let mut alpha_powers = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut claims_by_height: BTreeMap<usize, EF> = BTreeMap::new();
        let mut coefficients_by_height: BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>> =
            BTreeMap::new();

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (dims, values): (Vec<&DimAndNo>, Vec<&Vec<EF>>) =
                group.iter().map(|(d, v)| (*d, *v)).unzip();

            let coeffs: Vec<Vec<EF>> = values
                .iter()
                .map(|vals| vals.iter().map(|_| alpha_powers.next().unwrap()).collect())
                .collect();

            let claimed_sum: EF = values
                .iter()
                .zip(coeffs.iter())
                .flat_map(|(vals, cs)| vals.iter().zip(cs.iter()).map(|(v, c)| *v * *c))
                .sum::<EF>();
            claims_by_height.insert(log_height, claimed_sum);

            let indexed_coeffs: Vec<((usize, usize), Vec<EF>)> = coeffs
                .iter()
                .enumerate()
                .map(|(i, cs)| (size_index.find_position(dims[i].num), cs.clone()))
                .collect();
            coefficients_by_height.insert(log_height, indexed_coeffs);
        }

        // --- Verify batching proof of work ---
        challenger.observe(grinding_batching_witness[0]);
        if !challenger.check_witness(
            self.fri.grinding_bits_batching,
            grinding_batching_witness[1],
        ) {
            return Err(BaseFoldError::InvalidPowWitness);
        }

        // --- Phase 2: Verify sumcheck rounds (basefold: fold + simplified merge) ---
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut current_claim = claims_by_height
            .remove(&num_vars)
            .ok_or(BaseFoldError::SumcheckPhaseError)?;

        let min_log_height = matrices_by_log_height
            .keys()
            .min()
            .cloned()
            .unwrap_or(num_vars);
        let k = self.fri.log_final_poly_len.min(min_log_height);
        if k > 0 && final_poly.len() != (1usize << k) {
            return Err(BaseFoldError::InvalidInputError);
        }
        // [cross-round] Rebuild the same schedule the prover used. `present_heights`
        // must come from the same source (matrix dimensions) so prover/verifier
        // schedules are identical and the oracle observe order matches.
        let present_heights: std::collections::BTreeSet<usize> =
            matrices_by_log_height.keys().copied().collect();
        // The cross-round schedule applies to both plain cross-round and the
        // group-wise pruned (`both`) mode — both shift query indices per group.
        let schedule_cross_round = use_cross_round || use_both;
        let commit_schedule = compute_commit_schedule_cross_round(
            &present_heights,
            num_vars,
            k,
            schedule_cross_round,
        );
        // Number of committed IOPP oracles for this query proof must match the
        // schedule (group count). Guards against malformed proofs.
        if use_cross_round {
            for cr in iopp_cross_round.iter() {
                if cr.group_openings.len() != commit_schedule.len() {
                    return Err(BaseFoldError::InvalidInputError);
                }
            }
        }
        if use_both {
            let crp = iopp_cross_round_pruned.as_ref().unwrap();
            if crp.group_pruned.len() != commit_schedule.len()
                || crp.group_opened_rows.len() != commit_schedule.len()
            {
                return Err(BaseFoldError::InvalidInputError);
            }
        }
        let commit_start_rounds: std::collections::BTreeSet<usize> =
            commit_schedule.iter().map(|g| g.start_log_height).collect();

        if !iopp_oracles.is_empty() {
            challenger.observe(iopp_oracles[0].clone());
        }

        let mut folding_challenges: Vec<EF> = Vec::with_capacity(num_vars);
        let mut merge_betas: Vec<EF> = Vec::new();
        let mut eq_factor = EF::one();
        let mut oracle_idx: usize = 1;

        for round in (0..=num_vars).rev() {
            if round < num_vars && commit_start_rounds.contains(&round) {
                if oracle_idx < iopp_oracles.len() {
                    challenger.observe(iopp_oracles[oracle_idx].clone());
                    oracle_idx += 1;
                }
            } else if round == 0 && k == 0 {
                if oracle_idx < iopp_oracles.len() {
                    challenger.observe(iopp_oracles[oracle_idx].clone());
                    oracle_idx += 1;
                }
            } else if round == k && k > 0 && !commit_start_rounds.contains(&round) {
                for coeff in final_poly {
                    challenger.observe_ext_element(*coeff);
                }
            }
            if round == 0 {
                break;
            }

            // Normal sumcheck round: verify g(0) + g(1) = claim
            let uni_poly = poly_iter.next().ok_or(BaseFoldError::SumcheckPhaseError)?;
            if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != current_claim {
                return Err(BaseFoldError::SumcheckPhaseError);
            }
            uni_poly
                .coeffs
                .iter()
                .for_each(|c| challenger.observe_ext_element(*c));
            let r_fold: EF = challenger.sample_ext_element();
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate(&r_fold);

            // Accumulate eq factor
            let p_i = opening_point[round - 1];
            eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);

            // Basefold merge (deterministic, no merge poly)
            if let Some(branch_claim) = claims_by_height.remove(&(round - 1)) {
                let merge_beta: EF = challenger.sample_ext_element();
                merge_betas.push(merge_beta);

                current_claim = current_claim + merge_beta * branch_claim;

                eq_factor = EF::one();
            }
        }

        // --- Phase 3: Reconstruct combined EQ sum ---
        // With basefold little-endian folding, the final EQ is determined by the
        // folding challenges after the last merge (or all challenges if no merges).
        let fc_rev: Vec<EF> = folding_challenges.iter().rev().cloned().collect();

        // The final EQ only involves factors from after the last merge.
        // eq(p[0..min_height]; reversed tail of folding challenges)
        let combined_eq_sum = EqPolynomial::new(opening_point[..min_log_height].to_vec())
            .evaluate(&fc_rev[..min_log_height]);

        // --- Phase 4: Final codeword / polynomial check ---
        let combined_f_r: EF = current_claim / combined_eq_sum;

        if k == 0 {
            let expected_codeword = vec![combined_f_r; 1 << self.fri.log_blowup];
            let (expected_commitment, _) = self
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(expected_codeword, 2));

            let last_oracle = iopp_oracles
                .last()
                .ok_or(BaseFoldError::CommitmentCheckFailed)?;
            let last_bytes = bincode::serialize(last_oracle)
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
            let expected_bytes = bincode::serialize(&expected_commitment)
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
            if last_bytes != expected_bytes {
                return Err(BaseFoldError::CommitmentCheckFailed);
            }
        }

        let final_codeword: Option<Vec<EF>> = if k > 0 {
            let dft = EvalsDft::<F>::default();
            Some(self.encode_to_codeword(final_poly, &dft))
        } else {
            None
        };

        // --- Phase 5: Query proof of work ---
        challenger.observe(grinding_query_witness[0]);
        if !challenger.check_witness(self.fri.grinding_bits_query, grinding_query_witness[1]) {
            return Err(BaseFoldError::InvalidPowWitness);
        }

        // --- Phase 6: IOPP query verification ---
        let query_points: Vec<usize> = (0..self.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.fri.log_blowup))
            .collect();

        // [B6-5-step3] env-gated dispatch: pruned path uses single batched
        // `verify_queries_iopp_p3_pruned_basefold` (saves 17%+ proof bytes),
        // standard path keeps per-query loop.
        let all_queries_valid = if let Some(crp) = iopp_cross_round_pruned.as_ref() {
            // [both] cross-round + group-wise path-pruning.
            // Input-batch leaf sums come from the pruned input openings (same as
            // the pruned path); IOPP uses per-group `verify_batch_pruned` + local
            // cross-round fold via `query_to_unique_slot` lookup.
            let qop = match query_openings_pruned.as_ref() {
                Some(q) => q,
                None => return Err(BaseFoldError::InvalidInputError),
            };
            let num_rounds = matrices_size_batch.len();
            let n_queries = query_points.len();
            if qop.round_pruned.len() != num_rounds
                || qop.round_opened_values.len() != num_rounds
                || qop.query_to_unique_slot.len() != num_rounds
            {
                return Err(BaseFoldError::InvalidInputError);
            }

            // Step 1: input-batch pruned merkle verify (identical to pruned path).
            let mut input_ok = true;
            for (round_idx, batch_dims) in matrices_size_batch.iter().enumerate() {
                let codeword_dims: Vec<Dimensions> = batch_dims
                    .iter()
                    .map(|dim| Dimensions {
                        width: 0,
                        height: dim.height << self.fri.log_blowup,
                    })
                    .collect();
                let unique_opened = &qop.round_opened_values[round_idx];
                let q2u_round = &qop.query_to_unique_slot[round_idx];
                if q2u_round.len() != n_queries
                    || q2u_round.iter().any(|&s| (s as usize) >= unique_opened.len())
                {
                    input_ok = false;
                    break;
                }
                if self
                    .mmcs
                    .verify_batch_pruned(
                        &commitment_batch[round_idx],
                        &codeword_dims,
                        unique_opened,
                        &qop.round_pruned[round_idx],
                    )
                    .is_err()
                {
                    input_ok = false;
                    break;
                }
            }

            if !input_ok {
                false
            } else {
                // Step 2: per-query leaf sums (same formula as pruned path).
                let mut leaf_sums_per_query: Vec<BTreeMap<usize, EF>> =
                    Vec::with_capacity(n_queries);
                for q in 0..n_queries {
                    let sums: BTreeMap<usize, EF> = coefficients_by_height
                        .iter()
                        .map(|(&log_height, entries)| {
                            let sum = entries
                                .iter()
                                .map(|((batch_idx, mat_idx), coeffs)| {
                                    let slot = qop.query_to_unique_slot[*batch_idx][q] as usize;
                                    compute_dotproduct_mix(
                                        coeffs,
                                        &qop.round_opened_values[*batch_idx][slot][*mat_idx],
                                    )
                                })
                                .fold(EF::zero(), |acc, val| acc + val);
                            (log_height + self.fri.log_blowup, sum)
                        })
                        .collect();
                    leaf_sums_per_query.push(sums);
                }

                // Step 3: group-wise pruned IOPP verify.
                self.verify_cross_round_pruned_queries(
                    iopp_oracles.as_slice(),
                    &query_points,
                    &leaf_sums_per_query,
                    crp,
                    &commit_schedule,
                    &folding_challenges,
                    &merge_betas,
                    opening_point,
                    &combined_f_r,
                    final_codeword.as_deref(),
                )
                .is_ok()
            }
        } else if let Some(std_pruned) = iopp_pruned.as_ref() {
            // [D6-Audit-Fix1] env=1 path. Strong-binding guard above ensures
            // `query_openings_pruned` is also `Some(...)` here. We:
            //   (Step A1) verify each round's PCS opening once via
            //             `verify_batch_pruned` over the BFS-merged proof,
            //   (Step A2) reconstruct per-query opened values from
            //             `round_opened_values[r][q2u[r][q]]`,
            //   (Step A3) compute `leaf_sums_per_query` from the
            //             reconstructed values (same formula as the
            //             standard path), and
            //   (Step B)  feed the sums into the batched pruned IOPP
            //             verifier.
            let qop = query_openings_pruned
                .as_ref()
                .ok_or(BaseFoldError::InvalidInputError)?;
            let num_rounds = matrices_size_batch.len();
            let n_queries = query_points.len();

            if qop.round_pruned.len() != num_rounds
                || qop.round_opened_values.len() != num_rounds
                || qop.query_to_unique_slot.len() != num_rounds
            {
                return Err(BaseFoldError::InvalidInputError);
            }

            // Step A1: per-round pruned merkle verify.
            let mut per_round_ok = true;
            for (round_idx, batch_dims) in matrices_size_batch.iter().enumerate() {
                let codeword_dims: Vec<Dimensions> = batch_dims
                    .iter()
                    .map(|dim| Dimensions {
                        width: 0,
                        height: dim.height << self.fri.log_blowup,
                    })
                    .collect();
                let unique_opened = &qop.round_opened_values[round_idx];
                let q2u_round = &qop.query_to_unique_slot[round_idx];
                if q2u_round.len() != n_queries {
                    per_round_ok = false;
                    break;
                }
                // Slot indices must be in-range so the per-query sums below
                // never index-out-of-bounds.
                let unique_len = unique_opened.len();
                if q2u_round.iter().any(|&s| (s as usize) >= unique_len) {
                    per_round_ok = false;
                    break;
                }
                if self
                    .mmcs
                    .verify_batch_pruned(
                        &commitment_batch[round_idx],
                        &codeword_dims,
                        unique_opened,
                        &qop.round_pruned[round_idx],
                    )
                    .is_err()
                {
                    per_round_ok = false;
                    break;
                }
            }

            if !per_round_ok {
                false
            } else {
                // Step A2 + A3: rebuild per-query leaf_sums_by_log_height by
                // looking up each query's slot in the round's unique-leaves
                // table. Mirrors the standard path's per-query inner loop
                // but sources opened values from `round_opened_values`
                // instead of per-query `BatchOpening`s.
                let mut leaf_sums_per_query: Vec<BTreeMap<usize, EF>> =
                    Vec::with_capacity(n_queries);
                for q in 0..n_queries {
                    let sums: BTreeMap<usize, EF> = coefficients_by_height
                        .iter()
                        .map(|(&log_height, entries)| {
                            let sum = entries
                                .iter()
                                .map(|((batch_idx, mat_idx), coeffs)| {
                                    let slot = qop.query_to_unique_slot[*batch_idx][q] as usize;
                                    compute_dotproduct_mix(
                                        coeffs,
                                        &qop.round_opened_values[*batch_idx][slot][*mat_idx],
                                    )
                                })
                                .fold(EF::zero(), |acc, val| acc + val);
                            (log_height + self.fri.log_blowup, sum)
                        })
                        .collect();
                    leaf_sums_per_query.push(sums);
                }

                // Step B: single batched IOPP verify across all N queries.
                self.verify_queries_iopp_p3_pruned_basefold(
                    iopp_oracles.as_slice(),
                    &query_points,
                    &leaf_sums_per_query,
                    std_pruned,
                    &folding_challenges,
                    &merge_betas,
                    opening_point,
                    &combined_f_r,
                    final_codeword.as_deref(),
                )
                .is_ok()
            }
        } else if use_cross_round {
            // [cross-round] per-query, group-wise IOPP verification.
            iopp_cross_round
                .par_iter()
                .zip(query_openings.par_iter())
                .enumerate()
                .all(|(i, (cross_query, leaf_opening))| {
                    self.verify_cross_round_query_basefold(
                        &commitment_batch,
                        iopp_oracles.as_slice(),
                        query_points[i],
                        matrices_size_batch,
                        cross_query,
                        &commit_schedule,
                        leaf_opening,
                        &coefficients_by_height,
                        &folding_challenges,
                        &merge_betas,
                        opening_point,
                        &combined_f_r,
                        final_codeword.as_deref(),
                    )
                    .is_ok()
                })
        } else {
            iopp_queries
                .par_iter()
                .zip(query_openings.par_iter())
                .enumerate()
                .all(|(i, (query, leaf_opening))| {
                    self.verify_query_p3_batch_basefold(
                        &commitment_batch,
                        iopp_oracles.as_slice(),
                        query_points[i],
                        matrices_size_batch,
                        query,
                        leaf_opening,
                        &coefficients_by_height,
                        &folding_challenges,
                        &merge_betas,
                        opening_point,
                        &combined_f_r,
                        final_codeword.as_deref(),
                    )
                    .is_ok()
                })
        };

        if !all_queries_valid {
            return Err(BaseFoldError::FriFinalStepMisMatch);
        }

        Ok(())
    }
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    #[cfg(not(feature = "basefold"))]
    /// Verify a single IOPP query by folding the codeword through all rounds.
    ///
    /// At each round: merge in a new height group (if present), verify the Merkle opening,
    /// then fold via linear interpolation at the folding challenge.
    pub fn verify_iopp_query_p3(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        mut query_point: usize,
        leaf_sums_by_log_height: BTreeMap<usize, EF>,
        query_proof: &QueryProof<EF, FriMmcs>,
        folding_challenges: &[EF],
        merging_challenges: &[EF],
        expected_final_value: &EF,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.fri.log_blowup;

        // Prepend 1 so the first group (no merge) uses identity scaling
        let padded_merge_challenges: Vec<_> = std::iter::once(EF::one())
            .chain(merging_challenges.iter().cloned())
            .collect();

        let mut folded_eval = EF::zero();
        let mut merge_idx: usize = 0;
        let mut height_iter = leaf_sums_by_log_height.iter().rev().peekable();

        for (round, (&_r, commitment, opening)) in izip!(
            folding_challenges,
            iopp_commitments,
            &query_proof.commit_phase_openings
        )
        .enumerate()
        {
            let log_folded_height = log_max_height - round - 1;

            // Merge a new height group via linear interpolation if one enters at this level
            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                folded_eval += padded_merge_challenges[merge_idx] * (leaf_sum - folded_eval);
                merge_idx += 1;
            }

            // Verify Merkle opening for the sibling pair
            let sibling_index = query_point ^ 1;
            let pair_index = query_point >> 1;

            let mut pair_evals = vec![folded_eval; 2];
            pair_evals[sibling_index % 2] = opening.sibling_value;

            self.fri
                .mmcs
                .verify_batch(
                    commitment,
                    &[Dimensions {
                        width: 2,
                        height: 1 << log_folded_height,
                    }],
                    pair_index,
                    &[pair_evals.clone()],
                    &opening.opening_proof,
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            // Fold: interpolate the pair and evaluate at the folding challenge
            query_point = pair_index;
            let generator = EF::two_adic_generator(log_folded_height + 1)
                .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);

            let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
            let intercept = pair_evals[0] - slope * generator;
            folded_eval = intercept + slope * folding_challenges[round];
        }

        if folded_eval != *expected_final_value {
            return Err(BaseFoldError::FinalPolyMismatch);
        }
        Ok(())
    }

    #[cfg(not(feature = "basefold"))]
    /// Verify a single batch query: check leaf Merkle openings, compute per-height linear
    /// combinations from the opened leaves, then delegate to IOPP query verification.
    pub fn verify_query_p3_batch(
        &self,
        commitments: &[InputMmcs::Commitment],
        iopp_commitments: &[FriMmcs::Commitment],
        query_point: usize,
        matrices_size_batch: &[Vec<Dimensions>],
        query_proof: &QueryProof<EF, FriMmcs>,
        leaf_openings: &[BatchOpening<F, InputMmcs>],
        coefficients_by_height: &BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>>,
        folding_challenges: &[EF],
        merging_challenges: &[EF],
        expected_final_value: &EF,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        // Step 1: Verify leaf Merkle openings for each batch
        for (batch_dims, (commitment, opening)) in matrices_size_batch
            .iter()
            .zip(commitments.iter().zip(leaf_openings.iter()))
        {
            let max_log_height = batch_dims
                .iter()
                .map(|dim| log2_strict_usize(dim.height))
                .max()
                .unwrap_or(0);

            let codeword_dims: Vec<Dimensions> = batch_dims
                .iter()
                .map(|dim| Dimensions {
                    width: 0,
                    height: dim.height << self.fri.log_blowup,
                })
                .collect();

            self.mmcs
                .verify_batch(
                    commitment,
                    &codeword_dims,
                    query_point >> (folding_challenges.len() - max_log_height),
                    &opening.opened_values,
                    &opening.opening_proof,
                )
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        }

        // Step 2: Compute the linear combination of opened leaves per height group
        let leaf_sums_by_log_height: BTreeMap<usize, EF> = coefficients_by_height
            .iter()
            .map(|(&log_height, entries)| {
                let sum = entries
                    .par_iter()
                    .map(|((batch_idx, mat_idx), coeffs)| {
                        compute_dotproduct_mix(
                            coeffs,
                            &leaf_openings[*batch_idx].opened_values[*mat_idx],
                        )
                    })
                    .reduce(|| EF::zero(), |acc, val| acc + val);
                (log_height + self.fri.log_blowup, sum)
            })
            .collect();

        // Step 3: Verify IOPP folding
        self.verify_iopp_query_p3(
            iopp_commitments,
            query_point,
            leaf_sums_by_log_height,
            query_proof,
            folding_challenges,
            merging_challenges,
            expected_final_value,
        )
    }

    #[cfg(feature = "basefold")]
    /// Basefold IOPP query verification.
    ///
    /// Like `verify_iopp_query_p3` but uses the basefold merge formula:
    /// `F_new = eq_factor * F + merge_beta * G` instead of interpolation.
    pub fn verify_iopp_query_basefold(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        mut query_point: usize,
        leaf_sums_by_log_height: BTreeMap<usize, EF>,
        query_proof: &QueryProof<EF, FriMmcs>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.fri.log_blowup;

        let mut folded_eval = EF::zero();
        let mut merge_idx: usize = 0;
        let mut eq_factor = EF::one();
        let mut height_iter = leaf_sums_by_log_height.iter().rev().peekable();
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());

        for (round, (&_r, commitment, opening)) in izip!(
            folding_challenges,
            iopp_commitments,
            &query_proof.commit_phase_openings
        )
        .enumerate()
        {
            let log_folded_height = log_max_height - round - 1;

            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum;
                    eq_factor = EF::one();
                }
                merge_idx += 1;
            }

            let sibling_index = query_point ^ 1;
            let pair_index = query_point >> 1;

            let mut pair_evals = vec![folded_eval; 2];
            pair_evals[sibling_index % 2] = opening.sibling_value;

            self.fri
                .mmcs
                .verify_batch(
                    commitment,
                    &[Dimensions {
                        width: 2,
                        height: 1 << log_folded_height,
                    }],
                    pair_index,
                    &[pair_evals.clone()],
                    &opening.opening_proof,
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            query_point = pair_index;
            let generator = EF::two_adic_generator(log_folded_height + 1)
                .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);

            let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
            let intercept = pair_evals[0] - slope * generator;
            folded_eval = intercept + slope * folding_challenges[round];

            // Accumulate eq_factor for basefold merge
            let var_idx = num_vars - 1 - round;
            let p_i = opening_point[var_idx];
            let fc_i = folding_challenges[round];
            eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
        }

        let committed_rounds = query_proof.commit_phase_openings.len();
        for round in committed_rounds..num_vars {
            let log_folded_height = log_max_height - round - 1;

            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum;
                    eq_factor = EF::one();
                }
                merge_idx += 1;
            }

            let codeword = virtual_codeword
                .as_ref()
                .ok_or(BaseFoldError::FriFinalStepMisMatch)?;
            let pair_index = query_point >> 1;
            let even_idx = pair_index << 1;
            let pair_evals = [codeword[even_idx], codeword[even_idx | 1]];

            if round == committed_rounds && folded_eval != pair_evals[query_point & 1] {
                return Err(BaseFoldError::FriFinalStepMisMatch);
            }

            query_point = pair_index;
            let generator = EF::two_adic_generator(log_folded_height + 1)
                .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);
            let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
            let intercept = pair_evals[0] - slope * generator;
            folded_eval = intercept + slope * folding_challenges[round];

            if let Some(ref mut codeword) = virtual_codeword {
                *codeword = fold_codeword(codeword, folding_challenges[round]);
            }

            let var_idx = num_vars - 1 - round;
            let p_i = opening_point[var_idx];
            let fc_i = folding_challenges[round];
            eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
        }

        if folded_eval != *expected_final_value {
            return Err(BaseFoldError::FinalPolyMismatch);
        }
        Ok(())
    }

    /// [B6-5-step3] B-Stage 1' batched IOPP verify for basefold (no WHIR ood).
    ///
    /// Mirror of `verify_iopp_query_basefold` (per-query) collapsed to a
    /// single batched invocation that shares merkle paths across all N queries
    /// via `mmcs.verify_batch_pruned`. The per-(query, round) fold arithmetic
    /// (with `merge_betas` + `eq_factor`) is preserved verbatim from the
    /// per-query basefold path, so acceptance is equivalent to running
    /// `verify_iopp_query_basefold` N times independently.
    pub fn verify_queries_iopp_p3_pruned_basefold(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        query_points: &[usize],
        leaf_sums_by_log_height_per_query: &[BTreeMap<usize, EF>],
        iopp_pruned: &PrunedFriQueryProof<EF, FriMmcs>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let n = query_points.len();
        let num_vars = folding_challenges.len();
        let committed_rounds = iopp_commitments.len().min(num_vars);
        let log_max_height = num_vars + self.fri.log_blowup;

        if iopp_pruned.sibling_values.len() != n
            || iopp_pruned.round_pruned_proofs.len() < committed_rounds
            || leaf_sums_by_log_height_per_query.len() != n
        {
            return Err(BaseFoldError::InvalidInputError);
        }
        for sv in &iopp_pruned.sibling_values {
            if sv.len() < committed_rounds {
                return Err(BaseFoldError::InvalidInputError);
            }
        }

        // Per-query state, mirroring verify_iopp_query_basefold.
        let mut folded_evals: Vec<EF> = vec![EF::zero(); n];
        let mut merge_idxs: Vec<usize> = vec![0usize; n];
        let mut eq_factors: Vec<EF> = vec![EF::one(); n];
        let mut query_idxs: Vec<usize> = query_points.to_vec();
        let mut height_iters: Vec<_> = leaf_sums_by_log_height_per_query
            .iter()
            .map(|m| m.iter().rev().peekable())
            .collect();
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());

        for round in 0..committed_rounds {
            let log_folded_height = log_max_height - round - 1;

            // Step A: for each query, possibly merge a new height group, then
            // build the (left, right) row at this round.
            let mut pair_rows: Vec<[EF; 2]> = Vec::with_capacity(n);
            let mut pair_indices: Vec<usize> = Vec::with_capacity(n);

            for q in 0..n {
                if let Some((_, &leaf_sum)) =
                    height_iters[q].next_if(|(lh, _)| **lh == log_folded_height + 1)
                {
                    if merge_idxs[q] == 0 {
                        folded_evals[q] = leaf_sum;
                    } else {
                        folded_evals[q] = eq_factors[q] * folded_evals[q]
                            + merge_betas[merge_idxs[q] - 1] * leaf_sum;
                        eq_factors[q] = EF::one();
                    }
                    merge_idxs[q] += 1;
                }

                let sibling_bit = (query_idxs[q] ^ 1) & 1;
                let pair_idx = query_idxs[q] >> 1;
                let mut row = [folded_evals[q]; 2];
                row[sibling_bit] = iopp_pruned.sibling_values[q][round];

                pair_rows.push(row);
                pair_indices.push(pair_idx);
            }

            // Step B: dedup by pair_idx (colliding queries must produce same row).
            let mut sorted_unique: Vec<usize> = pair_indices.clone();
            sorted_unique.sort_unstable();
            sorted_unique.dedup();

            let mut row_by_pair: Vec<Option<[EF; 2]>> = vec![None; sorted_unique.len()];
            for q in 0..n {
                let k = sorted_unique
                    .binary_search(&pair_indices[q])
                    .expect("pair must be present");
                match &row_by_pair[k] {
                    None => row_by_pair[k] = Some(pair_rows[q]),
                    Some(existing) => {
                        if *existing != pair_rows[q] {
                            return Err(BaseFoldError::InvalidInputError);
                        }
                    }
                }
            }

            let opened_values_per_query: Vec<Vec<Vec<EF>>> = row_by_pair
                .iter()
                .map(|opt| {
                    let r = opt.expect("row must be filled");
                    vec![vec![r[0], r[1]]]
                })
                .collect();

            // Step C: single batched merkle verify for this round.
            let dims = [Dimensions {
                width: 2,
                height: 1 << log_folded_height,
            }];
            self.fri
                .mmcs
                .verify_batch_pruned(
                    &iopp_commitments[round],
                    &dims,
                    &opened_values_per_query,
                    &iopp_pruned.round_pruned_proofs[round],
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            // Step D: per-query fold (verbatim from verify_iopp_query_basefold).
            for q in 0..n {
                let pair_idx = pair_indices[q];
                query_idxs[q] = pair_idx;

                let pair_evals = pair_rows[q];
                let generator = EF::two_adic_generator(log_folded_height + 1)
                    .exp_u64(reverse_bits_len(query_idxs[q], log_folded_height) as u64);
                let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
                let intercept = pair_evals[0] - slope * generator;
                folded_evals[q] = intercept + slope * folding_challenges[round];

                // Accumulate eq_factor for basefold merge.
                let var_idx = num_vars - 1 - round;
                let p_i = opening_point[var_idx];
                let fc_i = folding_challenges[round];
                eq_factors[q] *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
            }
        }

        for round in committed_rounds..num_vars {
            let log_folded_height = log_max_height - round - 1;
            let codeword = virtual_codeword
                .as_ref()
                .ok_or(BaseFoldError::FriFinalStepMisMatch)?;

            for q in 0..n {
                if let Some((_, &leaf_sum)) =
                    height_iters[q].next_if(|(lh, _)| **lh == log_folded_height + 1)
                {
                    if merge_idxs[q] == 0 {
                        folded_evals[q] = leaf_sum;
                    } else {
                        folded_evals[q] = eq_factors[q] * folded_evals[q]
                            + merge_betas[merge_idxs[q] - 1] * leaf_sum;
                        eq_factors[q] = EF::one();
                    }
                    merge_idxs[q] += 1;
                }

                let pair_idx = query_idxs[q] >> 1;
                let even_idx = pair_idx << 1;
                let pair_evals = [codeword[even_idx], codeword[even_idx | 1]];

                if round == committed_rounds && folded_evals[q] != pair_evals[query_idxs[q] & 1] {
                    return Err(BaseFoldError::FriFinalStepMisMatch);
                }

                query_idxs[q] = pair_idx;
                let generator = EF::two_adic_generator(log_folded_height + 1)
                    .exp_u64(reverse_bits_len(query_idxs[q], log_folded_height) as u64);
                let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
                let intercept = pair_evals[0] - slope * generator;
                folded_evals[q] = intercept + slope * folding_challenges[round];

                let var_idx = num_vars - 1 - round;
                let p_i = opening_point[var_idx];
                let fc_i = folding_challenges[round];
                eq_factors[q] *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
            }

            if let Some(ref mut codeword) = virtual_codeword {
                *codeword = fold_codeword(codeword, folding_challenges[round]);
            }
        }

        // Final-poly check, per-query.
        for q in 0..n {
            if folded_evals[q] != *expected_final_value {
                return Err(BaseFoldError::FinalPolyMismatch);
            }
        }
        Ok(())
    }

    #[cfg(feature = "basefold")]
    /// Basefold batch query verification.
    pub fn verify_query_p3_batch_basefold(
        &self,
        commitments: &[InputMmcs::Commitment],
        iopp_commitments: &[FriMmcs::Commitment],
        query_point: usize,
        matrices_size_batch: &[Vec<Dimensions>],
        query_proof: &QueryProof<EF, FriMmcs>,
        leaf_openings: &[BatchOpening<F, InputMmcs>],
        coefficients_by_height: &BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        for (batch_dims, (commitment, opening)) in matrices_size_batch
            .iter()
            .zip(commitments.iter().zip(leaf_openings.iter()))
        {
            let max_log_height = batch_dims
                .iter()
                .map(|dim| log2_strict_usize(dim.height))
                .max()
                .unwrap_or(0);

            let codeword_dims: Vec<Dimensions> = batch_dims
                .iter()
                .map(|dim| Dimensions {
                    width: 0,
                    height: dim.height << self.fri.log_blowup,
                })
                .collect();

            self.mmcs
                .verify_batch(
                    commitment,
                    &codeword_dims,
                    query_point >> (folding_challenges.len() - max_log_height),
                    &opening.opened_values,
                    &opening.opening_proof,
                )
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        }

        let leaf_sums_by_log_height: BTreeMap<usize, EF> = coefficients_by_height
            .iter()
            .map(|(&log_height, entries)| {
                let sum = entries
                    .par_iter()
                    .map(|((batch_idx, mat_idx), coeffs)| {
                        compute_dotproduct_mix(
                            coeffs,
                            &leaf_openings[*batch_idx].opened_values[*mat_idx],
                        )
                    })
                    .reduce(|| EF::zero(), |acc, val| acc + val);
                (log_height + self.fri.log_blowup, sum)
            })
            .collect();

        self.verify_iopp_query_basefold(
            iopp_commitments,
            query_point,
            leaf_sums_by_log_height,
            query_proof,
            folding_challenges,
            merge_betas,
            opening_point,
            expected_final_value,
            final_codeword,
        )
    }

    /// [cross-round] Verify a single query against a cross-round proof.
    ///
    /// Mirrors [`Self::verify_query_p3_batch_basefold`] but walks the commit
    /// schedule group-by-group instead of round-by-round. For each group it
    /// verifies one wide (`2^log_folding`) Merkle row, then folds that row
    /// locally across `log_folding` rounds (see [`fold_row_cross_round`]).
    ///
    /// The leaf-sum injection and merge arithmetic match the per-round verifier:
    /// at the start of a group whose codeword height matches a leaf height, the
    /// running folded value is set (or merge-combined) from the leaf sum and the
    /// opened row's query position is checked against it.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_cross_round_query_basefold(
        &self,
        commitments: &[InputMmcs::Commitment],
        iopp_commitments: &[FriMmcs::Commitment],
        query_point: usize,
        matrices_size_batch: &[Vec<Dimensions>],
        cross_proof: &CrossRoundQueryProof<EF, FriMmcs>,
        commit_schedule: &[CommitGroup],
        leaf_openings: &[BatchOpening<F, InputMmcs>],
        coefficients_by_height: &BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        // --- Step 1: verify input MMCS leaf openings (identical to std path) ---
        for (batch_dims, (commitment, opening)) in matrices_size_batch
            .iter()
            .zip(commitments.iter().zip(leaf_openings.iter()))
        {
            let max_log_height = batch_dims
                .iter()
                .map(|dim| log2_strict_usize(dim.height))
                .max()
                .unwrap_or(0);

            let codeword_dims: Vec<Dimensions> = batch_dims
                .iter()
                .map(|dim| Dimensions {
                    width: 0,
                    height: dim.height << self.fri.log_blowup,
                })
                .collect();

            self.mmcs
                .verify_batch(
                    commitment,
                    &codeword_dims,
                    query_point >> (folding_challenges.len() - max_log_height),
                    &opening.opened_values,
                    &opening.opening_proof,
                )
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        }

        // --- Step 2: leaf sums per log_height (identical to std path) ---
        let leaf_sums_by_log_height: BTreeMap<usize, EF> = coefficients_by_height
            .iter()
            .map(|(&log_height, entries)| {
                let sum = entries
                    .iter()
                    .map(|((batch_idx, mat_idx), coeffs)| {
                        compute_dotproduct_mix(
                            coeffs,
                            &leaf_openings[*batch_idx].opened_values[*mat_idx],
                        )
                    })
                    .fold(EF::zero(), |acc, val| acc + val);
                (log_height + self.fri.log_blowup, sum)
            })
            .collect();

        // --- Step 3: walk the schedule, folding each group's row locally ---
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.fri.log_blowup;

        let mut folded_eval = EF::zero();
        let mut merge_idx: usize = 0;
        let mut eq_factor = EF::one();
        let mut height_iter = leaf_sums_by_log_height.iter().rev().peekable();
        let mut query_point = query_point;
        let mut round: usize = 0; // global folding rounds completed so far

        for (g, group) in commit_schedule.iter().enumerate() {
            let lf = group.log_folding;
            let step = &cross_proof.group_openings[g];
            if step.row_values.len() != (1usize << lf) {
                return Err(BaseFoldError::InvalidInputError);
            }
            // log_folded_height of this group's codeword (before its first fold).
            let log_fh0 = log_max_height - round - 1;

            // Leaf-sum injection at the group's starting height (if a leaf group
            // enters here). Matches the per-round verifier's height-match logic.
            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_fh0 + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum;
                    eq_factor = EF::one();
                }
                merge_idx += 1;
            }

            let pair_index_group = query_point >> lf;
            let local = query_point & ((1usize << lf) - 1);

            // Verify the wide Merkle row once for the whole group.
            let group_height = 1usize << (log_fh0 + 1 - lf); // codeword_len / 2^lf
            self.fri
                .mmcs
                .verify_batch(
                    &iopp_commitments[g],
                    &[Dimensions {
                        width: 1usize << lf,
                        height: group_height,
                    }],
                    pair_index_group,
                    &[step.row_values.clone()],
                    &step.opening_proof,
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            // Bind the running folded value to the opened row at the query's
            // own position. For the first group this is the leaf sum; for later
            // groups it is the value produced by the previous group's fold.
            // Either way IOPP consistency requires equality.
            if folded_eval != step.row_values[local] {
                return Err(BaseFoldError::FriFinalStepMisMatch);
            }

            // Locally fold the row across `lf` rounds.
            let challenges = &folding_challenges[round..round + lf];
            folded_eval =
                fold_row_cross_round(&step.row_values, pair_index_group, log_fh0 + 1, challenges);

            // Accumulate eq_factor over the lf rounds (matches per-round path).
            for (t, &fc_i) in challenges.iter().enumerate() {
                let var_idx = num_vars - 1 - (round + t);
                let p_i = opening_point[var_idx];
                eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
            }

            query_point = pair_index_group;
            round += lf;
        }

        // --- Step 4: early-stop virtual codeword tail (k > 0) ---
        // After the committed groups, `round == num_vars - k`. The remaining
        // `k` folds use the final-poly codeword, identical to the per-round path.
        let mut virtual_codeword = final_codeword.map(|c| c.to_vec());
        let committed_rounds = round;
        for round in committed_rounds..num_vars {
            let log_folded_height = log_max_height - round - 1;

            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum;
                    eq_factor = EF::one();
                }
                merge_idx += 1;
            }

            let codeword = virtual_codeword
                .as_ref()
                .ok_or(BaseFoldError::FriFinalStepMisMatch)?;
            let pair_index = query_point >> 1;
            let even_idx = pair_index << 1;
            let pair_evals = [codeword[even_idx], codeword[even_idx | 1]];

            if round == committed_rounds && folded_eval != pair_evals[query_point & 1] {
                return Err(BaseFoldError::FriFinalStepMisMatch);
            }

            query_point = pair_index;
            let generator = EF::two_adic_generator(log_folded_height + 1)
                .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);
            let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
            let intercept = pair_evals[0] - slope * generator;
            folded_eval = intercept + slope * folding_challenges[round];

            if let Some(ref mut codeword) = virtual_codeword {
                *codeword = fold_codeword(codeword, folding_challenges[round]);
            }

            let var_idx = num_vars - 1 - round;
            let p_i = opening_point[var_idx];
            let fc_i = folding_challenges[round];
            eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
        }

        if folded_eval != *expected_final_value {
            return Err(BaseFoldError::FinalPolyMismatch);
        }
        Ok(())
    }

    /// [both] Group-wise path-pruned cross-round IOPP verification across all N
    /// queries. For each commit group it runs ONE `verify_batch_pruned` over the
    /// BFS-merged wide rows, then folds each query's row locally (reusing
    /// [`fold_row_cross_round`]). `leaf_sums_per_query[q]` is the input-batch
    /// reduced sums (already reconstructed from the pruned input openings).
    #[allow(clippy::too_many_arguments)]
    pub fn verify_cross_round_pruned_queries(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        query_points: &[usize],
        leaf_sums_per_query: &[BTreeMap<usize, EF>],
        crp: &CrossRoundPrunedOpenings<EF, FriMmcs>,
        commit_schedule: &[CommitGroup],
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.fri.log_blowup;
        let n_queries = query_points.len();
        let num_groups = commit_schedule.len();

        if crp.group_pruned.len() != num_groups
            || crp.group_opened_rows.len() != num_groups
            || crp.query_to_unique_slot.len() != num_groups
        {
            return Err(BaseFoldError::InvalidInputError);
        }

        // Per-query running state (mirrors the per-query loop in
        // `verify_cross_round_query_basefold`, but advanced group-by-group).
        let mut folded_evals: Vec<EF> = vec![EF::zero(); n_queries];
        let mut merge_idx: Vec<usize> = vec![0; n_queries];
        let mut eq_factor: Vec<EF> = vec![EF::one(); n_queries];
        let mut height_iters: Vec<_> =
            leaf_sums_per_query.iter().map(|s| s.iter().rev().peekable()).collect();
        let mut q_points: Vec<usize> = query_points.to_vec();
        let mut round: usize = 0;

        for (g, group) in commit_schedule.iter().enumerate() {
            let lf = group.log_folding;
            let log_fh0 = log_max_height - round - 1;
            let group_height = 1usize << (log_fh0 + 1 - lf);
            let uniq_rows = &crp.group_opened_rows[g];
            for row in uniq_rows {
                if row.len() != (1usize << lf) {
                    return Err(BaseFoldError::InvalidInputError);
                }
            }

            // [soundness] Derive each query's row index and the sorted-unique set
            // OURSELVES (do not trust the prover's query_to_unique_slot). This
            // matches the per-round pruned verifier (`verify_queries_iopp_p3_pruned`)
            // which rebuilds `sorted_unique` from real indices. The opened rows
            // must be supplied in exactly this sorted order, so the slot a query
            // maps to is fixed by its true row index, not by a prover hint.
            let row_indices: Vec<usize> = q_points.iter().map(|&qp| qp >> lf).collect();
            let mut sorted_unique: Vec<usize> = row_indices.clone();
            sorted_unique.sort_unstable();
            sorted_unique.dedup();
            if uniq_rows.len() != sorted_unique.len() {
                return Err(BaseFoldError::InvalidInputError);
            }

            // ONE BFS-merged Merkle verify for all unique rows in this group.
            // verify_batch_pruned wants opened_values as [unique_slot][matrix][..];
            // each group commits a single matrix, so wrap each row in a Vec.
            // Rows are bound to `sorted_unique` order by the pruned proof's
            // strictly-ascending sorted_indices check inside verify_batch_pruned.
            let opened: Vec<Vec<Vec<EF>>> =
                uniq_rows.iter().map(|row| vec![row.clone()]).collect();
            self.fri
                .mmcs
                .verify_batch_pruned(
                    &iopp_commitments[g],
                    &[Dimensions { width: 1usize << lf, height: group_height }],
                    &opened,
                    &crp.group_pruned[g],
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            let challenges = &folding_challenges[round..round + lf];

            // Per-query: leaf-sum injection, binding check, local fold.
            for q in 0..n_queries {
                if let Some((_, &leaf_sum)) =
                    height_iters[q].next_if(|(lh, _)| **lh == log_fh0 + 1)
                {
                    if merge_idx[q] == 0 {
                        folded_evals[q] = leaf_sum;
                    } else {
                        folded_evals[q] =
                            eq_factor[q] * folded_evals[q] + merge_betas[merge_idx[q] - 1] * leaf_sum;
                        eq_factor[q] = EF::one();
                    }
                    merge_idx[q] += 1;
                }

                let pair_index_group = row_indices[q];
                let local = q_points[q] & ((1usize << lf) - 1);
                // Self-derived slot from the true row index (anti-cheat).
                let slot = sorted_unique
                    .binary_search(&pair_index_group)
                    .map_err(|_| BaseFoldError::InvalidInputError)?;
                let row = &uniq_rows[slot];

                // Bind the running folded value to the opened row at the query's
                // own position (same check as non-pruned cross-round).
                if folded_evals[q] != row[local] {
                    return Err(BaseFoldError::FriFinalStepMisMatch);
                }

                folded_evals[q] =
                    fold_row_cross_round(row, pair_index_group, log_fh0 + 1, challenges);

                for (t, &fc_i) in challenges.iter().enumerate() {
                    let var_idx = num_vars - 1 - (round + t);
                    let p_i = opening_point[var_idx];
                    eq_factor[q] *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
                }

                q_points[q] = pair_index_group;
            }
            round += lf;
        }

        // --- early-stop tail (per query), identical to non-pruned cross-round ---
        let committed_rounds = round;
        for q in 0..n_queries {
            let mut virtual_codeword = final_codeword.map(|c| c.to_vec());
            let mut query_point = q_points[q];
            let mut fe = folded_evals[q];
            let mut mi = merge_idx[q];
            let mut ef = eq_factor[q];
            for r in committed_rounds..num_vars {
                let log_folded_height = log_max_height - r - 1;
                if let Some((_, &leaf_sum)) =
                    height_iters[q].next_if(|(lh, _)| **lh == log_folded_height + 1)
                {
                    if mi == 0 {
                        fe = leaf_sum;
                    } else {
                        fe = ef * fe + merge_betas[mi - 1] * leaf_sum;
                        ef = EF::one();
                    }
                    mi += 1;
                }
                let codeword = virtual_codeword
                    .as_ref()
                    .ok_or(BaseFoldError::FriFinalStepMisMatch)?;
                let pair_index = query_point >> 1;
                let even_idx = pair_index << 1;
                let pair_evals = [codeword[even_idx], codeword[even_idx | 1]];
                if r == committed_rounds && fe != pair_evals[query_point & 1] {
                    return Err(BaseFoldError::FriFinalStepMisMatch);
                }
                query_point = pair_index;
                let generator = EF::two_adic_generator(log_folded_height + 1)
                    .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);
                let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
                let intercept = pair_evals[0] - slope * generator;
                fe = intercept + slope * folding_challenges[r];
                if let Some(ref mut cw) = virtual_codeword {
                    *cw = fold_codeword(cw, folding_challenges[r]);
                }
                let var_idx = num_vars - 1 - r;
                let p_i = opening_point[var_idx];
                let fc_i = folding_challenges[r];
                ef *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
            }
            if fe != *expected_final_value {
                return Err(BaseFoldError::FinalPolyMismatch);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::basefold::basefold_pcs::{BaseFoldPcs, BasefoldProof};
    use crate::basefold::mlpcs::MlPCS;
    use crate::basefold::sumcheck::SumcheckInstanceProof;
    use crate::utils::mlpoly::{MultilinearExtension, MultilinearPolynomial};
    use crate::utils::unipoly::UniPoly;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_field::AbstractExtensionField;
    use p3_field::AbstractField;
    use p3_field::{extension::BinomialExtensionField, Field};
    use p3_fri::FriConfig;
    use p3_fri::{BatchOpening, CommitPhaseProofStep, QueryProof};
    use p3_matrix::compressed::CompressedMatrix;
    #[cfg(feature = "basefold")]
    use p3_matrix::compressed::PaddingRow;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand_core::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    const D: u64 = 7;
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, D>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        FieldMerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;
    type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn get_col(mat: &RowMajorMatrix<F>, col: usize) -> Vec<F> {
        (0..mat.height()).map(|row| mat.get(row, col)).collect()
    }

    fn get_ef_col(mat: &RowMajorMatrix<EF>, col: usize) -> Vec<EF> {
        (0..mat.height()).map(|row| mat.get(row, col)).collect()
    }

    /// [Risk 1] Validate the cross-round local-fold arithmetic against the
    /// prover's `fold_codeword`. For a codeword of length `2^L`, folding it `k`
    /// times with challenges `[r_0,..,r_{k-1}]` and reading position `j >> k`
    /// must equal `fold_row_cross_round` applied to the row
    /// `codeword[(j>>k)<<k .. +2^k]` with the same challenges. We check this for
    /// every query position `j` and several `k`.
    #[test]
    fn test_cross_round_fold_matches_fold_codeword() {
        use crate::basefold::basefold_pcs::{fold_codeword, fold_row_cross_round};
        use p3_field::AbstractField;

        let mut rng = Xoroshiro128Plus::seed_from_u64(42);
        for log_len in 4..=8usize {
            let n = 1usize << log_len;
            // Random codeword over EF.
            let codeword: Vec<EF> = (0..n)
                .map(|_| {
                    let arr: [F; 4] = std::array::from_fn(|_| {
                        use rand::Rng;
                        F::from_canonical_u32(rng.gen::<u32>() % 1000)
                    });
                    EF::from_base_slice(&arr)
                })
                .collect();

            for k in 1..=(log_len - 1) {
                // Random folding challenges, one per local step.
                let challenges: Vec<EF> = (0..k)
                    .map(|_| {
                        let arr: [F; 4] = std::array::from_fn(|_| {
                            use rand::Rng;
                            F::from_canonical_u32(rng.gen::<u32>() % 1000 + 1)
                        });
                        EF::from_base_slice(&arr)
                    })
                    .collect();

                // Reference: fold the full codeword k times (prover path).
                let mut folded = codeword.clone();
                for &r in &challenges {
                    folded = fold_codeword(&folded, r);
                }
                // folded has length n >> k.

                // For every query position j, the cross-round local fold of the
                // row must match folded[j >> k].
                let row_len = 1usize << k;
                for j in 0..n {
                    let pair_index_group = j >> k;
                    let base = pair_index_group << k;
                    let row = &codeword[base..base + row_len];
                    let got = fold_row_cross_round(row, pair_index_group, log_len, &challenges);
                    assert_eq!(
                        got, folded[pair_index_group],
                        "mismatch at log_len={log_len} k={k} j={j}"
                    );
                }
            }
        }
    }

    /// Test open + verify (WHIR): multiple batches, different heights, no rotation.
    /// The last batch contains extension-field matrices (width divisible by 4).
    #[cfg(not(feature = "basefold"))]
    #[test]
    fn test_open_verify() {
        // Initialize tracing subscriber for performance profiling.
        // Control verbosity via RUST_LOG env var (e.g. RUST_LOG=debug).
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .try_init();

        let fri_num_queries = 90;
        let fri_blowup_bits = 1;
        let grinding_bits_query = 10;
        let grinding_bits_batching = 8;

        let batch_size = 3;

        // log_heights and widths must have the same dimensions.
        // The last batch is treated as EF matrices, so widths must be divisible by 4.
        let log_heights = vec![
            vec![19, 18, 17, 17, 16],
            vec![18, 16, 8],
            vec![12, 8, 10, 8],
        ];
        let widths = vec![vec![1, 4, 10, 8, 9], vec![3, 2, 5], vec![12, 16, 8, 12]];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let mut challenger = Challenger::new(perm.clone());
        let fri_config = FriConfig {
            log_blowup: fri_blowup_bits,
            num_queries: fri_num_queries,
            grinding_bits_query,
            grinding_bits_batching,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new(val_mmcs, fri_config);

        // Generate random matrices according to the specified dimensions
        let matrices: Vec<Vec<RowMajorMatrix<F>>> = widths
            .iter()
            .enumerate()
            .map(|(batch_idx, widths_in_batch)| {
                widths_in_batch
                    .iter()
                    .enumerate()
                    .map(|(i, &width)| {
                        let height = 1 << log_heights[batch_idx][i];
                        let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                        RowMajorMatrix::<F>::new(vec, width)
                    })
                    .collect()
            })
            .collect();

        // Wrap as CompressedMatrix (no actual padding in this test)
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = matrices
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
                    .collect()
            })
            .collect();

        // Interpret the last batch as extension-field matrices
        let ef_matrices: Vec<RowMajorMatrix<EF>> = {
            let last_batch = &matrices[batch_size - 1];
            let mut result = Vec::with_capacity(last_batch.len());
            for mat in last_batch {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d].clone();
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                result.push(RowMajorMatrix::<EF>::new(ef_values, ef_width));
            }
            result
        };

        let max_log_height = log_heights
            .iter()
            .flat_map(|v| v.iter())
            .max()
            .copied()
            .unwrap();

        // Single opening point of length max_log_height
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        // Compute opened_values
        let opened_values: Vec<Vec<Vec<EF>>> = {
            let mut result: Vec<Vec<Vec<EF>>> = matrices
                .iter()
                .take(batch_size - 1)
                .enumerate()
                .map(|(batch_idx, batch)| {
                    batch
                        .iter()
                        .enumerate()
                        .map(|(mat_idx, matrix)| {
                            (0..matrix.width())
                                .map(|j| {
                                    let poly = get_col(matrix, j);
                                    let poly = MultilinearPolynomial::from_evals(poly);
                                    let log_h = log_heights[batch_idx][mat_idx];
                                    poly.evaluate_mix(&opening_point[..log_h])
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let ef_batch: Vec<Vec<EF>> = ef_matrices
                .iter()
                .enumerate()
                .map(|(mat_idx, matrix)| {
                    (0..matrix.width())
                        .map(|j| {
                            let poly = get_ef_col(matrix, j);
                            let poly = MultilinearPolynomial::from_evals(poly);
                            let log_h = log_heights[batch_size - 1][mat_idx];
                            poly.evaluate_mix(&opening_point[..log_h])
                        })
                        .collect()
                })
                .collect();

            result.push(ef_batch);
            result
        };

        // Commit each batch (using compressed matrices)
        let (com, prover_data): (Vec<_>, Vec<_>) = (0..batch_size)
            .map(|i| pcs.commit(compressed_matrices[i].iter().collect()))
            .unzip();

        // Open (using compressed matrices)
        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut challenger,
            )
            .unwrap();

        // Verify
        let mut challenger2 = Challenger::new(perm.clone());

        let dims: Vec<Vec<Dimensions>> = widths
            .iter()
            .zip(log_heights.iter())
            .map(|(ws, hs)| {
                ws.iter()
                    .zip(hs.iter())
                    .map(|(&w, &log_h)| Dimensions {
                        width: w,
                        height: 1 << log_h,
                    })
                    .collect()
            })
            .collect();

        pcs.verify(
            com,
            &dims,
            &opening_point,
            &opened_values,
            &proof,
            &mut challenger2,
        )
        .unwrap();
    }

    /// Test open + verify (basefold): multiple batches, different heights, no rotation.
    /// Uses little-endian folding without WHIR out-of-domain sampling.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_basefold_open_verify() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .try_init();

        let fri_num_queries = 90;
        let fri_blowup_bits = 1;
        let grinding_bits_query = 10;
        let grinding_bits_batching = 8;

        let batch_size = 3;

        let log_heights = vec![
            vec![19, 18, 17, 17, 16],
            vec![18, 16, 8],
            vec![12, 8, 10, 8],
        ];
        let widths = vec![vec![1, 4, 10, 8, 9], vec![3, 2, 5], vec![12, 16, 8, 12]];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let mut challenger = Challenger::new(perm.clone());
        let fri_config = FriConfig {
            log_blowup: fri_blowup_bits,
            num_queries: fri_num_queries,
            grinding_bits_query,
            grinding_bits_batching,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new(val_mmcs, fri_config);

        let matrices: Vec<Vec<RowMajorMatrix<F>>> = widths
            .iter()
            .enumerate()
            .map(|(batch_idx, widths_in_batch)| {
                widths_in_batch
                    .iter()
                    .enumerate()
                    .map(|(i, &width)| {
                        let height = 1 << log_heights[batch_idx][i];
                        let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                        RowMajorMatrix::<F>::new(vec, width)
                    })
                    .collect()
            })
            .collect();

        // Wrap as CompressedMatrix (no actual padding in this test)
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = matrices
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
                    .collect()
            })
            .collect();

        let ef_matrices: Vec<RowMajorMatrix<EF>> = {
            let last_batch = &matrices[batch_size - 1];
            let mut result = Vec::with_capacity(last_batch.len());
            for mat in last_batch {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d].clone();
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                result.push(RowMajorMatrix::<EF>::new(ef_values, ef_width));
            }
            result
        };

        let max_log_height = log_heights
            .iter()
            .flat_map(|v| v.iter())
            .max()
            .copied()
            .unwrap();

        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        let opened_values: Vec<Vec<Vec<EF>>> = {
            let mut result: Vec<Vec<Vec<EF>>> = matrices
                .iter()
                .take(batch_size - 1)
                .enumerate()
                .map(|(batch_idx, batch)| {
                    batch
                        .iter()
                        .enumerate()
                        .map(|(mat_idx, matrix)| {
                            (0..matrix.width())
                                .map(|j| {
                                    let poly = get_col(matrix, j);
                                    let poly = MultilinearPolynomial::from_evals(poly);
                                    let log_h = log_heights[batch_idx][mat_idx];
                                    poly.evaluate_mix(&opening_point[..log_h])
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            let ef_batch: Vec<Vec<EF>> = ef_matrices
                .iter()
                .enumerate()
                .map(|(mat_idx, matrix)| {
                    (0..matrix.width())
                        .map(|j| {
                            let poly = get_ef_col(matrix, j);
                            let poly = MultilinearPolynomial::from_evals(poly);
                            let log_h = log_heights[batch_size - 1][mat_idx];
                            poly.evaluate_mix(&opening_point[..log_h])
                        })
                        .collect()
                })
                .collect();

            result.push(ef_batch);
            result
        };

        // Commit using compressed matrices
        let (com, prover_data): (Vec<_>, Vec<_>) = (0..batch_size)
            .map(|i| pcs.commit(compressed_matrices[i].iter().collect()))
            .unzip();

        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut challenger,
            )
            .unwrap();

        let mut challenger2 = Challenger::new(perm.clone());

        let dims: Vec<Vec<Dimensions>> = widths
            .iter()
            .zip(log_heights.iter())
            .map(|(ws, hs)| {
                ws.iter()
                    .zip(hs.iter())
                    .map(|(&w, &log_h)| Dimensions {
                        width: w,
                        height: 1 << log_h,
                    })
                    .collect()
            })
            .collect();

        pcs.verify(
            com,
            &dims,
            &opening_point,
            &opened_values,
            &proof,
            &mut challenger2,
        )
        .unwrap();
    }

    /// Test open + verify (basefold) with early-stop enabled.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_basefold_open_verify_early_stop() {
        let fri_num_queries = 10;
        let fri_blowup_bits = 1;
        let grinding_bits_query = 0;
        let grinding_bits_batching = 0;
        let log_final_poly_len = 3;

        let log_heights = vec![vec![8, 6]];
        let widths = vec![vec![4, 8]];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let mut challenger = Challenger::new(perm.clone());
        let fri_config = FriConfig {
            log_blowup: fri_blowup_bits,
            num_queries: fri_num_queries,
            grinding_bits_query,
            grinding_bits_batching,
            log_final_poly_len,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new(val_mmcs, fri_config);

        let matrices: Vec<Vec<RowMajorMatrix<F>>> = widths
            .iter()
            .enumerate()
            .map(|(batch_idx, widths_in_batch)| {
                widths_in_batch
                    .iter()
                    .enumerate()
                    .map(|(i, &width)| {
                        let height = 1 << log_heights[batch_idx][i];
                        let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                        RowMajorMatrix::<F>::new(vec, width)
                    })
                    .collect()
            })
            .collect();

        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = matrices
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
                    .collect()
            })
            .collect();

        let ef_matrices: Vec<RowMajorMatrix<EF>> = matrices[0]
            .iter()
            .map(|mat| {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d];
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                RowMajorMatrix::<EF>::new(ef_values, ef_width)
            })
            .collect();

        let max_log_height = log_heights
            .iter()
            .flat_map(|v| v.iter())
            .max()
            .copied()
            .unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        let opened_values: Vec<Vec<Vec<EF>>> = vec![ef_matrices
            .iter()
            .enumerate()
            .map(|(mat_idx, matrix)| {
                (0..matrix.width())
                    .map(|j| {
                        let poly = get_ef_col(matrix, j);
                        let poly = MultilinearPolynomial::from_evals(poly);
                        let log_h = log_heights[0][mat_idx];
                        poly.evaluate_mix(&opening_point[..log_h])
                    })
                    .collect()
            })
            .collect()];

        let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
            .iter()
            .map(|batch| pcs.commit(batch.iter().collect()))
            .unzip();

        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut challenger,
            )
            .unwrap();

        assert_eq!(proof.final_poly.len(), 1 << log_final_poly_len);
        assert_eq!(
            proof.iopp_oracles.len(),
            max_log_height - log_final_poly_len
        );
        if pcs.use_path_pruning {
            assert!(proof.iopp_pruned.is_some());
            assert!(proof.query_openings.pruned.is_some());
            assert!(proof.query_openings.per_query.is_empty());
        } else {
            assert!(proof.iopp_pruned.is_none());
        }

        let dims: Vec<Vec<Dimensions>> = widths
            .iter()
            .zip(log_heights.iter())
            .map(|(ws, hs)| {
                ws.iter()
                    .zip(hs.iter())
                    .map(|(&w, &log_h)| Dimensions {
                        width: w,
                        height: 1 << log_h,
                    })
                    .collect()
            })
            .collect();

        let mut challenger2 = Challenger::new(perm);
        pcs.verify(
            com,
            &dims,
            &opening_point,
            &opened_values,
            &proof,
            &mut challenger2,
        )
        .unwrap();
    }

    /// [cross-round] Schedule generation: verify the merge boundaries and gap
    /// merging for the canonical shrink-stage distribution {4,17,18,19}.
    #[test]
    fn test_cross_round_schedule() {
        use crate::basefold::basefold_pcs::{
            compute_commit_schedule_cross_round_capped, CommitGroup,
        };
        use std::collections::BTreeSet;

        let present: BTreeSet<usize> = [4, 17, 18, 19].into_iter().collect();

        // Unbounded (cap=0), no early stop: 19->18->17 each 1 round (adjacent
        // present heights), 17->4 merges the whole 16..5 gap (13 rounds), 4->0
        // folds 4.
        let greedy = compute_commit_schedule_cross_round_capped(&present, 19, 0, true, 0);
        assert_eq!(
            greedy,
            vec![
                CommitGroup { start_log_height: 19, log_folding: 1 },
                CommitGroup { start_log_height: 18, log_folding: 1 },
                CommitGroup { start_log_height: 17, log_folding: 13 },
                CommitGroup { start_log_height: 4, log_folding: 4 },
            ]
        );

        // Capped at 4: the wide 17->4 gap is split into cap-sized chunks
        // (17->13->9->5), then 5->4 (merge boundary at 4), then 4->0.
        let capped = compute_commit_schedule_cross_round_capped(&present, 19, 0, true, 4);
        assert_eq!(
            capped,
            vec![
                CommitGroup { start_log_height: 19, log_folding: 1 },
                CommitGroup { start_log_height: 18, log_folding: 1 },
                CommitGroup { start_log_height: 17, log_folding: 4 },
                CommitGroup { start_log_height: 13, log_folding: 4 },
                CommitGroup { start_log_height: 9, log_folding: 4 },
                CommitGroup { start_log_height: 5, log_folding: 1 },
                CommitGroup { start_log_height: 4, log_folding: 4 },
            ]
        );

        // Disabled => legacy per-round (19 groups, all log_folding=1).
        let baseline = compute_commit_schedule_cross_round_capped(&present, 19, 0, false, 4);
        assert_eq!(baseline.len(), 19);
        assert!(baseline.iter().all(|g| g.log_folding == 1));

        // Early stop k=4, capped at 4: never crosses k; the 17->4 gap splits
        // into 17->13->9->5->4 (stops at k=4).
        let sched_k4 = compute_commit_schedule_cross_round_capped(&present, 19, 4, true, 4);
        assert_eq!(
            sched_k4,
            vec![
                CommitGroup { start_log_height: 19, log_folding: 1 },
                CommitGroup { start_log_height: 18, log_folding: 1 },
                CommitGroup { start_log_height: 17, log_folding: 4 },
                CommitGroup { start_log_height: 13, log_folding: 4 },
                CommitGroup { start_log_height: 9, log_folding: 4 },
                CommitGroup { start_log_height: 5, log_folding: 1 },
            ]
        );

        // Every group must respect the cap.
        assert!(sched_k4.iter().all(|g| g.log_folding <= 4));
    }

    /// [cross-round] Compare baseline (per-round) vs cross-round on a single-batch
    /// input with a height gap. Asserts: (1) iopp commitment count drops,
    /// (2) both open+verify succeed, (3) proof sizes are reported.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_cross_round_vs_baseline() {
        // Single batch, EF matrices. Heights {10, 6, 4} => present {4,6,10},
        // gaps at 9,8,7 (between 10 and 6) and 5 (between 6 and 4).
        let log_heights: Vec<usize> = vec![10, 6, 4];
        let ef_widths: Vec<usize> = vec![2, 3, 1];

        let fri_num_queries = 20;
        let fri_blowup_bits = 1;
        let grinding_bits_query = 0;
        let grinding_bits_batching = 0;

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(123);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let make_fri = || FriConfig {
            log_blowup: fri_blowup_bits,
            num_queries: fri_num_queries,
            grinding_bits_query,
            grinding_bits_batching,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs.clone(),
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;

        // Build the base-field matrices (width = 4 * ef_width so they map to EF).
        let matrices: Vec<RowMajorMatrix<F>> = log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| {
                let height = 1usize << log_h;
                let width = efw * 4;
                let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                RowMajorMatrix::<F>::new(vec, width)
            })
            .collect();

        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = vec![matrices
            .iter()
            .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
            .collect()];

        let ef_matrices: Vec<RowMajorMatrix<EF>> = matrices
            .iter()
            .map(|mat| {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d];
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                RowMajorMatrix::<EF>::new(ef_values, ef_width)
            })
            .collect();

        let max_log_height = *log_heights.iter().max().unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        let opened_values: Vec<Vec<Vec<EF>>> = vec![ef_matrices
            .iter()
            .enumerate()
            .map(|(mat_idx, matrix)| {
                (0..matrix.width())
                    .map(|j| {
                        let poly = get_ef_col(matrix, j);
                        let poly = MultilinearPolynomial::from_evals(poly);
                        let log_h = log_heights[mat_idx];
                        poly.evaluate_mix(&opening_point[..log_h])
                    })
                    .collect()
            })
            .collect()];

        let dims: Vec<Vec<Dimensions>> = vec![log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| Dimensions {
                width: efw * 4,
                height: 1 << log_h,
            })
            .collect()];

        // Run open+verify for a given cross-round setting; return the proof.
        let run = |use_cross_round: bool| {
            let pcs = if use_cross_round {
                Pcs::new_with_cross_round(val_mmcs.clone(), make_fri())
            } else {
                Pcs::new(val_mmcs.clone(), make_fri())
            };
            let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
                .iter()
                .map(|batch| pcs.commit(batch.iter().collect()))
                .unzip();
            let mut ch_open = Challenger::new(perm.clone());
            let proof = pcs
                .open(
                    compressed_matrices.clone(),
                    prover_data,
                    &opening_point,
                    &opened_values,
                    &mut ch_open,
                )
                .unwrap();
            let mut ch_verify = Challenger::new(perm.clone());
            pcs.verify(com, &dims, &opening_point, &opened_values, &proof, &mut ch_verify)
                .expect("verify must succeed");
            proof
        };

        let proof_baseline = run(false);
        let proof_cross = run(true);

        // (1) iopp commitment count. For k==0 the layout is
        // (number of commit groups) + 1 final constant commitment.
        // Baseline per-round: 10 rounds + 1 = 11.
        // Cross-round, present {4,6,10}, num_vars=10: groups 10->6 (lf=4),
        // 6->4 (lf=2), 4->0 (lf=4) => 3 groups + 1 final = 4.
        println!(
            "[cross-round] iopp_oracles: baseline={} cross_round={}",
            proof_baseline.iopp_oracles.len(),
            proof_cross.iopp_oracles.len()
        );
        assert_eq!(proof_baseline.iopp_oracles.len(), 11);
        assert_eq!(proof_cross.iopp_oracles.len(), 4);
        assert!(
            proof_cross.iopp_oracles.len() < proof_baseline.iopp_oracles.len(),
            "cross-round must reduce the number of IOPP commitments"
        );
        assert!(proof_baseline.iopp_cross_round.is_empty());
        assert!(proof_cross.iopp_queries.is_empty());
        assert!(!proof_cross.iopp_cross_round.is_empty());

        // (2) proof size (informational — cross-round is a trade-off).
        let sz_baseline = bincode::serialized_size(&proof_baseline).unwrap();
        let sz_cross = bincode::serialized_size(&proof_cross).unwrap();
        println!(
            "[cross-round] proof size: baseline={sz_baseline} cross_round={sz_cross} (delta={})",
            sz_cross as i64 - sz_baseline as i64
        );
    }

    /// [cross-round] Early-stop (k > 0) compatibility: groups must not cross the
    /// final-polynomial boundary, and open+verify must succeed.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_cross_round_early_stop() {
        // Heights {10, 4} with log_final_poly_len=3 => present {4,10}, k=min(3,4)=3.
        // Schedule (cross-round): 10->4 (lf=6), then stop at 4 > k=3 is a merge
        // boundary; 4->3 (lf=1) is the last committed group; final poly covers k=3.
        let log_heights: Vec<usize> = vec![10, 4];
        let ef_widths: Vec<usize> = vec![2, 1];
        let log_final_poly_len = 3;

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(77);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 20,
            grinding_bits_query: 0,
            grinding_bits_batching: 0,
            log_final_poly_len,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new_with_cross_round(val_mmcs.clone(), fri_config);

        let matrices: Vec<RowMajorMatrix<F>> = log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| {
                let height = 1usize << log_h;
                let width = efw * 4;
                let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                RowMajorMatrix::<F>::new(vec, width)
            })
            .collect();
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = vec![matrices
            .iter()
            .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
            .collect()];

        let ef_matrices: Vec<RowMajorMatrix<EF>> = matrices
            .iter()
            .map(|mat| {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d];
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                RowMajorMatrix::<EF>::new(ef_values, ef_width)
            })
            .collect();

        let max_log_height = *log_heights.iter().max().unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();
        let opened_values: Vec<Vec<Vec<EF>>> = vec![ef_matrices
            .iter()
            .enumerate()
            .map(|(mat_idx, matrix)| {
                (0..matrix.width())
                    .map(|j| {
                        let poly = get_ef_col(matrix, j);
                        let poly = MultilinearPolynomial::from_evals(poly);
                        poly.evaluate_mix(&opening_point[..log_heights[mat_idx]])
                    })
                    .collect()
            })
            .collect()];
        let dims: Vec<Vec<Dimensions>> = vec![log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| Dimensions {
                width: efw * 4,
                height: 1 << log_h,
            })
            .collect()];

        let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
            .iter()
            .map(|batch| pcs.commit(batch.iter().collect()))
            .unzip();
        let mut ch_open = Challenger::new(perm.clone());
        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut ch_open,
            )
            .unwrap();

        assert_eq!(proof.final_poly.len(), 1 << log_final_poly_len);
        assert!(!proof.iopp_cross_round.is_empty());

        let mut ch_verify = Challenger::new(perm.clone());
        pcs.verify(com, &dims, &opening_point, &opened_values, &proof, &mut ch_verify)
            .expect("cross-round early-stop verify must succeed");
    }

    /// [cross-round] Capped split: a gap wider than the cap is split into
    /// several cap-sized groups. The boundaries introduced purely by the cap
    /// fall on heights with NO matrix injected, exercising the verifier's
    /// "continuation group" check (row[local] == previous group's fold). Uses
    /// present {2,16}, num_vars=16, cap=4 => 16->12->8->4 (continuations at 12,8)
    /// then 4->2 (merge boundary at 2).
    #[cfg(feature = "basefold")]
    #[test]
    fn test_cross_round_capped_split() {
        use crate::basefold::basefold_pcs::compute_commit_schedule_cross_round_capped;
        use std::collections::BTreeSet;

        // Sanity-check the schedule shape first.
        let present: BTreeSet<usize> = [2, 16].into_iter().collect();
        let sched = compute_commit_schedule_cross_round_capped(&present, 16, 0, true, 4);
        assert!(sched.iter().all(|g| g.log_folding <= 4));
        assert!(sched.len() >= 4, "wide gap must split into multiple groups");

        let log_heights: Vec<usize> = vec![16, 2];
        let ef_widths: Vec<usize> = vec![1, 1];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(2024);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 20,
            grinding_bits_query: 0,
            grinding_bits_batching: 0,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new_with_cross_round(val_mmcs.clone(), fri_config);

        let matrices: Vec<RowMajorMatrix<F>> = log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| {
                let height = 1usize << log_h;
                let width = efw * 4;
                let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                RowMajorMatrix::<F>::new(vec, width)
            })
            .collect();
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = vec![matrices
            .iter()
            .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
            .collect()];

        let ef_matrices: Vec<RowMajorMatrix<EF>> = matrices
            .iter()
            .map(|mat| {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d];
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                RowMajorMatrix::<EF>::new(ef_values, ef_width)
            })
            .collect();

        let max_log_height = *log_heights.iter().max().unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();
        let opened_values: Vec<Vec<Vec<EF>>> = vec![ef_matrices
            .iter()
            .enumerate()
            .map(|(mat_idx, matrix)| {
                (0..matrix.width())
                    .map(|j| {
                        let poly = get_ef_col(matrix, j);
                        let poly = MultilinearPolynomial::from_evals(poly);
                        poly.evaluate_mix(&opening_point[..log_heights[mat_idx]])
                    })
                    .collect()
            })
            .collect()];
        let dims: Vec<Vec<Dimensions>> = vec![log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| Dimensions {
                width: efw * 4,
                height: 1 << log_h,
            })
            .collect()];

        let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
            .iter()
            .map(|batch| pcs.commit(batch.iter().collect()))
            .unzip();
        let mut ch_open = Challenger::new(perm.clone());
        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut ch_open,
            )
            .unwrap();

        assert!(!proof.iopp_cross_round.is_empty());
        // Every opened group row must respect the cap (<= 2^4 values).
        for q in &proof.iopp_cross_round {
            for step in &q.group_openings {
                assert!(step.row_values.len() <= (1usize << 4));
            }
        }

        let mut ch_verify = Challenger::new(perm.clone());
        pcs.verify(com, &dims, &opening_point, &opened_values, &proof, &mut ch_verify)
            .expect("cross-round capped-split verify must succeed");
    }

    /// [both] cross-round + path-pruning (group-wise pruning) open/verify.
    /// With a height gap, cross-round merges rounds into wide groups AND
    /// path-pruning merges the per-query Merkle paths within each group.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_cross_round_with_path_pruning() {
        let log_heights: Vec<usize> = vec![10, 4];
        let ef_widths: Vec<usize> = vec![1, 1];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 20,
            grinding_bits_query: 0,
            grinding_bits_batching: 0,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        // both mode: cross-round AND path-pruning enabled together.
        let mut pcs = Pcs::new(val_mmcs.clone(), fri_config);
        pcs.use_path_pruning = true;
        pcs.use_cross_round = true;

        let matrices: Vec<RowMajorMatrix<F>> = log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| {
                let height = 1usize << log_h;
                let width = efw * 4;
                let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                RowMajorMatrix::<F>::new(vec, width)
            })
            .collect();
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = vec![matrices
            .iter()
            .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
            .collect()];

        let ef_matrices: Vec<RowMajorMatrix<EF>> = matrices
            .iter()
            .map(|mat| {
                let height = mat.height();
                let ef_width = mat.width() / 4;
                let mut ef_values = Vec::with_capacity(height * ef_width);
                for row in 0..height {
                    for ef_col in 0..ef_width {
                        let idx = row * mat.width() + ef_col * 4;
                        let mut arr = [F::zero(); 4];
                        for d in 0..4 {
                            arr[d] = mat.values[idx + d];
                        }
                        ef_values.push(EF::from_base_slice(&arr));
                    }
                }
                RowMajorMatrix::<EF>::new(ef_values, ef_width)
            })
            .collect();

        let max_log_height = *log_heights.iter().max().unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();
        let opened_values: Vec<Vec<Vec<EF>>> = vec![ef_matrices
            .iter()
            .enumerate()
            .map(|(mat_idx, matrix)| {
                (0..matrix.width())
                    .map(|j| {
                        let poly = get_ef_col(matrix, j);
                        let poly = MultilinearPolynomial::from_evals(poly);
                        poly.evaluate_mix(&opening_point[..log_heights[mat_idx]])
                    })
                    .collect()
            })
            .collect()];

        let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
            .iter()
            .map(|batch| pcs.commit(batch.iter().collect()))
            .unzip();
        let mut ch_open = Challenger::new(perm.clone());
        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut ch_open,
            )
            .expect("both-mode open must succeed");

        // both mode: IOPP uses cross-round-pruned, input uses pruned.
        assert!(proof.iopp_cross_round_pruned.is_some());
        assert!(proof.iopp_queries.is_empty());
        assert!(proof.iopp_cross_round.is_empty());
        assert!(proof.iopp_pruned.is_none());

        let dims: Vec<Vec<Dimensions>> = vec![log_heights
            .iter()
            .zip(ef_widths.iter())
            .map(|(&log_h, &efw)| Dimensions {
                width: efw * 4,
                height: 1 << log_h,
            })
            .collect()];

        let mut ch_verify = Challenger::new(perm.clone());
        pcs.verify(com, &dims, &opening_point, &opened_values, &proof, &mut ch_verify)
            .expect("both-mode (cross-round + path-pruning) verify must succeed");
    }

    /// Test open + verify (basefold) with compressed matrices that have actual padding rows.
    /// Simulates the real scenario where stored_height < total_height.
    #[cfg(feature = "basefold")]
    #[test]
    fn test_basefold_compressed_with_padding() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .try_init();

        let fri_num_queries = 90;
        let fri_blowup_bits = 1;
        let grinding_bits_query = 10;
        let grinding_bits_batching = 8;

        // Single batch, two matrices with different heights
        // stored_heights are smaller than total_heights (padding with zeros)
        let total_log_heights = vec![vec![12, 10]];
        let stored_heights = vec![vec![3000, 800]];
        let widths = vec![vec![3, 5]];

        let mut rng_bb = Xoroshiro128Plus::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut rng_bb,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let mut challenger = Challenger::new(perm.clone());
        let fri_config = FriConfig {
            log_blowup: fri_blowup_bits,
            num_queries: fri_num_queries,
            grinding_bits_query,
            grinding_bits_batching,
            log_final_poly_len: 0,
            mmcs: challenge_mmcs,
        };

        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new(val_mmcs, fri_config);

        // Build compressed matrices with zero padding
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = widths
            .iter()
            .enumerate()
            .map(|(batch_idx, ws)| {
                ws.iter()
                    .enumerate()
                    .map(|(mat_idx, &width)| {
                        let total_height = 1 << total_log_heights[batch_idx][mat_idx];
                        let stored = stored_heights[batch_idx][mat_idx];
                        let main_data: Vec<F> =
                            (0..stored * width).map(|_| rand::random()).collect();
                        let main = RowMajorMatrix::new(main_data, width);
                        CompressedMatrix::new(main, PaddingRow::Zero { width }, total_height)
                    })
                    .collect()
            })
            .collect();

        // Decompress for computing opened_values
        let full_matrices: Vec<Vec<RowMajorMatrix<F>>> = compressed_matrices
            .iter()
            .map(|batch| batch.iter().map(|cm| cm.decompress()).collect())
            .collect();

        let max_log_height = total_log_heights
            .iter()
            .flat_map(|v| v.iter())
            .max()
            .copied()
            .unwrap();

        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        let opened_values: Vec<Vec<Vec<EF>>> = full_matrices
            .iter()
            .enumerate()
            .map(|(batch_idx, batch)| {
                batch
                    .iter()
                    .enumerate()
                    .map(|(mat_idx, matrix)| {
                        (0..matrix.width())
                            .map(|j| {
                                let poly = get_col(matrix, j);
                                let poly = MultilinearPolynomial::from_evals(poly);
                                let log_h = total_log_heights[batch_idx][mat_idx];
                                poly.evaluate_mix(&opening_point[..log_h])
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let (com, prover_data): (Vec<_>, Vec<_>) = compressed_matrices
            .iter()
            .map(|batch| pcs.commit(batch.iter().collect()))
            .unzip();

        let proof = pcs
            .open(
                compressed_matrices,
                prover_data,
                &opening_point,
                &opened_values,
                &mut challenger,
            )
            .unwrap();

        let mut challenger2 = Challenger::new(perm.clone());

        let dims: Vec<Vec<Dimensions>> = widths
            .iter()
            .zip(total_log_heights.iter())
            .map(|(ws, hs)| {
                ws.iter()
                    .zip(hs.iter())
                    .map(|(&w, &log_h)| Dimensions {
                        width: w,
                        height: 1 << log_h,
                    })
                    .collect()
            })
            .collect();

        pcs.verify(
            com,
            &dims,
            &opening_point,
            &opened_values,
            &proof,
            &mut challenger2,
        )
        .unwrap();
    }

    #[cfg(feature = "basefold")]
    /// Generate a dummy proof for Basefold mode with the correct structure and sizes
    /// but all-zero data.
    ///
    /// Differs from WHIR dummy proof:
    /// - `out_of_domain_responses` is `None`
    /// - Sumcheck transcript has exactly `num_vars` uni_polys (no merge round polynomials;
    ///   basefold merges are simple linear combinations, not sumcheck rounds)
    fn generate_dummy_proof(
        log_heights: &Vec<Vec<usize>>,
        widths: &Vec<Vec<usize>>,
        fri_num_queries: usize,
        fri_blowup_bits: usize,
    ) -> BasefoldProof<EF, ChallengeMmcs, F, Vec<Vec<BatchOpening<F, ValMmcs>>>> {
        let num_batches = log_heights.len();
        let num_vars = log_heights
            .iter()
            .flat_map(|batch| batch.iter())
            .copied()
            .max()
            .unwrap_or(0);

        // --- sumcheck_transcript ---
        // Basefold: only num_vars uni_polys (degree 2 → 3 coefficients each).
        // Merges are simple linear combinations, not sumcheck rounds.
        let uni_polys: Vec<UniPoly<EF>> = (0..num_vars)
            .map(|_| UniPoly::from_coeff(vec![EF::zero(); 3]))
            .collect();

        // --- iopp_oracles ---
        let zero_commitment = <ValMmcs as p3_commit::Mmcs<F>>::Commitment::from([F::zero(); 8]);
        let iopp_oracles = vec![zero_commitment; num_vars + 1];

        // --- iopp_queries ---
        let iopp_queries: Vec<QueryProof<EF, ChallengeMmcs>> = (0..fri_num_queries)
            .map(|_| {
                let steps: Vec<CommitPhaseProofStep<EF, ChallengeMmcs>> = (0..num_vars)
                    .map(|round| {
                        let merkle_path_len = num_vars + fri_blowup_bits - round - 1;
                        CommitPhaseProofStep {
                            sibling_value: EF::zero(),
                            opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                        }
                    })
                    .collect();
                QueryProof {
                    commit_phase_openings: steps,
                }
            })
            .collect();

        // --- query_openings ---
        let query_openings: Vec<Vec<BatchOpening<F, ValMmcs>>> = (0..fri_num_queries)
            .map(|_| {
                (0..num_batches)
                    .map(|batch_idx| {
                        let opened_values: Vec<Vec<F>> = widths[batch_idx]
                            .iter()
                            .map(|&width| vec![F::zero(); width])
                            .collect();
                        let max_log_height_in_batch =
                            log_heights[batch_idx].iter().copied().max().unwrap_or(0);
                        let merkle_path_len = max_log_height_in_batch + fri_blowup_bits;
                        BatchOpening {
                            opened_values,
                            opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                        }
                    })
                    .collect()
            })
            .collect();

        // --- grinding witnesses ---
        let grinding_batching_witness: Vec<F> = vec![F::zero(); 2];
        let grinding_query_witness: Vec<F> = vec![F::zero(); 2];

        BasefoldProof {
            sumcheck_transcript: SumcheckInstanceProof::new(uni_polys),
            iopp_oracles,
            iopp_queries,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            out_of_domain_responses: None,
            final_poly: vec![],
            iopp_pruned: None,
            iopp_cross_round: Vec::new(),
            iopp_cross_round_pruned: None,
        }
    }

    #[cfg(feature = "basefold")]
    #[test]
    fn test_dummy_proof_size() {
        let log_heights = vec![
            vec![19, 18, 17, 17, 16],
            vec![18, 16, 8],
            vec![12, 8, 10, 8],
        ];
        let widths = vec![vec![1, 4, 10, 8, 9], vec![3, 2, 5], vec![12, 16, 8, 12]];

        let proof = generate_dummy_proof(&log_heights, &widths, 90, 1);

        let serialized = bincode::serialize(&proof).unwrap();
        println!(
            "Basefold dummy proof size: {} bytes ({:.2} KB)",
            serialized.len(),
            serialized.len() as f64 / 1024.0
        );

        let num_vars = 19;
        // Basefold: only num_vars uni_polys (no merge round polynomials)
        assert_eq!(proof.sumcheck_transcript.uni_polys.len(), num_vars);
        assert_eq!(proof.iopp_oracles.len(), num_vars + 1);
        assert_eq!(proof.iopp_queries.len(), 90);
        assert_eq!(proof.query_openings.len(), 90);
        assert!(proof.out_of_domain_responses.is_none());
        assert_eq!(proof.grinding_batching_witness.len(), 2);
        assert_eq!(proof.grinding_query_witness.len(), 2);
    }

    #[cfg(not(feature = "basefold"))]
    /// Generate a dummy proof with the correct structure and sizes but all-zero data.
    ///
    /// This is useful for estimating proof size, benchmarking serialization, or testing
    /// downstream consumers that only care about the proof shape.
    ///
    /// Params:
    /// - `log_heights`: `log_heights[batch][mat]` is the log2 of the height of each matrix
    /// - `widths`: `widths[batch][mat]` is the width (number of columns) of each matrix
    /// - `fri_num_queries`: number of FRI query rounds
    /// - `fri_blowup_bits`: log2 of the blowup factor
    fn generate_dummy_proof(
        log_heights: &Vec<Vec<usize>>,
        widths: &Vec<Vec<usize>>,
        fri_num_queries: usize,
        fri_blowup_bits: usize,
    ) -> BasefoldProof<EF, ChallengeMmcs, F, Vec<Vec<BatchOpening<F, ValMmcs>>>> {
        use std::collections::BTreeSet;

        let num_batches = log_heights.len();
        let num_vars = log_heights
            .iter()
            .flat_map(|batch| batch.iter())
            .copied()
            .max()
            .unwrap_or(0);

        // Collect distinct log_heights to determine the number of merge rounds
        let distinct_heights: BTreeSet<usize> = log_heights
            .iter()
            .flat_map(|batch| batch.iter().copied())
            .collect();
        let num_merge_rounds = if distinct_heights.is_empty() {
            0
        } else {
            distinct_heights.len() - 1
        };

        // --- sumcheck_transcript ---
        // Normal rounds: num_vars uni_polys (degree 2 → 3 coefficients each)
        // Merge rounds: one per additional height group
        let num_sumcheck_polys = num_vars + num_merge_rounds;
        let uni_polys: Vec<UniPoly<EF>> = (0..num_sumcheck_polys)
            .map(|_| UniPoly::from_coeff(vec![EF::zero(); 3]))
            .collect();

        // --- iopp_oracles ---
        // One commitment per round (num_vars rounds) + 1 for the final constant codeword
        let zero_commitment = <ValMmcs as p3_commit::Mmcs<F>>::Commitment::from([F::zero(); 8]);
        let iopp_oracles = vec![zero_commitment; num_vars + 1];

        // --- out_of_domain_responses ---
        let out_of_domain_responses: Vec<EF> = vec![EF::zero(); num_vars];

        // --- iopp_queries ---
        // Each query has num_vars CommitPhaseProofSteps.
        // Each step's Merkle path length = log2(codeword_height) at that round.
        // At round i (0-indexed), codeword_height = 2^(num_vars + blowup - i - 1),
        // so Merkle path length = num_vars + blowup - i - 1.
        let iopp_queries: Vec<QueryProof<EF, ChallengeMmcs>> = (0..fri_num_queries)
            .map(|_| {
                let steps: Vec<CommitPhaseProofStep<EF, ChallengeMmcs>> = (0..num_vars)
                    .map(|round| {
                        let merkle_path_len = num_vars + fri_blowup_bits - round - 1;
                        CommitPhaseProofStep {
                            sibling_value: EF::zero(),
                            opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                        }
                    })
                    .collect();
                QueryProof {
                    commit_phase_openings: steps,
                }
            })
            .collect();

        // --- query_openings ---
        // For each query, one BatchOpening per batch.
        // Each BatchOpening contains:
        //   - opened_values: one Vec<F> per matrix in that batch (length = width of that matrix)
        //   - opening_proof: Merkle path of length = max_log_height_in_batch + blowup
        let query_openings: Vec<Vec<BatchOpening<F, ValMmcs>>> = (0..fri_num_queries)
            .map(|_| {
                (0..num_batches)
                    .map(|batch_idx| {
                        let opened_values: Vec<Vec<F>> = widths[batch_idx]
                            .iter()
                            .map(|&width| vec![F::zero(); width])
                            .collect();
                        let max_log_height_in_batch =
                            log_heights[batch_idx].iter().copied().max().unwrap_or(0);
                        let merkle_path_len = max_log_height_in_batch + fri_blowup_bits;
                        BatchOpening {
                            opened_values,
                            opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                        }
                    })
                    .collect()
            })
            .collect();

        // --- grinding witnesses ---
        let grinding_batching_witness: Vec<F> = vec![F::zero(); 2];
        let grinding_query_witness: Vec<F> = vec![F::zero(); 2];

        BasefoldProof {
            sumcheck_transcript: SumcheckInstanceProof::new(uni_polys),
            iopp_oracles,
            iopp_queries,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            out_of_domain_responses: Some(out_of_domain_responses),
            final_poly: vec![],
            iopp_pruned: None,
            iopp_cross_round: Vec::new(),
            iopp_cross_round_pruned: None,
        }
    }

    #[cfg(not(feature = "basefold"))]
    #[test]
    fn test_dummy_proof_size() {
        let log_heights = vec![
            vec![19, 18, 17, 17, 16],
            vec![18, 16, 8],
            vec![12, 8, 10, 8],
        ];
        let widths = vec![vec![1, 4, 10, 8, 9], vec![3, 2, 5], vec![12, 16, 8, 12]];

        let proof = generate_dummy_proof(&log_heights, &widths, 90, 1);

        let serialized = bincode::serialize(&proof).unwrap();
        println!(
            "Dummy proof size: {} bytes ({:.2} KB)",
            serialized.len(),
            serialized.len() as f64 / 1024.0
        );

        // Verify structural correctness
        let num_vars = 19;
        let num_distinct_heights = 7; // {8, 10, 12, 16, 17, 18, 19}
        assert_eq!(
            proof.sumcheck_transcript.uni_polys.len(),
            num_vars + (num_distinct_heights - 1)
        );
        assert_eq!(proof.iopp_oracles.len(), num_vars + 1);
        assert_eq!(proof.iopp_queries.len(), 90);
        assert_eq!(proof.query_openings.len(), 90);
        assert_eq!(
            proof.out_of_domain_responses.as_ref().unwrap().len(),
            num_vars
        );
        assert_eq!(proof.grinding_batching_witness.len(), 2);
        assert_eq!(proof.grinding_query_witness.len(), 2);
    }
}
