use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    hash::FieldHasherVariable,
    sumcheck::SCBabyBearFriConfigVariable,
    BatchOpeningVariable, CircuitConfig, FriCommitPhaseProofStepVariable, FriQueryProofVariable,
};
use dt_core_executor::RiscvAirId;
use hashbrown::HashMap;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::PrunedBatchSchedule;

use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, Felt},
};
use dt_recursion_core::DIGEST_SIZE;
use dt_stark::sumcheck::{
    proof::{SCShardCommitment, SCShardOpenedValues},
    types::UniPolyEvals,
};
use pcs::utils::math::{is_power_of_two, Math};
#[cfg(feature = "verify")]
use std::fmt;
pub struct EqPolynomialVariable<C: CircuitConfig> {
    pub r: Vec<Ext<C::F, C::EF>>,
}

impl<C: CircuitConfig> EqPolynomialVariable<C> {
    pub fn new(r: Vec<Ext<C::F, C::EF>>) -> Self {
        Self { r }
    }

    pub fn evaluate(&self, builder: &mut Builder<C>, rx: &[Ext<C::F, C::EF>]) -> Ext<C::F, C::EF> {
        assert_eq!(self.r.len(), rx.len());
        let one: Ext<_, _> = builder.constant(C::EF::one());
        let mut result: Ext<_, _> = builder.constant(C::EF::one());
        for (r_i, rx_i) in self.r.iter().zip(rx.iter()) {
            let temp: Ext<_, _> = builder.eval(*r_i * *rx_i + (one - *r_i) * (one - *rx_i));
            result = builder.eval(result * temp);
        }
        result
    }

    fn evals(&self, builder: &mut Builder<C>) -> Vec<Ext<C::F, C::EF>> {
        let ell = self.r.len();

        let mut evals: Vec<Ext<C::F, C::EF>> = vec![builder.constant(C::EF::one()); ell.pow2()];
        let mut size = 1;
        for j in 0..ell {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = builder.eval(scalar * self.r[j]);
                evals[i - 1] = builder.eval(scalar - evals[i]);
            }
        }
        evals
    }
    //NOTE(yimin): USE EVALUATE at specific point.
    pub fn to_ml(&self, builder: &mut Builder<C>) -> MultilinearPolynomialVariable<C> {
        MultilinearPolynomialVariable::new(self.evals(builder))
    }
}
/// Eq(r_powers, opening_values)
pub struct EqPolyAlphaVariable<C: CircuitConfig> {
    pub r: Ext<C::F, C::EF>,
}
impl<C: CircuitConfig> EqPolyAlphaVariable<C> {
    pub fn new(r: Ext<C::F, C::EF>) -> Self {
        Self { r }
    }
}
impl<C: CircuitConfig> EqPolyAlphaVariable<C> {
    pub fn evaluate(
        &self,
        builder: &mut Builder<C>,
        inputs: &[Ext<C::F, C::EF>],
    ) -> Ext<C::F, C::EF> {
        assert!(!inputs.is_empty());
        /*
        alpha,alpha^2,alpha^4....
        */

        let one: Ext<C::F, C::EF> = builder.constant(C::EF::one());
        let mut r = self.r;

        let mut result = builder.eval(r * inputs[0] + (one - r) * (one - inputs[0]));

        for i in 1..inputs.len() {
            r = builder.eval(r * r);
            let temp: Ext<_, _> = builder.eval(r * inputs[i] + (one - r) * (one - inputs[i]));
            result = builder.eval(result * temp);
        }
        result
    }
}

#[derive(Debug, PartialEq)]
pub struct MultilinearPolynomialVariable<C: CircuitConfig> {
    pub evals: Vec<Ext<C::F, C::EF>>,
    num_vars: usize,
}

impl<C: CircuitConfig> Default for MultilinearPolynomialVariable<C> {
    fn default() -> Self {
        MultilinearPolynomialVariable::zero()
    }
}

impl<C: CircuitConfig> MultilinearPolynomialVariable<C> {
    pub fn new(evals: Vec<Ext<C::F, C::EF>>) -> Self {
        let num_vars = if evals.is_empty() {
            0
        } else {
            let num_vars = evals.len().ilog2() as usize;
            assert!(
                evals.len() == 1 || is_power_of_two(evals.len()),
                "Dense multi-linear polynomials must be made from a power of 2 (not {})",
                evals.len()
            );
            num_vars
        };

        MultilinearPolynomialVariable { num_vars, evals }
    }

    pub const fn zero() -> Self {
        MultilinearPolynomialVariable { num_vars: 0, evals: Vec::new() }
    }
}

#[derive(Clone)]
pub struct UniPolyVariable<C: CircuitConfig> {
    pub evals: Vec<Ext<C::F, C::EF>>,
}
#[cfg(feature = "verify")]
impl<C: CircuitConfig> fmt::Debug for UniPolyVariable<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let eval_val = self.evals.iter().map(|c| c.val).collect::<Vec<_>>();
        write!(f, "UniPoly {{ evals: {:?} }}", eval_val)
    }
}

impl<C: CircuitConfig> UniPolyVariable<C> {
    pub fn new(evals: Vec<Ext<C::F, C::EF>>) -> Self {
        Self { evals }
    }
    pub fn new_from_unipoly_evals(uni_poly: UniPolyEvals<Ext<C::F, C::EF>>) -> Self {
        Self { evals: uni_poly.evals }
    }
    pub fn degree(&self) -> usize {
        self.evals.len() - 1
    }
    pub fn eval_at_zero(&self) -> Ext<C::F, C::EF> {
        self.evals[0]
    }
    pub fn eval_at_one(&self, _builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        self.evals[1]
    }
    /// Evaluate the polynomial at an arbitrary point `r` using the O(d) barycentric
    /// Lagrange formula for consecutive-integer nodes {0, 1, ..., d}.
    pub fn evaluate(&self, builder: &mut Builder<C>, r: &Ext<C::F, C::EF>) -> Ext<C::F, C::EF>
    where
        Builder<C>: CircuitV2Builder<C>,
    {
        let d = self.evals.len() - 1;

        let weights = Self::barycentric_weights_consecutive(d);

        let mut r_minus: Vec<Ext<C::F, C::EF>> = Vec::with_capacity(d + 1);
        for i in 0..=d {
            let i_ef: Ext<_, _> = builder.constant(C::EF::from_canonical_usize(i));
            r_minus.push(builder.eval(*r - i_ef));
        }

        let mut prefix: Vec<Ext<C::F, C::EF>> = Vec::with_capacity(d + 2);
        prefix.push(builder.constant(C::EF::one()));
        for i in 0..=d {
            let next = builder.eval(prefix[i] * r_minus[i]);
            prefix.push(next);
        }

        let mut suffix: Vec<Ext<C::F, C::EF>> = vec![builder.constant(C::EF::one()); d + 2];
        for i in (0..=d).rev() {
            suffix[i] = builder.eval(suffix[i + 1] * r_minus[i]);
        }

        let mut result: Ext<_, _> = builder.constant(C::EF::zero());
        for i in 0..=d {
            let w_i: Ext<_, _> = builder.constant(weights[i]);
            let term: Ext<_, _> = builder.eval(self.evals[i] * w_i * prefix[i] * suffix[i + 1]);
            result = builder.eval(result + term);
        }
        result
    }

    /// Combined claim check + evaluation for linear rounds.
    /// Verifies claim == f(0) + f(1) and returns f(challenge).
    pub fn verify_and_evaluate(
        &self,
        builder: &mut Builder<C>,
        challenge: &Ext<C::F, C::EF>,
        claim: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF>
    where
        Builder<C>: CircuitV2Builder<C>,
    {
        let sum: Ext<_, _> = builder.eval(self.evals[0] + self.evals[1]);
        builder.assert_ext_eq(sum, claim);
        self.evaluate(builder, challenge)
    }

    /// Evaluate treating stored data as **coefficients** (Horner's method).
    /// Used by the PCS basefold path where `UniPoly` stores coefficients, not evaluations.
    pub fn evaluate_horner(
        &self,
        builder: &mut Builder<C>,
        r: &Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF>
    where
        Builder<C>: CircuitV2Builder<C>,
    {
        let mut result: Ext<_, _> = builder.constant(C::EF::zero());
        for coeff in self.evals.iter().rev() {
            result = builder.eval(result * *r + *coeff);
        }
        result
    }

    /// f(1) = sum of all coefficients (Horner at x = 1).
    /// Used by the PCS basefold path.
    pub fn eval_at_one_horner(&self, builder: &mut Builder<C>) -> Ext<C::F, C::EF>
    where
        Builder<C>: CircuitV2Builder<C>,
    {
        let mut sum: Ext<_, _> = builder.constant(C::EF::zero());
        for coeff in self.evals.iter() {
            sum = builder.eval(sum + *coeff);
        }
        sum
    }

    pub fn observe_into<H: FieldChallengerVariable<C, C::Bit>>(
        &self,
        builder: &mut Builder<C>,
        challenger: &mut H,
    ) {
        for eval in self.evals.iter() {
            let eval_felt = C::ext2felt(builder, *eval);
            challenger.observe_slice(builder, eval_felt);
        }
    }

    fn barycentric_weights_consecutive(d: usize) -> Vec<C::EF> {
        let mut weights = Vec::with_capacity(d + 1);
        let mut factorial_d = C::EF::one();
        for j in 1..=d {
            factorial_d *= C::EF::from_canonical_usize(j);
        }
        let sign_d = if d.is_multiple_of(2) { C::EF::one() } else { C::EF::neg_one() };
        weights.push((sign_d * factorial_d).inverse());

        for i in 1..=d {
            let prev = *weights.last().unwrap();
            let ratio = C::EF::neg_one() *
                C::EF::from_canonical_usize(d - i + 1) *
                C::EF::from_canonical_usize(i).inverse();
            weights.push(prev * ratio);
        }
        weights
    }
}

#[derive(Clone)]
pub struct SumcheckInstanceProofVariable<C: CircuitConfig> {
    pub uni_polys: Vec<UniPolyVariable<C>>,
}

#[derive(Clone)]
pub struct StackingReductionProofVariable<C: CircuitConfig> {
    pub sumcheck: SumcheckInstanceProofVariable<C>,
}
#[cfg(feature = "verify")]
impl<C: CircuitConfig> fmt::Debug for SumcheckInstanceProofVariable<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let uni_poly_val = self
            .uni_polys
            .iter()
            .map(|p| p.evals.iter().map(|c| c.val).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        write!(f, "SC_Proof {{ uni_polys: {:?} }}", uni_poly_val)
    }
}
/// [SS] Per-round path-pruned PCS input openings in circuit form.
///
/// Replaces `query_openings: Vec<Vec<BatchOpeningVariable>>` when env=1.
/// Contains the BFS-merged merkle proofs + opened values + q2u mapping
/// that let the circuit verify leaf openings without per-query auth paths.
#[derive(Clone)]
pub struct InputPrunedVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    /// Per-round pruned batch proof (length == num_batches).
    /// Each entry authenticates ALL unique query openings against the
    /// round's input MMCS commitment via a single BFS-merged merkle proof.
    pub round_pruned: Vec<PrunedBatchProofVariable<C, H>>,
    /// Per-round per-unique-slot opened values in circuit form.
    /// `round_opened_values[batch_idx][unique_slot][mat_idx]` is a
    /// `Vec<Felt<C::F>>` of the leaf data for that matrix.
    /// Layout: `[num_batches][num_unique_per_batch][num_mats][width]`.
    pub round_opened_values: Vec<Vec<Vec<Vec<Felt<C::F>>>>>,
    /// Per-round query→unique-slot mapping (native const hints).
    /// `query_to_unique_slot[batch_idx][query_idx]` gives the index
    /// into `round_opened_values[batch_idx]` for that query.
    pub query_to_unique_slot: Vec<Vec<usize>>,
}

#[derive(Clone)]
pub struct BasefoldProofVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub stack_log_height: Option<usize>,
    pub sumcheck_transcript: SumcheckInstanceProofVariable<C>,
    pub iopp_oracles: Vec<H::DigestVariable>,
    pub ood_values: Vec<Ext<C::F, C::EF>>,
    pub iopp_queries: Vec<FriQueryProofVariable<C, H>>,
    pub round_iopp: Option<WhirRoundQueryProofVariable<C, H>>,
    /// Path-pruned IOPP query proof (mutually exclusive with iopp_queries when populated).
    pub iopp_pruned: Option<PrunedFriQueryProofVariable<C, H>>,
    pub query_openings: Vec<Vec<BatchOpeningVariable<C, H>>>,
    /// [SS] Path-pruned PCS input openings. When `Some`, the circuit uses
    /// `verify_batch_pruned` per batch instead of N independent `verify_batch`
    /// calls, and extracts per-query opened values via `query_to_unique_slot`.
    /// Mutually exclusive with non-empty `query_openings`.
    pub input_pruned: Option<InputPrunedVariable<C, H>>,
    pub grinding_batching_witness: Vec<Felt<C::F>>,
    pub grinding_query_witness: Vec<Felt<C::F>>,
    /// Final polynomial coefficients for Basefold early stopping.
    pub final_poly: Vec<Ext<C::F, C::EF>>,
    pub stacking_reduction: Option<StackingReductionProofVariable<C>>,
}

#[derive(Clone)]
pub struct WhirRoundQueryProofVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub rounds: Vec<WhirIoppRoundVariable<C, H>>,
    pub pruned: Option<WhirRoundPrunedQueryProofVariable<C, H>>,
    pub query_witnesses: Vec<Vec<Felt<C::F>>>,
    pub folding_witnesses: Vec<Vec<Felt<C::F>>>,
}

#[derive(Clone)]
pub struct WhirRoundPrunedQueryProofVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub rounds: Vec<WhirPrunedIoppRoundVariable<C, H>>,
}

#[derive(Clone)]
pub struct WhirPrunedIoppRoundVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub pruned_proof: PrunedBatchProofVariable<C, H>,
    pub opened_rows: Vec<Vec<Vec<Ext<C::F, C::EF>>>>,
    pub query_to_unique_slot: Vec<usize>,
}

#[derive(Clone)]
pub struct WhirIoppRoundVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub query_proofs: Vec<WhirIoppRoundQueryVariable<C, H>>,
}

#[derive(Clone)]
pub struct WhirIoppRoundQueryVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    pub current_opening: FriCommitPhaseProofStepVariable<C, H>,
}
#[cfg(feature = "verify")]
impl<C: CircuitConfig, H: FieldHasherVariable<C>> fmt::Debug for BasefoldProofVariable<C, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Basefold_Proof_Var {{ unipolys: {:?} }}", self.sumcheck_transcript)
    }
}

// ============================================================
// Path-pruning extension (C-Stage 1).
//
// Mirrors `pcs::basefold::basefold_pcs::BasefoldProof::iopp_pruned` and
// `p3_fri::PrunedFriQueryProof` in circuit-witness form. The recursion
// circuit consumes these to verify N FRI queries against ONE shared
// path-pruned merkle proof per commit-phase round, instead of N
// independent merkle openings.
//
// Soundness invariant: the BFS layer-walk schedule (per_layer_pair_merge_mask
// + per_layer_sibling_pos) is a hint emitted by the prover; the circuit
// MUST cross-check it against `sorted_indices` (LSBs) before consuming.
// See `p3_merkle_tree::pruned::compute_bfs_schedule` for the native
// reference, and `pcs.rs::verify_batch_pruned` for the circuit gadget.
// ============================================================

/// Circuit-side analogue of `p3_merkle_tree::pruned::PrunedBatchProof`,
/// extended with prover-supplied BFS schedule hints.
#[derive(Clone)]
pub struct PrunedBatchProofVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    /// Queried leaf indices, strictly ascending (length == N).
    pub sorted_indices: Vec<Felt<C::F>>,
    /// All emitted sibling digests in BFS order.
    pub siblings: Vec<H::DigestVariable>,
    /// Per-layer count of emitted siblings (length == log_max_height).
    pub layer_widths: Vec<Felt<C::F>>,

    // ===== Prover-supplied BFS schedule hints (for static unrolling) =====
    /// Per-layer count of `active` entries entering the layer
    /// (length == log_max_height + 1; entry 0 == N, entry log_max_height == 1).
    pub per_layer_step_count: Vec<Felt<C::F>>,
    /// Per-(layer, step) BFS decision: 1 == PairMerge, 0 == ConsumeSibling.
    /// Per-(layer, step) BFS decision: encoded as Felt 0/1 hint; Stage2 will
    /// convert to native Bit via per-config bit_from_felt and CONSTRAIN it from sorted_indices
    /// LSB.
    pub per_layer_pair_merge_mask: Vec<Vec<Felt<C::F>>>,
    /// Per-(layer, step) side of the active node for `ConsumeSibling`.
    /// `false` means the active digest is the left child, `true` means it is
    /// the right child. This is derived from the native BFS schedule.
    pub per_layer_sibling_pos: Vec<Vec<bool>>,
    /// Native-side BFS schedule (PairMerge / ConsumeSibling list), used by the
    /// circuit gadget for static unrolling. Not part of the witness stream.
    /// None for dummy proofs (which must not flow through verify_batch_pruned).
    pub native_schedule: Option<PrunedBatchSchedule>,
}

/// Circuit-side analogue of `p3_fri::PrunedFriQueryProof`.
#[derive(Clone)]
pub struct PrunedFriQueryProofVariable<C: CircuitConfig, H: FieldHasherVariable<C>> {
    /// One pruned-batch proof per commit-phase round
    /// (length == number of commit-phase rounds).
    pub round_pruned_proofs: Vec<PrunedBatchProofVariable<C, H>>,
    /// Optional full opened rows for cross-round FRI openings.
    ///
    /// Empty for legacy arity-2 pruning. When populated, the shape is
    /// `[round][unique_slot][matrix_idx][row_values]`; FRI commit phases
    /// have one matrix, so `matrix_idx == 0`.
    pub round_opened_values: Vec<Vec<Vec<Vec<Ext<C::F, C::EF>>>>>,
    /// Per-(query, round) sibling values for the fold arithmetic
    /// (outer length == N, inner length == number of commit-phase rounds).
    pub sibling_values: Vec<Vec<Ext<C::F, C::EF>>>,
    /// A-fix: per-round mapping query_idx -> unique slot in the round's
    /// `PrunedBatchProof.sorted_indices` (sorted+deduped).
    /// Outer length == num_rounds, inner length == n_queries.
    /// Values are native `usize` const hints in `[0, M_r)` where
    /// `M_r = round_pruned_proofs[r].sorted_indices.len() <= N`.
    /// Used by `verify_iopp_query_p3_pruned` to merge N per-query leaves
    /// into M unique-slot leaves before `verify_batch_pruned`, mirroring
    /// the native `verify_challenges_pruned`'s `row_by_pair[k]` indexing.
    pub query_to_unique_slot: Vec<Vec<usize>>,
}

/// Reference: [dt_core::stark::sumcheck::SCStarkVerifyingKey]
#[derive(Clone, Debug)]
pub struct SCVerifyingKeyVariable<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>>
{
    pub commitment: SC::DigestVariable,
    pub pc_start: Felt<C::F>,
    pub program_global_seed: [[Felt<C::F>; 11]; 3],
    pub program_global_digest: [Felt<C::F>; 32],
    pub has_global_owner: bool,
    pub chip_information: Vec<(String, Dimensions)>,
    pub chip_ordering: HashMap<String, usize>,
    pub constraints_map: HashMap<String, usize>,
}

#[derive(Clone)]
pub struct SumcheckProofVariable<C: CircuitConfig> {
    pub unipolys: Vec<UniPolyVariable<C>>,
}

/// Reference: [dt_core::stark::SCShardProof]
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub struct SCShardProofVariable<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>> {
    pub commitment: SCShardCommitment<SC::DigestVariable>,
    pub opened_values: SCShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    pub opening_proof: BasefoldProofVariable<C, SC>,
    pub sumcheck_proof: SumcheckProofVariable<C>,
    pub dimensions: Vec<Vec<Dimensions>>,
    pub chip_ordering: HashMap<String, usize>,
    pub public_values: Vec<Felt<C::F>>,
}
impl<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>>
    SCShardProofVariable<C, SC>
{
    pub fn contains_cpu(&self) -> bool {
        RiscvAirId::cpu().iter().any(|id: &RiscvAirId| {
            let name = id.as_str();
            self.chip_ordering.contains_key(name) ||
                self.chip_ordering.contains_key(&format!("{name}PolyAir"))
        })
    }

    pub fn log_degree_cpu(&self) -> usize {
        self.opened_values.chips.iter().map(|c| c.log_height).max().unwrap_or(0)
    }

    pub fn contains_memory_init(&self) -> bool {
        self.chip_ordering.contains_key("MemoryGlobalInit") ||
            self.chip_ordering.contains_key("MemoryGlobalInitPolyAir")
    }

    pub fn contains_memory_finalize(&self) -> bool {
        self.chip_ordering.contains_key("MemoryGlobalFinalize") ||
            self.chip_ordering.contains_key("MemoryGlobalFinalizePolyAir")
    }
}

#[derive(Clone)]
pub struct MerkleProofVariable<C: CircuitConfig, HV: FieldHasherVariable<C>> {
    pub index: Vec<C::Bit>,
    pub path: Vec<HV::DigestVariable>,
}

impl<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>>
    SCVerifyingKeyVariable<C, SC>
{
    pub fn observe_into<Challenger>(&self, builder: &mut Builder<C>, challenger: &mut Challenger)
    where
        Challenger: CanObserveVariable<C, Felt<C::F>> + CanObserveVariable<C, SC::DigestVariable>,
    {
        let tag: Felt<C::F> = builder.eval(C::F::from_canonical_u32(0x3156_4b47));
        let version: Felt<C::F> = builder.eval(C::F::one());
        challenger.observe(builder, tag);
        challenger.observe(builder, version);
        challenger.observe(builder, self.commitment);
        if self.has_global_owner {
            challenger.observe(builder, self.pc_start);
            let kind = self.program_global_seed[2][0];
            challenger.observe(builder, kind);
            for coordinate in &self.program_global_seed[..2] {
                let canonical: [Felt<C::F>; 11] =
                    core::array::from_fn(|idx| builder.eval(coordinate[idx] * kind));
                challenger.observe_slice(builder, canonical);
            }
        }
    }

    /// Hash the verifying key into a single digest.
    /// Hash all SC key identity, including metadata deliberately omitted from native transcripts.
    pub fn hash(&self, builder: &mut Builder<C>) -> SC::DigestVariable
    where
        C::F: TwoAdicField,
        SC::DigestVariable: IntoIterator<Item = Felt<C::F>>,
    {
        let mut num_inputs = 2 + DIGEST_SIZE + 1 + 23 + 32 + 32 + (3 * self.chip_information.len());
        for (name, _) in self.chip_information.iter() {
            num_inputs += name.len();
        }
        let mut inputs = Vec::with_capacity(num_inputs);
        inputs.push(builder.eval(C::F::from_canonical_u32(0x3156_4b47)));
        inputs.push(builder.eval(C::F::one()));
        inputs.extend(self.commitment);
        inputs.push(self.pc_start);
        let kind = self.program_global_seed[2][0];
        inputs.push(kind);
        for coordinate in &self.program_global_seed[..2] {
            for value in coordinate {
                inputs.push(builder.eval(*value * kind));
            }
        }
        inputs.extend(self.program_global_digest);
        for value in dt_stark::global_d11::global146_identity_fields::<C::F>() {
            inputs.push(builder.eval(value));
        }
        for (name, dimension) in self.chip_information.iter() {
            inputs.push(builder.eval(C::F::from_canonical_usize(dimension.width)));
            inputs.push(builder.eval(C::F::from_canonical_usize(dimension.height)));
            inputs.push(builder.eval(C::F::from_canonical_usize(name.len())));
            for byte in name.as_bytes() {
                inputs.push(builder.eval(C::F::from_canonical_u8(*byte)));
            }
        }

        SC::hash(builder, &inputs)
    }
}
