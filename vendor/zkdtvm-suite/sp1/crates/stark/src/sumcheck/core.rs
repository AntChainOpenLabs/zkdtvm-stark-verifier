//! Core sumcheck protocol implementation.
//!
//! This module implements the sumcheck proving logic, including:
//! - Linear rounds with algebraic decomposition optimization
//! - Nonlinear rounds with skip optimization
//! - Padding optimization via compressed traces (`CompressedMatrix`)
//!
//! The eq polynomial uses **small-endian ordering**: the next variable to be
//! fixed occupies the lowest bit position. Consequently, elements for `eq(0;z)`
//! and `eq(1;z)` are **interleaved** (even/odd indices) rather than split into
//! contiguous halves.

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::{
    air::MachineAir,
    config::{Challenge, Val},
    sumcheck::{
        config::SCStarkGenericConfig,
        folder::{SumcheckConstraintFolder, SumcheckConstraintFolderExt},
        state::{ChipState, SumcheckState},
        types::UniPolyEvals,
        utils::{barycentric_weights, deinterleave, linear_combination_slices},
    },
};

/// Sumcheck protocol with padding optimization.
///
/// - Uses `SumcheckState` with per-chip `ChipState` for state management.
/// - Supports compressed traces (`CompressedMatrix`) for padding optimization: evaluations are
///   computed on non-padding rows only, with padding contribution added directly.
/// - Linear rounds (degree=1) use algebraic decomposition (`USE_ALGEBRAIC_DECOMP`).
/// - Nonlinear rounds use skip optimization.
pub struct SumcheckProtocol<
    'a,
    SC: SCStarkGenericConfig,
    A: MachineAir<Val<SC>>,
    AE: MachineAir<Challenge<SC>>,
    const USE_ALGEBRAIC_DECOMP: bool,
> {
    /// Current state of sumcheck
    pub state: SumcheckState<'a, SC, A, AE>,
    /// Univariate polynomials for each round (evaluation form)
    pub unipolys: Vec<UniPolyEvals<Challenge<SC>>>,
}

/// Result of computing univariate polynomial for a single round.
pub struct UnipolyResult<EF> {
    /// The combined univariate polynomial for this sumcheck round (eval form)
    pub poly: UniPolyEvals<EF>,
    /// Per-chip main constraint polynomials (eval form)
    pub unipolys_main: Vec<UniPolyEvals<EF>>,
    /// Per-chip permutation constraint polynomials (eval form)
    pub unipolys_perm: Vec<UniPolyEvals<EF>>,
    /// Auxiliary vectors for algebraic decomposition (if used)
    pub aux_vectors: Option<Vec<Vec<Vec<EF>>>>,
}

/// Result of computing univariate polynomial for a single chip in a round
pub struct UnipolyChipResult<EF> {
    /// The per-chip main constraint polynomial (eval form)
    pub unipoly_main: UniPolyEvals<EF>,
    /// The per-chip permutation constraint polynomial (eval form)
    pub unipoly_perm: UniPolyEvals<EF>,
    /// Auxiliary vectors for algebraic decomposition (if used)
    pub aux_vectors: Option<Vec<Vec<EF>>>,
}

impl<
        'a,
        SC: SCStarkGenericConfig,
        A: MachineAir<Val<SC>>,
        AE: MachineAir<Challenge<SC>>,
        const USE_ALGEBRAIC_DECOMP: bool,
    > SumcheckProtocol<'a, SC, A, AE, USE_ALGEBRAIC_DECOMP>
where
    SC::Val: Field,
    Challenge<SC>: ExtensionField<SC::Val>,
    A: for<'b> Air<SumcheckConstraintFolder<'b, SC>>,
    AE: for<'b> Air<SumcheckConstraintFolderExt<'b, SC>>,
{
    /// Create a new sumcheck protocol instance.
    ///
    /// # Arguments
    ///
    /// * `eq_challenges` - Random challenges for eq polynomial
    /// * `chip_states` - Initial chip states with compressed traces
    /// * `num_rounds` - Total number of rounds
    /// * `num_rounds_linear` - Number of linear rounds
    /// * `num_skip_rounds` - Number of skip rounds
    /// * `log_height_threshold` - Log height threshold
    /// * `permutation_challenges` - Permutation challenges
    /// * `public_values` - Public values
    /// * `public_values_ext` - Public values (extension field)
    /// * `num_chips_each_round` - Number of chips in each round
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eq_challenges: Vec<Challenge<SC>>,
        bit_expand_poly_points: Vec<Val<SC>>,
        chip_states: Vec<ChipState<'a, SC, A, AE>>,
        num_rounds: usize,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
        log_height_threshold: usize,
        permutation_challenges: [Challenge<SC>; 2],
        public_values: &'a [Val<SC>],
        public_values_ext: Vec<Challenge<SC>>,
        num_chips_each_round: Vec<usize>,
    ) -> Self {
        let state = SumcheckState::new(
            chip_states,
            eq_challenges,
            bit_expand_poly_points,
            num_rounds,
            num_rounds_linear,
            num_skip_rounds,
            log_height_threshold,
            permutation_challenges,
            public_values,
            public_values_ext,
            num_chips_each_round,
        );

        Self { state, unipolys: Vec::with_capacity(num_rounds) }
    }

    /// Execute the sumcheck proof protocol.
    ///
    /// Runs all linear rounds first (with algebraic decomposition), then all
    /// nonlinear rounds (with skip optimization).
    pub fn prove(&mut self, challenger: &mut <SC as SCStarkGenericConfig>::MlChallenger) {
        // Phase 1: Linear rounds (degree=1) with algebraic decomposition.
        for _ in 0..self.state.num_rounds_linear {
            self.process_linear_round(challenger);
        }

        // Phase 2: Nonlinear rounds with skip optimization.
        for _ in self.state.num_rounds_linear..self.state.num_rounds {
            self.process_nonlinear_round(challenger);
        }
    }

    /// Process a single linear round (degree=1) with algebraic decomposition.
    pub fn process_linear_round(
        &mut self,
        challenger: &mut <SC as SCStarkGenericConfig>::MlChallenger,
    ) {
        let univariate_result = self.compute_univariate_linear_round();

        univariate_result.poly.evals.iter().for_each(|eval| {
            <<SC as SCStarkGenericConfig>::MlChallenger as CanObserve<Val<SC>>>::observe_slice(
                challenger,
                eval.as_base_slice(),
            );
        });

        let round_challenge = FieldChallenger::sample_ext_element::<Challenge<SC>>(challenger);

        self.update_state_after_linear_round(univariate_result, round_challenge);
    }

    /// Process a single nonlinear round with skip optimization.
    pub fn process_nonlinear_round(
        &mut self,
        challenger: &mut <SC as SCStarkGenericConfig>::MlChallenger,
    ) {
        let univariate_result = self.compute_univariate_nonlinear_round();

        univariate_result.poly.evals.iter().for_each(|eval| {
            <<SC as SCStarkGenericConfig>::MlChallenger as CanObserve<Val<SC>>>::observe_slice(
                challenger,
                eval.as_base_slice(),
            );
        });

        let round_challenge = FieldChallenger::sample_ext_element::<Challenge<SC>>(challenger);

        self.update_state_after_nonlinear_round(univariate_result, round_challenge);
    }

    /// Compute the univariate polynomial for a linear round.
    ///
    /// Handles both existing (non-first-round) and newly introduced (first-round) chips.
    /// Returns per-chip main/permutation unipolys and optional auxiliary vectors for
    /// algebraic decomposition.
    pub fn compute_univariate_linear_round(&self) -> UnipolyResult<Challenge<SC>> {
        // Get current round's chips
        let prev_round_chips = self.state.prev_round_chips();
        let new_chips = self.state.new_chips();

        // Compute unipolys for existing chips (non-first round)
        let results_existing: Vec<UnipolyChipResult<Challenge<SC>>> = prev_round_chips
            .par_iter()
            .map(|chip| {
                chip.compute_unipoly_linear_non_first_round(
                    &self.state.permutation_challenges,
                    &self.state.public_values_ext,
                    &self.state.eq_poly,
                    USE_ALGEBRAIC_DECOMP,
                )
            })
            .collect();

        // Compute unipolys for new chips (first round)
        let results_new: Vec<UnipolyChipResult<Challenge<SC>>> = new_chips
            .par_iter()
            .map(|chip| {
                chip.compute_unipoly_linear_first_round(
                    &self.state.permutation_challenges,
                    self.state.public_values,
                    &self.state.eq_poly,
                    USE_ALGEBRAIC_DECOMP,
                )
            })
            .collect();

        // Destructure results by moving (not cloning) unipolys out.
        let total = results_existing.len() + results_new.len();
        let mut all_unipolys = Vec::with_capacity(total);
        let mut all_unipolys_perm = Vec::with_capacity(total);
        let mut aux_parts = Vec::with_capacity(total);
        let mut has_any_aux = false;

        for r in results_existing.into_iter().chain(results_new.into_iter()) {
            all_unipolys.push(r.unipoly_main);
            all_unipolys_perm.push(r.unipoly_perm);
            has_any_aux |= r.aux_vectors.is_some();
            aux_parts.push(r.aux_vectors);
        }

        let all_aux_vectors: Option<Vec<Vec<Vec<Challenge<SC>>>>> = if has_any_aux {
            Some(aux_parts.into_iter().map(std::option::Option::unwrap_or_default).collect())
        } else {
            None
        };

        let mut unipoly_sum = UniPolyEvals::sum_refs(all_unipolys.iter());
        crate::sumcheck::utils::multiply_evals_by_eq(&mut unipoly_sum, &self.state.eq_poly);

        let mut unipoly_perm_sum = UniPolyEvals::sum_refs(all_unipolys_perm.iter());
        if unipoly_perm_sum.evals.is_empty() {
            // All chips have empty permutation; produce a zero polynomial of the right length.
            unipoly_perm_sum.evals.resize(unipoly_sum.evals.len(), Challenge::<SC>::zero());
        } else {
            unipoly_perm_sum.extend_to(unipoly_sum.evals.len());
        }

        let unipoly = unipoly_sum + unipoly_perm_sum;

        debug_assert_eq!(
            unipoly.eval_one_plus_eval_zero(),
            self.state.claim,
            "linear round {}: f(0)+f(1) != claim",
            self.state.round_index
        );

        UnipolyResult {
            poly: unipoly,
            unipolys_main: all_unipolys,
            unipolys_perm: all_unipolys_perm,
            aux_vectors: all_aux_vectors,
        }
    }

    /// Compute the univariate polynomial for a nonlinear round.
    ///
    /// Uses skip optimization: the variable degree is `2^k - 1` where `k` is the
    /// number of skip rounds.
    pub fn compute_univariate_nonlinear_round(&self) -> UnipolyResult<Challenge<SC>> {
        let num_skip_rounds = self.state.num_skip_rounds;
        let var_degree = (1 << num_skip_rounds) - 1; // 2^k - 1

        // Get current round's chips
        let prev_round_chips = self.state.prev_round_chips();
        let new_chips = self.state.new_chips();

        // Compute unipolys for existing chips (non-first round)
        let results_existing: Vec<UnipolyChipResult<Challenge<SC>>> = prev_round_chips
            .par_iter()
            .map(|chip| {
                let degree = num_skip_rounds * chip.chip_degree * var_degree;
                chip.compute_unipoly_nonlinear_non_first_round(
                    var_degree,
                    degree,
                    &self.state.permutation_challenges,
                    &self.state.public_values_ext,
                    &self.state.eq_poly,
                    &self.state.bit_expand_poly,
                )
            })
            .collect();

        // Compute unipolys for new chips (first round)
        let results_new: Vec<UnipolyChipResult<Challenge<SC>>> = new_chips
            .par_iter()
            .map(|chip| {
                let degree = num_skip_rounds * chip.chip_degree * var_degree;
                chip.compute_unipoly_nonlinear_first_round(
                    var_degree,
                    degree,
                    &self.state.permutation_challenges,
                    self.state.public_values,
                    &self.state.eq_poly,
                    &self.state.bit_expand_poly,
                )
            })
            .collect();

        // Destructure results by moving (not cloning) unipolys out.
        let total = results_existing.len() + results_new.len();
        let mut all_unipolys = Vec::with_capacity(total);
        let mut all_unipolys_perm = Vec::with_capacity(total);
        for r in results_existing.into_iter().chain(results_new.into_iter()) {
            all_unipolys.push(r.unipoly_main);
            all_unipolys_perm.push(r.unipoly_perm);
        }

        let mut unipoly_sum = UniPolyEvals::sum_refs(all_unipolys.iter());
        crate::sumcheck::utils::multiply_evals_by_eq(&mut unipoly_sum, &self.state.eq_poly);

        let mut unipoly_perm_sum = UniPolyEvals::sum_refs(all_unipolys_perm.iter());
        if unipoly_perm_sum.evals.is_empty() {
            unipoly_perm_sum.evals.resize(unipoly_sum.evals.len(), Challenge::<SC>::zero());
        } else {
            unipoly_perm_sum.extend_to(unipoly_sum.evals.len());
        }

        let unipoly = unipoly_sum + unipoly_perm_sum;

        if cfg!(debug_assertions) {
            let sum = unipoly.sum_over_range(1 << self.state.num_skip_rounds);
            debug_assert_eq!(
                sum, self.state.claim,
                "nonlinear round {}: f(0)+f(1)+...+f(2^k-1) != claim",
                self.state.round_index
            );
        }

        UnipolyResult {
            poly: unipoly,
            unipolys_main: all_unipolys,
            unipolys_perm: all_unipolys_perm,
            aux_vectors: None,
        }
    }

    /// Update state after a linear round.
    ///
    /// 1. Record the round challenge.
    /// 2. Update the global claim from the combined univariate polynomial.
    /// 3. For each chip: evaluate per-chip claims, fold traces, update selectors.
    /// 4. Update auxiliary vectors for algebraic decomposition (if applicable).
    /// 5. Push the combined univariate polynomial.
    /// 6. Update the eq polynomial.
    /// 7. Advance the round index.
    pub fn update_state_after_linear_round(
        &mut self,
        unipoly_result: UnipolyResult<Challenge<SC>>,
        round_challenge: Challenge<SC>,
    ) {
        // 1. Record the round challenge.
        if self.state.round_index == 0 {
            self.state.sumcheck_challenges.reserve(self.state.num_rounds);
        }
        self.state.sumcheck_challenges.push(round_challenge);

        // 2. Update the global claim.
        self.state.claim = unipoly_result.poly.eval_at_point(round_challenge);

        let one_minus_challenge = Challenge::<SC>::one() - round_challenge;
        let num_prev = self.state.num_chips_prev_round();

        // Iterate over all current-round chips together with their per-chip unipolys.
        // The first `num_prev` entries are prev-round chips; the rest are new chips.
        self.state
            .current_round_chips_mut()
            .par_iter_mut()
            .zip(unipoly_result.unipolys_main)
            .zip(unipoly_result.unipolys_perm)
            .for_each(|((chip_state, poly_main), poly_perm)| {
                chip_state.claim = poly_main.eval_at_point(round_challenge);
                chip_state.perm_claim = poly_perm.eval_at_point(round_challenge);

                chip_state.update_traces(std::slice::from_ref(&round_challenge));

                chip_state.is_first_row_value *= one_minus_challenge;
                chip_state.is_last_row_value *= round_challenge;
            });

        // 5. Update auxiliary vectors for algebraic decomposition (if applicable). Must happen
        //    before eq_poly.update() because we need the current eq_challenge.
        if USE_ALGEBRAIC_DECOMP && self.state.round_index + 1 < self.state.num_rounds_linear {
            if let Some(all_chip_evals) = unipoly_result.aux_vectors {
                let eq_challenge =
                    self.state.eq_poly.eq_challenges[self.state.eq_poly.eq_challenges.len() -
                        self.state.eq_poly.num_vars_fixed -
                        2];

                // Split evals into prev-round chips and new chips.
                let (evals_prev, evals_new) = all_chip_evals.split_at(num_prev);

                // Update aux vectors for prev-round chips (non-first-round).
                if num_prev > 0 {
                    let prev_chips = &mut self.state.chip_states[..num_prev];
                    // Collect existing aux_vectors into a temporary Vec for the batch function.
                    let mut aux_batch: Vec<Vec<Vec<Challenge<SC>>>> = prev_chips
                        .iter_mut()
                        .map(|cs| cs.aux_vectors.take().unwrap_or_default())
                        .collect();
                    Self::update_aux_vectors_non_first_round(
                        &mut aux_batch,
                        &round_challenge,
                        &eq_challenge,
                        evals_prev,
                    );
                    // Distribute back to each ChipState.
                    for (cs, aux) in self.state.chip_states[..num_prev].iter_mut().zip(aux_batch) {
                        cs.aux_vectors = Some(aux);
                    }
                }

                // Generate aux vectors for new chips (first-round).
                let num_new = self.state.num_chips_new();
                if num_new > 0 {
                    let mut aux_batch: Vec<Vec<Vec<Challenge<SC>>>> = Vec::new();
                    Self::update_aux_vectors_first_round(
                        &mut aux_batch,
                        num_new,
                        &round_challenge,
                        &eq_challenge,
                        evals_new,
                    );
                    // Distribute to each new ChipState.
                    for (cs, aux) in self.state.chip_states[num_prev..num_prev + num_new]
                        .iter_mut()
                        .zip(aux_batch)
                    {
                        cs.aux_vectors = Some(aux);
                    }
                }
            }
        }

        // 6. Push the combined univariate polynomial.
        self.unipolys.push(unipoly_result.poly);

        // 7. Update the eq polynomial.
        self.state.eq_poly.update(round_challenge);

        // 8. Advance the round index.
        self.state.round_index += 1;
    }

    /// Generate auxiliary vectors for first-round chips.
    ///
    /// Extracts `part0`/`part1` via **deinterleaving** (even/odd indices) to match
    /// the small-endian eq polynomial layout.
    ///
    /// Output: `slot = vec![part0, part1]`, where:
    /// - `part0[i]` — eq(0;z) positions (even indices in the input)
    /// - `part1[i]` — eq(1;z) positions (odd indices in the input)
    fn update_aux_vectors_first_round(
        aux_for_eval_at_zero: &mut Vec<Vec<Vec<Challenge<SC>>>>,
        num_chips_new: usize,
        round_challenge: &Challenge<SC>,
        eq_challenge: &Challenge<SC>,
        chips_evals_except_zero_one_first_round: &[Vec<Vec<Challenge<SC>>>],
    ) {
        debug_assert_eq!(chips_evals_except_zero_one_first_round.len(), num_chips_new);

        let old_len = aux_for_eval_at_zero.len();
        let new_len = old_len + num_chips_new;
        aux_for_eval_at_zero.resize_with(new_len, Vec::new);

        let eq_0_z = Challenge::<SC>::one() - *eq_challenge;
        let inv_eq_0_z = eq_0_z.inverse();
        let inv_eq_1_z = eq_challenge.inverse();

        let r_factor = *round_challenge * (*round_challenge - Val::<SC>::one());

        aux_for_eval_at_zero[old_len..new_len].par_iter_mut().enumerate().for_each(
            |(chip_idx, slot)| {
                let evals = &chips_evals_except_zero_one_first_round[chip_idx];

                if evals.is_empty() {
                    return;
                }

                let num_eval_points = evals.len();
                debug_assert_eq!(evals[0].len() % 2, 0, "Each b vector length must be even");

                let xs: Vec<Challenge<SC>> = (0..num_eval_points)
                    .map(|x| Challenge::<SC>::from_canonical_usize(x + 2))
                    .collect();
                let bary_w = barycentric_weights(&xs, round_challenge);

                let mut base_factors = Vec::with_capacity(num_eval_points);
                for (idx, a_k) in bary_w.iter().enumerate() {
                    let k = idx + 2;
                    let denom_inv = Val::<SC>::from_canonical_usize(k * (k - 1)).inverse();
                    base_factors.push(*a_k * denom_inv * r_factor);
                }

                let weights0: Vec<Challenge<SC>> =
                    base_factors.iter().map(|f| *f * inv_eq_0_z).collect();
                let weights1: Vec<Challenge<SC>> =
                    base_factors.iter().map(|f| *f * inv_eq_1_z).collect();

                // Deinterleave: even indices → part0 (b₀=0), odd indices → part1 (b₀=1)
                #[allow(clippy::type_complexity)]
                let deinterleaved: Vec<(Vec<Challenge<SC>>, Vec<Challenge<SC>>)> =
                    evals.par_iter().map(|row| deinterleave(row)).collect();
                let evals_part0: Vec<&[Challenge<SC>]> =
                    deinterleaved.iter().map(|(e, _)| e.as_slice()).collect();
                let evals_part1: Vec<&[Challenge<SC>]> =
                    deinterleaved.iter().map(|(_, o)| o.as_slice()).collect();

                let part0 = linear_combination_slices(&weights0, &evals_part0);
                let part1 = linear_combination_slices(&weights1, &evals_part1);

                *slot = vec![part0, part1];
            },
        );
    }

    /// Update auxiliary vectors for non-first-round chips.
    ///
    /// Extracts `part0`/`part1` via **deinterleaving** (even/odd indices) to match
    /// the small-endian eq polynomial layout.
    ///
    /// - **Input**: `slot = vec![part0, part1]` from previous round, each length `L`. New evals
    ///   (points `2..d`) also have length `L`.
    /// - **Output**: `vec![new_part0, new_part1]`, each length `L/2`.
    fn update_aux_vectors_non_first_round(
        aux_for_eval_at_zero: &mut [Vec<Vec<Challenge<SC>>>],
        round_challenge: &Challenge<SC>,
        eq_challenge: &Challenge<SC>,
        chips_evals_except_zero_one_non_first_round: &[Vec<Vec<Challenge<SC>>>],
    ) {
        let num_chips_prev_round = aux_for_eval_at_zero.len();
        debug_assert_eq!(num_chips_prev_round, chips_evals_except_zero_one_non_first_round.len());

        let eq_0_z = Challenge::<SC>::one() - *eq_challenge;
        let inv_eq_0_z = eq_0_z.inverse();
        let inv_eq_1_z = eq_challenge.inverse();

        aux_for_eval_at_zero.par_iter_mut().enumerate().for_each(|(chip_idx, slot)| {
            let num_eval_points_no_01 = chips_evals_except_zero_one_non_first_round[chip_idx].len();

            if num_eval_points_no_01 == 0 {
                return;
            }

            // Build interpolation points xs = [0, 1, 2, ..., d]
            let xs: Vec<Challenge<SC>> = (0..(num_eval_points_no_01 + 2))
                .map(Challenge::<SC>::from_canonical_usize)
                .collect();

            // Collect all evaluation vectors: slot[0], slot[1], new_evals[0], ...
            let mut evals_all: Vec<&[Challenge<SC>]> = Vec::with_capacity(xs.len());
            evals_all.push(slot[0].as_slice());
            evals_all.push(slot[1].as_slice());
            evals_all.extend(
                chips_evals_except_zero_one_non_first_round[chip_idx].iter().map(Vec::as_slice),
            );

            debug_assert_eq!(evals_all.len(), xs.len());
            debug_assert_eq!(evals_all[0].len() % 2, 0, "b-vector length must be even");

            let bary_w = barycentric_weights(&xs, round_challenge);

            // Deinterleave all vectors: even indices → part0, odd indices → part1
            #[allow(clippy::type_complexity)]
            let deinterleaved: Vec<(Vec<Challenge<SC>>, Vec<Challenge<SC>>)> =
                evals_all.par_iter().map(|row| deinterleave(row)).collect();
            let evals_part0: Vec<&[Challenge<SC>]> =
                deinterleaved.iter().map(|(e, _)| e.as_slice()).collect();
            let evals_part1: Vec<&[Challenge<SC>]> =
                deinterleaved.iter().map(|(_, o)| o.as_slice()).collect();

            let weights0: Vec<Challenge<SC>> = bary_w.iter().map(|a| *a * inv_eq_0_z).collect();
            let weights1: Vec<Challenge<SC>> = bary_w.iter().map(|a| *a * inv_eq_1_z).collect();

            let new_part0 = linear_combination_slices(&weights0, &evals_part0);
            let new_part1 = linear_combination_slices(&weights1, &evals_part1);

            *slot = vec![new_part0, new_part1];
        });
    }

    /// Update state after a nonlinear round with skip folding.
    pub fn update_state_after_nonlinear_round(
        &mut self,
        unipoly_result: UnipolyResult<Challenge<SC>>,
        round_challenge: Challenge<SC>,
    ) {
        let num_rounds = self.state.num_rounds;

        // update sumcheck challenges
        if self.state.round_index == 0 {
            self.state.sumcheck_challenges.reserve(num_rounds);
        }
        self.state.sumcheck_challenges.push(round_challenge);

        // update global claim
        self.state.claim = unipoly_result.poly.eval_at_point(round_challenge);

        let extended_challenge = self.state.bit_expand_poly.evals_all(round_challenge);
        let extended_challenge_prod = extended_challenge
            .iter()
            .fold(Challenge::<SC>::one(), |acc, &challenge| acc * challenge);
        let one_minus_extended_challenge_prod =
            extended_challenge.iter().fold(Challenge::<SC>::one(), |acc, &challenge| {
                acc * (Challenge::<SC>::one() - challenge)
            });

        // Update each chip: claims, traces, and selectors.
        self.state
            .current_round_chips_mut()
            .par_iter_mut()
            .zip(unipoly_result.unipolys_main)
            .zip(unipoly_result.unipolys_perm)
            .for_each(|((chip_state, poly), poly_perm)| {
                chip_state.claim = poly.eval_at_point(round_challenge);
                chip_state.perm_claim = poly_perm.eval_at_point(round_challenge);
                chip_state.update_traces(&extended_challenge);
                chip_state.is_first_row_value *= one_minus_extended_challenge_prod;
                chip_state.is_last_row_value *= extended_challenge_prod;
            });

        self.unipolys.push(unipoly_result.poly);
        self.state.eq_poly.update(round_challenge);
        self.state.round_index += 1;
    }
}
