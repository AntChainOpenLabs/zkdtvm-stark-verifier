use dt_stark::{
    air::{MachineAir, PolyAirExtendable},
    sumcheck::{config::SCStarkGenericConfig, core::UnipolyResult, types::UniPolyEvals},
    Challenge, Val,
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::state::SumcheckState;
use dt_stark::sumcheck::{core::UnipolyChipResult, utils::barycentric_weights};

pub struct SumcheckProtocol<
    'a,
    SC: SCStarkGenericConfig,
    A: MachineAir<Val<SC>>,
    const D: usize,
    const USE_ALGEBRAIC_DECOMP: bool,
> where
    Val<SC>: PolyAirExtendable<D>,
{
    /// Current state of sumcheck
    pub state: SumcheckState<'a, SC, A, D>,
    /// Univariate polynomials for each round (evaluation form)
    pub unipolys: Vec<UniPolyEvals<Challenge<SC>>>,
}

impl<
        'a,
        SC: SCStarkGenericConfig,
        A: MachineAir<Val<SC>>,
        const D: usize,
        const USE_ALGEBRAIC_DECOMP: bool,
    > SumcheckProtocol<'a, SC, A, D, USE_ALGEBRAIC_DECOMP>
where
    SC::Val: Field,
    Challenge<SC>: ExtensionField<SC::Val>,
    Val<SC>: PolyAirExtendable<D>,
    A: for<'b> dt_stark::air::FullAir<
            crate::evaluator::ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>,
        > + for<'b> dt_stark::air::FullAir<
            crate::evaluator::ConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>,
        >,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eq_challenges: Vec<Challenge<SC>>,
        chip_states: Vec<crate::state::ChipState<'a, SC, A, D>>,
        num_rounds: usize,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
        log_height_threshold: usize,
        perm_alpha: Challenge<SC>,
        beta_powers: &'a [Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &'a [Val<SC>],
        num_chips_each_round: Vec<usize>,
    ) -> Self {
        let state = SumcheckState::new(
            chip_states,
            eq_challenges,
            num_rounds,
            num_rounds_linear,
            num_skip_rounds,
            log_height_threshold,
            perm_alpha,
            beta_powers,
            beta_septix,
            public_values,
            num_chips_each_round,
        );
        Self { state, unipolys: Vec::with_capacity(num_rounds) }
    }

    pub fn prove(&mut self, challenger: &mut <SC as SCStarkGenericConfig>::MlChallenger) {
        for _ in 0..self.state.num_rounds_linear {
            self.process_linear_round(challenger);
        }
    }

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

    pub fn compute_univariate_linear_round(&self) -> UnipolyResult<Challenge<SC>> {
        let prev_round_chips = self.state.prev_round_chips();
        let new_chips = self.state.new_chips();
        let eq_challenge = self.state.eq_poly.eq_challenges
            [self.state.eq_poly.eq_challenges.len() - self.state.eq_poly.num_vars_fixed - 1];
        let eq_challenge_inv = if eq_challenge == Challenge::<SC>::zero() {
            None
        } else {
            Some(eq_challenge.inverse())
        };

        let results_existing: Vec<UnipolyChipResult<Challenge<SC>>> = prev_round_chips
            .par_iter()
            .map(|chip| {
                chip.compute_unipoly_linear_non_first_round(
                    self.state.perm_alpha,
                    self.state.beta_powers,
                    self.state.beta_septix,
                    self.state.public_values,
                    &self.state.eq_poly,
                    eq_challenge_inv,
                    USE_ALGEBRAIC_DECOMP,
                )
            })
            .collect();

        let results_new: Vec<UnipolyChipResult<Challenge<SC>>> = new_chips
            .par_iter()
            .map(|chip| {
                chip.compute_unipoly_linear_first_round(
                    self.state.perm_alpha,
                    self.state.beta_powers,
                    self.state.beta_septix,
                    self.state.public_values,
                    &self.state.eq_poly,
                    USE_ALGEBRAIC_DECOMP,
                )
            })
            .collect();

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

        let all_aux_vectors = if has_any_aux {
            Some(aux_parts.into_iter().map(Option::unwrap_or_default).collect())
        } else {
            None
        };

        let mut unipoly_sum = UniPolyEvals::sum_refs(all_unipolys.iter());
        Self::multiply_evals_by_eq_linear(&mut unipoly_sum, &self.state.eq_poly);

        let mut unipoly_perm_sum = UniPolyEvals::sum_refs(all_unipolys_perm.iter());
        unipoly_perm_sum.extend_to(unipoly_sum.evals.len());

        let unipoly = unipoly_sum + unipoly_perm_sum;

        if cfg!(debug_assertions) {
            debug_assert_eq!(
                unipoly.eval_one_plus_eval_zero(),
                self.state.claim,
                "linear round {}: f(0)+f(1) != claim",
                self.state.round_index
            );
        }

        UnipolyResult {
            poly: unipoly,
            unipolys_main: all_unipolys,
            unipolys_perm: all_unipolys_perm,
            aux_vectors: all_aux_vectors,
        }
    }

    pub fn update_state_after_linear_round(
        &mut self,
        unipoly_result: UnipolyResult<Challenge<SC>>,
        round_challenge: Challenge<SC>,
    ) {
        if self.state.round_index == 0 {
            self.state.sumcheck_challenges.reserve(self.state.num_rounds);
        }
        self.state.sumcheck_challenges.push(round_challenge);
        self.state.claim = unipoly_result.poly.eval_at_point(round_challenge);

        let one_minus_challenge = Challenge::<SC>::one() - round_challenge;
        let num_prev = self.state.num_chips_prev_round();

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

        if USE_ALGEBRAIC_DECOMP && self.state.round_index + 1 < self.state.num_rounds_linear {
            if let Some(all_chip_evals) = unipoly_result.aux_vectors {
                let eq_challenge =
                    self.state.eq_poly.eq_challenges[self.state.eq_poly.eq_challenges.len() -
                        self.state.eq_poly.num_vars_fixed -
                        2];
                let (evals_prev, evals_new) = all_chip_evals.split_at(num_prev);

                if num_prev > 0 {
                    let prev_chips = &mut self.state.chip_states[..num_prev];
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
                    for (cs, aux) in self.state.chip_states[..num_prev].iter_mut().zip(aux_batch) {
                        cs.aux_vectors = Some(aux);
                    }
                }

                let num_new = self.state.num_chips_new();
                if num_new > 0 {
                    let mut aux_batch = Vec::new();
                    Self::update_aux_vectors_first_round(
                        &mut aux_batch,
                        num_new,
                        &round_challenge,
                        &eq_challenge,
                        evals_new,
                    );
                    for (cs, aux) in self.state.chip_states[num_prev..num_prev + num_new]
                        .iter_mut()
                        .zip(aux_batch)
                    {
                        cs.aux_vectors = Some(aux);
                    }
                }
            }
        }

        self.unipolys.push(unipoly_result.poly);
        self.state.eq_poly.update(round_challenge);
        self.state.round_index += 1;
    }

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

                let eval_rows: Vec<&[Challenge<SC>]> = evals.iter().map(Vec::as_slice).collect();
                let (part0, part1) =
                    Self::linear_combination_deinterleaved(&weights0, &weights1, &eval_rows);
                *slot = vec![part0, part1];
            },
        );
    }

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

            let xs: Vec<Challenge<SC>> = (0..(num_eval_points_no_01 + 2))
                .map(Challenge::<SC>::from_canonical_usize)
                .collect();

            let mut evals_all: Vec<&[Challenge<SC>]> = Vec::with_capacity(xs.len());
            evals_all.push(slot[0].as_slice());
            evals_all.push(slot[1].as_slice());
            evals_all.extend(
                chips_evals_except_zero_one_non_first_round[chip_idx].iter().map(Vec::as_slice),
            );

            let bary_w = barycentric_weights(&xs, round_challenge);
            let weights0: Vec<Challenge<SC>> = bary_w.iter().map(|a| *a * inv_eq_0_z).collect();
            let weights1: Vec<Challenge<SC>> = bary_w.iter().map(|a| *a * inv_eq_1_z).collect();
            let (new_part0, new_part1) =
                Self::linear_combination_deinterleaved(&weights0, &weights1, &evals_all);
            *slot = vec![new_part0, new_part1];
        });
    }

    fn linear_combination_deinterleaved(
        weights0: &[Challenge<SC>],
        weights1: &[Challenge<SC>],
        rows: &[&[Challenge<SC>]],
    ) -> (Vec<Challenge<SC>>, Vec<Challenge<SC>>) {
        debug_assert_eq!(weights0.len(), rows.len());
        debug_assert_eq!(weights1.len(), rows.len());
        if rows.is_empty() {
            return (Vec::new(), Vec::new());
        }
        debug_assert!(
            rows.iter().all(|row| row.len() == rows[0].len()),
            "aux rows must have equal length"
        );

        let even_len = rows[0].len().div_ceil(2);
        let odd_len = rows[0].len() / 2;
        let evens = (0..even_len)
            .into_par_iter()
            .map(|i| {
                let row_idx = 2 * i;
                let mut acc = Challenge::<SC>::zero();
                for (weight, row) in weights0.iter().zip(rows.iter()) {
                    acc += *weight * row[row_idx];
                }
                acc
            })
            .collect();
        let odds = (0..odd_len)
            .into_par_iter()
            .map(|i| {
                let row_idx = 2 * i + 1;
                let mut acc = Challenge::<SC>::zero();
                for (weight, row) in weights1.iter().zip(rows.iter()) {
                    acc += *weight * row[row_idx];
                }
                acc
            })
            .collect();
        (evens, odds)
    }

    fn multiply_evals_by_eq_linear(
        poly: &mut UniPolyEvals<Challenge<SC>>,
        eq_poly: &dt_stark::sumcheck::types::EqPoly<Val<SC>, Challenge<SC>>,
    ) {
        let challenge =
            eq_poly.eq_challenges[eq_poly.eq_challenges.len() - eq_poly.num_vars_fixed - 1];
        debug_assert!(
            eq_poly.num_vars_fixed < eq_poly.num_linear_vars,
            "linear-only protocol called after linear rounds"
        );
        poly.extend_to(poly.evals.len() + 1);
        for (i, v) in poly.evals.iter_mut().enumerate() {
            let x = Challenge::<SC>::from_canonical_usize(i);
            let eq_val =
                (Challenge::<SC>::one() - challenge) * (Challenge::<SC>::one() - x) + challenge * x;
            *v *= eq_poly.eval * eq_val;
        }
    }
}
