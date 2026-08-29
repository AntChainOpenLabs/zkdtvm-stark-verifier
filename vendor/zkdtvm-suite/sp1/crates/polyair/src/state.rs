use crate::{
    evaluator::{first_round_evaluation, nofirst_round_evaluation, ConstraintFolder},
    Chip,
};
use dt_stark::{
    air::{FullAir, MachineAir, PolyAirExtendable},
    sumcheck::{
        config::SCStarkGenericConfig,
        core::UnipolyChipResult,
        proof::SCChipOpenedValues,
        trace::ChipTrace,
        types::{BitExpandPoly, EqPoly, UniPolyEvals},
    },
    Challenge, SCAirOpenedValues, Val,
};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_matrix::{
    compressed::{
        padding_row_sum, padding_row_to_base_vec, padding_row_to_challenge_vec, CompressedMatrix,
        PaddingRow,
    },
    Matrix,
};
use p3_maybe_rayon::prelude::*;

pub struct ChipState<'a, SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize>
where
    Val<SC>: PolyAirExtendable<D>,
{
    /// Chip index within the global chip list.
    pub idx: usize,
    /// Log2 of the trace height.
    pub log_height: usize,
    /// Base-field chip reference (used for constraint evaluation in the first round).
    pub chip: &'a Chip<A, Val<SC>, D>,
    /// reserved poly (main or preprocessed).
    pub reserved_poly: ChipTrace<'a, SC>,
    /// precompute linear combination.
    pub precompute_lc: ChipTrace<'a, SC>,
    /// Permutation trace (compressed, tri-state representation).
    pub permutation: ChipTrace<'a, SC>,
    /// Current value of `is_first_row` selector (for boundary constraints).
    pub is_first_row_value: Challenge<SC>,
    /// Current value of `is_last_row` selector.
    pub is_last_row_value: Challenge<SC>,
    /// Constraint claim value for this chip in the current round.
    pub claim: Challenge<SC>,
    /// Permutation claim value for this chip in the current round.
    pub perm_claim: Challenge<SC>,
    /// Local cumulative sum for this chip.
    pub local_cumulative_sum: Challenge<SC>,
    /// Precomputed powers of alpha for constraint randomization.
    pub powers_of_alpha: Vec<Challenge<SC>>,
    /// Number of constraints for this chip.
    pub num_constraints: usize,
    /// Auxiliary vectors for algebraic decomposition optimization.
    pub aux_vectors: Option<Vec<Vec<Challenge<SC>>>>,
    /// Precomputed padding-row evaluation for main constraints (excludes the
    /// `local_cumulative_sum` permutation constraint) with the last-row selector
    /// set to zero.
    pub padding_eval_main_nonlast: Challenge<SC>,
    /// Precomputed padding-row evaluation for main constraints with the
    /// last-row selector set to one. Padding rows share values, but only the
    /// final padding row carries the folded last-row selector.
    pub padding_eval_main_last: Challenge<SC>,
    /// Precomputed padding-row evaluation for the permutation constraint only
    /// (the `local_cumulative_sum` term). Multiplied by the number of padding
    /// rows each round.
    pub padding_eval_perm: Challenge<SC>,
}

impl<'a, SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize> ChipState<'a, SC, A, D>
where
    Val<SC>: PolyAirExtendable<D>,
{
    pub fn scaled_local_cumulative_sum(&self) -> Challenge<SC> {
        // TODO: is log_height < 31 a reasonable assumption?
        assert!(self.log_height < 31, "log_height too large");
        let height_inv = Val::<SC>::from_canonical_usize(1 << self.log_height).inverse();
        self.local_cumulative_sum * height_inv
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        idx: usize,
        log_height: usize,
        chip: &'a Chip<A, Val<SC>, D>,
        reserved_poly: &'a CompressedMatrix<Val<SC>, Val<SC>>,
        precompute_lc: &'a CompressedMatrix<Challenge<SC>, Challenge<SC>>,
        permutation: &'a CompressedMatrix<Challenge<SC>, Challenge<SC>>,
        local_cumulative_sum: Challenge<SC>,
        powers_of_alpha: Vec<Challenge<SC>>,
        num_constraints: usize,
        // first permutation challenge
        _perm_alpha: Challenge<SC>,
        // second permutation challenge
        _beta_powers: &[Challenge<SC>],
        _beta_septix: Challenge<SC>,
        _public_values: &'a [Val<SC>],
    ) -> Self
    where
        A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>>,
    {
        // Compute padding_eval_perm: sum of all elements in the permutation padding row,
        // minus local_cumulative_sum / 2^log_height. When there is no permutation
        // (padding_row is None), use zero.
        let padding_eval_perm = if let PaddingRow::None = &permutation.padding_row {
            Challenge::<SC>::zero()
        } else {
            let row_sum = padding_row_sum(&permutation.padding_row);
            assert!(log_height < 31, "log_height too large");
            let height_inv = Val::<SC>::from_canonical_usize(1 << log_height).inverse();
            (row_sum - local_cumulative_sum * height_inv) * *powers_of_alpha.last().unwrap()
        };

        let eval_padding_main = |is_last_row: Val<SC>| {
            let reserved_padding_vec = padding_row_to_base_vec(&reserved_poly.padding_row);
            let precompute_padding_vec = padding_row_to_challenge_vec(&precompute_lc.padding_row);
            let perm_padding_vec = padding_row_to_challenge_vec(&permutation.padding_row);

            let reserved_view =
                p3_matrix::dense::RowMajorMatrixView::new_row(&reserved_padding_vec);
            let precompute_view =
                p3_matrix::dense::RowMajorMatrixView::new_row(&precompute_padding_vec);
            let perm_view = p3_matrix::dense::RowMajorMatrixView::new_row(&perm_padding_vec);

            let perm_empty = permutation.main.width() == 0;
            let constraint_reducer = if perm_empty {
                powers_of_alpha.clone()
            } else {
                powers_of_alpha[..powers_of_alpha.len() - 1].to_vec()
            };

            let mut accumulator = Challenge::<SC>::zero();
            let mut folder = ConstraintFolder {
                public: _public_values,
                alpha: _perm_alpha,
                beta_powers: _beta_powers,
                beta_septix: _beta_septix,
                precomputed: precompute_view,
                reserved_poly: reserved_view,
                is_first_row: Val::<SC>::zero(),
                is_last_row,
                local_sum: local_cumulative_sum,
                permutation: perm_view,
                multiplicitys: vec![],
                batch_size: chip.logup_batch_size(),
                accumulator: &mut accumulator,
                constraint_reducer: &constraint_reducer,
                constraint_index: 0,
            };

            chip.air.eval(&mut folder);
            chip.air.lookup(&mut folder);
            folder.constrain_lookup();
            accumulator
        };

        let (padding_eval_main_nonlast, padding_eval_main_last) =
            if matches!(precompute_lc.padding_row, PaddingRow::None) &&
                matches!(reserved_poly.padding_row, PaddingRow::None) &&
                matches!(permutation.padding_row, PaddingRow::None)
            {
                (Challenge::<SC>::zero(), Challenge::<SC>::zero())
            } else {
                (eval_padding_main(Val::<SC>::zero()), eval_padding_main(Val::<SC>::one()))
            };
        Self {
            idx,
            log_height,
            chip,
            reserved_poly: ChipTrace::FirstRound(reserved_poly),
            precompute_lc: ChipTrace::FirstRoundExt(precompute_lc),
            permutation: ChipTrace::FirstRoundExt(permutation),
            is_first_row_value: Challenge::<SC>::one(),
            is_last_row_value: Challenge::<SC>::one(),
            claim: Challenge::<SC>::zero(),
            perm_claim: Challenge::<SC>::zero(),
            local_cumulative_sum,
            powers_of_alpha,
            num_constraints,
            aux_vectors: None,
            padding_eval_main_nonlast,
            padding_eval_main_last,
            padding_eval_perm,
        }
    }

    /// Update traces: for each challenge, fold preprocessed, main, and permutation concurrently.
    /// Linear round: pass a single-element slice; nonlinear round: pass multiple challenges.
    pub fn update_traces(&mut self, challenges: &[Challenge<SC>]) {
        for &challenge in challenges {
            let reserved_poly = &mut self.reserved_poly;
            let precompute_lc = &mut self.precompute_lc;
            let perm = &mut self.permutation;
            join(
                || {
                    reserved_poly.update(challenge);
                    precompute_lc.update(challenge);
                },
                || perm.update(challenge),
            );
        }
    }

    /// Returns summation input matrices (compressed) for the first round.
    ///
    /// Uses `get_summation_input_base` for preprocessed and main (they must be `FirstRound`),
    /// and `get_summation_input_ext` for permutation (must be `FirstRoundExt`). Panics if
    /// the wrong `ChipTrace` variant is used.
    #[allow(clippy::type_complexity)]
    pub fn get_summation_input_first_round(
        &self,
        point: usize,
        var_degree: usize,
        bit_expand_poly: Option<&BitExpandPoly<Val<SC>>>,
    ) -> (
        CompressedMatrix<Val<SC>, Val<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
    ) {
        let reserved_poly_input =
            self.reserved_poly.get_summation_input_base(point, var_degree, bit_expand_poly);
        let precompute_lc =
            self.precompute_lc.get_summation_input_ext(point, var_degree, bit_expand_poly);
        let permutation_input =
            self.permutation.get_summation_input_ext(point, var_degree, bit_expand_poly);
        (reserved_poly_input, precompute_lc, permutation_input)
    }

    /// Returns summation input matrices (compressed) for non-first round.
    ///
    /// Uses `get_summation_input_hybrid` for preprocessed and main (they must be `NonFirstRound`),
    /// and `get_summation_input_ext` for permutation (must be `NonFirstRoundExt`). Panics if
    /// the wrong `ChipTrace` variant is used.
    #[allow(clippy::type_complexity)]
    pub fn get_summation_input_non_first_round(
        &self,
        point: usize,
        var_degree: usize,
        bit_expand_poly: Option<&BitExpandPoly<Val<SC>>>,
    ) -> (
        CompressedMatrix<Val<SC>, Challenge<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
    ) {
        let reserved_poly_input =
            self.reserved_poly.get_summation_input_hybrid(point, var_degree, bit_expand_poly);
        let precompute_lc =
            self.precompute_lc.get_summation_input_ext(point, var_degree, bit_expand_poly);
        let permutation_input =
            self.permutation.get_summation_input_ext(point, var_degree, bit_expand_poly);
        (reserved_poly_input, precompute_lc, permutation_input)
    }

    fn padding_main_eval(&self, last_value: Challenge<SC>) -> Challenge<SC> {
        self.padding_eval_main_nonlast +
            (self.padding_eval_main_last - self.padding_eval_main_nonlast) * last_value
    }

    fn padding_main_is_zero(&self) -> bool {
        self.padding_eval_main_nonlast == Challenge::<SC>::zero() &&
            self.padding_eval_main_last == Challenge::<SC>::zero()
    }

    fn padding_main_contribution(
        &self,
        stored_height: usize,
        total_height: usize,
        coeffs: Option<&[Challenge<SC>]>,
        last_value: Challenge<SC>,
    ) -> Challenge<SC> {
        let num_padding = total_height.saturating_sub(stored_height);
        if num_padding == 0 || self.padding_main_is_zero() {
            return Challenge::<SC>::zero();
        }

        let nonlast = self.padding_eval_main_nonlast;
        let last_delta =
            (self.padding_eval_main_last - self.padding_eval_main_nonlast) * last_value;

        if let Some(coeffs) = coeffs {
            let padding_coeff_sum = (stored_height..total_height)
                .map(|row| coeffs.get(row).copied().unwrap_or_else(Challenge::<SC>::zero))
                .sum::<Challenge<SC>>();
            let last_coeff =
                coeffs.get(total_height - 1).copied().unwrap_or_else(Challenge::<SC>::zero);
            nonlast * padding_coeff_sum + last_delta * last_coeff
        } else {
            nonlast * Val::<SC>::from_canonical_usize(num_padding) + last_delta
        }
    }

    fn push_padding_main_evals(
        &self,
        out: &mut Vec<Challenge<SC>>,
        stored_height: usize,
        total_height: usize,
        coeffs: Option<&[Challenge<SC>]>,
        last_value: Challenge<SC>,
    ) {
        let num_padding = total_height.saturating_sub(stored_height);
        if num_padding == 0 {
            return;
        }
        if self.padding_main_is_zero() {
            out.extend(vec![Challenge::<SC>::zero(); num_padding]);
            return;
        }

        let nonlast = self.padding_eval_main_nonlast;
        let last_eval = self.padding_main_eval(last_value);
        for row in stored_height..total_height {
            let mut value = if row + 1 == total_height { last_eval } else { nonlast };
            if let Some(coeffs) = coeffs {
                value *= coeffs.get(row).copied().unwrap_or_else(Challenge::<SC>::zero);
            }
            out.push(value);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_eval_linear_non_first_round(
        &self,
        point: usize,
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> (Challenge<SC>, Option<Vec<Challenge<SC>>>)
    where
        A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    {
        let (reserved_poly_input, precompute_lc_input, permutation_input) =
            self.get_summation_input_non_first_round(point, 1, None);

        let one_minus_point = Val::<SC>::one() - Val::<SC>::from_canonical_usize(point);
        let first_value = self.is_first_row_value * one_minus_point;
        let point_val = Val::<SC>::from_canonical_usize(point);
        let last_value = self.is_last_row_value * point_val;

        let mut block_row_evals = nofirst_round_evaluation(
            &self.chip.air,
            public_values,
            &reserved_poly_input,
            &precompute_lc_input,
            &permutation_input,
            perm_alpha,
            beta_powers,
            beta_septix,
            self.local_cumulative_sum,
            self.chip.logup_batch_size(),
            &self.powers_of_alpha,
            first_value,
            last_value,
        );

        if !eq_poly.coeffs.is_empty() {
            let coeffs = eq_poly.coeffs.last().expect("checked non-empty eq_poly coeffs");
            for (i, value) in block_row_evals.iter_mut().enumerate() {
                let coeff = coeffs.get(i).copied().unwrap();
                *value *= coeff;
            }
        }

        let stored_height = reserved_poly_input.stored_height();
        let total_height = reserved_poly_input.total_height;
        let coeffs = eq_poly.coeffs.last().map(Vec::as_slice);
        let padding_contribution =
            self.padding_main_contribution(stored_height, total_height, coeffs, last_value);

        let total_sum =
            block_row_evals.par_iter().copied().sum::<Challenge<SC>>() + padding_contribution;

        if use_algebraic_decomp {
            self.push_padding_main_evals(
                &mut block_row_evals,
                stored_height,
                total_height,
                coeffs,
                last_value,
            );
            (total_sum, Some(block_row_evals))
        } else {
            (total_sum, None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_eval_linear_first_round(
        &self,
        point: usize,
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> (Challenge<SC>, Option<Vec<Challenge<SC>>>)
    where
        A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>>,
    {
        let (reserved_poly_input, precompute_lc_input, permutation_input) =
            self.get_summation_input_first_round(point, 1, None);

        let first_value = match point {
            0 => Val::<SC>::one(),
            1 => Val::<SC>::zero(),
            _ => -Val::<SC>::from_canonical_usize(point - 1),
        };
        let last_value = match point {
            0 => Val::<SC>::zero(),
            1 => Val::<SC>::one(),
            _ => Val::<SC>::from_canonical_usize(point),
        };

        let mut block_row_evals = first_round_evaluation(
            &self.chip.air,
            public_values,
            &reserved_poly_input,
            &precompute_lc_input,
            &permutation_input,
            perm_alpha,
            beta_powers,
            beta_septix,
            self.local_cumulative_sum,
            self.chip.logup_batch_size(),
            &self.powers_of_alpha,
            first_value,
            last_value,
        );

        if !eq_poly.coeffs.is_empty() {
            let coeffs = eq_poly.coeffs.last().expect("checked non-empty eq_poly coeffs");
            for (i, value) in block_row_evals.iter_mut().enumerate() {
                let coeff = coeffs.get(i).copied().unwrap_or_else(Challenge::<SC>::zero);
                *value *= coeff;
            }
        }

        let stored_height = reserved_poly_input.stored_height();
        let total_height = reserved_poly_input.total_height;
        let coeffs = eq_poly.coeffs.last().map(Vec::as_slice);
        let padding_contribution = self.padding_main_contribution(
            stored_height,
            total_height,
            coeffs,
            Challenge::<SC>::from_base(last_value),
        );

        let total_sum =
            block_row_evals.par_iter().copied().sum::<Challenge<SC>>() + padding_contribution;

        if use_algebraic_decomp {
            self.push_padding_main_evals(
                &mut block_row_evals,
                stored_height,
                total_height,
                coeffs,
                Challenge::<SC>::from_base(last_value),
            );
            (total_sum, Some(block_row_evals))
        } else {
            (total_sum, None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_unipoly_linear_non_first_round(
        &self,
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        eq_challenge_inv: Option<Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> UnipolyChipResult<Challenge<SC>>
    where
        A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    {
        let perm_empty = self.permutation.total_height() == 0;
        let mut chip_row_evaluations = None;
        let mut evals = Vec::with_capacity(self.chip.degree + 1);
        let mut evals_perm = Vec::with_capacity(2);
        let mut points = Vec::with_capacity(self.chip.degree + 1);

        let (eval_at_zero, _) = if let Some(aux_vec) = self.aux_vectors.as_ref() {
            (aux_vec[0].par_iter().copied().sum(), None)
        } else {
            self.compute_eval_linear_non_first_round(
                0,
                perm_alpha,
                beta_powers,
                beta_septix,
                public_values,
                eq_poly,
                use_algebraic_decomp,
            )
        };

        let eq_challenge =
            eq_poly.eq_challenges[eq_poly.eq_challenges.len() - eq_poly.num_vars_fixed - 1];
        let eval_at_one = if eq_challenge == Challenge::<SC>::zero() {
            Challenge::<SC>::zero()
        } else {
            let temp = self.claim - (Challenge::<SC>::one() - eq_challenge) * eval_at_zero;
            eq_challenge_inv.expect("non-zero eq challenge must have an inverse") * temp
        };

        evals.push(eval_at_zero);
        evals.push(eval_at_one);
        points.push(Challenge::<SC>::zero());
        points.push(Challenge::<SC>::one());

        if !perm_empty {
            let mut eval_perm_at_zero = self.permutation.get_sum_perm_rows_linear(0);
            eval_perm_at_zero -= self.local_cumulative_sum *
                Challenge::<SC>::from_canonical_usize(self.permutation.total_height()) /
                Challenge::<SC>::from_canonical_usize(2 << self.log_height);

            let eval_perm_at_one = if let Some(last_power) = self.powers_of_alpha.last() {
                self.perm_claim / *last_power - eval_perm_at_zero
            } else {
                Challenge::<SC>::zero()
            };

            evals_perm.push(eval_perm_at_zero);
            evals_perm.push(eval_perm_at_one);
        }

        if self.chip.degree > 1 {
            let extra_points: Vec<usize> = (2..=self.chip.degree).collect();
            let computed: Vec<_> = extra_points
                .par_iter()
                .map(|&point| {
                    let (eval, aux_vec) = self.compute_eval_linear_non_first_round(
                        point,
                        perm_alpha,
                        beta_powers,
                        beta_septix,
                        public_values,
                        eq_poly,
                        use_algebraic_decomp,
                    );
                    (point, eval, aux_vec)
                })
                .collect();

            for (point, eval, aux_vec) in computed {
                evals.push(eval);
                points.push(Challenge::<SC>::from_canonical_usize(point));
                if let Some(v) = aux_vec {
                    if point == 2 {
                        chip_row_evaluations = Some(vec![v]);
                    } else if let Some(ref mut chip_evals) = chip_row_evaluations {
                        chip_evals.push(v);
                    }
                }
            }
        }

        if !perm_empty {
            if let Some(last_power) = self.powers_of_alpha.last() {
                for eval_perm in &mut evals_perm {
                    *eval_perm *= *last_power;
                }
            }
        }

        UnipolyChipResult {
            unipoly_main: UniPolyEvals::new(evals),
            unipoly_perm: UniPolyEvals::new(evals_perm),
            aux_vectors: chip_row_evaluations,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_unipoly_linear_first_round(
        &self,
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> UnipolyChipResult<Challenge<SC>>
    where
        A: for<'b> FullAir<ConstraintFolder<'b, Val<SC>, Val<SC>, Challenge<SC>>>,
    {
        let perm_empty = self.permutation.total_height() == 0;
        let mut chip_row_evaluations = None;
        let mut evals = Vec::with_capacity(self.chip.degree + 1);
        let mut evals_perm = Vec::with_capacity(2);
        let mut points = Vec::with_capacity(self.chip.degree + 1);

        evals.push(Challenge::<SC>::zero());
        evals.push(Challenge::<SC>::zero());
        points.push(Challenge::<SC>::zero());
        points.push(Challenge::<SC>::one());

        if !perm_empty {
            let mut eval_perm_at_zero: Challenge<SC> = self.permutation.get_sum_perm_rows_linear(0);
            eval_perm_at_zero -=
                self.local_cumulative_sum * Val::<SC>::from_canonical_usize(2).inverse();

            let mut eval_perm_at_one: Challenge<SC> = self.permutation.get_sum_perm_rows_linear(1);
            eval_perm_at_one -=
                self.local_cumulative_sum * Val::<SC>::from_canonical_usize(2).inverse();

            evals_perm.push(eval_perm_at_zero);
            evals_perm.push(eval_perm_at_one);
        }

        if self.chip.degree > 1 {
            let extra_points: Vec<usize> = (2..=self.chip.degree).collect();
            let computed: Vec<_> = extra_points
                .par_iter()
                .map(|&point| {
                    let (eval, aux_vec) = self.compute_eval_linear_first_round(
                        point,
                        perm_alpha,
                        beta_powers,
                        beta_septix,
                        public_values,
                        eq_poly,
                        use_algebraic_decomp,
                    );
                    (point, eval, aux_vec)
                })
                .collect();

            for (point, eval, aux_vec) in computed {
                evals.push(eval);
                points.push(Challenge::<SC>::from_canonical_usize(point));
                if let Some(v) = aux_vec {
                    if point == 2 {
                        chip_row_evaluations = Some(vec![v]);
                    } else if let Some(ref mut chip_evals) = chip_row_evaluations {
                        chip_evals.push(v);
                    }
                }
            }
        }

        if !perm_empty {
            if let Some(last_power) = self.powers_of_alpha.last() {
                for eval_perm in &mut evals_perm {
                    *eval_perm *= *last_power;
                }
            }
        }

        UnipolyChipResult {
            unipoly_main: UniPolyEvals::new(evals),
            unipoly_perm: UniPolyEvals::new(evals_perm),
            aux_vectors: chip_row_evaluations,
        }
    }
}

pub struct SumcheckState<'a, SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize>
where
    Val<SC>: PolyAirExtendable<D>,
{
    /// Current round index (0-based).
    pub round_index: usize,
    /// Aggregated claim value across all chips.
    pub claim: Challenge<SC>,
    /// Per-chip states, ordered by chip index.
    pub chip_states: Vec<ChipState<'a, SC, A, D>>,
    /// Equality polynomial used in the sumcheck protocol.
    pub eq_poly: EqPoly<Val<SC>, Challenge<SC>>,
    /// Random challenges chosen at each sumcheck round.
    pub sumcheck_challenges: Vec<Challenge<SC>>,
    /// Total number of sumcheck rounds.
    pub num_rounds: usize,
    /// Number of linear rounds.
    pub num_rounds_linear: usize,
    /// Number of skip (nonlinear) rounds.
    pub num_skip_rounds: usize,
    /// Log2 height threshold for chip introduction.
    pub log_height_threshold: usize,
    /// Permutation challenges alpha.
    pub perm_alpha: Challenge<SC>,
    /// Permutation challenges beta.
    pub beta_powers: &'a [Challenge<SC>],
    /// Beta raised to the 7th power in the septic extension.
    pub beta_septix: Challenge<SC>,
    /// Public values (base field).
    pub public_values: &'a [Val<SC>],
    /// Cumulative number of chips participating up to each round.
    pub num_chips_each_round: Vec<usize>,
}

impl<'a, SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize>
    SumcheckState<'a, SC, A, D>
where
    Val<SC>: PolyAirExtendable<D>,
{
    /// Create a new `SumcheckState`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chip_states: Vec<ChipState<'a, SC, A, D>>,
        eq_challenges: Vec<Challenge<SC>>,
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
        // Initial claim is zero; the first round derives it from chip unipolys times eq.
        let claim = Challenge::<SC>::zero();

        // Create the eq polynomial (nonlinear-round variable degree = 2^num_skip_rounds - 1).
        let degree = (1 << num_skip_rounds) - 1;
        let eq_poly = EqPoly::new(eq_challenges, num_rounds_linear, degree);

        Self {
            round_index: 0,
            claim,
            chip_states,
            eq_poly,
            sumcheck_challenges: vec![],
            num_rounds,
            num_rounds_linear,
            num_skip_rounds,
            log_height_threshold,
            perm_alpha,
            beta_powers,
            beta_septix,
            public_values,
            num_chips_each_round,
        }
    }

    /// Returns the chips participating in the current round.
    #[allow(dead_code)]
    pub fn current_round_chips(&self) -> &[ChipState<'a, SC, A, D>] {
        let num_chips = self.num_chips_current_round();
        &self.chip_states[..num_chips]
    }

    /// Returns the chips participating in the current round (mutable).
    pub fn current_round_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, D>] {
        let num_chips = self.num_chips_current_round();
        &mut self.chip_states[..num_chips]
    }

    /// Returns the chips newly introduced in the current round.
    pub fn new_chips(&self) -> &[ChipState<'a, SC, A, D>] {
        let prev_num_chips = self.num_chips_prev_round();
        let curr_num_chips = self.num_chips_current_round();
        &self.chip_states[prev_num_chips..curr_num_chips]
    }

    /// Returns the chips newly introduced in the current round (mutable).
    #[allow(dead_code)]
    pub fn new_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, D>] {
        let prev_num_chips = self.num_chips_prev_round();
        let curr_num_chips = self.num_chips_current_round();
        &mut self.chip_states[prev_num_chips..curr_num_chips]
    }

    /// Returns the chips that existed in the previous round.
    pub fn prev_round_chips(&self) -> &[ChipState<'a, SC, A, D>] {
        let prev_num_chips = self.num_chips_prev_round();
        &self.chip_states[..prev_num_chips]
    }

    /// Returns the chips that existed in the previous round (mutable).
    #[allow(dead_code)]
    pub fn prev_round_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, D>] {
        let prev_num_chips = self.num_chips_prev_round();
        &mut self.chip_states[..prev_num_chips]
    }

    /// Returns the number of chips in the previous round.
    pub fn num_chips_prev_round(&self) -> usize {
        if self.round_index == 0 {
            0
        } else {
            self.num_chips_each_round[self.round_index - 1]
        }
    }

    /// Returns the number of newly introduced chips in the current round.
    pub fn num_chips_new(&self) -> usize {
        if self.round_index == 0 {
            self.num_chips_each_round[0]
        } else {
            let prev_num_chips = self.num_chips_each_round[self.round_index - 1];
            let curr_num_chips = self.num_chips_each_round[self.round_index];
            curr_num_chips - prev_num_chips
        }
    }

    /// Returns the total number of chips participating in the current round.
    pub fn num_chips_current_round(&self) -> usize {
        self.num_chips_each_round[self.round_index]
    }
}

pub fn compute_eq_poly_coeffs<EF: Field>(eq_challenges: &[EF]) -> Vec<Vec<EF>> {
    let mut ret = Vec::with_capacity(eq_challenges.len());
    eq_challenges.iter().enumerate().for_each(|(i, &r)| {
        let cur = vec![EF::one() - r, r];
        if i == 0 {
            ret.push(cur);
        } else {
            let prev = ret.last().unwrap();
            // prev ⊗ cur: prev in high bits (outer), cur in low bits (inner)
            ret.push(
                prev.par_iter()
                    .flat_map(|&x| cur.iter().map(|&y| x * y).collect::<Vec<EF>>())
                    .collect(),
            );
        }
    });
    ret
}

pub fn batch_eval<F: Field, EF: ExtensionField<F>>(
    mat: &CompressedMatrix<F, F>,
    eq_poly_coeff: &[EF],
) -> Vec<EF> {
    assert_eq!(mat.total_height, eq_poly_coeff.len());
    let storage_height = mat.stored_height();
    let padding_height = mat.total_height - storage_height;
    let eval0 = mat.main.columnwise_dot_product(&eq_poly_coeff[..storage_height]);
    let temp: EF = eq_poly_coeff[storage_height..].into_par_iter().copied().sum();
    let eval: Vec<EF> = if padding_height != 0 {
        let padding_row = padding_row_to_base_vec(&mat.padding_row);
        padding_row
            .into_par_iter()
            .zip(eval0.into_par_iter())
            .map(|(i, eval)| temp * i + eval)
            .collect()
    } else {
        eval0
    };
    eval
}

pub fn finalize<SC: SCStarkGenericConfig>(
    log_heights: &[usize],
    main: &Vec<(String, CompressedMatrix<Val<SC>, Val<SC>>)>,
    prep: &Vec<Option<&CompressedMatrix<Val<SC>, Val<SC>>>>,
    perm: &Vec<CompressedMatrix<Challenge<SC>, Challenge<SC>>>,
    eq_poly_coeffs: &Vec<Vec<Challenge<SC>>>,
    local_sums: &Vec<Challenge<SC>>,
) -> Vec<SCChipOpenedValues<Val<SC>, Challenge<SC>>> {
    let num_chips = log_heights.len();
    assert_eq!(num_chips, main.len());
    assert_eq!(num_chips, prep.len());
    assert_eq!(num_chips, perm.len());
    let height_zero_eq_poly_coeff = [Challenge::<SC>::one()];
    (0..num_chips)
        .into_par_iter()
        .map(|chip_index| {
            let log_height = log_heights[chip_index];
            let eq_poly_coeff: &[Challenge<SC>] = if log_height == 0 {
                height_zero_eq_poly_coeff.as_slice()
            } else {
                eq_poly_coeffs[log_height - 1].as_slice()
            };
            assert_eq!(eq_poly_coeff.len(), 1 << log_height);
            let main_eval = batch_eval(&main[chip_index].1, eq_poly_coeff);
            let prep_eval = prep[chip_index]
                .as_ref()
                .map(|mat| batch_eval(mat, eq_poly_coeff))
                .unwrap_or(vec![]);
            let perm_eval = batch_eval(&perm[chip_index], eq_poly_coeff);

            let preprocessed = SCAirOpenedValues { local: prep_eval };
            let main = SCAirOpenedValues { local: main_eval };
            let permutation = SCAirOpenedValues { local: perm_eval };
            SCChipOpenedValues {
                preprocessed,
                main,
                permutation,
                local_cumulative_sum: local_sums[chip_index],
                log_height,
                _field: core::marker::PhantomData,
            }
        })
        .collect()
}
