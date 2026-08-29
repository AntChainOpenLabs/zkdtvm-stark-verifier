//! Sumcheck protocol state management.
//!
//! This module defines the two core state types used during the sumcheck protocol:
//! - [`ChipState`]: per-chip state holding compressed traces, selectors, and constraint
//!   evaluations.
//! - [`SumcheckState`]: global state aggregating all chips, the eq polynomial, and round
//!   bookkeeping.

use p3_air::Air;
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PackedValue};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{get_max_constraint_degree_sc, SymbolicAirBuilder};

use super::{
    proof::SCChipOpenedValues,
    trace::{
        padding_row_sum, padding_row_to_base_vec, padding_row_to_challenge_vec, ChipTrace,
        CompressedMatrix, PaddingRow,
    },
    types::{BitExpandPoly, EqPoly, UniPolyEvals},
};
use crate::{
    air::MachineAir,
    config::Val,
    sumcheck::{
        config::SCStarkGenericConfig,
        core::UnipolyChipResult,
        folder::{
            PaddingRowConstraintFolder, SumcheckConstraintFolder, SumcheckConstraintFolderExt,
        },
        utils::selectors::{
            get_first_sel_ext_packed, get_first_sel_packed, get_last_sel_ext_packed,
            get_last_sel_packed,
        },
    },
    Challenge, Chip, PackedChallenge, PackedExt, PackedVal, SCAirOpenedValues, PROOF_MAX_NUM_PVS,
};
/// Per-chip state during the sumcheck protocol.
///
/// Encapsulates all chip-specific fields extracted from the former `SumcheckCommon`
/// and `SumcheckState` structures.
pub struct ChipState<'a, SC: SCStarkGenericConfig, A, AE> {
    /// Chip index within the global chip list.
    pub idx: usize,
    /// Log2 of the trace height.
    pub log_height: usize,
    /// Maximum constraint degree of this chip.
    pub chip_degree: usize,
    /// Base-field chip reference (used for constraint evaluation in the first round).
    pub chip: &'a Chip<Val<SC>, A>,
    /// Extension-field chip reference (used for constraint evaluation in non-first rounds).
    pub chip_ext: &'a Chip<Challenge<SC>, AE>,
    /// Preprocessed trace (compressed, tri-state representation).
    pub preprocessed: Option<ChipTrace<'a, SC>>,
    /// Main trace (compressed, tri-state representation).
    pub main: ChipTrace<'a, SC>,
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
    /// The round in which this chip is introduced.
    pub round_introduced: usize,
    /// Precomputed padding-row evaluation for main constraints (excludes the
    /// `local_cumulative_sum` permutation constraint). Multiplied by the number
    /// of padding rows each round.
    pub padding_eval_main: Challenge<SC>,
    /// Precomputed padding-row evaluation for the permutation constraint only
    /// (the `local_cumulative_sum` term). Multiplied by the number of padding
    /// rows each round.
    pub padding_eval_perm: Challenge<SC>,
}

impl<SC: SCStarkGenericConfig, A, AE> std::fmt::Debug for ChipState<'_, SC, A, AE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChipState")
            .field("idx", &self.idx)
            .field("log_height", &self.log_height)
            .field("chip_degree", &self.chip_degree)
            .field("preprocessed", &"Option<ChipTrace>")
            .field("main", &"ChipTrace")
            .field("permutation", &"ChipTrace")
            .field("is_first_row_value", &self.is_first_row_value)
            .field("is_last_row_value", &self.is_last_row_value)
            .field("claim", &self.claim)
            .field("perm_claim", &self.perm_claim)
            .field("local_cumulative_sum", &self.local_cumulative_sum)
            .field("num_constraints", &self.num_constraints)
            .field("round_introduced", &self.round_introduced)
            .field("padding_eval_main", &self.padding_eval_main)
            .field("padding_eval_perm", &self.padding_eval_perm)
            .finish()
    }
}

impl<'a, SC: SCStarkGenericConfig, A, AE> ChipState<'a, SC, A, AE>
where
    for<'b> A: Air<SumcheckConstraintFolder<'b, SC>> + MachineAir<Val<SC>>,
    for<'b> AE: Air<SumcheckConstraintFolderExt<'b, SC>> + MachineAir<Challenge<SC>>,
{
    /// Returns `local_cumulative_sum / 2^log_height` (scaled by inverse of trace height).
    pub fn scaled_local_cumulative_sum(&self) -> Challenge<SC> {
        // TODO: is log_height < 31 a reasonable assumption?
        assert!(self.log_height < 31, "log_height too large");
        let height_inv = Val::<SC>::from_canonical_usize(1 << self.log_height).inverse();
        self.local_cumulative_sum * height_inv
    }

    /// * `powers_of_alpha` - Precomputed powers of alpha for constraint randomization
    /// * `num_constraints` - Number of constraints for this chip
    /// * `round_introduced` - The round in which this chip is introduced
    /// * `permutation_challenges` - Challenges for the permutation argument
    /// * `public_values` - Public values (base field)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        idx: usize,
        log_height: usize,
        chip: &'a Chip<Val<SC>, A>,
        chip_ext: &'a Chip<Challenge<SC>, AE>,
        preprocessed: Option<&'a CompressedMatrix<Val<SC>, Val<SC>>>,
        main: &'a CompressedMatrix<Val<SC>, Val<SC>>,
        permutation: &'a CompressedMatrix<Challenge<SC>, Challenge<SC>>,
        local_cumulative_sum: Challenge<SC>,
        powers_of_alpha: Vec<Challenge<SC>>,
        num_constraints: usize,
        round_introduced: usize,
        permutation_challenges: &[Challenge<SC>; 2],
        public_values: &'a [Val<SC>],
    ) -> Self
    where
        A: Air<SymbolicAirBuilder<Val<SC>>> + for<'b> Air<PaddingRowConstraintFolder<'b, SC>>,
    {
        let chip_degree = get_max_constraint_degree_sc(
            &chip.air,
            chip.air.preprocessed_width(),
            PROOF_MAX_NUM_PVS,
        )
        .max(3);

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

        // Compute padding_eval_main: evaluate constraints on the padding row using
        // PaddingRowConstraintFolder. When the main trace has PaddingRow::None, all rows
        // are actual data and there is nothing to evaluate, so the contribution is zero.
        let padding_eval_main = if matches!(main.padding_row, PaddingRow::None) {
            Challenge::<SC>::zero()
        } else {
            // Build base field padding row vectors for preprocessed and main.
            let preprocessed_padding_vec = preprocessed.map_or_else(
                || vec![Val::<SC>::zero(); chip.air.preprocessed_width()],
                |p| padding_row_to_base_vec(&p.padding_row),
            );
            let main_padding_vec = padding_row_to_base_vec(&main.padding_row);
            let perm_padding_vec = padding_row_to_challenge_vec(&permutation.padding_row);

            let preprocessed_view =
                RowMajorMatrixView::new(&preprocessed_padding_vec, preprocessed_padding_vec.len());
            let main_view = RowMajorMatrixView::new(&main_padding_vec, main_padding_vec.len());
            let perm_view = RowMajorMatrixView::new(&perm_padding_vec, perm_padding_vec.len());

            let perm_empty = permutation.main.width() == 0;

            let mut folder = PaddingRowConstraintFolder {
                preprocessed: preprocessed_view,
                main: main_view,
                permutation: perm_view,
                permutation_challenges: permutation_challenges.as_slice(),
                local_cumulative_sum: &local_cumulative_sum,
                is_first_row: Val::<SC>::zero(), // TODO: is this correct?
                is_last_row: Val::<SC>::one(),   // TODO: is this correct?
                powers_of_alpha: &powers_of_alpha
                    [..powers_of_alpha.len() - if perm_empty { 0 } else { 1 }],
                accumulator: Challenge::<SC>::zero(),
                public_values,
                constraint_index: 0,
            };

            chip.eval(&mut folder);
            folder.accumulator
        };

        Self {
            idx,
            log_height,
            chip_degree,
            chip,
            chip_ext,
            preprocessed: preprocessed.map(ChipTrace::FirstRound),
            main: ChipTrace::FirstRound(main),
            permutation: ChipTrace::FirstRoundExt(permutation),
            is_first_row_value: Challenge::<SC>::one(),
            is_last_row_value: Challenge::<SC>::one(),
            claim: Challenge::<SC>::zero(),
            perm_claim: Challenge::<SC>::zero(),
            local_cumulative_sum,
            powers_of_alpha,
            num_constraints,
            aux_vectors: None,
            round_introduced,
            padding_eval_main,
            padding_eval_perm,
        }
    }

    /// Update traces: for each challenge, fold preprocessed, main, and permutation concurrently.
    /// Linear round: pass a single-element slice; nonlinear round: pass multiple challenges.
    pub fn update_traces(&mut self, challenges: &[Challenge<SC>]) {
        for &challenge in challenges {
            let main = &mut self.main;
            let perm = &mut self.permutation;
            match &mut self.preprocessed {
                Some(prep) => {
                    join(
                        || {
                            prep.update(challenge);
                            main.update(challenge);
                        },
                        || perm.update(challenge),
                    );
                }
                None => {
                    join(|| main.update(challenge), || perm.update(challenge));
                }
            }
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
        Option<CompressedMatrix<Val<SC>, Val<SC>>>,
        CompressedMatrix<Val<SC>, Val<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
    ) {
        let preprocessed_input = self
            .preprocessed
            .as_ref()
            .map(|trace| trace.get_summation_input_base(point, var_degree, bit_expand_poly));
        let main_input = self.main.get_summation_input_base(point, var_degree, bit_expand_poly);
        let permutation_input =
            self.permutation.get_summation_input_ext(point, var_degree, bit_expand_poly);
        (preprocessed_input, main_input, permutation_input)
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
        Option<CompressedMatrix<Val<SC>, Challenge<SC>>>,
        CompressedMatrix<Val<SC>, Challenge<SC>>,
        CompressedMatrix<Challenge<SC>, Challenge<SC>>,
    ) {
        let preprocessed_input = self
            .preprocessed
            .as_ref()
            .map(|trace| trace.get_summation_input_hybrid(point, var_degree, bit_expand_poly));
        let main_input = self.main.get_summation_input_hybrid(point, var_degree, bit_expand_poly);
        let permutation_input =
            self.permutation.get_summation_input_ext(point, var_degree, bit_expand_poly);
        (preprocessed_input, main_input, permutation_input)
    }

    /// Compute single unipoly for nonlinear round (non-first round).
    ///
    /// # Arguments
    ///
    /// * `var_degree` - Degree of the variable in the polynomial (2^k - 1)
    /// * `degree` - Degree of the output univariate polynomial
    /// * `eq_poly` - The eq polynomial with current coefficients and weights
    ///
    /// # Returns
    ///
    /// `UnipolyChipResult` with main/perm unipolys and `aux_vectors: None`.
    pub fn compute_unipoly_nonlinear_non_first_round(
        &self,
        var_degree: usize,
        degree: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values_ext: &[Challenge<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        bit_expand_poly: &BitExpandPoly<Val<SC>>,
    ) -> UnipolyChipResult<Challenge<SC>> {
        let perm_empty = self.permutation.total_height() == 0;
        let degree_perm = degree / self.chip_degree;

        let eq_challenge =
            eq_poly.eq_challenges[eq_poly.eq_challenges.len() - eq_poly.num_vars_fixed - 1];
        let factors = EqPoly::<Val<SC>, Challenge<SC>>::compute_eq_poly_coeffs_single(
            eq_challenge,
            var_degree,
            &eq_poly.weights,
        );

        let independent_points: Vec<usize> = (0..=degree).filter(|&p| p != var_degree).collect();

        let computed: Vec<(usize, Challenge<SC>, Option<Challenge<SC>>)> = independent_points
            .par_iter()
            .map(|&point| {
                let (eval, eval_perm_opt) = self.compute_eval_nonlinear_non_first_round(
                    point,
                    var_degree,
                    degree_perm,
                    permutation_challenges,
                    public_values_ext,
                    eq_poly,
                    bit_expand_poly,
                );
                (point, eval, eval_perm_opt)
            })
            .collect();

        let mut evals = vec![Challenge::<SC>::zero(); degree + 1];
        let mut evals_perm = Vec::with_capacity(degree_perm + 1);
        evals_perm.resize(degree_perm + 1, Challenge::<SC>::zero());

        let mut acc = Challenge::<SC>::zero();
        let mut perm_acc = Challenge::<SC>::zero();

        for &(point, eval, eval_perm_opt) in &computed {
            evals[point] = eval;
            if point < var_degree {
                acc += eval * factors[point];
            }
            if !perm_empty {
                if let Some(ep) = eval_perm_opt {
                    evals_perm[point] = ep;
                    if point < var_degree {
                        perm_acc += ep;
                    }
                }
            }
        }

        let eval_at_var_degree = if factors[var_degree] != Challenge::<SC>::zero() {
            (self.claim - acc) * factors[var_degree].inverse()
        } else {
            Challenge::<SC>::zero()
        };
        evals[var_degree] = eval_at_var_degree;

        if !perm_empty {
            evals_perm[var_degree] = self.perm_claim - perm_acc;
        }

        UnipolyChipResult {
            unipoly_main: UniPolyEvals::new(evals),
            unipoly_perm: UniPolyEvals::new(evals_perm),
            aux_vectors: None,
        }
    }

    /// Compute single unipoly for nonlinear round (first round).
    ///
    /// # Arguments
    ///
    /// * `var_degree` - Degree of the variable in the polynomial (2^k - 1)
    /// * `degree` - Degree of the output univariate polynomial
    /// * `permutation_challenges` - Permutation challenges for constraint evaluation
    /// * `public_values` - Public values (base field version, for first round)
    /// * `eq_poly` - The eq polynomial with current coefficients and weights
    /// * `bit_expand_poly` - Bit expansion polynomial for expanding points in skip rounds
    ///
    /// # Returns
    ///
    /// `UnipolyChipResult` with main/perm unipolys and `aux_vectors: None`.
    pub fn compute_unipoly_nonlinear_first_round(
        &self,
        var_degree: usize,
        degree: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        bit_expand_poly: &BitExpandPoly<Val<SC>>,
    ) -> UnipolyChipResult<Challenge<SC>> {
        let perm_empty = self.permutation.total_height() == 0;
        let degree_perm = degree / self.chip_degree;

        let eq_challenge =
            eq_poly.eq_challenges[eq_poly.eq_challenges.len() - eq_poly.num_vars_fixed - 1];
        let factors = EqPoly::<Val<SC>, Challenge<SC>>::compute_eq_poly_coeffs_single(
            eq_challenge,
            var_degree,
            &eq_poly.weights,
        );

        let independent_points: Vec<usize> = (0..=degree).filter(|&p| p != var_degree).collect();

        let computed: Vec<(usize, Challenge<SC>, Option<Challenge<SC>>)> = independent_points
            .par_iter()
            .map(|&point| {
                let (eval, eval_perm_opt) = self.compute_eval_nonlinear_first_round(
                    point,
                    var_degree,
                    degree_perm,
                    permutation_challenges,
                    public_values,
                    eq_poly,
                    bit_expand_poly,
                );
                (point, eval, eval_perm_opt)
            })
            .collect();

        let mut evals = vec![Challenge::<SC>::zero(); degree + 1];
        let mut evals_perm = Vec::with_capacity(degree_perm + 1);
        evals_perm.resize(degree_perm + 1, Challenge::<SC>::zero());

        let mut acc = Challenge::<SC>::zero();
        let mut perm_acc = Challenge::<SC>::zero();

        for &(point, eval, eval_perm_opt) in &computed {
            evals[point] = eval;
            if point < var_degree {
                acc += eval * factors[point];
            }
            if !perm_empty {
                if let Some(ep) = eval_perm_opt {
                    evals_perm[point] = ep;
                    if point < var_degree {
                        perm_acc += ep;
                    }
                }
            }
        }

        let eval_at_var_degree = if factors[var_degree] != Challenge::<SC>::zero() {
            (self.claim - acc) * factors[var_degree].inverse()
        } else {
            Challenge::<SC>::zero()
        };
        evals[var_degree] = eval_at_var_degree;

        if !perm_empty {
            evals_perm[var_degree] = self.perm_claim - perm_acc;
        }

        UnipolyChipResult {
            unipoly_main: UniPolyEvals::new(evals),
            unipoly_perm: UniPolyEvals::new(evals_perm),
            aux_vectors: None,
        }
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    /// Compute evaluation for nonlinear round (non-first round).
    /// Returns `(eval, Option<eval_perm>)`; `eval_perm` is `None` when `point > degree_perm`
    /// since `unipoly_perm` has degree `degree_perm`.
    fn compute_eval_nonlinear_non_first_round(
        &self,
        point: usize,
        var_degree: usize,
        degree_perm: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values_ext: &[Challenge<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        bit_expand_poly: &BitExpandPoly<Val<SC>>,
    ) -> (Challenge<SC>, Option<Challenge<SC>>) {
        // get traces for summation according to `point`
        // case 1: point <= var_degree
        // eg. k=3, var_degree=2^k-1=7, point=3
        // get rows with indices 8*i+3 from the compressed preprocessed, main
        // and permutation trace; these rows forms the input traces.
        // case 2: point > var_degree
        // get expanded points using `BitExpandPoly::eval_all`, and then
        // fold the compressed preprocessed, main and permutation traces k
        // times in row to get input traces.
        let (preprocessed_input, main_input, permutation_input) =
            self.get_summation_input_non_first_round(point, var_degree, Some(bit_expand_poly));

        let empty_matrix: RowMajorMatrix<Challenge<SC>> = RowMajorMatrix::default(0, 0);
        let preprocessed = preprocessed_input.as_ref().map_or(&empty_matrix, |mat| &mat.main);
        let main = &main_input.main;
        let permutation = &permutation_input.main;

        let total_height = main_input.height();
        let stored_height = main_input.stored_height();
        let prep_width = preprocessed.width();
        let main_width = main.width();
        let perm_width = permutation.width();

        let points = bit_expand_poly.evals_all(Val::<SC>::from_canonical_usize(point));
        // first_value = prod_i (1-points[i]) * self.is_first_row_value
        let first_value =
            points.iter().fold(self.is_first_row_value, |acc, &p| acc * (Val::<SC>::one() - p));
        // last_value = prod_i points[i] * self.is_last_row_value
        let last_value = points.iter().fold(self.is_last_row_value, |acc, &p| acc * p);

        let packed_permutation_challenges =
            permutation_challenges.iter().map(|c| PackedExt::<SC>::from(*c)).collect::<Vec<_>>();
        let packed_local_cumulative_sum = PackedExt::<SC>::from_f(self.local_cumulative_sum);

        let ext_d = <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
        let eq_poly_coeffs_opt = if !eq_poly.coeffs.is_empty() {
            Some(eq_poly.coeffs.last().unwrap().as_slice())
        } else {
            None
        };

        let block_starts: Vec<usize> = (0..stored_height).step_by(PackedVal::<SC>::WIDTH).collect();
        let mut eval: Challenge<SC> = block_starts
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); prep_width],
                        vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); main_width],
                        vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); perm_width],
                        Challenge::<SC>::zero(),
                    )
                },
                |(mut prep_buf, mut main_buf, mut perm_buf, mut acc), i_start| {
                    let is_first_row = PackedExt::<SC>::new(get_first_sel_ext_packed(
                        i_start,
                        first_value,
                        total_height,
                        stored_height,
                    ));
                    let is_last_row = PackedExt::<SC>::new(get_last_sel_ext_packed(
                        i_start,
                        last_value,
                        total_height,
                        stored_height,
                    ));

                    for col in 0..prep_width {
                        prep_buf[col] =
                            PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    if i_start + offset < preprocessed.height() {
                                        preprocessed.get(i_start + offset, col).as_base_slice()[i]
                                    } else {
                                        Val::<SC>::zero()
                                    }
                                })
                            }));
                    }

                    for col in 0..main_width {
                        main_buf[col] =
                            PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    if i_start + offset < main.height() {
                                        main.get(i_start + offset, col).as_base_slice()[i]
                                    } else {
                                        Val::<SC>::zero()
                                    }
                                })
                            }));
                    }

                    for col in 0..perm_width {
                        perm_buf[col] =
                            PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    if i_start + offset < permutation.height() {
                                        permutation.get(i_start + offset, col).as_base_slice()[i]
                                    } else {
                                        Val::<SC>::zero()
                                    }
                                })
                            }));
                    }

                    let mut folder: SumcheckConstraintFolderExt<'_, SC> =
                        SumcheckConstraintFolderExt {
                            preprocessed: RowMajorMatrixView::new_row(&prep_buf),
                            main: RowMajorMatrixView::new_row(&main_buf),
                            permutation: RowMajorMatrixView::new_row(&perm_buf),
                            permutation_challenges: &packed_permutation_challenges,
                            is_first_row,
                            is_last_row,
                            powers_of_alpha: &self.powers_of_alpha,
                            accumulator: PackedExt::<SC>::zero(),
                            public_values: public_values_ext,
                            constraint_index: 0,
                            local_cumulative_sum: &packed_local_cumulative_sum,
                        };

                    self.chip_ext.eval(&mut folder);
                    let mut row_value_packed = folder.accumulator;

                    if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                        let eq = PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < eq_coeffs.len() {
                                    eq_coeffs[i_start + offset].as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        }));
                        row_value_packed *= eq;
                    }

                    let valid_count_in_pack = std::cmp::min(
                        stored_height.saturating_sub(i_start),
                        PackedVal::<SC>::WIDTH,
                    );
                    let base_slice = AbstractExtensionField::<PackedVal<SC>>::as_base_slice(
                        &row_value_packed.inner,
                    );
                    for idx_in_packing in 0..valid_count_in_pack {
                        let mut base_arr = vec![Val::<SC>::zero(); ext_d];
                        for coeff_idx in 0..ext_d {
                            base_arr[coeff_idx] = base_slice[coeff_idx].as_slice()[idx_in_packing];
                        }
                        acc += Challenge::<SC>::from_base_slice(&base_arr[..ext_d]);
                    }

                    (prep_buf, main_buf, perm_buf, acc)
                },
            )
            .map(|(_, _, _, acc)| acc)
            .sum();

        // Add padding rows' contribution, accounting for per-row eq coefficients.
        let num_padding = main_input.total_height - stored_height;
        let num_padding_rows = Val::<SC>::from_canonical_usize(num_padding);

        if self.padding_eval_main != Challenge::<SC>::zero() {
            if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                let padding_contribution: Challenge<SC> = (stored_height..main_input.total_height)
                    .map(|row| {
                        if row < eq_coeffs.len() {
                            self.padding_eval_main * eq_coeffs[row]
                        } else {
                            Challenge::<SC>::zero()
                        }
                    })
                    .sum();
                eval += padding_contribution;
            } else {
                eval += self.padding_eval_main * num_padding_rows;
            }
        }

        // eval for the last constraint of permutation (only needed when point <= degree_perm and
        // permutation is not empty)
        let eval_perm_opt = if point <= degree_perm && perm_width > 0 {
            let mut eval_perm = permutation.values.iter().copied().sum::<Challenge<SC>>();
            eval_perm -=
                self.scaled_local_cumulative_sum() * Val::<SC>::from_canonical_usize(stored_height);
            eval_perm *= *self.powers_of_alpha.last().unwrap();
            eval_perm += self.padding_eval_perm * num_padding_rows;
            Some(eval_perm)
        } else {
            None
        };

        (eval, eval_perm_opt)
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    /// Compute evaluation for nonlinear round (first round).
    ///
    /// This computes the main constraint evaluation for first round nonlinear variables
    /// using the base field trace values and `SumcheckConstraintFolder`.
    /// Returns `(eval, Option<eval_perm>)`; `eval_perm` is `None` when `point > degree_perm`.
    fn compute_eval_nonlinear_first_round(
        &self,
        point: usize,
        var_degree: usize,
        degree_perm: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        bit_expand_poly: &BitExpandPoly<Val<SC>>,
    ) -> (Challenge<SC>, Option<Challenge<SC>>) {
        let (preprocessed_input, main_input, permutation_input) =
            self.get_summation_input_first_round(point, var_degree, Some(bit_expand_poly));

        let empty_matrix = RowMajorMatrix::default(0, 0);
        let preprocessed = preprocessed_input.as_ref().map_or(&empty_matrix, |mat| &mat.main);
        let main = &main_input.main;
        let permutation = &permutation_input.main;

        let total_height = main_input.height();
        let stored_height = main_input.stored_height();
        let prep_width = preprocessed.width();
        let main_width = main.width();
        let perm_width = permutation.width();

        let points = bit_expand_poly.evals_all(Val::<SC>::from_canonical_usize(point));
        // first_value = prod_i (1-points[i]) * self.is_first_row_value
        let first_value =
            points.iter().fold(Val::<SC>::one(), |acc, &p| acc * (Val::<SC>::one() - p));
        // last_value = prod_i points[i] * self.is_last_row_value
        let last_value = points.iter().fold(Val::<SC>::one(), |acc, &p| acc * p);

        let packed_permutation_challenges = permutation_challenges
            .iter()
            .map(|c| PackedChallenge::<SC>::from_f(*c))
            .collect::<Vec<_>>();
        let packed_local_cumulative_sum = PackedChallenge::<SC>::from_f(self.local_cumulative_sum);

        let ext_d = <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
        let eq_poly_coeffs_opt = if !eq_poly.coeffs.is_empty() {
            Some(eq_poly.coeffs.last().unwrap().as_slice())
        } else {
            None
        };

        let block_starts: Vec<usize> = (0..stored_height).step_by(PackedVal::<SC>::WIDTH).collect();
        let mut eval: Challenge<SC> = block_starts
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![PackedVal::<SC>::zero(); prep_width],
                        vec![PackedVal::<SC>::zero(); main_width],
                        vec![PackedChallenge::<SC>::zero(); perm_width],
                        Challenge::<SC>::zero(),
                    )
                },
                |(mut prep_buf, mut main_buf, mut perm_buf, mut acc), i_start| {
                    let is_first_row =
                        get_first_sel_packed(i_start, first_value, total_height, stored_height);
                    let is_last_row =
                        get_last_sel_packed(i_start, last_value, total_height, stored_height);

                    for col in 0..prep_width {
                        prep_buf[col] = PackedVal::<SC>::from_fn(|offset| {
                            if i_start + offset < preprocessed.height() {
                                preprocessed.get(i_start + offset, col)
                            } else {
                                Val::<SC>::zero()
                            }
                        });
                    }

                    for col in 0..main_width {
                        main_buf[col] = PackedVal::<SC>::from_fn(|offset| {
                            if i_start + offset < main.height() {
                                main.get(i_start + offset, col)
                            } else {
                                Val::<SC>::zero()
                            }
                        });
                    }

                    for col in 0..perm_width {
                        perm_buf[col] = PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < permutation.height() {
                                    permutation.get(i_start + offset, col).as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        });
                    }

                    let mut folder = SumcheckConstraintFolder {
                        preprocessed: RowMajorMatrixView::new_row(&prep_buf),
                        main: RowMajorMatrixView::new_row(&main_buf),
                        permutation: RowMajorMatrixView::new_row(&perm_buf),
                        permutation_challenges: &packed_permutation_challenges,
                        local_cumulative_sum: &packed_local_cumulative_sum,
                        is_first_row,
                        is_last_row,
                        powers_of_alpha: &self.powers_of_alpha,
                        accumulator: PackedChallenge::<SC>::zero(),
                        public_values,
                        constraint_index: 0,
                    };

                    self.chip.eval(&mut folder);
                    let mut row_value_packed = folder.accumulator;

                    if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                        let eq = PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < eq_coeffs.len() {
                                    eq_coeffs[i_start + offset].as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        });
                        row_value_packed *= eq;
                    }

                    let valid_count_in_pack = std::cmp::min(
                        stored_height.saturating_sub(i_start),
                        PackedVal::<SC>::WIDTH,
                    );
                    let base_slice =
                        AbstractExtensionField::<PackedVal<SC>>::as_base_slice(&row_value_packed);
                    for idx_in_packing in 0..valid_count_in_pack {
                        let mut base_arr = vec![Val::<SC>::zero(); ext_d];
                        for coeff_idx in 0..ext_d {
                            base_arr[coeff_idx] = base_slice[coeff_idx].as_slice()[idx_in_packing];
                        }
                        acc += Challenge::<SC>::from_base_slice(&base_arr[..ext_d]);
                    }

                    (prep_buf, main_buf, perm_buf, acc)
                },
            )
            .map(|(_, _, _, acc)| acc)
            .sum();

        // Add padding rows' contribution, accounting for per-row eq coefficients.
        let num_padding = main_input.total_height - stored_height;
        let num_padding_rows = Val::<SC>::from_canonical_usize(num_padding);

        if self.padding_eval_main != Challenge::<SC>::zero() {
            if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                let padding_contribution: Challenge<SC> = (stored_height..main_input.total_height)
                    .map(|row| {
                        if row < eq_coeffs.len() {
                            self.padding_eval_main * eq_coeffs[row]
                        } else {
                            Challenge::<SC>::zero()
                        }
                    })
                    .sum();
                eval += padding_contribution;
            } else {
                eval += self.padding_eval_main * num_padding_rows;
            }
        }

        // Compute permutation eval (only when point <= degree_perm and permutation is not empty)
        let eval_perm_opt = if point <= degree_perm && perm_width > 0 {
            let mut eval_perm = permutation.values.iter().copied().sum::<Challenge<SC>>();
            eval_perm -=
                self.scaled_local_cumulative_sum() * Val::<SC>::from_canonical_usize(stored_height);
            eval_perm *= *self.powers_of_alpha.last().unwrap();
            eval_perm += self.padding_eval_perm * num_padding_rows;
            Some(eval_perm)
        } else {
            None
        };

        (eval, eval_perm_opt)
    }

    /// Compute permutation evaluation for first round at points `[0, var_degree]`.
    ///
    /// Note: This function is no longer used, permutation eval is computed
    /// as part of `compute_eval_first_round_nonlinear`.
    #[allow(dead_code)]
    fn compute_perm_eval_first_round(&self, point: usize, var_degree: usize) -> Challenge<SC> {
        let num_eval_blocks = var_degree + 1; // 2^k
        let block_height = 1 << self.log_height; // 2^{n} (original height)
        let block_size = block_height / num_eval_blocks; // 2^{n-k}

        // L = local_sum / 2^k
        let last_col_block_sum =
            self.local_cumulative_sum / Challenge::<SC>::from_canonical_usize(num_eval_blocks);

        // Sum permutation trace rows from start_row to end_row
        let start_row = point * block_size;
        let end_row = start_row + block_size;
        let block_sum = if let ChipTrace::FirstRound(mat) = &self.permutation {
            let mut sum = Challenge::<SC>::zero();
            if mat.main.height() >= end_row {
                for row in start_row..end_row {
                    for col in 0..mat.main.width() {
                        sum += Challenge::<SC>::from_base(mat.main.get(row, col));
                    }
                }
            }
            sum
        } else {
            Challenge::<SC>::zero()
        };

        block_sum - last_col_block_sum
    }

    /// Compute univariate polynomial for a linear round (non-first round) with algebraic
    /// decomposition.
    ///
    /// # Arguments
    ///
    /// * `permutation_challenges` - Permutation challenges
    /// * `public_values_ext` - Public values in extension field
    /// * `eq_poly` - Equality polynomial
    /// * `use_algebraic_decomp` - Whether to use algebraic decomposition optimization
    ///
    /// # Returns
    ///
    /// `UnipolyChipResult` with main/perm unipolys and optional auxiliary vectors.
    pub fn compute_unipoly_linear_non_first_round(
        &self,
        permutation_challenges: &[Challenge<SC>],
        public_values_ext: &[Challenge<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> UnipolyChipResult<Challenge<SC>> {
        let perm_empty = self.permutation.total_height() == 0;
        let mut chip_row_evaluations = None;
        let mut evals = Vec::with_capacity(self.chip_degree + 1);
        // deg(g_perm) = 1, it can be determined by the evaluations at 0, 1
        let mut evals_perm =
            if perm_empty { vec![Challenge::<SC>::zero(); 2] } else { Vec::with_capacity(2) };
        let mut points = Vec::with_capacity(self.chip_degree + 1);

        let (eval_at_zero, _) = if let Some(aux_vec) = self.aux_vectors.as_ref() {
            (aux_vec[0].par_iter().copied().sum(), None)
        } else {
            self.compute_eval_linear_non_first_round(
                0,
                permutation_challenges,
                public_values_ext,
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
            eq_challenge.inverse() * temp
        };

        evals.push(eval_at_zero);
        evals.push(eval_at_one);
        points.push(Challenge::<SC>::zero());
        points.push(Challenge::<SC>::one());

        if !perm_empty {
            // evaluate permutation at point 0
            let mut eval_perm_at_zero = self.permutation.get_sum_perm_rows_linear(0);
            // Treat `local_sum` as a virtual column: [local_sum / 2^n, local_sum / 2^n, ...,
            // local_sum / 2^n], where `n` is the initial total height of the chip.
            // The values in this column remain unchanged during folding operations,
            // except that the column length decreases as folding progresses.
            // (local_sum / 2^n) * 2^current_height / 2
            eval_perm_at_zero -= self.local_cumulative_sum *
                Challenge::<SC>::from_canonical_usize(self.permutation.total_height()) /
                Challenge::<SC>::from_canonical_usize(2 << self.log_height);

            // evaluate permutation at point 1
            let eval_perm_at_one = if let Some(last_power) = self.powers_of_alpha.last() {
                self.perm_claim / *last_power - eval_perm_at_zero
            } else {
                Challenge::<SC>::zero()
            };

            evals_perm.push(eval_perm_at_zero);
            evals_perm.push(eval_perm_at_one);
        }

        // If degree > 1, compute evaluations at extra points (in parallel).
        if self.chip_degree > 1 {
            let extra_points: Vec<usize> = (2..=self.chip_degree).collect();
            let computed: Vec<_> = extra_points
                .par_iter()
                .map(|&point| {
                    let (eval, aux_vec) = self.compute_eval_linear_non_first_round(
                        point,
                        permutation_challenges,
                        public_values_ext,
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

        // Multiply by the last power of alpha.
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

    /// Compute univariate polynomial for a linear round (first round) with algebraic decomposition.
    ///
    /// # Arguments
    ///
    /// * `permutation_challenges` - Permutation challenges
    /// * `public_values` - Public values in base field
    /// * `eq_poly` - Equality polynomial
    /// * `use_algebraic_decomp` - Whether to compute auxiliary vectors for algebraic decomposition
    ///
    /// # Returns
    ///
    /// `UnipolyChipResult<Challenge<SC>>` containing:
    ///   - Main univariate polynomial
    ///   - Permutation univariate polynomial
    ///   - Auxiliary vectors (if `use_algebraic_decomp` is true)
    pub fn compute_unipoly_linear_first_round(
        &self,
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> UnipolyChipResult<Challenge<SC>> {
        let perm_empty = self.permutation.total_height() == 0;
        let mut chip_row_evaluations = None;
        let mut evals = Vec::with_capacity(self.chip_degree + 1);
        // deg(g_perm) = 1
        let mut evals_perm =
            if perm_empty { vec![Challenge::<SC>::zero(); 2] } else { Vec::with_capacity(2) };
        let mut points = Vec::with_capacity(self.chip_degree + 1);

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

        if self.chip_degree > 1 {
            let extra_points: Vec<usize> = (2..=self.chip_degree).collect();
            let computed: Vec<_> = extra_points
                .par_iter()
                .map(|&point| {
                    let (eval, aux_vec) = self.compute_eval_linear_first_round(
                        point,
                        permutation_challenges,
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

        // alpha^i * g_perm(X)
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

    /// Compute evaluation for a linear round (non-first round) at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point at which to evaluate (0, 1, 2, ...)
    /// * `permutation_challenges` - Permutation challenges
    /// * `public_values_ext` - Public values in extension field
    /// * `eq_poly` - Equality polynomial
    /// * `use_algebraic_decomp` - Whether to compute auxiliary vectors for algebraic decomposition
    ///
    /// # Returns
    ///
    /// * (Challenge<SC>, Option<Vec<Challenge<SC>>>)
    ///   - Main constraint evaluation
    ///   - Auxiliary vector (if `use_algebraic_decomp` is true)
    #[allow(clippy::too_many_lines)]
    fn compute_eval_linear_non_first_round(
        &self,
        point: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values_ext: &[Challenge<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> (Challenge<SC>, Option<Vec<Challenge<SC>>>) {
        let (preprocessed_input, main_input, permutation_input) =
            self.get_summation_input_non_first_round(point, 1, None);

        let empty_matrix: RowMajorMatrix<Challenge<SC>> = RowMajorMatrix::default(0, 0);
        let preprocessed: &RowMajorMatrix<Challenge<SC>> = match &preprocessed_input {
            Some(mat) => &mat.main,
            None => &empty_matrix,
        };
        let main: &RowMajorMatrix<Challenge<SC>> = &main_input.main;
        let permutation: &RowMajorMatrix<Challenge<SC>> = &permutation_input.main;

        let total_height = main_input.height();
        let stored_height = main_input.stored_height();
        let prep_width = preprocessed.width();
        let main_width = main.width();
        let perm_width = permutation.width();

        // (1 - point) * self.is_first_row_value
        let one_minus_point = Val::<SC>::one() - Val::<SC>::from_canonical_usize(point);
        let first_value = self.is_first_row_value * one_minus_point;
        // point * self.is_last_row_value
        let point_val = Val::<SC>::from_canonical_usize(point);
        let last_value = self.is_last_row_value * point_val;

        let packed_permutation_challenges =
            permutation_challenges.iter().map(|c| PackedExt::<SC>::from(*c)).collect::<Vec<_>>();
        let packed_local_cumulative_sum = PackedExt::<SC>::from_f(self.local_cumulative_sum);

        let ext_d = <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
        let eq_poly_coeffs_opt = if !eq_poly.coeffs.is_empty() {
            Some(eq_poly.coeffs.last().unwrap().as_slice())
        } else {
            None
        };

        let block_starts: Vec<usize> = (0..stored_height).step_by(PackedVal::<SC>::WIDTH).collect();
        let nested: Vec<Vec<Challenge<SC>>> = block_starts
            .into_par_iter()
            .map(|i_start| {
                let is_first_row = PackedExt::<SC>::new(get_first_sel_ext_packed(
                    i_start,
                    first_value,
                    total_height,
                    stored_height,
                ));
                let is_last_row = PackedExt::<SC>::new(get_last_sel_ext_packed(
                    i_start,
                    last_value,
                    total_height,
                    stored_height,
                ));

                let mut prep_buf =
                    vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); prep_width];
                let mut main_buf =
                    vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); main_width];
                let mut perm_buf =
                    vec![PackedExt::<SC>::new(PackedChallenge::<SC>::zero()); perm_width];

                for col in 0..prep_width {
                    prep_buf[col] =
                        PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < preprocessed.height() {
                                    preprocessed.get(i_start + offset, col).as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        }));
                }

                for col in 0..main_width {
                    main_buf[col] =
                        PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < main.height() {
                                    main.get(i_start + offset, col).as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        }));
                }

                for col in 0..perm_width {
                    perm_buf[col] =
                        PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                if i_start + offset < permutation.height() {
                                    permutation.get(i_start + offset, col).as_base_slice()[i]
                                } else {
                                    Val::<SC>::zero()
                                }
                            })
                        }));
                }

                let mut folder: SumcheckConstraintFolderExt<'_, SC> = SumcheckConstraintFolderExt {
                    preprocessed: RowMajorMatrixView::new_row(&prep_buf),
                    main: RowMajorMatrixView::new_row(&main_buf),
                    permutation: RowMajorMatrixView::new_row(&perm_buf),
                    permutation_challenges: &packed_permutation_challenges,
                    is_first_row,
                    is_last_row,
                    powers_of_alpha: &self.powers_of_alpha,
                    accumulator: PackedExt::<SC>::zero(),
                    public_values: public_values_ext,
                    constraint_index: 0,
                    local_cumulative_sum: &packed_local_cumulative_sum,
                };

                self.chip_ext.eval(&mut folder);
                let mut row_value_packed = folder.accumulator;

                if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                    let eq = PackedExt::<SC>::new(PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            if i_start + offset < eq_coeffs.len() {
                                eq_coeffs[i_start + offset].as_base_slice()[i]
                            } else {
                                Val::<SC>::zero()
                            }
                        })
                    }));
                    row_value_packed *= eq;
                }

                let base_slice =
                    AbstractExtensionField::<PackedVal<SC>>::as_base_slice(&row_value_packed.inner);
                (0..std::cmp::min(PackedVal::<SC>::WIDTH, stored_height - i_start))
                    .map(|idx_in_packing| {
                        let mut base_arr = vec![Val::<SC>::zero(); ext_d];
                        for coeff_idx in 0..ext_d {
                            base_arr[coeff_idx] = base_slice[coeff_idx].as_slice()[idx_in_packing];
                        }
                        Challenge::<SC>::from_base_slice(&base_arr[..ext_d])
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut block_row_evals: Vec<Challenge<SC>> = nested.into_iter().flatten().collect();

        // Compute padding rows' contribution.
        let num_padding = main_input.total_height - stored_height;
        let padding_is_zero = self.padding_eval_main == Challenge::<SC>::zero();

        let padding_contribution: Challenge<SC> = if padding_is_zero {
            Challenge::<SC>::zero()
        } else if !eq_poly.coeffs.is_empty() {
            // Each padding row has the same constraint evaluation (`padding_eval_main`),
            // but must be multiplied by its own eq coefficient.
            let eq_poly_coeffs = eq_poly.coeffs.last().unwrap();
            (stored_height..main_input.total_height)
                .map(|row| {
                    if row < eq_poly_coeffs.len() {
                        self.padding_eval_main * eq_poly_coeffs[row]
                    } else {
                        Challenge::<SC>::zero()
                    }
                })
                .sum()
        } else {
            self.padding_eval_main * Val::<SC>::from_canonical_usize(num_padding)
        };

        let total_sum: Challenge<SC> =
            block_row_evals.par_iter().copied().sum::<Challenge<SC>>() + padding_contribution;

        if use_algebraic_decomp {
            if padding_is_zero {
                block_row_evals.extend(vec![Challenge::<SC>::zero(); num_padding]);
            } else if !eq_poly.coeffs.is_empty() {
                let eq_poly_coeffs = eq_poly.coeffs.last().unwrap();
                for row in stored_height..main_input.total_height {
                    let eq_coeff = if row < eq_poly_coeffs.len() {
                        eq_poly_coeffs[row]
                    } else {
                        Challenge::<SC>::zero()
                    };
                    block_row_evals.push(self.padding_eval_main * eq_coeff);
                }
            } else {
                block_row_evals.extend(vec![self.padding_eval_main; num_padding]);
            }
            (total_sum, Some(block_row_evals))
        } else {
            (total_sum, None)
        }
    }

    /// Compute evaluation for a linear round (first round) at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point at which to evaluate (0, 1, 2, ...)
    /// * `permutation_challenges` - Permutation challenges
    /// * `public_values` - Public values in base field
    /// * `eq_poly` - Equality polynomial
    /// * `use_algebraic_decomp` - Whether to compute auxiliary vectors for algebraic decomposition
    ///
    /// # Returns
    ///
    /// * (Challenge<SC>, Option<Vec<Challenge<SC>>>)
    ///   - Main constraint evaluation
    ///   - Auxiliary vector (if `use_algebraic_decomp` is true)
    #[allow(clippy::too_many_lines)]
    fn compute_eval_linear_first_round(
        &self,
        point: usize,
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
        eq_poly: &EqPoly<Val<SC>, Challenge<SC>>,
        use_algebraic_decomp: bool,
    ) -> (Challenge<SC>, Option<Vec<Challenge<SC>>>) {
        let (preprocessed_input, main_input, permutation_input) =
            self.get_summation_input_first_round(point, 1, None);

        let empty_matrix: RowMajorMatrix<Val<SC>> = RowMajorMatrix::default(0, 0);
        let preprocessed: &RowMajorMatrix<Val<SC>> = match &preprocessed_input {
            Some(mat) => &mat.main,
            None => &empty_matrix,
        };
        let main: &RowMajorMatrix<Val<SC>> = &main_input.main;
        let permutation: &RowMajorMatrix<Challenge<SC>> = &permutation_input.main;

        let total_height = main_input.height();
        let stored_height = main_input.stored_height();
        let prep_width = preprocessed.width();
        let main_width = main.width();
        let perm_width = permutation.width();

        // (1 - point) * Val::<SC>::one()
        let first_value = match point {
            0 => Val::<SC>::one(),
            1 => Val::<SC>::zero(),
            _ => -Val::<SC>::from_canonical_usize(point - 1),
        };
        // point * Val::<SC>::one()
        let last_value = match point {
            0 => Val::<SC>::zero(),
            1 => Val::<SC>::one(),
            _ => Val::<SC>::from_canonical_usize(point),
        };

        let packed_permutation_challenges = permutation_challenges
            .iter()
            .map(|c| PackedChallenge::<SC>::from_f(*c))
            .collect::<Vec<_>>();
        let packed_local_cumulative_sum = PackedChallenge::<SC>::from_f(self.local_cumulative_sum);

        let ext_d = <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
        let eq_poly_coeffs_opt = if !eq_poly.coeffs.is_empty() {
            Some(eq_poly.coeffs.last().unwrap().as_slice())
        } else {
            None
        };

        let block_starts: Vec<usize> = (0..stored_height).step_by(PackedVal::<SC>::WIDTH).collect();
        let nested: Vec<Vec<Challenge<SC>>> = block_starts
            .into_par_iter()
            .map(|i_start| {
                let is_first_row: PackedVal<SC> =
                    get_first_sel_packed(i_start, first_value, total_height, stored_height);
                let is_last_row: PackedVal<SC> =
                    get_last_sel_packed(i_start, last_value, total_height, stored_height);

                let mut prep_buf = vec![PackedVal::<SC>::zero(); prep_width];
                let mut main_buf = vec![PackedVal::<SC>::zero(); main_width];
                let mut perm_buf = vec![PackedChallenge::<SC>::zero(); perm_width];

                for col in 0..prep_width {
                    prep_buf[col] = PackedVal::<SC>::from_fn(|offset| {
                        if i_start + offset < preprocessed.height() {
                            preprocessed.get(i_start + offset, col)
                        } else {
                            Val::<SC>::zero()
                        }
                    });
                }

                for col in 0..main_width {
                    main_buf[col] = PackedVal::<SC>::from_fn(|offset| {
                        if i_start + offset < main.height() {
                            main.get(i_start + offset, col)
                        } else {
                            Val::<SC>::zero()
                        }
                    });
                }

                for col in 0..perm_width {
                    perm_buf[col] = PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            if i_start + offset < permutation.height() {
                                permutation.get(i_start + offset, col).as_base_slice()[i]
                            } else {
                                Val::<SC>::zero()
                            }
                        })
                    });
                }

                let mut folder = SumcheckConstraintFolder {
                    preprocessed: RowMajorMatrixView::new_row(&prep_buf),
                    main: RowMajorMatrixView::new_row(&main_buf),
                    permutation: RowMajorMatrixView::new_row(&perm_buf),
                    permutation_challenges: &packed_permutation_challenges,
                    local_cumulative_sum: &packed_local_cumulative_sum,
                    is_first_row,
                    is_last_row,
                    powers_of_alpha: &self.powers_of_alpha,
                    accumulator: PackedChallenge::<SC>::zero(),
                    public_values,
                    constraint_index: 0,
                };

                self.chip.eval(&mut folder);
                let mut row_value_packed = folder.accumulator;

                if let Some(eq_coeffs) = eq_poly_coeffs_opt {
                    let eq = PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            if i_start + offset < eq_coeffs.len() {
                                eq_coeffs[i_start + offset].as_base_slice()[i]
                            } else {
                                Val::<SC>::zero()
                            }
                        })
                    });
                    row_value_packed *= eq;
                }

                let base_slice =
                    AbstractExtensionField::<PackedVal<SC>>::as_base_slice(&row_value_packed);
                (0..std::cmp::min(PackedVal::<SC>::WIDTH, stored_height - i_start))
                    .map(|idx_in_packing| {
                        let mut base_arr = vec![Val::<SC>::zero(); ext_d];
                        for coeff_idx in 0..ext_d {
                            base_arr[coeff_idx] = base_slice[coeff_idx].as_slice()[idx_in_packing];
                        }
                        Challenge::<SC>::from_base_slice(&base_arr[..ext_d])
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut block_row_evals: Vec<Challenge<SC>> = nested.into_iter().flatten().collect();

        // Compute padding rows' contribution.
        let num_padding = main_input.total_height - stored_height;
        let padding_is_zero = self.padding_eval_main == Challenge::<SC>::zero();

        let padding_contribution: Challenge<SC> = if padding_is_zero {
            Challenge::<SC>::zero()
        } else if !eq_poly.coeffs.is_empty() {
            // Each padding row has the same constraint evaluation (`padding_eval_main`),
            // but must be multiplied by its own eq coefficient.
            let eq_poly_coeffs = eq_poly.coeffs.last().unwrap();
            (stored_height..main_input.total_height)
                .map(|row| {
                    if row < eq_poly_coeffs.len() {
                        self.padding_eval_main * eq_poly_coeffs[row]
                    } else {
                        Challenge::<SC>::zero()
                    }
                })
                .sum()
        } else {
            self.padding_eval_main * Val::<SC>::from_canonical_usize(num_padding)
        };

        let total_sum: Challenge<SC> =
            block_row_evals.par_iter().copied().sum::<Challenge<SC>>() + padding_contribution;

        if use_algebraic_decomp {
            if padding_is_zero {
                block_row_evals.extend(vec![Challenge::<SC>::zero(); num_padding]);
            } else if !eq_poly.coeffs.is_empty() {
                let eq_poly_coeffs = eq_poly.coeffs.last().unwrap();
                for row in stored_height..main_input.total_height {
                    let eq_coeff = if row < eq_poly_coeffs.len() {
                        eq_poly_coeffs[row]
                    } else {
                        Challenge::<SC>::zero()
                    };
                    block_row_evals.push(self.padding_eval_main * eq_coeff);
                }
            } else {
                block_row_evals.extend(vec![self.padding_eval_main; num_padding]);
            }
            (total_sum, Some(block_row_evals))
        } else {
            (total_sum, None)
        }
    }
}

/// Global sumcheck state aggregating all chips.
///
/// Contains the per-chip [`ChipState`] vector together with global fields such as
/// the equality polynomial, round configuration, and challenge history.
pub struct SumcheckState<'a, SC: SCStarkGenericConfig, A, AE> {
    /// Current round index (0-based).
    pub round_index: usize,
    /// Aggregated claim value across all chips.
    pub claim: Challenge<SC>,
    /// Per-chip states, ordered by chip index.
    pub chip_states: Vec<ChipState<'a, SC, A, AE>>,
    /// Equality polynomial used in the sumcheck protocol.
    pub eq_poly: EqPoly<Val<SC>, Challenge<SC>>,
    /// Bit-expansion polynomial for expanding one challenge into multiple in skip rounds.
    pub bit_expand_poly: BitExpandPoly<Val<SC>>,
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
    /// Permutation challenges.
    pub permutation_challenges: [Challenge<SC>; 2],
    /// Public values (base field).
    pub public_values: &'a [Val<SC>],
    /// Public values (extension field).
    pub public_values_ext: Vec<Challenge<SC>>,
    /// Cumulative number of chips participating up to each round.
    pub num_chips_each_round: Vec<usize>,
}

impl<SC: SCStarkGenericConfig, A, AE> std::fmt::Debug for SumcheckState<'_, SC, A, AE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumcheckState")
            .field("round_index", &self.round_index)
            .field("claim", &self.claim)
            .field("chip_states", &self.chip_states)
            .field("eq_poly", &self.eq_poly)
            .field("bit_expand_poly", &self.bit_expand_poly)
            .field("sumcheck_challenges", &self.sumcheck_challenges)
            .field("num_rounds", &self.num_rounds)
            .field("num_rounds_linear", &self.num_rounds_linear)
            .field("num_skip_rounds", &self.num_skip_rounds)
            .field("log_height_threshold", &self.log_height_threshold)
            .field("permutation_challenges", &self.permutation_challenges)
            .field("public_values_ext", &self.public_values_ext)
            .field("num_chips_each_round", &self.num_chips_each_round)
            .finish()
    }
}

impl<'a, SC: SCStarkGenericConfig, A, AE> SumcheckState<'a, SC, A, AE> {
    /// Create a new `SumcheckState`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chip_states: Vec<ChipState<'a, SC, A, AE>>,
        eq_challenges: Vec<Challenge<SC>>,
        bit_expand_poly_points: Vec<Val<SC>>,
        num_rounds: usize,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
        log_height_threshold: usize,
        permutation_challenges: [Challenge<SC>; 2],
        public_values: &'a [Val<SC>],
        public_values_ext: Vec<Challenge<SC>>,
        num_chips_each_round: Vec<usize>,
    ) -> Self {
        // Initial claim is zero; the first round derives it from chip unipolys times eq.
        let claim = Challenge::<SC>::zero();

        // Create the eq polynomial (nonlinear-round variable degree = 2^num_skip_rounds - 1).
        let degree = (1 << num_skip_rounds) - 1;
        let eq_poly = EqPoly::new(eq_challenges, num_rounds_linear, degree);

        let bit_expand_poly = BitExpandPoly::new(bit_expand_poly_points);

        Self {
            round_index: 0,
            claim,
            chip_states,
            eq_poly,
            bit_expand_poly,
            sumcheck_challenges: vec![],
            num_rounds,
            num_rounds_linear,
            num_skip_rounds,
            log_height_threshold,
            permutation_challenges,
            public_values,
            public_values_ext,
            num_chips_each_round,
        }
    }

    /// Returns the chips participating in the current round.
    #[allow(dead_code)]
    pub fn current_round_chips(&self) -> &[ChipState<'a, SC, A, AE>] {
        let num_chips = self.num_chips_current_round();
        &self.chip_states[..num_chips]
    }

    /// Returns the chips participating in the current round (mutable).
    pub fn current_round_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, AE>] {
        let num_chips = self.num_chips_current_round();
        &mut self.chip_states[..num_chips]
    }

    /// Returns the chips newly introduced in the current round.
    pub fn new_chips(&self) -> &[ChipState<'a, SC, A, AE>] {
        let prev_num_chips = self.num_chips_prev_round();
        let curr_num_chips = self.num_chips_current_round();
        &self.chip_states[prev_num_chips..curr_num_chips]
    }

    /// Returns the chips newly introduced in the current round (mutable).
    #[allow(dead_code)]
    pub fn new_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, AE>] {
        let prev_num_chips = self.num_chips_prev_round();
        let curr_num_chips = self.num_chips_current_round();
        &mut self.chip_states[prev_num_chips..curr_num_chips]
    }

    /// Returns the chips that existed in the previous round.
    pub fn prev_round_chips(&self) -> &[ChipState<'a, SC, A, AE>] {
        let prev_num_chips = self.num_chips_prev_round();
        &self.chip_states[..prev_num_chips]
    }

    /// Returns the chips that existed in the previous round (mutable).
    #[allow(dead_code)]
    pub fn prev_round_chips_mut(&mut self) -> &mut [ChipState<'a, SC, A, AE>] {
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

    /// Finalize the sumcheck protocol and return opened values for each chip.
    ///
    /// Each chip produces a single `local` row from its preprocessed, main, and
    /// permutation traces, along with its cumulative sums.
    pub fn finalize(self) -> Vec<SCChipOpenedValues<Val<SC>, Challenge<SC>>>
    where
        Challenge<SC>: ExtensionField<Val<SC>>,
    {
        self.chip_states
            .into_iter()
            .map(|cs| {
                let preprocessed = match &cs.preprocessed {
                    Some(prep) => SCAirOpenedValues { local: prep.get_row_for_opening(0) },
                    None => SCAirOpenedValues { local: vec![] },
                };
                let main = SCAirOpenedValues { local: cs.main.get_row_for_opening(0) };
                let permutation =
                    SCAirOpenedValues { local: cs.permutation.get_row_for_opening(0) };
                SCChipOpenedValues {
                    preprocessed,
                    main,
                    permutation,
                    local_cumulative_sum: cs.local_cumulative_sum,
                    log_height: cs.log_height,
                    _field: core::marker::PhantomData,
                }
            })
            .collect()
    }
}
