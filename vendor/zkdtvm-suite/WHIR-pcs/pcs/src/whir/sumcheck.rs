use crate::utils::mlpoly::MultilinearPolynomial;
use crate::utils::unipoly::UniPoly;
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Eq, PartialEq, Error)]
pub enum SumcheckError {
    #[error("invalid proof input")]
    InvalidProofInput,
    #[error("sumcheck round inconsistency")]
    SumcheckRoundInconsistency,
    #[error("inconsistency of prover message with claimed sum")]
    InconsistencyWithClaimedSum,
    #[error("inconsistency of proof with evaluation claim")]
    InconsistencyWithEval,
}

/// Univariate polynomials generated in sumcheck protocol
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckInstanceProof<F> {
    pub uni_polys: Vec<UniPoly<F>>,
}

pub(crate) enum PairProductLeftInput<'a, F: Field, ExtF: Field> {
    Base(&'a RowMajorMatrix<F>),
    Ext(&'a RowMajorMatrix<ExtF>),
}

impl<'a, F: Field, ExtF: Field> PairProductLeftInput<'a, F, ExtF> {
    fn height(&self) -> usize {
        match self {
            Self::Base(matrix) => matrix.height(),
            Self::Ext(matrix) => matrix.height(),
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Base(matrix) => matrix.width(),
            Self::Ext(matrix) => matrix.width(),
        }
    }
}

impl<F: Field> SumcheckInstanceProof<F> {
    pub fn new(uni_polys: Vec<UniPoly<F>>) -> Self {
        Self { uni_polys }
    }

    fn validate_prover_inputs<ExtF: Field>(
        num_rounds: usize,
        polys: &[MultilinearPolynomial<ExtF>],
        combined_degree: usize,
    ) -> Result<usize, SumcheckError> {
        if combined_degree != 2 || polys.is_empty() {
            return Err(SumcheckError::InvalidProofInput);
        }

        let num_vars = polys[0].num_vars();
        if num_rounds > num_vars || polys.iter().any(|poly| poly.num_vars() != num_vars) {
            return Err(SumcheckError::InvalidProofInput);
        }

        Ok(polys.len())
    }

    /// Run `num_rounds` rounds of the sumcheck protocol for degree-2 combined polynomials.
    ///
    /// In each round, the prover constructs a univariate polynomial `g(X)` such that
    /// `g(0) + g(1) = claim`. The optimization exploits this identity: only `g(0)` and
    /// `g(2)` are computed directly, while `g(1) = claim - g(0)` is derived.
    ///
    /// Params
    /// - `claim`: The current sumcheck claim; updated each round to `g(r_j)`
    /// - `num_rounds`: Number of variables to bind (one challenge per round)
    /// - `polys`: Multilinear polynomials to combine; each is halved every round
    /// - `comb_func`: Combination function applied to polynomial evaluations (must be degree 2)
    /// - `combined_degree`: Must be 2
    /// - `transcript`: Fiat-Shamir transcript for challenge generation
    ///
    /// Returns `(proof, challenges, final_evals)`:
    /// - `proof`: The univariate polynomials from each round
    /// - `challenges`: The random folding challenges `[r_0, r_1, ..., r_{n-1}]`
    /// - `final_evals`: Each polynomial evaluated at the final point (single element)
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, level = "debug", name = "Sumcheck normal round prove")]
    pub fn sumcheck_prove_normal_round<
        Func,
        ExtF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    >(
        claim: &ExtF,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<ExtF>>,
        comb_func: &Func,
        combined_degree: usize,
        transcript: &mut Challenger,
    ) -> Result<(SumcheckInstanceProof<ExtF>, Vec<ExtF>, Vec<ExtF>), SumcheckError>
    where
        Func: Fn(&[ExtF]) -> ExtF + std::marker::Sync,
    {
        let polys_len = Self::validate_prover_inputs(num_rounds, polys, combined_degree)?;
        let mut r = Vec::with_capacity(num_rounds);
        let mut uni_polys = Vec::with_capacity(num_rounds);
        let mut running_claim = *claim;

        for _round in 0..num_rounds {
            let mle_half = polys[0].len() / 2;

            // Little-endian folding: use even/odd indexed elements instead of low/high halves.
            // g(0) = Σ_i comb_func(poly[2i] for each poly)  (even positions, xₙ₋₁=0)
            // g(2) = Σ_i comb_func(2*poly[2i+1] - poly[2i]) (extrapolated at X=2)
            let (eval_0, eval_2) = (0..mle_half)
                .into_par_iter()
                .fold(
                    || (ExtF::zero(), ExtF::zero(), vec![ExtF::zero(); polys_len]),
                    |(mut acc_0, mut acc_2, mut params), poly_term_i| {
                        for (i, poly) in polys.iter().enumerate() {
                            params[i] = poly[2 * poly_term_i];
                        }
                        acc_0 += comb_func(&params);

                        for (i, poly) in polys.iter().enumerate() {
                            let even = poly[2 * poly_term_i];
                            let odd = poly[2 * poly_term_i + 1];
                            params[i] = odd + odd - even;
                        }
                        acc_2 += comb_func(&params);

                        (acc_0, acc_2, params)
                    },
                )
                .map(|(eval_0, eval_2, _params)| (eval_0, eval_2))
                .reduce(
                    || (ExtF::zero(), ExtF::zero()),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                );

            let eval_1 = running_claim - eval_0;
            let round_uni_poly = UniPoly::from_evals_quadratic_012(eval_0, eval_1, eval_2);

            round_uni_poly.coeffs.iter().for_each(|coeff| {
                transcript.observe_ext_element(*coeff);
            });

            let r_j = transcript.sample_ext_element();
            r.push(r_j);

            running_claim = round_uni_poly.evaluate(&r_j);

            polys.par_iter_mut().for_each(|poly| {
                poly.bound_poly_var_bottom(&r_j);
            });
            uni_polys.push(round_uni_poly);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        Ok((
            SumcheckInstanceProof::<ExtF>::new(uni_polys),
            r,
            final_evals,
        ))
    }

    /// Specialized degree-2 sumcheck for `Σ_j polys[2j] * polys[2j + 1]`.
    ///
    /// This is the hot path for stacked opening reduction and avoids building a
    /// temporary parameter vector for every hypercube pair.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(
        skip_all,
        level = "debug",
        name = "Sumcheck interleaved pair-products prove"
    )]
    pub fn sumcheck_prove_interleaved_pair_products<
        ExtF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    >(
        claim: &ExtF,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<ExtF>>,
        transcript: &mut Challenger,
    ) -> Result<(SumcheckInstanceProof<ExtF>, Vec<ExtF>, Vec<ExtF>), SumcheckError> {
        let polys_len = Self::validate_prover_inputs(num_rounds, polys, 2)?;
        if polys_len % 2 != 0 {
            return Err(SumcheckError::InvalidProofInput);
        }

        let mut r = Vec::with_capacity(num_rounds);
        let mut uni_polys = Vec::with_capacity(num_rounds);
        let mut running_claim = *claim;
        let num_pairs = polys_len / 2;

        for _round in 0..num_rounds {
            let mle_half = polys[0].len() / 2;

            let (eval_0, eval_2) = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut e0 = ExtF::zero();
                    let mut e2 = ExtF::zero();

                    for pair_idx in 0..num_pairs {
                        let left = &polys[2 * pair_idx];
                        let right = &polys[2 * pair_idx + 1];

                        let left_even = left[2 * poly_term_i];
                        let right_even = right[2 * poly_term_i];
                        e0 += left_even * right_even;

                        let left_odd = left[2 * poly_term_i + 1];
                        let right_odd = right[2 * poly_term_i + 1];
                        let left_at_2 = left_odd + left_odd - left_even;
                        let right_at_2 = right_odd + right_odd - right_even;
                        e2 += left_at_2 * right_at_2;
                    }

                    (e0, e2)
                })
                .reduce(
                    || (ExtF::zero(), ExtF::zero()),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                );

            let eval_1 = running_claim - eval_0;
            let round_uni_poly = UniPoly::from_evals_quadratic_012(eval_0, eval_1, eval_2);

            round_uni_poly.coeffs.iter().for_each(|coeff| {
                transcript.observe_ext_element(*coeff);
            });

            let r_j = transcript.sample_ext_element();
            r.push(r_j);

            running_claim = round_uni_poly.evaluate(&r_j);

            polys.par_iter_mut().for_each(|poly| {
                poly.bound_poly_var_bottom(&r_j);
            });
            uni_polys.push(round_uni_poly);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        Ok((
            SumcheckInstanceProof::<ExtF>::new(uni_polys),
            r,
            final_evals,
        ))
    }

    /// Matrix-oriented pair-product sumcheck for `Σ_col F_col * Q_col`.
    ///
    /// The left side may borrow an existing base-field stacked matrix. In that
    /// case the first round reads it directly and only allocates the folded
    /// extension-field working matrix after the first challenge.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(
        skip_all,
        level = "debug",
        name = "Sumcheck matrix pair-products prove"
    )]
    pub(crate) fn sumcheck_prove_pair_products<
        'a,
        ExtF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    >(
        claim: &ExtF,
        num_rounds: usize,
        left_inputs: &[PairProductLeftInput<'a, F, ExtF>],
        q_matrices: &mut [RowMajorMatrix<ExtF>],
        transcript: &mut Challenger,
    ) -> Result<(SumcheckInstanceProof<ExtF>, Vec<ExtF>, Vec<ExtF>), SumcheckError> {
        Self::validate_pair_product_matrix_inputs(num_rounds, left_inputs, q_matrices)?;

        let mut r = Vec::with_capacity(num_rounds);
        let mut uni_polys = Vec::with_capacity(num_rounds);
        let mut running_claim = *claim;
        let mut folded_lefts: Option<Vec<RowMajorMatrix<ExtF>>> = None;

        for _round in 0..num_rounds {
            let current_height = q_matrices[0].height();
            let mle_half = current_height / 2;

            let (eval_0, eval_2) = (0..mle_half)
                .into_par_iter()
                .map(|row| {
                    Self::pair_product_matrix_round_contribution(
                        row,
                        left_inputs,
                        folded_lefts.as_deref(),
                        q_matrices,
                    )
                })
                .reduce(
                    || (ExtF::zero(), ExtF::zero()),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                );

            let eval_1 = running_claim - eval_0;
            let round_uni_poly = UniPoly::from_evals_quadratic_012(eval_0, eval_1, eval_2);

            round_uni_poly.coeffs.iter().for_each(|coeff| {
                transcript.observe_ext_element(*coeff);
            });

            let r_j = transcript.sample_ext_element();
            r.push(r_j);

            running_claim = round_uni_poly.evaluate(&r_j);

            if let Some(folded) = folded_lefts.as_mut() {
                Self::fold_ext_matrices_in_place(folded, &r_j);
            } else {
                folded_lefts = Some(Self::fold_left_inputs(left_inputs, &r_j));
            }
            Self::fold_ext_matrices_in_place(q_matrices, &r_j);

            uni_polys.push(round_uni_poly);
        }

        let final_evals =
            Self::pair_product_matrix_final_evals(left_inputs, folded_lefts.as_deref(), q_matrices);

        Ok((
            SumcheckInstanceProof::<ExtF>::new(uni_polys),
            r,
            final_evals,
        ))
    }

    fn validate_pair_product_matrix_inputs<ExtF: ExtensionField<F>>(
        num_rounds: usize,
        left_inputs: &[PairProductLeftInput<'_, F, ExtF>],
        q_matrices: &[RowMajorMatrix<ExtF>],
    ) -> Result<(), SumcheckError> {
        if left_inputs.is_empty() || left_inputs.len() != q_matrices.len() {
            return Err(SumcheckError::InvalidProofInput);
        }

        let height = left_inputs[0].height();
        if height == 0 || !height.is_power_of_two() {
            return Err(SumcheckError::InvalidProofInput);
        }
        let num_vars = height.ilog2() as usize;
        if num_rounds > num_vars {
            return Err(SumcheckError::InvalidProofInput);
        }

        let mut total_width = 0usize;
        for (left, q) in left_inputs.iter().zip(q_matrices.iter()) {
            if left.height() != height || q.height() != height || left.width() != q.width() {
                return Err(SumcheckError::InvalidProofInput);
            }
            total_width += left.width();
        }
        if total_width == 0 {
            return Err(SumcheckError::InvalidProofInput);
        }

        Ok(())
    }

    fn pair_product_matrix_round_contribution<ExtF: ExtensionField<F>>(
        row: usize,
        left_inputs: &[PairProductLeftInput<'_, F, ExtF>],
        folded_lefts: Option<&[RowMajorMatrix<ExtF>]>,
        q_matrices: &[RowMajorMatrix<ExtF>],
    ) -> (ExtF, ExtF) {
        let mut eval_0 = ExtF::zero();
        let mut eval_2 = ExtF::zero();

        for matrix_idx in 0..q_matrices.len() {
            let q_even = q_matrices[matrix_idx].row_slice(2 * row);
            let q_odd = q_matrices[matrix_idx].row_slice(2 * row + 1);

            match folded_lefts {
                Some(folded) => {
                    let f_even = folded[matrix_idx].row_slice(2 * row);
                    let f_odd = folded[matrix_idx].row_slice(2 * row + 1);
                    for col in 0..q_matrices[matrix_idx].width() {
                        let left_even = f_even[col];
                        let right_even = q_even[col];
                        eval_0 += left_even * right_even;

                        let left_odd = f_odd[col];
                        let right_odd = q_odd[col];
                        let left_at_2 = left_odd + left_odd - left_even;
                        let right_at_2 = right_odd + right_odd - right_even;
                        eval_2 += left_at_2 * right_at_2;
                    }
                }
                None => match &left_inputs[matrix_idx] {
                    PairProductLeftInput::Base(matrix) => {
                        let f_even = matrix.row_slice(2 * row);
                        let f_odd = matrix.row_slice(2 * row + 1);
                        for col in 0..q_matrices[matrix_idx].width() {
                            let left_even = ExtF::from_base(f_even[col]);
                            let right_even = q_even[col];
                            eval_0 += left_even * right_even;

                            let left_odd = ExtF::from_base(f_odd[col]);
                            let right_odd = q_odd[col];
                            let left_at_2 = left_odd + left_odd - left_even;
                            let right_at_2 = right_odd + right_odd - right_even;
                            eval_2 += left_at_2 * right_at_2;
                        }
                    }
                    PairProductLeftInput::Ext(matrix) => {
                        let f_even = matrix.row_slice(2 * row);
                        let f_odd = matrix.row_slice(2 * row + 1);
                        for col in 0..q_matrices[matrix_idx].width() {
                            let left_even = f_even[col];
                            let right_even = q_even[col];
                            eval_0 += left_even * right_even;

                            let left_odd = f_odd[col];
                            let right_odd = q_odd[col];
                            let left_at_2 = left_odd + left_odd - left_even;
                            let right_at_2 = right_odd + right_odd - right_even;
                            eval_2 += left_at_2 * right_at_2;
                        }
                    }
                },
            }
        }

        (eval_0, eval_2)
    }

    fn fold_left_inputs<ExtF: ExtensionField<F>>(
        left_inputs: &[PairProductLeftInput<'_, F, ExtF>],
        r: &ExtF,
    ) -> Vec<RowMajorMatrix<ExtF>> {
        left_inputs
            .iter()
            .map(|left| match left {
                PairProductLeftInput::Base(matrix) => {
                    Self::fold_base_matrix_to_ext::<ExtF>(matrix, r)
                }
                PairProductLeftInput::Ext(matrix) => Self::fold_ext_matrix_to_ext(matrix, r),
            })
            .collect()
    }

    fn fold_base_matrix_to_ext<ExtF: ExtensionField<F>>(
        matrix: &RowMajorMatrix<F>,
        r: &ExtF,
    ) -> RowMajorMatrix<ExtF> {
        let half_height = matrix.height() / 2;
        let width = matrix.width();
        let mut values = vec![ExtF::zero(); half_height * width];
        values
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out_row)| {
                let even = matrix.row_slice(2 * row);
                let odd = matrix.row_slice(2 * row + 1);
                for col in 0..width {
                    let even = ExtF::from_base(even[col]);
                    let odd = ExtF::from_base(odd[col]);
                    out_row[col] = even + *r * (odd - even);
                }
            });
        RowMajorMatrix::new(values, width)
    }

    fn fold_ext_matrix_to_ext<ExtF: ExtensionField<F>>(
        matrix: &RowMajorMatrix<ExtF>,
        r: &ExtF,
    ) -> RowMajorMatrix<ExtF> {
        let half_height = matrix.height() / 2;
        let width = matrix.width();
        let mut values = vec![ExtF::zero(); half_height * width];
        values
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out_row)| {
                let even = matrix.row_slice(2 * row);
                let odd = matrix.row_slice(2 * row + 1);
                for col in 0..width {
                    out_row[col] = even[col] + *r * (odd[col] - even[col]);
                }
            });
        RowMajorMatrix::new(values, width)
    }

    fn fold_ext_matrices_in_place<ExtF: ExtensionField<F>>(
        matrices: &mut [RowMajorMatrix<ExtF>],
        r: &ExtF,
    ) {
        matrices.par_iter_mut().for_each(|matrix| {
            let half_height = matrix.height() / 2;
            let width = matrix.width();
            for row in 0..half_height {
                for col in 0..width {
                    let even_idx = (2 * row) * width + col;
                    let odd_idx = even_idx + width;
                    let even = matrix.values[even_idx];
                    let odd = matrix.values[odd_idx];
                    matrix.values[row * width + col] = even + *r * (odd - even);
                }
            }
            matrix.values.truncate(half_height * width);
        });
    }

    fn pair_product_matrix_final_evals<ExtF: ExtensionField<F>>(
        left_inputs: &[PairProductLeftInput<'_, F, ExtF>],
        folded_lefts: Option<&[RowMajorMatrix<ExtF>]>,
        q_matrices: &[RowMajorMatrix<ExtF>],
    ) -> Vec<ExtF> {
        let total_width: usize = q_matrices.iter().map(|matrix| matrix.width()).sum();
        let mut final_evals = Vec::with_capacity(2 * total_width);

        for matrix_idx in 0..q_matrices.len() {
            let q_row = q_matrices[matrix_idx].row_slice(0);
            match folded_lefts {
                Some(folded) => {
                    let f_row = folded[matrix_idx].row_slice(0);
                    for col in 0..q_matrices[matrix_idx].width() {
                        final_evals.push(f_row[col]);
                        final_evals.push(q_row[col]);
                    }
                }
                None => match &left_inputs[matrix_idx] {
                    PairProductLeftInput::Base(matrix) => {
                        let f_row = matrix.row_slice(0);
                        for col in 0..q_matrices[matrix_idx].width() {
                            final_evals.push(ExtF::from_base(f_row[col]));
                            final_evals.push(q_row[col]);
                        }
                    }
                    PairProductLeftInput::Ext(matrix) => {
                        let f_row = matrix.row_slice(0);
                        for col in 0..q_matrices[matrix_idx].width() {
                            final_evals.push(f_row[col]);
                            final_evals.push(q_row[col]);
                        }
                    }
                },
            }
        }

        final_evals
    }
}
