use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::utils::eqpoly::EqPolynomial;
use crate::utils::mlpoly::MultilinearPolynomial;
use crate::utils::unipoly::UniPoly;
use crate::utils::math::compute_dotproduct;

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

impl<F: Field> SumcheckInstanceProof<F> {
    pub fn new(uni_polys: Vec<UniPoly<F>>) -> Self {
        Self { uni_polys }
    }

    /// Merge two sumcheck instances by performing one round of sumcheck on their virtual concatenation.
    ///
    /// Given two instances `[F_0, EQ_0]` and `[F_1, EQ_1]`, virtually concatenate them as
    /// `F = [F_0 || F_1]` and `EQ = [EQ_0 || EQ_1]`, then run one sumcheck round to produce
    /// a univariate polynomial `g(X) = Σ_cube F(X, cube) · EQ(X, cube)` and fold both
    /// polynomials using the resulting challenge `r`.
    ///
    /// The key optimization: `g(0) = claim_0` and `g(1) = claim_1` are already known
    /// (they are the inner products of the respective instances), so we only need to
    /// compute `g(2)` directly.
    ///
    /// Returns `(g(X), r, [F_folded, EQ_folded])` where:
    /// - `F_folded  = F_0 + r · (F_1 - F_0)`
    /// - `EQ_folded = EQ_0 + r · (EQ_1 - EQ_0)`
    #[tracing::instrument(skip_all, level="debug", name = "Sumcheck merge two instances")]
    pub fn sumcheck_prove_merge_two_instances<ExtF: ExtensionField<F> + From<F>, Challenger: FieldChallenger<F>>(
        claim_0: &ExtF,
        claim_1: &ExtF,
        sumcheck_instance_0: &Vec<MultilinearPolynomial<ExtF>>,      // [F_0, EQ_0]
        sumcheck_instance_1: &Vec<MultilinearPolynomial<ExtF>>,      // [F_1, EQ_1]
        transcript: &mut Challenger,
    ) -> Result<(UniPoly<ExtF>, ExtF, Vec<MultilinearPolynomial<ExtF>>), SumcheckError>
    {
        assert_eq!(sumcheck_instance_0[0].len(), sumcheck_instance_1[0].len());
        assert_eq!(sumcheck_instance_0[1].len(), sumcheck_instance_1[1].len());
        assert_eq!(sumcheck_instance_0.len(), 2);
        assert_eq!(sumcheck_instance_1.len(), 2);

        let poly_len = sumcheck_instance_0[0].len();

        // g(0) = claim_0, g(1) = claim_1 — already known, no computation needed.
        // Only compute g(2) by evaluating at X=2: F(2) = 2*F_1 - F_0, EQ(2) = 2*EQ_1 - EQ_0
        let eval_2: ExtF = (0..poly_len)
            .into_par_iter()
            .map(|idx| {
                let f_at_2 = sumcheck_instance_1[0].evals[idx] + sumcheck_instance_1[0].evals[idx]
                    - sumcheck_instance_0[0].evals[idx];
                let eq_at_2 = sumcheck_instance_1[1].evals[idx] + sumcheck_instance_1[1].evals[idx]
                    - sumcheck_instance_0[1].evals[idx];
                f_at_2 * eq_at_2
            })
            .sum();

        let eval_points = vec![*claim_0, *claim_1, eval_2];
        let round_uni_poly = UniPoly::from_evals(&eval_points);
        round_uni_poly.coeffs.iter().for_each(|coeff| {
            transcript.observe_ext_element(coeff.clone());
        });

        let r = transcript.sample_ext_element();

        // Fold: F_folded = F_0 + r * (F_1 - F_0), EQ_folded = EQ_0 + r * (EQ_1 - EQ_0)
        let (folded_f, folded_eq): (Vec<_>, Vec<_>) = (0..poly_len)
            .into_par_iter()
            .map(|idx| {
                let f0 = sumcheck_instance_0[0].evals[idx];
                let f1 = sumcheck_instance_1[0].evals[idx];
                let s0 = sumcheck_instance_0[1].evals[idx];
                let s1 = sumcheck_instance_1[1].evals[idx];

                (
                    f0 + r * (f1 - f0),
                    s0 + r * (s1 - s0)
                )
            })
            .unzip();

        let folded_f_and_eq = vec![
            MultilinearPolynomial::new(folded_f),
            MultilinearPolynomial::new(folded_eq),
        ];

        Ok((round_uni_poly, r, folded_f_and_eq))
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
    /// - `combined_degree`: Must be 2 (asserted)
    /// - `transcript`: Fiat-Shamir transcript for challenge generation
    ///
    /// Returns `(proof, challenges, final_evals)`:
    /// - `proof`: The univariate polynomials from each round
    /// - `challenges`: The random folding challenges `[r_0, r_1, ..., r_{n-1}]`
    /// - `final_evals`: Each polynomial evaluated at the final point (single element)
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, level="debug", name = "Sumcheck normal round prove")]
    pub fn sumcheck_prove_normal_round<Func, ExtF: ExtensionField<F>, Challenger: FieldChallenger<F>>(
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
        assert_eq!(combined_degree, 2);

        let mut r = Vec::with_capacity(num_rounds);
        let mut uni_polys = Vec::with_capacity(num_rounds);
        let polys_len = polys.len();
        let mut running_claim = *claim;

        for _round in 0..num_rounds {
            let mle_half = polys[0].len() / 2;

            // Little-endian folding: use even/odd indexed elements instead of low/high halves.
            // g(0) = Σ_i comb_func(poly[2i] for each poly)  (even positions, xₙ₋₁=0)
            // g(2) = Σ_i comb_func(2*poly[2i+1] - poly[2i]) (extrapolated at X=2)
            let (eval_0, eval_2) = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut params = vec![ExtF::zero(); polys_len];

                    for (i, poly) in polys.iter().enumerate() {
                        params[i] = poly[2 * poly_term_i];
                    }
                    let e0 = comb_func(&params);

                    for (i, poly) in polys.iter().enumerate() {
                        let even = poly[2 * poly_term_i];
                        let odd = poly[2 * poly_term_i + 1];
                        params[i] = odd + odd - even;
                    }
                    let e2 = comb_func(&params);

                    (e0, e2)
                })
                .reduce(
                    || (ExtF::zero(), ExtF::zero()),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2)
                );

            let eval_1 = running_claim - eval_0;

            let eval_points = vec![eval_0, eval_1, eval_2];
            let round_uni_poly = UniPoly::from_evals(&eval_points);

            round_uni_poly.coeffs.iter().for_each(|coeff| {
                transcript.observe_ext_element(coeff.clone());
            });

            let r_j = transcript.sample_ext_element();
            r.push(r_j);

            running_claim = round_uni_poly.evaluate(&r_j);

            polys.par_iter_mut().for_each(|poly| {
                poly.bound_poly_var_bottom(&r_j);
                poly.evals.truncate(poly.len());
            });
            uni_polys.push(round_uni_poly);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        Ok((SumcheckInstanceProof::<ExtF>::new(uni_polys), r, final_evals))
    }

    /// Perform one WHIR (Weighted Hypercube Interpolation Reduction) round.
    ///
    /// Samples an out-of-domain challenge `alpha` and a linear-combination coefficient `gamma`.
    /// Computes `y = <polys[0], eq(alpha, ·)>` — the out-of-domain evaluation of the
    /// polynomial at the point defined by successive squares of `alpha`.
    /// Then folds `eq(alpha, ·)` into the EQ polynomial: `polys[1] += gamma · eq(alpha, ·)`.
    ///
    /// Returns `(alpha, gamma, y)`.
    #[tracing::instrument(skip_all, level="debug", name = "Sumcheck WHIR round")]
    pub fn sumcheck_prove_whir_round<ExtF: ExtensionField<F> + From<F>, Challenger: FieldChallenger<F>>(
        polys: &mut Vec<MultilinearPolynomial<ExtF>>,
        transcript: &mut Challenger,
    ) -> Result<(ExtF, ExtF, ExtF), SumcheckError> {
        // out-of-domain challenge
        let alpha: ExtF = transcript.sample_ext_element();
        // random coefficient used for linear combination
        let gamma: ExtF = transcript.sample_ext_element();
        let log_poly_len = polys[0].get_num_vars();

        // (alpha, alpha^2, alpha^{2^2}, ..., )
        let powers_of_alpha: Vec<ExtF> = std::iter::successors(Some(alpha.clone()), |prev| Some(prev.clone() * prev.clone()))
            .take(log_poly_len)
            .collect();

        // eq(alpha, ·) evaluated over the boolean hypercube
        let eq_eval_at_alpha = EqPolynomial::new(powers_of_alpha).to_ml();

        // y = <polys[0], eq(alpha, ·)>, the out-of-domain evaluation
        let y = compute_dotproduct(&polys[0], &eq_eval_at_alpha);
        transcript.observe_ext_element(y);

        // Fold into the EQ polynomial: polys[1] += gamma * eq(alpha, ·)
        polys[1].par_iter_mut().zip(eq_eval_at_alpha.into_par_iter()).for_each(|(a, b)| {
            *a += *b * gamma;
        });

        Ok((alpha, gamma, y))
    }
}
