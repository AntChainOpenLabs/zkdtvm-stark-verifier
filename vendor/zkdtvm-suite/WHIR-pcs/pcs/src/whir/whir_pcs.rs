use std::collections::BTreeMap;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Powers, TwoAdicField};
use p3_fri::prover::{answer_queries_pruned, answer_query};
use p3_fri::BatchOpening;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::utils::eqpoly::EqPolynomial;
use crate::utils::math::{compute_dotproduct, compute_dotproduct_mix};
use crate::utils::mlpoly::MultilinearPolynomial;
use crate::whir::mlpcs::{MlCommitOptions, MlPCS};
use crate::whir::sumcheck::SumcheckInstanceProof;
use crate::whir::whir_helpers::{
    with_thread_local_evals_dft, MatricesSizeIndex, StackedBatchLayout,
};

use crate::whir::whir_types::CoefficientsByHeight;
pub use crate::whir::whir_types::{
    compute_commit_schedule, compute_commit_schedule_with_log_foldings, CommitGroup, DimAndNo,
    PrunedQueryOpenings, WhirConfig, WhirError, WhirInputProof, WhirPcs, WhirPcsProverData,
    WhirProof, WhirVerificationTrace, WhirVerifiedBatchStep, WhirVerifiedGroup, WhirVerifiedRound,
};

// =====================================================================
// WHIR algorithm
// =====================================================================
impl<F, InputMmcs, FriMmcs, EF, Challenger> MlPCS for WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField + 'static,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    type Field = F;
    type ExtensionField = EF;
    type ProverData = WhirPcsProverData<F, InputMmcs>;
    type Commitment = InputMmcs::Commitment;
    type BatchProof = WhirProof<EF, FriMmcs, F, WhirInputProof<F, InputMmcs>>;
    type Challenger = Challenger;
    type Error = WhirError<FriMmcs::Error, InputMmcs::Error>;
    type VerificationTrace =
        WhirVerificationTrace<EF, InputMmcs::VerificationTrace, FriMmcs::VerificationTrace>;

    #[tracing::instrument(skip_all, level = "debug", name = "WHIR::commit")]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.commit_with_options(evaluations, MlCommitOptions::default())
    }

    #[tracing::instrument(skip_all, level = "debug", name = "WHIR::commit_with_options")]
    fn commit_with_options(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
        options: MlCommitOptions,
    ) -> (Self::Commitment, Self::ProverData) {
        if let Some(stacking) = options.stacking {
            let stack_log_height = stacking.log_height.unwrap_or_else(|| {
                StackedBatchLayout::max_log_height_from_matrices(&evaluations)
                    .expect("invalid stacking input dimensions")
            });
            self.commit_stacked_impl(evaluations, stack_log_height, stacking.cache_stacked_matrix)
        } else {
            let (commitment, prover_data) = self.commit_impl(evaluations);
            (commitment, WhirPcsProverData::unstacked(prover_data))
        }
    }

    /// WHIR open: generates a batch opening proof using little-endian folding
    ///.
    ///
    /// At merge points, the EQ prefix matches due to little-endian folding,
    /// so we use a simple random linear combination instead of a full merge sumcheck round.
    #[tracing::instrument(skip_all, level = "debug", name = "WHIR::open")]
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<F>>>,
        prover_data_batch: Vec<Self::ProverData>,
        opening_point: &[EF],
        opened_values: &[Vec<Vec<EF>>],
        challenger: &mut Challenger,
    ) -> Result<Self::BatchProof, Self::Error> {
        self.validate_open_inputs(&polynomials_batch, &prover_data_batch, opened_values)?;

        if prover_data_batch
            .iter()
            .any(|data| data.stacked_log_height().is_some())
        {
            let mut stack_log_height = None;
            for data in &prover_data_batch {
                let current = data
                    .stacked_log_height()
                    .ok_or(WhirError::InvalidInputError)?;
                if stack_log_height.is_some_and(|expected| expected != current) {
                    return Err(WhirError::InvalidInputError);
                }
                stack_log_height = Some(current);
            }
            let stack_log_height = stack_log_height.ok_or(WhirError::InvalidInputError)?;
            return self.open_stacked(
                polynomials_batch,
                prover_data_batch,
                opening_point,
                opened_values,
                challenger,
                stack_log_height,
            );
        }

        let prover_data_batch = prover_data_batch
            .into_iter()
            .map(|data| {
                data.into_unstacked_mmcs()
                    .map_err(|_| WhirError::InvalidInputError)
            })
            .collect::<Result<Vec<_>, _>>()?;

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
        let mut claims_by_height: BTreeMap<usize, EF> = BTreeMap::new();

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

            // The group's opening claim is the coefficient-weighted sum of the
            // opened values — the exact field element the verifier computes —
            // so the full-hypercube ⟨F, eq⟩ dot products below are redundant
            // (kept as debug assertions).
            let claimed_sum: EF = values
                .iter()
                .zip(coefficients.iter())
                .flat_map(|(vals, cs)| vals.iter().zip(cs.iter()).map(|(v, c)| *v * *c))
                .sum();
            claims_by_height.insert(log_height, claimed_sum);

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
            self.find_pow_witness(challenger, self.config.fri.grinding_bits_batching)?;

        // --- Phase 2: WHIR sumcheck folding ---
        let merge_function = |x: &[EF]| x.iter().copied().product::<EF>();
        let highest_f = f_polys_by_height
            .remove(&max_log_height)
            .expect("at least one matrix is required");
        let eq_polynomial = EqPolynomial::new(opening_point.to_vec()).to_ml();
        let mut current_polys = vec![highest_f, eq_polynomial];

        let mut running_claim: EF = claims_by_height
            .remove(&max_log_height)
            .expect("top height group must exist");
        debug_assert_eq!(
            running_claim,
            compute_dotproduct(&current_polys[0].evals, &current_polys[1].evals),
            "opened-value claim must equal the ⟨F, eq⟩ dot product"
        );

        let mut branch_claims: BTreeMap<usize, EF> = claims_by_height;
        #[cfg(debug_assertions)]
        for (&log_height, f_poly) in f_polys_by_height.iter() {
            let branch_eq = EqPolynomial::new(opening_point[..log_height].to_vec()).to_ml();
            debug_assert_eq!(
                *branch_claims
                    .get(&log_height)
                    .expect("branch claim must exist"),
                compute_dotproduct(&f_poly.evals, &branch_eq.evals),
                "branch opened-value claim must equal its ⟨F, eq⟩ dot product"
            );
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
        let k = self.config.fri.log_final_poly_len.min(min_log_height);
        let commit_schedule = compute_commit_schedule(num_vars, k);
        let mut final_poly_evals: Vec<EF> = Vec::new();

        for group in commit_schedule.iter() {
            let round = group.start_round;
            let codeword = with_thread_local_evals_dft(|dft| {
                self.encode_to_codeword(&current_polys[0].evals, self.config.fri.log_blowup, dft)
            });
            let (root, tree) = self
                .config
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, 2));

            iopp_commitments.push(root.clone());
            iopp_prover_data.push(tree);
            challenger.observe(root);

            // Normal sumcheck round
            let (sc_proof, r_vec, _) = SumcheckInstanceProof::sumcheck_prove_normal_round(
                &running_claim,
                1,
                &mut current_polys,
                &merge_function,
                2,
                challenger,
            )
            .map_err(|_| WhirError::SumcheckPhaseError)?;
            running_claim = sc_proof.uni_polys[0].evaluate(&r_vec[0]);
            sumcheck_polys.push(sc_proof.uni_polys[0].clone());

            // Accumulate eq factor: eq(p[round-1]; r_fold)
            // round here counts down: at round=n, we bind x[n-1] to r_fold,
            // so the eq factor is eq(p[n-1]; r_fold) = eq(p[round-1]; r_fold)
            let r_fold = r_vec[0];
            let p_i = opening_point[round - 1];
            eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);

            // WHIR merge: simplified merge using EQ prefix matching
            if let Some(branch_f) = f_polys_by_height.remove(&(round - 1)) {
                debug_assert_eq!(branch_f.len(), current_polys[0].len());

                let branch_claim = branch_claims
                    .remove(&(round - 1))
                    .expect("branch claim must exist for this height group");

                // Sample merge coefficient
                let merge_beta: EF = challenger.sample_ext_element();

                // F_new = eq_factor * F + merge_beta * G
                // EQ_new = eq(p[0..round-1]; cube) = branch_eq
                let branch_eq = EqPolynomial::new(opening_point[..(round - 1)].to_vec()).to_ml();

                current_polys[0]
                    .evals
                    .par_iter_mut()
                    .zip(branch_f.evals.par_iter())
                    .for_each(|(f_val, g_val)| {
                        *f_val = eq_factor * *f_val + merge_beta * *g_val;
                    });

                current_polys[1] = branch_eq;

                running_claim += merge_beta * branch_claim;

                eq_factor = EF::one();
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
                .map_err(|_| WhirError::SumcheckPhaseError)?;
                running_claim = sc_proof.uni_polys[0].evaluate(&r_vec[0]);
                sumcheck_polys.push(sc_proof.uni_polys[0].clone());

                let r_fold = r_vec[0];
                let p_i = opening_point[round - 1];
                eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);
            }
        } else {
            let codeword = with_thread_local_evals_dft(|dft| {
                self.encode_to_codeword(&current_polys[0].evals, self.config.fri.log_blowup, dft)
            });
            let (root, tree) = self
                .config
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, 2));
            iopp_commitments.push(root.clone());
            iopp_prover_data.push(tree);
            challenger.observe(root);
        }

        // --- Phase 3: Query proof of work ---
        let grinding_query_data =
            self.find_pow_witness(challenger, self.config.fri.grinding_bits_query)?;

        // --- Phase 4: IOPP query generation ---
        let query_points: Vec<usize> = (0..self.config.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.config.fri.log_blowup))
            .collect();

        // Path-pruning uses one batched opening per committed round instead
        // of duplicating Merkle authentication paths for every query.
        let use_path_pruning = self.config.path_pruning;

        let query_openings_bundle: WhirInputProof<F, InputMmcs> = if use_path_pruning {
            let num_batches = prover_data_batch.len();
            let mut round_pruned = Vec::with_capacity(num_batches);
            let mut round_opened_values: Vec<Vec<Vec<Vec<F>>>> = Vec::with_capacity(num_batches);
            let mut q2u: Vec<Vec<u32>> = Vec::with_capacity(num_batches);

            for (batch_idx, prover_data) in prover_data_batch.iter().enumerate() {
                let shift = max_log_height - max_log_height_per_batch[batch_idx];
                let shifted_per_query: Vec<usize> =
                    query_points.iter().map(|&p| p >> shift).collect();

                let (uniq_opened, pruned_proof) = self
                    .mmcs
                    .open_batch_pruned(&shifted_per_query, prover_data.as_ref());

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

            WhirInputProof {
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
                            let (values, proof) =
                                self.mmcs.open_batch(shifted_point, prover_data.as_ref());
                            BatchOpening {
                                opened_values: values,
                                opening_proof: proof,
                            }
                        })
                        .collect()
                })
                .collect();
            WhirInputProof::from_per_query(qo)
        };

        let (iopp_queries, iopp_pruned) = if use_path_pruning {
            let pruned = answer_queries_pruned(&self.config.fri, &iopp_prover_data, &query_points);
            (Vec::new(), Some(pruned))
        } else {
            let queries = query_points
                .iter()
                .map(|&point| answer_query(&self.config.fri, &iopp_prover_data, point))
                .collect::<Vec<_>>();
            (queries, None)
        };

        Ok(WhirProof {
            stack_log_height: None,
            sumcheck_transcript: SumcheckInstanceProof {
                uni_polys: sumcheck_polys,
            },
            iopp_oracles: iopp_commitments,
            ood_values: Vec::new(),
            iopp_queries,
            round_iopp: None,
            query_openings: query_openings_bundle,
            grinding_batching_witness: grinding_batching_data,
            grinding_query_witness: grinding_query_data,
            final_poly: final_poly_evals,
            iopp_pruned,
            stacking_reduction: None,
        })
    }

    /// Verify the batch opening proof (whir variant).
    ///
    /// No WHIR out-of-domain sampling. Merges are deterministic (no merge sumcheck polynomials).
    #[tracing::instrument(skip_all, level = "debug", name = "WHIR::verify")]
    fn verify(
        &self,
        commitment_batch: Vec<Self::Commitment>,
        matrices_size_batch: &[Vec<Dimensions>],
        opening_point: &[Self::ExtensionField],
        opened_values_batch: &[Vec<Vec<Self::ExtensionField>>],
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::VerificationTrace, Self::Error> {
        self.validate_verify_inputs(&commitment_batch, matrices_size_batch, opened_values_batch)?;

        if let Some(stack_log_height) = proof.stack_log_height {
            return self
                .verify_stacked(
                    commitment_batch,
                    matrices_size_batch,
                    opening_point,
                    opened_values_batch,
                    proof,
                    challenger,
                    stack_log_height,
                )
                .map(|_| WhirVerificationTrace {
                    stacked: true,
                    ..Default::default()
                });
        }

        let mut verification_trace = WhirVerificationTrace::default();

        let WhirProof {
            stack_log_height: _,
            sumcheck_transcript,
            iopp_oracles,
            ood_values: _,
            iopp_queries,
            round_iopp,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned,
            stacking_reduction: _,
        } = proof;
        // [D6] Split bundle into per-query openings and optional pruned variant.
        let WhirInputProof {
            per_query: query_openings,
            pruned: query_openings_pruned,
        } = query_openings;

        if round_iopp.is_some() {
            return Err(WhirError::InvalidInputError);
        }

        // [D6] Strong binding rule: pruned PCS openings are only valid together
        // with pruned IOPP queries. Any other combination is rejected to avoid
        // verifying a partially-pruned proof.
        match (iopp_pruned.is_some(), query_openings_pruned.is_some()) {
            (true, true) | (false, false) => {}
            _ => return Err(WhirError::InvalidInputError),
        }

        if grinding_batching_witness.len() != 2 || grinding_query_witness.len() != 2 {
            return Err(WhirError::InvalidInputError);
        }

        let num_vars = opening_point.len();

        // --- Flatten and group by log_height ---
        let flat_dims: Vec<DimAndNo> = matrices_size_batch
            .iter()
            .flat_map(|batch| batch.iter())
            .enumerate()
            .map(|(idx, dim)| DimAndNo {
                dim: *dim,
                num: idx,
            })
            .collect();
        let flat_opened_values: Vec<&Vec<EF>> =
            opened_values_batch.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height =
            Self::group_dims_by_log_height(&flat_dims, &flat_opened_values);
        let log_max_height = matrices_by_log_height.keys().max().cloned().unwrap_or(0);
        // [F-022] The tallest committed matrix must match the opening point
        // dimension. A taller height from an untrusted proof would make the
        // per-batch `index >> (num_vars - log_height)` shift underflow.
        if log_max_height != num_vars {
            return Err(WhirError::InvalidInputError);
        }

        let size_index = MatricesSizeIndex::new(matrices_size_batch);

        // --- Phase 1: Compute claimed sums and query coefficients per height group ---
        let alpha: EF = challenger.sample_ext_element();
        verification_trace.alpha = Some(alpha);
        let mut alpha_powers = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut claims_by_height: BTreeMap<usize, EF> = BTreeMap::new();
        let mut coefficients_by_height: CoefficientsByHeight<EF> = BTreeMap::new();
        let mut batch_accumulator = EF::zero();

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (dims, values): (Vec<&DimAndNo>, Vec<&Vec<EF>>) =
                group.iter().map(|(d, v)| (*d, *v)).unzip();

            let first_step = verification_trace.batch_steps.len();
            let group_accumulator = batch_accumulator;
            let group_value_count = values.iter().map(|vals| vals.len()).sum::<usize>();
            let mut group_offset = 0usize;
            let mut coeffs = Vec::with_capacity(values.len());
            for (matrix_in_group, vals) in values.iter().enumerate() {
                let (batch_idx, matrix_idx) = size_index.find_position(dims[matrix_in_group].num);
                let mut matrix_coeffs = Vec::with_capacity(vals.len());
                for (value_idx, value) in vals.iter().copied().enumerate() {
                    let coefficient = alpha_powers.next().unwrap();
                    let accumulator_in = batch_accumulator;
                    batch_accumulator += coefficient * value;
                    group_offset += 1;
                    let is_group_end = group_offset == group_value_count;
                    verification_trace.batch_steps.push(WhirVerifiedBatchStep {
                        log_height,
                        batch_idx,
                        matrix_idx,
                        value_idx,
                        value,
                        coefficient,
                        coefficient_out: coefficient * alpha,
                        accumulator_in,
                        accumulator_out: batch_accumulator,
                        group_accumulator_in: group_accumulator,
                        group_accumulator_out: if is_group_end {
                            batch_accumulator
                        } else {
                            group_accumulator
                        },
                        is_group_start: group_offset == 1,
                        is_group_end,
                    });
                    matrix_coeffs.push(coefficient);
                }
                coeffs.push(matrix_coeffs);
            }

            let claimed_sum = batch_accumulator - group_accumulator;
            claims_by_height.insert(log_height, claimed_sum);
            verification_trace.groups.push(WhirVerifiedGroup {
                log_height,
                claim: claimed_sum,
                first_step,
                step_count: group_value_count,
            });

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
            self.config.fri.grinding_bits_batching,
            grinding_batching_witness[1],
        ) {
            return Err(WhirError::InvalidPowWitness);
        }

        // --- Phase 2: Verify sumcheck rounds (fold + simplified merge) ---
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut current_claim = claims_by_height
            .remove(&num_vars)
            .ok_or(WhirError::SumcheckPhaseError)?;

        let min_log_height = matrices_by_log_height
            .keys()
            .min()
            .cloned()
            .unwrap_or(num_vars);
        let k = self.config.fri.log_final_poly_len.min(min_log_height);
        if k > 0 && final_poly.len() != (1usize << k) {
            return Err(WhirError::InvalidInputError);
        }
        let commit_schedule = compute_commit_schedule(num_vars, k);
        let commit_start_rounds: std::collections::BTreeSet<usize> =
            commit_schedule.iter().map(|g| g.start_round).collect();

        // [F-019/F-020] Pin the exact IOPP oracle count before observing them.
        // Without this, a malicious proof can append extra tail oracles: with
        // k>0 that moves the verification past the final-codeword early-stop
        // check (degree bound bypass), and with k==0 an oracle inserted before
        // the real final commitment is still observed into the query-PoW /
        // sampling transcript (biasing Fiat-Shamir) while `iopp_oracles.last()`
        // keeps pointing at the honest final commitment. The committed-round
        // count is otherwise taken from prover-controlled
        // `commit_phase_openings.len()`, so it must be fixed here.
        let expected_iopp_oracles = commit_schedule.len() + usize::from(k == 0);
        if iopp_oracles.len() != expected_iopp_oracles {
            return Err(WhirError::InvalidInputError);
        }

        if !iopp_oracles.is_empty() {
            challenger.observe(iopp_oracles[0].clone());
        }

        let mut folding_challenges: Vec<EF> = Vec::with_capacity(num_vars);
        let mut merge_betas: Vec<EF> = Vec::new();
        let mut eq_factor = EF::one();
        let mut oracle_idx: usize = 1;

        for round in (0..=num_vars).rev() {
            let should_observe_next_oracle = (round < num_vars
                && commit_start_rounds.contains(&round))
                || (round == 0 && k == 0);
            if should_observe_next_oracle {
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
            let uni_poly = poly_iter.next().ok_or(WhirError::SumcheckPhaseError)?;
            let trace_round = folding_challenges.len();
            let claim_in = current_claim;
            let eq_in = eq_factor;
            if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != current_claim {
                return Err(WhirError::SumcheckPhaseError);
            }
            uni_poly
                .coeffs
                .iter()
                .for_each(|c| challenger.observe_ext_element(*c));
            let r_fold: EF = challenger.sample_ext_element();
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate(&r_fold);
            let claim_folded = current_claim;
            let claim_acc = uni_poly.coeffs[1] + r_fold * uni_poly.coeffs[2];

            // Accumulate eq factor
            let p_i = opening_point[round - 1];
            let round_eq_factor = p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);
            eq_factor *= round_eq_factor;
            let eq_folded = eq_factor;

            // WHIR merge (deterministic, no merge poly)
            let mut merge_height = None;
            let mut captured_merge_beta = None;
            let mut captured_branch_claim = None;
            if let Some(branch_claim) = claims_by_height.remove(&(round - 1)) {
                let merge_beta: EF = challenger.sample_ext_element();
                merge_betas.push(merge_beta);

                current_claim += merge_beta * branch_claim;

                eq_factor = EF::one();
                merge_height = Some(round - 1);
                captured_merge_beta = Some(merge_beta);
                captured_branch_claim = Some(branch_claim);
            }
            verification_trace.rounds.push(WhirVerifiedRound {
                round: trace_round,
                claim_in,
                coefficients: uni_poly.coeffs.clone(),
                r_fold,
                claim_acc,
                claim_folded,
                eq_in,
                eq_factor: round_eq_factor,
                eq_folded,
                merge_height,
                merge_beta: captured_merge_beta,
                branch_claim: captured_branch_claim,
                claim_out: current_claim,
                eq_out: eq_factor,
            });
        }

        // --- Phase 3: Reconstruct combined EQ sum ---
        // With whir little-endian folding, the final EQ is determined by the
        // folding challenges after the last merge (or all challenges if no merges).
        let fc_rev: Vec<EF> = folding_challenges.iter().rev().cloned().collect();

        // The final EQ only involves factors from after the last merge.
        // eq(p[0..min_height]; reversed tail of folding challenges)
        let combined_eq_sum = EqPolynomial::new(opening_point[..min_log_height].to_vec())
            .evaluate(&fc_rev[..min_log_height]);

        // --- Phase 4: Final codeword / polynomial check ---
        let combined_f_r: EF = current_claim / combined_eq_sum;
        verification_trace.combined_eq_sum = Some(combined_eq_sum);
        verification_trace.combined_f_r = Some(combined_f_r);

        if k == 0 {
            let expected_codeword = vec![combined_f_r; 1 << self.config.fri.log_blowup];
            let (expected_commitment, _) = self
                .config
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(expected_codeword, 2));

            let last_oracle = iopp_oracles
                .last()
                .ok_or(WhirError::CommitmentCheckFailed)?;
            let last_bytes =
                bincode::serialize(last_oracle).map_err(|_| WhirError::CommitmentCheckFailed)?;
            let expected_bytes = bincode::serialize(&expected_commitment)
                .map_err(|_| WhirError::CommitmentCheckFailed)?;
            if last_bytes != expected_bytes {
                return Err(WhirError::CommitmentCheckFailed);
            }
        }

        let final_codeword: Option<Vec<EF>> = if k > 0 {
            Some(with_thread_local_evals_dft(|dft| {
                self.encode_to_codeword(final_poly, self.config.fri.log_blowup, dft)
            }))
        } else {
            None
        };

        // --- Phase 5: Query proof of work ---
        challenger.observe(grinding_query_witness[0]);
        if !challenger.check_witness(
            self.config.fri.grinding_bits_query,
            grinding_query_witness[1],
        ) {
            return Err(WhirError::InvalidPowWitness);
        }

        // --- Phase 6: IOPP query verification ---
        let query_points: Vec<usize> = (0..self.config.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.config.fri.log_blowup))
            .collect();

        // [F-018] Pin the query counts before iterating. The standard path
        // zips `iopp_queries` with `query_openings`; without this guard a
        // malicious proof could supply short vectors and `zip` would silently
        // truncate (an empty zip makes `.all(..)` vacuously true), bypassing
        // IOPP soundness amplification. Mirror the FRI verifier's check.
        if iopp_pruned.is_none()
            && (iopp_queries.len() != self.config.fri.num_queries
                || query_openings.len() != self.config.fri.num_queries)
        {
            return Err(WhirError::InvalidInputError);
        }

        // [B6-5-step3] env-gated dispatch: pruned path uses single batched
        // `verify_queries_iopp_p3_pruned_whir` (saves 17%+ proof bytes),
        // standard path keeps per-query loop.
        let all_queries_valid = if let Some(std_pruned) = iopp_pruned.as_ref() {
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
                .ok_or(WhirError::InvalidInputError)?;
            let num_rounds = matrices_size_batch.len();
            let n_queries = query_points.len();

            if qop.round_pruned.len() != num_rounds
                || qop.round_opened_values.len() != num_rounds
                || qop.query_to_unique_slot.len() != num_rounds
            {
                return Err(WhirError::InvalidInputError);
            }

            // Step A1: per-round pruned merkle verify.
            let mut per_round_ok = true;
            for (round_idx, batch_dims) in matrices_size_batch.iter().enumerate() {
                let codeword_dims: Vec<Dimensions> = batch_dims
                    .iter()
                    .map(|dim| Dimensions {
                        width: 0,
                        height: dim.height << self.config.fri.log_blowup,
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
                // [F-017] Bind the proof's embedded pruned indices to the
                // transcript-sampled (per-batch right-shifted, sorted+deduped)
                // query indices by value.
                let batch_max_log_height = batch_dims
                    .iter()
                    .map(|dim| log2_strict_usize(dim.height))
                    .max()
                    .unwrap_or(0);
                let shift = log_max_height - batch_max_log_height;
                let mut sorted_unique: Vec<usize> =
                    query_points.iter().map(|&p| p >> shift).collect();
                sorted_unique.sort_unstable();
                sorted_unique.dedup();
                if let Some(recovered) = self
                    .mmcs
                    .recover_pruned_indices(&qop.round_pruned[round_idx])
                {
                    if recovered.len() != sorted_unique.len()
                        || recovered
                            .iter()
                            .zip(sorted_unique.iter())
                            .any(|(&got, &want)| got as usize != want)
                    {
                        per_round_ok = false;
                        break;
                    }
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
                            (log_height + self.config.fri.log_blowup, sum)
                        })
                        .collect();
                    leaf_sums_per_query.push(sums);
                }

                // Step B: single batched IOPP verify across all N queries.
                self.verify_queries_iopp_p3_pruned_whir(
                    iopp_oracles.as_slice(),
                    &query_points,
                    &leaf_sums_per_query,
                    std_pruned,
                    &[],
                    &folding_challenges,
                    &merge_betas,
                    opening_point,
                    &combined_f_r,
                    final_codeword.as_deref(),
                )
                .is_ok()
            }
        } else {
            let query_results = iopp_queries
                .par_iter()
                .zip(query_openings.par_iter())
                .enumerate()
                .map(|(i, (query, leaf_opening))| {
                    self.verify_query_p3_batch_whir(
                        i,
                        &commitment_batch,
                        iopp_oracles.as_slice(),
                        query_points[i],
                        matrices_size_batch,
                        query,
                        leaf_opening,
                        &coefficients_by_height,
                        alpha,
                        &folding_challenges,
                        &merge_betas,
                        opening_point,
                        &combined_f_r,
                        final_codeword.as_deref(),
                    )
                    .ok()
                })
                .collect::<Vec<_>>();
            if query_results.iter().any(Option::is_none) {
                false
            } else {
                verification_trace.queries =
                    query_results.into_iter().map(Option::unwrap).collect();
                true
            }
        };

        if !all_queries_valid {
            return Err(WhirError::FriFinalStepMisMatch);
        }

        Ok(verification_trace)
    }
}

#[cfg(test)]
#[path = "whir_pcs_tests.rs"]
mod tests;
