use std::collections::BTreeMap;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::dft_eval::EvalsDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_fri::prover::{
    answer_queries_pruned, answer_queries_pruned_with_log_foldings, answer_query_with_log_foldings,
};
use p3_fri::BatchOpening;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::reverse_bits_len;
use web_time::Instant;

use crate::utils::eqpoly::EqPolynomial;
use crate::utils::math::compute_dotproduct_mix;
use crate::utils::mlpoly::MultilinearPolynomial;
use crate::utils::unipoly::UniPoly;
use crate::whir::profile;
use crate::whir::sumcheck::{PairProductLeftInput, SumcheckInstanceProof};
use crate::whir::whir_helpers::{
    with_thread_local_evals_dft, StackedBatchCoefficients, StackedBatchLayout,
};
use crate::whir::whir_types::{
    PrunedQueryOpenings, SharedMmcsProverData, StackingReductionProof, WhirError, WhirInputProof,
    WhirIoppRound, WhirPcs, WhirPcsProverData, WhirProof, WhirPrunedIoppRound,
    WhirRoundPrunedQueryProof, WhirRoundQueryConfig, WhirRoundQueryProof, WhirRoundSchedule,
};

type WhirStackedResult<F, InputMmcs, FriMmcs, EF, T> =
    Result<T, WhirError<<FriMmcs as Mmcs<EF>>::Error, <InputMmcs as Mmcs<F>>::Error>>;
type SumcheckRounds<EF> = (Vec<UniPoly<EF>>, Vec<EF>);
type PrunedIoppRows<EF, FriMmcs> = (Vec<Vec<EF>>, WhirPrunedIoppRound<EF, FriMmcs>);
type StackedWhirProof<F, InputMmcs, FriMmcs, EF> =
    WhirProof<EF, FriMmcs, F, WhirInputProof<F, InputMmcs>>;
type StackedProofResult<F, InputMmcs, FriMmcs, EF> =
    WhirStackedResult<F, InputMmcs, FriMmcs, EF, StackedWhirProof<F, InputMmcs, FriMmcs, EF>>;
type PrunedIoppRowsResult<F, InputMmcs, FriMmcs, EF> =
    WhirStackedResult<F, InputMmcs, FriMmcs, EF, PrunedIoppRows<EF, FriMmcs>>;

struct StackedWhirRoundOutput<F: Field, InputMmcs: Mmcs<F>, EF: Field, FriMmcs: Mmcs<EF>> {
    iopp_round: Option<WhirIoppRound<EF, FriMmcs>>,
    pruned_iopp_round: Option<WhirPrunedIoppRound<EF, FriMmcs>>,
    query_witness: Vec<F>,
    folding_witness: Vec<F>,
    first_round_input_openings: Option<WhirInputProof<F, InputMmcs>>,
    sumcheck_polys: Vec<UniPoly<EF>>,
}

struct StackedOpeningPreparation<F: Field, InputMmcs: Mmcs<F>, EF: Field> {
    combined_evals: Vec<EF>,
    running_claim: EF,
    opening_point: Vec<EF>,
    stacked_data: Vec<SharedMmcsProverData<F, InputMmcs>>,
    reduction_proof: Option<StackingReductionProof<EF>>,
}

struct StackedWhirProverState<'a, F: Field, InputMmcs: Mmcs<F>, EF: Field, FriMmcs: Mmcs<EF>> {
    stack_log_height: usize,
    stacked_data: &'a [SharedMmcsProverData<F, InputMmcs>],
    dft: &'a EvalsDft<F>,
    current_polys: &'a mut Vec<MultilinearPolynomial<EF>>,
    running_claim: &'a mut EF,
    iopp_commitments: &'a mut Vec<FriMmcs::Commitment>,
    iopp_prover_data: &'a mut Vec<FriMmcs::ProverData<RowMajorMatrix<EF>>>,
    ood_values: &'a mut Vec<EF>,
    final_poly_evals: &'a mut Vec<EF>,
    consumed_rounds: &'a mut usize,
}

struct SymbolicWeightTerm<EF> {
    coeff: EF,
    point: Vec<EF>,
}

fn base_matrix_into_ext<F: Field, EF: ExtensionField<F>>(
    matrix: RowMajorMatrix<F>,
) -> RowMajorMatrix<EF> {
    let width = matrix.width();
    let values = matrix.values.into_iter().map(EF::from_base).collect();
    RowMajorMatrix::new(values, width)
}

// Rows are independent and each `out[row]` keeps the same sequential per-column
// accumulation order, so parallelizing over rows is proof-byte-identical.
fn accumulate_base_matrix_columns<F: Field, EF: ExtensionField<F>>(
    matrix: &RowMajorMatrix<F>,
    coeffs: &[EF],
    out: &mut [EF],
) {
    debug_assert_eq!(matrix.width(), coeffs.len());
    debug_assert_eq!(matrix.height(), out.len());
    let width = matrix.width();
    if width == 0 {
        return;
    }
    out.par_iter_mut()
        .zip(matrix.values.par_chunks_exact(width))
        .for_each(|(out_value, matrix_row)| {
            for col in 0..width {
                *out_value += coeffs[col] * matrix_row[col];
            }
        });
}

fn accumulate_ext_matrix_columns<EF: Field>(
    matrix: &RowMajorMatrix<EF>,
    coeffs: &[EF],
    out: &mut [EF],
) {
    debug_assert_eq!(matrix.width(), coeffs.len());
    debug_assert_eq!(matrix.height(), out.len());
    let width = matrix.width();
    if width == 0 {
        return;
    }
    out.par_iter_mut()
        .zip(matrix.values.par_chunks_exact(width))
        .for_each(|(out_value, matrix_row)| {
            for col in 0..width {
                *out_value += coeffs[col] * matrix_row[col];
            }
        });
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField + 'static,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    fn pow2_ext_point(z: EF, len: usize) -> Vec<EF> {
        let mut powers = Vec::with_capacity(len);
        let mut cur = z;
        for _ in 0..len {
            powers.push(cur);
            cur *= cur;
        }
        powers
    }

    fn accumulate_eq_evals(target: &mut [EF], point: &[EF], scale: EF) {
        let eq_evals = EqPolynomial::new(point.to_vec()).evals();
        debug_assert_eq!(target.len(), eq_evals.len());
        target
            .par_iter_mut()
            .zip(eq_evals.par_iter())
            .for_each(|(acc, eq)| *acc += scale * *eq);
    }

    fn eq_eval(left: EF, right: EF) -> EF {
        left * right + (EF::one() - left) * (EF::one() - right)
    }

    fn fold_symbolic_weight_terms(
        terms: &mut [SymbolicWeightTerm<EF>],
        alpha: EF,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, ()> {
        for term in terms {
            let point = term.point.pop().ok_or(WhirError::InvalidInputError)?;
            term.coeff *= Self::eq_eval(point, alpha);
        }
        Ok(())
    }

    fn evaluate_mle_evals_at_point(
        evals: &[EF],
        point: &[EF],
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, EF> {
        if evals.len() != (1usize << point.len()) {
            return Err(WhirError::InvalidInputError);
        }
        if evals.is_empty() {
            return Err(WhirError::InvalidInputError);
        }

        let mut folded = evals.to_vec();
        for &alpha in point.iter().rev() {
            let half = folded.len() / 2;
            for idx in 0..half {
                let even = folded[2 * idx];
                let odd = folded[2 * idx + 1];
                folded[idx] = even + alpha * (odd - even);
            }
            folded.truncate(half);
        }
        folded.first().copied().ok_or(WhirError::InvalidInputError)
    }

    fn symbolic_final_accumulator(
        final_poly: &[EF],
        terms: &[SymbolicWeightTerm<EF>],
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, EF> {
        let mut acc = EF::zero();
        for term in terms {
            acc += term.coeff * Self::evaluate_mle_evals_at_point(final_poly, &term.point)?;
        }
        Ok(acc)
    }

    fn codeword_query_point(
        row_index: usize,
        codeword_log_height: usize,
        mle_vars: usize,
        log_blowup: usize,
    ) -> Vec<EF> {
        if mle_vars == 0 {
            return Vec::new();
        }
        // Match `encode_to_codeword`: skipped DFT keeps bit-reversed output
        // for nonzero blowup, and the MLE table folds bottom variables first.
        let exponent = if log_blowup == 0 {
            row_index
        } else {
            reverse_bits_len(row_index, codeword_log_height)
        };
        let z = EF::two_adic_generator(codeword_log_height).exp_u64(exponent as u64);
        let mut point = Self::pow2_ext_point(z, mle_vars);
        point.reverse();
        point
    }

    fn commit_iopp_codeword(
        &self,
        evals: &[EF],
        log_folding: usize,
        log_blowup: usize,
        dft: &EvalsDft<F>,
        challenger: &mut Challenger,
    ) -> (FriMmcs::Commitment, FriMmcs::ProverData<RowMajorMatrix<EF>>) {
        let codeword = profile::time("open.whir_iopp_encode_dft_ms", || {
            self.encode_to_codeword(evals, log_blowup, dft)
        });
        let (root, tree) = profile::time("open.whir_iopp_leaf_hash_and_tree_ms", || {
            self.config
                .fri
                .mmcs
                .commit_matrix(RowMajorMatrix::new(codeword, 1usize << log_folding))
        });
        challenger.observe(root.clone());
        (root, tree)
    }

    fn prove_sumcheck_rounds(
        running_claim: &mut EF,
        num_rounds: usize,
        current_polys: &mut Vec<MultilinearPolynomial<EF>>,
        challenger: &mut Challenger,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, SumcheckRounds<EF>> {
        let (sc_proof, challenges, _) = profile::time("open.whir_sumcheck_rounds_ms", || {
            SumcheckInstanceProof::sumcheck_prove_interleaved_pair_products(
                running_claim,
                num_rounds,
                current_polys,
                challenger,
            )
        })
        .map_err(|_| WhirError::SumcheckPhaseError)?;
        let final_challenge = challenges.last().ok_or(WhirError::InvalidInputError)?;
        let final_poly = sc_proof
            .uni_polys
            .last()
            .ok_or(WhirError::InvalidInputError)?;
        *running_claim = final_poly.evaluate(final_challenge);
        Ok((sc_proof.uni_polys, challenges))
    }

    fn prove_sumcheck_rounds_with_folding_pow(
        &self,
        running_claim: &mut EF,
        num_rounds: usize,
        current_polys: &mut Vec<MultilinearPolynomial<EF>>,
        challenger: &mut Challenger,
        grinding_bits_folding: usize,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, (SumcheckRounds<EF>, Vec<F>)> {
        let mut uni_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut folding_witness = Vec::with_capacity(2 * num_rounds);

        for _ in 0..num_rounds {
            let (round_polys, round_challenges) =
                Self::prove_sumcheck_rounds(running_claim, 1, current_polys, challenger)?;
            let uni_poly = round_polys
                .into_iter()
                .next()
                .ok_or(WhirError::SumcheckPhaseError)?;
            let challenge = round_challenges
                .into_iter()
                .next()
                .ok_or(WhirError::SumcheckPhaseError)?;
            uni_polys.push(uni_poly);
            challenges.push(challenge);

            if grinding_bits_folding > 0 {
                folding_witness.extend(self.find_pow_witness(challenger, grinding_bits_folding)?);
            }
        }

        Ok(((uni_polys, challenges), folding_witness))
    }

    fn open_stacked_input_batches(
        &self,
        query_points: &[usize],
        stacked_data: &[SharedMmcsProverData<F, InputMmcs>],
    ) -> Vec<Vec<BatchOpening<F, InputMmcs>>> {
        profile::time("open.whir_input_mmcs_open_ms", || {
            query_points
                .par_iter()
                .map(|&point| {
                    stacked_data
                        .iter()
                        .map(|mmcs_prover_data| {
                            let (values, proof) =
                                self.mmcs.open_batch(point, mmcs_prover_data.as_ref());
                            BatchOpening {
                                opened_values: values,
                                opening_proof: proof,
                            }
                        })
                        .collect()
                })
                .collect()
        })
    }

    fn sorted_unique_slots(indices: &[usize]) -> (Vec<usize>, Vec<u32>) {
        let mut sorted_unique = indices.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        let query_to_unique_slot = indices
            .iter()
            .map(|index| {
                sorted_unique
                    .binary_search(index)
                    .expect("index must be present") as u32
            })
            .collect();
        (sorted_unique, query_to_unique_slot)
    }

    fn open_stacked_input_batches_pruned(
        &self,
        query_points: &[usize],
        stacked_data: &[SharedMmcsProverData<F, InputMmcs>],
    ) -> WhirInputProof<F, InputMmcs> {
        let (_, query_to_unique_slot) = Self::sorted_unique_slots(query_points);
        let round_results = profile::time("open.whir_input_mmcs_open_ms", || {
            stacked_data
                .par_iter()
                .map(|mmcs_prover_data| {
                    let (opened_values, pruned_proof) = self
                        .mmcs
                        .open_batch_pruned(query_points, mmcs_prover_data.as_ref());
                    (pruned_proof, opened_values)
                })
                .collect::<Vec<_>>()
        });

        let mut round_pruned = Vec::with_capacity(round_results.len());
        let mut round_opened_values = Vec::with_capacity(round_results.len());
        for (pruned_proof, opened_values) in round_results {
            round_pruned.push(pruned_proof);
            round_opened_values.push(opened_values);
        }

        WhirInputProof {
            per_query: Vec::new(),
            pruned: Some(PrunedQueryOpenings {
                round_pruned,
                round_opened_values,
                query_to_unique_slot: vec![query_to_unique_slot; stacked_data.len()],
            }),
        }
    }

    pub fn restore_elided_first_stacked_input_batch_pruned(
        &self,
        proof: &mut WhirProof<EF, FriMmcs, F, WhirInputProof<F, InputMmcs>>,
        first_batch_data: &WhirPcsProverData<F, InputMmcs>,
        expected_batches: usize,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, bool> {
        if expected_batches == 0 {
            return Err(WhirError::InvalidInputError);
        }
        if proof.stack_log_height.is_none() || proof.round_iopp.is_none() {
            return Ok(false);
        }
        let pruned = match proof.query_openings.pruned.as_mut() {
            Some(pruned) => pruned,
            None => return Ok(false),
        };
        let current_batches = pruned.round_pruned.len();
        if current_batches == expected_batches {
            return Ok(false);
        }
        if current_batches + 1 != expected_batches
            || current_batches == 0
            || pruned.round_opened_values.len() != current_batches
            || pruned.query_to_unique_slot.len() != current_batches
        {
            return Err(WhirError::InvalidInputError);
        }

        let sorted_unique = self
            .mmcs
            .recover_pruned_indices(&pruned.round_pruned[0])
            .ok_or(WhirError::InvalidInputError)?
            .into_iter()
            .map(|idx| idx as usize)
            .collect::<Vec<_>>();
        let retained_q2u = pruned.query_to_unique_slot[0].clone();
        let mut query_points = Vec::with_capacity(retained_q2u.len());
        for &slot in &retained_q2u {
            let slot = slot as usize;
            if slot >= sorted_unique.len() {
                return Err(WhirError::InvalidInputError);
            }
            query_points.push(sorted_unique[slot]);
        }
        if pruned
            .query_to_unique_slot
            .iter()
            .any(|q2u| q2u != &retained_q2u)
        {
            return Err(WhirError::InvalidInputError);
        }

        let (first_batch_mmcs_data, _stacked) = first_batch_data
            .clone()
            .into_stacked()
            .map_err(|_| WhirError::InvalidInputError)?;
        let (opened_values, pruned_proof) = self
            .mmcs
            .open_batch_pruned(&query_points, first_batch_mmcs_data.as_ref());
        let (check_sorted, check_q2u) = Self::sorted_unique_slots(&query_points);
        if check_sorted != sorted_unique || check_q2u != retained_q2u {
            return Err(WhirError::InvalidInputError);
        }

        pruned.round_pruned.insert(0, pruned_proof);
        pruned.round_opened_values.insert(0, opened_values);
        pruned.query_to_unique_slot.insert(0, check_q2u);
        Ok(true)
    }

    fn open_iopp_rows_pruned(
        &self,
        query_points: &[usize],
        commit: &FriMmcs::ProverData<RowMajorMatrix<EF>>,
        log_folding: usize,
    ) -> PrunedIoppRowsResult<F, InputMmcs, FriMmcs, EF> {
        let row_width = 1usize << log_folding;
        let row_indices = query_points
            .iter()
            .map(|point| point >> log_folding)
            .collect::<Vec<_>>();
        let (_, query_to_unique_slot) = Self::sorted_unique_slots(&row_indices);
        let (opened_rows, pruned_proof) = profile::time("open.whir_iopp_mmcs_open_ms", || {
            self.config.fri.mmcs.open_batch_pruned(&row_indices, commit)
        });

        let rows_by_query = query_to_unique_slot
            .iter()
            .map(|&slot| {
                let row = opened_rows
                    .get(slot as usize)
                    .ok_or(WhirError::InvalidInputError)?;
                if row.len() != 1 || row[0].len() != row_width {
                    return Err(WhirError::InvalidInputError);
                }
                Ok(row[0].clone())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((
            rows_by_query,
            WhirPrunedIoppRound {
                pruned_proof,
                opened_rows,
                query_to_unique_slot,
            },
        ))
    }

    fn verify_stacked_input_pruned(
        &self,
        commitment_batch: &[InputMmcs::Commitment],
        stacked_dims_by_batch: &[Vec<Dimensions>],
        coeffs_by_batch: &[StackedBatchCoefficients<EF>],
        query_points: &[usize],
        pruned: &PrunedQueryOpenings<F, InputMmcs>,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, Vec<EF>> {
        let num_batches = coeffs_by_batch.len();
        if pruned.round_pruned.len() != num_batches
            || pruned.round_opened_values.len() != num_batches
            || pruned.query_to_unique_slot.len() != num_batches
            || commitment_batch.len() != num_batches
            || stacked_dims_by_batch.len() != num_batches
        {
            return Err(WhirError::InvalidInputError);
        }

        for batch_idx in 0..num_batches {
            let opened_values = &pruned.round_opened_values[batch_idx];
            let q2u = &pruned.query_to_unique_slot[batch_idx];
            if q2u.len() != query_points.len()
                || q2u
                    .iter()
                    .any(|&slot| (slot as usize) >= opened_values.len())
            {
                return Err(WhirError::InvalidInputError);
            }
            // [F-017] Bind the proof's embedded pruned indices to the
            // transcript-sampled query points (sorted+deduped) by value.
            let (sorted_unique, _) = Self::sorted_unique_slots(query_points);
            if let Some(recovered) = self
                .mmcs
                .recover_pruned_indices(&pruned.round_pruned[batch_idx])
            {
                if recovered.len() != sorted_unique.len()
                    || recovered
                        .iter()
                        .zip(sorted_unique.iter())
                        .any(|(&got, &want)| got as usize != want)
                {
                    return Err(WhirError::InvalidInputError);
                }
            }
            self.mmcs
                .verify_batch_pruned(
                    &commitment_batch[batch_idx],
                    &stacked_dims_by_batch[batch_idx],
                    opened_values,
                    &pruned.round_pruned[batch_idx],
                )
                .map_err(|_| WhirError::CommitmentCheckFailed)?;
        }

        let mut leaf_sums = Vec::with_capacity(query_points.len());
        for query_idx in 0..query_points.len() {
            let mut leaf_sum = EF::zero();
            for (batch_idx, coeffs) in coeffs_by_batch.iter().enumerate() {
                let slot = pruned.query_to_unique_slot[batch_idx][query_idx] as usize;
                let opened_values = &pruned.round_opened_values[batch_idx][slot];
                if opened_values.len() != 1 {
                    return Err(WhirError::InvalidInputError);
                }
                leaf_sum += compute_dotproduct_mix(&coeffs.column_coeffs, &opened_values[0]);
            }
            leaf_sums.push(leaf_sum);
        }
        Ok(leaf_sums)
    }

    fn verify_iopp_rows_pruned(
        &self,
        commitment: &FriMmcs::Commitment,
        query_points: &[usize],
        current_codeword_log: usize,
        log_folding: usize,
        pruned_round: &WhirPrunedIoppRound<EF, FriMmcs>,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, Vec<Vec<EF>>> {
        if log_folding == 0 || current_codeword_log < log_folding {
            return Err(WhirError::InvalidInputError);
        }
        let row_width = 1usize << log_folding;
        let log_row_height = current_codeword_log - log_folding;
        let row_indices = query_points
            .iter()
            .map(|point| point >> log_folding)
            .collect::<Vec<_>>();
        let (sorted_unique, expected_q2u) = Self::sorted_unique_slots(&row_indices);
        if pruned_round.opened_rows.len() != sorted_unique.len()
            || pruned_round.query_to_unique_slot != expected_q2u
        {
            return Err(WhirError::InvalidInputError);
        }
        // [F-017] Bind the proof's embedded pruned indices to OUR
        // transcript-sampled row indices by value.
        if let Some(recovered) = self
            .config
            .fri
            .mmcs
            .recover_pruned_indices(&pruned_round.pruned_proof)
        {
            if recovered.len() != sorted_unique.len()
                || recovered
                    .iter()
                    .zip(sorted_unique.iter())
                    .any(|(&got, &want)| got as usize != want)
            {
                return Err(WhirError::InvalidInputError);
            }
        }
        for opened_row in &pruned_round.opened_rows {
            if opened_row.len() != 1 || opened_row[0].len() != row_width {
                return Err(WhirError::InvalidInputError);
            }
        }

        self.config
            .fri
            .mmcs
            .verify_batch_pruned(
                commitment,
                &[Dimensions {
                    width: row_width,
                    height: 1 << log_row_height,
                }],
                &pruned_round.opened_rows,
                &pruned_round.pruned_proof,
            )
            .map_err(WhirError::CommitPhaseMmcsError)?;

        pruned_round
            .query_to_unique_slot
            .iter()
            .map(|&slot| {
                pruned_round
                    .opened_rows
                    .get(slot as usize)
                    .and_then(|row| row.first())
                    .cloned()
                    .ok_or(WhirError::InvalidInputError)
            })
            .collect()
    }

    fn prepare_stacked_opening_inputs(
        &self,
        polynomials_batch: &[Vec<CompressedMatrix<F>>],
        prover_data_batch: Vec<WhirPcsProverData<F, InputMmcs>>,
        opened_values: &[Vec<Vec<EF>>],
        full_opening_point: &[EF],
        stack_log_height: usize,
        challenger: &mut Challenger,
    ) -> WhirStackedResult<F, InputMmcs, FriMmcs, EF, StackedOpeningPreparation<F, InputMmcs, EF>>
    {
        use crate::whir::whir_helpers::{
            build_q_matrix_for_batch, build_stacked_evaluations, compute_q_at_point_for_batch,
            reduction_target_for_batch,
        };

        let stack_height = 1usize << stack_log_height;

        let dimensions_by_batch = polynomials_batch
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|matrix| Dimensions {
                        width: matrix.width(),
                        height: matrix.height(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let layouts = dimensions_by_batch
            .iter()
            .map(|dims| {
                StackedBatchLayout::from_dimensions(dims, stack_log_height, EF::D)
                    .map_err(|_| WhirError::InvalidInputError)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let raw_stacked_data = prover_data_batch
            .into_iter()
            .map(|data| {
                data.into_stacked()
                    .map_err(|_| WhirError::InvalidInputError)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Bind the claimed openings before deriving the reduction challenge.
        for batch_values in opened_values.iter() {
            for mat_values in batch_values.iter() {
                for v in mat_values.iter() {
                    challenger.observe_ext_element(*v);
                }
            }
        }

        let lambda: EF = challenger.sample_ext_element();

        // Track whether each batch is opened as base columns or flattened
        // extension columns; Q-table construction must mirror that layout.
        let flattened_flags: Vec<bool> = dimensions_by_batch
            .iter()
            .zip(opened_values.iter())
            .map(|(dims, values)| self.batch_uses_flattened_ext_dims(dims, values))
            .collect();

        // Compute the reduction claim T = sum_i lambda^i * opened_value_i.
        let mut target = EF::zero();
        let mut lambda_power = EF::one();
        for ((dims, values), (layout, &uses_flat)) in dimensions_by_batch
            .iter()
            .zip(opened_values.iter())
            .zip(layouts.iter().zip(flattened_flags.iter()))
        {
            let (t_batch, _consumed, next_power) = reduction_target_for_batch::<EF, F>(
                layout,
                dims,
                values,
                lambda,
                lambda_power,
                uses_flat,
            );
            target += t_batch;
            lambda_power = next_power;
        }

        // Build matrix-oriented pair-product inputs. If the prover data carries
        // a cached stacked matrix, the F side borrows it directly. Otherwise we
        // materialize one row-major EF matrix for the F side.
        let mut q_matrices: Vec<RowMajorMatrix<EF>> = Vec::with_capacity(raw_stacked_data.len());
        let mut owned_f_matrices: Vec<Option<RowMajorMatrix<EF>>> =
            Vec::with_capacity(raw_stacked_data.len());

        lambda_power = EF::one();
        for (((batch, layout), &uses_flat), (_mmcs_prover_data, stacked)) in polynomials_batch
            .iter()
            .zip(layouts.iter())
            .zip(flattened_flags.iter())
            .zip(raw_stacked_data.iter())
        {
            if &stacked.layout != layout {
                return Err(WhirError::InvalidInputError);
            }
            if let Some(stacked_evaluations) = &stacked.cached_evaluations {
                if stacked_evaluations.height() != stack_height
                    || stacked_evaluations.width() != layout.width
                {
                    return Err(WhirError::InvalidInputError);
                }
            }

            let (q_matrix, _consumed, next_power) = build_q_matrix_for_batch::<EF, F>(
                layout,
                full_opening_point,
                lambda,
                lambda_power,
                uses_flat,
            );
            lambda_power = next_power;
            q_matrices.push(q_matrix);

            if stacked.cached_evaluations.is_some() {
                owned_f_matrices.push(None);
            } else {
                let stacked_mat =
                    build_stacked_evaluations(&batch.iter().collect::<Vec<_>>(), layout);
                owned_f_matrices.push(Some(base_matrix_into_ext::<F, EF>(stacked_mat)));
            }
        }

        // Prove sum_j F_j(u) * Q_j(u) = T with the specialized degree-2
        // pair-product sumcheck.
        let (reduction_sumcheck, u_challenges, final_evals) = {
            let f_inputs: Vec<PairProductLeftInput<'_, F, EF>> = raw_stacked_data
                .iter()
                .zip(owned_f_matrices.iter())
                .map(|((_mmcs_prover_data, stacked), owned)| {
                    if let Some(stacked_evaluations) = &stacked.cached_evaluations {
                        Ok(PairProductLeftInput::Base(stacked_evaluations.as_ref()))
                    } else {
                        owned
                            .as_ref()
                            .map(PairProductLeftInput::Ext)
                            .ok_or(WhirError::InvalidInputError)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            SumcheckInstanceProof::sumcheck_prove_pair_products(
                &target,
                stack_log_height,
                &f_inputs,
                &mut q_matrices,
                challenger,
            )
            .map_err(|_| WhirError::SumcheckPhaseError)?
        };

        // u = challenges from the reduction sumcheck (LSB-first binding order).
        // Reverse to MSB-first for EqPolynomial/selector_eq convention.
        let u: Vec<EF> = u_challenges.into_iter().rev().collect();

        // Recompute Q_j(u) for each stacked column at the sampled point.
        lambda_power = EF::one();
        let mut q_at_u_by_batch: Vec<Vec<EF>> = Vec::with_capacity(layouts.len());
        for (layout, &uses_flat) in layouts.iter().zip(flattened_flags.iter()) {
            let (q_batch, _consumed, next_power) = compute_q_at_point_for_batch::<EF, F>(
                layout,
                full_opening_point,
                &u,
                lambda,
                lambda_power,
                uses_flat,
            );
            lambda_power = next_power;
            q_at_u_by_batch.push(q_batch);
        }

        // Fold all stacked F columns into one polynomial opened by WHIR.
        let mut combined_evals = vec![EF::zero(); stack_height];
        for ((q_batch, (_mmcs_prover_data, stacked)), owned) in q_at_u_by_batch
            .iter()
            .zip(raw_stacked_data.iter())
            .zip(owned_f_matrices.iter())
        {
            if let Some(stacked_evaluations) = &stacked.cached_evaluations {
                accumulate_base_matrix_columns(
                    stacked_evaluations.as_ref(),
                    q_batch,
                    &mut combined_evals,
                );
            } else {
                let f_matrix = owned.as_ref().ok_or(WhirError::InvalidInputError)?;
                accumulate_ext_matrix_columns(f_matrix, q_batch, &mut combined_evals);
            }
        }

        // The final sumcheck claim is sum_j F_j(u) * Q_j(u).
        let running_claim = final_evals
            .chunks_exact(2)
            .fold(EF::zero(), |acc, pair| acc + pair[0] * pair[1]);

        Ok(StackedOpeningPreparation {
            combined_evals,
            running_claim,
            opening_point: u,
            stacked_data: raw_stacked_data
                .into_iter()
                .map(|(mmcs_prover_data, _stacked)| mmcs_prover_data)
                .collect(),
            reduction_proof: Some(StackingReductionProof {
                sumcheck: reduction_sumcheck,
            }),
        })
    }

    fn prove_stacked_whir_query_round(
        &self,
        round_idx: usize,
        round: &WhirRoundSchedule,
        round_schedule: &[WhirRoundSchedule],
        round_config: WhirRoundQueryConfig,
        state: &mut StackedWhirProverState<'_, F, InputMmcs, EF, FriMmcs>,
        challenger: &mut Challenger,
    ) -> WhirStackedResult<
        F,
        InputMmcs,
        FriMmcs,
        EF,
        StackedWhirRoundOutput<F, InputMmcs, EF, FriMmcs>,
    > {
        if round.start_round != state.stack_log_height - *state.consumed_rounds {
            return Err(WhirError::InvalidInputError);
        }
        let current_codeword_log = round.codeword_log;
        let log_row_height = round.row_log_height();
        let remaining_dim = round.poly_log_after();
        let ((sumcheck_polys, group_challenges), folding_witness) = self
            .prove_sumcheck_rounds_with_folding_pow(
                state.running_claim,
                round.log_folding,
                state.current_polys,
                challenger,
                round_config.grinding_bits_folding,
            )?;

        let is_last_round = round_idx + 1 == round_schedule.len();
        let mut ood_point = None;
        let mut ood_value = None;
        if is_last_round {
            *state.final_poly_evals = state.current_polys[0].evals.clone();
            for coeff in state.final_poly_evals.iter() {
                challenger.observe_ext_element(*coeff);
            }
        } else {
            let next_round = round_schedule[round_idx + 1];
            let (root, tree) = self.commit_iopp_codeword(
                &state.current_polys[0].evals,
                next_round.log_folding,
                next_round.log_blowup,
                state.dft,
                challenger,
            );
            state.iopp_commitments.push(root);
            state.iopp_prover_data.push(tree);

            let z0 = challenger.sample_ext_element();
            let z0_point = Self::pow2_ext_point(z0, remaining_dim);
            let y0 = state.current_polys[0].evaluate_mix(&z0_point);
            challenger.observe_ext_element(y0);
            state.ood_values.push(y0);
            ood_point = Some(z0_point);
            ood_value = Some(y0);
        }

        let query_witness = self.find_pow_witness(challenger, round_config.grinding_bits_query)?;
        let query_points = (0..round_config.num_queries)
            .map(|_| challenger.sample_bits(current_codeword_log))
            .collect::<Vec<_>>();

        let first_round_input_openings = if round_idx == 0 {
            Some(if self.config.path_pruning {
                self.open_stacked_input_batches_pruned(&query_points, state.stacked_data)
            } else {
                WhirInputProof::from_per_query(
                    self.open_stacked_input_batches(&query_points, state.stacked_data),
                )
            })
        } else {
            None
        };

        let (rows_by_query, mut iopp_round, pruned_iopp_round) = if self.config.path_pruning {
            let (rows_by_query, pruned_iopp_round) = self.open_iopp_rows_pruned(
                &query_points,
                &state.iopp_prover_data[round_idx],
                round.log_folding,
            )?;
            (rows_by_query, None, Some(pruned_iopp_round))
        } else {
            let mut query_proofs = query_points
                .iter()
                .map(|&point| {
                    self.open_iopp_row_full(
                        &state.iopp_prover_data[round_idx],
                        point,
                        round.log_folding,
                    )
                })
                .collect::<Vec<_>>();
            let rows_by_query = query_proofs
                .iter()
                .map(|query| query.current_opening.opened_values.clone())
                .collect::<Vec<_>>();
            for query in &mut query_proofs {
                query.next_opening = None;
            }
            (rows_by_query, Some(WhirIoppRound { query_proofs }), None)
        };

        let gamma = challenger.sample_ext_element();
        if let (Some(point), Some(value)) = (ood_point.as_deref(), ood_value) {
            *state.running_claim += gamma * value;
            Self::accumulate_eq_evals(&mut state.current_polys[1].evals, point, gamma);
        }

        let mut gamma_power = gamma * gamma;
        for (&query_point, opened_row) in query_points.iter().zip(rows_by_query.iter()) {
            let row_index = query_point >> round.log_folding;
            let yi = self.fold_opened_iopp_row(
                opened_row.clone(),
                row_index,
                round.log_folding,
                log_row_height,
                &group_challenges,
            )?;
            *state.running_claim += gamma_power * yi;

            let query_point = Self::codeword_query_point(
                row_index,
                log_row_height,
                remaining_dim,
                round.log_blowup,
            );
            Self::accumulate_eq_evals(&mut state.current_polys[1].evals, &query_point, gamma_power);
            gamma_power *= gamma;
        }

        *state.consumed_rounds += round.log_folding;

        Ok(StackedWhirRoundOutput {
            iopp_round: iopp_round.take(),
            pruned_iopp_round,
            query_witness,
            folding_witness,
            first_round_input_openings,
            sumcheck_polys,
        })
    }

    pub(crate) fn open_stacked(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<F>>>,
        prover_data_batch: Vec<WhirPcsProverData<F, InputMmcs>>,
        opening_point: &[EF],
        opened_values: &[Vec<Vec<EF>>],
        challenger: &mut Challenger,
        stack_log_height: usize,
    ) -> StackedProofResult<F, InputMmcs, FriMmcs, EF> {
        let full_opening_point =
            self.extend_stacked_opening_point(opening_point, stack_log_height, challenger)?;

        let StackedOpeningPreparation {
            combined_evals,
            mut running_claim,
            opening_point: whir_opening_point,
            stacked_data,
            reduction_proof,
        } = profile::time("open.whir_stacking_eq_ms", || {
            self.prepare_stacked_opening_inputs(
                &polynomials_batch,
                prover_data_batch,
                opened_values,
                &full_opening_point,
                stack_log_height,
                challenger,
            )
        })?;

        let grinding_batching_data =
            self.find_pow_witness(challenger, self.config.fri.grinding_bits_batching)?;

        let k = self.config.fri.log_final_poly_len.min(stack_log_height);
        let commit_schedule = self.commit_schedule(stack_log_height, k);
        let committed_groups = commit_schedule.len();

        if let Some(round_query_configs) = self.config.round_query_configs(committed_groups) {
            let round_schedule = self
                .whir_round_schedule(stack_log_height, k)
                .ok_or(WhirError::InvalidInputError)?;
            if committed_groups == 0 {
                return Err(WhirError::InvalidInputError);
            }

            return with_thread_local_evals_dft(|dft| {
                let eq_polynomial = EqPolynomial::new(whir_opening_point.clone()).to_ml();
                let mut current_polys =
                    vec![MultilinearPolynomial::new(combined_evals), eq_polynomial];

                let mut sumcheck_polys = Vec::new();
                let mut iopp_commitments = Vec::with_capacity(committed_groups);
                let mut iopp_prover_data = Vec::with_capacity(committed_groups);
                let mut ood_values = Vec::with_capacity(committed_groups.saturating_sub(1));
                let mut final_poly_evals = Vec::new();

                let (first_root, first_tree) = self.commit_iopp_codeword(
                    &current_polys[0].evals,
                    round_schedule[0].log_folding,
                    round_schedule[0].log_blowup,
                    dft,
                    challenger,
                );
                iopp_commitments.push(first_root);
                iopp_prover_data.push(first_tree);

                let mut pruned_iopp = self.config.path_pruning.then(|| WhirRoundPrunedQueryProof {
                    rounds: Vec::with_capacity(committed_groups),
                });
                let mut round_iopp = WhirRoundQueryProof {
                    rounds: if self.config.path_pruning {
                        Vec::new()
                    } else {
                        Vec::with_capacity(committed_groups)
                    },
                    pruned: None,
                    query_witnesses: Vec::with_capacity(committed_groups),
                    folding_witnesses: Vec::with_capacity(committed_groups),
                };
                let mut first_round_input_openings: Option<WhirInputProof<F, InputMmcs>> = None;
                let mut consumed_rounds = 0usize;

                {
                    let mut whir_state = StackedWhirProverState {
                        stack_log_height,
                        stacked_data: &stacked_data,
                        dft,
                        current_polys: &mut current_polys,
                        running_claim: &mut running_claim,
                        iopp_commitments: &mut iopp_commitments,
                        iopp_prover_data: &mut iopp_prover_data,
                        ood_values: &mut ood_values,
                        final_poly_evals: &mut final_poly_evals,
                        consumed_rounds: &mut consumed_rounds,
                    };

                    for (round_idx, round) in round_schedule.iter().enumerate() {
                        let output = self.prove_stacked_whir_query_round(
                            round_idx,
                            round,
                            &round_schedule,
                            round_query_configs[round_idx],
                            &mut whir_state,
                            challenger,
                        )?;
                        if let Some(input_openings) = output.first_round_input_openings {
                            first_round_input_openings = Some(input_openings);
                        }
                        sumcheck_polys.extend(output.sumcheck_polys);
                        if let Some(iopp_round) = output.iopp_round {
                            round_iopp.rounds.push(iopp_round);
                        }
                        if let Some(pruned_round) = output.pruned_iopp_round {
                            pruned_iopp
                                .as_mut()
                                .ok_or(WhirError::InvalidInputError)?
                                .rounds
                                .push(pruned_round);
                        }
                        round_iopp.query_witnesses.push(output.query_witness);
                        round_iopp.folding_witnesses.push(output.folding_witness);
                    }
                }
                round_iopp.pruned = pruned_iopp;

                Ok(WhirProof {
                    stack_log_height: Some(stack_log_height),
                    sumcheck_transcript: SumcheckInstanceProof {
                        uni_polys: sumcheck_polys,
                    },
                    iopp_oracles: iopp_commitments,
                    ood_values,
                    iopp_queries: Vec::new(),
                    round_iopp: Some(round_iopp),
                    query_openings: first_round_input_openings
                        .ok_or(WhirError::InvalidInputError)?,
                    grinding_batching_witness: grinding_batching_data,
                    grinding_query_witness: Vec::new(),
                    final_poly: final_poly_evals,
                    iopp_pruned: None,
                    stacking_reduction: reduction_proof,
                })
            });
        }

        let eq_polynomial = EqPolynomial::new(whir_opening_point.clone()).to_ml();
        let mut current_polys = vec![MultilinearPolynomial::new(combined_evals), eq_polynomial];

        let mut sumcheck_polys = Vec::new();
        let mut iopp_commitments = Vec::new();
        let mut iopp_prover_data = Vec::new();
        let mut iopp_log_foldings = commit_schedule
            .iter()
            .map(|group| group.log_folding)
            .collect::<Vec<_>>();
        let mut final_poly_evals: Vec<EF> = Vec::new();

        for group in commit_schedule.iter() {
            let (root, tree) = with_thread_local_evals_dft(|dft| {
                self.commit_iopp_codeword(
                    &current_polys[0].evals,
                    group.log_folding,
                    self.config.fri.log_blowup,
                    dft,
                    challenger,
                )
            });
            iopp_commitments.push(root);
            iopp_prover_data.push(tree);

            let (polys, _) = Self::prove_sumcheck_rounds(
                &mut running_claim,
                group.log_folding,
                &mut current_polys,
                challenger,
            )?;
            sumcheck_polys.extend(polys);
        }

        if k > 0 {
            final_poly_evals = current_polys[0].evals.clone();
            for coeff in &final_poly_evals {
                challenger.observe_ext_element(*coeff);
            }
            let (polys, _) =
                Self::prove_sumcheck_rounds(&mut running_claim, k, &mut current_polys, challenger)?;
            sumcheck_polys.extend(polys);
        } else {
            let (root, tree) = with_thread_local_evals_dft(|dft| {
                self.commit_iopp_codeword(
                    &current_polys[0].evals,
                    1,
                    self.config.fri.log_blowup,
                    dft,
                    challenger,
                )
            });
            iopp_commitments.push(root);
            iopp_prover_data.push(tree);
            iopp_log_foldings.push(1);
        }

        let grinding_query_data =
            self.find_pow_witness(challenger, self.config.fri.grinding_bits_query)?;

        let query_points: Vec<usize> = (0..self.config.fri.num_queries)
            .map(|_| challenger.sample_bits(stack_log_height + self.config.fri.log_blowup))
            .collect();

        let use_path_pruning = self.config.path_pruning;

        let query_openings_bundle: WhirInputProof<F, InputMmcs> = if use_path_pruning {
            let mut sorted_dedup = query_points.clone();
            sorted_dedup.sort_unstable();
            sorted_dedup.dedup();
            let q2u_round = query_points
                .iter()
                .map(|&q| sorted_dedup.binary_search(&q).unwrap() as u32)
                .collect::<Vec<_>>();

            let round_results = profile::time("open.whir_input_mmcs_open_ms", || {
                stacked_data
                    .par_iter()
                    .map(|mmcs_prover_data| {
                        let (uniq_opened, pruned_proof) = self
                            .mmcs
                            .open_batch_pruned(&query_points, mmcs_prover_data.as_ref());
                        (pruned_proof, uniq_opened)
                    })
                    .collect::<Vec<_>>()
            });

            let mut round_pruned = Vec::with_capacity(round_results.len());
            let mut round_opened_values = Vec::with_capacity(round_results.len());
            for (pruned_proof, uniq_opened) in round_results {
                round_pruned.push(pruned_proof);
                round_opened_values.push(uniq_opened);
            }
            let q2u = vec![q2u_round; stacked_data.len()];

            WhirInputProof {
                per_query: Vec::new(),
                pruned: Some(PrunedQueryOpenings {
                    round_pruned,
                    round_opened_values,
                    query_to_unique_slot: q2u,
                }),
            }
        } else {
            let qo: Vec<Vec<BatchOpening<F, InputMmcs>>> =
                profile::time("open.whir_input_mmcs_open_ms", || {
                    query_points
                        .par_iter()
                        .map(|&point| {
                            stacked_data
                                .iter()
                                .map(|mmcs_prover_data| {
                                    let (values, proof) =
                                        self.mmcs.open_batch(point, mmcs_prover_data.as_ref());
                                    BatchOpening {
                                        opened_values: values,
                                        opening_proof: proof,
                                    }
                                })
                                .collect()
                        })
                        .collect()
                });
            WhirInputProof::from_per_query(qo)
        };

        let (iopp_queries, iopp_pruned) = if use_path_pruning {
            let committed_groups = commit_schedule.len();
            let pruned = if self.cross_round_enabled(stack_log_height, k) {
                answer_queries_pruned_with_log_foldings(
                    &self.config.fri,
                    &iopp_prover_data[..committed_groups],
                    &query_points,
                    &iopp_log_foldings[..committed_groups],
                )
            } else {
                answer_queries_pruned(
                    &self.config.fri,
                    &iopp_prover_data[..committed_groups],
                    &query_points,
                )
            };
            (Vec::new(), Some(pruned))
        } else {
            let queries = query_points
                .iter()
                .map(|&point| {
                    answer_query_with_log_foldings(
                        &self.config.fri,
                        &iopp_prover_data,
                        point,
                        &iopp_log_foldings,
                    )
                })
                .collect::<Vec<_>>();
            (queries, None)
        };

        Ok(WhirProof {
            stack_log_height: Some(stack_log_height),
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
            stacking_reduction: reduction_proof,
        })
    }

    fn verify_stacked_whir_round_iopp(
        &self,
        commitment_batch: &[InputMmcs::Commitment],
        stacked_dims_by_batch: &[Vec<Dimensions>],
        coeffs_by_batch: &[StackedBatchCoefficients<EF>],
        input_openings: &WhirInputProof<F, InputMmcs>,
        iopp_oracles: &[FriMmcs::Commitment],
        ood_values: &[EF],
        round_iopp: &WhirRoundQueryProof<EF, FriMmcs, F>,
        final_poly: &[EF],
        sumcheck_transcript: &SumcheckInstanceProof<EF>,
        round_schedule: &[WhirRoundSchedule],
        stack_log_height: usize,
        mut current_claim: EF,
        full_opening_point: &[EF],
        challenger: &mut Challenger,
    ) -> Result<(), WhirError<FriMmcs::Error, InputMmcs::Error>> {
        let committed_groups = round_schedule.len();
        let pruned_iopp = round_iopp.pruned.as_ref();
        if committed_groups == 0
            || iopp_oracles.len() != committed_groups
            || ood_values.len() != committed_groups.saturating_sub(1)
        {
            return Err(WhirError::InvalidInputError);
        }
        if let Some(pruned) = pruned_iopp {
            if !round_iopp.rounds.is_empty() || pruned.rounds.len() != committed_groups {
                return Err(WhirError::InvalidInputError);
            }
            if !input_openings.per_query.is_empty() || input_openings.pruned.is_none() {
                return Err(WhirError::InvalidInputError);
            }
        } else if round_iopp.rounds.len() != committed_groups || input_openings.pruned.is_some() {
            return Err(WhirError::InvalidInputError);
        }

        let consumed_sumcheck_rounds = round_schedule
            .iter()
            .map(|round| round.log_folding)
            .sum::<usize>();
        if consumed_sumcheck_rounds > stack_log_height
            || sumcheck_transcript.uni_polys.len() != consumed_sumcheck_rounds
        {
            return Err(WhirError::InvalidInputError);
        }
        let final_log_height = stack_log_height - consumed_sumcheck_rounds;
        if final_poly.len() != (1usize << final_log_height) {
            return Err(WhirError::InvalidInputError);
        }

        let round_query_configs = self
            .config
            .round_query_configs(committed_groups)
            .ok_or(WhirError::InvalidInputError)?;
        if round_iopp.query_witnesses.len() != committed_groups {
            return Err(WhirError::InvalidInputError);
        }

        challenger.observe(iopp_oracles[0].clone());

        let mut weight_terms = vec![SymbolicWeightTerm {
            coeff: EF::one(),
            point: full_opening_point.to_vec(),
        }];
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut consumed_rounds = 0usize;

        let phase_rounds = Instant::now();
        for (round_idx, round) in round_schedule.iter().enumerate() {
            if round.start_round != stack_log_height - consumed_rounds {
                return Err(WhirError::InvalidInputError);
            }
            let current_codeword_log = round.codeword_log;
            let log_row_height = round.row_log_height();
            let remaining_dim = round.poly_log_after();
            let mut group_challenges = Vec::with_capacity(round.log_folding);
            let round_config = round_query_configs[round_idx];

            for _ in 0..round.log_folding {
                let uni_poly = poly_iter.next().ok_or(WhirError::SumcheckPhaseError)?;
                if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != current_claim {
                    return Err(WhirError::SumcheckPhaseError);
                }
                uni_poly
                    .coeffs
                    .iter()
                    .for_each(|c| challenger.observe_ext_element(*c));
                let r_fold = challenger.sample_ext_element();
                current_claim = uni_poly.evaluate(&r_fold);
                group_challenges.push(r_fold);
                Self::fold_symbolic_weight_terms(&mut weight_terms, r_fold)?;

                if round_config.grinding_bits_folding > 0 {
                    let folding_witness = &round_iopp.folding_witnesses[round_idx];
                    let witness_offset = 2 * group_challenges.len().saturating_sub(1);
                    if folding_witness.len() != 2 * round.log_folding
                        || witness_offset + 1 >= folding_witness.len()
                    {
                        return Err(WhirError::InvalidInputError);
                    }
                    challenger.observe(folding_witness[witness_offset]);
                    if !challenger.check_witness(
                        round_config.grinding_bits_folding,
                        folding_witness[witness_offset + 1],
                    ) {
                        return Err(WhirError::InvalidPowWitness);
                    }
                }
            }

            let is_last_round = round_idx + 1 == committed_groups;
            let ood_point = if is_last_round {
                for coeff in final_poly {
                    challenger.observe_ext_element(*coeff);
                }
                None
            } else {
                challenger.observe(iopp_oracles[round_idx + 1].clone());
                let z0 = challenger.sample_ext_element();
                let y0 = ood_values[round_idx];
                challenger.observe_ext_element(y0);
                Some((Self::pow2_ext_point(z0, remaining_dim), y0))
            };

            let witness = &round_iopp.query_witnesses[round_idx];
            if witness.len() != 2 {
                return Err(WhirError::InvalidInputError);
            }
            challenger.observe(witness[0]);
            if !challenger.check_witness(round_config.grinding_bits_query, witness[1]) {
                return Err(WhirError::InvalidPowWitness);
            }
            let query_points = (0..round_config.num_queries)
                .map(|_| challenger.sample_bits(current_codeword_log))
                .collect::<Vec<_>>();
            let phase = Instant::now();
            let opened_rows_by_query = if let Some(pruned) = pruned_iopp {
                self.verify_iopp_rows_pruned(
                    &iopp_oracles[round_idx],
                    &query_points,
                    current_codeword_log,
                    round.log_folding,
                    &pruned.rounds[round_idx],
                )?
            } else {
                let round_proof = &round_iopp.rounds[round_idx];
                if round_proof.query_proofs.len() != query_points.len() {
                    return Err(WhirError::InvalidInputError);
                }
                round_proof
                    .query_proofs
                    .iter()
                    .zip(query_points.iter())
                    .map(|(query_proof, &query_point)| {
                        if query_proof.next_opening.is_some() {
                            return Err(WhirError::InvalidInputError);
                        }
                        self.verify_iopp_step_full(
                            &iopp_oracles[round_idx],
                            query_point,
                            current_codeword_log,
                            round.log_folding,
                            &query_proof.current_opening,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            profile::add_ms("verify.whir_iopp_rows_us", phase.elapsed().as_micros());

            let phase = Instant::now();
            let input_leaf_sums = if round_idx == 0 {
                if let Some(pruned) = input_openings.pruned.as_ref() {
                    self.verify_stacked_input_pruned(
                        commitment_batch,
                        stacked_dims_by_batch,
                        coeffs_by_batch,
                        &query_points,
                        pruned,
                    )?
                } else {
                    if input_openings.per_query.len() != query_points.len() {
                        return Err(WhirError::InvalidInputError);
                    }
                    let mut leaf_sums = Vec::with_capacity(query_points.len());
                    for (query_idx, leaf_opening) in input_openings.per_query.iter().enumerate() {
                        if leaf_opening.len() != coeffs_by_batch.len() {
                            return Err(WhirError::InvalidInputError);
                        }
                        for (batch_idx, opening) in leaf_opening.iter().enumerate() {
                            self.mmcs
                                .verify_batch(
                                    &commitment_batch[batch_idx],
                                    &stacked_dims_by_batch[batch_idx],
                                    query_points[query_idx],
                                    &opening.opened_values,
                                    &opening.opening_proof,
                                )
                                .map_err(|_| WhirError::CommitmentCheckFailed)?;
                        }
                        leaf_sums.push(
                            coeffs_by_batch
                                .iter()
                                .zip(leaf_opening.iter())
                                .map(|(coeffs, opening)| {
                                    compute_dotproduct_mix(
                                        &coeffs.column_coeffs,
                                        &opening.opened_values[0],
                                    )
                                })
                                .sum::<EF>(),
                        );
                    }
                    leaf_sums
                }
            } else {
                Vec::new()
            };
            profile::add_ms("verify.whir_input_batch_us", phase.elapsed().as_micros());

            let gamma = challenger.sample_ext_element();
            if let Some((point, value)) = ood_point.as_ref() {
                current_claim += gamma * *value;
                weight_terms.push(SymbolicWeightTerm {
                    coeff: gamma,
                    point: point.clone(),
                });
            }

            let phase = Instant::now();
            let mut gamma_power = gamma * gamma;
            for (query_idx, (&query_point, opened_row)) in query_points
                .iter()
                .zip(opened_rows_by_query.iter())
                .enumerate()
            {
                let local_index = query_point & ((1usize << round.log_folding) - 1);

                if round_idx == 0 && opened_row[local_index] != input_leaf_sums[query_idx] {
                    return Err(WhirError::FriFinalStepMisMatch);
                }

                let row_index = query_point >> round.log_folding;
                let yi = self.fold_opened_iopp_row(
                    opened_row.clone(),
                    row_index,
                    round.log_folding,
                    log_row_height,
                    &group_challenges,
                )?;
                current_claim += gamma_power * yi;

                let query_point = Self::codeword_query_point(
                    row_index,
                    log_row_height,
                    remaining_dim,
                    round.log_blowup,
                );
                weight_terms.push(SymbolicWeightTerm {
                    coeff: gamma_power,
                    point: query_point,
                });
                gamma_power *= gamma;
            }
            profile::add_ms("verify.whir_fold_rows_us", phase.elapsed().as_micros());

            consumed_rounds += round.log_folding;
        }
        profile::add_ms(
            "verify.whir_rounds_total_us",
            phase_rounds.elapsed().as_micros(),
        );

        if poly_iter.next().is_some() {
            return Err(WhirError::InvalidInputError);
        }

        let phase = Instant::now();
        let final_acc = Self::symbolic_final_accumulator(final_poly, &weight_terms)?;
        profile::add_ms("verify.whir_final_acc_us", phase.elapsed().as_micros());
        if final_acc != current_claim {
            return Err(WhirError::FinalPolyMismatch);
        }

        Ok(())
    }

    pub(crate) fn verify_stacked(
        &self,
        commitment_batch: Vec<InputMmcs::Commitment>,
        matrices_size_batch: &[Vec<Dimensions>],
        opening_point: &[EF],
        opened_values_batch: &[Vec<Vec<EF>>],
        proof: &WhirProof<EF, FriMmcs, F, WhirInputProof<F, InputMmcs>>,
        challenger: &mut Challenger,
        stack_log_height: usize,
    ) -> Result<(), WhirError<FriMmcs::Error, InputMmcs::Error>> {
        let WhirProof {
            stack_log_height: _,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned,
            stacking_reduction: _,
        } = proof;

        if round_iopp.is_some() {
            if iopp_pruned.is_some() || !iopp_queries.is_empty() {
                return Err(WhirError::InvalidInputError);
            }
        } else {
            let query_openings_pruned = query_openings.pruned.as_ref();
            match (iopp_pruned.is_some(), query_openings_pruned.is_some()) {
                (true, true) | (false, false) => {}
                _ => return Err(WhirError::InvalidInputError),
            }
            if grinding_query_witness.len() != 2 {
                return Err(WhirError::InvalidInputError);
            }
        }
        if grinding_batching_witness.len() != 2 {
            return Err(WhirError::InvalidInputError);
        }

        let phase = Instant::now();
        let full_opening_point =
            self.extend_stacked_opening_point(opening_point, stack_log_height, challenger)?;
        profile::add_ms("verify.whir_extend_point_us", phase.elapsed().as_micros());

        let phase = Instant::now();
        let layouts = matrices_size_batch
            .iter()
            .map(|dims| {
                StackedBatchLayout::from_dimensions(dims, stack_log_height, EF::D)
                    .map_err(|_| WhirError::InvalidInputError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        profile::add_ms("verify.whir_layouts_us", phase.elapsed().as_micros());

        // ── Stacking reduction verification ──
        // 1. Absorb opened_values
        let phase = Instant::now();
        for batch_values in opened_values_batch.iter() {
            for mat_values in batch_values.iter() {
                for v in mat_values.iter() {
                    challenger.observe_ext_element(*v);
                }
            }
        }
        profile::add_ms("verify.whir_absorb_opened_us", phase.elapsed().as_micros());

        // 2. Sample λ
        let lambda: EF = challenger.sample_ext_element();

        let flattened_flags: Vec<bool> = matrices_size_batch
            .iter()
            .zip(opened_values_batch.iter())
            .map(|(dims, values)| self.batch_uses_flattened_ext_dims(dims, values))
            .collect();

        // 3. Compute T = Σ λ^i · original_claim_i
        let phase = Instant::now();
        let mut target = EF::zero();
        let mut lambda_power = EF::one();
        for ((dims, values), (layout, &uses_flat)) in matrices_size_batch
            .iter()
            .zip(opened_values_batch.iter())
            .zip(layouts.iter().zip(flattened_flags.iter()))
        {
            let (t_batch, _consumed, next_power) =
                crate::whir::whir_helpers::reduction_target_for_batch::<EF, F>(
                    layout,
                    dims,
                    values,
                    lambda,
                    lambda_power,
                    uses_flat,
                );
            target += t_batch;
            lambda_power = next_power;
        }
        profile::add_ms(
            "verify.whir_reduction_target_us",
            phase.elapsed().as_micros(),
        );

        // 4. Verify reduction sumcheck
        let reduction = proof
            .stacking_reduction
            .as_ref()
            .ok_or(WhirError::InvalidInputError)?;
        if reduction.sumcheck.uni_polys.len() != stack_log_height {
            return Err(WhirError::InvalidInputError);
        }
        let phase = Instant::now();
        let mut reduction_claim = target;
        let mut u = Vec::with_capacity(stack_log_height);
        for uni_poly in &reduction.sumcheck.uni_polys {
            if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != reduction_claim {
                return Err(WhirError::SumcheckPhaseError);
            }
            uni_poly
                .coeffs
                .iter()
                .for_each(|c| challenger.observe_ext_element(*c));
            let r_j = challenger.sample_ext_element();
            reduction_claim = uni_poly.evaluate(&r_j);
            u.push(r_j);
        }
        // Reverse u from LSB-first (binding order) to MSB-first (EqPolynomial convention).
        u.reverse();
        profile::add_ms(
            "verify.whir_reduction_sumcheck_us",
            phase.elapsed().as_micros(),
        );

        // 5. Compute q_c = Q_c(u) and verify final claim
        let phase = Instant::now();
        lambda_power = EF::one();
        let total_stacked_width: usize = layouts.iter().map(|l| l.width).sum();
        let mut q_at_u_all: Vec<EF> = Vec::with_capacity(total_stacked_width);
        let mut coeffs_by_batch = Vec::with_capacity(matrices_size_batch.len());
        for (layout, &uses_flat) in layouts.iter().zip(flattened_flags.iter()) {
            let (q_batch, _consumed, next_power) =
                crate::whir::whir_helpers::compute_q_at_point_for_batch::<EF, F>(
                    layout,
                    &full_opening_point,
                    &u,
                    lambda,
                    lambda_power,
                    uses_flat,
                );
            lambda_power = next_power;
            coeffs_by_batch.push(StackedBatchCoefficients {
                column_coeffs: q_batch.clone(),
                chunk_coeffs: q_batch.clone(),
            });
            q_at_u_all.extend(q_batch);
        }
        profile::add_ms("verify.whir_q_at_u_us", phase.elapsed().as_micros());

        // (The final claim will be checked by the WHIR opening:
        //  reduction_claim == Σ_c F_c(u) * Q_c(u), which equals
        //  combined_evals(u) since combined_evals = Σ q_c * F_c.)
        let mut current_claim = reduction_claim;
        let whir_opening_point = u;

        challenger.observe(grinding_batching_witness[0]);
        if !challenger.check_witness(
            self.config.fri.grinding_bits_batching,
            grinding_batching_witness[1],
        ) {
            return Err(WhirError::InvalidPowWitness);
        }

        let k = self.config.fri.log_final_poly_len.min(stack_log_height);
        let commit_schedule = self.commit_schedule(stack_log_height, k);
        let iopp_log_foldings = commit_schedule
            .iter()
            .map(|group| group.log_folding)
            .collect::<Vec<_>>();
        let stacked_dims_by_batch = layouts
            .iter()
            .map(|layout| {
                layout
                    .stacked_dimensions(self.config.fri.log_blowup)
                    .to_vec()
            })
            .collect::<Vec<_>>();

        if let Some(round_iopp) = round_iopp.as_ref() {
            let round_schedule = self
                .whir_round_schedule(stack_log_height, k)
                .ok_or(WhirError::InvalidInputError)?;
            return self.verify_stacked_whir_round_iopp(
                &commitment_batch,
                &stacked_dims_by_batch,
                &coeffs_by_batch,
                query_openings,
                iopp_oracles,
                ood_values,
                round_iopp,
                final_poly,
                sumcheck_transcript,
                &round_schedule,
                stack_log_height,
                current_claim,
                &whir_opening_point,
                challenger,
            );
        }

        let query_openings_pruned = query_openings.pruned.as_ref();
        let query_openings = &query_openings.per_query;

        if sumcheck_transcript.uni_polys.len() != stack_log_height {
            return Err(WhirError::InvalidInputError);
        }
        if k > 0 && final_poly.len() != (1usize << k) {
            return Err(WhirError::InvalidInputError);
        }
        let expected_iopp_oracles = commit_schedule.len() + usize::from(k == 0);
        if iopp_oracles.len() != expected_iopp_oracles {
            return Err(WhirError::InvalidInputError);
        }
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();

        let mut folding_challenges: Vec<EF> = Vec::with_capacity(stack_log_height);
        let mut oracle_idx = 0usize;
        let mut schedule_idx = 0usize;
        for round in (0..=stack_log_height).rev() {
            if schedule_idx < commit_schedule.len()
                && commit_schedule[schedule_idx].start_round == round
            {
                challenger.observe(iopp_oracles[oracle_idx].clone());
                oracle_idx += 1;
                schedule_idx += 1;
            } else if round == 0 && k == 0 {
                if oracle_idx < iopp_oracles.len() {
                    challenger.observe(iopp_oracles[oracle_idx].clone());
                    oracle_idx += 1;
                }
            } else if round == k && k > 0 {
                for coeff in final_poly {
                    challenger.observe_ext_element(*coeff);
                }
            }
            if round == 0 {
                break;
            }

            let uni_poly = poly_iter.next().ok_or(WhirError::SumcheckPhaseError)?;
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
        }

        let fc_rev: Vec<EF> = folding_challenges.iter().rev().copied().collect();
        let combined_eq_sum = EqPolynomial::new(whir_opening_point.clone()).evaluate(&fc_rev);
        let combined_f_r = current_claim / combined_eq_sum;

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

        let final_codeword = if k > 0 {
            Some(with_thread_local_evals_dft(|dft| {
                self.encode_to_codeword(final_poly, self.config.fri.log_blowup, dft)
            }))
        } else {
            None
        };

        challenger.observe(grinding_query_witness[0]);
        if !challenger.check_witness(
            self.config.fri.grinding_bits_query,
            grinding_query_witness[1],
        ) {
            return Err(WhirError::InvalidPowWitness);
        }

        let query_points: Vec<usize> = (0..self.config.fri.num_queries)
            .map(|_| challenger.sample_bits(stack_log_height + self.config.fri.log_blowup))
            .collect();

        // [F-018] Pin the query counts before the standard path zips
        // `iopp_queries` with `query_openings`; a malicious proof with short
        // vectors would otherwise truncate the zip (vacuously passing `.all`)
        // and bypass IOPP soundness. Mirror the FRI verifier's check.
        if iopp_pruned.is_none()
            && (iopp_queries.len() != self.config.fri.num_queries
                || query_openings.len() != self.config.fri.num_queries)
        {
            return Err(WhirError::InvalidInputError);
        }

        let all_queries_valid = if let Some(std_pruned) = iopp_pruned.as_ref() {
            let qop = query_openings_pruned.ok_or(WhirError::InvalidInputError)?;
            let n_queries = query_points.len();
            if qop.round_pruned.len() != layouts.len()
                || qop.round_opened_values.len() != layouts.len()
                || qop.query_to_unique_slot.len() != layouts.len()
            {
                return Err(WhirError::InvalidInputError);
            }

            let mut per_round_ok = true;
            for batch_idx in 0..layouts.len() {
                let unique_opened = &qop.round_opened_values[batch_idx];
                let q2u_round = &qop.query_to_unique_slot[batch_idx];
                if q2u_round.len() != n_queries
                    || q2u_round
                        .iter()
                        .any(|&s| (s as usize) >= unique_opened.len())
                {
                    per_round_ok = false;
                    break;
                }
                if self
                    .mmcs
                    .verify_batch_pruned(
                        &commitment_batch[batch_idx],
                        &stacked_dims_by_batch[batch_idx],
                        unique_opened,
                        &qop.round_pruned[batch_idx],
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
                let mut leaf_sums_per_query = Vec::with_capacity(n_queries);
                for q in 0..n_queries {
                    let sum = coeffs_by_batch
                        .iter()
                        .enumerate()
                        .map(|(batch_idx, coeffs)| {
                            let slot = qop.query_to_unique_slot[batch_idx][q] as usize;
                            compute_dotproduct_mix(
                                &coeffs.column_coeffs,
                                &qop.round_opened_values[batch_idx][slot][0],
                            )
                        })
                        .sum::<EF>();
                    leaf_sums_per_query.push(BTreeMap::from([(
                        stack_log_height + self.config.fri.log_blowup,
                        sum,
                    )]));
                }

                self.verify_queries_iopp_p3_pruned_whir(
                    iopp_oracles.as_slice(),
                    &query_points,
                    &leaf_sums_per_query,
                    std_pruned,
                    &iopp_log_foldings,
                    &folding_challenges,
                    &[],
                    &whir_opening_point,
                    &combined_f_r,
                    final_codeword.as_deref(),
                )
                .is_ok()
            }
        } else {
            iopp_queries
                .par_iter()
                .zip(query_openings.par_iter())
                .enumerate()
                .all(|(i, (query, leaf_opening))| {
                    for (batch_idx, opening) in leaf_opening.iter().enumerate() {
                        if self
                            .mmcs
                            .verify_batch(
                                &commitment_batch[batch_idx],
                                &stacked_dims_by_batch[batch_idx],
                                query_points[i],
                                &opening.opened_values,
                                &opening.opening_proof,
                            )
                            .is_err()
                        {
                            return false;
                        }
                    }

                    let leaf_sum = coeffs_by_batch
                        .iter()
                        .zip(leaf_opening.iter())
                        .map(|(coeffs, opening)| {
                            compute_dotproduct_mix(&coeffs.column_coeffs, &opening.opened_values[0])
                        })
                        .sum::<EF>();
                    let leaf_sums =
                        BTreeMap::from([(stack_log_height + self.config.fri.log_blowup, leaf_sum)]);

                    self.verify_iopp_query_whir(
                        iopp_oracles.as_slice(),
                        query_points[i],
                        &leaf_sums,
                        query,
                        &iopp_log_foldings,
                        &folding_challenges,
                        &[],
                        &whir_opening_point,
                        &combined_f_r,
                        final_codeword.as_deref(),
                    )
                    .is_ok()
                })
        };

        if !all_queries_valid {
            return Err(WhirError::FriFinalStepMisMatch);
        }
        Ok(())
    }
}
