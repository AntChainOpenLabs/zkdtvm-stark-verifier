use std::collections::BTreeMap;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_fri::{BatchOpening, CommitPhaseProofStep, PrunedFriQueryProof, QueryProof};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::whir::profile;
use crate::whir::whir_types::{
    CoefficientsByHeight, WhirError, WhirIoppRoundQuery, WhirPcs, WhirVerifiedLeafStep,
    WhirVerifiedQuery, WhirVerifiedQueryFoldStep,
};

fn fold_codeword<EF: TwoAdicField>(codeword: &[EF], beta: EF) -> Vec<EF> {
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

fn fold_codeword_block<EF: TwoAdicField>(
    block: &[EF],
    block_index: usize,
    remaining_log_folding: usize,
    log_row_height: usize,
    beta: EF,
) -> Vec<EF> {
    debug_assert!(remaining_log_folding > 0);
    debug_assert_eq!(block.len(), 1usize << remaining_log_folding);
    let half = block.len() / 2;
    let log_current = log_row_height + remaining_log_folding;
    let log_folded_height = log_current - 1;
    let generator = EF::two_adic_generator(log_current);
    let row_factor = if log_row_height == 0 {
        EF::one()
    } else {
        generator.exp_u64(reverse_bits_len(block_index, log_row_height) as u64)
    };

    (0..half)
        .map(|i| {
            let local_exp = reverse_bits_len(i, remaining_log_folding - 1) << log_row_height;
            let g = row_factor * generator.exp_u64(local_exp as u64);
            debug_assert_eq!(
                g,
                generator.exp_u64(reverse_bits_len(
                    (block_index << (remaining_log_folding - 1)) | i,
                    log_folded_height
                ) as u64)
            );
            let slope = (block[2 * i + 1] - block[2 * i]) / (-g - g);
            let intercept = block[2 * i] - slope * g;
            intercept + slope * beta
        })
        .collect()
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn open_iopp_row_full(
        &self,
        commit: &FriMmcs::ProverData<p3_matrix::dense::RowMajorMatrix<EF>>,
        index: usize,
        log_folding: usize,
    ) -> WhirIoppRoundQuery<EF, FriMmcs> {
        WhirIoppRoundQuery {
            current_opening: self.open_iopp_step_full(commit, index, log_folding),
            next_opening: None,
        }
    }

    pub(crate) fn open_iopp_step_full(
        &self,
        commit: &FriMmcs::ProverData<p3_matrix::dense::RowMajorMatrix<EF>>,
        index: usize,
        log_folding: usize,
    ) -> CommitPhaseProofStep<EF, FriMmcs> {
        assert!(log_folding > 0, "log_folding must be positive");
        let width = 1usize << log_folding;
        let row_index = index >> log_folding;
        let local_index = index & (width - 1);
        let (mut opened_rows, opening_proof) = profile::time("open.whir_iopp_mmcs_open_ms", || {
            self.config.fri.mmcs.open_batch(row_index, commit)
        });
        assert_eq!(opened_rows.len(), 1);
        let opened_row = opened_rows.pop().unwrap();
        assert_eq!(
            opened_row.len(),
            width,
            "Committed data should match the requested folding factor"
        );
        CommitPhaseProofStep {
            sibling_value: opened_row[local_index ^ 1],
            opened_values: opened_row,
            opening_proof,
        }
    }

    pub(crate) fn verify_iopp_step_full(
        &self,
        commitment: &FriMmcs::Commitment,
        index: usize,
        current_codeword_log: usize,
        log_folding: usize,
        step: &CommitPhaseProofStep<EF, FriMmcs>,
    ) -> Result<Vec<EF>, WhirError<FriMmcs::Error, InputMmcs::Error>> {
        if log_folding == 0 || current_codeword_log < log_folding {
            return Err(WhirError::InvalidInputError);
        }
        let row_width = 1usize << log_folding;
        let log_row_height = current_codeword_log - log_folding;
        let row_index = index >> log_folding;
        let local_index = index & (row_width - 1);
        if step.opened_values.len() != row_width
            || step.sibling_value != step.opened_values[local_index ^ 1]
        {
            return Err(WhirError::InvalidInputError);
        }

        self.config
            .fri
            .mmcs
            .verify_batch(
                commitment,
                &[Dimensions {
                    width: row_width,
                    height: 1 << log_row_height,
                }],
                row_index,
                std::slice::from_ref(&step.opened_values),
                &step.opening_proof,
            )
            .map_err(WhirError::CommitPhaseMmcsError)?;

        Ok(step.opened_values.clone())
    }

    pub(crate) fn fold_opened_iopp_row(
        &self,
        mut opened_row: Vec<EF>,
        row_index: usize,
        log_folding: usize,
        log_row_height: usize,
        folding_challenges: &[EF],
    ) -> Result<EF, WhirError<FriMmcs::Error, InputMmcs::Error>> {
        if log_folding == 0 || folding_challenges.len() != log_folding {
            return Err(WhirError::InvalidInputError);
        }
        if opened_row.len() != (1usize << log_folding) {
            return Err(WhirError::InvalidInputError);
        }
        for local_round in 0..log_folding {
            opened_row = fold_codeword_block(
                &opened_row,
                row_index,
                log_folding - local_round,
                log_row_height,
                folding_challenges[local_round],
            );
        }
        opened_row
            .first()
            .copied()
            .ok_or(WhirError::InvalidInputError)
    }

    /// WHIR IOPP query verification.
    ///
    /// Like `verify_iopp_query_p3` but uses the whir merge formula:
    /// `F_new = eq_factor * F + merge_beta * G` instead of interpolation.
    pub fn verify_iopp_query_whir(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        mut query_point: usize,
        leaf_sums_by_log_height: &BTreeMap<usize, EF>,
        query_proof: &QueryProof<EF, FriMmcs>,
        commit_log_foldings: &[usize],
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<
        (
            Vec<WhirVerifiedQueryFoldStep<EF>>,
            EF,
            Vec<FriMmcs::VerificationTrace>,
        ),
        WhirError<FriMmcs::Error, InputMmcs::Error>,
    > {
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.config.fri.log_blowup;

        // [F-019] The commit-phase fold loop zips `iopp_commitments` with
        // `query_proof.commit_phase_openings`. A short openings vector would
        // truncate the zip and leave folding rounds to be (mis)handled by the
        // tail/final-codeword path, bypassing the committed-round structure.
        // The caller pins `iopp_commitments.len()` to the trusted schedule, so
        // require the per-query openings to match it exactly.
        if query_proof.commit_phase_openings.len() != iopp_commitments.len() {
            return Err(WhirError::InvalidInputError);
        }

        let mut folded_eval = EF::zero();
        let mut merge_idx: usize = 0;
        let mut eq_factor = EF::one();
        let mut height_iter = leaf_sums_by_log_height.iter().rev().peekable();
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());
        let mut consumed_rounds = 0usize;
        let mut verified_steps = Vec::with_capacity(num_vars);
        let mut merkle_traces = Vec::with_capacity(iopp_commitments.len().min(num_vars));

        for (opening_idx, (commitment, opening)) in iopp_commitments
            .iter()
            .zip(&query_proof.commit_phase_openings)
            .enumerate()
        {
            if consumed_rounds >= num_vars {
                break;
            }
            let log_folding = commit_log_foldings.get(opening_idx).copied().unwrap_or(1);
            if log_folding == 0 || consumed_rounds + log_folding > num_vars {
                return Err(WhirError::InvalidInputError);
            }

            let current_codeword_log = log_max_height - consumed_rounds;
            let log_row_height = current_codeword_log
                .checked_sub(log_folding)
                .ok_or(WhirError::InvalidInputError)?;
            let row_width = 1usize << log_folding;
            let row_index = query_point >> log_folding;
            let local_index = query_point & (row_width - 1);

            let log_folded_height = current_codeword_log - 1;

            let folded_before_merge = folded_eval;
            let eq_before_merge = eq_factor;
            let mut merged_leaf = None;
            if let Some((&leaf_height, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                    merged_leaf = Some((leaf_height, leaf_sum, None));
                } else {
                    let merge_beta = merge_betas[merge_idx - 1];
                    folded_eval = eq_factor * folded_eval + merge_beta * leaf_sum;
                    eq_factor = EF::one();
                    merged_leaf = Some((leaf_height, leaf_sum, Some(merge_beta)));
                }
                merge_idx += 1;
            }

            let mut opened_row = if log_folding == 1 && opening.opened_values.is_empty() {
                let sibling_index = query_point ^ 1;
                let mut pair_evals = vec![folded_eval; 2];
                pair_evals[sibling_index % 2] = opening.sibling_value;
                pair_evals
            } else {
                if opening.opened_values.len() != row_width {
                    return Err(WhirError::InvalidInputError);
                }
                if opening.opened_values[local_index] != folded_eval {
                    return Err(WhirError::FriFinalStepMisMatch);
                }
                opening.opened_values.clone()
            };

            let merkle_trace = self
                .config
                .fri
                .mmcs
                .verify_batch_with_trace(
                    commitment,
                    &[Dimensions {
                        width: row_width,
                        height: 1 << log_row_height,
                    }],
                    row_index,
                    &[opened_row.clone()],
                    &opening.opening_proof,
                )
                .map_err(WhirError::CommitPhaseMmcsError)?;
            merkle_traces.push(merkle_trace);

            let mut local_index = local_index;
            for local_round in 0..log_folding {
                let round = consumed_rounds + local_round;
                let log_folded_height = log_max_height - round - 1;
                if local_round > 0
                    && height_iter
                        .peek()
                        .is_some_and(|(lh, _)| **lh == log_folded_height + 1)
                {
                    return Err(WhirError::InvalidInputError);
                }
                if opened_row[local_index] != folded_eval {
                    return Err(WhirError::FriFinalStepMisMatch);
                }

                let local_pair_index = local_index >> 1;
                let pair_evals = [
                    opened_row[local_pair_index << 1],
                    opened_row[(local_pair_index << 1) | 1],
                ];
                let query_point_in = query_point;
                let folded_value_in = folded_eval;
                let eq_in = eq_factor;

                let pair_index = query_point >> 1;
                let generator = EF::two_adic_generator(log_folded_height + 1)
                    .exp_u64(reverse_bits_len(pair_index, log_folded_height) as u64);
                let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
                let intercept = pair_evals[0] - slope * generator;
                folded_eval = intercept + slope * folding_challenges[round];

                if local_round + 1 < log_folding {
                    opened_row = fold_codeword_block(
                        &opened_row,
                        row_index,
                        log_folding - local_round,
                        log_row_height,
                        folding_challenges[round],
                    );
                }

                query_point = pair_index;
                local_index >>= 1;

                let var_idx = num_vars - 1 - round;
                let p_i = opening_point[var_idx];
                let fc_i = folding_challenges[round];
                eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
                verified_steps.push(WhirVerifiedQueryFoldStep {
                    round,
                    query_point_in,
                    query_point_out: query_point,
                    pair: pair_evals,
                    generator,
                    folding_challenge: folding_challenges[round],
                    folded_before_merge,
                    folded_value_in,
                    folded_value_out: folded_eval,
                    eq_before_merge,
                    eq_in,
                    eq_out: eq_factor,
                    merged_leaf: merged_leaf.take(),
                });
            }
            consumed_rounds += log_folding;
        }

        for round in consumed_rounds..num_vars {
            let log_folded_height = log_max_height - round - 1;

            let folded_before_merge = folded_eval;
            let eq_before_merge = eq_factor;
            let mut merged_leaf = None;
            if let Some((&leaf_height, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                    merged_leaf = Some((leaf_height, leaf_sum, None));
                } else {
                    let merge_beta = merge_betas[merge_idx - 1];
                    folded_eval = eq_factor * folded_eval + merge_beta * leaf_sum;
                    eq_factor = EF::one();
                    merged_leaf = Some((leaf_height, leaf_sum, Some(merge_beta)));
                }
                merge_idx += 1;
            }

            let codeword = virtual_codeword
                .as_ref()
                .ok_or(WhirError::FriFinalStepMisMatch)?;
            let pair_index = query_point >> 1;
            let even_idx = pair_index << 1;
            let pair_evals = [codeword[even_idx], codeword[even_idx | 1]];
            let query_point_in = query_point;
            let folded_value_in = folded_eval;
            let eq_in = eq_factor;

            if round == consumed_rounds && folded_eval != pair_evals[query_point & 1] {
                return Err(WhirError::FriFinalStepMisMatch);
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
            verified_steps.push(WhirVerifiedQueryFoldStep {
                round,
                query_point_in,
                query_point_out: query_point,
                pair: pair_evals,
                generator,
                folding_challenge: folding_challenges[round],
                folded_before_merge,
                folded_value_in,
                folded_value_out: folded_eval,
                eq_before_merge,
                eq_in,
                eq_out: eq_factor,
                merged_leaf,
            });
        }

        if folded_eval != *expected_final_value {
            return Err(WhirError::FinalPolyMismatch);
        }
        Ok((verified_steps, folded_eval, merkle_traces))
    }

    /// Path-pruned batched IOPP verification for WHIR-style openings.
    pub fn verify_queries_iopp_p3_pruned_whir(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        query_points: &[usize],
        leaf_sums_by_log_height_per_query: &[BTreeMap<usize, EF>],
        iopp_pruned: &PrunedFriQueryProof<EF, FriMmcs>,
        commit_log_foldings: &[usize],
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<(), WhirError<FriMmcs::Error, InputMmcs::Error>> {
        let n = query_points.len();
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.config.fri.log_blowup;
        let mut active_log_foldings = Vec::new();
        let mut scheduled_rounds = 0usize;
        if commit_log_foldings.is_empty() {
            let committed_rounds = iopp_commitments.len().min(num_vars);
            active_log_foldings.resize(committed_rounds, 1);
        } else {
            for &log_folding in commit_log_foldings {
                if scheduled_rounds >= num_vars {
                    break;
                }
                if log_folding == 0 || scheduled_rounds + log_folding > num_vars {
                    return Err(WhirError::InvalidInputError);
                }
                active_log_foldings.push(log_folding);
                scheduled_rounds += log_folding;
            }
        }
        let committed_groups = active_log_foldings.len();

        if iopp_pruned.sibling_values.len() != n
            || iopp_pruned.round_pruned_proofs.len() < committed_groups
            || leaf_sums_by_log_height_per_query.len() != n
        {
            return Err(WhirError::InvalidInputError);
        }
        for sv in &iopp_pruned.sibling_values {
            if sv.len() < committed_groups {
                return Err(WhirError::InvalidInputError);
            }
        }
        if !iopp_pruned.round_opened_values.is_empty()
            && iopp_pruned.round_opened_values.len() < committed_groups
        {
            return Err(WhirError::InvalidInputError);
        }
        if !iopp_pruned.query_to_unique_slot.is_empty()
            && iopp_pruned.query_to_unique_slot.len() < committed_groups
        {
            return Err(WhirError::InvalidInputError);
        }

        let mut folded_evals: Vec<EF> = vec![EF::zero(); n];
        let mut merge_idxs: Vec<usize> = vec![0usize; n];
        let mut eq_factors: Vec<EF> = vec![EF::one(); n];
        let mut query_idxs: Vec<usize> = query_points.to_vec();
        let mut height_iters: Vec<_> = leaf_sums_by_log_height_per_query
            .iter()
            .map(|m| m.iter().rev().peekable())
            .collect();
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());
        let mut consumed_rounds = 0usize;

        for (opening_idx, &log_folding) in active_log_foldings.iter().enumerate() {
            let current_codeword_log = log_max_height - consumed_rounds;
            let log_row_height = current_codeword_log
                .checked_sub(log_folding)
                .ok_or(WhirError::InvalidInputError)?;
            let row_width = 1usize << log_folding;
            let first_log_folded_height = current_codeword_log - 1;

            let mut row_indices: Vec<usize> = Vec::with_capacity(n);
            let mut local_indices: Vec<usize> = Vec::with_capacity(n);
            let mut rows_per_query: Vec<Vec<EF>> = Vec::with_capacity(n);
            let opened_rows_for_round = iopp_pruned
                .round_opened_values
                .get(opening_idx)
                .filter(|rows| !rows.is_empty());

            let mut sorted_unique = Vec::with_capacity(n);
            for &query_idx in &query_idxs {
                let row_index = query_idx >> log_folding;
                sorted_unique.push(row_index);
            }
            sorted_unique.sort_unstable();
            sorted_unique.dedup();

            // [F-017] Bind the proof's embedded pruned indices to OUR
            // transcript-sampled row indices by value. `verify_batch_pruned`
            // authenticates rows at the proof's own `sorted_indices`; without
            // this the prover could authenticate genuine-but-unsampled rows and
            // have us fold them as if they sat at the sampled positions.
            if let Some(recovered) = self
                .config
                .fri
                .mmcs
                .recover_pruned_indices(&iopp_pruned.round_pruned_proofs[opening_idx])
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

            for q in 0..n {
                if let Some((_, &leaf_sum)) =
                    height_iters[q].next_if(|(lh, _)| **lh == first_log_folded_height + 1)
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

                let row_index = query_idxs[q] >> log_folding;
                let local_index = query_idxs[q] & (row_width - 1);
                let slot = sorted_unique
                    .binary_search(&row_index)
                    .map_err(|_| WhirError::InvalidInputError)?;
                if let Some(q2u_round) = iopp_pruned.query_to_unique_slot.get(opening_idx) {
                    if q2u_round.len() != n || q2u_round[q] as usize != slot {
                        return Err(WhirError::InvalidInputError);
                    }
                }

                let row = if let Some(opened_rows) = opened_rows_for_round {
                    if opened_rows.len() != sorted_unique.len()
                        || slot >= opened_rows.len()
                        || opened_rows[slot].len() != 1
                        || opened_rows[slot][0].len() != row_width
                    {
                        return Err(WhirError::InvalidInputError);
                    }
                    if opened_rows[slot][0][local_index] != folded_evals[q] {
                        return Err(WhirError::FriFinalStepMisMatch);
                    }
                    opened_rows[slot][0].clone()
                } else {
                    if log_folding != 1 {
                        return Err(WhirError::InvalidInputError);
                    }
                    let sibling_bit = local_index ^ 1;
                    let mut row = vec![folded_evals[q]; 2];
                    row[sibling_bit] = iopp_pruned.sibling_values[q][opening_idx];
                    row
                };

                rows_per_query.push(row);
                row_indices.push(row_index);
                local_indices.push(local_index);
            }

            let mut row_by_index: Vec<Option<Vec<EF>>> = vec![None; sorted_unique.len()];
            for q in 0..n {
                let slot = sorted_unique
                    .binary_search(&row_indices[q])
                    .map_err(|_| WhirError::InvalidInputError)?;
                match &row_by_index[slot] {
                    None => row_by_index[slot] = Some(rows_per_query[q].clone()),
                    Some(existing) => {
                        if *existing != rows_per_query[q] {
                            return Err(WhirError::InvalidInputError);
                        }
                    }
                }
            }

            let opened_values_per_query: Vec<Vec<Vec<EF>>> = row_by_index
                .iter()
                .map(|opt| {
                    let row = opt.clone().expect("row must be filled");
                    vec![row]
                })
                .collect();

            let dims = [Dimensions {
                width: row_width,
                height: 1 << log_row_height,
            }];
            self.config
                .fri
                .mmcs
                .verify_batch_pruned(
                    &iopp_commitments[opening_idx],
                    &dims,
                    &opened_values_per_query,
                    &iopp_pruned.round_pruned_proofs[opening_idx],
                )
                .map_err(WhirError::CommitPhaseMmcsError)?;

            for q in 0..n {
                let mut opened_row = rows_per_query[q].clone();
                let mut local_index = local_indices[q];
                let row_index = row_indices[q];
                for local_round in 0..log_folding {
                    let round = consumed_rounds + local_round;
                    let log_folded_height = log_max_height - round - 1;
                    if local_round > 0
                        && height_iters[q]
                            .peek()
                            .is_some_and(|(lh, _)| **lh == log_folded_height + 1)
                    {
                        return Err(WhirError::InvalidInputError);
                    }
                    if opened_row[local_index] != folded_evals[q] {
                        return Err(WhirError::FriFinalStepMisMatch);
                    }

                    let local_pair_index = local_index >> 1;
                    let pair_evals = [
                        opened_row[local_pair_index << 1],
                        opened_row[(local_pair_index << 1) | 1],
                    ];

                    let pair_idx = query_idxs[q] >> 1;
                    let generator = EF::two_adic_generator(log_folded_height + 1)
                        .exp_u64(reverse_bits_len(pair_idx, log_folded_height) as u64);
                    let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
                    let intercept = pair_evals[0] - slope * generator;
                    folded_evals[q] = intercept + slope * folding_challenges[round];

                    if local_round + 1 < log_folding {
                        opened_row = fold_codeword_block(
                            &opened_row,
                            row_index,
                            log_folding - local_round,
                            log_row_height,
                            folding_challenges[round],
                        );
                    }

                    query_idxs[q] = pair_idx;
                    local_index >>= 1;

                    let var_idx = num_vars - 1 - round;
                    let p_i = opening_point[var_idx];
                    let fc_i = folding_challenges[round];
                    eq_factors[q] *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
                }
            }
            consumed_rounds += log_folding;
        }

        for round in consumed_rounds..num_vars {
            let log_folded_height = log_max_height - round - 1;
            let codeword = virtual_codeword
                .as_ref()
                .ok_or(WhirError::FriFinalStepMisMatch)?;

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

                if round == consumed_rounds && folded_evals[q] != pair_evals[query_idxs[q] & 1] {
                    return Err(WhirError::FriFinalStepMisMatch);
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

        for q in 0..n {
            if folded_evals[q] != *expected_final_value {
                return Err(WhirError::FinalPolyMismatch);
            }
        }
        Ok(())
    }

    /// WHIR batch query verification.
    pub fn verify_query_p3_batch_whir(
        &self,
        query_idx: usize,
        commitments: &[InputMmcs::Commitment],
        iopp_commitments: &[FriMmcs::Commitment],
        query_point: usize,
        matrices_size_batch: &[Vec<Dimensions>],
        query_proof: &QueryProof<EF, FriMmcs>,
        leaf_openings: &[BatchOpening<F, InputMmcs>],
        coefficients_by_height: &CoefficientsByHeight<EF>,
        alpha: EF,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
        final_codeword: Option<&[EF]>,
    ) -> Result<
        WhirVerifiedQuery<EF, InputMmcs::VerificationTrace, FriMmcs::VerificationTrace>,
        WhirError<FriMmcs::Error, InputMmcs::Error>,
    > {
        let input_merkle = matrices_size_batch
            .iter()
            .zip(commitments.iter().zip(leaf_openings.iter()))
            .map(|(batch_dims, (commitment, opening))| {
                let max_log_height = batch_dims
                    .iter()
                    .map(|dim| log2_strict_usize(dim.height))
                    .max()
                    .unwrap_or(0);

                let codeword_dims: Vec<Dimensions> = batch_dims
                    .iter()
                    .map(|dim| Dimensions {
                        width: 0,
                        height: dim.height << self.config.fri.log_blowup,
                    })
                    .collect();

                self.mmcs
                    .verify_batch_with_trace(
                        commitment,
                        &codeword_dims,
                        query_point >> (folding_challenges.len() - max_log_height),
                        &opening.opened_values,
                        &opening.opening_proof,
                    )
                    .map_err(|_| WhirError::CommitmentCheckFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut leaf_steps = Vec::new();
        let mut leaf_sums_by_log_height = BTreeMap::new();
        for (&log_height, entries) in coefficients_by_height {
            let codeword_height = log_height + self.config.fri.log_blowup;
            let mut accumulator = EF::zero();
            for ((batch_idx, mat_idx), coeffs) in entries {
                let opened = &leaf_openings[*batch_idx].opened_values[*mat_idx];
                if opened.len() != coeffs.len() && opened.len() != coeffs.len() * EF::D {
                    return Err(WhirError::InvalidInputError);
                }
                for (value_idx, coefficient) in coeffs.iter().copied().enumerate() {
                    let value = if opened.len() == coeffs.len() {
                        EF::from_base(opened[value_idx])
                    } else {
                        let start = value_idx * EF::D;
                        EF::from_base_slice(&opened[start..start + EF::D])
                    };
                    let accumulator_in = accumulator;
                    accumulator += coefficient * value;
                    leaf_steps.push(WhirVerifiedLeafStep {
                        log_height: codeword_height,
                        batch_idx: *batch_idx,
                        matrix_idx: *mat_idx,
                        value_idx,
                        value,
                        coefficient,
                        coefficient_out: coefficient * alpha,
                        accumulator_in,
                        accumulator_out: accumulator,
                    });
                }
            }
            leaf_sums_by_log_height.insert(codeword_height, accumulator);
        }

        let (fold_steps, final_value, iopp_merkle) = self.verify_iopp_query_whir(
            iopp_commitments,
            query_point,
            &leaf_sums_by_log_height,
            query_proof,
            &[],
            folding_challenges,
            merge_betas,
            opening_point,
            expected_final_value,
            final_codeword,
        )?;
        Ok(WhirVerifiedQuery {
            query_idx,
            query_point,
            leaf_steps,
            leaf_sums_by_log_height,
            fold_steps,
            final_value,
            input_merkle,
            iopp_merkle,
        })
    }
}
