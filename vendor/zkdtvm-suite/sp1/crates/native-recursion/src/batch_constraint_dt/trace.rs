use core::borrow::BorrowMut;

use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_field::{AbstractExtensionField, AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    batch_constraint_dt::columns::{
        BatchSumcheckCols, BatchTranscriptInputsCols, BATCH_INTERP_MATRIX, BATCH_SUMCHECK_EVALS,
        NUM_BATCH_SUMCHECK_COLS, NUM_BATCH_TRANSCRIPT_INPUTS_COLS,
    },
    config::{D_EF, EF, F},
    system_dt::{RecursionRecord, RecursionSumcheckRoundRecord},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchTranscriptInputsTraceGenerator;

impl BatchTranscriptInputsTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        // The PolyAir finalize path indexes equality coefficients by
        // `log_height - 1`, so a one-row chip is unsupported. Fusing the
        // transcript inputs makes a single-proof native node exactly one
        // logical row; keep that row stored and pad its domain to height two.
        batch_transcript_input_row_count(record).max(2).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = batch_transcript_input_row_count(record);
        let mut values = zeroed_trace_values(row_count, NUM_BATCH_TRANSCRIPT_INPUTS_COLS);
        let mut trace_rows = values[..row_count * NUM_BATCH_TRANSCRIPT_INPUTS_COLS]
            .chunks_exact_mut(NUM_BATCH_TRANSCRIPT_INPUTS_COLS);
        visit_batch_transcript_input_rows(record, |row| {
            fill_input_row(trace_rows.next().expect("batch transcript row count"), row);
        });
        debug_assert!(trace_rows.next().is_none());
        compressed_values(
            values,
            NUM_BATCH_TRANSCRIPT_INPUTS_COLS,
            row_count.max(2).next_power_of_two(),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchSumcheckTraceGenerator;

impl BatchSumcheckTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        batch_sumcheck_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = batch_sumcheck_row_count(record);
        let width = NUM_BATCH_SUMCHECK_COLS;
        let mut values = zeroed_trace_values(row_count, width);
        let mut trace_rows = values[..row_count * width].chunks_exact_mut(width);
        visit_batch_sumcheck_rows(record, |row| {
            fill_sumcheck_row(trace_rows.next().expect("batch sumcheck row count"), row);
        });
        debug_assert!(trace_rows.next().is_none());
        compressed_values(values, width, row_count.max(1).next_power_of_two())
    }
}

#[derive(Debug, Clone)]
pub enum BatchTranscriptInputRow {
    Fused {
        proof_idx: usize,
        c_chips: usize,
        perm_alpha: [F; D_EF],
        perm_beta: [F; D_EF],
        alpha: [F; D_EF],
    },
}

#[derive(Debug, Clone)]
pub enum BatchSumcheckRow {
    Seed {
        proof_idx: usize,
        num_public_values: usize,
        num_rounds: usize,
        c_chips: usize,
        summary_id_base: usize,
    },
    Round {
        proof_idx: usize,
        num_public_values: usize,
        num_rounds: usize,
        c_chips: usize,
        round: RecursionSumcheckRoundRecord,
        eq_challenge: [F; D_EF],
    },
}

fn batch_transcript_input_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().filter(|proof| !proof.batch_constraint.is_empty()).count()
}

fn visit_batch_transcript_input_rows(
    record: &RecursionRecord,
    mut visit: impl FnMut(BatchTranscriptInputRow),
) {
    for proof in record.proof_records.iter().filter(|proof| !proof.batch_constraint.is_empty()) {
        let batch = &proof.batch_constraint;
        visit(BatchTranscriptInputRow::Fused {
            proof_idx: proof.proof_idx,
            c_chips: batch.c_chips,
            perm_alpha: batch.perm_alpha,
            perm_beta: batch.perm_beta,
            alpha: batch.alpha,
        });
    }
}

pub fn batch_transcript_input_rows(record: &RecursionRecord) -> Vec<BatchTranscriptInputRow> {
    let mut rows = Vec::with_capacity(batch_transcript_input_row_count(record));
    visit_batch_transcript_input_rows(record, |row| rows.push(row));
    rows
}

fn batch_sumcheck_row_count(record: &RecursionRecord) -> usize {
    record
        .proof_records
        .iter()
        .filter(|proof| !proof.batch_constraint.is_empty())
        .map(|proof| 1 + proof.batch_constraint.rounds.len())
        .sum()
}

fn visit_batch_sumcheck_rows(record: &RecursionRecord, mut visit: impl FnMut(BatchSumcheckRow)) {
    for proof in record.proof_records.iter().filter(|proof| !proof.batch_constraint.is_empty()) {
        let batch = &proof.batch_constraint;
        visit(BatchSumcheckRow::Seed {
            proof_idx: proof.proof_idx,
            num_public_values: batch.num_public_values,
            num_rounds: batch.num_rounds,
            c_chips: batch.c_chips,
            summary_id_base: proof.proof_shape.segment_id_base(),
        });
        for round in &batch.rounds {
            let opening_idx = batch.num_rounds - 1 - round.round_idx;
            visit(BatchSumcheckRow::Round {
                proof_idx: proof.proof_idx,
                num_public_values: batch.num_public_values,
                num_rounds: batch.num_rounds,
                c_chips: batch.c_chips,
                round: *round,
                eq_challenge: batch.eq_challenges[opening_idx],
            });
        }
    }
}

pub fn batch_sumcheck_rows(record: &RecursionRecord) -> Vec<BatchSumcheckRow> {
    let mut rows = Vec::with_capacity(batch_sumcheck_row_count(record));
    visit_batch_sumcheck_rows(record, |row| rows.push(row));
    rows
}

fn fill_input_row(values: &mut [F], row: BatchTranscriptInputRow) {
    debug_assert_eq!(values.len(), NUM_BATCH_TRANSCRIPT_INPUTS_COLS);
    let cols: &mut BatchTranscriptInputsCols<F> = values.borrow_mut();
    match row {
        BatchTranscriptInputRow::Fused { proof_idx, c_chips, perm_alpha, perm_beta, alpha } => {
            cols.proof_idx = f(proof_idx);
            cols.is_valid = F::one();
            cols.c_chips = f(c_chips);
            cols.event_values[..D_EF].copy_from_slice(&perm_alpha);
            cols.event_values[D_EF..2 * D_EF].copy_from_slice(&perm_beta);
            cols.event_values[2 * D_EF..3 * D_EF].copy_from_slice(&alpha);
        }
    }
}

#[cfg(test)]
pub(crate) fn input_row(row: BatchTranscriptInputRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_BATCH_TRANSCRIPT_INPUTS_COLS];
    fill_input_row(&mut values, row);
    values
}

fn fill_sumcheck_row(values: &mut [F], row: BatchSumcheckRow) {
    debug_assert_eq!(values.len(), NUM_BATCH_SUMCHECK_COLS);
    let cols: &mut BatchSumcheckCols<F> = values.borrow_mut();
    match row {
        BatchSumcheckRow::Seed { proof_idx, num_rounds, c_chips, summary_id_base, .. } => {
            cols.proof_idx = f(proof_idx);
            cols.is_seed = F::one();
            cols.r_rounds = f(num_rounds);
            cols.c_chips = f(c_chips);
            cols.summary_id_base = f(summary_id_base);
        }
        BatchSumcheckRow::Round { proof_idx, num_rounds, c_chips, round, eq_challenge, .. } => {
            cols.proof_idx = f(proof_idx);
            cols.is_round = F::one();
            cols.round_idx = f(round.round_idx);
            cols.r_rounds = f(num_rounds);
            cols.c_chips = f(c_chips);
            let coefficients = interpolation_coefficients(round.evals);
            for (dst, coefficient) in cols.coefficients.iter_mut().zip(coefficients.iter().skip(1))
            {
                *dst = ext_limbs(coefficient);
            }
            cols.challenge = round.challenge;
            cols.eq_challenge = eq_challenge;
            cols.claim_in = round.claim_in;
            let [acc_3, acc_2, acc_1] = horner_accumulators(round.evals, round.challenge);
            cols.acc_3 = acc_3;
            cols.acc_2 = acc_2;
            cols.acc_1 = acc_1;
            cols.claim_out = round.claim_out;
        }
    }
}

#[cfg(test)]
pub(crate) fn sumcheck_row(row: BatchSumcheckRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_BATCH_SUMCHECK_COLS];
    fill_sumcheck_row(&mut values, row);
    values
}

/// Horner accumulators for the production degree-four round unipoly.
pub fn horner_accumulators(
    evals: [[F; D_EF]; BATCH_SUMCHECK_EVALS],
    challenge: [F; D_EF],
) -> [[F; D_EF]; 3] {
    let coeffs = interpolation_coefficients(evals);
    let r = EF::from_base_slice(&challenge);
    let acc_3 = coeffs[3] + r * coeffs[4];
    let acc_2 = coeffs[2] + r * acc_3;
    let acc_1 = coeffs[1] + r * acc_2;
    [ext_limbs(&acc_3), ext_limbs(&acc_2), ext_limbs(&acc_1)]
}

pub fn interpolation_coefficients(
    evals: [[F; D_EF]; BATCH_SUMCHECK_EVALS],
) -> [EF; BATCH_SUMCHECK_EVALS] {
    let values = evals.iter().map(|limbs| EF::from_base_slice(limbs)).collect::<Vec<_>>();
    core::array::from_fn(|row| {
        if row == 0 {
            values[0]
        } else {
            lin_comb(&values, &BATCH_INTERP_MATRIX[row])
        }
    })
}

fn lin_comb(values: &[EF], coeffs: &[(i64, u32)]) -> EF {
    debug_assert_eq!(values.len(), coeffs.len());
    let mut acc = EF::zero();
    for (value, &(num, den)) in values.iter().zip(coeffs.iter()) {
        acc += *value * rational(num, den);
    }
    acc
}

fn rational(num: i64, den: u32) -> F {
    let abs = F::from_canonical_u64(num.unsigned_abs());
    let signed = if num < 0 { -abs } else { abs };
    signed * F::from_canonical_u32(den).inverse()
}

fn ext_limbs(value: &EF) -> [F; D_EF] {
    value.as_base_slice().try_into().expect("active extension degree is D_EF")
}

fn zeroed_trace_values(row_count: usize, width: usize) -> Vec<F> {
    vec![F::zero(); row_count.max(1) * width]
}

fn compressed_values(values: Vec<F>, width: usize, height: usize) -> CompressedMatrix<F> {
    let main = RowMajorMatrix::new(values, width);
    let padding = if main.height() < height {
        PaddingRow::General(vec![F::zero(); width])
    } else {
        PaddingRow::None
    };
    CompressedMatrix::new(main, padding, height)
}

fn f(value: usize) -> F {
    F::from_canonical_usize(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::borrow::Borrow;

    use dt_stark::sumcheck::types::UniPolyEvals;
    use p3_field::{AbstractExtensionField, AbstractField};

    use crate::system_dt::{
        RecursionBatchConstraintRecord, RecursionProofRecord, RecursionSumcheckRoundRecord,
    };

    #[test]
    fn interpolation_matrix_matches_nodes_zero_to_four() {
        let evals = core::array::from_fn(|eval_idx| {
            core::array::from_fn(|limb_idx| F::from_canonical_usize(10 * eval_idx + limb_idx + 1))
        });
        let coeffs = interpolation_coefficients(evals);
        for node in 0..BATCH_SUMCHECK_EVALS {
            let x = EF::from_base(F::from_canonical_usize(node));
            let mut powers = EF::one();
            let mut value = EF::zero();
            for coeff in coeffs {
                value += coeff * powers;
                powers *= x;
            }
            assert_eq!(ext_limbs(&value), evals[node]);
        }
    }

    #[test]
    fn horner_eval_matches_verifier_unipoly_eval_at_point() {
        let evals = core::array::from_fn(|eval_idx| {
            core::array::from_fn(|limb_idx| {
                F::from_canonical_usize(17 + 11 * eval_idx + limb_idx * limb_idx)
            })
        });
        let challenge: [F; D_EF] =
            core::array::from_fn(|limb_idx| F::from_canonical_usize(3 + 7 * limb_idx));
        let coeffs = interpolation_coefficients(evals);
        let r = EF::from_base_slice(&challenge);
        let ours = coeffs[0] + r * (coeffs[1] + r * (coeffs[2] + r * (coeffs[3] + r * coeffs[4])));
        let verifier =
            UniPolyEvals::new(evals.iter().map(|limbs| EF::from_base_slice(limbs)).collect())
                .eval_at_point(r);
        assert_eq!(ours, verifier);
    }

    #[test]
    fn compact_writer_round_trips_claims_and_horner_state() {
        let evals: [[F; D_EF]; BATCH_SUMCHECK_EVALS] = core::array::from_fn(|eval_idx| {
            core::array::from_fn(|limb_idx| {
                F::from_canonical_usize(31 + 17 * eval_idx + 7 * eval_idx * eval_idx + 3 * limb_idx)
            })
        });
        let challenge =
            core::array::from_fn(|limb_idx| F::from_canonical_usize(41 + 13 * limb_idx));
        let claim_in = core::array::from_fn(|limb_idx| evals[0][limb_idx] + evals[1][limb_idx]);
        let r = EF::from_base_slice(&challenge);
        let claim_out = ext_limbs(
            &UniPolyEvals::new(evals.iter().map(|limbs| EF::from_base_slice(limbs)).collect())
                .eval_at_point(r),
        );
        let values = sumcheck_row(BatchSumcheckRow::Round {
            proof_idx: 9,
            num_public_values: 0,
            num_rounds: 1,
            c_chips: 1,
            round: RecursionSumcheckRoundRecord {
                round_idx: 0,
                evals,
                challenge,
                claim_in,
                claim_out,
            },
            eq_challenge: [F::zero(); D_EF],
        });
        let cols: &BatchSumcheckCols<F> = values.as_slice().borrow();

        assert_eq!(cols.claim_in, claim_in);
        assert_eq!(cols.claim_out, claim_out);
        assert_eq!([cols.acc_3, cols.acc_2, cols.acc_1], horner_accumulators(evals, challenge));
        assert_eq!(values.len(), NUM_BATCH_SUMCHECK_COLS);
    }

    #[test]
    fn fused_inputs_are_one_row_per_proof_and_pad_to_height_two() {
        let record = sample_batch_record();
        assert_eq!(batch_transcript_input_rows(&record).len(), 1);
        assert_eq!(BatchTranscriptInputsTraceGenerator::trace_height(&record), 2);
        let trace = BatchTranscriptInputsTraceGenerator::generate_trace_compressed(&record);
        assert_eq!(trace.stored_height(), 1);
        assert_eq!(trace.total_height, 2);

        let values = input_row(batch_transcript_input_rows(&record).remove(0));
        let cols: &BatchTranscriptInputsCols<F> = values.as_slice().borrow();
        assert_eq!(&cols.event_values[..D_EF], &[F::from_canonical_u32(11); D_EF]);
        assert_eq!(&cols.event_values[D_EF..2 * D_EF], &[F::from_canonical_u32(13); D_EF]);
        assert_eq!(&cols.event_values[2 * D_EF..], &[F::from_canonical_u32(17); D_EF]);
    }

    #[test]
    fn sumcheck_uses_seed_plus_rounds_without_tail_and_reverses_openings() {
        let record = sample_batch_record();
        let rows = batch_sumcheck_rows(&record);
        assert_eq!(rows.len(), 4);
        assert!(matches!(rows[0], BatchSumcheckRow::Seed { .. }));
        for (round_idx, row) in rows[1..].iter().enumerate() {
            let BatchSumcheckRow::Round { eq_challenge, .. } = row else {
                panic!("expected round row");
            };
            assert_eq!(
                *eq_challenge,
                record.proof_records[0].batch_constraint.eq_challenges[2 - round_idx]
            );
        }
        assert_eq!(BatchSumcheckTraceGenerator::trace_height(&record), 4);
    }

    fn sample_batch_record() -> RecursionRecord {
        let rounds = (0..3)
            .map(|round_idx| RecursionSumcheckRoundRecord {
                round_idx,
                challenge: [F::from_canonical_usize(round_idx + 1); D_EF],
                ..Default::default()
            })
            .collect();
        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 2,
                batch_constraint: RecursionBatchConstraintRecord {
                    num_public_values: 0,
                    num_rounds: 3,
                    c_chips: 1,
                    perm_alpha: [F::from_canonical_u32(11); D_EF],
                    perm_beta: [F::from_canonical_u32(13); D_EF],
                    alpha: [F::from_canonical_u32(17); D_EF],
                    eq_challenges: vec![
                        [F::from_canonical_u32(19); D_EF],
                        [F::from_canonical_u32(23); D_EF],
                        [F::from_canonical_u32(29); D_EF],
                    ],
                    rounds,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}
