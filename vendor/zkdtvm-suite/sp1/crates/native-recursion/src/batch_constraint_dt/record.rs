use crate::{
    batch_constraint_dt::columns::{
        batch_active_shape_limbs, batch_seed_prefix_limbs, BATCH_COMMITMENT_LIMBS,
        BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS, BATCH_ROUND_EVENT_LIMBS, BATCH_SUMCHECK_EVALS,
    },
    child_views::NativeChildViews,
    config::{D_EF, F},
    system_dt::{
        RecursionBatchConstraintRecord, RecursionBatchCumSumRecord, RecursionRecord,
        RecursionSumcheckRoundRecord, RecursionTranscriptEvent, RecursionTranscriptEventKind,
    },
};
use dt_stark::{sumcheck::config::SCStarkGenericConfig, Challenge as StarkChallenge};
use p3_field::{AbstractExtensionField, AbstractField};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchConstraintRecordError {
    InvalidRoundShape,
    UnsupportedNonlinearRounds {
        num_rounds_nonlinear: usize,
    },
    SumcheckRoundCountMismatch {
        expected: usize,
        actual: usize,
    },
    SumcheckEvalLengthMismatch {
        round_idx: usize,
        expected: usize,
        actual: usize,
    },
    MissingTranscriptEvent {
        tidx: usize,
    },
    TranscriptEventTidxMismatch {
        expected: usize,
        actual: usize,
    },
    TranscriptEventKindMismatch {
        tidx: usize,
        expected: RecursionTranscriptEventKind,
        actual: RecursionTranscriptEventKind,
    },
    TranscriptEventValueMismatch {
        tidx: usize,
        expected: F,
        actual: F,
    },
}

/// Capture batch/sumcheck material after the sequential transcript walk. The
/// narrow claim chain is produced in the same pass because dependency-heavy
/// terminal construction consumes it. No sumcheck identity is checked here.
pub fn record_batch_constraint_materials_from_views<SC>(
    record: &mut RecursionRecord,
    proof_idx: usize,
    views: &NativeChildViews<'_, SC>,
    publish_opening_point: bool,
    publish_terminal_outputs: bool,
) -> Result<(), BatchConstraintRecordError>
where
    SC: SCStarkGenericConfig<Val = F>,
    StarkChallenge<SC>: AbstractExtensionField<F>,
{
    let proof = views.proof.proof();
    let num_public_values = views.layout.num_observed_public_values();
    let c_chips = views.proof.chip_count();
    let verifier_log_height = views
        .proof
        .verifier_round_log_height()
        .map_err(|_| BatchConstraintRecordError::InvalidRoundShape)?;
    let round_shape = views
        .verifier_config
        .round_shape(verifier_log_height)
        .map_err(|_| BatchConstraintRecordError::InvalidRoundShape)?;
    if round_shape.num_rounds_nonlinear != 0 {
        return Err(BatchConstraintRecordError::UnsupportedNonlinearRounds {
            num_rounds_nonlinear: round_shape.num_rounds_nonlinear,
        });
    }
    let num_rounds = round_shape.num_rounds;
    let unipolys = &proof.sumcheck_proof.unipolys;
    if unipolys.len() != num_rounds {
        return Err(BatchConstraintRecordError::SumcheckRoundCountMismatch {
            expected: num_rounds,
            actual: unipolys.len(),
        });
    }

    let events = &record.proof_record_mut(proof_idx).transcript.events;
    let layout = BatchTranscriptLayout::new(
        num_public_values,
        c_chips,
        num_rounds,
        views.layout.contains_global_bus(),
    );

    let mut cum_sums = Vec::with_capacity(c_chips);
    for (chip_idx, opening) in proof.opened_values.chips.iter().enumerate() {
        let lcs = ext_limbs(&opening.local_cumulative_sum);
        for (offset, value) in lcs.iter().copied().enumerate() {
            expect_event(
                events,
                layout.e6_tidx(chip_idx) + offset,
                RecursionTranscriptEventKind::Observe,
                value,
            )?;
        }
        cum_sums.push(RecursionBatchCumSumRecord { chip_idx, lcs });
    }

    let perm_alpha =
        read_ext_events(events, layout.e3_tidx(), RecursionTranscriptEventKind::Sample)?;
    let perm_beta =
        read_ext_events(events, layout.e3_tidx() + D_EF, RecursionTranscriptEventKind::Sample)?;
    let alpha = read_ext_events(events, layout.e7_tidx(), RecursionTranscriptEventKind::Sample)?;

    let mut eq_challenges = Vec::with_capacity(num_rounds);
    for round_idx in 0..num_rounds {
        eq_challenges.push(read_ext_events(
            events,
            layout.e8_tidx(round_idx),
            RecursionTranscriptEventKind::Sample,
        )?);
    }

    let mut rounds = Vec::with_capacity(num_rounds);
    let mut claim = StarkChallenge::<SC>::zero();
    for (round_idx, unipoly) in unipolys.iter().enumerate() {
        if unipoly.evals.len() != BATCH_SUMCHECK_EVALS {
            return Err(BatchConstraintRecordError::SumcheckEvalLengthMismatch {
                round_idx,
                expected: BATCH_SUMCHECK_EVALS,
                actual: unipoly.evals.len(),
            });
        }
        let mut evals = [[F::zero(); D_EF]; BATCH_SUMCHECK_EVALS];
        let e9_tidx = layout.e9_tidx(round_idx);
        for eval_idx in 0..BATCH_SUMCHECK_EVALS {
            evals[eval_idx] = ext_limbs(&unipoly.evals[eval_idx]);
            for limb_idx in 0..D_EF {
                expect_event(
                    events,
                    e9_tidx + eval_idx * D_EF + limb_idx,
                    RecursionTranscriptEventKind::Observe,
                    evals[eval_idx][limb_idx],
                )?;
            }
        }

        let challenge = read_ext_events(
            events,
            e9_tidx + BATCH_SUMCHECK_EVALS * D_EF,
            RecursionTranscriptEventKind::Sample,
        )?;
        let challenge_ext = StarkChallenge::<SC>::from_base_slice(&challenge);
        let claim_in = ext_limbs(&claim);
        claim = unipoly.eval_at_point(challenge_ext);
        rounds.push(RecursionSumcheckRoundRecord {
            round_idx,
            evals,
            challenge,
            claim_in,
            claim_out: ext_limbs(&claim),
        });
    }

    let batch_constraint = RecursionBatchConstraintRecord {
        num_public_values,
        num_rounds,
        c_chips,
        cum_sums,
        perm_alpha,
        perm_beta,
        alpha,
        eq_challenges,
        rounds,
        last_claim: ext_limbs(&claim),
        publish_opening_point,
        publish_terminal_outputs,
    };
    record.proof_record_mut(proof_idx).batch_constraint = batch_constraint;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchTranscriptLayout {
    pub num_public_values: usize,
    pub c_chips: usize,
    pub num_rounds: usize,
    pub seed_prefix_limbs: usize,
}

impl BatchTranscriptLayout {
    pub const fn new(
        num_public_values: usize,
        c_chips: usize,
        num_rounds: usize,
        contains_global_bus: bool,
    ) -> Self {
        Self {
            num_public_values,
            c_chips,
            num_rounds,
            seed_prefix_limbs: batch_seed_prefix_limbs(contains_global_bus),
        }
    }

    pub const fn e2_base(&self) -> usize {
        self.seed_prefix_limbs + self.num_public_values + BATCH_COMMITMENT_LIMBS
    }

    pub const fn e3_tidx(&self) -> usize {
        self.e2_base() + batch_active_shape_limbs(self.c_chips)
    }

    pub const fn e6_base(&self) -> usize {
        self.e3_tidx() + BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS
    }

    pub const fn e6_tidx(&self, chip_idx: usize) -> usize {
        self.e6_base() + D_EF * chip_idx
    }

    pub const fn e7_tidx(&self) -> usize {
        self.e6_base() + D_EF * self.c_chips
    }

    pub const fn e8_base(&self) -> usize {
        self.e7_tidx() + D_EF
    }

    pub const fn e8_tidx(&self, round_idx: usize) -> usize {
        self.e8_base() + D_EF * round_idx
    }

    pub const fn e9_base(&self) -> usize {
        self.e8_base() + D_EF * self.num_rounds
    }

    pub const fn e9_tidx(&self, round_idx: usize) -> usize {
        self.e9_base() + BATCH_ROUND_EVENT_LIMBS * round_idx
    }
}

fn read_ext_events(
    events: &[RecursionTranscriptEvent],
    tidx_base: usize,
    kind: RecursionTranscriptEventKind,
) -> Result<[F; D_EF], BatchConstraintRecordError> {
    let mut values = [F::zero(); D_EF];
    for (idx, value) in values.iter_mut().enumerate() {
        *value = expect_event(events, tidx_base + idx, kind, None)?;
    }
    Ok(values)
}

fn expect_event(
    events: &[RecursionTranscriptEvent],
    tidx: usize,
    kind: RecursionTranscriptEventKind,
    expected_value: impl Into<Option<F>>,
) -> Result<F, BatchConstraintRecordError> {
    let event =
        events.get(tidx).ok_or(BatchConstraintRecordError::MissingTranscriptEvent { tidx })?;
    if event.tidx != tidx {
        return Err(BatchConstraintRecordError::TranscriptEventTidxMismatch {
            expected: tidx,
            actual: event.tidx,
        });
    }
    if event.kind != kind {
        return Err(BatchConstraintRecordError::TranscriptEventKindMismatch {
            tidx,
            expected: kind,
            actual: event.kind,
        });
    }
    if let Some(expected) = expected_value.into() {
        if event.value != expected {
            return Err(BatchConstraintRecordError::TranscriptEventValueMismatch {
                tidx,
                expected,
                actual: event.value,
            });
        }
    }
    Ok(event.value)
}

fn ext_limbs<EF>(value: &EF) -> [F; D_EF]
where
    EF: AbstractExtensionField<F>,
{
    value.as_base_slice().try_into().expect("active extension degree is D_EF")
}
