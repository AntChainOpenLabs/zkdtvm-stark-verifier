use core::borrow::BorrowMut;
use std::sync::Arc;

use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    config::F,
    system_dt::{RecursionRecord, SpecSpongeBlock},
    transcript_dt::sponge::columns::{TranscriptSpongeCols, NUM_TRANSCRIPT_SPONGE_COLS},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct TranscriptSpongeTraceGenerator;

impl TranscriptSpongeTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        transcript_sponge_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_row_major(record: &RecursionRecord) -> RowMajorMatrix<F> {
        let height = Self::trace_height(record);
        let rows = transcript_sponge_rows_cached(record);
        let mut trace = Vec::with_capacity(height * NUM_TRANSCRIPT_SPONGE_COLS);

        if rows.is_empty() {
            trace.extend(padding_row());
        } else {
            for row in rows.iter() {
                trace.extend(trace_row(row));
            }
            for _ in rows.len()..height {
                trace.extend(padding_row());
            }
        }

        RowMajorMatrix::new(trace, NUM_TRANSCRIPT_SPONGE_COLS)
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let rows = transcript_sponge_rows_cached(record);
        if rows.is_empty() {
            let main = RowMajorMatrix::new(padding_row(), NUM_TRANSCRIPT_SPONGE_COLS);
            return CompressedMatrix::new(main, PaddingRow::None, 1);
        }

        if crate::env_var("DT_NATIVE_D10_ALIGN").is_ok() {
            // Optional diagnostic: report whether transcript-limb window offsets
            // are uniform per consumer class (kept for schedule-alignment
            // investigations).
            let locator = crate::system_dt::spec_sponge::SpongeWindowLocator::from_blocks(&rows);
            let items = record.proof_records.iter().flat_map(|proof| {
                proof.whir.round_rows.iter().flat_map(move |row| {
                    (0..33usize).filter_map(move |slot| {
                        let mut offset = slot as isize;
                        if slot >= 8 {
                            offset -= if row.is_round { 8 } else { 0 };
                            offset += if row.round_has_oracle { 8 } else { 0 };
                        }
                        let t = row.tidx as isize + offset;
                        (t >= 0).then_some((slot, t as usize))
                    })
                })
            });
            crate::system_dt::spec_sponge::d10_alignment_census("WhirRound", &locator, items);
        }
        let height = Self::trace_height(record);
        let mut trace = Vec::with_capacity(rows.len() * NUM_TRANSCRIPT_SPONGE_COLS);
        for row in rows.iter() {
            trace.extend(trace_row(row));
        }

        let main = RowMajorMatrix::new(trace, NUM_TRANSCRIPT_SPONGE_COLS);
        let padding =
            if rows.len() < height { PaddingRow::General(padding_row()) } else { PaddingRow::None };
        CompressedMatrix::new(main, padding, height)
    }
}

pub fn transcript_sponge_rows(record: &RecursionRecord) -> Vec<SpecSpongeBlock> {
    transcript_sponge_rows_cached(record).as_ref().to_vec()
}

pub fn transcript_sponge_rows_cached(record: &RecursionRecord) -> Arc<[SpecSpongeBlock]> {
    Arc::clone(
        record
            .tracegen_artifacts
            .transcript_sponge
            .get_or_init(|| transcript_sponge_rows_uncached(record).into()),
    )
}

fn transcript_sponge_rows_uncached(record: &RecursionRecord) -> Vec<SpecSpongeBlock> {
    record
        .proof_records
        .iter()
        .flat_map(|proof| {
            assert!(
                !proof.transcript.sponge_blocks.is_empty(),
                "proof {} has no source-captured transcript sponge rows",
                proof.proof_idx
            );
            proof.transcript.sponge_blocks.clone()
        })
        .collect()
}

pub fn transcript_sponge_row_count(record: &RecursionRecord) -> usize {
    if let Some(rows) = record.tracegen_artifacts.transcript_sponge.get() {
        return rows.len();
    }
    record
        .proof_records
        .iter()
        .map(|proof| {
            assert!(
                !proof.transcript.sponge_blocks.is_empty(),
                "proof {} has no source-captured transcript sponge rows",
                proof.proof_idx
            );
            proof.transcript.sponge_blocks.len()
        })
        .sum()
}

fn padding_row() -> Vec<F> {
    vec![F::zero(); NUM_TRANSCRIPT_SPONGE_COLS]
}

pub fn trace_row(row: &SpecSpongeBlock) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_TRANSCRIPT_SPONGE_COLS];
    let cols: &mut TranscriptSpongeCols<F> = values.as_mut_slice().borrow_mut();

    cols.proof_idx = F::from_canonical_usize(row.proof_idx);
    cols.is_proof_start = F::from_bool(row.is_proof_start);
    cols.is_proof_last = F::from_bool(row.is_proof_last);
    cols.is_valid = F::one();
    cols.tidx = F::from_canonical_usize(row.tidx);
    cols.prev_rate = row.prev_rate;
    cols.input16 = row.input16;
    cols.output16 = row.output16;
    cols.absorb_mask = row.absorb_mask.map(F::from_bool);
    cols.squeeze_mask = row.squeeze_mask.map(F::from_bool);
    cols.prev_s_count = F::from_canonical_usize(row.prev_s_count);

    values
}
