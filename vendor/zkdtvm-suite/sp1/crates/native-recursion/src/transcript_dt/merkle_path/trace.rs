use core::borrow::BorrowMut;

use dt_stark::sumcheck::trace::{CompressedMatrix, PaddingRow};
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    config::F,
    system_dt::{RecursionMerklePathOp, RecursionMerklePathRow, RecursionRecord},
    transcript_dt::merkle_path::columns::{MerklePathCols, NUM_MERKLE_PATH_COLS},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct MerklePathTraceGenerator;

impl MerklePathTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        merkle_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_row_major(record: &RecursionRecord) -> RowMajorMatrix<F> {
        let height = Self::trace_height(record);
        let mut trace = vec![F::zero(); height * NUM_MERKLE_PATH_COLS];

        for (row, values) in record
            .proof_records
            .iter()
            .flat_map(|proof| proof.merkle_path.rows())
            .zip(trace.chunks_exact_mut(NUM_MERKLE_PATH_COLS))
        {
            fill_trace_row(values, row);
        }

        RowMajorMatrix::new(trace, NUM_MERKLE_PATH_COLS)
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = merkle_row_count(record);
        if row_count == 0 {
            let main = RowMajorMatrix::new(padding_row(), NUM_MERKLE_PATH_COLS);
            return CompressedMatrix::new(main, PaddingRow::None, 1);
        }

        let height = row_count.next_power_of_two();
        let rows: Vec<_> =
            record.proof_records.iter().flat_map(|proof| proof.merkle_path.rows()).collect();
        let mut trace = vec![F::zero(); row_count * NUM_MERKLE_PATH_COLS];
        trace
            .par_chunks_exact_mut(NUM_MERKLE_PATH_COLS)
            .zip(rows.into_par_iter())
            .for_each(|(values, row)| fill_trace_row(values, row));

        let main = RowMajorMatrix::new(trace, NUM_MERKLE_PATH_COLS);
        let padding =
            if row_count < height { PaddingRow::General(padding_row()) } else { PaddingRow::None };
        CompressedMatrix::new(main, padding, height)
    }
}

pub(crate) fn merkle_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = &RecursionMerklePathRow> {
    record.proof_records.iter().flat_map(|proof| proof.merkle_path.rows())
}

pub fn merkle_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.merkle_path.row_count()).sum()
}

fn padding_row() -> Vec<F> {
    vec![F::zero(); NUM_MERKLE_PATH_COLS]
}

pub fn trace_row(row: &RecursionMerklePathRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_MERKLE_PATH_COLS];
    fill_trace_row(&mut values, row);
    values
}

pub(crate) fn fill_trace_row(values: &mut [F], row: &RecursionMerklePathRow) {
    debug_assert_eq!(values.len(), NUM_MERKLE_PATH_COLS);
    let cols: &mut MerklePathCols<F> = values.borrow_mut();

    cols.proof_idx = F::from_canonical_usize(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_leaf_absorb = F::from_bool(matches!(row.op, RecursionMerklePathOp::LeafAbsorb));
    cols.is_inject = F::from_bool(matches!(row.op, RecursionMerklePathOp::InjectCompress));
    cols.is_last = F::from_bool(row.is_last);
    cols.is_leaf_first = F::from_bool(row.is_leaf_first);
    cols.is_leaf_last = F::from_bool(row.is_leaf_last);
    cols.unit_key = F::from_canonical_usize(row.unit_key);
    cols.commit_id = F::from_canonical_usize(row.commit_id);
    cols.level = F::from_canonical_usize(row.level);
    cols.block_idx = if row.is_last {
        F::from_canonical_usize(row.root_cnt).inverse()
    } else {
        F::from_canonical_usize(row.block_idx)
    };
    cols.idx = F::from_canonical_usize(row.cur_idx);
    cols.left_idx = if matches!(row.op, RecursionMerklePathOp::LeafAbsorb) {
        F::from_canonical_usize(row.absorb_cnt).inverse()
    } else {
        F::from_canonical_usize(row.left_idx)
    };
    cols.left_cnt = F::from_canonical_usize(row.left_cnt);
    cols.right_cnt = F::from_canonical_usize(row.right_cnt);
    cols.root_cnt = F::from_canonical_usize(row.root_cnt);
    cols.absorb_cnt = F::from_canonical_usize(row.absorb_cnt);
    cols.prev_state = row.prev_state;
    cols.chunk = row.chunk;
    cols.chunk_mask = row.chunk_mask.map(F::from_bool);
    cols.input = row.input;
    cols.output = row.output;
}
