use core::borrow::{Borrow, BorrowMut};
use std::collections::BTreeMap;

use dt_stark::{
    koalabear_poseidon2::whir_config,
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    batch_constraint_dt::{
        columns::{BATCH_SUMCHECK_EVALS, BATCH_VK_TAG_VERSION_LIMBS},
        trace::{
            batch_sumcheck_rows, batch_transcript_input_rows, BatchSumcheckRow,
            BatchTranscriptInputRow,
        },
        BatchTranscriptLayout,
    },
    child_views::KOALABEAR_MAX_TRACE_LOG_HEIGHT,
    config::{DIGEST_SIZE, D_EF, F, POSEIDON2_WIDTH},
    proof_shape_dt::{
        proof_height_set_rows, proof_shape_binder_rows, trace::ProofShapeBinderRow,
        PROOF_SHAPE_BATCH_MAIN, PROOF_SHAPE_BATCH_PERMUTATION, PROOF_SHAPE_BATCH_PREPROCESSED,
    },
    system_dt::{
        RecursionMerklePathOp, RecursionProofRecord, RecursionRecord, RecursionTranscriptEventKind,
        RecursionWhirBatchEvalRow, RecursionWhirLeafExtStreamRow,
        RecursionWhirLeafExtStreamTraceRow, RecursionWhirLeafStreamRow, RecursionWhirQueryFoldRow,
        RecursionWhirRoundRow, WHIR_BATCH_MAIN, WHIR_BATCH_PERMUTATION,
    },
    transcript_dt::merkle_path::trace::merkle_row_iter,
    whir_dt::columns::{
        whir_unit_key, WhirBatchEvalCols, WhirLeafExtStreamCols, WhirLeafStreamCols,
        WhirQueryFoldCols, WhirRoundCols, WhirSampleBandCols, WhirSampleBandPreprocessedCols,
        WhirTwiddleCols, WhirTwiddlePreprocessedCols, NUM_WHIR_BATCH_EVAL_COLS,
        NUM_WHIR_LEAF_EXT_STREAM_COLS, NUM_WHIR_LEAF_STREAM_COLS, NUM_WHIR_QUERY_FOLD_COLS,
        NUM_WHIR_ROUND_COLS, NUM_WHIR_SAMPLE_BAND_COLS, NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS,
        NUM_WHIR_TWIDDLE_COLS, NUM_WHIR_TWIDDLE_PREPROCESSED_COLS, WHIR_FINAL_ROOT_DIGEST_LANES,
        WHIR_INPUT_PERMUTATION_PATH_SLOT, WHIR_IOPP_ORACLE_PATH_SLOT_BASE,
        WHIR_LEAF_BASE_LIMBS_PER_ROW, WHIR_LEAF_BLOCKS_PER_ROW, WHIR_LEAF_EXT_LIMBS_PER_ROW,
        WHIR_QUERY_PAIR_LEAF_BLOCKS, WHIR_ROLE_COMPRESS, WHIR_ROLE_CORE, WHIR_ROLE_COUNT,
        WHIR_ROLE_SHRINK, WHIR_ROUND_MAX_TRANSCRIPT_EVENTS, WHIR_SAMPLE_BAND_ROWS,
        WHIR_TWIDDLE_ROWS, WHIR_TWIDDLE_TABLES,
    },
};

#[cfg(test)]
use crate::system_dt::RecursionTranscriptEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirRoleConfig {
    pub role_id: usize,
    pub num_queries: usize,
    pub batching_bits: usize,
    pub log_blowup: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirSampleBandConfig {
    pub query_bits: usize,
    pub shift: usize,
    pub high_max: usize,
    pub high_bits: usize,
}

pub fn whir_role_configs() -> [WhirRoleConfig; WHIR_ROLE_COUNT] {
    [
        whir_role_config_from_stage(WHIR_ROLE_CORE, "core"),
        whir_role_config_from_stage(WHIR_ROLE_COMPRESS, "compress"),
        whir_role_config_from_stage(WHIR_ROLE_SHRINK, "shrink"),
    ]
}

pub fn whir_role_config(role_id: usize) -> WhirRoleConfig {
    whir_role_configs()
        .into_iter()
        .find(|config| config.role_id == role_id)
        .expect("WHIR role id must exist in generated role-config table")
}

fn whir_role_config_from_stage(role_id: usize, stage: &'static str) -> WhirRoleConfig {
    let stage_config = whir_config().stage(stage);
    WhirRoleConfig {
        role_id,
        num_queries: stage_config
            .num_queries
            .unwrap_or_else(|| panic!("{stage} WHIR num_queries must be present in JSON config")),
        batching_bits: stage_config.grinding_bits_batching.unwrap_or_else(|| {
            panic!("{stage} WHIR grinding_bits_batching must be present in JSON config")
        }),
        log_blowup: stage_config
            .log_blowup
            .unwrap_or_else(|| panic!("{stage} WHIR log_blowup must be present in JSON config")),
    }
}

pub fn whir_sample_band_rows() -> Vec<WhirSampleBandConfig> {
    (1..=KOALABEAR_MAX_TRACE_LOG_HEIGHT)
        .map(|query_bits| {
            sample_band_for_query_bits(query_bits)
                .expect("sample band domain is bounded by max trace log height")
        })
        .collect()
}

pub fn sample_band_for_query_bits(query_bits: usize) -> Option<WhirSampleBandConfig> {
    if !(1..=KOALABEAR_MAX_TRACE_LOG_HEIGHT).contains(&query_bits) {
        return None;
    }
    let shift = 1usize.checked_shl(query_bits as u32)?;
    let high_max = ((F::ORDER_U64 - 1) >> query_bits) as usize;
    Some(WhirSampleBandConfig { query_bits, shift, high_max, high_bits: bit_width(high_max) })
}

fn bit_width(value: usize) -> usize {
    usize::BITS as usize - value.leading_zeros() as usize
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirTwiddleTraceGenerator;

impl WhirTwiddleTraceGenerator {
    pub const fn trace_height() -> usize {
        WHIR_TWIDDLE_ROWS
    }

    pub fn generate_preprocessed_trace() -> CompressedMatrix<F> {
        let mut values = zeroed_trace_values(WHIR_TWIDDLE_ROWS, NUM_WHIR_TWIDDLE_PREPROCESSED_COLS);
        for (byte, trace_row) in
            values.chunks_exact_mut(NUM_WHIR_TWIDDLE_PREPROCESSED_COLS).enumerate()
        {
            fill_twiddle_preprocessed_row(trace_row, byte);
        }
        compressed_values(values, NUM_WHIR_TWIDDLE_PREPROCESSED_COLS, Self::trace_height())
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let mut values = zeroed_trace_values(WHIR_TWIDDLE_ROWS, NUM_WHIR_TWIDDLE_COLS);
        for (byte, trace_row) in values.chunks_exact_mut(NUM_WHIR_TWIDDLE_COLS).enumerate() {
            fill_twiddle_main_row(trace_row, record, byte);
        }
        compressed_values(values, NUM_WHIR_TWIDDLE_COLS, Self::trace_height())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirSampleBandTraceGenerator;

impl WhirSampleBandTraceGenerator {
    pub const fn trace_height() -> usize {
        WHIR_SAMPLE_BAND_ROWS.next_power_of_two()
    }

    pub fn generate_preprocessed_trace() -> CompressedMatrix<F> {
        let mut values =
            zeroed_trace_values(WHIR_SAMPLE_BAND_ROWS, NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS);
        for (slot, trace_row) in
            values.chunks_exact_mut(NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS).enumerate()
        {
            let config = sample_band_for_query_bits(slot + 1)
                .expect("sample band domain is bounded by max trace log height");
            fill_sample_band_preprocessed_row(trace_row, config);
        }
        compressed_values(values, NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS, Self::trace_height())
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let mut mults = [0u32; WHIR_SAMPLE_BAND_ROWS];
        for row in whir_query_fold_row_iter(record).filter(|row| row.is_seed) {
            if let Some(slot) = row.query_bits.checked_sub(1).filter(|slot| *slot < mults.len()) {
                mults[slot] = mults[slot].saturating_add(1);
            }
        }
        let mut values = zeroed_trace_values(WHIR_SAMPLE_BAND_ROWS, NUM_WHIR_SAMPLE_BAND_COLS);
        for (slot, trace_row) in values.chunks_exact_mut(NUM_WHIR_SAMPLE_BAND_COLS).enumerate() {
            fill_sample_band_main_row(trace_row, mults[slot]);
        }
        compressed_values(values, NUM_WHIR_SAMPLE_BAND_COLS, Self::trace_height())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirRoundTraceGenerator;

impl WhirRoundTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        whir_round_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = whir_round_row_count(record);
        let mut values = zeroed_trace_values(row_count, NUM_WHIR_ROUND_COLS);
        for (trace_row, row) in values[..row_count * NUM_WHIR_ROUND_COLS]
            .chunks_exact_mut(NUM_WHIR_ROUND_COLS)
            .zip(whir_round_row_iter(record))
        {
            fill_round_row(trace_row, row);
        }
        compressed_values(values, NUM_WHIR_ROUND_COLS, row_count.max(1).next_power_of_two())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirBatchEvalTraceGenerator;

impl WhirBatchEvalTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        whir_batch_eval_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = whir_batch_eval_row_count(record);
        let mut values = zeroed_trace_values(row_count, NUM_WHIR_BATCH_EVAL_COLS);
        for (trace_row, row) in values[..row_count * NUM_WHIR_BATCH_EVAL_COLS]
            .chunks_exact_mut(NUM_WHIR_BATCH_EVAL_COLS)
            .zip(whir_batch_eval_row_iter(record))
        {
            fill_batch_eval_row(trace_row, row);
        }
        compressed_values(values, NUM_WHIR_BATCH_EVAL_COLS, row_count.max(1).next_power_of_two())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirQueryFoldTraceGenerator;

impl WhirQueryFoldTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        whir_query_fold_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = whir_query_fold_row_count(record);
        let mut values = zeroed_trace_values(row_count, NUM_WHIR_QUERY_FOLD_COLS);
        for (trace_row, row) in values[..row_count * NUM_WHIR_QUERY_FOLD_COLS]
            .chunks_exact_mut(NUM_WHIR_QUERY_FOLD_COLS)
            .zip(whir_query_fold_row_iter(record))
        {
            fill_query_fold_row(trace_row, row);
        }
        compressed_values(values, NUM_WHIR_QUERY_FOLD_COLS, row_count.max(1).next_power_of_two())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirLeafStreamTraceGenerator;

impl WhirLeafStreamTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        whir_leaf_stream_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = whir_leaf_stream_row_count(record);
        let rows: Vec<_> = whir_leaf_stream_row_iter(record).collect();
        let mut values = zeroed_trace_values(row_count, NUM_WHIR_LEAF_STREAM_COLS);
        values[..row_count * NUM_WHIR_LEAF_STREAM_COLS]
            .par_chunks_exact_mut(NUM_WHIR_LEAF_STREAM_COLS)
            .zip(rows.into_par_iter())
            .for_each(|(trace_row, row)| fill_leaf_stream_row(trace_row, row));
        compressed_values(values, NUM_WHIR_LEAF_STREAM_COLS, row_count.max(1).next_power_of_two())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WhirLeafExtStreamTraceGenerator;

impl WhirLeafExtStreamTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        whir_leaf_ext_stream_row_count(record).max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let row_count = whir_leaf_ext_stream_row_count(record);
        let rows: Vec<_> = whir_leaf_ext_stream_trace_row_iter(record).collect();
        let mut values = zeroed_trace_values(row_count, NUM_WHIR_LEAF_EXT_STREAM_COLS);
        values[..row_count * NUM_WHIR_LEAF_EXT_STREAM_COLS]
            .par_chunks_exact_mut(NUM_WHIR_LEAF_EXT_STREAM_COLS)
            .zip(rows.into_par_iter())
            .for_each(|(trace_row, (proof_idx, row))| {
                fill_leaf_ext_stream_trace_row(trace_row, proof_idx, row)
            });
        compressed_values(
            values,
            NUM_WHIR_LEAF_EXT_STREAM_COLS,
            row_count.max(1).next_power_of_two(),
        )
    }
}

pub(crate) fn whir_round_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = &RecursionWhirRoundRow> {
    record.proof_records.iter().flat_map(|proof| proof.whir.round_rows.iter())
}

pub fn whir_round_rows(record: &RecursionRecord) -> Vec<RecursionWhirRoundRow> {
    whir_round_row_iter(record).copied().collect()
}

fn whir_round_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.whir.round_rows.len()).sum()
}

pub fn whir_batch_eval_rows(record: &RecursionRecord) -> Vec<RecursionWhirBatchEvalRow> {
    whir_batch_eval_row_iter(record).copied().collect()
}

fn whir_batch_eval_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = &RecursionWhirBatchEvalRow> {
    record.proof_records.iter().flat_map(|proof| proof.whir.batch_eval_rows.iter())
}

fn whir_batch_eval_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.whir.batch_eval_rows.len()).sum()
}

pub fn whir_query_fold_rows(record: &RecursionRecord) -> Vec<RecursionWhirQueryFoldRow> {
    whir_query_fold_row_iter(record).copied().collect()
}

fn whir_query_fold_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = &RecursionWhirQueryFoldRow> {
    record.proof_records.iter().flat_map(|proof| proof.whir.query_fold_rows.iter())
}

fn whir_query_fold_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.whir.query_fold_rows.len()).sum()
}

pub fn whir_leaf_stream_rows(record: &RecursionRecord) -> Vec<RecursionWhirLeafStreamRow> {
    whir_leaf_stream_row_iter(record).copied().collect()
}

fn whir_leaf_stream_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = &RecursionWhirLeafStreamRow> {
    record.proof_records.iter().flat_map(|proof| proof.whir.leaf_stream_rows.iter())
}

fn whir_leaf_stream_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.whir.leaf_stream_rows.len()).sum()
}

pub fn whir_leaf_ext_stream_rows(record: &RecursionRecord) -> Vec<RecursionWhirLeafExtStreamRow> {
    whir_leaf_ext_stream_row_iter(record).collect()
}

fn whir_leaf_ext_stream_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = RecursionWhirLeafExtStreamRow> + '_ {
    whir_leaf_ext_stream_trace_row_iter(record)
        .map(|(proof_idx, row)| row.to_semantic_row(proof_idx))
}

fn whir_leaf_ext_stream_trace_row_iter(
    record: &RecursionRecord,
) -> impl Iterator<Item = (usize, &RecursionWhirLeafExtStreamTraceRow)> {
    record.proof_records.iter().flat_map(|proof| {
        proof.whir.leaf_ext_stream_rows.iter().map(move |row| (proof.proof_idx, row))
    })
}

fn whir_leaf_ext_stream_row_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().map(|proof| proof.whir.leaf_ext_stream_rows.len()).sum()
}

fn fill_twiddle_preprocessed_row(values: &mut [F], byte: usize) {
    debug_assert_eq!(values.len(), NUM_WHIR_TWIDDLE_PREPROCESSED_COLS);
    let cols: &mut WhirTwiddlePreprocessedCols<F> = values.borrow_mut();
    cols.byte = f(byte);
    cols.values = core::array::from_fn(|table_id| twiddle_value(table_id, byte));
}

pub fn twiddle_value(table_id: usize, byte: usize) -> F {
    debug_assert!(table_id < WHIR_TWIDDLE_TABLES);
    debug_assert!(byte < WHIR_TWIDDLE_ROWS);
    let exponent = (byte as u64) << (8 * table_id);
    F::two_adic_generator(24).exp_u64(exponent)
}

fn fill_twiddle_main_row(values: &mut [F], record: &RecursionRecord, byte: usize) {
    debug_assert_eq!(values.len(), NUM_WHIR_TWIDDLE_COLS);
    let cols: &mut WhirTwiddleCols<F> = values.borrow_mut();
    for proof in &record.proof_records {
        for table_id in 0..WHIR_TWIDDLE_TABLES {
            let mult = proof.whir.twiddle_mults.get(byte).map(|row| row[table_id]).unwrap_or(0);
            cols.mults[table_id] += f_u32(mult);
        }
    }
}

fn fill_sample_band_preprocessed_row(values: &mut [F], config: WhirSampleBandConfig) {
    debug_assert_eq!(values.len(), NUM_WHIR_SAMPLE_BAND_PREPROCESSED_COLS);
    let cols: &mut WhirSampleBandPreprocessedCols<F> = values.borrow_mut();
    cols.query_bits = f(config.query_bits);
    cols.shift = f(config.shift);
    cols.high_max = f(config.high_max);
    cols.high_bits = f(config.high_bits);
}

fn fill_sample_band_main_row(values: &mut [F], mult: u32) {
    debug_assert_eq!(values.len(), NUM_WHIR_SAMPLE_BAND_COLS);
    let cols: &mut WhirSampleBandCols<F> = values.borrow_mut();
    cols.mult = f_u32(mult);
}

pub(crate) fn fill_round_row(values: &mut [F], row: &RecursionWhirRoundRow) {
    debug_assert_eq!(values.len(), NUM_WHIR_ROUND_COLS);
    let cols: &mut WhirRoundCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_pow_batch = f_bool(row.is_pow_batch);
    cols.is_preamble = f_bool(row.is_preamble);
    cols.is_round = f_bool(row.is_round);
    cols.is_final = f_bool(row.is_final);
    cols.is_final_perm = f_bool(row.is_final_perm);
    cols.final_root_perm_step_flags = row.final_root_perm_step_flags.map(f_bool);
    cols.round = f(row.round);
    cols.tidx = f(row.tidx);
    cols.query_bits = f(row.query_bits);
    cols.r_rounds = f(row.r_rounds);
    cols.c_chips = f(row.c_chips);
    cols.w_qbase = f(row.w_qbase);
    cols.opening_idx = f(row.opening_idx);
    cols.opening_point = row.opening_point;
    cols.height_group_rank = f(row.height_group_rank);
    cols.height_group_log_height = f(row.height_group_log_height);
    cols.group_claim_log_height = f(row.group_claim_log_height);
    cols.group_claim = row.group_claim;
    cols.commit_id = f(row.commit_id);
    cols.commit_root = row.commit_root;
    cols.event_value =
        core::array::from_fn(
            |idx| if idx < 32 { row.event_value[idx] } else { row.event_value_last },
        );
    cols.pow_sample_high = f(row.pow_sample_high);
    cols.round_has_oracle = f_bool(row.round_has_oracle);
    cols.chain_recv_round = f(row.chain_recv_round);
    cols.chain_recv_tidx = f(row.chain_recv_tidx);
    cols.chain_recv_claim = row.chain_recv_claim;
    cols.chain_recv_eq = row.chain_recv_eq;
    cols.chain_recv_pending_is_merge = f_bool(row.chain_recv_pending_is_merge);
    cols.chain_recv_pending_beta = row.chain_recv_pending_beta;
    cols.chain_recv_pending_eq = row.chain_recv_pending_eq;
    cols.chain_send_round = f(row.chain_send_round);
    cols.chain_send_tidx = f(row.chain_send_tidx);
    cols.chain_send_claim = row.chain_send_claim;
    cols.chain_send_eq = row.chain_send_eq;
    cols.chain_send_pending_is_merge = f_bool(row.chain_send_pending_is_merge);
    cols.chain_send_pending_beta = row.chain_send_pending_beta;
    cols.chain_send_pending_eq = row.chain_send_pending_eq;
    cols.r_fold = row.r_fold;
    cols.is_merge = f_bool(row.is_merge);
    cols.emit_prep_seed = f_bool(row.emit_prep_seed);
    cols.merge_log_height = f(row.merge_log_height);
    cols.cfr = row.cfr;
    cols.claim_acc = row.claim_acc;
    cols.claim_folded = row.claim_folded;
    cols.eq_factor = row.eq_factor;
    cols.eq_folded = row.eq_folded;
    cols.bcast_mult = f_u32(row.bcast_mult);
    cols.query_init_mult = f_u32(row.query_init_mult);
    cols.summary_id_base = f(row.summary_id_base);
    cols.commitment_root_send_mult = f_u32(row.commitment_root_send_mult);
    cols.final_root_poseidon2_recv_mult = f_u32(row.final_root_poseidon2_recv_mult);
    if row.is_final_perm {
        write_final_root_recv_state(cols, row.final_root_poseidon2_input);
        write_final_root_poseidon2_output(cols, row.final_root_poseidon2_output);
        let send_state = final_root_next_state(
            row.cfr,
            row.log_blowup,
            final_root_perm_step(row),
            row.final_root_poseidon2_input,
            row.final_root_poseidon2_output,
        );
        write_final_root_send_state(cols, send_state);
    }
    if row.is_final {
        write_final_root_recv_state(cols, row.final_root_poseidon2_output);
        write_final_root_send_state(cols, final_root_seed_state(row.cfr));
    }
}

fn round_row(row: impl Borrow<RecursionWhirRoundRow>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_WHIR_ROUND_COLS];
    fill_round_row(&mut values, row.borrow());
    values
}

pub(crate) fn fill_batch_eval_row(values: &mut [F], row: &RecursionWhirBatchEvalRow) {
    debug_assert_eq!(values.len(), NUM_WHIR_BATCH_EVAL_COLS);
    let cols: &mut WhirBatchEvalCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_start = f_bool(row.is_start);
    cols.is_group_end = f_bool(row.is_group_end);
    cols.cursor = f(row.cursor);
    cols.chain_recv_cursor = f(row.chain_recv_cursor);
    cols.chain_send_cursor = f(row.chain_send_cursor);
    cols.chain_recv_log_height = f(row.chain_recv_log_height);
    cols.chain_recv_batch_id = f(row.chain_recv_batch_id);
    cols.chain_recv_batch_pos = f(row.chain_recv_batch_pos);
    cols.chain_recv_value_idx = f(row.chain_recv_value_idx);
    cols.chain_recv_segment_element_count = f(row.chain_recv_segment_element_count);
    cols.alpha_tidx = f(row.alpha_tidx);
    cols.alpha = row.alpha;
    cols.pow_in = row.pow_in;
    cols.acc_in = row.acc_in;
    cols.group_base_in = row.group_base_in;
    cols.pow_out = row.pow_out;
    cols.acc_out = row.acc_out;
    cols.group_base_out = row.group_base_out;
    cols.value = row.value;
    cols.log_height = f(row.log_height);
    cols.batch_id = f(row.batch_id);
    cols.batch_pos = f(row.batch_pos);
    cols.chip_idx = f(row.chip_idx);
    cols.static_chip_id = f(row.static_chip_id);
    cols.width = f(row.width);
    cols.value_idx = f(row.value_idx);
    cols.segment_element_count = f(row.segment_element_count);
    cols.is_value = f_bool(row.is_value);
    cols.is_segment_start = f_bool(row.is_segment_start);
    cols.is_segment_end = f_bool(row.is_segment_end);
    cols.is_first_value = f_bool(row.is_first_value);
    cols.is_group_start = f_bool(row.is_group_start);
    cols.is_perm_batch = f_bool(row.is_perm_batch);
    cols.group_log_height_gap = f(row.group_log_height_gap);
    cols.batch_dim_recv_mult = f_u32(row.batch_dim_recv_mult);
    cols.opened_eval_send_mult = f_u32(row.opened_eval_send_mult);
    cols.pow_seed_cnt = f_u32(row.pow_seed_cnt);
}

fn batch_eval_row(row: impl Borrow<RecursionWhirBatchEvalRow>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_WHIR_BATCH_EVAL_COLS];
    fill_batch_eval_row(&mut values, row.borrow());
    values
}

pub(crate) fn fill_query_fold_row(values: &mut [F], row: &RecursionWhirQueryFoldRow) {
    debug_assert_eq!(values.len(), NUM_WHIR_QUERY_FOLD_COLS);
    let cols: &mut WhirQueryFoldCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_seed = f_bool(row.is_seed);
    cols.is_round = f_bool(row.is_round);
    cols.query_idx = f(row.query_idx);
    cols.cursor = f(row.cursor);
    cols.w_qbase = f(row.w_qbase);
    cols.query_bits = f(row.query_bits);
    cols.r_rounds = f(row.r_rounds);
    cols.query_sample = row.query_sample;
    cols.query_sample_raw = row.query_sample_raw;
    cols.query_sample_high = f(row.query_sample_high);
    cols.query_sample_shift = f(row.query_sample_shift);
    cols.query_sample_high_max = f(row.query_sample_high_max);
    cols.query_sample_high_bits = f(row.query_sample_high_bits);
    cols.query_sample_high_gap_inv = row.query_sample_high_gap_inv;
    cols.idx = row.idx;
    cols.idx_bit = f_bool(row.idx_bit);
    cols.idx_tail_bit0 = f_bool(row.idx_tail_bit0);
    cols.idx_tail_bit1 = f_bool(row.idx_tail_bit1);
    cols.x = row.x;
    cols.acc = row.acc;
    cols.ipw = row.ipw;
    cols.folded = row.folded;
    cols.f0 = row.f0;
    cols.f1 = row.f1;
    cols.chain_send_cursor = f(row.chain_send_cursor);
    cols.chain_send_idx = row.chain_send_idx;
    cols.chain_send_idx_bit = f_bool(row.chain_send_idx_bit);
    cols.chain_send_x = row.chain_send_x;
    cols.chain_send_acc = row.chain_send_acc;
    cols.chain_send_ipw = row.chain_send_ipw;
    cols.chain_send_folded = row.chain_send_folded;
    cols.r_fold = row.r_fold;
    cols.is_merge = f_bool(row.is_merge);
    cols.is_assign = f_bool(row.is_assign);
    cols.merge_cursor_inv =
        if row.is_merge && row.cursor != 0 { f(row.cursor).inverse() } else { F::zero() };
    cols.merge_beta = row.merge_beta;
    cols.merge_eq = row.merge_eq;
    cols.emit_prep_seed = f_bool(row.emit_prep_seed);
    cols.cfr = row.cfr;
    cols.leaf_sum = row.leaf_sum;
    cols.twiddle_bytes = row.twiddle_bytes.map(|byte| f(byte as usize));
    cols.twiddle_values = row.twiddle_values;
    cols.twiddle_product_01 = row.twiddle_product_01;
}

fn query_fold_row(row: impl Borrow<RecursionWhirQueryFoldRow>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_WHIR_QUERY_FOLD_COLS];
    fill_query_fold_row(&mut values, row.borrow());
    values
}

pub(crate) fn fill_leaf_stream_row(values: &mut [F], row: &RecursionWhirLeafStreamRow) {
    debug_assert_eq!(values.len(), NUM_WHIR_LEAF_STREAM_COLS);
    let cols: &mut WhirLeafStreamCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_unit_start = f_bool(row.is_unit_start);
    cols.is_unit_end = f_bool(row.is_unit_end);
    cols.idx = f(row.idx);
    cols.serve_cnt = f(row.serve_cnt);
    cols.chain_recv_cursor = f(row.chain_recv_cursor);
    cols.chain_send_cursor = f(row.chain_send_cursor);
    cols.log_height = f(row.log_height);
    cols.batch_id = f(row.batch_id);
    cols.chain_recv_log_height = f(row.chain_recv_log_height);
    cols.chain_recv_batch_id = f(row.chain_recv_batch_id);
    cols.is_unit_key_start = f_bool(row.is_unit_key_start);
    cols.unit_key_gap = f(row.unit_key_gap);
    cols.alpha = row.alpha;
    cols.pow_in = row.pow_in;
    cols.acc_in = row.acc_in;
    cols.slot_pows = row.slot_pows;
    cols.pow_out = row.pow_out;
    cols.acc_out = row.acc_out;
    cols.values = row.values;
    cols.chunk_mask = row.chunk_mask.map(f_bool);
    cols.unit_key = f(row.unit_key);
    cols.block_idx = f(row.block_idx);
    debug_assert_eq!(WHIR_LEAF_BASE_LIMBS_PER_ROW, 8);
}

fn leaf_stream_row(row: impl Borrow<RecursionWhirLeafStreamRow>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_WHIR_LEAF_STREAM_COLS];
    fill_leaf_stream_row(&mut values, row.borrow());
    values
}

impl RecursionWhirLeafExtStreamTraceRow {
    fn to_semantic_row(self, proof_idx: usize) -> RecursionWhirLeafExtStreamRow {
        RecursionWhirLeafExtStreamRow {
            proof_idx,
            is_unit_start: false,
            is_unit_end: self.is_unit_end,
            idx: self.idx,
            serve_cnt: self.serve_cnt,
            cursor: self.chain_recv_cursor,
            chain_recv_cursor: self.chain_recv_cursor,
            chain_send_cursor: self.chain_recv_cursor + 1,
            log_height: self.log_height,
            batch_id: WHIR_BATCH_PERMUTATION,
            chain_recv_log_height: self.log_height,
            chain_recv_batch_id: if self.is_unit_key_start {
                WHIR_BATCH_MAIN
            } else {
                WHIR_BATCH_PERMUTATION
            },
            is_unit_key_start: self.is_unit_key_start,
            unit_key_gap: 0,
            alpha: self.alpha,
            pow_in: self.pow_in,
            acc_in: self.acc_in,
            slot_pows: core::array::from_fn(|slot| {
                if slot == 0 {
                    self.pow_in
                } else {
                    self.slot_pows[slot - 1]
                }
            }),
            pow_out: self.pow_out,
            acc_out: self.acc_out,
            value_blocks: self.value_blocks,
            chunk_masks: core::array::from_fn(|block| {
                core::array::from_fn(|idx| {
                    self.element_masks[(block * WHIR_LEAF_BASE_LIMBS_PER_ROW + idx) / D_EF]
                })
            }),
            unit_key: whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, self.log_height),
            block_idx: self.block_idx,
        }
    }
}

fn fill_leaf_ext_stream_trace_row(
    values: &mut [F],
    proof_idx: usize,
    row: &RecursionWhirLeafExtStreamTraceRow,
) {
    debug_assert_eq!(values.len(), NUM_WHIR_LEAF_EXT_STREAM_COLS);
    let cols: &mut WhirLeafExtStreamCols<F> = values.borrow_mut();
    cols.proof_idx = f(proof_idx);
    cols.is_unit_end = f_bool(row.is_unit_end);
    cols.idx = f(row.idx);
    cols.serve_cnt = f(row.serve_cnt);
    cols.chain_recv_cursor = f(row.chain_recv_cursor);
    cols.log_height = f(row.log_height);
    cols.is_unit_key_start = f_bool(row.is_unit_key_start);
    cols.alpha = row.alpha;
    cols.pow_in = row.pow_in;
    cols.acc_in = row.acc_in;
    cols.slot_pows = row.slot_pows;
    cols.pow_out = row.pow_out;
    cols.acc_out = row.acc_out;
    cols.values = core::array::from_fn(|idx| {
        row.value_blocks[idx / WHIR_LEAF_BASE_LIMBS_PER_ROW][idx % WHIR_LEAF_BASE_LIMBS_PER_ROW]
    });
    cols.element_masks = row.element_masks.map(f_bool);
    cols.block_idx = f(row.block_idx);
    debug_assert_eq!(WHIR_LEAF_EXT_LIMBS_PER_ROW, WHIR_LEAF_BLOCKS_PER_ROW * 8);
}

pub(crate) fn fill_leaf_ext_stream_row(values: &mut [F], row: &RecursionWhirLeafExtStreamRow) {
    fill_leaf_ext_stream_trace_row(values, row.proof_idx, &(*row).into());
}

fn leaf_ext_stream_row(row: impl Borrow<RecursionWhirLeafExtStreamRow>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_WHIR_LEAF_EXT_STREAM_COLS];
    fill_leaf_ext_stream_row(&mut values, row.borrow());
    values
}

pub(crate) fn zeroed_trace_values(row_count: usize, width: usize) -> Vec<F> {
    vec![F::zero(); row_count.max(1) * width]
}

pub(crate) fn compressed_values(
    values: Vec<F>,
    width: usize,
    height: usize,
) -> CompressedMatrix<F> {
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

fn f_u32(value: u32) -> F {
    F::from_canonical_u32(value)
}

fn f_bool(value: bool) -> F {
    F::from_bool(value)
}

fn final_root_codeword_limb(cfr: [F; D_EF], idx: usize) -> F {
    cfr[idx % D_EF]
}

fn final_root_seed_state(cfr: [F; D_EF]) -> [F; POSEIDON2_WIDTH] {
    let mut state = [F::zero(); POSEIDON2_WIDTH];
    for (idx, lane) in state.iter_mut().enumerate().take(WHIR_FINAL_ROOT_DIGEST_LANES) {
        *lane = final_root_codeword_limb(cfr, idx);
    }
    state
}

fn duplicated_digest_state(output: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    core::array::from_fn(|idx| output[idx % WHIR_FINAL_ROOT_DIGEST_LANES])
}

fn final_root_next_state(
    combined_f_r: [F; D_EF],
    log_blowup: usize,
    step: usize,
    input: [F; POSEIDON2_WIDTH],
    output: [F; POSEIDON2_WIDTH],
) -> [F; POSEIDON2_WIDTH] {
    match step {
        0 => {
            let mut next = output;
            next[0] = final_root_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES);
            next[1] = final_root_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES + 1);
            next
        }
        1 => {
            if log_blowup == 1 {
                output
            } else {
                duplicated_digest_state(output)
            }
        }
        2 => {
            if log_blowup == 1 {
                input
            } else if log_blowup == 2 {
                output
            } else {
                duplicated_digest_state(output)
            }
        }
        3 => {
            if log_blowup == 3 {
                output
            } else {
                input
            }
        }
        _ => panic!("unsupported WHIR final-root step {step}"),
    }
}

fn final_root_perm_step(row: &RecursionWhirRoundRow) -> usize {
    row.final_root_perm_step_flags.iter().position(|&flag| flag).unwrap_or(0)
}

fn write_final_root_recv_state(cols: &mut WhirRoundCols<F>, state: [F; POSEIDON2_WIDTH]) {
    cols.event_value[..WHIR_FINAL_ROOT_DIGEST_LANES]
        .copy_from_slice(&state[..WHIR_FINAL_ROOT_DIGEST_LANES]);
    cols.r_fold.copy_from_slice(&state[WHIR_FINAL_ROOT_DIGEST_LANES..13]);
    cols.claim_acc[..3].copy_from_slice(&state[13..POSEIDON2_WIDTH]);
}

fn write_final_root_poseidon2_output(cols: &mut WhirRoundCols<F>, output: [F; POSEIDON2_WIDTH]) {
    cols.claim_folded.copy_from_slice(&output[..D_EF]);
    cols.eq_factor.copy_from_slice(&output[D_EF..2 * D_EF]);
    cols.eq_folded.copy_from_slice(&output[2 * D_EF..3 * D_EF]);
    cols.event_value[WHIR_ROUND_MAX_TRANSCRIPT_EVENTS - 1] = output[3 * D_EF];
}

fn write_final_root_send_state(cols: &mut WhirRoundCols<F>, state: [F; POSEIDON2_WIDTH]) {
    cols.event_value[16..32].copy_from_slice(&state);
}

pub type WhirBusResidualReport = BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>;

pub fn whir_bus_residual_report(record: &RecursionRecord) -> WhirBusResidualReport {
    let mut report = WhirBusResidualReport::new();
    let checks: [(&'static str, BTreeMap<Vec<u32>, i64>); 18] = [
        ("1021 WhirTwiddlePow", twiddle_residual(record)),
        ("11 WhirSampleBand", sample_band_residual(record)),
        ("1044 WhirLeafPowSeed", leaf_pow_seed_residual(record)),
        ("1024 WhirGroupClaim", group_claim_residual(record)),
        ("1023 WhirRoundBcast", round_bcast_residual(record)),
        ("1025 WhirQueryLeafSum", query_leaf_sum_residual(record)),
        ("1032 WhirRoundChain", round_chain_residual(record)),
        ("1043 WhirFinalRootChain", final_root_chain_residual(record)),
        ("1026 WhirQueryChain", query_chain_residual(record)),
        ("1030 WhirQueryInit", query_init_residual(record)),
        ("1027 WhirEvalChain", eval_chain_residual(record)),
        ("1028 WhirLeafChain", leaf_chain_residual(record)),
        ("1009 BatchDim", batch_dim_residual(record)),
        ("1022 ProofShapeSummary", summary_residual(record)),
        ("1011 HeightGroup", height_group_residual(record)),
        ("1017 BatchOpeningPoint", opening_point_residual(record)),
        ("1007 TranscriptEvent", transcript_event_residual(record)),
        ("1002 CommitmentRoot", commitment_root_residual(record)),
    ];
    for (name, residual) in checks {
        if !residual.is_empty() {
            report.insert(name, residual);
        }
    }
    let residual = merkle_leaf_block_residual(record);
    if !residual.is_empty() {
        report.insert("MerkleLeafBlock", residual);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{DIGEST_SIZE, D_EF, EF, POSEIDON2_WIDTH, SC},
        machine_dt::{
            build_core_native_recursion_program, build_dual_segment_reduce_program,
            build_native_recursion_program, build_root_shrink_program, core_recording_machine,
            native_recording_machine, native_recording_machine_for_stage,
        },
        proof_shape_dt::{
            PROOF_SHAPE_BATCH_MAIN, PROOF_SHAPE_BATCH_PERMUTATION, PROOF_SHAPE_BATCH_PREPROCESSED,
        },
        statement_dt::{
            NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
            STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        },
        symbolic_expr_fixed_dt::RecursionChildRole,
        system_dt::{
            RecordingStage, RecursionBatchConstraintRecord, RecursionMerklePathRow,
            RecursionNativeProgram, RecursionProofRecord, RecursionProofShapeChip,
            RecursionProofShapeRecord, RecursionRecord, RecursionStatementRole,
            RecursionSumcheckRoundRecord, RecursionTranscriptEventKind, RecursionWhirBatchEvalRow,
            RecursionWhirLeafExtStreamRow, RecursionWhirLeafStreamRow, RecursionWhirQueryFoldRow,
            RecursionWhirRecord, RecursionWhirRoundRow, StatementConfigRow, WhirBatchRlc,
            WhirOpenedMatrices, WhirOpenedMatrix, WhirQueryPairSource, WhirQueryReplayInput,
            WhirQueryRoundControl, WhirSpecFoldSeed, WhirSpecFoldShape, WHIR_BATCH_PERMUTATION,
        },
        transcript_dt::{
            merkle_path::{MerklePathCols, MerklePathTraceGenerator, NUM_MERKLE_PATH_COLS},
            poseidon2::RecursionPoseidon2Memo,
        },
        whir_dt::{
            air::{
                eval_query_fold_historical, lookup_query_fold_mirror, WhirLeafExtStreamAir,
                WhirQueryFoldAir,
            },
            columns::{
                WHIR_INPUT_PERMUTATION_PATH_SLOT, WHIR_LEAF_RLC_SLOTS, WHIR_UNIT_KEY_SLOT_STRIDE,
            },
        },
    };
    use dt_stark::{
        air::{FullAir, FullAirBuilder, MachineAir, PairCol},
        sumcheck::config::SCStarkGenericConfig,
    };
    use p3_air::BaseAir;
    use p3_field::{AbstractExtensionField, AbstractField, Field};
    use p3_matrix::{
        dense::{RowMajorMatrix, RowMajorMatrixView},
        Matrix,
    };
    use polyair::{
        evaluator::ConstraintFolder, permutation::fused_precompute_reserved_permutation,
        prover::SCMachineProver, Chip,
    };
    use std::{borrow::Borrow, ops::Deref};

    #[derive(Debug)]
    struct LeafExtRowEvaluation {
        first: EF,
        nonfirst: EF,
        lookup_multiplicities: Vec<F>,
    }

    #[derive(Debug, Clone, Copy)]
    enum QueryFoldTestRole {
        Current,
        Historical,
        LookupMirror,
    }

    /// Test-only synchronized machine pair. The primary chip compiles either
    /// the current or historical QueryFold relation; the mirror chip compiles
    /// every production lookup with the opposite direction over the identical
    /// trace. Thus proof verification exercises the real permutation argument
    /// instead of disabling cross-chip lookups.
    #[derive(Debug, Clone, Copy)]
    struct QueryFoldSynchronizedAir {
        air: WhirQueryFoldAir,
        role: QueryFoldTestRole,
    }

    impl QueryFoldSynchronizedAir {
        fn new(role: QueryFoldTestRole) -> Self {
            Self { air: WhirQueryFoldAir::default(), role }
        }
    }

    impl<Fld: Field> BaseAir<Fld> for QueryFoldSynchronizedAir {
        fn width(&self) -> usize {
            BaseAir::<Fld>::width(&self.air)
        }
    }

    impl<AB: FullAirBuilder> FullAir<AB> for QueryFoldSynchronizedAir {
        fn width(&self) -> usize {
            FullAir::<AB>::width(&self.air)
        }

        fn required_max_beta_power(&self) -> usize {
            FullAir::<AB>::required_max_beta_power(&self.air)
        }

        fn reserved_poly(&self) -> Vec<PairCol> {
            FullAir::<AB>::reserved_poly(&self.air)
        }

        fn precompute_lc(&self, builder: &mut AB) {
            FullAir::<AB>::precompute_lc(&self.air, builder);
        }

        fn eval(&self, builder: &mut AB) {
            match self.role {
                QueryFoldTestRole::Current => FullAir::<AB>::eval(&self.air, builder),
                QueryFoldTestRole::Historical => eval_query_fold_historical(builder),
                QueryFoldTestRole::LookupMirror => {}
            }
        }

        fn lookup(&self, builder: &mut AB) {
            match self.role {
                QueryFoldTestRole::Current | QueryFoldTestRole::Historical => {
                    FullAir::<AB>::lookup(&self.air, builder);
                }
                QueryFoldTestRole::LookupMirror => lookup_query_fold_mirror(builder),
            }
        }
    }

    impl MachineAir<F> for QueryFoldSynchronizedAir {
        type Record = RecursionRecord;
        type Program = RecursionNativeProgram<F>;

        fn name(&self) -> String {
            match self.role {
                QueryFoldTestRole::Current => "WhirQueryFoldCurrentTest",
                QueryFoldTestRole::Historical => "WhirQueryFoldHistoricalTest",
                QueryFoldTestRole::LookupMirror => "WhirQueryFoldLookupMirrorTest",
            }
            .to_string()
        }

        fn preprocessed_width(&self) -> usize {
            usize::from(matches!(self.role, QueryFoldTestRole::LookupMirror))
        }

        fn preprocessed_num_rows(
            &self,
            _program: &Self::Program,
            _instrs_len: usize,
        ) -> Option<usize> {
            matches!(self.role, QueryFoldTestRole::LookupMirror).then_some(2)
        }

        fn generate_preprocessed_trace(
            &self,
            _program: &Self::Program,
        ) -> Option<CompressedMatrix<F>> {
            matches!(self.role, QueryFoldTestRole::LookupMirror).then(|| {
                CompressedMatrix::new(
                    RowMajorMatrix::new(vec![F::zero(); 2], 1),
                    PaddingRow::None,
                    2,
                )
            })
        }

        fn num_rows(&self, input: &Self::Record) -> Option<usize> {
            Some(WhirQueryFoldTraceGenerator::trace_height(input))
        }

        fn generate_trace(
            &self,
            input: &Self::Record,
            _output: &mut Self::Record,
        ) -> CompressedMatrix<F> {
            WhirQueryFoldTraceGenerator::generate_trace_compressed(input)
        }

        fn included(&self, _record: &Self::Record) -> bool {
            true
        }

        fn local_only(&self) -> bool {
            true
        }
    }

    fn authoritative_leaf_group(
        element_count: usize,
        row_idx: usize,
    ) -> (Vec<RecursionWhirLeafStreamRow>, Vec<RecursionWhirLeafExtStreamRow>) {
        assert!((1..=2 * WHIR_LEAF_RLC_SLOTS).contains(&element_count));
        let values = authoritative_ext_values(element_count, row_idx);
        authoritative_leaf_group_from_values(values, row_idx)
    }

    fn authoritative_ext_values(element_count: usize, row_idx: usize) -> Vec<[F; D_EF]> {
        (0..element_count).map(|slot| ext(800 + row_idx * 17 + slot)).collect()
    }

    fn authoritative_leaf_group_from_values(
        values: Vec<[F; D_EF]>,
        row_idx: usize,
    ) -> (Vec<RecursionWhirLeafStreamRow>, Vec<RecursionWhirLeafExtStreamRow>) {
        let alpha = ext(700);
        let main_opened = ext(750 + row_idx);
        let element_count = values.len();
        assert!(!values.is_empty());
        let opened = WhirOpenedMatrices {
            matrices: vec![
                WhirOpenedMatrix {
                    batch_id: 1,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: 1,
                    log_height: 4,
                    values: vec![main_opened],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PERMUTATION,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: element_count * D_EF,
                    log_height: 4,
                    values: values.clone(),
                },
            ],
        };
        let rlc = WhirBatchRlc::from_opened_matrices(&opened, alpha);
        let mut permutation_opening = Vec::with_capacity(element_count * D_EF);
        for value in values {
            permutation_opening.extend(value);
        }
        let leaf_openings = vec![
            Vec::<Vec<F>>::new(),
            vec![vec![F::from_canonical_usize(751 + row_idx)]],
            vec![permutation_opening],
        ];
        let start_pows = rlc.group_start_pows(1);
        let (base, extension) = rlc
            .leaf_group_stream_rows(0, 5, row_idx, &leaf_openings, 1, start_pows[&5])
            .expect("authoritative base/ext leaf rows");
        assert_eq!(base.len(), 1);
        assert_eq!(extension.len(), element_count.div_ceil(WHIR_LEAF_RLC_SLOTS));
        assert!(base[0].is_unit_start);
        assert!(!extension[0].is_unit_start);
        assert!(extension[0].is_unit_key_start);
        assert_eq!(extension[0].unit_key_gap, 0);
        assert_eq!(extension[0].chain_recv_batch_id, 1);
        for row in &extension[1..] {
            assert!(!row.is_unit_start);
            assert!(!row.is_unit_key_start);
            assert_eq!(row.unit_key_gap, 0);
            assert_eq!(row.chain_recv_batch_id, WHIR_BATCH_PERMUTATION);
        }
        (base, extension)
    }

    fn authoritative_leaf_ext_rows(slot_counts: &[usize]) -> Vec<RecursionWhirLeafExtStreamRow> {
        slot_counts
            .iter()
            .copied()
            .enumerate()
            .map(|(row_idx, slot_count)| {
                let (_, mut extension) = authoritative_leaf_group(slot_count, row_idx);
                extension.pop().expect("one ext row")
            })
            .collect()
    }

    fn leaf_ext_record(rows: Vec<RecursionWhirLeafExtStreamRow>) -> RecursionRecord {
        let mut record = RecursionRecord::default();
        record.proof_records = vec![RecursionProofRecord {
            proof_idx: 0,
            whir: RecursionWhirRecord {
                leaf_ext_stream_rows: rows.into_iter().map(Into::into).collect(),
                ..Default::default()
            },
            ..Default::default()
        }];
        record
    }

    fn query_rows_for_leaf_sum(
        proof_idx: usize,
        query_idx: usize,
        leaf_sum: [F; D_EF],
    ) -> Vec<RecursionWhirQueryFoldRow> {
        let shape = WhirSpecFoldShape {
            role_id: WHIR_ROLE_CORE,
            num_rounds: 1,
            c_chips: 1,
            num_public_values: 0,
            num_queries: 1,
            batching_bits: 0,
            query_bits: 5,
            log_blowup: 4,
            w0_tidx: 100,
        };
        WhirQueryReplayInput {
            seed: WhirSpecFoldSeed { proof_idx, shape, opening_point: vec![[F::zero(); D_EF]] },
            query_idx,
            w_qbase: 500,
            query_sample_raw: F::zero(),
            query_sample: 0,
            controls: vec![WhirQueryRoundControl {
                r_fold: ext(17),
                is_merge: true,
                is_assign: true,
                merge_beta: [F::zero(); D_EF],
                merge_eq: [F::zero(); D_EF],
                emit_prep_seed: false,
                cfr: leaf_sum,
            }],
            pair_source: WhirQueryPairSource::Explicit(vec![(leaf_sum, leaf_sum)]),
            leaf_sums_by_log_height: BTreeMap::from([(5, leaf_sum)]),
        }
        .query_fold_rows()
        .expect("one-round synchronized QueryFold fixture")
    }

    fn query_rows_with_disabled_initial_assignment(
        proof_idx: usize,
        query_idx: usize,
        leaf_sum: [F; D_EF],
    ) -> Vec<RecursionWhirQueryFoldRow> {
        let shape = WhirSpecFoldShape {
            role_id: WHIR_ROLE_CORE,
            num_rounds: 1,
            c_chips: 1,
            num_public_values: 0,
            num_queries: 1,
            batching_bits: 0,
            query_bits: 5,
            log_blowup: 4,
            w0_tidx: 100,
        };
        WhirQueryReplayInput {
            seed: WhirSpecFoldSeed { proof_idx, shape, opening_point: vec![[F::zero(); D_EF]] },
            query_idx,
            w_qbase: 500,
            query_sample_raw: F::zero(),
            query_sample: 0,
            controls: vec![WhirQueryRoundControl {
                r_fold: ext(17),
                is_merge: true,
                is_assign: false,
                merge_beta: [F::zero(); D_EF],
                merge_eq: [F::zero(); D_EF],
                emit_prep_seed: false,
                cfr: [F::zero(); D_EF],
            }],
            pair_source: WhirQueryPairSource::Explicit(vec![(
                [F::zero(); D_EF],
                [F::zero(); D_EF],
            )]),
            leaf_sums_by_log_height: BTreeMap::from([(5, leaf_sum)]),
        }
        .query_fold_rows()
        .expect("one-round QueryFold boundary fixture")
    }

    fn query_gate_evaluations(rows: Vec<RecursionWhirQueryFoldRow>) -> Vec<EF> {
        let mut record = RecursionRecord::default();
        record.proof_records = vec![RecursionProofRecord {
            proof_idx: rows[0].proof_idx,
            whir: RecursionWhirRecord { query_fold_rows: rows, ..Default::default() },
            ..Default::default()
        }];
        let main = WhirQueryFoldTraceGenerator::generate_trace_compressed(&record);
        let chip = Chip::<WhirQueryFoldAir, F, D_EF>::new(WhirQueryFoldAir::default());
        let alpha = EF::from_canonical_u32(211);
        let beta = EF::from_canonical_u32(223) + <EF as AbstractExtensionField<F>>::monomial(1);
        let mut powers = beta.powers();
        let beta_powers = (0..=chip.required_max_beta_power())
            .map(|_| powers.next().expect("infinite beta powers"))
            .collect::<Vec<_>>();
        let beta_septix =
            beta_powers[7] - beta * EF::from_canonical_u32(3) - EF::from_canonical_u32(5);
        let (precomputed, reserved, permutation, local_sum) = fused_precompute_reserved_permutation(
            &chip.air,
            None,
            &main,
            &[],
            alpha,
            &beta_powers,
            beta_septix,
            chip.num_precompute(),
            chip.reserved_poly(),
            chip.logup_batch_size(),
            chip.num_lookup(),
        );
        let reducers =
            (0..chip.num_alpha).map(|idx| EF::from_canonical_usize(307 + idx)).collect::<Vec<_>>();
        (0..reserved.stored_height())
            .map(|row_idx| {
                let precomputed_row = precomputed.main.row_slice(row_idx);
                let reserved_row = reserved.main.row_slice(row_idx);
                let permutation_row = permutation.main.row_slice(row_idx);
                let mut accumulator = EF::zero();
                let mut folder = ConstraintFolder::<F, F, EF> {
                    public: &[],
                    alpha,
                    beta_powers: &beta_powers,
                    beta_septix,
                    precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                    reserved_poly: RowMajorMatrixView::new_row(reserved_row.as_ref()),
                    is_first_row: F::zero(),
                    is_last_row: F::zero(),
                    local_sum,
                    permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                    multiplicitys: Vec::new(),
                    batch_size: chip.logup_batch_size(),
                    accumulator: &mut accumulator,
                    constraint_reducer: &reducers,
                    constraint_index: 0,
                };
                chip.air.eval(&mut folder);
                accumulator
            })
            .collect()
    }

    #[test]
    fn query_fold_round_trace_keeps_seed_only_columns_zero() {
        let rows = query_rows_for_leaf_sum(0, 0, ext(19));
        assert_eq!(rows.len(), 2);

        let record = RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 0,
                whir: RecursionWhirRecord { query_fold_rows: rows, ..Default::default() },
                ..Default::default()
            }],
            ..Default::default()
        };
        let trace = WhirQueryFoldTraceGenerator::generate_trace_compressed(&record);
        assert_eq!(trace.stored_height(), 2);

        let seed_row = trace.main.row_slice(0);
        let round_row = trace.main.row_slice(1);
        let seed: &WhirQueryFoldCols<F> = seed_row.deref().borrow();
        let round: &WhirQueryFoldCols<F> = round_row.deref().borrow();

        assert_eq!(seed.is_seed, F::one());
        assert_eq!(round.is_round, F::one());

        assert_ne!(seed.w_qbase, F::zero());
        assert!([
            seed.query_sample_shift,
            seed.query_sample_high_max,
            seed.query_sample_high_bits,
            seed.query_sample_high_gap_inv,
        ]
        .into_iter()
        .all(|value| value != F::zero()));
        assert!(seed.twiddle_values.iter().all(|value| *value != F::zero()));
        assert_ne!(seed.twiddle_product_01, F::zero());

        assert_eq!(round.w_qbase, F::zero());
        assert_eq!(
            [
                round.query_sample_shift,
                round.query_sample_high_max,
                round.query_sample_high_bits,
                round.query_sample_high_gap_inv,
            ],
            [F::zero(); 4]
        );
        assert_eq!(round.twiddle_values, [F::zero(); WHIR_TWIDDLE_TABLES]);
        assert_eq!(round.twiddle_product_01, F::zero());
    }

    #[test]
    fn query_max_edge_and_x_recurrence_are_enforced_without_deleted_columns() {
        let leaf_sum = ext(19);
        let honest = query_rows_for_leaf_sum(0, 0, leaf_sum);
        assert!(query_gate_evaluations(honest.clone())
            .into_iter()
            .all(|evaluation| evaluation == EF::zero()));

        let mut max_edge = honest.clone();
        let seed = &mut max_edge[0];
        seed.query_sample_high_max = 7;
        seed.query_sample_high = 7;
        seed.query_sample_high_gap_inv = F::zero();
        seed.query_sample = F::zero();
        seed.query_sample_raw = F::from_canonical_usize(7 * seed.query_sample_shift);
        seed.chain_send_idx = F::zero();
        seed.chain_send_idx_bit = false;
        assert!(query_gate_evaluations(max_edge.clone())
            .into_iter()
            .all(|evaluation| evaluation == EF::zero()));

        let invalid_seed = &mut max_edge[0];
        invalid_seed.query_sample = F::one();
        invalid_seed.query_sample_raw += F::one();
        invalid_seed.chain_send_idx = F::one();
        invalid_seed.chain_send_idx_bit = true;
        assert_ne!(query_gate_evaluations(max_edge)[0], EF::zero());

        let mut bad_x = honest;
        bad_x[1].chain_send_x += F::one();
        assert_ne!(query_gate_evaluations(bad_x)[1], EF::zero());
    }

    #[test]
    fn first_merge_assignment_is_exact_in_compiled_query_fold_air_and_lookups() {
        let leaf_sum = ext(901);
        assert_ne!(leaf_sum, [F::zero(); D_EF]);

        let honest_rows = query_rows_for_leaf_sum(0, 0, leaf_sum);
        assert!(query_gate_evaluations(honest_rows.clone())
            .into_iter()
            .all(|evaluation| evaluation == EF::zero()));

        let boundary_rows = query_rows_with_disabled_initial_assignment(0, 0, leaf_sum);
        let boundary_round = boundary_rows[1];
        assert!(boundary_round.is_merge);
        assert!(!boundary_round.is_assign);
        assert_eq!(boundary_round.cursor, 0);
        assert_eq!(boundary_round.leaf_sum, leaf_sum);
        assert_eq!(boundary_round.f0, [F::zero(); D_EF]);
        assert_eq!(boundary_round.f1, [F::zero(); D_EF]);
        assert_ne!(query_gate_evaluations(boundary_rows.clone())[1], EF::zero());

        let honest_record = RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 0,
                whir: RecursionWhirRecord { query_fold_rows: honest_rows, ..Default::default() },
                ..Default::default()
            }],
            ..Default::default()
        };
        let boundary_record = RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 0,
                whir: RecursionWhirRecord { query_fold_rows: boundary_rows, ..Default::default() },
                ..Default::default()
            }],
            ..Default::default()
        };

        let historical_config = SC::compressed();
        let historical_machine = polyair::SCStarkMachine::new(
            historical_config.clone(),
            vec![
                Chip::<QueryFoldSynchronizedAir, F, D_EF>::new(QueryFoldSynchronizedAir::new(
                    QueryFoldTestRole::Historical,
                )),
                Chip::<QueryFoldSynchronizedAir, F, D_EF>::new(QueryFoldSynchronizedAir::new(
                    QueryFoldTestRole::LookupMirror,
                )),
            ],
            0,
            false,
        );
        let historical_prover = polyair::prover::SumcheckProver { machine: historical_machine };
        let (historical_pk, historical_vk) =
            historical_prover.setup(&RecursionNativeProgram::<F>::default());
        let mut historical_prover_challenger = historical_config.mlchallenger();
        let mut historical_verifier_challenger = historical_config.mlchallenger();
        let historical_boundary_proof = historical_prover
            .prove(
                &historical_pk,
                vec![boundary_record.clone()],
                &mut historical_prover_challenger,
                (),
                1,
                0,
            )
            .expect("historical QueryFold relation must produce the counterexample proof");
        historical_prover
            .machine()
            .verify(
                &historical_vk,
                &historical_boundary_proof,
                &mut historical_verifier_challenger,
                1,
                0,
            )
            .expect(
                "historical compiled relation and synchronized lookups accept the counterexample",
            );

        let current_config = SC::compressed();
        let current_machine = polyair::SCStarkMachine::new(
            current_config.clone(),
            vec![
                Chip::<QueryFoldSynchronizedAir, F, D_EF>::new(QueryFoldSynchronizedAir::new(
                    QueryFoldTestRole::Current,
                )),
                Chip::<QueryFoldSynchronizedAir, F, D_EF>::new(QueryFoldSynchronizedAir::new(
                    QueryFoldTestRole::LookupMirror,
                )),
            ],
            0,
            false,
        );
        let current_prover = polyair::prover::SumcheckProver { machine: current_machine };
        let (current_pk, current_vk) =
            current_prover.setup(&RecursionNativeProgram::<F>::default());

        let mut honest_prover_challenger = current_config.mlchallenger();
        let mut honest_verifier_challenger = current_config.mlchallenger();
        let honest_proof = current_prover
            .prove(&current_pk, vec![honest_record], &mut honest_prover_challenger, (), 1, 0)
            .expect("current QueryFold proof for the valid assignment");
        current_prover
            .machine()
            .verify(&current_vk, &honest_proof, &mut honest_verifier_challenger, 1, 0)
            .expect("current compiled relation and synchronized lookups accept the honest witness");

        let mut boundary_prover_challenger = current_config.mlchallenger();
        let mut boundary_verifier_challenger = current_config.mlchallenger();
        let boundary_proof = current_prover
            .prove(&current_pk, vec![boundary_record], &mut boundary_prover_challenger, (), 1, 0)
            .expect("compiled QueryFold proof material for the boundary assignment");
        assert!(
            current_prover
                .machine()
                .verify(&current_vk, &boundary_proof, &mut boundary_verifier_challenger, 1, 0,)
                .is_err(),
            "the exact cursor gadget must reject the counterexample with compiled lookups active"
        );
    }

    fn ext_merkle_blocks(
        rows: &[RecursionWhirLeafExtStreamRow],
    ) -> Vec<(usize, [bool; DIGEST_SIZE], [F; DIGEST_SIZE])> {
        rows.iter()
            .flat_map(|row| {
                row.chunk_masks
                    .iter()
                    .copied()
                    .zip(row.value_blocks.iter().copied())
                    .enumerate()
                    .filter_map(|(block, (mask, chunk))| {
                        mask[0].then_some((row.block_idx + block, mask, chunk))
                    })
            })
            .collect()
    }

    fn base_merkle_blocks(
        rows: &[RecursionWhirLeafStreamRow],
    ) -> Vec<(usize, [bool; DIGEST_SIZE], [F; DIGEST_SIZE])> {
        rows.iter()
            .filter_map(|row| {
                row.chunk_mask[0].then_some((row.block_idx, row.chunk_mask, row.values))
            })
            .collect()
    }

    fn query_merkle_blocks(
        row: RecursionWhirQueryFoldRow,
    ) -> Vec<(usize, [bool; DIGEST_SIZE], [F; DIGEST_SIZE])> {
        (0..WHIR_QUERY_PAIR_LEAF_BLOCKS)
            .map(|block| {
                (
                    block,
                    query_pair_leaf_mask_for_test(block),
                    query_pair_leaf_chunk_for_test(row, block),
                )
            })
            .collect()
    }

    fn one_level_merkle_component(
        proof_idx: usize,
        commit_id: usize,
        unit_key: usize,
        idx: usize,
        blocks: &[(usize, [bool; DIGEST_SIZE], [F; DIGEST_SIZE])],
        absorb_cnt: usize,
        root_cnt: usize,
    ) -> (Vec<RecursionMerklePathRow>, [F; DIGEST_SIZE]) {
        assert!(!blocks.is_empty());
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let mut rows = Vec::with_capacity(blocks.len() + 1);
        let mut prev_state = [F::zero(); POSEIDON2_WIDTH];
        for (position, (block_idx, mask, chunk)) in blocks.iter().copied().enumerate() {
            let row = RecursionMerklePathRow::leaf_absorb(
                proof_idx,
                unit_key,
                commit_id,
                block_idx,
                idx,
                absorb_cnt,
                false,
                position == 0,
                position + 1 == blocks.len(),
                prev_state,
                chunk,
                mask,
                &poseidon2_memo,
            );
            prev_state = row.output;
            rows.push(row);
        }
        let mut root = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            0,
            idx,
            digest_from_poseidon_output(prev_state),
            core::array::from_fn(|lane| F::from_canonical_usize(10_000 + lane)),
            true,
            &poseidon2_memo,
        );
        root.root_cnt = root_cnt;
        let digest = digest_from_poseidon_output(root.output);
        rows.push(root);
        (rows, digest)
    }

    fn synchronized_leaf_record(
        base: Vec<RecursionWhirLeafStreamRow>,
        extension: Vec<RecursionWhirLeafExtStreamRow>,
        query_rows: Vec<RecursionWhirQueryFoldRow>,
        input_merkle: Vec<RecursionMerklePathRow>,
        query_merkle: Vec<RecursionMerklePathRow>,
        input_root: [F; DIGEST_SIZE],
        query_root: [F; DIGEST_SIZE],
    ) -> RecursionRecord {
        let proof_idx = 0;
        let base_row = base.first().expect("synchronized leaf record needs a Base row");
        let (base_merkle, base_root) = one_level_merkle_component(
            proof_idx,
            base_row.batch_id,
            base_row.unit_key,
            base_row.idx,
            &base_merkle_blocks(&base),
            1,
            1,
        );
        let mut query_root_events = RecursionWhirRoundRow::default().event_value;
        query_root_events[..DIGEST_SIZE].copy_from_slice(&query_root);
        let mut proof = RecursionProofRecord {
            proof_idx,
            transcript: crate::system_dt::RecursionTranscriptRecord {
                events: query_root
                    .into_iter()
                    .enumerate()
                    .map(|(tidx, value)| {
                        transcript_event(tidx, RecursionTranscriptEventKind::Observe, value)
                    })
                    .chain(core::iter::once(transcript_event(
                        query_rows[0].w_qbase,
                        RecursionTranscriptEventKind::Sample,
                        query_rows[0].query_sample_raw,
                    )))
                    .collect(),
                ..Default::default()
            },
            whir: RecursionWhirRecord {
                round_rows: vec![
                    RecursionWhirRoundRow {
                        proof_idx,
                        commit_id: base_row.batch_id,
                        commit_root: base_root,
                        commitment_root_send_mult: 1,
                        ..Default::default()
                    },
                    RecursionWhirRoundRow {
                        proof_idx,
                        commit_id: WHIR_BATCH_PERMUTATION,
                        commit_root: input_root,
                        commitment_root_send_mult: 1,
                        ..Default::default()
                    },
                    RecursionWhirRoundRow {
                        proof_idx,
                        is_preamble: true,
                        tidx: 0,
                        commit_id: 100,
                        commit_root: query_root,
                        event_value: query_root_events,
                        commitment_root_send_mult: 1,
                        ..Default::default()
                    },
                ],
                query_fold_rows: query_rows,
                leaf_stream_rows: base,
                leaf_ext_stream_rows: extension.into_iter().map(Into::into).collect(),
                ..Default::default()
            },
            ..Default::default()
        };
        for row in base_merkle.into_iter().chain(input_merkle).chain(query_merkle) {
            proof.merkle_path.push_row(row);
        }
        RecursionRecord { proof_records: vec![proof], ..Default::default() }
    }

    #[derive(Clone)]
    struct SynchronizedLeafFixture {
        base: Vec<RecursionWhirLeafStreamRow>,
        extension: Vec<RecursionWhirLeafExtStreamRow>,
        query_rows: Vec<RecursionWhirQueryFoldRow>,
        input_merkle: Vec<RecursionMerklePathRow>,
        query_merkle: Vec<RecursionMerklePathRow>,
        input_root: [F; DIGEST_SIZE],
        query_root: [F; DIGEST_SIZE],
    }

    fn synchronized_leaf_fixture(values: Vec<[F; D_EF]>) -> SynchronizedLeafFixture {
        let (base, mut extension) = authoritative_leaf_group_from_values(values, 0);
        extension.last_mut().expect("Ext fixture row").serve_cnt = 1;
        let leaf_sum = extension.last().expect("Ext fixture row").acc_out;
        let query_rows = query_rows_for_leaf_sum(0, 0, leaf_sum);
        let input_unit_key = whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, 5);
        let (input_merkle, input_root) = one_level_merkle_component(
            0,
            WHIR_BATCH_PERMUTATION,
            input_unit_key,
            0,
            &ext_merkle_blocks(&extension),
            1,
            1,
        );
        let query_round = query_rows[1];
        let query_unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE, 4);
        let (query_merkle, query_root) = one_level_merkle_component(
            0,
            100,
            query_unit_key,
            query_round.chain_send_idx.as_canonical_u32() as usize,
            &query_merkle_blocks(query_round),
            1,
            1,
        );
        SynchronizedLeafFixture {
            base,
            extension,
            query_rows,
            input_merkle,
            query_merkle,
            input_root,
            query_root,
        }
    }

    fn synchronized_leaf_fixture_record(
        fixture: &SynchronizedLeafFixture,
        authenticated_input_root: [F; DIGEST_SIZE],
        authenticated_query_root: [F; DIGEST_SIZE],
    ) -> RecursionRecord {
        synchronized_leaf_record(
            fixture.base.clone(),
            fixture.extension.clone(),
            fixture.query_rows.clone(),
            fixture.input_merkle.clone(),
            fixture.query_merkle.clone(),
            authenticated_input_root,
            authenticated_query_root,
        )
    }

    fn leaf_ext_materialized_evaluations(main: &CompressedMatrix<F>) -> Vec<LeafExtRowEvaluation> {
        let chip = Chip::<WhirLeafExtStreamAir, F, D_EF>::new(WhirLeafExtStreamAir::default());
        let perm_alpha = EF::from_canonical_u32(211);
        let beta = EF::from_canonical_u32(223) + <EF as AbstractExtensionField<F>>::monomial(1);
        let mut powers = beta.powers();
        let beta_powers = (0..=chip.required_max_beta_power())
            .map(|_| powers.next().expect("infinite beta powers"))
            .collect::<Vec<_>>();
        let beta_septix =
            beta_powers[7] - beta * EF::from_canonical_u32(3) - EF::from_canonical_u32(5);
        let (precomputed, reserved, permutation, local_sum) = fused_precompute_reserved_permutation(
            &chip.air,
            None,
            main,
            &[],
            perm_alpha,
            &beta_powers,
            beta_septix,
            chip.num_precompute(),
            chip.reserved_poly(),
            chip.logup_batch_size(),
            chip.num_lookup(),
        );
        let reducers =
            (0..chip.num_alpha).map(|idx| EF::from_canonical_usize(307 + idx)).collect::<Vec<_>>();
        let reserved_ext = RowMajorMatrix::new(
            reserved.main.values.iter().copied().map(EF::from_base).collect(),
            reserved.main.width(),
        );
        let mut evaluations = Vec::with_capacity(reserved.stored_height());
        for row_idx in 0..reserved.stored_height() {
            let precomputed_row = precomputed.main.row_slice(row_idx);
            let reserved_row = reserved.main.row_slice(row_idx);
            let permutation_row = permutation.main.row_slice(row_idx);
            let mut first_accumulator = EF::zero();
            let mut first = ConstraintFolder::<F, F, EF> {
                public: &[],
                alpha: perm_alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_row.as_ref()),
                is_first_row: F::zero(),
                is_last_row: F::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut first_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut first);
            chip.air.lookup(&mut first);
            let lookup_multiplicities = first.multiplicitys.clone();
            first.constrain_lookup();
            drop(first);

            let reserved_ext_row = reserved_ext.row_slice(row_idx);
            let mut nonfirst_accumulator = EF::zero();
            let mut nonfirst = ConstraintFolder::<F, EF, EF> {
                public: &[],
                alpha: perm_alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_ext_row.as_ref()),
                is_first_row: EF::zero(),
                is_last_row: EF::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut nonfirst_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut nonfirst);
            chip.air.lookup(&mut nonfirst);
            nonfirst.constrain_lookup();
            drop(nonfirst);
            evaluations.push(LeafExtRowEvaluation {
                first: first_accumulator,
                nonfirst: nonfirst_accumulator,
                lookup_multiplicities,
            });
        }

        if reserved.stored_height() < reserved.total_height {
            let row_idx = reserved.stored_height();
            let precomputed_row = precomputed.row_slice(row_idx);
            let reserved_row = reserved.row_slice(row_idx);
            let permutation_row = permutation.row_slice(row_idx);
            let mut accumulator = EF::zero();
            let mut padding = ConstraintFolder::<F, F, EF> {
                public: &[],
                alpha: perm_alpha,
                beta_powers: &beta_powers,
                beta_septix,
                precomputed: RowMajorMatrixView::new_row(precomputed_row.as_ref()),
                reserved_poly: RowMajorMatrixView::new_row(reserved_row.as_ref()),
                is_first_row: F::zero(),
                is_last_row: F::zero(),
                local_sum,
                permutation: RowMajorMatrixView::new_row(permutation_row.as_ref()),
                multiplicitys: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut padding);
            chip.air.lookup(&mut padding);
            assert!(padding.multiplicitys.iter().all(|value| *value == F::zero()));
            padding.constrain_lookup();
            drop(padding);
            assert_eq!(accumulator, EF::zero(), "WhirLeafExtStream padding");
        }
        evaluations
    }

    #[test]
    fn leaf_ext_actual_materialization_covers_slots_lookup_order_and_padding() {
        let rows = authoritative_leaf_ext_rows(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let record = leaf_ext_record(rows.clone());
        let main = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&record);
        let evaluations = leaf_ext_materialized_evaluations(&main);
        assert_eq!(evaluations.len(), rows.len());

        for (row_idx, evaluation) in evaluations.iter().enumerate() {
            let slot_count = row_idx + 1;
            assert_eq!(evaluation.first, EF::zero(), "first row {row_idx}");
            assert_eq!(evaluation.nonfirst, EF::zero(), "nonfirst row {row_idx}");
            let block_mults =
                [0, 1, 3, 4, 6].map(
                    |elem_idx| {
                        if elem_idx < slot_count {
                            F::one()
                        } else {
                            F::zero()
                        }
                    },
                );
            let mut expected = vec![-F::one(), F::zero()];
            expected.extend(block_mults);
            expected.push(F::zero());
            assert_eq!(evaluation.lookup_multiplicities, expected, "lookup order row {row_idx}");
        }

        let padded_record = leaf_ext_record(authoritative_leaf_ext_rows(&[1, 8, 3]));
        let padded = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&padded_record);
        assert_eq!(padded.stored_height(), 3);
        assert_eq!(padded.total_height, 4);
        let padded_evaluations = leaf_ext_materialized_evaluations(&padded);
        assert_eq!(padded_evaluations.len(), 3);

        let (_, multi_rows) = authoritative_leaf_group(13, 99);
        assert_eq!(multi_rows.len(), 2);
        assert_eq!(multi_rows[0].chain_send_cursor, multi_rows[1].chain_recv_cursor);
        assert_eq!(multi_rows[0].pow_out, multi_rows[1].pow_in);
        assert_eq!(multi_rows[0].acc_out, multi_rows[1].acc_in);
        assert!(!multi_rows[0].is_unit_end);
        assert!(multi_rows[1].is_unit_end);
        let multi = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&leaf_ext_record(
            multi_rows,
        ));
        for evaluation in leaf_ext_materialized_evaluations(&multi) {
            assert_eq!(evaluation.first, EF::zero());
            assert_eq!(evaluation.nonfirst, EF::zero());
        }
    }

    #[test]
    fn leaf_ext_early_endpoint_reaches_the_authenticated_roots() {
        let (base, mut full_ext) = authoritative_leaf_group(13, 0);
        assert_eq!(full_ext.len(), 2);
        full_ext[1].serve_cnt = 1;
        let full_sum = full_ext[1].acc_out;
        let partial_sum = full_ext[0].acc_out;
        assert_ne!(partial_sum, full_sum);

        let mut partial_ext = vec![full_ext[0]];
        partial_ext[0].is_unit_end = true;
        partial_ext[0].serve_cnt = 1;

        let full_query = query_rows_for_leaf_sum(0, 0, full_sum);
        let partial_query = query_rows_for_leaf_sum(0, 0, partial_sum);
        assert!(query_gate_evaluations(full_query.clone())
            .iter()
            .all(|value| *value == EF::zero()));
        assert!(query_gate_evaluations(partial_query.clone())
            .iter()
            .all(|value| *value == EF::zero()));

        let input_unit_key = whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, 5);
        let (full_input_merkle, full_input_root) = one_level_merkle_component(
            0,
            WHIR_BATCH_PERMUTATION,
            input_unit_key,
            0,
            &ext_merkle_blocks(&full_ext),
            1,
            1,
        );
        let (partial_input_merkle, partial_input_root) = one_level_merkle_component(
            0,
            WHIR_BATCH_PERMUTATION,
            input_unit_key,
            0,
            &ext_merkle_blocks(&partial_ext),
            1,
            1,
        );
        assert_ne!(partial_input_root, full_input_root);

        let full_round = full_query[1];
        let partial_round = partial_query[1];
        let query_unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE, 4);
        let (full_query_merkle, full_query_root) = one_level_merkle_component(
            0,
            100,
            query_unit_key,
            full_round.chain_send_idx.as_canonical_u32() as usize,
            &query_merkle_blocks(full_round),
            1,
            1,
        );
        let (partial_query_merkle, partial_query_root) = one_level_merkle_component(
            0,
            100,
            query_unit_key,
            partial_round.chain_send_idx.as_canonical_u32() as usize,
            &query_merkle_blocks(partial_round),
            1,
            1,
        );
        assert_ne!(partial_query_root, full_query_root);

        let honest = synchronized_leaf_record(
            base.clone(),
            full_ext.clone(),
            full_query.clone(),
            full_input_merkle.clone(),
            full_query_merkle.clone(),
            full_input_root,
            full_query_root,
        );
        assert!(leaf_chain_residual(&honest).is_empty());
        assert!(query_leaf_sum_residual(&honest).is_empty());
        let honest_leaf_block_residual = merkle_leaf_block_residual(&honest);
        assert!(
            honest_leaf_block_residual.is_empty(),
            "unexpected honest MerkleLeafBlock residual: {honest_leaf_block_residual:?}"
        );
        assert!(commitment_root_residual(&honest).is_empty());
        let honest_transcript_residual = transcript_event_residual(&honest);
        assert!(
            honest_transcript_residual.is_empty(),
            "unexpected honest transcript residual: {honest_transcript_residual:?}"
        );

        // Synchronization depth 1: retaining later Merkle rows with zero demand
        // fails the compiled support equation before any endpoint bus can detach.
        let later_absorb = honest.proof_records[0]
            .merkle_path
            .rows()
            .iter()
            .position(|row| {
                matches!(row.op, RecursionMerklePathOp::LeafAbsorb) &&
                    row.commit_id == WHIR_BATCH_PERMUTATION &&
                    row.block_idx == full_ext[1].block_idx
            })
            .expect("later Ext leaf block");
        let mut unsupported_trace = MerklePathTraceGenerator::generate_trace_compressed(&honest);
        let row_start = later_absorb * NUM_MERKLE_PATH_COLS;
        unsupported_trace.main.values
            [row_start + core::mem::offset_of!(MerklePathCols<u8>, absorb_cnt)] = F::zero();
        unsupported_trace.main.values
            [row_start + core::mem::offset_of!(MerklePathCols<u8>, left_idx)] = F::zero();
        let unsupported_binding = unsupported_trace.main.row_slice(later_absorb);
        let unsupported_cols: &MerklePathCols<F> = unsupported_binding.as_ref().borrow();
        assert_eq!(unsupported_cols.absorb_cnt, F::zero());
        assert_eq!(unsupported_cols.left_idx, F::zero());
        assert_ne!(
            unsupported_cols.absorb_cnt * unsupported_cols.left_idx -
                unsupported_cols.is_leaf_absorb,
            F::zero(),
            "`absorb_cnt * left_idx - is_leaf_absorb` must be the local support failure"
        );

        // Synchronization depth 2: ending after the first full Ext row balances
        // LeafChain, but retained demand for the omitted row leaves the exact
        // commit_id-carrying MerkleLeafBlock key unmatched.
        let omitted_producer = synchronized_leaf_record(
            base.clone(),
            partial_ext.clone(),
            full_query.clone(),
            full_input_merkle.clone(),
            full_query_merkle.clone(),
            full_input_root,
            full_query_root,
        );
        assert!(leaf_chain_residual(&omitted_producer).is_empty());
        let later_mask = full_ext[1].chunk_masks[0];
        let later_chunk = full_ext[1].value_blocks[0];
        let later_key = merkle_leaf_block_key(
            0,
            WHIR_BATCH_PERMUTATION,
            input_unit_key,
            0,
            full_ext[1].block_idx,
            later_mask,
            later_chunk,
        );
        assert_eq!(merkle_leaf_block_residual(&omitted_producer).get(&later_key), Some(&-1));

        // Synchronization depth 3: after the leaf sum is changed, retaining the
        // old pair values fails the exact QueryFold assign equation.
        let mut changed_sum_query = full_query.clone();
        changed_sum_query[1].leaf_sum = partial_sum;
        let selected = if changed_sum_query[1].idx_bit {
            changed_sum_query[1].f1
        } else {
            changed_sum_query[1].f0
        };
        assert_ne!(selected, changed_sum_query[1].leaf_sum);
        let changed_sum_evaluations = query_gate_evaluations(changed_sum_query.clone());
        assert_eq!(changed_sum_evaluations[0], EF::zero());
        assert_ne!(changed_sum_evaluations[1], EF::zero());
        let changed_sum = synchronized_leaf_record(
            base.clone(),
            partial_ext.clone(),
            changed_sum_query,
            partial_input_merkle.clone(),
            full_query_merkle.clone(),
            full_input_root,
            full_query_root,
        );
        assert!(query_leaf_sum_residual(&changed_sum).is_empty());

        // Synchronization depth 4: recomputing the QueryFold pair changes both
        // producer keys; the old IOPP Merkle consumer is now the first bus failure.
        let changed_pair = synchronized_leaf_record(
            base.clone(),
            partial_ext.clone(),
            partial_query.clone(),
            partial_input_merkle.clone(),
            full_query_merkle.clone(),
            full_input_root,
            full_query_root,
        );
        let old_pair_key = merkle_leaf_block_key(
            0,
            100,
            query_unit_key,
            0,
            0,
            query_pair_leaf_mask_for_test(0),
            query_pair_leaf_chunk_for_test(full_round, 0),
        );
        let new_pair_key = merkle_leaf_block_key(
            0,
            100,
            query_unit_key,
            0,
            0,
            query_pair_leaf_mask_for_test(0),
            query_pair_leaf_chunk_for_test(partial_round, 0),
        );
        let changed_pair_residual = merkle_leaf_block_residual(&changed_pair);
        assert_eq!(changed_pair_residual.get(&old_pair_key), Some(&-1));
        assert_eq!(changed_pair_residual.get(&new_pair_key), Some(&1));

        // Synchronization depth 5: after both Merkle/Poseidon components are
        // rebuilt, every earlier relation balances and the authenticated old
        // roots are the remaining MerkleCommitmentRootBus failures.
        let fully_synchronized = synchronized_leaf_record(
            base,
            partial_ext,
            partial_query,
            partial_input_merkle,
            partial_query_merkle,
            full_input_root,
            full_query_root,
        );
        assert!(leaf_chain_residual(&fully_synchronized).is_empty());
        assert!(query_leaf_sum_residual(&fully_synchronized).is_empty());
        assert!(merkle_leaf_block_residual(&fully_synchronized).is_empty());
        assert!(transcript_event_residual(&fully_synchronized).is_empty());
        let root_residual = commitment_root_residual(&fully_synchronized);
        assert_eq!(
            root_residual.get(&commitment_root_key(0, WHIR_BATCH_PERMUTATION, full_input_root)),
            Some(&1)
        );
        assert_eq!(
            root_residual.get(&commitment_root_key(0, WHIR_BATCH_PERMUTATION, partial_input_root)),
            Some(&-1)
        );
        assert_eq!(root_residual.get(&commitment_root_key(0, 100, full_query_root)), Some(&1));
        assert_eq!(root_residual.get(&commitment_root_key(0, 100, partial_query_root)), Some(&-1));
    }

    #[test]
    fn leaf_ext_count_multiplicity_and_order_variants_reach_the_authenticated_roots() {
        let full_values = authoritative_ext_values(13, 0);
        let honest = synchronized_leaf_fixture(full_values.clone());
        let honest_record =
            synchronized_leaf_fixture_record(&honest, honest.input_root, honest.query_root);
        assert!(leaf_chain_residual(&honest_record).is_empty());
        assert!(query_leaf_sum_residual(&honest_record).is_empty());
        assert!(merkle_leaf_block_residual(&honest_record).is_empty());
        assert!(commitment_root_residual(&honest_record).is_empty());
        assert!(transcript_event_residual(&honest_record).is_empty());

        let assert_root_rejection = |label: &str, variant: SynchronizedLeafFixture| {
            for evaluation in leaf_ext_materialized_evaluations(
                &WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&leaf_ext_record(
                    variant.extension.clone(),
                )),
            ) {
                assert_eq!(evaluation.first, EF::zero(), "{label}: first evaluator");
                assert_eq!(evaluation.nonfirst, EF::zero(), "{label}: non-first evaluator");
            }
            let record =
                synchronized_leaf_fixture_record(&variant, honest.input_root, honest.query_root);
            assert!(leaf_chain_residual(&record).is_empty(), "{label}: LeafChain");
            assert!(query_leaf_sum_residual(&record).is_empty(), "{label}: QueryLeafSum");
            assert!(merkle_leaf_block_residual(&record).is_empty(), "{label}: MerkleLeafBlock");
            assert!(transcript_event_residual(&record).is_empty(), "{label}: transcript");
            let roots = commitment_root_residual(&record);
            assert_eq!(
                roots.get(&commitment_root_key(0, WHIR_BATCH_PERMUTATION, honest.input_root,)),
                Some(&1),
                "{label}: authenticated input root"
            );
            assert_eq!(
                roots.get(&commitment_root_key(0, WHIR_BATCH_PERMUTATION, variant.input_root,)),
                Some(&-1),
                "{label}: recomputed input root"
            );
            assert_eq!(
                roots.get(&commitment_root_key(0, 100, honest.query_root)),
                Some(&1),
                "{label}: authenticated query root"
            );
            assert_eq!(
                roots.get(&commitment_root_key(0, 100, variant.query_root)),
                Some(&-1),
                "{label}: recomputed query root"
            );
        };

        let partial = synchronized_leaf_fixture(full_values[..5].to_vec());
        assert_eq!(partial.extension.len(), 1);
        assert!(partial.extension[0].is_unit_end);
        assert!(!partial.extension[0].chunk_masks[3][1]);
        assert_root_rejection("partial Ext row ending early", partial);

        let omitted_later = synchronized_leaf_fixture(full_values[..8].to_vec());
        assert_eq!(omitted_later.extension.len(), 1);
        assert!(omitted_later.extension[0].chunk_masks.iter().flatten().all(|bit| *bit));
        assert_root_rejection("full Ext row ending with later row omitted", omitted_later);

        let mut duplicate_row_values = full_values[..8].to_vec();
        duplicate_row_values.extend_from_slice(&full_values[..8]);
        duplicate_row_values.extend_from_slice(&full_values[8..]);
        let duplicated = synchronized_leaf_fixture(duplicate_row_values);
        assert_eq!(duplicated.extension.len(), 3);
        assert_eq!(duplicated.extension[0].value_blocks, duplicated.extension[1].value_blocks);
        assert_root_rejection("duplicated Ext row", duplicated);

        let mut reordered_values = full_values;
        reordered_values.swap(2, 10);
        let reordered = synchronized_leaf_fixture(reordered_values);
        assert_eq!(reordered.extension.len(), 2);
        assert_root_rejection("reordered Ext elements", reordered);
    }

    #[test]
    fn leaf_ext_trace_preserves_proof_order_and_empty_zero_row() {
        let mut first = authoritative_leaf_ext_rows(&[2]).pop().expect("first Ext row");
        first.proof_idx = 7;
        let mut second = authoritative_leaf_ext_rows(&[6]).pop().expect("second Ext row");
        second.proof_idx = 3;
        let record = RecursionRecord {
            proof_records: vec![
                RecursionProofRecord {
                    proof_idx: 7,
                    whir: RecursionWhirRecord {
                        leaf_ext_stream_rows: vec![first.into()],
                        ..Default::default()
                    },
                    ..Default::default()
                },
                RecursionProofRecord {
                    proof_idx: 3,
                    whir: RecursionWhirRecord {
                        leaf_ext_stream_rows: vec![second.into()],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let ordered = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&record);
        assert_eq!(ordered.stored_height(), 2);
        let first_row = ordered.main.row_slice(0);
        let second_row = ordered.main.row_slice(1);
        let first_cols: &WhirLeafExtStreamCols<F> = first_row.deref().borrow();
        let second_cols: &WhirLeafExtStreamCols<F> = second_row.deref().borrow();
        assert_eq!(first_cols.proof_idx, F::from_canonical_usize(7));
        assert_eq!(second_cols.proof_idx, F::from_canonical_usize(3));

        let empty =
            WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&RecursionRecord::default());
        assert_eq!(empty.stored_height(), 1);
        assert_eq!(empty.total_height, 1);
        assert!(empty.main.values.iter().all(|value| *value == F::zero()));
        let empty_evaluations = leaf_ext_materialized_evaluations(&empty);
        assert_eq!(empty_evaluations.len(), 1);
        assert_eq!(empty_evaluations[0].first, EF::zero());
        assert_eq!(empty_evaluations[0].nonfirst, EF::zero());
        assert!(empty_evaluations[0].lookup_multiplicities.iter().all(|value| *value == F::zero()));
    }

    #[test]
    fn leaf_ext_merkle_blocks_follow_flat_ext5_limb_order() {
        let (_, mut extension) = authoritative_leaf_group(8, 21);
        let row = extension.pop().expect("one full Ext row");
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let mut proof = RecursionProofRecord {
            proof_idx: row.proof_idx,
            whir: RecursionWhirRecord {
                leaf_ext_stream_rows: vec![row.into()],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut prev_state = [F::zero(); POSEIDON2_WIDTH];
        for block in 0..WHIR_LEAF_BLOCKS_PER_ROW {
            let merkle_row = RecursionMerklePathRow::leaf_absorb(
                row.proof_idx,
                row.unit_key,
                WHIR_BATCH_PERMUTATION,
                row.block_idx + block,
                row.idx,
                1,
                false,
                block == 0,
                block + 1 == WHIR_LEAF_BLOCKS_PER_ROW,
                prev_state,
                row.value_blocks[block],
                row.chunk_masks[block],
                &poseidon2_memo,
            );
            prev_state = merkle_row.output;
            proof.merkle_path.push_row(merkle_row);
        }
        let record = RecursionRecord { proof_records: vec![proof], ..Default::default() };
        assert!(
            merkle_leaf_block_residual(&record).is_empty(),
            "the five Ext blocks must use flat ext5 limb order"
        );

        let mut bad_value = record.clone();
        bad_value.proof_records[0].whir.leaf_ext_stream_rows[0].value_blocks[1][0] += F::one();
        assert!(
            !merkle_leaf_block_residual(&bad_value).is_empty(),
            "changing an Ext limb must leave a MerkleLeafBlock residual"
        );

        let mut bad_commit = record.clone();
        let mut merkle_rows = bad_commit.proof_records[0].merkle_path.rows().to_vec();
        merkle_rows[0].commit_id += 1;
        bad_commit.proof_records[0].merkle_path = Default::default();
        for row in merkle_rows {
            bad_commit.proof_records[0].merkle_path.push_row(row);
        }
        assert!(
            !merkle_leaf_block_residual(&bad_commit).is_empty(),
            "changing the commitment tree id must leave a MerkleLeafBlock residual"
        );

        let mut cross_proof = record.clone();
        let mut merkle_rows = cross_proof.proof_records[0].merkle_path.rows().to_vec();
        merkle_rows[0].proof_idx = 1;
        cross_proof.proof_records[0].merkle_path = Default::default();
        for merkle_row in merkle_rows {
            cross_proof.proof_records[0].merkle_path.push_row(merkle_row);
        }
        let source_key = merkle_leaf_block_key(
            row.proof_idx,
            WHIR_BATCH_PERMUTATION,
            row.unit_key,
            row.idx,
            row.block_idx,
            row.chunk_masks[0],
            row.value_blocks[0],
        );
        let reused_key = merkle_leaf_block_key(
            1,
            WHIR_BATCH_PERMUTATION,
            row.unit_key,
            row.idx,
            row.block_idx,
            row.chunk_masks[0],
            row.value_blocks[0],
        );
        let cross_proof_residual = merkle_leaf_block_residual(&cross_proof);
        assert_eq!(cross_proof_residual.get(&source_key), Some(&1));
        assert_eq!(cross_proof_residual.get(&reused_key), Some(&-1));

        let mut bad_block = record;
        bad_block.proof_records[0].whir.leaf_ext_stream_rows[0].block_idx += 1;
        assert!(
            !merkle_leaf_block_residual(&bad_block).is_empty(),
            "changing the Ext block index must leave a MerkleLeafBlock residual"
        );
    }

    #[test]
    fn leaf_ext_mask_power_and_key_start_tampering_fail_full_evaluation() {
        let record = leaf_ext_record(authoritative_leaf_ext_rows(&[5]));
        let honest = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(&record);
        let width = honest.main.width();

        let mut bad_mask = honest.clone();
        let mask_offset = core::mem::offset_of!(WhirLeafExtStreamCols<u8>, element_masks) + 5;
        bad_mask.main.values[mask_offset] = F::from_canonical_usize(2);
        let bad_mask_eval = &leaf_ext_materialized_evaluations(&bad_mask)[0];
        assert_ne!(bad_mask_eval.first, EF::zero());
        assert_ne!(bad_mask_eval.nonfirst, EF::zero());

        let mut bad_power = honest.clone();
        let power_offset = core::mem::offset_of!(WhirLeafExtStreamCols<u8>, slot_pows);
        bad_power.main.values[power_offset] += F::one();
        let bad_power_eval = &leaf_ext_materialized_evaluations(&bad_power)[0];
        assert_ne!(bad_power_eval.first, EF::zero());
        assert_ne!(bad_power_eval.nonfirst, EF::zero());

        let mut bad_acc = honest.clone();
        let acc_out_offset = core::mem::offset_of!(WhirLeafExtStreamCols<u8>, acc_out);
        bad_acc.main.values[acc_out_offset] += F::one();
        let bad_acc_eval = &leaf_ext_materialized_evaluations(&bad_acc)[0];
        assert_ne!(bad_acc_eval.first, EF::zero());
        assert_ne!(bad_acc_eval.nonfirst, EF::zero());

        let mut bad_key_start = honest;
        let key_start_offset = core::mem::offset_of!(WhirLeafExtStreamCols<u8>, is_unit_key_start);
        bad_key_start.main.values[key_start_offset] = F::from_canonical_usize(2);
        let bad_key_start_eval = &leaf_ext_materialized_evaluations(&bad_key_start)[0];
        assert_ne!(bad_key_start_eval.first, EF::zero());
        assert_ne!(bad_key_start_eval.nonfirst, EF::zero());

        let (_, multi_rows) = authoritative_leaf_group(13, 3);
        let mut bad_serve = WhirLeafExtStreamTraceGenerator::generate_trace_compressed(
            &leaf_ext_record(multi_rows),
        );
        let serve_offset = core::mem::offset_of!(WhirLeafExtStreamCols<u8>, serve_cnt);
        bad_serve.main.values[serve_offset] = F::one();
        let bad_serve_eval = &leaf_ext_materialized_evaluations(&bad_serve)[0];
        assert_ne!(bad_serve_eval.first, EF::zero());
        assert_ne!(bad_serve_eval.nonfirst, EF::zero());

        assert_eq!(width, NUM_WHIR_LEAF_EXT_STREAM_COLS);
    }

    #[test]
    fn leaf_ext_writer_derives_deleted_metadata_and_first_slot_power() {
        let row = authoritative_leaf_ext_rows(&[3]).pop().expect("one row");
        let honest = leaf_ext_stream_row(row);
        let mut redundant_tamper = row;
        redundant_tamper.chain_send_cursor += 17;
        redundant_tamper.batch_id = 0;
        redundant_tamper.chain_recv_log_height += 11;
        redundant_tamper.chain_recv_batch_id += 9;
        redundant_tamper.unit_key += 23;
        redundant_tamper.is_unit_start = !redundant_tamper.is_unit_start;
        redundant_tamper.unit_key_gap += 31;
        redundant_tamper.slot_pows[0][0] += F::one();
        assert_eq!(
            leaf_ext_stream_row(redundant_tamper),
            honest,
            "deleted semantic mirrors must not regain committed authority"
        );
    }

    #[test]
    fn leaf_ext_requires_the_product_main_predecessor() {
        let (base, extension) = authoritative_leaf_group(3, 0);
        let complete = RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 0,
                whir: RecursionWhirRecord {
                    leaf_stream_rows: base,
                    leaf_ext_stream_rows: extension.clone().into_iter().map(Into::into).collect(),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(
            leaf_chain_residual(&complete).is_empty(),
            "the authenticated main predecessor must feed the first Ext row"
        );

        let missing_base = leaf_ext_record(extension);
        assert!(
            !leaf_chain_residual(&missing_base).is_empty(),
            "an Ext-only height group must leave an unmatched mandatory chain receive"
        );
    }

    #[test]
    fn base_batch_relabel_is_rejected_outside_the_leaf_chain() {
        let fixture = synchronized_leaf_fixture(authoritative_ext_values(3, 0));
        let honest =
            synchronized_leaf_fixture_record(&fixture, fixture.input_root, fixture.query_root);
        assert!(leaf_chain_residual(&honest).is_empty());
        assert!(merkle_leaf_block_residual(&honest).is_empty());
        assert!(commitment_root_residual(&honest).is_empty());
        let authenticated_main_root = honest.proof_records[0].whir.round_rows[0].commit_root;
        let old_base = fixture.base[0];
        assert!(!old_base.is_unit_end, "the predecessor must send into Ext");
        assert!(old_base.is_unit_start);
        assert_eq!(old_base.unit_key_gap, 0);

        // Synchronously relabel the Base unit, its LeafChain send, and the first
        // Ext receive. Because this is the first row, the Range8 multiplicity
        // `is_unit_key_start - is_unit_start` stays zero.
        let mut relabeled_fixture = fixture.clone();
        let predecessor = &mut relabeled_fixture.base[0];
        predecessor.batch_id = WHIR_BATCH_PERMUTATION;
        predecessor.unit_key = whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, 5);
        let first_ext = &mut relabeled_fixture.extension[0];
        first_ext.is_unit_key_start = false;
        first_ext.chain_recv_batch_id = WHIR_BATCH_PERMUTATION;
        first_ext.unit_key_gap = 0;
        let relabeled_base_row = *predecessor;
        let relabeled_base = leaf_stream_row(relabeled_base_row);
        let relabeled_base_cols: &WhirLeafStreamCols<F> = relabeled_base.as_slice().borrow();
        assert_eq!(
            relabeled_base_cols.chunk_mask[0] *
                (relabeled_base_cols.unit_key -
                    F::from_canonical_usize(WHIR_UNIT_KEY_SLOT_STRIDE) *
                        relabeled_base_cols.batch_id -
                    relabeled_base_cols.log_height),
            F::zero(),
            "the synchronized relabel must satisfy the Base routing equation"
        );
        assert_eq!(
            relabeled_base_cols.is_unit_key_start - relabeled_base_cols.is_unit_start,
            F::zero(),
            "the first Base row must not create a Range8 receive"
        );

        // Branch 1: keeping the original main Merkle consumer leaves the exact
        // commit_id/unit_key-carrying leaf-block keys unmatched.
        let mut stale_merkle = honest.clone();
        stale_merkle.proof_records[0].whir.leaf_stream_rows = relabeled_fixture.base.clone();
        stale_merkle.proof_records[0].whir.leaf_ext_stream_rows =
            relabeled_fixture.extension.clone().into_iter().map(Into::into).collect();
        assert!(
            leaf_chain_residual(&stale_merkle).is_empty(),
            "this synchronized change intentionally preserves the LeafChain multiset"
        );
        let old_base_key = merkle_leaf_block_key(
            0,
            old_base.batch_id,
            old_base.unit_key,
            old_base.idx,
            old_base.block_idx,
            old_base.chunk_mask,
            old_base.values,
        );
        let new_base_key = merkle_leaf_block_key(
            0,
            WHIR_BATCH_PERMUTATION,
            relabeled_base_row.unit_key,
            relabeled_base_row.idx,
            relabeled_base_row.block_idx,
            relabeled_base_row.chunk_mask,
            relabeled_base_row.values,
        );
        let stale_leaf_residual = merkle_leaf_block_residual(&stale_merkle);
        assert_eq!(stale_leaf_residual.get(&old_base_key), Some(&-1));
        assert_eq!(stale_leaf_residual.get(&new_base_key), Some(&1));

        // Branch 2: rebuilding the Base Merkle/Poseidon component under commit
        // 2 balances every leaf block. Keeping the prescribed main and
        // permutation roots then exposes the relabeled component at the root bus.
        let mut propagated = synchronized_leaf_fixture_record(
            &relabeled_fixture,
            fixture.input_root,
            fixture.query_root,
        );
        let relabeled_root = propagated.proof_records[0].whir.round_rows[0].commit_root;
        assert_ne!(relabeled_root, fixture.input_root);
        propagated.proof_records[0].whir.round_rows[0].commit_id = old_base.batch_id;
        propagated.proof_records[0].whir.round_rows[0].commit_root = authenticated_main_root;
        assert!(leaf_chain_residual(&propagated).is_empty());
        assert!(merkle_leaf_block_residual(&propagated).is_empty());
        assert!(transcript_event_residual(&propagated).is_empty());
        let propagated_roots = commitment_root_residual(&propagated);
        assert_eq!(
            propagated_roots.get(&commitment_root_key(
                0,
                old_base.batch_id,
                authenticated_main_root,
            )),
            Some(&1)
        );
        assert_eq!(
            propagated_roots.get(&commitment_root_key(0, WHIR_BATCH_PERMUTATION, relabeled_root,)),
            Some(&-1)
        );
    }

    #[test]
    fn product_l1_l4_permutation_segments_have_main_predecessors() {
        let statement_config = |class_ids: &[usize]| {
            class_ids
                .iter()
                .copied()
                .map(|class_id| StatementConfigRow { class_id, digest: [F::zero(); DIGEST_SIZE] })
                .collect::<Vec<_>>()
        };
        let core_machine = core_recording_machine();
        let l1 =
            build_core_native_recursion_program(&core_machine).expect("compile L1 test program");
        let l1_child = native_recording_machine(&l1).expect("build L1 child machine");
        let l2_config = statement_config(&[STATEMENT_CONFIG_CLASS_BAKED_LIFT]);
        let l2_bootstrap = build_native_recursion_program(
            &l1_child,
            RecursionStatementRole::ReduceL2,
            RecursionChildRole::Compress,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false,
            l2_config.clone(),
        )
        .expect("compile L2 bootstrap test program");
        let l2_bootstrap_child =
            native_recording_machine(&l2_bootstrap).expect("build L2 bootstrap child machine");
        let l2 = build_dual_segment_reduce_program(
            &l1_child,
            &l2_bootstrap_child,
            RecursionStatementRole::ReduceL2,
            l2_config,
        )
        .expect("compile L2 test program");
        let l2_child = native_recording_machine(&l2).expect("build L2 child machine");
        let l3 = build_dual_segment_reduce_program(
            &l1_child,
            &l2_child,
            RecursionStatementRole::ReduceL3,
            statement_config(&[STATEMENT_CONFIG_CLASS_BAKED_LIFT, STATEMENT_CONFIG_CLASS_BAKED_L2]),
        )
        .expect("compile L3 test program");
        let l3_child = native_recording_machine_for_stage(&l3, RecordingStage::Shrink)
            .expect("build L3 shrink child machine");
        let l4 = build_root_shrink_program(
            &l3_child,
            statement_config(&[STATEMENT_CONFIG_CLASS_BAKED_L3]),
        )
        .expect("compile L4 test program");

        for (layer, program) in [("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4)] {
            let mut permutation_chips = 0usize;
            for chip in &program.constraint_program.chips {
                if chip.lookup_multiplicity_roots.is_empty() {
                    continue;
                }
                permutation_chips += 1;
                assert!(
                    chip.widths.main > 0,
                    "{layer} chip {} has permutation values without a main predecessor",
                    chip.chip_name
                );
            }
            assert!(permutation_chips > 0, "{layer} must exercise WhirLeafExtStream");
        }
    }

    #[test]
    fn role_config_rows_are_pinned_to_active_json_config() {
        let configs = whir_role_configs();
        for (role_id, stage_name) in [
            (WHIR_ROLE_CORE, "core"),
            (WHIR_ROLE_COMPRESS, "compress"),
            (WHIR_ROLE_SHRINK, "shrink"),
        ] {
            let generated = configs
                .iter()
                .find(|config| config.role_id == role_id)
                .expect("generated role config");
            let stage = whir_config().stage(stage_name);
            assert_eq!(generated.num_queries, stage.num_queries.expect("json num_queries"));
            assert_eq!(
                generated.batching_bits,
                stage.grinding_bits_batching.expect("json batching bits")
            );
            assert_eq!(generated.log_blowup, stage.log_blowup.expect("json log_blowup"));
            assert_eq!(stage.stacking, Some(false));
            assert_eq!(stage.path_pruning, Some(false));
        }
    }

    #[test]
    fn sample_band_rows_cover_query_geometry_domain() {
        let rows = whir_sample_band_rows();
        assert_eq!(rows.len(), KOALABEAR_MAX_TRACE_LOG_HEIGHT);
        assert_eq!(rows[20].query_bits, 21);
        assert_eq!(rows[20].high_bits, 10);
        assert_eq!(rows[21].query_bits, 22);
        assert_eq!(rows[21].high_bits, 9);
        assert_eq!(rows[19].query_bits, 20);
        assert_eq!(rows[19].high_bits, 11);
        assert!(sample_band_for_query_bits(0).is_none());
        assert!(sample_band_for_query_bits(KOALABEAR_MAX_TRACE_LOG_HEIGHT + 1).is_none());
    }

    #[test]
    fn sample_band_residual_rejects_payload_tamper() {
        let mut record = sample_query_geometry_record(21, 19);
        assert!(sample_band_residual(&record).is_empty(), "sample band bus must balance");

        record.proof_records[0].whir.query_fold_rows[0].query_sample_high_bits += 1;
        assert!(
            !sample_band_residual(&record).is_empty(),
            "tampering generated sample-band payload must leave an 11 residual"
        );
    }

    #[test]
    fn query_init_residual_rejects_query_bits_tamper() {
        let mut record = sample_query_geometry_record(21, 19);
        assert!(query_init_residual(&record).is_empty(), "query init bus must balance");

        record.proof_records[0].whir.query_fold_rows[0].query_bits += 1;
        assert!(
            !query_init_residual(&record).is_empty(),
            "tampering QueryFold query_bits must leave a 1030 residual"
        );
    }

    #[test]
    fn twiddle_table_matches_power_decomposition() {
        let g = F::two_adic_generator(24);
        assert_eq!(twiddle_value(0, 0), F::one());
        assert_eq!(twiddle_value(0, 7), g.exp_u64(7));
        assert_eq!(twiddle_value(1, 7), g.exp_u64(7 << 8));
        assert_eq!(twiddle_value(2, 7), g.exp_u64(7 << 16));
    }

    #[test]
    fn round_chain_carries_preamble_assign_into_first_broadcast() {
        let record = sample_whir_round_record();
        let rows = whir_round_rows(&record);
        assert_eq!(rows.len(), 4);

        let preamble_values = round_row(rows[1]);
        let first_round_values = round_row(rows[2]);
        let preamble: &WhirRoundCols<F> = preamble_values.as_slice().borrow();
        let first_round: &WhirRoundCols<F> = first_round_values.as_slice().borrow();

        assert_eq!(preamble.chain_send_pending_is_merge, F::one());
        assert_eq!(first_round.chain_recv_pending_is_merge, F::one());
        assert_eq!(first_round.chain_recv_pending_beta, [F::zero(); D_EF]);
        assert_eq!(first_round.chain_recv_pending_eq, [F::zero(); D_EF]);
    }

    #[test]
    fn round_chain_balances_and_rejects_pending_tamper() {
        let record = sample_whir_round_record();
        assert_round_chain_balanced(&record);

        let mut tampered = record;
        tampered.proof_records[0].whir.round_rows[2].chain_recv_pending_is_merge = false;
        let residual = round_chain_residual(&tampered);
        assert!(
            !residual.is_empty(),
            "tampering the C-indexed pending merge must leave a 1032 residual"
        );
    }

    #[test]
    fn query_chain_carries_lookahead_bit_in_residual() {
        let record = sample_whir_query_chain_record();
        assert_query_chain_balanced(&record);

        let mut tampered = record;
        tampered.proof_records[0].whir.query_fold_rows[1].chain_send_idx_bit = true;
        let residual = query_chain_residual(&tampered);
        assert!(
            !residual.is_empty(),
            "tampering the carried lookahead bit must leave a 1026 residual"
        );
    }

    #[test]
    fn authenticated_cursor_prevents_round_control_permutation() {
        let record = sample_whir_query_chain_record();
        assert_query_chain_balanced(&record);
        assert!(round_bcast_residual(&record).is_empty());

        let mut tampered = record;
        let query_rows = &mut tampered.proof_records[0].whir.query_fold_rows;
        let first_control = query_rows[1].r_fold;
        query_rows[1].r_fold = query_rows[2].r_fold;
        query_rows[2].r_fold = first_control;

        assert!(
            query_chain_residual(&tampered).is_empty(),
            "round controls are intentionally outside the QueryChain payload"
        );
        assert!(
            !round_bcast_residual(&tampered).is_empty(),
            "RoundBcast must bind each control payload to its authenticated cursor"
        );
    }

    #[test]
    fn query_init_carries_cfr_to_seed_terminal() {
        let record = sample_whir_query_init_record();
        assert_query_init_balanced(&record);

        let mut tampered = record;
        tampered.proof_records[0].whir.query_fold_rows[0].cfr[0] += F::one();
        let residual = query_init_residual(&tampered);
        assert!(!residual.is_empty(), "tampering the seed cfr must leave a 1030 residual");
    }

    #[test]
    fn cross_chip_whir_buses_balance_and_reject_payload_tamper() {
        let record = sample_whir_cross_bus_record();
        assert!(twiddle_residual(&record).is_empty(), "twiddle bus must balance");
        assert!(leaf_pow_seed_residual(&record).is_empty(), "leaf pow seed bus must balance");
        assert!(group_claim_residual(&record).is_empty(), "group claim bus must balance");
        assert!(round_bcast_residual(&record).is_empty(), "round bcast bus must balance");
        assert!(query_leaf_sum_residual(&record).is_empty(), "query leaf sum bus must balance");

        let mut tampered_round = record.clone();
        tampered_round.proof_records[0].whir.query_fold_rows[1].r_fold[0] += F::one();
        assert!(
            !round_bcast_residual(&tampered_round).is_empty(),
            "tampering C's round challenge must leave a 1023 residual"
        );

        let mut tampered_leaf = record;
        tampered_leaf.proof_records[0].whir.leaf_stream_rows[0].acc_out[0] += F::one();
        assert!(
            !query_leaf_sum_residual(&tampered_leaf).is_empty(),
            "tampering D's leaf sum must leave a 1025 residual"
        );
    }

    #[test]
    fn internal_whir_chain_buses_balance_and_reject_payload_tamper() {
        let record = sample_whir_internal_chain_record();
        assert!(eval_chain_residual(&record).is_empty(), "eval chain bus must balance");
        assert!(leaf_chain_residual(&record).is_empty(), "leaf chain bus must balance");

        let mut tampered_eval = record.clone();
        tampered_eval.proof_records[0].whir.batch_eval_rows[0].pow_out[0] += F::one();
        assert!(
            !eval_chain_residual(&tampered_eval).is_empty(),
            "tampering B's outgoing pow must leave a 1027 residual"
        );

        let mut tampered_leaf = record;
        tampered_leaf.proof_records[0].whir.leaf_ext_stream_rows[0].acc_in[0] += F::one();
        assert!(
            !leaf_chain_residual(&tampered_leaf).is_empty(),
            "tampering D2's incoming accumulator must leave a 1028 residual"
        );
    }

    #[test]
    fn leaf_ext_compact_retained_row_matches_the_semantic_writer() {
        assert_eq!(core::mem::size_of::<RecursionWhirLeafExtStreamRow>(), 568);
        assert_eq!(core::mem::size_of::<RecursionWhirLeafExtStreamTraceRow>(), 456);

        let row = authoritative_leaf_ext_rows(&[8]).pop().expect("one full Ext row");
        let retained = RecursionWhirLeafExtStreamTraceRow::from(row);
        assert_eq!(retained.to_semantic_row(row.proof_idx), row);

        let mut semantic_values = vec![F::zero(); NUM_WHIR_LEAF_EXT_STREAM_COLS];
        fill_leaf_ext_stream_row(&mut semantic_values, &row);
        let mut retained_values = vec![F::zero(); NUM_WHIR_LEAF_EXT_STREAM_COLS];
        fill_leaf_ext_stream_trace_row(&mut retained_values, row.proof_idx, &retained);
        assert_eq!(retained_values, semantic_values);
    }

    #[test]
    fn authenticated_shape_buses_balance_and_reject_payload_tamper() {
        let record = sample_whir_authenticated_shape_record();
        assert!(batch_dim_residual(&record).is_empty(), "batch dim bus must balance");
        assert!(summary_residual(&record).is_empty(), "summary bus must balance");
        assert!(height_group_residual(&record).is_empty(), "height group bus must balance");
        assert!(opening_point_residual(&record).is_empty(), "opening point bus must balance");

        let mut tampered_dim = record.clone();
        tampered_dim.proof_records[0].whir.batch_eval_rows[1].width += 1;
        assert!(
            !batch_dim_residual(&tampered_dim).is_empty(),
            "tampering B's batch width must leave a 1009 residual"
        );

        let mut tampered_summary = record.clone();
        tampered_summary.proof_records[0].whir.round_rows[0].r_rounds += 1;
        assert!(
            !summary_residual(&tampered_summary).is_empty(),
            "tampering WHIR's round-count summary input must leave a 1022 residual"
        );

        let mut tampered_opening = record;
        tampered_opening.proof_records[0].whir.round_rows[2].opening_point[0] += F::one();
        assert!(
            !opening_point_residual(&tampered_opening).is_empty(),
            "tampering WHIR's opening point must leave a 1017 residual"
        );
    }

    #[test]
    fn transcript_and_merkle_buses_balance_and_reject_payload_tamper() {
        let record = sample_whir_transcript_merkle_record();
        assert!(transcript_event_residual(&record).is_empty(), "transcript bus must balance");
        assert!(commitment_root_residual(&record).is_empty(), "commitment root bus must balance");
        assert!(merkle_leaf_block_residual(&record).is_empty(), "leaf block bus must balance");

        let mut tampered_event = record.clone();
        tampered_event.proof_records[0].transcript.events[3].value += F::one();
        assert!(
            !transcript_event_residual(&tampered_event).is_empty(),
            "tampering a WHIR-owned W event must leave a 1007 residual"
        );

        let mut tampered_root = record.clone();
        tampered_root.proof_records[0].whir.round_rows[1].commit_root[0] += F::one();
        assert!(
            !commitment_root_residual(&tampered_root).is_empty(),
            "tampering WHIR's IOPP root must leave a 1002 residual"
        );

        let mut tampered_leaf = record;
        tampered_leaf.proof_records[0].whir.query_fold_rows[1].f0[0] += F::one();
        assert!(
            !merkle_leaf_block_residual(&tampered_leaf).is_empty(),
            "tampering C's pair leaf payload must leave a leaf-block residual"
        );
    }

    #[test]
    fn batch_sumcheck_transcript_residual_reverses_eq_challenge_indices() {
        let record = sample_batch_transcript_record();
        let residual = transcript_event_residual(&record);
        assert!(
            residual.is_empty(),
            "multi-round batch transcript must consume E8 challenges by opening index: {residual:?}"
        );

        let mut tampered = record;
        tampered.proof_records[0].batch_constraint.eq_challenges.swap(0, 2);
        assert!(
            !transcript_event_residual(&tampered).is_empty(),
            "relabeling reversed E8 challenges must leave a 1007 residual"
        );
    }

    fn sample_whir_round_record() -> RecursionRecord {
        let claim = ext(11);
        let eq_one = one_ext();
        let claim_folded = ext(25);
        let eq_factor = ext(2);
        let eq_folded = ext(2);
        let pow_seed = RecursionWhirRoundRow {
            proof_idx: 3,
            is_pow_batch: true,
            tidx: 5,
            r_rounds: 1,
            chain_recv_round: 1,
            chain_recv_tidx: 47,
            chain_recv_claim: claim_folded,
            chain_recv_eq: eq_folded,
            chain_send_tidx: 8,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            role_config_recv_mult: 1,
            summary_recv_mult: 1,
            ..Default::default()
        };

        let mut preamble = RecursionWhirRoundRow {
            proof_idx: 3,
            is_preamble: true,
            tidx: 8,
            r_rounds: 1,
            chain_recv_tidx: 8,
            chain_send_tidx: 16,
            chain_send_claim: claim,
            chain_send_eq: eq_one,
            chain_send_pending_is_merge: true,
            group_claim_log_height: 1,
            height_group_log_height: 1,
            group_claim: claim,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            height_group_recv_mult: 1,
            group_claim_recv_mult: 1,
            ..Default::default()
        };
        preamble.chain_recv_eq = [F::zero(); D_EF];

        let r = ext(2);
        let c0 = ext(3);
        let c1 = ext(5);
        let c2 = ext(3);
        let claim_acc = ext(11);
        let mut event_value = [F::zero(); 32];
        event_value[8..13].copy_from_slice(&c0);
        event_value[13..18].copy_from_slice(&c1);
        event_value[18..23].copy_from_slice(&c2);
        event_value[23..28].copy_from_slice(&r);
        let first_round = RecursionWhirRoundRow {
            proof_idx: 3,
            is_round: true,
            round: 0,
            tidx: 16,
            r_rounds: 1,
            chain_recv_tidx: 16,
            chain_send_tidx: 36,
            chain_recv_claim: claim,
            chain_recv_eq: eq_one,
            chain_recv_pending_is_merge: true,
            chain_send_round: 1,
            chain_send_claim: claim_folded,
            chain_send_eq: eq_folded,
            r_fold: r,
            opening_point: eq_one,
            claim_acc,
            claim_folded,
            eq_factor,
            eq_folded,
            event_value,
            bcast_mult: 1,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            opening_point_recv_mult: 1,
            ..Default::default()
        };

        let final_row = RecursionWhirRoundRow {
            proof_idx: 3,
            is_final: true,
            tidx: 36,
            r_rounds: 1,
            w_qbase: 47,
            chain_recv_round: 1,
            chain_send_round: 1,
            chain_recv_tidx: 36,
            chain_send_tidx: 47,
            chain_recv_claim: claim_folded,
            chain_send_claim: claim_folded,
            chain_recv_eq: eq_folded,
            chain_send_eq: eq_folded,
            cfr: c0,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            ..Default::default()
        };

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 3,
                whir: RecursionWhirRecord {
                    round_rows: vec![pow_seed, preamble, first_round, final_row],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn assert_round_chain_balanced(record: &RecursionRecord) {
        let residual = round_chain_residual(record);
        assert!(residual.is_empty(), "unexpected WhirRoundChain residual: {residual:?}");
    }

    fn sample_whir_query_chain_record() -> RecursionRecord {
        let proof_idx = 3;
        let query_idx = 4;
        let x0 = F::from_canonical_usize(7);
        let x1 = F::from_canonical_usize(9);
        let x2 = F::from_canonical_usize(13);
        let inv4 = F::from_canonical_usize(4).inverse();
        let inv8 = F::from_canonical_usize(8).inverse();
        let high_gap_inv = F::from_canonical_usize(508).inverse();
        let folded0 = ext(30);
        let folded1 = ext(40);
        let folded2 = ext(50);
        let r_fold0 = ext(60);
        let r_fold1 = ext(70);
        let query_bits = 3;
        let r_rounds = 2;

        let seed = RecursionWhirQueryFoldRow {
            proof_idx,
            is_seed: true,
            query_idx,
            query_sample: F::from_canonical_usize(5),
            query_sample_raw: F::from_canonical_usize(5),
            query_sample_high_gap_inv: high_gap_inv,
            query_bits,
            r_rounds,
            cursor: 2,
            idx: F::one(),
            idx_bit: true,
            x: x2,
            acc: inv4,
            ipw: inv8,
            folded: folded2,
            chain_send_cursor: 0,
            chain_send_idx: F::from_canonical_usize(5),
            chain_send_idx_bit: true,
            chain_send_x: x0,
            chain_send_ipw: F::from_canonical_usize(2).inverse(),
            chain_send_folded: folded0,
            ..Default::default()
        };
        let round0 = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx,
            cursor: 0,
            query_bits,
            r_rounds,
            idx: F::from_canonical_usize(5),
            idx_bit: true,
            x: x0,
            ipw: F::from_canonical_usize(2).inverse(),
            folded: folded0,
            chain_send_cursor: 1,
            chain_send_idx: F::from_canonical_usize(2),
            chain_send_idx_bit: false,
            chain_send_x: x1,
            chain_send_ipw: inv4,
            chain_send_folded: folded1,
            r_fold: r_fold0,
            ..Default::default()
        };
        let round1 = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx,
            cursor: 1,
            query_bits,
            r_rounds,
            idx: F::from_canonical_usize(2),
            idx_bit: false,
            x: x1,
            ipw: inv4,
            folded: folded1,
            chain_send_cursor: 2,
            chain_send_idx: F::one(),
            chain_send_idx_bit: true,
            chain_send_x: x2,
            chain_send_acc: inv4,
            chain_send_ipw: inv8,
            chain_send_folded: folded2,
            r_fold: r_fold1,
            ..Default::default()
        };
        let round_bcast0 = RecursionWhirRoundRow {
            proof_idx,
            is_round: true,
            round: 0,
            r_fold: r_fold0,
            merge_log_height: 3,
            bcast_mult: 1,
            ..Default::default()
        };
        let round_bcast1 = RecursionWhirRoundRow {
            proof_idx,
            is_round: true,
            round: 1,
            r_fold: r_fold1,
            merge_log_height: 2,
            bcast_mult: 1,
            ..Default::default()
        };

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                whir: RecursionWhirRecord {
                    round_rows: vec![round_bcast0, round_bcast1],
                    query_fold_rows: vec![seed, round0, round1],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn sample_whir_query_init_record() -> RecursionRecord {
        let proof_idx = 3;
        let w_qbase = 47;
        let cfr = ext(90);
        let high_gap_inv = F::from_canonical_usize(508).inverse();
        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                whir: RecursionWhirRecord {
                    round_rows: vec![RecursionWhirRoundRow {
                        proof_idx,
                        is_final: true,
                        w_qbase,
                        cfr,
                        query_init_mult: 1,
                        ..Default::default()
                    }],
                    query_fold_rows: vec![RecursionWhirQueryFoldRow {
                        proof_idx,
                        is_seed: true,
                        w_qbase,
                        query_sample_high_gap_inv: high_gap_inv,
                        cfr,
                        folded: cfr,
                        idx: F::one(),
                        idx_bit: true,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn sample_whir_cross_bus_record() -> RecursionRecord {
        let proof_idx = 7;
        let alpha = ext(3);
        let claim = ext(20);
        let r_fold = ext(40);
        let beta = ext(50);
        let merge_eq = ext(60);
        let cfr = ext(70);
        let leaf_sum = ext(80);
        let twiddle_bytes = [1, 2, 3];
        let twiddle_values = [
            twiddle_value(0, twiddle_bytes[0] as usize),
            twiddle_value(1, twiddle_bytes[1] as usize),
            twiddle_value(2, twiddle_bytes[2] as usize),
        ];

        let mut twiddle_mults = vec![[0u32; WHIR_TWIDDLE_TABLES]; WHIR_TWIDDLE_ROWS];
        for (table_id, byte) in twiddle_bytes.iter().copied().enumerate() {
            twiddle_mults[byte as usize][table_id] = 1;
        }

        let role_config = whir_role_config(WHIR_ROLE_CORE);
        let mut role_config_mults = [0u32; WHIR_ROLE_COUNT];
        role_config_mults[WHIR_ROLE_CORE] = 2;

        let round_payload = RecursionWhirRoundRow {
            proof_idx,
            is_round: true,
            round: 0,
            r_fold,
            chain_recv_pending_is_merge: true,
            chain_recv_pending_beta: beta,
            chain_recv_pending_eq: merge_eq,
            merge_log_height: 5,
            cfr,
            bcast_mult: 2,
            ..Default::default()
        };
        let group_consumer = RecursionWhirRoundRow {
            proof_idx,
            is_preamble: true,
            group_claim_log_height: 3,
            group_claim: claim,
            group_claim_recv_mult: 1,
            ..Default::default()
        };
        let role_consumer = RecursionWhirRoundRow {
            proof_idx,
            is_pow_batch: true,
            role_id: role_config.role_id,
            num_queries: role_config.num_queries,
            batching_bits: role_config.batching_bits,
            query_bits: 22,
            log_blowup: role_config.log_blowup,
            r_rounds: 21,
            role_config_recv_mult: 1,
            ..Default::default()
        };

        let alpha_sender = RecursionWhirBatchEvalRow {
            proof_idx,
            is_start: true,
            is_group_start: true,
            log_height: 5,
            role_id: role_config.role_id,
            role_num_queries: role_config.num_queries,
            role_batching_bits: role_config.batching_bits,
            role_log_blowup: 0,
            role_config_recv_mult: 1,
            alpha,
            pow_in: one_ext(),
            pow_seed_cnt: 1,
            ..Default::default()
        };
        let group_sender = RecursionWhirBatchEvalRow {
            proof_idx,
            is_group_end: true,
            log_height: 3,
            acc_out: claim,
            group_claim_send_mult: 1,
            ..Default::default()
        };

        // One deduped instance serving both merge rows (serve_cnt = 2).
        let leaf0 = RecursionWhirLeafStreamRow {
            proof_idx,
            is_unit_start: true,
            is_unit_end: true,
            idx: 0,
            serve_cnt: 2,
            log_height: 5,
            alpha,
            pow_in: one_ext(),
            acc_out: leaf_sum,
            ..Default::default()
        };

        let seed = RecursionWhirQueryFoldRow {
            proof_idx,
            is_seed: true,
            twiddle_bytes,
            twiddle_values,
            ..Default::default()
        };
        let query0 = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx: 0,
            cursor: 0,
            query_bits: 5,
            r_fold,
            is_merge: true,
            merge_beta: beta,
            merge_eq,
            cfr,
            leaf_sum,
            ..Default::default()
        };
        let query1 = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx: 1,
            cursor: 0,
            query_bits: 5,
            r_fold,
            is_merge: true,
            merge_beta: beta,
            merge_eq,
            cfr,
            leaf_sum,
            ..Default::default()
        };

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                whir: RecursionWhirRecord {
                    role_config_mults,
                    twiddle_mults,
                    round_rows: vec![role_consumer, group_consumer, round_payload],
                    batch_eval_rows: vec![alpha_sender, group_sender],
                    query_fold_rows: vec![seed, query0, query1],
                    leaf_stream_rows: vec![leaf0],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn sample_whir_internal_chain_record() -> RecursionRecord {
        let proof_idx = 17;
        let alpha = ext(5);
        let pow0 = one_ext();
        let pow1 = ext(30);
        let acc0 = [F::zero(); D_EF];
        let acc1 = ext(40);
        let base0 = [F::zero(); D_EF];
        let base1 = ext(50);

        let eval0 = RecursionWhirBatchEvalRow {
            proof_idx,
            is_start: true,
            chain_recv_cursor: 0,
            chain_send_cursor: 1,
            alpha,
            pow_in: pow0,
            acc_in: acc0,
            group_base_in: base0,
            pow_out: pow1,
            acc_out: acc1,
            group_base_out: base1,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            ..Default::default()
        };
        let eval1 = RecursionWhirBatchEvalRow {
            proof_idx,
            is_value: true,
            chain_recv_cursor: 1,
            chain_send_cursor: 0,
            alpha,
            pow_in: pow1,
            acc_in: acc1,
            group_base_in: base1,
            pow_out: pow0,
            acc_out: acc0,
            group_base_out: base0,
            chain_recv_mult: 1,
            chain_send_mult: 1,
            ..Default::default()
        };

        // One linear instance across the base and ext chips —
        // base row starts (no recv), ext row ends (no send).
        let leaf0 = RecursionWhirLeafStreamRow {
            proof_idx,
            is_unit_start: true,
            idx: 0,
            chain_recv_cursor: 0,
            chain_send_cursor: 1,
            log_height: 5,
            batch_id: 1,
            alpha,
            pow_in: pow0,
            acc_in: acc0,
            pow_out: pow1,
            acc_out: acc1,
            ..Default::default()
        };
        let leaf_ext = RecursionWhirLeafExtStreamRow {
            proof_idx,
            is_unit_end: true,
            idx: 0,
            chain_recv_cursor: 1,
            chain_send_cursor: 2,
            log_height: 5,
            batch_id: WHIR_BATCH_PERMUTATION,
            chain_recv_log_height: 5,
            chain_recv_batch_id: 1,
            is_unit_key_start: true,
            alpha,
            pow_in: pow1,
            acc_in: acc1,
            pow_out: pow0,
            acc_out: acc0,
            chunk_masks: core::array::from_fn(|block| {
                core::array::from_fn(|limb| block == 0 && limb < D_EF)
            }),
            ..Default::default()
        };

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                whir: RecursionWhirRecord {
                    batch_eval_rows: vec![eval0, eval1],
                    leaf_stream_rows: vec![leaf0],
                    leaf_ext_stream_rows: vec![leaf_ext.into()],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn sample_whir_authenticated_shape_record() -> RecursionRecord {
        let proof_idx = 19;
        let chip = RecursionProofShapeChip {
            chip_idx: 0,
            static_chip_id: 11,
            stable_air_id: 43,
            log_height: 5,
            prep_width: 2,
            main_width: 3,
            perm_width: 4,
            constraint_count: 7,
            gate_count: 7,
        };
        let opening = ext(90);
        let num_rounds = 5;
        let c_chips = 1;
        let num_public_values = 2;

        let summary_consumer = RecursionWhirRoundRow {
            proof_idx,
            is_pow_batch: true,
            r_rounds: num_rounds,
            c_chips,
            num_public_values,
            summary_recv_mult: 1,
            ..Default::default()
        };
        let height_consumer = RecursionWhirRoundRow {
            proof_idx,
            is_preamble: true,
            height_group_rank: 0,
            height_group_log_height: chip.log_height,
            height_group_recv_mult: 1,
            ..Default::default()
        };
        let opening_consumer = RecursionWhirRoundRow {
            proof_idx,
            is_round: true,
            opening_idx: num_rounds - 1 - 1,
            opening_point: opening,
            opening_point_recv_mult: 1,
            ..Default::default()
        };

        let prep_dim = RecursionWhirBatchEvalRow {
            proof_idx,
            batch_id: PROOF_SHAPE_BATCH_PREPROCESSED,
            batch_pos: 0,
            chip_idx: chip.chip_idx,
            static_chip_id: chip.static_chip_id,
            width: chip.prep_width,
            log_height: chip.log_height,
            batch_dim_recv_mult: 1,
            ..Default::default()
        };
        let main_dim = RecursionWhirBatchEvalRow {
            proof_idx,
            batch_id: PROOF_SHAPE_BATCH_MAIN,
            batch_pos: chip.chip_idx,
            chip_idx: chip.chip_idx,
            static_chip_id: chip.static_chip_id,
            width: chip.main_width,
            log_height: chip.log_height,
            batch_dim_recv_mult: 1,
            ..Default::default()
        };
        let perm_dim = RecursionWhirBatchEvalRow {
            proof_idx,
            batch_id: PROOF_SHAPE_BATCH_PERMUTATION,
            batch_pos: chip.chip_idx,
            chip_idx: chip.chip_idx,
            static_chip_id: chip.static_chip_id,
            width: chip.perm_width,
            log_height: chip.log_height,
            batch_dim_recv_mult: 1,
            ..Default::default()
        };

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                proof_shape: RecursionProofShapeRecord {
                    role_id: WHIR_ROLE_CORE,
                    num_public_values,
                    vk_meta: vec![
                        F::zero();
                        crate::proof_shape_dt::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
                    ],
                    vk_meta_send_mults: vec![
                        0;
                        crate::proof_shape_dt::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
                    ],
                    public_values: vec![F::from_canonical_usize(1), F::from_canonical_usize(2)],
                    chips: vec![chip],
                    publish_whir_inputs: true,
                    ..Default::default()
                },
                batch_constraint: RecursionBatchConstraintRecord {
                    num_public_values,
                    num_rounds,
                    c_chips,
                    eq_challenges: vec![[F::zero(); D_EF]; num_rounds],
                    rounds: vec![RecursionSumcheckRoundRecord {
                        round_idx: 1,
                        challenge: opening,
                        ..Default::default()
                    }],
                    publish_opening_point: true,
                    ..Default::default()
                },
                whir: RecursionWhirRecord {
                    round_rows: vec![summary_consumer, height_consumer, opening_consumer],
                    batch_eval_rows: vec![prep_dim, main_dim, perm_dim],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn sample_whir_transcript_merkle_record() -> RecursionRecord {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let proof_idx = 23;
        let commit_id = 100;
        let in_digest = core::array::from_fn(|idx| F::from_canonical_usize(100 + idx));
        let sibling = core::array::from_fn(|idx| F::from_canonical_usize(120 + idx));
        let root_row0 = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            0,
            0,
            in_digest,
            sibling,
            true,
            &poseidon2_memo,
        );
        let root_row1 = RecursionMerklePathRow::path_compress(
            proof_idx,
            commit_id,
            0,
            0,
            in_digest,
            sibling,
            true,
            &poseidon2_memo,
        );
        let root = digest_from_poseidon_output(root_row0.output);

        let mut pow_events = [F::zero(); 32];
        pow_events[0] = F::from_canonical_usize(10);
        pow_events[1] = F::from_canonical_usize(11);
        pow_events[2] = F::from_canonical_usize(12);
        let pow_row = RecursionWhirRoundRow {
            proof_idx,
            is_pow_batch: true,
            tidx: 0,
            event_value: pow_events,
            ..Default::default()
        };

        let mut preamble_events = [F::zero(); 32];
        preamble_events[..DIGEST_SIZE].copy_from_slice(&root);
        let preamble_row = RecursionWhirRoundRow {
            proof_idx,
            is_preamble: true,
            tidx: 3,
            num_queries: 2,
            commit_id,
            commit_root: root,
            event_value: preamble_events,
            commitment_root_send_mult: 2,
            ..Default::default()
        };

        let query_sample = F::from_canonical_usize(13);
        let seed_row = RecursionWhirQueryFoldRow {
            proof_idx,
            is_seed: true,
            query_idx: 0,
            w_qbase: 11,
            query_sample_raw: query_sample,
            ..Default::default()
        };

        let f0 = ext(30);
        let f1 = ext(40);
        let depth = 4;
        let seed_idx = 7;
        // Pair-leaf identity: depth = query_bits - 1 - cursor.
        let query_bits = depth + 1;
        let unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE, depth);
        let query_row = RecursionWhirQueryFoldRow {
            proof_idx,
            is_round: true,
            query_idx: 0,
            query_bits,
            chain_send_idx: F::from_canonical_usize(seed_idx),
            f0,
            f1,
            ..Default::default()
        };

        let chunk0 = query_pair_leaf_chunk_for_test(query_row, 0);
        let mask0 = query_pair_leaf_mask_for_test(0);
        let chunk1 = query_pair_leaf_chunk_for_test(query_row, 1);
        let mask1 = query_pair_leaf_mask_for_test(1);
        let leaf0 = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            commit_id,
            0,
            seed_idx,
            1,
            false,
            true,
            false,
            [F::zero(); POSEIDON2_WIDTH],
            chunk0,
            mask0,
            &poseidon2_memo,
        );
        let leaf1 = RecursionMerklePathRow::leaf_absorb(
            proof_idx,
            unit_key,
            commit_id,
            1,
            seed_idx,
            1,
            false,
            false,
            true,
            leaf0.output,
            chunk1,
            mask1,
            &poseidon2_memo,
        );

        let mut events = Vec::new();
        events.push(transcript_event(0, RecursionTranscriptEventKind::Observe, pow_events[0]));
        events.push(transcript_event(1, RecursionTranscriptEventKind::Observe, pow_events[1]));
        events.push(transcript_event(2, RecursionTranscriptEventKind::Sample, pow_events[2]));
        for (idx, value) in root.into_iter().enumerate() {
            events.push(transcript_event(3 + idx, RecursionTranscriptEventKind::Observe, value));
        }
        events.push(transcript_event(11, RecursionTranscriptEventKind::Sample, query_sample));

        let mut proof = RecursionProofRecord {
            proof_idx,
            transcript: crate::system_dt::RecursionTranscriptRecord {
                events,
                ..Default::default()
            },
            whir: RecursionWhirRecord {
                round_rows: vec![pow_row, preamble_row],
                query_fold_rows: vec![seed_row, query_row],
                ..Default::default()
            },
            ..Default::default()
        };
        proof.merkle_path.push_row(root_row0);
        proof.merkle_path.push_row(root_row1);
        proof.merkle_path.push_row(leaf0);
        proof.merkle_path.push_row(leaf1);

        RecursionRecord { proof_records: vec![proof], ..Default::default() }
    }

    fn sample_batch_transcript_record() -> RecursionRecord {
        let proof_idx = 29;
        let num_public_values = 0;
        let num_rounds = 3;
        let c_chips = 1;
        let perm_alpha = ext(100);
        let perm_beta = ext(110);
        let alpha = ext(120);
        let eq_challenges = vec![ext(200), ext(210), ext(220)];
        let rounds = (0..num_rounds)
            .map(|round_idx| RecursionSumcheckRoundRecord {
                round_idx,
                evals: core::array::from_fn(|eval_idx| {
                    ext(300 + round_idx * 100 + eval_idx * D_EF)
                }),
                challenge: ext(600 + round_idx * 10),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let layout = BatchTranscriptLayout::new(num_public_values, c_chips, num_rounds, true);
        let mut events = Vec::new();
        {
            let mut push_ext_events =
                |base: usize, kind: RecursionTranscriptEventKind, values: [F; D_EF]| {
                    events.extend(
                        values
                            .into_iter()
                            .enumerate()
                            .map(|(offset, value)| transcript_event(base + offset, kind, value)),
                    );
                };

            push_ext_events(layout.e3_tidx(), RecursionTranscriptEventKind::Sample, perm_alpha);
            push_ext_events(
                layout.e3_tidx() + D_EF,
                RecursionTranscriptEventKind::Sample,
                perm_beta,
            );
            push_ext_events(layout.e7_tidx(), RecursionTranscriptEventKind::Sample, alpha);
            for (opening_idx, eq_challenge) in eq_challenges.iter().copied().enumerate() {
                push_ext_events(
                    layout.e8_tidx(opening_idx),
                    RecursionTranscriptEventKind::Sample,
                    eq_challenge,
                );
            }
            for round in &rounds {
                let e9 = layout.e9_tidx(round.round_idx);
                for (eval_idx, eval) in round.evals.into_iter().enumerate() {
                    push_ext_events(
                        e9 + eval_idx * D_EF,
                        RecursionTranscriptEventKind::Observe,
                        eval,
                    );
                }
                push_ext_events(
                    e9 + BATCH_SUMCHECK_EVALS * D_EF,
                    RecursionTranscriptEventKind::Sample,
                    round.challenge,
                );
            }
        }

        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                transcript: crate::system_dt::RecursionTranscriptRecord {
                    events,
                    ..Default::default()
                },
                batch_constraint: RecursionBatchConstraintRecord {
                    num_public_values,
                    num_rounds,
                    c_chips,
                    perm_alpha,
                    perm_beta,
                    alpha,
                    eq_challenges,
                    rounds,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn assert_query_chain_balanced(record: &RecursionRecord) {
        let residual = query_chain_residual(record);
        assert!(residual.is_empty(), "unexpected WhirQueryChain residual: {residual:?}");
    }

    fn assert_query_init_balanced(record: &RecursionRecord) {
        let residual = query_init_residual(record);
        assert!(residual.is_empty(), "unexpected WhirQueryInit residual: {residual:?}");
    }

    fn sample_query_geometry_record(query_bits: usize, r_rounds: usize) -> RecursionRecord {
        let proof_idx = 0;
        let band = sample_band_for_query_bits(query_bits).expect("sample band for fixture");
        let seed = RecursionWhirQueryFoldRow {
            proof_idx,
            is_seed: true,
            w_qbase: 17,
            query_bits,
            r_rounds,
            query_sample_shift: band.shift,
            query_sample_high_max: band.high_max,
            query_sample_high_bits: band.high_bits,
            ..Default::default()
        };
        let final_row = RecursionWhirRoundRow {
            proof_idx,
            is_final: true,
            w_qbase: 17,
            query_bits,
            r_rounds,
            query_init_mult: 1,
            ..Default::default()
        };
        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx,
                whir: RecursionWhirRecord {
                    round_rows: vec![final_row],
                    query_fold_rows: vec![seed],
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}

fn twiddle_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for proof in &record.proof_records {
        for (byte, row) in proof.whir.twiddle_mults.iter().enumerate() {
            for (table_id, mult) in row.iter().copied().enumerate() {
                apply_residual(
                    &mut residual,
                    vec![
                        table_id as u32,
                        byte as u32,
                        twiddle_value(table_id, byte).as_canonical_u32(),
                    ],
                    i64::from(mult),
                );
            }
        }
    }
    for row in whir_query_fold_row_iter(record).filter(|row| row.is_seed) {
        for table_id in 0..WHIR_TWIDDLE_TABLES {
            apply_residual(
                &mut residual,
                vec![
                    table_id as u32,
                    u32::from(row.twiddle_bytes[table_id]),
                    row.twiddle_values[table_id].as_canonical_u32(),
                ],
                -1,
            );
        }
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn sample_band_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_query_fold_row_iter(record) {
        if !row.is_seed {
            continue;
        }
        if let Some(config) = sample_band_for_query_bits(row.query_bits) {
            apply_residual(&mut residual, sample_band_key(config), 1);
        }
        apply_residual(
            &mut residual,
            vec![
                row.query_bits as u32,
                row.query_sample_shift as u32,
                row.query_sample_high_max as u32,
                row.query_sample_high_bits as u32,
            ],
            -1,
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn sample_band_key(config: WhirSampleBandConfig) -> Vec<u32> {
    vec![
        config.query_bits as u32,
        config.shift as u32,
        config.high_max as u32,
        config.high_bits as u32,
    ]
}

fn leaf_pow_seed_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_batch_eval_row_iter(record) {
        if row.is_group_start {
            apply_residual(
                &mut residual,
                leaf_pow_seed_key(
                    row.proof_idx,
                    row.log_height + row.role_log_blowup,
                    row.alpha,
                    row.pow_in,
                ),
                row.pow_seed_cnt as i64,
            );
        }
    }
    for row in whir_leaf_stream_row_iter(record).filter(|row| row.is_unit_start) {
        apply_residual(
            &mut residual,
            leaf_pow_seed_key(row.proof_idx, row.log_height, row.alpha, row.pow_in),
            -1,
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn group_claim_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_batch_eval_row_iter(record) {
        let claim = core::array::from_fn(|idx| row.acc_out[idx] - row.group_base_in[idx]);
        apply_residual(
            &mut residual,
            group_claim_key(row.proof_idx, row.log_height, claim),
            flag_i64(row.is_group_end),
        );
    }
    for row in whir_round_row_iter(record) {
        apply_residual(
            &mut residual,
            group_claim_key(row.proof_idx, row.group_claim_log_height, row.group_claim),
            -(flag_i64(row.is_preamble) + flag_i64(row.is_merge)),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn round_bcast_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_round_row_iter(record) {
        apply_residual(&mut residual, round_bcast_key_from_round(*row), row.bcast_mult as i64);
    }
    for row in whir_query_fold_row_iter(record).filter(|row| row.is_round) {
        apply_residual(&mut residual, round_bcast_key_from_query(*row), -1);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn query_leaf_sum_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_leaf_stream_row_iter(record).filter(|row| row.is_unit_end) {
        apply_residual(
            &mut residual,
            query_leaf_sum_key(row.proof_idx, row.idx, row.log_height, row.acc_out),
            row.serve_cnt as i64,
        );
    }
    for row in whir_leaf_ext_stream_rows(record).into_iter().filter(|row| row.is_unit_end) {
        apply_residual(
            &mut residual,
            query_leaf_sum_key(row.proof_idx, row.idx, row.log_height, row.acc_out),
            row.serve_cnt as i64,
        );
    }
    for row in whir_query_fold_row_iter(record).filter(|row| row.is_merge) {
        let merge_idx = row.idx.as_canonical_u32() as usize;
        apply_residual(
            &mut residual,
            query_leaf_sum_key(row.proof_idx, merge_idx, row.query_bits - row.cursor, row.leaf_sum),
            -1,
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn round_chain_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_round_row_iter(record) {
        let values = round_row(row);
        let cols: &WhirRoundCols<F> = values.as_slice().borrow();
        let mult = field_i64(cols.is_valid) - field_i64(cols.is_final_perm);
        apply_residual(&mut residual, round_chain_key_recv(cols), -mult);
        apply_residual(&mut residual, round_chain_key_send(cols), mult);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn final_root_chain_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_round_row_iter(record) {
        let values = round_row(row);
        let cols: &WhirRoundCols<F> = values.as_slice().borrow();
        let mult = field_i64(cols.is_final) + field_i64(cols.is_final_perm);
        apply_residual(&mut residual, final_root_chain_key_recv(cols), -mult);
        apply_residual(&mut residual, final_root_chain_key_send(cols), mult);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn query_chain_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_query_fold_row_iter(record) {
        let values = query_fold_row(row);
        let cols: &WhirQueryFoldCols<F> = values.as_slice().borrow();
        apply_residual(&mut residual, query_chain_key_recv(cols), -1);
        apply_residual(&mut residual, query_chain_key_send(cols), 1);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn query_init_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_round_row_iter(record) {
        let values = round_row(row);
        let cols: &WhirRoundCols<F> = values.as_slice().borrow();
        apply_residual(
            &mut residual,
            query_init_key_from_round(cols),
            field_i64(cols.query_init_mult),
        );
    }
    for row in whir_query_fold_row_iter(record) {
        let values = query_fold_row(row);
        let cols: &WhirQueryFoldCols<F> = values.as_slice().borrow();
        apply_residual(&mut residual, query_init_key_from_query(cols), -field_i64(cols.is_seed));
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn eval_chain_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_batch_eval_row_iter(record) {
        let values = batch_eval_row(row);
        let cols: &WhirBatchEvalCols<F> = values.as_slice().borrow();
        let mult = field_i64(cols.is_start) + field_i64(cols.is_value);
        apply_residual(&mut residual, eval_chain_key_recv(cols), -mult);
        apply_residual(&mut residual, eval_chain_key_send(cols), mult);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn leaf_chain_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_leaf_stream_row_iter(record) {
        let values = leaf_stream_row(row);
        let cols: &WhirLeafStreamCols<F> = values.as_slice().borrow();
        let recv_mult = field_i64(cols.is_valid) - field_i64(cols.is_unit_start);
        let send_mult = field_i64(cols.is_valid) - field_i64(cols.is_unit_end);
        apply_residual(&mut residual, leaf_chain_key_recv(cols), -recv_mult);
        apply_residual(&mut residual, leaf_chain_key_send(cols), send_mult);
    }
    for row in whir_leaf_ext_stream_rows(record) {
        let values = leaf_ext_stream_row(row);
        let cols: &WhirLeafExtStreamCols<F> = values.as_slice().borrow();
        let recv_mult = field_i64(cols.element_masks[0]);
        let send_mult = field_i64(cols.element_masks[0]) - field_i64(cols.is_unit_end);
        apply_residual(&mut residual, leaf_ext_chain_key_recv(cols), -recv_mult);
        apply_residual(&mut residual, leaf_ext_chain_key_send(cols), send_mult);
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn batch_dim_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::Chip {
            proof_idx,
            chip,
            prev_prep_matrix_idx,
            publish_batch_dim,
            ..
        } = row
        {
            if !publish_batch_dim {
                continue;
            }
            if chip.has_prep() {
                apply_residual(
                    &mut residual,
                    batch_dim_key(
                        proof_idx,
                        PROOF_SHAPE_BATCH_PREPROCESSED,
                        prev_prep_matrix_idx,
                        chip.chip_idx,
                        chip.static_chip_id,
                        chip.prep_width,
                        chip.log_height,
                    ),
                    1,
                );
            }
            apply_residual(
                &mut residual,
                batch_dim_key(
                    proof_idx,
                    PROOF_SHAPE_BATCH_MAIN,
                    chip.chip_idx,
                    chip.chip_idx,
                    chip.static_chip_id,
                    chip.main_width,
                    chip.log_height,
                ),
                1,
            );
            apply_residual(
                &mut residual,
                batch_dim_key(
                    proof_idx,
                    PROOF_SHAPE_BATCH_PERMUTATION,
                    chip.chip_idx,
                    chip.chip_idx,
                    chip.static_chip_id,
                    chip.perm_width,
                    chip.log_height,
                ),
                1,
            );
        }
    }
    for row in whir_batch_eval_row_iter(record) {
        apply_residual(
            &mut residual,
            batch_dim_key(
                row.proof_idx,
                row.batch_id,
                row.batch_pos,
                row.chip_idx,
                row.static_chip_id,
                row.width,
                row.log_height,
            ),
            -(row.batch_dim_recv_mult as i64),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn summary_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::E5 { proof_idx, prev, summary_send_mult, .. } = row {
            let Some(proof) =
                record.proof_records.iter().find(|proof| proof.proof_idx == proof_idx)
            else {
                continue;
            };
            if summary_send_mult != 0 {
                apply_residual(
                    &mut residual,
                    summary_key(
                        proof_idx,
                        prev.first_log_height,
                        prev.chip_idx,
                        proof.proof_shape.num_public_values,
                        proof.proof_shape.segment_id_base(),
                    ),
                    summary_send_mult as i64,
                );
            }
        }
    }
    for row in batch_sumcheck_rows(record) {
        if let BatchSumcheckRow::Seed { proof_idx, num_rounds, c_chips, summary_id_base, .. } = row
        {
            let Some(proof) =
                record.proof_records.iter().find(|proof| proof.proof_idx == proof_idx)
            else {
                continue;
            };
            apply_residual(
                &mut residual,
                summary_key(
                    proof_idx,
                    num_rounds,
                    c_chips,
                    proof.proof_shape.num_public_values,
                    summary_id_base,
                ),
                -1,
            );
        }
    }
    for row in whir_round_row_iter(record) {
        let Some(proof) =
            record.proof_records.iter().find(|proof| proof.proof_idx == row.proof_idx)
        else {
            continue;
        };
        apply_residual(
            &mut residual,
            summary_key(
                row.proof_idx,
                row.r_rounds,
                row.c_chips,
                proof.proof_shape.num_public_values,
                row.summary_id_base,
            ),
            -flag_i64(row.is_pow_batch),
        );
    }
    for proof in &record.proof_records {
        if proof.proof_shape.publish_whir_inputs {
            apply_residual(
                &mut residual,
                summary_key(
                    proof.proof_idx,
                    proof.batch_constraint.num_rounds,
                    proof.batch_constraint.c_chips,
                    proof.proof_shape.num_public_values,
                    proof.proof_shape.segment_id_base(),
                ),
                -1,
            );
        }
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn height_group_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_height_set_rows(record) {
        if row.publish_external && row.member_count != 0 {
            apply_residual(
                &mut residual,
                height_group_key(row.proof_idx, row.rank, row.height_cursor),
                1,
            );
        }
    }
    for row in whir_round_row_iter(record) {
        apply_residual(
            &mut residual,
            height_group_key(row.proof_idx, row.height_group_rank, row.height_group_log_height),
            -(flag_i64(row.is_preamble) + flag_i64(row.is_merge)),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn opening_point_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in batch_sumcheck_rows(record) {
        if let BatchSumcheckRow::Round { proof_idx, num_rounds, round, .. } = row {
            let publish_opening_point = record
                .proof_records
                .iter()
                .find(|proof| proof.proof_idx == proof_idx)
                .is_some_and(|proof| proof.batch_constraint.publish_opening_point);
            if publish_opening_point {
                apply_residual(
                    &mut residual,
                    opening_point_key(proof_idx, num_rounds - 1 - round.round_idx, round.challenge),
                    1,
                );
            }
        }
    }
    for row in whir_round_row_iter(record) {
        apply_residual(
            &mut residual,
            opening_point_key(row.proof_idx, row.opening_idx, row.opening_point),
            -flag_i64(row.is_round),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn transcript_event_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for proof in &record.proof_records {
        let has_vk_header = transcript_has_canonical_vk_header(proof);
        for event in &proof.transcript.events {
            let is_vk_header = has_vk_header &&
                event.tidx < BATCH_VK_TAG_VERSION_LIMBS &&
                matches!(event.kind, RecursionTranscriptEventKind::Observe);
            apply_residual(
                &mut residual,
                transcript_event_key(
                    proof.proof_idx,
                    event.tidx,
                    matches!(event.kind, RecursionTranscriptEventKind::Sample),
                    event.value,
                ),
                1 + i64::from(is_vk_header),
            );
        }
    }
    apply_proof_shape_transcript_recvs(&mut residual, record);
    apply_batch_transcript_recvs(&mut residual, record);
    for row in whir_round_row_iter(record) {
        let values = round_row(row);
        let cols: &WhirRoundCols<F> = values.as_slice().borrow();
        for idx in 0..WHIR_ROUND_MAX_TRANSCRIPT_EVENTS {
            let mult = whir_round_event_mult_for_test(cols, idx);
            if mult != 0 {
                apply_residual(
                    &mut residual,
                    transcript_event_key(
                        cols.proof_idx.as_canonical_u32() as usize,
                        whir_round_event_tidx_for_test(cols, idx),
                        whir_round_event_is_sample_for_test(cols, idx),
                        cols.event_value[idx],
                    ),
                    -mult,
                );
            }
        }
    }
    for row in whir_batch_eval_row_iter(record).filter(|row| row.is_start) {
        for idx in 0..D_EF {
            apply_residual(
                &mut residual,
                transcript_event_key(row.proof_idx, row.alpha_tidx + idx, true, row.alpha[idx]),
                -1,
            );
        }
    }
    for row in whir_query_fold_row_iter(record).filter(|row| row.is_seed) {
        apply_residual(
            &mut residual,
            transcript_event_key(
                row.proof_idx,
                row.w_qbase + row.query_idx,
                true,
                row.query_sample_raw,
            ),
            -1,
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn apply_proof_shape_transcript_recvs(
    residual: &mut BTreeMap<Vec<u32>, i64>,
    record: &RecursionRecord,
) {
    for row in proof_shape_binder_rows(record) {
        match row {
            ProofShapeBinderRow::VkCommit { proof_idx, values, .. } => {
                apply_transcript_recv_range(
                    residual,
                    proof_idx,
                    0,
                    [true; BATCH_VK_TAG_VERSION_LIMBS],
                    [
                        F::from_canonical_u32(crate::batch_constraint_dt::BATCH_VK_TAG_V1),
                        F::from_canonical_u32(crate::batch_constraint_dt::BATCH_VK_VERSION_V1),
                    ],
                );
                apply_transcript_recv_range(
                    residual,
                    proof_idx,
                    BATCH_VK_TAG_VERSION_LIMBS,
                    [true; 8],
                    values,
                );
            }
            ProofShapeBinderRow::VkMeta { proof_idx, base, values, shape_mask, .. } => {
                apply_transcript_recv_range(
                    residual,
                    proof_idx,
                    base + BATCH_VK_TAG_VERSION_LIMBS,
                    shape_mask,
                    values,
                );
            }
            ProofShapeBinderRow::E1 { proof_idx, tidx_base: base, values, .. } |
            ProofShapeBinderRow::E5 { proof_idx, tidx_base: base, values, .. } => {
                apply_transcript_recv_range(residual, proof_idx, base, [true; 8], values);
            }
            ProofShapeBinderRow::PublicValues { proof_idx, base, values, mask, .. } => {
                apply_transcript_recv_range(residual, proof_idx, base, mask, values);
            }
            ProofShapeBinderRow::ActiveShapeHeader { proof_idx, tidx_base, c_chips, .. } => {
                let mut values = [F::zero(); 8];
                values[0] = F::from_canonical_u32(dt_stark::air::ACTIVE_SHAPE_TAG_V1);
                values[1] = F::from_canonical_u32(dt_stark::air::ACTIVE_SHAPE_VERSION_V2);
                values[2] = F::from_canonical_usize(c_chips);
                apply_transcript_recv_range(
                    residual,
                    proof_idx,
                    tidx_base,
                    [true, true, true, false, false, false, false, false],
                    values,
                );
            }
            ProofShapeBinderRow::Chip { proof_idx, chip, prev_tidx_acc, .. } => {
                let mut values = [F::zero(); 8];
                values[0] = F::from_canonical_u32(chip.stable_air_id & 0xffff);
                values[1] = F::from_canonical_u32(chip.stable_air_id >> 16);
                values[2] = F::from_canonical_usize(chip.log_height);
                values[3] = F::from_canonical_usize(chip.main_width);
                values[4] = F::from_canonical_usize(chip.chip_idx);
                apply_transcript_recv_range(
                    residual,
                    proof_idx,
                    prev_tidx_acc,
                    [true, true, true, true, true, false, false, false],
                    values,
                );
            }
        }
    }
}

fn apply_transcript_recv_range<const N: usize>(
    residual: &mut BTreeMap<Vec<u32>, i64>,
    proof_idx: usize,
    base: usize,
    mask: [bool; N],
    values: [F; N],
) {
    for (offset, (enabled, value)) in mask.into_iter().zip(values).enumerate() {
        if enabled {
            apply_residual(
                residual,
                transcript_event_key(proof_idx, base + offset, false, value),
                -1,
            );
        }
    }
}

fn apply_batch_transcript_recvs(residual: &mut BTreeMap<Vec<u32>, i64>, record: &RecursionRecord) {
    for row in batch_transcript_input_rows(record) {
        let BatchTranscriptInputRow::Fused { proof_idx, c_chips, perm_alpha, perm_beta, alpha } =
            row;
        let Some(proof) = record.proof_records.iter().find(|proof| proof.proof_idx == proof_idx)
        else {
            continue;
        };
        let layout = BatchTranscriptLayout::new(
            proof.proof_shape.num_public_values,
            c_chips,
            0,
            proof.proof_shape.role_id == 0,
        );
        if transcript_has_canonical_vk_header(proof) {
            apply_residual(
                residual,
                transcript_event_key(
                    proof_idx,
                    0,
                    false,
                    F::from_canonical_u32(crate::batch_constraint_dt::BATCH_VK_TAG_V1),
                ),
                -1,
            );
            apply_residual(
                residual,
                transcript_event_key(
                    proof_idx,
                    1,
                    false,
                    F::from_canonical_u32(crate::batch_constraint_dt::BATCH_VK_VERSION_V1),
                ),
                -1,
            );
        }
        for (offset, value) in perm_alpha.into_iter().chain(perm_beta).enumerate() {
            apply_residual(
                residual,
                transcript_event_key(proof_idx, layout.e3_tidx() + offset, true, value),
                -1,
            );
        }
        for (offset, value) in alpha.into_iter().enumerate() {
            apply_residual(
                residual,
                transcript_event_key(proof_idx, layout.e7_tidx() + offset, true, value),
                -1,
            );
        }
    }

    for row in batch_sumcheck_rows(record) {
        let BatchSumcheckRow::Round {
            proof_idx,
            num_public_values,
            num_rounds,
            c_chips,
            round,
            eq_challenge,
        } = row
        else {
            continue;
        };
        let layout = BatchTranscriptLayout::new(
            num_public_values,
            c_chips,
            num_rounds,
            record
                .proof_records
                .iter()
                .find(|proof| proof.proof_idx == proof_idx)
                .is_some_and(|proof| proof.proof_shape.role_id == 0),
        );
        for (offset, value) in eq_challenge.into_iter().enumerate() {
            apply_residual(
                residual,
                transcript_event_key(
                    proof_idx,
                    layout.e8_tidx(num_rounds - 1 - round.round_idx) + offset,
                    true,
                    value,
                ),
                -1,
            );
        }
        let e9 = layout.e9_tidx(round.round_idx);
        for (offset, value) in round.evals.into_iter().flatten().enumerate() {
            apply_residual(
                residual,
                transcript_event_key(proof_idx, e9 + offset, false, value),
                -1,
            );
        }
        for (offset, value) in round.challenge.into_iter().enumerate() {
            apply_residual(
                residual,
                transcript_event_key(
                    proof_idx,
                    e9 + BATCH_SUMCHECK_EVALS * D_EF + offset,
                    true,
                    value,
                ),
                -1,
            );
        }
    }
}

fn transcript_has_canonical_vk_header(proof: &RecursionProofRecord) -> bool {
    [crate::batch_constraint_dt::BATCH_VK_TAG_V1, crate::batch_constraint_dt::BATCH_VK_VERSION_V1]
        .into_iter()
        .enumerate()
        .all(|(tidx, expected)| {
            proof.transcript.events.iter().any(|event| {
                event.tidx == tidx &&
                    matches!(event.kind, RecursionTranscriptEventKind::Observe) &&
                    event.value == F::from_canonical_u32(expected)
            })
        })
}

fn commitment_root_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_round_row_iter(record) {
        apply_residual(
            &mut residual,
            commitment_root_key(row.proof_idx, row.commit_id, row.commit_root),
            row.commitment_root_send_mult as i64,
        );
    }
    for row in proof_shape_binder_rows(record) {
        match row {
            ProofShapeBinderRow::VkCommit {
                proof_idx, role_id, values, publish_external, ..
            } => apply_residual(
                &mut residual,
                commitment_root_key(
                    proof_idx,
                    crate::proof_shape_dt::PROOF_SHAPE_COMMIT_VK,
                    values,
                ),
                proof_shape_root_mult(role_id, publish_external),
            ),
            ProofShapeBinderRow::E1 { proof_idx, role_id, values, publish_external, .. } => {
                apply_residual(
                    &mut residual,
                    commitment_root_key(
                        proof_idx,
                        crate::proof_shape_dt::PROOF_SHAPE_COMMIT_MAIN,
                        values,
                    ),
                    proof_shape_root_mult(role_id, publish_external),
                )
            }
            ProofShapeBinderRow::E5 { proof_idx, role_id, values, publish_external, .. } => {
                apply_residual(
                    &mut residual,
                    commitment_root_key(
                        proof_idx,
                        crate::proof_shape_dt::PROOF_SHAPE_COMMIT_PERMUTATION,
                        values,
                    ),
                    proof_shape_root_mult(role_id, publish_external),
                )
            }
            _ => {}
        }
    }
    for row in merkle_row_iter(record).filter(|row| row.root_cnt != 0) {
        apply_residual(
            &mut residual,
            commitment_root_key(
                row.proof_idx,
                row.commit_id,
                digest_from_poseidon_output(row.output),
            ),
            -(row.root_cnt as i64),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn merkle_leaf_block_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in whir_query_fold_row_iter(record).filter(|row| row.is_round) {
        let depth = row.query_bits - 1 - row.cursor;
        let unit_key = whir_unit_key(WHIR_IOPP_ORACLE_PATH_SLOT_BASE + row.cursor, depth);
        let pair_idx = row.chain_send_idx.as_canonical_u32() as usize;
        for block in 0..WHIR_QUERY_PAIR_LEAF_BLOCKS {
            apply_residual(
                &mut residual,
                merkle_leaf_block_key(
                    row.proof_idx,
                    100 + row.cursor,
                    unit_key,
                    pair_idx,
                    block,
                    query_pair_leaf_mask_for_test(block),
                    query_pair_leaf_chunk_for_test(*row, block),
                ),
                1,
            );
        }
    }
    for row in whir_leaf_stream_row_iter(record) {
        apply_residual(
            &mut residual,
            merkle_leaf_block_key(
                row.proof_idx,
                row.batch_id,
                row.unit_key,
                row.idx,
                row.block_idx,
                row.chunk_mask,
                row.values,
            ),
            row.chunk_mask[0] as i64,
        );
    }
    for row in whir_leaf_ext_stream_rows(record) {
        let values = leaf_ext_stream_row(row);
        let cols: &WhirLeafExtStreamCols<F> = values.as_slice().borrow();
        let unit_key = whir_unit_key(
            WHIR_INPUT_PERMUTATION_PATH_SLOT,
            cols.log_height.as_canonical_u32() as usize,
        );
        for block in 0..WHIR_LEAF_BLOCKS_PER_ROW {
            let mask = core::array::from_fn(|idx| {
                cols.element_masks[(block * WHIR_LEAF_BASE_LIMBS_PER_ROW + idx) / D_EF] == F::one()
            });
            let chunk =
                core::array::from_fn(|idx| cols.values[block * WHIR_LEAF_BASE_LIMBS_PER_ROW + idx]);
            apply_residual(
                &mut residual,
                merkle_leaf_block_key(
                    cols.proof_idx.as_canonical_u32() as usize,
                    WHIR_BATCH_PERMUTATION,
                    unit_key,
                    cols.idx.as_canonical_u32() as usize,
                    cols.block_idx.as_canonical_u32() as usize + block,
                    mask,
                    chunk,
                ),
                field_i64(cols.element_masks[block * WHIR_LEAF_BASE_LIMBS_PER_ROW / D_EF]),
            );
        }
    }
    for row in
        merkle_row_iter(record).filter(|row| matches!(row.op, RecursionMerklePathOp::LeafAbsorb))
    {
        apply_residual(
            &mut residual,
            merkle_leaf_block_key(
                row.proof_idx,
                row.commit_id,
                row.unit_key,
                row.cur_idx,
                row.block_idx,
                row.chunk_mask,
                row.chunk,
            ),
            -(row.absorb_cnt as i64),
        );
    }
    residual.retain(|_, value| *value != 0);
    residual
}

fn round_chain_key_recv(cols: &WhirRoundCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.chain_recv_round.as_canonical_u32(),
        cols.chain_recv_tidx.as_canonical_u32(),
    ];
    key.extend(cols.chain_recv_claim.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.chain_recv_eq.iter().map(|value| value.as_canonical_u32()));
    key.push(cols.chain_recv_pending_is_merge.as_canonical_u32());
    key.extend(cols.chain_recv_pending_beta.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.chain_recv_pending_eq.iter().map(|value| value.as_canonical_u32()));
    key
}

fn round_chain_key_send(cols: &WhirRoundCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.chain_send_round.as_canonical_u32(),
        cols.chain_send_tidx.as_canonical_u32(),
    ];
    key.extend(cols.chain_send_claim.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.chain_send_eq.iter().map(|value| value.as_canonical_u32()));
    key.push(cols.chain_send_pending_is_merge.as_canonical_u32());
    key.extend(cols.chain_send_pending_beta.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.chain_send_pending_eq.iter().map(|value| value.as_canonical_u32()));
    key
}

fn eval_chain_key_recv(cols: &WhirBatchEvalCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.chain_recv_cursor.as_canonical_u32(),
        cols.chain_recv_log_height.as_canonical_u32(),
        cols.chain_recv_batch_id.as_canonical_u32(),
        cols.chain_recv_batch_pos.as_canonical_u32(),
        cols.chain_recv_value_idx.as_canonical_u32(),
        cols.chain_recv_segment_element_count.as_canonical_u32(),
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_in.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_in.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.group_base_in.iter().map(|value| value.as_canonical_u32()));
    key
}

fn eval_chain_key_send(cols: &WhirBatchEvalCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.chain_send_cursor.as_canonical_u32(),
        cols.log_height.as_canonical_u32(),
        cols.batch_id.as_canonical_u32(),
        cols.batch_pos.as_canonical_u32(),
        cols.value_idx.as_canonical_u32(),
        cols.segment_element_count.as_canonical_u32(),
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_out.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_out.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.group_base_out.iter().map(|value| value.as_canonical_u32()));
    key
}

fn leaf_chain_key_recv(cols: &WhirLeafStreamCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.idx.as_canonical_u32(),
        cols.chain_recv_cursor.as_canonical_u32(),
        cols.chain_recv_log_height.as_canonical_u32(),
        cols.chain_recv_batch_id.as_canonical_u32(),
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_in.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_in.iter().map(|value| value.as_canonical_u32()));
    key
}

fn leaf_chain_key_send(cols: &WhirLeafStreamCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.idx.as_canonical_u32(),
        cols.chain_send_cursor.as_canonical_u32(),
        cols.log_height.as_canonical_u32(),
        cols.batch_id.as_canonical_u32(),
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_out.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_out.iter().map(|value| value.as_canonical_u32()));
    key
}

fn leaf_ext_chain_key_recv(cols: &WhirLeafExtStreamCols<F>) -> Vec<u32> {
    let recv_batch = F::from_canonical_usize(WHIR_BATCH_PERMUTATION) - cols.is_unit_key_start;
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.idx.as_canonical_u32(),
        cols.chain_recv_cursor.as_canonical_u32(),
        cols.log_height.as_canonical_u32(),
        recv_batch.as_canonical_u32(),
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_in.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_in.iter().map(|value| value.as_canonical_u32()));
    key
}

fn leaf_ext_chain_key_send(cols: &WhirLeafExtStreamCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.idx.as_canonical_u32(),
        (cols.chain_recv_cursor + F::one()).as_canonical_u32(),
        cols.log_height.as_canonical_u32(),
        WHIR_BATCH_PERMUTATION as u32,
    ];
    key.extend(cols.alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.pow_out.iter().map(|value| value.as_canonical_u32()));
    key.extend(cols.acc_out.iter().map(|value| value.as_canonical_u32()));
    key
}

fn batch_dim_key(
    proof_idx: usize,
    batch_id: usize,
    batch_pos: usize,
    chip_idx: usize,
    static_chip_id: usize,
    width: usize,
    log_height: usize,
) -> Vec<u32> {
    vec![
        proof_idx as u32,
        batch_id as u32,
        batch_pos as u32,
        chip_idx as u32,
        static_chip_id as u32,
        width as u32,
        log_height as u32,
    ]
}

fn summary_key(
    proof_idx: usize,
    r_rounds: usize,
    c_chips: usize,
    num_public_values: usize,
    static_chip_id_base: usize,
) -> Vec<u32> {
    vec![
        proof_idx as u32,
        r_rounds as u32,
        c_chips as u32,
        num_public_values as u32,
        static_chip_id_base as u32,
    ]
}

fn height_group_key(proof_idx: usize, rank: usize, log_height: usize) -> Vec<u32> {
    vec![proof_idx as u32, rank as u32, log_height as u32]
}

fn opening_point_key(proof_idx: usize, opening_idx: usize, opening_point: [F; D_EF]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, opening_idx as u32];
    key.extend(opening_point.iter().map(|value| value.as_canonical_u32()));
    key
}

fn transcript_event_key(proof_idx: usize, tidx: usize, is_sample: bool, value: F) -> Vec<u32> {
    vec![proof_idx as u32, tidx as u32, is_sample as u32, value.as_canonical_u32()]
}

fn commitment_root_key(proof_idx: usize, commit_id: usize, root: [F; DIGEST_SIZE]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, commit_id as u32];
    key.extend(root.iter().map(|value| value.as_canonical_u32()));
    key
}

fn merkle_leaf_block_key(
    proof_idx: usize,
    commit_id: usize,
    unit_key: usize,
    idx: usize,
    block_idx: usize,
    mask: [bool; DIGEST_SIZE],
    chunk: [F; DIGEST_SIZE],
) -> Vec<u32> {
    let mut key =
        vec![proof_idx as u32, commit_id as u32, unit_key as u32, idx as u32, block_idx as u32];
    // Mirrors the AIR's bitmask fold (one slot, sum 2^i * mask_i).
    key.push(mask.iter().enumerate().map(|(i, &b)| (b as u32) << i).sum());
    key.extend(chunk.iter().map(|value| value.as_canonical_u32()));
    key
}

fn final_root_recv_state_lane(cols: &WhirRoundCols<F>, lane: usize) -> F {
    debug_assert!(lane < POSEIDON2_WIDTH);
    if lane < WHIR_FINAL_ROOT_DIGEST_LANES {
        cols.event_value[lane]
    } else if lane < 13 {
        cols.r_fold[lane - WHIR_FINAL_ROOT_DIGEST_LANES]
    } else {
        cols.claim_acc[lane - 13]
    }
}

fn final_root_send_state_lane(cols: &WhirRoundCols<F>, lane: usize) -> F {
    debug_assert!(lane < POSEIDON2_WIDTH);
    cols.event_value[16 + lane]
}

fn final_root_chain_key_recv(cols: &WhirRoundCols<F>) -> Vec<u32> {
    let mut key = vec![cols.proof_idx.as_canonical_u32(), cols.opening_idx.as_canonical_u32()];
    key.extend(
        (0..POSEIDON2_WIDTH).map(|lane| final_root_recv_state_lane(cols, lane).as_canonical_u32()),
    );
    key
}

fn final_root_chain_key_send(cols: &WhirRoundCols<F>) -> Vec<u32> {
    let mut key =
        vec![cols.proof_idx.as_canonical_u32(), cols.height_group_rank.as_canonical_u32()];
    key.extend(
        (0..POSEIDON2_WIDTH).map(|lane| final_root_send_state_lane(cols, lane).as_canonical_u32()),
    );
    key
}

fn query_chain_key_recv(cols: &WhirQueryFoldCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.query_idx.as_canonical_u32(),
        cols.cursor.as_canonical_u32(),
        cols.query_bits.as_canonical_u32(),
        cols.r_rounds.as_canonical_u32(),
        cols.idx.as_canonical_u32(),
        cols.idx_bit.as_canonical_u32(),
        cols.x.as_canonical_u32(),
        cols.acc.as_canonical_u32(),
        cols.ipw.as_canonical_u32(),
    ];
    key.extend(cols.folded.iter().map(|value| value.as_canonical_u32()));
    key
}

fn query_chain_key_send(cols: &WhirQueryFoldCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.query_idx.as_canonical_u32(),
        cols.chain_send_cursor.as_canonical_u32(),
        cols.query_bits.as_canonical_u32(),
        cols.r_rounds.as_canonical_u32(),
        cols.chain_send_idx.as_canonical_u32(),
        cols.chain_send_idx_bit.as_canonical_u32(),
        cols.chain_send_x.as_canonical_u32(),
        cols.chain_send_acc.as_canonical_u32(),
        cols.chain_send_ipw.as_canonical_u32(),
    ];
    key.extend(cols.chain_send_folded.iter().map(|value| value.as_canonical_u32()));
    key
}

fn query_init_key_from_round(cols: &WhirRoundCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.w_qbase.as_canonical_u32(),
        cols.query_bits.as_canonical_u32(),
        cols.r_rounds.as_canonical_u32(),
    ];
    key.extend(cols.cfr.iter().map(|value| value.as_canonical_u32()));
    key
}

fn query_init_key_from_query(cols: &WhirQueryFoldCols<F>) -> Vec<u32> {
    let mut key = vec![
        cols.proof_idx.as_canonical_u32(),
        cols.w_qbase.as_canonical_u32(),
        cols.query_bits.as_canonical_u32(),
        cols.r_rounds.as_canonical_u32(),
    ];
    key.extend(cols.cfr.iter().map(|value| value.as_canonical_u32()));
    key
}

fn leaf_pow_seed_key(
    proof_idx: usize,
    codeword_log_height: usize,
    alpha: [F; D_EF],
    pow: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, codeword_log_height as u32];
    key.extend(alpha.iter().map(|value| value.as_canonical_u32()));
    key.extend(pow.iter().map(|value| value.as_canonical_u32()));
    key
}

fn group_claim_key(proof_idx: usize, log_height: usize, claim: [F; D_EF]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, log_height as u32];
    key.extend(claim.iter().map(|value| value.as_canonical_u32()));
    key
}

fn round_bcast_key_from_round(row: RecursionWhirRoundRow) -> Vec<u32> {
    let mut key = vec![row.proof_idx as u32, row.round as u32];
    key.extend(row.r_fold.iter().map(|value| value.as_canonical_u32()));
    key.push(row.chain_recv_pending_is_merge as u32);
    key.extend(row.chain_recv_pending_beta.iter().map(|value| value.as_canonical_u32()));
    key.extend(row.chain_recv_pending_eq.iter().map(|value| value.as_canonical_u32()));
    key.push(row.emit_prep_seed as u32);
    key.push(row.merge_log_height as u32);
    key
}

fn round_bcast_key_from_query(row: RecursionWhirQueryFoldRow) -> Vec<u32> {
    let mut key = vec![row.proof_idx as u32, row.cursor as u32];
    key.extend(row.r_fold.iter().map(|value| value.as_canonical_u32()));
    key.push(row.is_merge as u32);
    key.extend(row.merge_beta.iter().map(|value| value.as_canonical_u32()));
    key.extend(row.merge_eq.iter().map(|value| value.as_canonical_u32()));
    key.push(row.emit_prep_seed as u32);
    key.push((row.query_bits - row.cursor) as u32);
    key
}

fn query_leaf_sum_key(proof_idx: usize, idx: usize, log_height: usize, sum: [F; D_EF]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, idx as u32, log_height as u32];
    key.extend(sum.iter().map(|value| value.as_canonical_u32()));
    key
}

fn apply_residual(residual: &mut BTreeMap<Vec<u32>, i64>, key: Vec<u32>, delta: i64) {
    *residual.entry(key).or_default() += delta;
}

fn whir_round_event_tidx_for_test(cols: &WhirRoundCols<F>, idx: usize) -> usize {
    let mut offset = idx;
    if idx >= 8 {
        offset = offset - field_i64(cols.is_round) as usize * 8 +
            field_i64(cols.round_has_oracle) as usize * 8;
    }
    cols.tidx.as_canonical_u32() as usize + offset
}

fn whir_round_event_is_sample_for_test(cols: &WhirRoundCols<F>, idx: usize) -> bool {
    (idx == 2 && cols.is_pow_batch == F::one()) ||
        ((23..WHIR_ROUND_MAX_TRANSCRIPT_EVENTS).contains(&idx) && cols.is_round == F::one()) ||
        (idx == 10 && cols.is_final == F::one())
}

fn whir_round_event_mult_for_test(cols: &WhirRoundCols<F>, idx: usize) -> i64 {
    let pow_mult = if idx < 3 { field_i64(cols.is_pow_batch) } else { 0 };
    let preamble_mult = if idx < 8 { field_i64(cols.is_preamble) } else { 0 };
    let round_mult = if idx < 8 {
        field_i64(cols.round_has_oracle)
    } else if idx < 28 {
        field_i64(cols.is_round)
    } else {
        field_i64(cols.is_merge)
    };
    let final_mult = if idx < 11 { field_i64(cols.is_final) } else { 0 };
    pow_mult + preamble_mult + round_mult + final_mult
}

fn query_pair_leaf_mask_for_test(block: usize) -> [bool; DIGEST_SIZE] {
    core::array::from_fn(|idx| block == 0 || idx < 2)
}

fn query_pair_leaf_chunk_for_test(
    row: RecursionWhirQueryFoldRow,
    block: usize,
) -> [F; DIGEST_SIZE] {
    core::array::from_fn(|idx| match (block, idx) {
        (0, 0..=4) => row.f0[idx],
        (0, 5..=7) => row.f1[idx - 5],
        (1, 0..=1) => row.f1[idx + 3],
        _ => F::zero(),
    })
}

fn proof_shape_root_mult(role_id: usize, publish_external: bool) -> i64 {
    if publish_external {
        whir_role_config(role_id).num_queries as i64
    } else {
        0
    }
}

fn digest_from_poseidon_output(output: [F; POSEIDON2_WIDTH]) -> [F; DIGEST_SIZE] {
    core::array::from_fn(|idx| output[idx])
}

#[cfg(test)]
fn transcript_event(
    tidx: usize,
    kind: RecursionTranscriptEventKind,
    value: F,
) -> RecursionTranscriptEvent {
    RecursionTranscriptEvent { tidx, kind, value }
}

fn field_i64(value: F) -> i64 {
    value.as_canonical_u32() as i64
}

fn flag_i64(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

#[cfg(test)]
fn ext(seed: usize) -> [F; D_EF] {
    core::array::from_fn(|idx| F::from_canonical_usize(seed + idx))
}

#[cfg(test)]
fn one_ext() -> [F; D_EF] {
    core::array::from_fn(|idx| if idx == 0 { F::one() } else { F::zero() })
}
