use core::borrow::BorrowMut;

use dt_stark::{
    air::{ACTIVE_SHAPE_TAG_V1, ACTIVE_SHAPE_VERSION_V2},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    batch_constraint_dt::columns::{
        batch_seed_prefix_limbs_for_role_id, BATCH_VK_TAG_VERSION_LIMBS,
    },
    config::{D_EF, F},
    proof_shape_dt::{
        bus::{
            PROOF_SHAPE_COMMIT_MAIN, PROOF_SHAPE_COMMIT_PERMUTATION, PROOF_SHAPE_COMMIT_VK,
            PROOF_SHAPE_VK_META_COMMIT_BASE, PROOF_SHAPE_VK_META_COMMIT_ELTS,
            PROOF_SHAPE_VK_META_PC_START,
        },
        columns::{
            NativeChipMetadataCols, NativeChipMetadataPreprocessedCols, ProofHeightSetCols,
            ProofShapeBinderCols, NUM_NATIVE_CHIP_METADATA_COLS,
            NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS, NUM_PROOF_HEIGHT_SET_COLS,
            NUM_PROOF_SHAPE_BINDER_COLS,
        },
        record::proof_shape_range_value,
    },
    system_dt::{
        RecursionNativeChipMetadataRequest, RecursionProofRecord, RecursionProofShapeChip,
        RecursionProofShapeRecord, RecursionRecord,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeChipMetadataTraceGenerator;

impl NativeChipMetadataTraceGenerator {
    pub fn trace_height(metadata: &[RecursionNativeChipMetadataRequest]) -> usize {
        metadata.len().max(1).next_power_of_two()
    }

    pub fn generate_preprocessed_trace(
        metadata: &[RecursionNativeChipMetadataRequest],
    ) -> CompressedMatrix<F> {
        let rows = metadata.iter().map(native_chip_metadata_preprocessed_row).collect::<Vec<_>>();
        compressed_rows(
            rows,
            NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS,
            Self::trace_height(metadata),
        )
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        metadata: &[RecursionNativeChipMetadataRequest],
    ) -> CompressedMatrix<F> {
        let rows = native_chip_metadata_trace_rows(record, metadata)
            .into_iter()
            .map(|request| native_chip_metadata_main_row(record, request))
            .collect::<Vec<_>>();
        compressed_rows(rows, NUM_NATIVE_CHIP_METADATA_COLS, Self::trace_height(metadata))
    }
}

pub fn native_chip_metadata_trace_rows(
    _record: &RecursionRecord,
    metadata: &[RecursionNativeChipMetadataRequest],
) -> Vec<RecursionNativeChipMetadataRequest> {
    metadata.to_vec()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProofShapeBinderTraceGenerator;

impl ProofShapeBinderTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        proof_shape_binder_rows(record).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let rows = proof_shape_binder_rows(record).into_iter().map(binder_row).collect::<Vec<_>>();
        compressed_rows(rows, NUM_PROOF_SHAPE_BINDER_COLS, Self::trace_height(record))
    }
}

#[derive(Debug, Clone)]
pub enum ProofShapeBinderRow {
    VkCommit {
        proof_idx: usize,
        role_id: usize,
        values: [F; 8],
        shape_send_mults: [u32; 8],
        publish_external: bool,
    },
    VkMeta {
        proof_idx: usize,
        base: usize,
        values: [F; 8],
        shape_mask: [bool; 8],
        shape_send_mults: [u32; 8],
        publish_external: bool,
    },
    PublicValues {
        proof_idx: usize,
        base: usize,
        shape_idx_base: usize,
        values: [F; 8],
        mask: [bool; 8],
        send_mults: [u32; 8],
        global_packed_send: bool,
        publish_external: bool,
    },
    E1 {
        proof_idx: usize,
        role_id: usize,
        tidx_base: usize,
        values: [F; 8],
        c_chips: usize,
        publish_external: bool,
    },
    ActiveShapeHeader {
        proof_idx: usize,
        role_id: usize,
        tidx_base: usize,
        c_chips: usize,
        prev: ShapeChainState,
    },
    Chip {
        proof_idx: usize,
        role_id: usize,
        chip: RecursionProofShapeChip,
        prev_chip_idx: usize,
        prev_log_height: usize,
        prev_static_chip_id: usize,
        prev_tidx_acc: usize,
        prev_prep_matrix_idx: usize,
        prev_first_log_height: usize,
        shape_chip_count: usize,
        range_val: usize,
        publish_external: bool,
        publish_batch_dim: bool,
    },
    E5 {
        proof_idx: usize,
        role_id: usize,
        tidx_base: usize,
        values: [F; 8],
        prev: ShapeChainState,
        publish_external: bool,
        summary_send_mult: u32,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeChainState {
    pub chip_idx: usize,
    pub log_height: usize,
    pub static_chip_id: usize,
    pub tidx_acc: usize,
    pub prep_matrix_idx: usize,
    pub first_log_height: usize,
    pub shape_chip_count: usize,
}

pub fn proof_shape_binder_rows(record: &RecursionRecord) -> Vec<ProofShapeBinderRow> {
    let mut rows = Vec::new();
    for proof in record.proof_records.iter().filter(|proof| !proof.proof_shape.is_empty()) {
        push_binder_rows_for_proof(proof, &mut rows);
    }
    rows
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProofHeightSetTraceGenerator;

impl ProofHeightSetTraceGenerator {
    pub fn trace_height(record: &RecursionRecord) -> usize {
        proof_height_set_rows(record).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(record: &RecursionRecord) -> CompressedMatrix<F> {
        let rows = proof_height_set_rows(record).into_iter().map(height_row).collect::<Vec<_>>();
        compressed_rows(rows, NUM_PROOF_HEIGHT_SET_COLS, Self::trace_height(record))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProofHeightSetRow {
    pub proof_idx: usize,
    pub is_first: bool,
    pub is_last: bool,
    pub height_cursor: usize,
    pub member_count: usize,
    pub rank: usize,
    pub publish_external: bool,
}

pub fn proof_height_set_rows(record: &RecursionRecord) -> Vec<ProofHeightSetRow> {
    let mut rows = Vec::new();
    for proof in record.proof_records.iter().filter(|proof| !proof.proof_shape.is_empty()) {
        let counts = height_member_counts(&proof.proof_shape);
        let mut rank = 0usize;
        for (row_idx, height_cursor) in (0..=24).rev().enumerate() {
            let member_count = counts[height_cursor];
            rows.push(ProofHeightSetRow {
                proof_idx: proof.proof_idx,
                is_first: row_idx == 0,
                is_last: height_cursor == 0,
                height_cursor,
                member_count,
                rank,
                publish_external: proof.proof_shape.publish_external
                    || proof.proof_shape.publish_whir_inputs,
            });
            if member_count != 0 {
                rank += 1;
            }
        }
    }
    rows
}

fn push_binder_rows_for_proof(proof: &RecursionProofRecord, rows: &mut Vec<ProofShapeBinderRow>) {
    let shape = &proof.proof_shape;
    let seed_prefix_limbs = batch_seed_prefix_limbs_for_role_id(shape.role_id);
    let publish_whir_inputs = shape.publish_external || shape.publish_whir_inputs;
    assert!(matches!(
        shape.vk_meta.len(),
        crate::proof_shape_dt::bus::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
            | crate::proof_shape_dt::bus::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
    ));
    assert_eq!(shape.vk_meta_send_mults.len(), shape.vk_meta.len());
    rows.push(ProofShapeBinderRow::VkCommit {
        proof_idx: proof.proof_idx,
        role_id: shape.role_id,
        values: shape.vk_commit,
        shape_send_mults: shape.vk_meta_send_mults[PROOF_SHAPE_VK_META_COMMIT_BASE
            ..PROOF_SHAPE_VK_META_COMMIT_BASE + PROOF_SHAPE_VK_META_COMMIT_ELTS]
            .try_into()
            .expect("VK commit metadata chunk has width 8"),
        publish_external: publish_whir_inputs,
    });

    for base in (PROOF_SHAPE_VK_META_PC_START..shape.vk_meta.len()).step_by(8) {
        let end = (base + 8).min(shape.vk_meta.len());
        let live = end - base;
        let mut values = [F::zero(); 8];
        let mut shape_mask = [false; 8];
        let mut shape_send_mults = [0u32; 8];
        values[..live].copy_from_slice(&shape.vk_meta[base..end]);
        shape_mask[..live].fill(true);
        shape_send_mults[..live].copy_from_slice(&shape.vk_meta_send_mults[base..end]);
        rows.push(ProofShapeBinderRow::VkMeta {
            proof_idx: proof.proof_idx,
            base,
            values,
            shape_mask,
            shape_send_mults,
            publish_external: shape.publish_external,
        });
    }

    let observed_public_values = shape.public_values.len();
    for (chunk_idx, chunk) in shape.public_values[..observed_public_values].chunks(8).enumerate() {
        let mut values = [F::zero(); 8];
        let mut mask = [false; 8];
        let mut send_mults = [0u32; 8];
        for (i, value) in chunk.iter().copied().enumerate() {
            values[i] = value;
            mask[i] = true;
            let index = 8 * chunk_idx + i;
            send_mults[i] = if shape.publish_external {
                shape.public_value_send_mults.get(index).copied().unwrap_or_else(|| {
                    panic!(
                        "missing public value send mult for proof {} public index {}",
                        proof.proof_idx, index
                    )
                })
            } else {
                0
            };
        }
        rows.push(ProofShapeBinderRow::PublicValues {
            proof_idx: proof.proof_idx,
            base: seed_prefix_limbs + 8 * chunk_idx,
            shape_idx_base: 8 * chunk_idx,
            values,
            mask,
            send_mults,
            global_packed_send: shape.role_id == crate::whir_dt::WHIR_ROLE_CORE &&
                8 * chunk_idx >= 48,
            publish_external: shape.publish_external,
        });
    }

    rows.push(ProofShapeBinderRow::E1 {
        proof_idx: proof.proof_idx,
        role_id: shape.role_id,
        tidx_base: shape.e1_tidx_base(),
        values: shape.main_commit,
        c_chips: shape.chips.len(),
        publish_external: publish_whir_inputs,
    });

    let mut prev = ShapeChainState {
        chip_idx: 0,
        log_height: 25,
        static_chip_id: 0,
        tidx_acc: shape.e1_tidx_base() + 8,
        prep_matrix_idx: 0,
        first_log_height: 0,
        shape_chip_count: shape.chips.len(),
    };
    rows.push(ProofShapeBinderRow::ActiveShapeHeader {
        proof_idx: proof.proof_idx,
        role_id: shape.role_id,
        tidx_base: prev.tidx_acc,
        c_chips: shape.chips.len(),
        prev,
    });
    prev.tidx_acc += crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_HEADER_LIMBS;
    for chip in &shape.chips {
        let range_val = proof_shape_range_value(prev.log_height, prev.static_chip_id, chip);
        rows.push(ProofShapeBinderRow::Chip {
            proof_idx: proof.proof_idx,
            role_id: shape.role_id,
            chip: *chip,
            prev_chip_idx: prev.chip_idx,
            prev_log_height: prev.log_height,
            prev_static_chip_id: prev.static_chip_id,
            prev_tidx_acc: prev.tidx_acc,
            prev_prep_matrix_idx: prev.prep_matrix_idx,
            prev_first_log_height: prev.first_log_height,
            shape_chip_count: prev.shape_chip_count,
            range_val,
            publish_external: shape.publish_external,
            publish_batch_dim: publish_whir_inputs,
        });
        let first_log_height =
            if prev.chip_idx == 0 { chip.log_height } else { prev.first_log_height };
        prev = ShapeChainState {
            chip_idx: chip.chip_idx + 1,
            log_height: chip.log_height,
            static_chip_id: chip.static_chip_id,
            tidx_acc: prev.tidx_acc
                + crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_ENTRY_LIMBS,
            prep_matrix_idx: prev.prep_matrix_idx + usize::from(chip.has_prep()),
            first_log_height,
            shape_chip_count: prev.shape_chip_count,
        };
    }

    rows.push(ProofShapeBinderRow::E5 {
        proof_idx: proof.proof_idx,
        role_id: shape.role_id,
        tidx_base: shape.e5_tidx_base(),
        values: shape.permutation_commit,
        prev,
        publish_external: publish_whir_inputs,
        summary_send_mult: if publish_whir_inputs {
            3 + u32::from(shape.publish_terminal_summary)
        } else {
            0
        },
    });
}

fn native_chip_metadata_preprocessed_row(row: &RecursionNativeChipMetadataRequest) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_NATIVE_CHIP_METADATA_PREPROCESSED_COLS];
    let cols: &mut NativeChipMetadataPreprocessedCols<F> = values.as_mut_slice().borrow_mut();
    cols.role_id = f(row.role_id);
    cols.chip_id = f(row.chip_id);
    cols.stable_air_id_lo = f_u32(row.stable_air_id & 0xffff);
    cols.stable_air_id_hi = f_u32(row.stable_air_id >> 16);
    cols.prep_width = f(row.prep_width);
    cols.main_width = f(row.main_width);
    cols.perm_width = f(row.perm_width);
    cols.constraint_count = f(row.constraint_count);
    cols.gate_count = f(row.gate_count);
    values
}

fn native_chip_metadata_main_row(
    record: &RecursionRecord,
    row: RecursionNativeChipMetadataRequest,
) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_NATIVE_CHIP_METADATA_COLS];
    let cols: &mut NativeChipMetadataCols<F> = values.as_mut_slice().borrow_mut();
    cols.mult = F::from_canonical_u32(record.native_chip_metadata.count_for(row));
    values
}

fn binder_row(row: ProofShapeBinderRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_PROOF_SHAPE_BINDER_COLS];
    let cols: &mut ProofShapeBinderCols<F> = values.as_mut_slice().borrow_mut();
    match row {
        ProofShapeBinderRow::VkCommit {
            proof_idx,
            role_id,
            values,
            shape_send_mults,
            publish_external,
        } => {
            common_event_row(cols, proof_idx, BATCH_VK_TAG_VERSION_LIMBS, values, [true; 8]);
            cols.is_vk_commit = F::one();
            fill_whir_role_config(cols, role_id, publish_external);
            cols.commit_id = f(PROOF_SHAPE_COMMIT_VK);
            cols.shape_idx_base = f(PROOF_SHAPE_VK_META_COMMIT_BASE);
            cols.shape_value_send_mask = [true; 8].map(f_bool);
            cols.shape_value_send_mults = shape_send_mults.map(f_u32);
        }
        ProofShapeBinderRow::VkMeta {
            proof_idx,
            base,
            values,
            shape_mask,
            shape_send_mults,
            publish_external: _,
        } => {
            common_event_row(
                cols,
                proof_idx,
                base + BATCH_VK_TAG_VERSION_LIMBS,
                values,
                shape_mask,
            );
            cols.is_vk_meta = F::one();
            cols.shape_idx_base = f(base);
            cols.shape_value_send_mask = shape_mask.map(f_bool);
            cols.shape_value_send_mults = shape_send_mults.map(f_u32);
        }
        ProofShapeBinderRow::PublicValues {
            proof_idx,
            base,
            shape_idx_base,
            values,
            mask,
            send_mults,
            global_packed_send,
            publish_external: _,
        } => {
            common_event_row(cols, proof_idx, base, values, mask);
            cols.is_public_values = F::one();
            cols.shape_idx_base = f(shape_idx_base);
            cols.shape_value_send_mask = mask.map(f_bool);
            cols.shape_value_send_mults = send_mults.map(f_u32);
            // `commit_id` is unused on public-value rows and acts as the one-bit
            // packed Global-row send multiplicity there.
            cols.commit_id = f_bool(global_packed_send);
        }
        ProofShapeBinderRow::E1 {
            proof_idx,
            role_id,
            tidx_base,
            values,
            c_chips,
            publish_external,
        } => {
            common_event_row(cols, proof_idx, tidx_base, values, [true; 8]);
            cols.is_e1 = F::one();
            fill_whir_role_config(cols, role_id, publish_external);
            cols.commit_id = f(PROOF_SHAPE_COMMIT_MAIN);
            cols.chain_send_log_height = f(25);
            cols.chain_send_tidx_acc = f(tidx_base + 8);
            cols.chain_send_shape_chip_count = f(c_chips);
        }
        ProofShapeBinderRow::ActiveShapeHeader { proof_idx, role_id, tidx_base, c_chips, prev } => {
            let mut event_values = [F::zero(); 8];
            event_values[0] = f_u32(ACTIVE_SHAPE_TAG_V1);
            event_values[1] = f_u32(ACTIVE_SHAPE_VERSION_V2);
            event_values[2] = f(c_chips);
            common_event_row(
                cols,
                proof_idx,
                tidx_base,
                event_values,
                [true, true, true, false, false, false, false, false],
            );
            cols.is_active_shape_header = F::one();
            cols.role_id = f(role_id);
            cols.prev_chip_idx = f(prev.chip_idx);
            cols.prev_log_height = f(prev.log_height);
            cols.prev_static_chip_id = f(prev.static_chip_id);
            cols.prev_tidx_acc = f(prev.tidx_acc);
            cols.prev_prep_matrix_idx = f(prev.prep_matrix_idx);
            cols.prev_first_log_height = f(prev.first_log_height);
            cols.prev_shape_chip_count = f(prev.shape_chip_count);
            cols.chain_send_chip_idx = f(prev.chip_idx);
            cols.chain_send_log_height = f(prev.log_height);
            cols.chain_send_static_chip_id = f(prev.static_chip_id);
            cols.chain_send_tidx_acc = f(prev.tidx_acc
                + crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_HEADER_LIMBS);
            cols.chain_send_prep_matrix_idx = f(prev.prep_matrix_idx);
            cols.chain_send_first_log_height = f(prev.first_log_height);
            cols.chain_send_shape_chip_count = f(prev.shape_chip_count);
        }
        ProofShapeBinderRow::Chip {
            proof_idx,
            role_id,
            chip,
            prev_chip_idx,
            prev_log_height,
            prev_static_chip_id,
            prev_tidx_acc,
            prev_prep_matrix_idx,
            prev_first_log_height,
            shape_chip_count,
            range_val,
            publish_external: _,
            publish_batch_dim,
        } => {
            let first = prev_chip_idx == 0;
            cols.proof_idx = f(proof_idx);
            cols.is_valid = F::one();
            cols.is_chip = F::one();
            cols.role_id = f(role_id);
            cols.chip_idx = f(chip.chip_idx);
            cols.static_chip_id = f(chip.static_chip_id);
            cols.stable_air_id_lo = f_u32(chip.stable_air_id & 0xffff);
            cols.stable_air_id_hi = f_u32(chip.stable_air_id >> 16);
            cols.seg_bit = f(chip.static_chip_id / 128);
            cols.log_height = f(chip.log_height);
            cols.prep_width = f(chip.prep_width);
            cols.main_width = f(chip.main_width);
            cols.perm_width = f(chip.perm_width);
            cols.constraint_count = f(chip.constraint_count);
            cols.gate_count = f(chip.gate_count);
            cols.has_prep = f_bool(chip.has_prep());
            cols.prep_width_inv =
                if chip.prep_width == 0 { F::zero() } else { f(chip.prep_width).inverse() };
            cols.prev_chip_idx = f(prev_chip_idx);
            cols.prev_log_height = f(prev_log_height);
            cols.prev_static_chip_id = f(prev_static_chip_id);
            cols.prev_seg_bit = f(prev_static_chip_id / 128);
            cols.prev_tidx_acc = f(prev_tidx_acc);
            cols.prev_prep_matrix_idx = f(prev_prep_matrix_idx);
            cols.prev_first_log_height = f(prev_first_log_height);
            cols.prev_shape_chip_count = f(shape_chip_count);
            cols.is_first_chip = f_bool(first);
            cols.prev_chip_idx_inv =
                if prev_chip_idx == 0 { F::zero() } else { f(prev_chip_idx).inverse() };
            cols.chain_send_chip_idx = f(chip.chip_idx + 1);
            cols.chain_send_log_height = f(chip.log_height);
            cols.chain_send_static_chip_id = f(chip.static_chip_id);
            cols.chain_send_tidx_acc =
                f(prev_tidx_acc
                    + crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_ENTRY_LIMBS);
            cols.chain_send_prep_matrix_idx =
                f(prev_prep_matrix_idx + usize::from(chip.has_prep()));
            cols.chain_send_first_log_height =
                f(if prev_chip_idx == 0 { chip.log_height } else { prev_first_log_height });
            cols.chain_send_shape_chip_count = f(shape_chip_count);
            cols.tidx_base = f(prev_tidx_acc);
            cols.event_values[0] = f_u32(chip.stable_air_id & 0xffff);
            cols.event_values[1] = f_u32(chip.stable_air_id >> 16);
            cols.event_values[2] = f(chip.log_height);
            cols.event_values[3] = f(chip.main_width);
            cols.event_values[4] = f(chip.chip_idx);
            cols.event_recv_mask[..5].fill(F::one());
            cols.is_group_start = f_bool(chip.log_height != prev_log_height);
            cols.range_val = f(range_val);
            cols.chip_meta_send_mult = f(chip.gate_count + chip.perm_width / D_EF + 1);
            cols.batch_dim_prep_send_mult = f_bool(publish_batch_dim && chip.has_prep());
            cols.batch_dim_perm_send_mult = f_bool(publish_batch_dim);
        }
        ProofShapeBinderRow::E5 {
            proof_idx,
            role_id,
            tidx_base,
            values,
            prev,
            publish_external,
            summary_send_mult,
        } => {
            common_event_row(cols, proof_idx, tidx_base, values, [true; 8]);
            cols.is_e5 = F::one();
            fill_whir_role_config(cols, role_id, publish_external);
            cols.commit_id = f(PROOF_SHAPE_COMMIT_PERMUTATION);
            cols.prev_chip_idx = f(prev.chip_idx);
            cols.prev_log_height = f(prev.log_height);
            cols.prev_static_chip_id = f(prev.static_chip_id);
            cols.prev_seg_bit = f(prev.static_chip_id / 128);
            cols.prev_tidx_acc = f(prev.tidx_acc);
            cols.prev_prep_matrix_idx = f(prev.prep_matrix_idx);
            cols.prev_first_log_height = f(prev.first_log_height);
            cols.prev_shape_chip_count = f(prev.shape_chip_count);
            cols.summary_send_mult = F::from_canonical_u32(summary_send_mult);
            cols.fold_plan_source_mult = f(prev.chip_idx + 2);
        }
    }
    values
}

fn fill_whir_role_config(
    cols: &mut ProofShapeBinderCols<F>,
    role_id: usize,
    publish_external: bool,
) {
    cols.role_id = f(role_id);
    cols.whir_role_config_recv_mult = f_bool(publish_external);
}

fn common_event_row(
    cols: &mut ProofShapeBinderCols<F>,
    proof_idx: usize,
    tidx_base: usize,
    values: [F; 8],
    mask: [bool; 8],
) {
    cols.proof_idx = f(proof_idx);
    cols.is_valid = F::one();
    cols.tidx_base = f(tidx_base);
    cols.event_values = values;
    cols.event_recv_mask = mask.map(f_bool);
}

fn height_row(row: ProofHeightSetRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_PROOF_HEIGHT_SET_COLS];
    let cols: &mut ProofHeightSetCols<F> = values.as_mut_slice().borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_first = f_bool(row.is_first);
    cols.is_last = f_bool(row.is_last);
    cols.height_cursor = f(row.height_cursor);
    cols.member_count = f(row.member_count);
    cols.member_count_inv =
        if row.member_count == 0 { F::zero() } else { f(row.member_count).inverse() };
    cols.present = f_bool(row.member_count != 0);
    cols.rank = f(row.rank);
    cols.height_group_send_mult = f_bool(row.publish_external && row.member_count != 0);
    values
}

fn height_member_counts(shape: &RecursionProofShapeRecord) -> [usize; 25] {
    let mut counts = [0usize; 25];
    for chip in &shape.chips {
        counts[chip.log_height] += 2 + usize::from(chip.has_prep());
    }
    counts
}

fn compressed_rows(rows: Vec<Vec<F>>, width: usize, height: usize) -> CompressedMatrix<F> {
    if rows.is_empty() {
        return CompressedMatrix::new(
            RowMajorMatrix::new(vec![F::zero(); width], width),
            PaddingRow::None,
            1,
        );
    }
    let flat = rows.into_iter().flatten().collect::<Vec<_>>();
    let main = RowMajorMatrix::new(flat, width);
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

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;
    use std::collections::BTreeMap;

    use super::*;
    use crate::batch_constraint_dt::columns::BATCH_CORE_SEED_PREFIX_LIMBS;
    use p3_field::{AbstractField, PrimeField32};

    use crate::proof_shape_dt::air::{binder_lookup_ops, LookupDirection};

    fn sample_record(publish_external: bool) -> RecursionRecord {
        RecursionRecord {
            proof_records: vec![RecursionProofRecord {
                proof_idx: 7,
                proof_shape: RecursionProofShapeRecord {
                    role_id: 0,
                    num_public_values: 9,
                    vk_commit: eight_values(10),
                    vk_meta: core_vk_meta_values(20).to_vec(),
                    vk_meta_send_mults:
                        vec![0; crate::proof_shape_dt::bus::PROOF_SHAPE_VK_META_VALUE_COUNT],
                    public_values: (0..9).map(|value| f(100 + value)).collect(),
                    public_value_send_mults: vec![u32::from(publish_external); 9],
                    main_commit: eight_values(200),
                    permutation_commit: eight_values(300),
                    chips: vec![
                        RecursionProofShapeChip {
                            chip_idx: 0,
                            static_chip_id: 1,
                            stable_air_id: 0x1111_2222,
                            log_height: 4,
                            prep_width: 2,
                            main_width: 12,
                            perm_width: 5,
                            constraint_count: 8,
                            gate_count: 8,
                        },
                        RecursionProofShapeChip {
                            chip_idx: 1,
                            static_chip_id: 3,
                            stable_air_id: 0x3333_4444,
                            log_height: 4,
                            prep_width: 0,
                            main_width: 9,
                            perm_width: 5,
                            constraint_count: 6,
                            gate_count: 6,
                        },
                        RecursionProofShapeChip {
                            chip_idx: 2,
                            static_chip_id: 0,
                            stable_air_id: 43,
                            log_height: 2,
                            prep_width: 1,
                            main_width: 4,
                            perm_width: 5,
                            constraint_count: 3,
                            gate_count: 3,
                        },
                    ],
                    publish_external,
                    publish_terminal_summary: false,
                    publish_whir_inputs: false,
                },
                ..RecursionProofRecord::default()
            }],
            ..RecursionRecord::default()
        }
    }

    #[test]
    fn binder_rows_follow_transcript_layout_and_chain_state() {
        let record = sample_record(false);
        let rows = proof_shape_binder_rows(&record);
        assert_eq!(rows.len(), 12);

        match &rows[0] {
            ProofShapeBinderRow::VkCommit { proof_idx, values, publish_external, .. } => {
                assert_eq!(*proof_idx, 7);
                assert_eq!(*values, eight_values(10));
                assert!(!publish_external);
            }
            row => panic!("unexpected row 0: {row:?}"),
        }

        match &rows[4] {
            ProofShapeBinderRow::PublicValues { proof_idx, base, mask, .. } => {
                assert_eq!(*proof_idx, 7);
                assert_eq!(*base, BATCH_CORE_SEED_PREFIX_LIMBS);
                assert_eq!(*mask, [true; 8]);
            }
            row => panic!("unexpected first public-values row: {row:?}"),
        }
        match &rows[5] {
            ProofShapeBinderRow::PublicValues { base, mask, .. } => {
                assert_eq!(*base, BATCH_CORE_SEED_PREFIX_LIMBS + 8);
                assert_eq!(*mask, [true, false, false, false, false, false, false, false]);
            }
            row => panic!("unexpected second public-values row: {row:?}"),
        }
        match &rows[6] {
            ProofShapeBinderRow::E1 { tidx_base, values, .. } => {
                assert_eq!(*tidx_base, BATCH_CORE_SEED_PREFIX_LIMBS + 9);
                assert_eq!(*values, eight_values(200));
            }
            row => panic!("unexpected E1 row: {row:?}"),
        }

        match &rows[7] {
            ProofShapeBinderRow::ActiveShapeHeader { tidx_base, c_chips, prev, .. } => {
                assert_eq!(*tidx_base, BATCH_CORE_SEED_PREFIX_LIMBS + 9 + 8);
                assert_eq!(*c_chips, 3);
                assert_eq!(prev.tidx_acc, *tidx_base);
                assert_eq!(prev.shape_chip_count, 3);
            }
            row => panic!("unexpected active-shape header row: {row:?}"),
        }

        match &rows[8] {
            ProofShapeBinderRow::Chip {
                prev_chip_idx,
                prev_log_height,
                prev_static_chip_id,
                prev_tidx_acc,
                prev_prep_matrix_idx,
                prev_first_log_height,
                range_val,
                ..
            } => {
                assert_eq!(*prev_chip_idx, 0);
                assert_eq!(*prev_log_height, 25);
                assert_eq!(*prev_static_chip_id, 0);
                assert_eq!(*prev_tidx_acc, BATCH_CORE_SEED_PREFIX_LIMBS + 9 + 8 + 3);
                assert_eq!(*prev_prep_matrix_idx, 0);
                assert_eq!(*prev_first_log_height, 0);
                assert_eq!(*range_val, 20);
            }
            row => panic!("unexpected first chip row: {row:?}"),
        }
        match &rows[9] {
            ProofShapeBinderRow::Chip {
                prev_chip_idx,
                prev_log_height,
                prev_static_chip_id,
                prev_tidx_acc,
                prev_prep_matrix_idx,
                prev_first_log_height,
                range_val,
                ..
            } => {
                assert_eq!(*prev_chip_idx, 1);
                assert_eq!(*prev_log_height, 4);
                assert_eq!(*prev_static_chip_id, 1);
                assert_eq!(*prev_tidx_acc, BATCH_CORE_SEED_PREFIX_LIMBS + 9 + 8 + 3 + 5);
                assert_eq!(*prev_prep_matrix_idx, 1);
                assert_eq!(*prev_first_log_height, 4);
                assert_eq!(*range_val, 1);
            }
            row => panic!("unexpected second chip row: {row:?}"),
        }
        match &rows[11] {
            ProofShapeBinderRow::E5 { tidx_base, values, prev, .. } => {
                assert_eq!(*tidx_base, BATCH_CORE_SEED_PREFIX_LIMBS + 9 + 8 + 3 + 3 * 5 + 10);
                assert_eq!(*values, eight_values(300));
                assert_eq!(prev.chip_idx, 3);
                assert_eq!(prev.tidx_acc, BATCH_CORE_SEED_PREFIX_LIMBS + 9 + 8 + 3 + 3 * 5);
                assert_eq!(prev.prep_matrix_idx, 2);
                assert_eq!(prev.first_log_height, 4);
            }
            row => panic!("unexpected E5 row: {row:?}"),
        }
    }

    #[test]
    fn binder_trace_sets_commit_ids_for_commitment_rows() {
        let record = sample_record(true);
        let rows = proof_shape_binder_rows(&record);

        let vk_row = binder_row(rows[0].clone());
        let vk_cols: &ProofShapeBinderCols<F> = vk_row.as_slice().borrow();
        assert_eq!(vk_cols.commit_id, f(PROOF_SHAPE_COMMIT_VK));
        assert_eq!(vk_cols.shape_idx_base, f(PROOF_SHAPE_VK_META_COMMIT_BASE));
        assert_eq!(vk_cols.shape_value_send_mask, [F::one(); 8]);

        let vk_meta_a = binder_row(rows[1].clone());
        let vk_meta_a_cols: &ProofShapeBinderCols<F> = vk_meta_a.as_slice().borrow();
        assert_eq!(vk_meta_a_cols.shape_idx_base, f(PROOF_SHAPE_VK_META_PC_START));
        assert_eq!(vk_meta_a_cols.shape_value_send_mask, [F::one(); 8]);

        let vk_meta_b = binder_row(rows[3].clone());
        let vk_meta_b_cols: &ProofShapeBinderCols<F> = vk_meta_b.as_slice().borrow();
        assert_eq!(
            vk_meta_b_cols.shape_idx_base,
            f(crate::proof_shape_dt::bus::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT - 8)
        );
        assert_eq!(
            vk_meta_b_cols.shape_value_send_mask,
            [F::one(); 8]
        );

        let e1_row = binder_row(rows[6].clone());
        let e1_cols: &ProofShapeBinderCols<F> = e1_row.as_slice().borrow();
        assert_eq!(e1_cols.commit_id, f(PROOF_SHAPE_COMMIT_MAIN));

        let header_row = binder_row(rows[7].clone());
        let header_cols: &ProofShapeBinderCols<F> = header_row.as_slice().borrow();
        assert_eq!(header_cols.is_active_shape_header, F::one());
        assert_eq!(
            header_cols.event_values[..3],
            [
                F::from_canonical_u32(ACTIVE_SHAPE_TAG_V1),
                F::from_canonical_u32(ACTIVE_SHAPE_VERSION_V2),
                f(3),
            ]
        );
        assert_eq!(
            header_cols.event_recv_mask,
            [F::one(), F::one(), F::one(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero()]
        );

        let first_chip_row = binder_row(rows[8].clone());
        let first_chip_cols: &ProofShapeBinderCols<F> = first_chip_row.as_slice().borrow();
        assert_eq!(
            first_chip_cols.event_values[..5],
            [F::from_canonical_u32(0x2222), F::from_canonical_u32(0x1111), f(4), f(12), F::zero(),]
        );
        assert_eq!(first_chip_cols.event_recv_mask[..5], [F::one(); 5]);

        let e5_row = binder_row(rows[11].clone());
        let e5_cols: &ProofShapeBinderCols<F> = e5_row.as_slice().borrow();
        assert_eq!(e5_cols.commit_id, f(PROOF_SHAPE_COMMIT_PERMUTATION));
    }

    #[test]
    fn binder_chain_bus_balances_under_lookup_order() {
        let record = sample_record(false);
        let mut balance = TraceBusBalance::default();
        let mut fold_plan_source_mult = None;

        for row in proof_shape_binder_rows(&record) {
            let values = binder_row(row);
            let cols: &ProofShapeBinderCols<F> = values.as_slice().borrow();
            let chip_meta_mult =
                cols.is_chip * (cols.gate_count + cols.perm_width * f(D_EF).inverse() + F::one());
            let ops = binder_lookup_ops(cols, f(261), chip_meta_mult, false);
            assert_eq!(ops.len(), 34);

            let (direction, mult) = ops[25];
            balance.apply(direction, chain_recv_key(cols), mult);
            let (direction, mult) = ops[26];
            balance.apply(direction, chain_send_key(cols), mult);
            if cols.is_e5 == F::one() {
                let (direction, mult) = ops[33];
                assert_eq!(direction, LookupDirection::Send);
                fold_plan_source_mult = Some(mult);
            }
        }

        balance.assert_balanced("ProofShapeChain");
        assert_eq!(
            fold_plan_source_mult,
            Some(f(5)),
            "one Fold receiver + three Challenge rows + one fused Batch input row"
        );
    }

    #[test]
    fn chip_meta_multiplicity_is_exact_and_cannot_be_zeroed() {
        let record = sample_record(true);
        for row in proof_shape_binder_rows(&record) {
            let values = binder_row(row);
            let cols: &ProofShapeBinderCols<F> = values.as_slice().borrow();
            if cols.is_chip == F::one() {
                let expected = cols.gate_count + cols.perm_width * f(D_EF).inverse() + F::one();
                assert_eq!(cols.chip_meta_send_mult, expected);
                let zeroed_constraint = F::zero() - cols.is_chip * expected;
                assert_ne!(
                    zeroed_constraint,
                    F::zero(),
                    "zeroing chip-meta multiplicity must violate its exact AIR equality"
                );
            }
        }
    }

    #[test]
    fn set_rows_count_height_memberships() {
        let record = sample_record(true);

        let height_rows = proof_height_set_rows(&record);
        assert_eq!(height_rows.len(), 25);
        assert_eq!(height_rows[0].height_cursor, 24);
        assert!(height_rows[0].is_first);
        assert_eq!(height_rows[0].rank, 0);

        let height_four =
            height_rows.iter().find(|row| row.height_cursor == 4).expect("height 4 row");
        assert_eq!(height_four.member_count, 5);
        assert_eq!(height_four.rank, 0);
        assert!(height_four.publish_external);

        let height_two =
            height_rows.iter().find(|row| row.height_cursor == 2).expect("height 2 row");
        assert_eq!(height_two.member_count, 3);
        assert_eq!(height_two.rank, 1);

        let height_zero = height_rows.last().expect("height 0 row");
        assert_eq!(height_zero.height_cursor, 0);
        assert!(height_zero.is_last);
        assert_eq!(height_zero.member_count, 0);
        assert_eq!(height_zero.rank, 2);
    }

    #[test]
    fn height_group_send_mult_only_marks_published_present_heights() {
        let published = sample_record(true);
        for row in proof_height_set_rows(&published) {
            let values = height_row(row);
            let cols: &ProofHeightSetCols<F> = values.as_slice().borrow();
            let expected = cols.present;
            assert_eq!(cols.height_group_send_mult, expected);
        }

        let unpublished = sample_record(false);
        for row in proof_height_set_rows(&unpublished) {
            let values = height_row(row);
            let cols: &ProofHeightSetCols<F> = values.as_slice().borrow();
            assert_eq!(cols.height_group_send_mult, F::zero());
        }
    }

    #[test]
    fn whir_inputs_publish_batch_dims_commit_roots_and_height_groups_only() {
        let mut record = sample_record(false);
        record.proof_records[0].proof_shape.publish_whir_inputs = true;

        let binder_rows = proof_shape_binder_rows(&record);
        let vk_row = binder_row(binder_rows[0].clone());
        let vk_cols: &ProofShapeBinderCols<F> = vk_row.as_slice().borrow();
        assert_eq!(vk_cols.whir_role_config_recv_mult, F::one());

        let chip_row = binder_rows
            .iter()
            .find(|row| matches!(row, ProofShapeBinderRow::Chip { .. }))
            .expect("chip row")
            .clone();
        let chip_values = binder_row(chip_row);
        let chip_cols: &ProofShapeBinderCols<F> = chip_values.as_slice().borrow();
        assert_eq!(chip_cols.batch_dim_prep_send_mult, F::one());
        assert_eq!(chip_cols.batch_dim_perm_send_mult, F::one());

        let present_height = proof_height_set_rows(&record)
            .into_iter()
            .find(|row| row.member_count != 0)
            .expect("present height row");
        let height_values = height_row(present_height);
        let height_cols: &ProofHeightSetCols<F> = height_values.as_slice().borrow();
        assert_eq!(height_cols.height_group_send_mult, F::one());
    }

    #[test]
    fn summary_send_mult_distinguishes_whir_only_and_full_external_publish() {
        let mut whir_only = sample_record(false);
        whir_only.proof_records[0].proof_shape.publish_whir_inputs = true;
        let whir_rows = proof_shape_binder_rows(&whir_only);
        let whir_summary = whir_rows
            .iter()
            .find_map(|row| match row {
                ProofShapeBinderRow::E5 { summary_send_mult, .. } => Some(*summary_send_mult),
                _ => None,
            })
            .expect("E5 row exists");
        assert_eq!(whir_summary, 3);

        let full_rows = proof_shape_binder_rows(&sample_record(true));
        let full_summary = full_rows
            .iter()
            .find_map(|row| match row {
                ProofShapeBinderRow::E5 { summary_send_mult, .. } => Some(*summary_send_mult),
                _ => None,
            })
            .expect("E5 row exists");
        assert_eq!(full_summary, 3);

        let mut terminal = sample_record(true);
        terminal.proof_records[0].proof_shape.publish_terminal_summary = true;
        let terminal_rows = proof_shape_binder_rows(&terminal);
        let terminal_summary = terminal_rows
            .iter()
            .find_map(|row| match row {
                ProofShapeBinderRow::E5 { summary_send_mult, .. } => Some(*summary_send_mult),
                _ => None,
            })
            .expect("E5 row exists");
        assert_eq!(terminal_summary, 4);
    }

    #[derive(Default)]
    struct TraceBusBalance {
        net: BTreeMap<Vec<u32>, i64>,
    }

    impl TraceBusBalance {
        fn apply(&mut self, direction: LookupDirection, key: Vec<u32>, mult: F) {
            let mult = mult.as_canonical_u32() as i64;
            if mult == 0 {
                return;
            }
            let sign = match direction {
                LookupDirection::Send => 1,
                LookupDirection::Recv => -1,
            };
            *self.net.entry(key).or_default() += sign * mult;
        }

        fn assert_balanced(self, bus_name: &str) {
            let nonzero = self.net.into_iter().filter(|(_, value)| *value != 0).collect::<Vec<_>>();
            assert!(nonzero.is_empty(), "{bus_name} imbalance: {nonzero:?}");
        }
    }

    fn chain_recv_key(cols: &ProofShapeBinderCols<F>) -> Vec<u32> {
        vec![
            f_u32(cols.proof_idx),
            f_u32(cols.prev_chip_idx),
            f_u32(cols.prev_log_height),
            f_u32(cols.prev_static_chip_id),
            f_u32(cols.prev_tidx_acc),
            f_u32(cols.prev_prep_matrix_idx),
            f_u32(cols.prev_first_log_height),
            f_u32(cols.prev_shape_chip_count),
        ]
    }

    fn chain_send_key(cols: &ProofShapeBinderCols<F>) -> Vec<u32> {
        vec![
            f_u32(cols.proof_idx),
            f_u32(cols.chain_send_chip_idx),
            f_u32(cols.chain_send_log_height),
            f_u32(cols.chain_send_static_chip_id),
            f_u32(cols.chain_send_tidx_acc),
            f_u32(cols.chain_send_prep_matrix_idx),
            f_u32(cols.chain_send_first_log_height),
            f_u32(cols.chain_send_shape_chip_count),
        ]
    }

    fn f_u32(value: F) -> u32 {
        value.as_canonical_u32()
    }

    fn eight_values(base: usize) -> [F; 8] {
        core::array::from_fn(|idx| f(base + idx))
    }

    fn core_vk_meta_values(
        base: usize,
    ) -> [F; crate::proof_shape_dt::bus::PROOF_SHAPE_VK_META_VALUE_COUNT] {
        core::array::from_fn(|idx| f(base + idx))
    }
}
