use std::collections::BTreeMap;

#[cfg(test)]
use crate::config::DIGEST_SIZE;
#[cfg(test)]
use crate::transcript_dt::poseidon2::RecursionPoseidon2Memo;
use crate::{
    config::{D_EF, EF, F, POSEIDON2_WIDTH},
    system_dt::record::{
        RecursionBatchConstraintRecord, RecursionWhirBatchEvalRow, RecursionWhirLeafExtStreamRow,
        RecursionWhirLeafExtStreamTraceRow, RecursionWhirLeafStreamRow, RecursionWhirQueryFoldRow,
        RecursionWhirRoundRow,
    },
    transcript_dt::poseidon2::RecursionPoseidon2Output,
    whir_dt::{
        columns::{
            whir_unit_key, WHIR_BATCHING_POW_BITS, WHIR_BATCHING_POW_HIGH_MAX,
            WHIR_FINAL_ROOT_DIGEST_LANES, WHIR_FINAL_ROOT_POSEIDON2_PERMS,
            WHIR_INPUT_MAIN_PATH_SLOT, WHIR_INPUT_PERMUTATION_PATH_SLOT,
            WHIR_INPUT_PREPROCESSED_PATH_SLOT, WHIR_LEAF_BASE_LIMBS_PER_ROW,
            WHIR_LEAF_BLOCKS_PER_ROW, WHIR_LEAF_RLC_SLOTS, WHIR_QUERY_POW_BITS,
            WHIR_QUERY_POW_HIGH_MAX, WHIR_TWIDDLE_TABLES,
        },
        trace::{sample_band_for_query_bits, twiddle_value, WhirSampleBandConfig},
    },
};
use dt_stark::sumcheck::proof::SCShardOpenedValues;
use p3_field::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use p3_matrix::Dimensions;
use serde::{Deserialize, Serialize};

pub const WHIR_BATCH_PREPROCESSED: usize = 0;
pub const WHIR_BATCH_MAIN: usize = 1;
pub const WHIR_BATCH_PERMUTATION: usize = 2;
const WHIR_BATCH_COUNT: usize = WHIR_BATCH_PERMUTATION + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirSpecFoldShape {
    pub role_id: usize,
    pub num_rounds: usize,
    pub c_chips: usize,
    pub num_public_values: usize,
    pub num_queries: usize,
    pub batching_bits: usize,
    pub query_bits: usize,
    pub log_blowup: usize,
    pub w0_tidx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirSpecFoldSeed {
    pub proof_idx: usize,
    pub shape: WhirSpecFoldShape,
    pub opening_point: Vec<[F; D_EF]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirFinalRootSponge {
    pub inputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub outputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub num_perms: usize,
    pub digest: [F; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirFinalRootRowFields {
    pub inputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub outputs: [[F; POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub recv_mults: [u32; WHIR_FINAL_ROOT_POSEIDON2_PERMS],
    pub digest: [F; 8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirOpenedMatrix {
    pub batch_id: usize,
    pub batch_pos: usize,
    pub chip_idx: usize,
    pub width: usize,
    pub log_height: usize,
    pub values: Vec<[F; D_EF]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirOpenedMatrices {
    pub matrices: Vec<WhirOpenedMatrix>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirBatchRlcStep {
    pub cursor: usize,
    pub log_height: usize,
    pub batch_id: usize,
    pub batch_pos: usize,
    pub chip_idx: usize,
    pub width: usize,
    pub value_idx: usize,
    pub value: [F; D_EF],
    pub alpha_power: [F; D_EF],
    pub alpha_power_out: [F; D_EF],
    pub acc_in: [F; D_EF],
    pub acc_out: [F; D_EF],
    pub group_base_in: [F; D_EF],
    pub group_base_out: [F; D_EF],
    pub is_segment_start: bool,
    pub is_group_start: bool,
    pub is_group_end: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirBatchRlcSegment {
    pub log_height: usize,
    pub batch_id: usize,
    pub batch_pos: usize,
    pub chip_idx: usize,
    pub width: usize,
    pub first_cursor: usize,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirBatchRlcGroup {
    pub log_height: usize,
    pub claim: [F; D_EF],
    pub first_cursor: usize,
    pub element_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirBatchRlc {
    pub alpha: [F; D_EF],
    pub segments: Vec<WhirBatchRlcSegment>,
    pub groups: Vec<WhirBatchRlcGroup>,
    pub steps: Vec<WhirBatchRlcStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirRoundReplayInput {
    pub seed: WhirSpecFoldSeed,
    /// Replay-segment authority from the verified proof-shape view. It is
    /// installed while round rows are constructed, never patched afterward.
    pub summary_id_base: usize,
    pub group_claims: Vec<WhirBatchRlcGroup>,
    pub sumcheck_coeffs: Vec<[[F; D_EF]; 3]>,
    pub r_folds: Vec<[F; D_EF]>,
    pub merge_betas_by_height: BTreeMap<usize, [F; D_EF]>,
    pub iopp_oracles: Vec<[F; 8]>,
    pub batching_pow_events: [F; 3],
    pub query_pow_events: [F; 3],
    pub prep_seed_round: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhirQueryRoundControl {
    pub r_fold: [F; D_EF],
    pub is_merge: bool,
    pub is_assign: bool,
    pub merge_beta: [F; D_EF],
    pub merge_eq: [F; D_EF],
    pub emit_prep_seed: bool,
    pub cfr: [F; D_EF],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirQueryPairSource {
    Explicit(Vec<([F; D_EF], [F; D_EF])>),
    Siblings(Vec<[F; D_EF]>),
}

/// Compact round replay authority consumed by device trace generation.
///
/// It contains only transcript controls, exact output positions, range
/// publications, and final-root provider inputs. No semantic round-row mirror
/// is allocated on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirCompactRoundAuthority {
    pub controls: Vec<WhirQueryRoundControl>,
    pub w_qbase: usize,
    pub output_count: usize,
    pub final_row_idx: usize,
    pub batching_pow_sample_high: usize,
    pub query_pow_sample_high: usize,
    pub final_root: WhirFinalRootRowFields,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhirQueryReplayInput {
    pub seed: WhirSpecFoldSeed,
    pub query_idx: usize,
    pub w_qbase: usize,
    pub query_sample_raw: F,
    pub query_sample: usize,
    pub controls: Vec<WhirQueryRoundControl>,
    pub pair_source: WhirQueryPairSource,
    pub leaf_sums_by_log_height: BTreeMap<usize, [F; D_EF]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirSpecFoldError {
    RoundCountMismatch {
        expected: usize,
        actual: usize,
    },
    PcsBatchCountMismatch {
        expected_min: usize,
        actual: usize,
    },
    DimensionCountMismatch {
        batch_id: usize,
        expected: usize,
        actual: usize,
    },
    OpenedWidthMismatch {
        batch_id: usize,
        batch_pos: usize,
        chip_idx: usize,
        expected_width: usize,
        actual_values: usize,
    },
    NonPowerOfTwoHeight {
        batch_id: usize,
        batch_pos: usize,
        height: usize,
    },
    PrepFirstHeightNotMax {
        first_log_height: usize,
        max_log_height: usize,
    },
    RoundReplayRoundCountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    RoundReplayOracleCountMismatch {
        expected: usize,
        actual: usize,
    },
    DuplicateHeightGroup {
        log_height: usize,
    },
    MissingHeightGroup {
        log_height: usize,
    },
    MissingMergeBeta {
        log_height: usize,
    },
    UnexpectedMergeBeta {
        log_height: usize,
    },
    UnsupportedQueryBits {
        query_bits: usize,
    },
    QueryReplayRoundCountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    QueryReplayRoundIndexMismatch {
        expected: usize,
        actual: usize,
    },
    QueryOpeningBatchCountMismatch {
        expected_min: usize,
        actual: usize,
    },
    QueryOpeningMatrixCountMismatch {
        batch_id: usize,
        expected_min: usize,
        actual: usize,
    },
    QueryOpeningWidthMismatch {
        batch_id: usize,
        batch_pos: usize,
        expected_width: usize,
        actual_width: usize,
    },
    QuerySampleLowMismatch {
        expected: usize,
        actual: usize,
    },
    QuerySampleOutOfRange {
        query_sample: usize,
        query_bits: usize,
    },
    PowSampleOutOfRange {
        phase: &'static str,
        high: usize,
        max: usize,
    },
    MissingQueryLeafSum {
        log_height: usize,
    },
    ZeroQueryTwiddle {
        round: usize,
    },
}

impl WhirSpecFoldSeed {
    pub fn from_batch(
        proof_idx: usize,
        shape: WhirSpecFoldShape,
        batch: &RecursionBatchConstraintRecord,
    ) -> Result<Self, WhirSpecFoldError> {
        if batch.rounds.len() != shape.num_rounds {
            return Err(WhirSpecFoldError::RoundCountMismatch {
                expected: shape.num_rounds,
                actual: batch.rounds.len(),
            });
        }

        let opening_point = (0..shape.num_rounds)
            .map(|idx| batch.rounds[shape.num_rounds - 1 - idx].challenge)
            .collect();
        Ok(Self { proof_idx, shape, opening_point })
    }
}

impl WhirOpenedMatrices {
    pub fn from_child_openings(
        dimensions: &[Vec<Dimensions>],
        opened_values: &SCShardOpenedValues<F, EF>,
    ) -> Result<Self, WhirSpecFoldError> {
        if dimensions.len() < 2 {
            return Err(WhirSpecFoldError::PcsBatchCountMismatch {
                expected_min: 2,
                actual: dimensions.len(),
            });
        }
        let chip_count = opened_values.chips.len();
        if dimensions[WHIR_BATCH_MAIN].len() != chip_count {
            return Err(WhirSpecFoldError::DimensionCountMismatch {
                batch_id: WHIR_BATCH_MAIN,
                expected: chip_count,
                actual: dimensions[WHIR_BATCH_MAIN].len(),
            });
        }

        let preprocessed_count =
            opened_values.chips.iter().filter(|chip| !chip.preprocessed.local.is_empty()).count();
        if dimensions[WHIR_BATCH_PREPROCESSED].len() != preprocessed_count {
            return Err(WhirSpecFoldError::DimensionCountMismatch {
                batch_id: WHIR_BATCH_PREPROCESSED,
                expected: preprocessed_count,
                actual: dimensions[WHIR_BATCH_PREPROCESSED].len(),
            });
        }

        let has_permutation_batch = dimensions.len() > WHIR_BATCH_PERMUTATION;
        if has_permutation_batch && dimensions[WHIR_BATCH_PERMUTATION].len() != chip_count {
            return Err(WhirSpecFoldError::DimensionCountMismatch {
                batch_id: WHIR_BATCH_PERMUTATION,
                expected: chip_count,
                actual: dimensions[WHIR_BATCH_PERMUTATION].len(),
            });
        }

        let mut matrices = Vec::new();
        let mut prep_pos = 0;
        for (chip_idx, chip) in opened_values.chips.iter().enumerate() {
            if !chip.preprocessed.local.is_empty() {
                matrices.push(opened_matrix(
                    WHIR_BATCH_PREPROCESSED,
                    prep_pos,
                    chip_idx,
                    dimensions[WHIR_BATCH_PREPROCESSED][prep_pos],
                    &chip.preprocessed.local,
                    false,
                )?);
                prep_pos += 1;
            }
        }
        for (chip_idx, chip) in opened_values.chips.iter().enumerate() {
            matrices.push(opened_matrix(
                WHIR_BATCH_MAIN,
                chip_idx,
                chip_idx,
                dimensions[WHIR_BATCH_MAIN][chip_idx],
                &chip.main.local,
                false,
            )?);
        }
        if has_permutation_batch {
            for (chip_idx, chip) in opened_values.chips.iter().enumerate() {
                matrices.push(opened_matrix(
                    WHIR_BATCH_PERMUTATION,
                    chip_idx,
                    chip_idx,
                    dimensions[WHIR_BATCH_PERMUTATION][chip_idx],
                    &chip.permutation.local,
                    true,
                )?);
            }
        }

        Ok(Self { matrices })
    }

    pub fn assert_prep_first_height_is_max(&self) -> Result<(), WhirSpecFoldError> {
        let mut prep_matrices =
            self.matrices.iter().filter(|matrix| matrix.batch_id == WHIR_BATCH_PREPROCESSED);
        let Some(first) = prep_matrices.next() else {
            return Ok(());
        };
        let max_log_height =
            prep_matrices.fold(first.log_height, |acc, matrix| acc.max(matrix.log_height));
        if first.log_height != max_log_height {
            return Err(WhirSpecFoldError::PrepFirstHeightNotMax {
                first_log_height: first.log_height,
                max_log_height,
            });
        }
        Ok(())
    }
}

impl WhirBatchRlc {
    pub fn from_opened_matrices(opened: &WhirOpenedMatrices, alpha: [F; D_EF]) -> Self {
        let alpha_ext = limbs_to_ext(alpha);
        let mut groups_by_height: BTreeMap<usize, Vec<&WhirOpenedMatrix>> = BTreeMap::new();
        for matrix in &opened.matrices {
            groups_by_height.entry(matrix.log_height).or_default().push(matrix);
        }

        let mut alpha_power = EF::one();
        let mut prefix_acc = EF::zero();
        let mut cursor = 0;
        let mut segments = Vec::new();
        let mut groups = Vec::new();
        let mut steps = Vec::new();

        for (&log_height, matrices) in groups_by_height.iter().rev() {
            let first_cursor = cursor;
            let group_base = prefix_acc;
            let group_value_count = matrices.iter().map(|matrix| matrix.values.len()).sum();
            let mut group_offset = 0;

            for matrix in matrices {
                segments.push(WhirBatchRlcSegment {
                    log_height,
                    batch_id: matrix.batch_id,
                    batch_pos: matrix.batch_pos,
                    chip_idx: matrix.chip_idx,
                    width: matrix.width,
                    first_cursor: cursor,
                    element_count: matrix.values.len(),
                });
                for (value_idx, value_limbs) in matrix.values.iter().copied().enumerate() {
                    let value = limbs_to_ext(value_limbs);
                    let acc_in = prefix_acc;
                    prefix_acc += alpha_power * value;
                    group_offset += 1;
                    let is_group_end = group_offset == group_value_count;
                    let group_base_out = if is_group_end { prefix_acc } else { group_base };

                    steps.push(WhirBatchRlcStep {
                        cursor,
                        log_height,
                        batch_id: matrix.batch_id,
                        batch_pos: matrix.batch_pos,
                        chip_idx: matrix.chip_idx,
                        width: matrix.width,
                        value_idx,
                        value: value_limbs,
                        alpha_power: ext_limbs(&alpha_power),
                        alpha_power_out: ext_limbs(&(alpha_power * alpha_ext)),
                        acc_in: ext_limbs(&acc_in),
                        acc_out: ext_limbs(&prefix_acc),
                        group_base_in: ext_limbs(&group_base),
                        group_base_out: ext_limbs(&group_base_out),
                        is_segment_start: value_idx == 0,
                        is_group_start: group_offset == 1,
                        is_group_end,
                    });

                    alpha_power *= alpha_ext;
                    cursor += 1;
                }
            }

            groups.push(WhirBatchRlcGroup {
                log_height,
                claim: ext_limbs(&(prefix_acc - group_base)),
                first_cursor,
                element_count: group_value_count,
            });
        }

        Self { alpha, segments, groups, steps }
    }

    pub fn query_leaf_sums<B>(
        &self,
        leaf_openings: &[B],
        log_blowup: usize,
    ) -> Result<BTreeMap<usize, [F; D_EF]>, WhirSpecFoldError>
    where
        B: AsRef<[Vec<F>]>,
    {
        self.validate_query_leaf_openings(leaf_openings)?;

        let mut sums = BTreeMap::<usize, EF>::new();
        for group in &self.groups {
            sums.entry(group.log_height + log_blowup).or_insert(EF::zero());
        }
        for step in &self.steps {
            let row = &leaf_openings[step.batch_id].as_ref()[step.batch_pos];
            let value = if step.batch_id == WHIR_BATCH_PERMUTATION {
                let start = step.value_idx * D_EF;
                let end = start + D_EF;
                if end > row.len() {
                    return Err(WhirSpecFoldError::QueryOpeningWidthMismatch {
                        batch_id: step.batch_id,
                        batch_pos: step.batch_pos,
                        expected_width: step.width,
                        actual_width: row.len(),
                    });
                }
                EF::from_base_slice(&row[start..end])
            } else {
                row.get(step.value_idx).copied().map(EF::from_base).ok_or(
                    WhirSpecFoldError::QueryOpeningWidthMismatch {
                        batch_id: step.batch_id,
                        batch_pos: step.batch_pos,
                        expected_width: step.width,
                        actual_width: row.len(),
                    },
                )?
            };
            let key = step.log_height + log_blowup;
            let alpha_power = limbs_to_ext(step.alpha_power);
            *sums.entry(key).or_insert(EF::zero()) += alpha_power * value;
        }
        Ok(sums.into_iter().map(|(height, value)| (height, ext_limbs(&value))).collect())
    }

    /// The alpha-power schedule value at each height-group start —
    /// `pow(h) = alpha^(number of values in all taller groups)`. Query-independent
    /// by construction.
    /// Note: WhirBatchEval's group-start `pow_in` must equal these values; the
    /// alignment assert lives at the record call site.
    pub fn group_start_pows(&self, log_blowup: usize) -> BTreeMap<usize, EF> {
        self.groups
            .iter()
            .map(|group| {
                let pow = self
                    .steps
                    .get(group.first_cursor)
                    .map(|step| limbs_to_ext(step.alpha_power))
                    .or_else(|| self.steps.last().map(|step| limbs_to_ext(step.alpha_power_out)))
                    .unwrap_or_else(EF::one);
                (group.log_height + log_blowup, pow)
            })
            .collect()
    }

    /// Build ONE deduped height-group instance keyed (codeword height,
    /// truncated leaf index). Rows start at cursor 0 with acc = 0 and
    /// pow = `start_pow` (the schedule seed recv'd from bus 1044); there is no
    /// per-query cycle row — the intra-instance chain is linear with
    /// boundary-gated mults.
    pub fn leaf_group_stream_rows<B>(
        &self,
        proof_idx: usize,
        codeword_log_height: usize,
        idx: usize,
        leaf_openings: &[B],
        log_blowup: usize,
        start_pow: EF,
    ) -> Result<
        (Vec<RecursionWhirLeafStreamRow>, Vec<RecursionWhirLeafExtStreamRow>),
        WhirSpecFoldError,
    >
    where
        B: AsRef<[Vec<F>]>,
    {
        self.validate_query_leaf_openings(leaf_openings)?;
        self.leaf_group_stream_rows_validated(
            proof_idx,
            codeword_log_height,
            idx,
            leaf_openings,
            log_blowup,
            start_pow,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn leaf_group_stream_rows_validated<B>(
        &self,
        proof_idx: usize,
        codeword_log_height: usize,
        idx: usize,
        leaf_openings: &[B],
        log_blowup: usize,
        start_pow: EF,
    ) -> Result<
        (Vec<RecursionWhirLeafStreamRow>, Vec<RecursionWhirLeafExtStreamRow>),
        WhirSpecFoldError,
    >
    where
        B: AsRef<[Vec<F>]>,
    {
        let alpha_ext = limbs_to_ext(self.alpha);
        let mut current_pow = start_pow;
        let mut current_acc = EF::zero();
        let mut cursor = 0usize;
        let mut next_step = 0usize;
        let mut next_segment = 0usize;
        let mut block_idx_by_batch = [0usize; WHIR_BATCH_COUNT];
        let mut base_rows = Vec::new();
        let mut ext_rows = Vec::new();
        let mut base_pack: Option<LeafBasePack> = None;
        let mut ext_pack: Option<LeafExtPack> = None;
        let mut found_group = false;

        for group in &self.groups {
            let codeword_height = group.log_height + log_blowup;
            if codeword_height != codeword_log_height {
                // Skip this group's segments/steps to keep the walk offsets aligned.
                while next_segment < self.segments.len() &&
                    self.segments[next_segment].log_height == group.log_height
                {
                    next_step += self.segments[next_segment].element_count;
                    next_segment += 1;
                }
                continue;
            }
            found_group = true;
            let group_start_cursor = cursor;
            let mut emitted_group_row = false;

            while next_segment < self.segments.len() &&
                self.segments[next_segment].log_height == group.log_height
            {
                let segment = &self.segments[next_segment];
                let segment_steps = &self.steps[next_step..next_step + segment.element_count];
                let row_values = &leaf_openings[segment.batch_id].as_ref()[segment.batch_pos];
                let unit_key = whir_unit_key(input_path_slot(segment.batch_id), codeword_height);

                if segment.batch_id == WHIR_BATCH_PERMUTATION {
                    flush_base_pack(
                        &mut base_pack,
                        &mut base_rows,
                        &mut current_pow,
                        &mut current_acc,
                        &mut cursor,
                        &mut emitted_group_row,
                        &mut block_idx_by_batch,
                    );
                    for step in segment_steps {
                        let start = step.value_idx * D_EF;
                        let end = start + D_EF;
                        let value = row_values[start..end]
                            .try_into()
                            .expect("query leaf opening width was validated");
                        push_ext_leaf_value(
                            &mut ext_pack,
                            &mut ext_rows,
                            &mut current_pow,
                            &mut current_acc,
                            &mut cursor,
                            &mut emitted_group_row,
                            &mut block_idx_by_batch,
                            LeafPackMeta {
                                proof_idx,
                                idx,
                                log_height: codeword_height,
                                batch_id: segment.batch_id,
                                alpha: self.alpha,
                                unit_key,
                            },
                            value,
                            alpha_ext,
                        );
                    }
                } else {
                    flush_ext_pack(
                        &mut ext_pack,
                        &mut ext_rows,
                        &mut current_pow,
                        &mut current_acc,
                        &mut cursor,
                        &mut emitted_group_row,
                        &mut block_idx_by_batch,
                    );
                    for step in segment_steps {
                        push_base_leaf_value(
                            &mut base_pack,
                            &mut base_rows,
                            &mut current_pow,
                            &mut current_acc,
                            &mut cursor,
                            &mut emitted_group_row,
                            &mut block_idx_by_batch,
                            LeafPackMeta {
                                proof_idx,
                                idx,
                                log_height: codeword_height,
                                batch_id: segment.batch_id,
                                alpha: self.alpha,
                                unit_key,
                            },
                            row_values[step.value_idx],
                            alpha_ext,
                        );
                    }
                }

                next_step += segment.element_count;
                next_segment += 1;
            }
            flush_base_pack(
                &mut base_pack,
                &mut base_rows,
                &mut current_pow,
                &mut current_acc,
                &mut cursor,
                &mut emitted_group_row,
                &mut block_idx_by_batch,
            );
            flush_ext_pack(
                &mut ext_pack,
                &mut ext_rows,
                &mut current_pow,
                &mut current_acc,
                &mut cursor,
                &mut emitted_group_row,
                &mut block_idx_by_batch,
            );

            if !emitted_group_row {
                base_rows.push(zero_leaf_sum_row(
                    proof_idx,
                    idx,
                    cursor,
                    codeword_height,
                    0,
                    self.alpha,
                    current_pow,
                    current_acc,
                ));
                cursor += 1;
            }

            set_last_leaf_group_row_unit_end(
                &mut base_rows,
                &mut ext_rows,
                group_start_cursor,
                cursor,
            );
        }

        if !found_group {
            return Err(WhirSpecFoldError::MissingQueryLeafSum { log_height: codeword_log_height });
        }

        annotate_leaf_key_chain(&mut base_rows, &mut ext_rows);
        Ok((base_rows, ext_rows))
    }

    fn validate_query_leaf_openings<B>(&self, leaf_openings: &[B]) -> Result<(), WhirSpecFoldError>
    where
        B: AsRef<[Vec<F>]>,
    {
        let required_batches = self
            .segments
            .iter()
            .map(|segment| segment.batch_id)
            .max()
            .map_or(0, |batch_id| batch_id + 1);
        if leaf_openings.len() < required_batches {
            return Err(WhirSpecFoldError::QueryOpeningBatchCountMismatch {
                expected_min: required_batches,
                actual: leaf_openings.len(),
            });
        }
        for segment in &self.segments {
            let batch = leaf_openings[segment.batch_id].as_ref();
            if batch.len() <= segment.batch_pos {
                return Err(WhirSpecFoldError::QueryOpeningMatrixCountMismatch {
                    batch_id: segment.batch_id,
                    expected_min: segment.batch_pos + 1,
                    actual: batch.len(),
                });
            }
            let actual_width = batch[segment.batch_pos].len();
            if actual_width != segment.width {
                return Err(WhirSpecFoldError::QueryOpeningWidthMismatch {
                    batch_id: segment.batch_id,
                    batch_pos: segment.batch_pos,
                    expected_width: segment.width,
                    actual_width,
                });
            }
        }
        Ok(())
    }
}

fn input_path_slot(batch_id: usize) -> usize {
    match batch_id {
        WHIR_BATCH_PREPROCESSED => WHIR_INPUT_PREPROCESSED_PATH_SLOT,
        WHIR_BATCH_MAIN => WHIR_INPUT_MAIN_PATH_SLOT,
        WHIR_BATCH_PERMUTATION => WHIR_INPUT_PERMUTATION_PATH_SLOT,
        _ => panic!("unsupported WHIR input batch id {batch_id}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeafPackMeta {
    proof_idx: usize,
    idx: usize,
    log_height: usize,
    batch_id: usize,
    alpha: [F; D_EF],
    unit_key: usize,
}

#[derive(Debug, Clone)]
struct LeafBasePack {
    meta: LeafPackMeta,
    is_unit_start: bool,
    block_idx: usize,
    pow_in: EF,
    acc_in: EF,
    pow: EF,
    acc: EF,
    slot_pows: [[F; D_EF]; WHIR_LEAF_BASE_LIMBS_PER_ROW],
    values: [F; WHIR_LEAF_BASE_LIMBS_PER_ROW],
    chunk_mask: [bool; WHIR_LEAF_BASE_LIMBS_PER_ROW],
    len: usize,
}

#[derive(Debug, Clone)]
struct LeafExtPack {
    meta: LeafPackMeta,
    is_unit_start: bool,
    block_idx: usize,
    pow_in: EF,
    acc_in: EF,
    pow: EF,
    acc: EF,
    slot_pows: [[F; D_EF]; WHIR_LEAF_RLC_SLOTS],
    value_blocks: [[F; WHIR_LEAF_BASE_LIMBS_PER_ROW]; WHIR_LEAF_BLOCKS_PER_ROW],
    chunk_masks: [[bool; WHIR_LEAF_BASE_LIMBS_PER_ROW]; WHIR_LEAF_BLOCKS_PER_ROW],
    len: usize,
}

impl From<RecursionWhirLeafExtStreamRow> for RecursionWhirLeafExtStreamTraceRow {
    fn from(row: RecursionWhirLeafExtStreamRow) -> Self {
        Self {
            is_unit_end: row.is_unit_end,
            is_unit_key_start: row.is_unit_key_start,
            element_masks: core::array::from_fn(|elem_idx| {
                let flat_idx = elem_idx * D_EF;
                row.chunk_masks[flat_idx / WHIR_LEAF_BASE_LIMBS_PER_ROW]
                    [flat_idx % WHIR_LEAF_BASE_LIMBS_PER_ROW]
            }),
            idx: row.idx,
            serve_cnt: row.serve_cnt,
            chain_recv_cursor: row.chain_recv_cursor,
            log_height: row.log_height,
            block_idx: row.block_idx,
            alpha: row.alpha,
            pow_in: row.pow_in,
            acc_in: row.acc_in,
            slot_pows: core::array::from_fn(|slot| row.slot_pows[slot + 1]),
            pow_out: row.pow_out,
            acc_out: row.acc_out,
            value_blocks: row.value_blocks,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_base_leaf_value(
    pack: &mut Option<LeafBasePack>,
    rows: &mut Vec<RecursionWhirLeafStreamRow>,
    current_pow: &mut EF,
    current_acc: &mut EF,
    cursor: &mut usize,
    emitted_group_row: &mut bool,
    block_idx_by_batch: &mut [usize; WHIR_BATCH_COUNT],
    meta: LeafPackMeta,
    value: F,
    alpha_ext: EF,
) {
    let should_flush = pack
        .as_ref()
        .is_some_and(|pack| pack.meta != meta || pack.len == WHIR_LEAF_BASE_LIMBS_PER_ROW);
    if should_flush {
        flush_base_pack(
            pack,
            rows,
            current_pow,
            current_acc,
            cursor,
            emitted_group_row,
            block_idx_by_batch,
        );
    }
    if pack.is_none() {
        let acc = if !*emitted_group_row { EF::zero() } else { *current_acc };
        let block_idx = block_idx_by_batch[meta.batch_id];
        *pack = Some(LeafBasePack {
            meta,
            is_unit_start: !*emitted_group_row,
            block_idx,
            pow_in: *current_pow,
            acc_in: *current_acc,
            pow: *current_pow,
            acc,
            slot_pows: [[F::zero(); D_EF]; WHIR_LEAF_BASE_LIMBS_PER_ROW],
            values: [F::zero(); WHIR_LEAF_BASE_LIMBS_PER_ROW],
            chunk_mask: [false; WHIR_LEAF_BASE_LIMBS_PER_ROW],
            len: 0,
        });
    }
    let pack = pack.as_mut().expect("base pack was initialized");
    let slot = pack.len;
    pack.slot_pows[slot] = ext_limbs(&pack.pow);
    pack.values[slot] = value;
    pack.chunk_mask[slot] = true;
    pack.acc += pack.pow * EF::from_base(value);
    pack.pow *= alpha_ext;
    pack.len += 1;
}

fn flush_base_pack(
    pack: &mut Option<LeafBasePack>,
    rows: &mut Vec<RecursionWhirLeafStreamRow>,
    current_pow: &mut EF,
    current_acc: &mut EF,
    cursor: &mut usize,
    emitted_group_row: &mut bool,
    block_idx_by_batch: &mut [usize; WHIR_BATCH_COUNT],
) {
    let Some(mut pack) = pack.take() else {
        return;
    };
    if pack.len == 0 {
        return;
    }
    for slot in pack.len..WHIR_LEAF_BASE_LIMBS_PER_ROW {
        pack.slot_pows[slot] = ext_limbs(&pack.pow);
    }
    let meta = pack.meta;
    rows.push(RecursionWhirLeafStreamRow {
        proof_idx: meta.proof_idx,
        is_unit_start: pack.is_unit_start,
        idx: meta.idx,
        cursor: *cursor,
        chain_recv_cursor: *cursor,
        chain_send_cursor: *cursor + 1,
        log_height: meta.log_height,
        batch_id: meta.batch_id,
        alpha: meta.alpha,
        pow_in: ext_limbs(&pack.pow_in),
        acc_in: ext_limbs(&pack.acc_in),
        slot_pows: pack.slot_pows,
        pow_out: ext_limbs(&pack.pow),
        acc_out: ext_limbs(&pack.acc),
        values: pack.values,
        chunk_mask: pack.chunk_mask,
        unit_key: meta.unit_key,
        block_idx: pack.block_idx,
        ..Default::default()
    });
    *current_pow = pack.pow;
    *current_acc = pack.acc;
    *cursor += 1;
    *emitted_group_row = true;
    block_idx_by_batch[meta.batch_id] += 1;
}

#[allow(clippy::too_many_arguments)]
fn push_ext_leaf_value(
    pack: &mut Option<LeafExtPack>,
    rows: &mut Vec<RecursionWhirLeafExtStreamRow>,
    current_pow: &mut EF,
    current_acc: &mut EF,
    cursor: &mut usize,
    emitted_group_row: &mut bool,
    block_idx_by_batch: &mut [usize; WHIR_BATCH_COUNT],
    meta: LeafPackMeta,
    value: [F; D_EF],
    alpha_ext: EF,
) {
    let should_flush =
        pack.as_ref().is_some_and(|pack| pack.meta != meta || pack.len == WHIR_LEAF_RLC_SLOTS);
    if should_flush {
        flush_ext_pack(
            pack,
            rows,
            current_pow,
            current_acc,
            cursor,
            emitted_group_row,
            block_idx_by_batch,
        );
    }
    if pack.is_none() {
        let acc = if !*emitted_group_row { EF::zero() } else { *current_acc };
        let block_idx = block_idx_by_batch[meta.batch_id];
        *pack = Some(LeafExtPack {
            meta,
            is_unit_start: !*emitted_group_row,
            block_idx,
            pow_in: *current_pow,
            acc_in: *current_acc,
            pow: *current_pow,
            acc,
            slot_pows: [[F::zero(); D_EF]; WHIR_LEAF_RLC_SLOTS],
            value_blocks: [[F::zero(); WHIR_LEAF_BASE_LIMBS_PER_ROW]; WHIR_LEAF_BLOCKS_PER_ROW],
            chunk_masks: [[false; WHIR_LEAF_BASE_LIMBS_PER_ROW]; WHIR_LEAF_BLOCKS_PER_ROW],
            len: 0,
        });
    }
    let pack = pack.as_mut().expect("ext pack was initialized");
    let elem_idx = pack.len;
    pack.slot_pows[elem_idx] = ext_limbs(&pack.pow);
    for (limb_idx, limb) in value.iter().copied().enumerate() {
        let flat_idx = elem_idx * D_EF + limb_idx;
        pack.value_blocks[flat_idx / WHIR_LEAF_BASE_LIMBS_PER_ROW]
            [flat_idx % WHIR_LEAF_BASE_LIMBS_PER_ROW] = limb;
        pack.chunk_masks[flat_idx / WHIR_LEAF_BASE_LIMBS_PER_ROW]
            [flat_idx % WHIR_LEAF_BASE_LIMBS_PER_ROW] = true;
    }
    pack.acc += pack.pow * EF::from_base_slice(&value);
    pack.pow *= alpha_ext;
    pack.len += 1;
}

fn flush_ext_pack(
    pack: &mut Option<LeafExtPack>,
    rows: &mut Vec<RecursionWhirLeafExtStreamRow>,
    current_pow: &mut EF,
    current_acc: &mut EF,
    cursor: &mut usize,
    emitted_group_row: &mut bool,
    block_idx_by_batch: &mut [usize; WHIR_BATCH_COUNT],
) {
    let Some(mut pack) = pack.take() else {
        return;
    };
    if pack.len == 0 {
        return;
    }
    for elem_idx in pack.len..WHIR_LEAF_RLC_SLOTS {
        pack.slot_pows[elem_idx] = ext_limbs(&pack.pow);
    }
    let meta = pack.meta;
    rows.push(RecursionWhirLeafExtStreamRow {
        proof_idx: meta.proof_idx,
        is_unit_start: pack.is_unit_start,
        idx: meta.idx,
        cursor: *cursor,
        chain_recv_cursor: *cursor,
        chain_send_cursor: *cursor + 1,
        log_height: meta.log_height,
        batch_id: meta.batch_id,
        alpha: meta.alpha,
        pow_in: ext_limbs(&pack.pow_in),
        acc_in: ext_limbs(&pack.acc_in),
        slot_pows: pack.slot_pows,
        pow_out: ext_limbs(&pack.pow),
        acc_out: ext_limbs(&pack.acc),
        value_blocks: pack.value_blocks,
        chunk_masks: pack.chunk_masks,
        unit_key: meta.unit_key,
        block_idx: pack.block_idx,
        ..Default::default()
    });
    *current_pow = pack.pow;
    *current_acc = pack.acc;
    *cursor += 1;
    *emitted_group_row = true;
    let block_count = (pack.len * D_EF).div_ceil(WHIR_LEAF_BASE_LIMBS_PER_ROW);
    block_idx_by_batch[meta.batch_id] += block_count;
}

fn annotate_leaf_key_chain(
    base_rows: &mut [RecursionWhirLeafStreamRow],
    ext_rows: &mut [RecursionWhirLeafExtStreamRow],
) {
    // Each slice is cursor-ordered by construction. Merge them once so
    // annotation remains linear even for wide permutation segments.
    let mut base_idx = 0usize;
    let mut ext_idx = 0usize;
    let mut expected_cursor = 0usize;
    let mut prev: Option<(usize, usize)> = None;
    while base_idx < base_rows.len() || ext_idx < ext_rows.len() {
        let take_base = match (base_rows.get(base_idx), ext_rows.get(ext_idx)) {
            (Some(base), Some(ext)) => base.cursor <= ext.cursor,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let (cursor, log_height, batch_id) = if take_base {
            let row = &base_rows[base_idx];
            (row.cursor, row.log_height, row.batch_id)
        } else {
            let row = &ext_rows[ext_idx];
            (row.cursor, row.log_height, row.batch_id)
        };
        assert_eq!(cursor, expected_cursor, "leaf instance cursors must be dense and ordered");
        expected_cursor += 1;
        let (key_start, gap, recv_h, recv_b) = match prev {
            None => (true, 0, 0, 0),
            Some((ph, pb)) => {
                assert_eq!(ph, log_height, "leaf instance must stay at one height");
                if batch_id != pb {
                    let gap = batch_id
                        .checked_sub(pb + 1)
                        .expect("leaf unit batch id must strictly increase within a height");
                    (true, gap, ph, pb)
                } else {
                    (false, 0, ph, pb)
                }
            }
        };
        if take_base {
            let row = &mut base_rows[base_idx];
            row.chain_recv_log_height = recv_h;
            row.chain_recv_batch_id = recv_b;
            row.is_unit_key_start = key_start;
            row.unit_key_gap = gap;
            base_idx += 1;
        } else {
            let row = &mut ext_rows[ext_idx];
            row.chain_recv_log_height = recv_h;
            row.chain_recv_batch_id = recv_b;
            row.is_unit_key_start = key_start;
            row.unit_key_gap = gap;
            ext_idx += 1;
        }
        prev = Some((log_height, batch_id));
    }
}

fn zero_leaf_sum_row(
    proof_idx: usize,
    idx: usize,
    cursor: usize,
    log_height: usize,
    batch_id: usize,
    alpha: [F; D_EF],
    current_pow: EF,
    current_acc: EF,
) -> RecursionWhirLeafStreamRow {
    let pow = ext_limbs(&current_pow);
    RecursionWhirLeafStreamRow {
        proof_idx,
        is_unit_start: true,
        is_unit_end: true,
        idx,
        cursor,
        chain_recv_cursor: cursor,
        chain_send_cursor: cursor + 1,
        log_height,
        batch_id,
        alpha,
        pow_in: pow,
        acc_in: ext_limbs(&current_acc),
        slot_pows: [pow; WHIR_LEAF_BASE_LIMBS_PER_ROW],
        pow_out: pow,
        acc_out: ext_limbs(&EF::zero()),
        ..Default::default()
    }
}

fn set_last_leaf_group_row_unit_end(
    base_rows: &mut [RecursionWhirLeafStreamRow],
    ext_rows: &mut [RecursionWhirLeafExtStreamRow],
    start_cursor: usize,
    end_cursor: usize,
) {
    if start_cursor == end_cursor {
        return;
    }
    let last_cursor = end_cursor - 1;
    if ext_rows.last().is_some_and(|row| row.cursor == last_cursor) {
        ext_rows.last_mut().expect("checked final Ext row").is_unit_end = true;
    } else {
        let row = base_rows.last_mut().expect("non-empty leaf group has a final row");
        assert_eq!(row.cursor, last_cursor, "final Base row must have the last cursor");
        row.is_unit_end = true;
    }
}

impl WhirRoundReplayInput {
    /// Transcript index of the first WHIR query sample.
    ///
    /// Compact device tracegen needs this control value before it expands any
    /// round rows. Computing it is purely structural and deliberately does
    /// not invoke Poseidon or mutate the request-local permutation memo.
    pub fn query_sample_tidx_base(&self) -> Result<usize, WhirSpecFoldError> {
        let shape = self.seed.shape;
        let mut group_by_height = BTreeMap::<usize, (usize, [F; D_EF])>::new();
        for (rank, group) in self.group_claims.iter().enumerate() {
            if group_by_height.insert(group.log_height, (rank, group.claim)).is_some() {
                return Err(WhirSpecFoldError::DuplicateHeightGroup {
                    log_height: group.log_height,
                });
            }
        }
        let preamble_tidx = shape.w0_tidx + D_EF + 3;
        let mut round_tidx = preamble_tidx + 8;
        for round in 0..shape.num_rounds {
            round_tidx += round_event_stride(round, shape.num_rounds, &group_by_height);
        }
        Ok(round_tidx + 11)
    }

    /// Replay only the compact authority needed by GPU round/query expansion.
    pub fn compact_round_authority(
        &self,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Result<WhirCompactRoundAuthority, WhirSpecFoldError> {
        let shape = self.seed.shape;
        let rounds = shape.num_rounds;
        if self.seed.opening_point.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "opening_point",
                expected: rounds,
                actual: self.seed.opening_point.len(),
            });
        }
        if self.sumcheck_coeffs.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "sumcheck_coeffs",
                expected: rounds,
                actual: self.sumcheck_coeffs.len(),
            });
        }
        if self.r_folds.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "r_folds",
                expected: rounds,
                actual: self.r_folds.len(),
            });
        }
        if self.iopp_oracles.len() != rounds + 1 {
            return Err(WhirSpecFoldError::RoundReplayOracleCountMismatch {
                expected: rounds + 1,
                actual: self.iopp_oracles.len(),
            });
        }
        let mut group_by_height = BTreeMap::<usize, (usize, [F; D_EF])>::new();
        for (rank, group) in self.group_claims.iter().enumerate() {
            if group_by_height.insert(group.log_height, (rank, group.claim)).is_some() {
                return Err(WhirSpecFoldError::DuplicateHeightGroup {
                    log_height: group.log_height,
                });
            }
        }
        let (_, tallest_claim) = group_by_height
            .get(&rounds)
            .copied()
            .ok_or(WhirSpecFoldError::MissingHeightGroup { log_height: rounds })?;
        for &height in self.merge_betas_by_height.keys() {
            if height >= rounds || !group_by_height.contains_key(&height) {
                return Err(WhirSpecFoldError::UnexpectedMergeBeta { log_height: height });
            }
        }

        let mut recv_claim = tallest_claim;
        let mut recv_eq = one_ext_limbs();
        let mut recv_pending_is_merge = true;
        let mut recv_pending_beta = [F::zero(); D_EF];
        let mut recv_pending_eq = [F::zero(); D_EF];
        let mut controls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let opening_idx = rounds - round - 1;
            let merge = group_by_height.get(&opening_idx).copied();
            let merge_beta = if merge.is_some() {
                self.merge_betas_by_height
                    .get(&opening_idx)
                    .copied()
                    .ok_or(WhirSpecFoldError::MissingMergeBeta { log_height: opening_idx })?
            } else {
                [F::zero(); D_EF]
            };
            controls.push(WhirQueryRoundControl {
                r_fold: self.r_folds[round],
                is_merge: recv_pending_is_merge,
                is_assign: recv_pending_is_merge && round == 0,
                merge_beta: recv_pending_beta,
                merge_eq: recv_pending_eq,
                emit_prep_seed: self.prep_seed_round == Some(round),
                cfr: [F::zero(); D_EF],
            });

            let coeffs = self.sumcheck_coeffs[round];
            let r = limbs_to_ext(self.r_folds[round]);
            let claim_acc = limbs_to_ext(coeffs[1]) + r * limbs_to_ext(coeffs[2]);
            let claim_folded = limbs_to_ext(coeffs[0]) + r * claim_acc;
            let one = EF::one();
            let z = limbs_to_ext(self.seed.opening_point[opening_idx]);
            let eq_folded = limbs_to_ext(recv_eq) * (z * r + (one - z) * (one - r));
            let mut send_claim = claim_folded;
            let mut send_eq = eq_folded;
            if let Some((_, claim)) = merge {
                send_claim += limbs_to_ext(merge_beta) * limbs_to_ext(claim);
                send_eq = EF::one();
            }
            recv_claim = ext_limbs(&send_claim);
            recv_eq = ext_limbs(&send_eq);
            recv_pending_is_merge = merge.is_some();
            recv_pending_beta = merge_beta;
            recv_pending_eq =
                if merge.is_some() { ext_limbs(&eq_folded) } else { [F::zero(); D_EF] };
        }
        let final_eq_inv = limbs_to_ext(recv_eq).try_inverse().unwrap_or(EF::zero());
        let cfr = ext_limbs(&(limbs_to_ext(recv_claim) * final_eq_inv));
        for control in &mut controls {
            control.cfr = cfr;
        }
        let final_root =
            WhirFinalRootSponge::from_combined_f_r(cfr, shape.log_blowup, poseidon2_output)
                .round_fields();
        let output_count = rounds.checked_add(3 + WHIR_FINAL_ROOT_POSEIDON2_PERMS).ok_or(
            WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "output_count",
                expected: rounds,
                actual: usize::MAX,
            },
        )?;
        Ok(WhirCompactRoundAuthority {
            controls,
            w_qbase: self.query_sample_tidx_base()?,
            output_count,
            final_row_idx: rounds + 2 + WHIR_FINAL_ROOT_POSEIDON2_PERMS,
            batching_pow_sample_high: whir_pow_sample_high(
                "batching",
                self.batching_pow_events[2],
                WHIR_BATCHING_POW_BITS,
                WHIR_BATCHING_POW_HIGH_MAX,
            )?,
            query_pow_sample_high: whir_pow_sample_high(
                "query",
                self.query_pow_events[2],
                WHIR_QUERY_POW_BITS,
                WHIR_QUERY_POW_HIGH_MAX,
            )?,
            final_root,
        })
    }

    pub fn round_rows(
        &self,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Result<Vec<RecursionWhirRoundRow>, WhirSpecFoldError> {
        self.round_rows_impl(poseidon2_output).map(|(rows, _)| rows)
    }

    fn round_rows_impl(
        &self,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Result<(Vec<RecursionWhirRoundRow>, WhirFinalRootRowFields), WhirSpecFoldError> {
        let shape = self.seed.shape;
        let rounds = shape.num_rounds;
        if self.seed.opening_point.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "opening_point",
                expected: rounds,
                actual: self.seed.opening_point.len(),
            });
        }
        if self.sumcheck_coeffs.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "sumcheck_coeffs",
                expected: rounds,
                actual: self.sumcheck_coeffs.len(),
            });
        }
        if self.r_folds.len() != rounds {
            return Err(WhirSpecFoldError::RoundReplayRoundCountMismatch {
                field: "r_folds",
                expected: rounds,
                actual: self.r_folds.len(),
            });
        }
        if self.iopp_oracles.len() != rounds + 1 {
            return Err(WhirSpecFoldError::RoundReplayOracleCountMismatch {
                expected: rounds + 1,
                actual: self.iopp_oracles.len(),
            });
        }
        let mut group_by_height = BTreeMap::<usize, (usize, [F; D_EF])>::new();
        for (rank, group) in self.group_claims.iter().enumerate() {
            if group_by_height.insert(group.log_height, (rank, group.claim)).is_some() {
                return Err(WhirSpecFoldError::DuplicateHeightGroup {
                    log_height: group.log_height,
                });
            }
        }
        let (tallest_rank, tallest_claim) = group_by_height
            .get(&rounds)
            .copied()
            .ok_or(WhirSpecFoldError::MissingHeightGroup { log_height: rounds })?;
        for &height in self.merge_betas_by_height.keys() {
            if height >= rounds || !group_by_height.contains_key(&height) {
                return Err(WhirSpecFoldError::UnexpectedMergeBeta { log_height: height });
            }
        }

        let mut rows = Vec::with_capacity(rounds + 3 + WHIR_FINAL_ROOT_POSEIDON2_PERMS);
        let pow_tidx = shape.w0_tidx + D_EF;
        let preamble_tidx = pow_tidx + 3;
        let mut round_tidx = preamble_tidx + 8;
        for round in 0..rounds {
            round_tidx += round_event_stride(round, rounds, &group_by_height);
        }
        let final_tidx = round_tidx;
        let w_qbase = final_tidx + 11;

        let mut pow_row = base_round_row(self.seed.proof_idx, shape, w_qbase, self.summary_id_base);
        pow_row.is_pow_batch = true;
        pow_row.tidx = pow_tidx;
        pow_row.event_value[0] = self.batching_pow_events[0];
        pow_row.event_value[1] = self.batching_pow_events[1];
        pow_row.event_value[2] = self.batching_pow_events[2];
        pow_row.pow_sample_high = whir_pow_sample_high(
            "batching",
            self.batching_pow_events[2],
            WHIR_BATCHING_POW_BITS,
            WHIR_BATCHING_POW_HIGH_MAX,
        )?;
        pow_row.chain_send_tidx = preamble_tidx;
        pow_row.chain_recv_mult = 1;
        pow_row.chain_send_mult = 1;
        pow_row.role_config_recv_mult = 1;
        pow_row.summary_recv_mult = 1;
        rows.push(pow_row);

        let mut preamble_row =
            base_round_row(self.seed.proof_idx, shape, w_qbase, self.summary_id_base);
        preamble_row.is_preamble = true;
        preamble_row.tidx = preamble_tidx;
        preamble_row.chain_recv_tidx = preamble_tidx;
        preamble_row.chain_send_tidx = preamble_tidx + 8;
        preamble_row.height_group_rank = tallest_rank;
        preamble_row.height_group_log_height = rounds;
        preamble_row.group_claim_log_height = rounds;
        preamble_row.group_claim = tallest_claim;
        preamble_row.commit_id = 100;
        preamble_row.commit_root = self.iopp_oracles[0];
        preamble_row.event_value[..8].copy_from_slice(&self.iopp_oracles[0]);
        preamble_row.chain_send_claim = tallest_claim;
        preamble_row.chain_send_eq = one_ext_limbs();
        preamble_row.chain_send_pending_is_merge = true;
        preamble_row.bcast_mult = 0;
        preamble_row.commitment_root_send_mult = shape.num_queries as u32;
        preamble_row.chain_recv_mult = 1;
        preamble_row.chain_send_mult = 1;
        preamble_row.height_group_recv_mult = 1;
        preamble_row.group_claim_recv_mult = 1;
        rows.push(preamble_row);

        let mut recv_round = 0usize;
        let mut recv_tidx = preamble_tidx + 8;
        let mut recv_claim = tallest_claim;
        let mut recv_eq = one_ext_limbs();
        let mut recv_pending_is_merge = true;
        let mut recv_pending_beta = [F::zero(); D_EF];
        let mut recv_pending_eq = [F::zero(); D_EF];

        let mut tidx = recv_tidx;
        for round in 0..rounds {
            let opening_idx = rounds - round - 1;
            let expected_merge_height = rounds - round - 1;
            let merge = group_by_height.get(&expected_merge_height).copied();
            let merge_beta = if merge.is_some() {
                self.merge_betas_by_height.get(&expected_merge_height).copied().ok_or(
                    WhirSpecFoldError::MissingMergeBeta { log_height: expected_merge_height },
                )?
            } else {
                [F::zero(); D_EF]
            };
            let coeffs = self.sumcheck_coeffs[round];
            let r_fold = self.r_folds[round];
            let c0 = limbs_to_ext(coeffs[0]);
            let c1 = limbs_to_ext(coeffs[1]);
            let c2 = limbs_to_ext(coeffs[2]);
            // Preserve both sides in the generated row. The AIR checks
            // g(0)+g(1)=claim; tracegen does not pre-verify that identity.
            let r = limbs_to_ext(r_fold);
            let claim_acc = c1 + r * c2;
            let claim_folded = c0 + r * claim_acc;
            let one = EF::one();
            let z = limbs_to_ext(self.seed.opening_point[opening_idx]);
            let eq_factor = z * r + (one - z) * (one - r);
            let eq_folded = limbs_to_ext(recv_eq) * eq_factor;
            let mut send_claim = claim_folded;
            let mut send_eq = eq_folded;
            if let Some((_, claim)) = merge {
                send_claim += limbs_to_ext(merge_beta) * limbs_to_ext(claim);
                send_eq = EF::one();
            }
            let merge_height = merge.map(|_| expected_merge_height);
            let is_merge = merge_height.is_some();
            let (height_group_rank, height_group_log_height, group_claim) =
                if let Some(merge_height) = merge_height {
                    let (rank, claim) = group_by_height.get(&merge_height).copied().ok_or(
                        WhirSpecFoldError::MissingHeightGroup { log_height: merge_height },
                    )?;
                    (rank, merge_height, claim)
                } else {
                    (0, 0, [F::zero(); D_EF])
                };

            let mut row = base_round_row(self.seed.proof_idx, shape, w_qbase, self.summary_id_base);
            row.is_round = true;
            row.round = round;
            row.tidx = tidx;
            row.opening_idx = opening_idx;
            row.opening_point = self.seed.opening_point[opening_idx];
            row.height_group_rank = height_group_rank;
            row.height_group_log_height = height_group_log_height;
            row.group_claim_log_height = height_group_log_height;
            row.group_claim = group_claim;
            row.round_has_oracle = round > 0;
            if row.round_has_oracle {
                row.commit_id = 100 + round;
                row.commit_root = self.iopp_oracles[round];
                row.event_value[..8].copy_from_slice(&self.iopp_oracles[round]);
            }
            for coeff_idx in 0..3 {
                let start = 8 + coeff_idx * D_EF;
                row.event_value[start..start + D_EF].copy_from_slice(&coeffs[coeff_idx]);
            }
            row.event_value[23..28].copy_from_slice(&r_fold);
            if is_merge {
                row.event_value[28..32].copy_from_slice(&merge_beta[..4]);
                row.event_value_last = merge_beta[4];
            }
            row.chain_recv_round = recv_round;
            row.chain_recv_tidx = recv_tidx;
            row.chain_recv_claim = recv_claim;
            row.chain_recv_eq = recv_eq;
            row.chain_recv_pending_is_merge = recv_pending_is_merge;
            row.chain_recv_pending_beta = recv_pending_beta;
            row.chain_recv_pending_eq = recv_pending_eq;
            row.chain_send_round = recv_round + 1;
            row.chain_send_tidx = tidx + round_event_stride(round, rounds, &group_by_height);
            row.chain_send_claim = ext_limbs(&send_claim);
            row.chain_send_eq = ext_limbs(&send_eq);
            row.chain_send_pending_is_merge = is_merge;
            row.chain_send_pending_beta = merge_beta;
            row.chain_send_pending_eq =
                if is_merge { ext_limbs(&eq_folded) } else { [F::zero(); D_EF] };
            row.r_fold = r_fold;
            row.is_merge = is_merge;
            row.emit_prep_seed = self.prep_seed_round == Some(round);
            row.merge_log_height = shape.num_rounds + shape.log_blowup - round;
            row.claim_acc = ext_limbs(&claim_acc);
            row.claim_folded = ext_limbs(&claim_folded);
            row.eq_factor = ext_limbs(&eq_factor);
            row.eq_folded = ext_limbs(&eq_folded);
            row.bcast_mult = shape.num_queries as u32;
            row.chain_recv_mult = 1;
            row.chain_send_mult = 1;
            row.opening_point_recv_mult = 1;
            row.height_group_recv_mult = u32::from(is_merge);
            row.group_claim_recv_mult = u32::from(is_merge);
            row.commitment_root_send_mult =
                if row.round_has_oracle { shape.num_queries as u32 } else { 0 };
            rows.push(row);

            recv_round += 1;
            recv_tidx = tidx + round_event_stride(round, rounds, &group_by_height);
            recv_claim = ext_limbs(&send_claim);
            recv_eq = ext_limbs(&send_eq);
            recv_pending_is_merge = is_merge;
            recv_pending_beta = merge_beta;
            recv_pending_eq = if is_merge { ext_limbs(&eq_folded) } else { [F::zero(); D_EF] };
            tidx = recv_tidx;
        }

        let final_eq = limbs_to_ext(recv_eq);
        let final_eq_inv = final_eq.try_inverse().unwrap_or(EF::zero());
        let cfr = limbs_to_ext(recv_claim) * final_eq_inv;
        let cfr_limbs = ext_limbs(&cfr);
        let final_root_sponge =
            WhirFinalRootSponge::from_combined_f_r(cfr_limbs, shape.log_blowup, poseidon2_output);
        let final_root = final_root_sponge.round_fields();
        // `final_root.digest` and the claimed final oracle are both retained;
        // their equality is enforced by the recursive AIR buses.
        for row in rows.iter_mut().filter(|row| row.is_round) {
            row.cfr = cfr_limbs;
        }

        let mut final_row =
            base_round_row(self.seed.proof_idx, shape, w_qbase, self.summary_id_base);
        final_row.is_final = true;
        final_row.tidx = final_tidx;
        final_row.chain_recv_round = recv_round;
        final_row.chain_recv_tidx = recv_tidx;
        final_row.chain_recv_claim = recv_claim;
        final_row.chain_recv_eq = recv_eq;
        final_row.chain_recv_pending_is_merge = recv_pending_is_merge;
        final_row.chain_recv_pending_beta = recv_pending_beta;
        final_row.chain_recv_pending_eq = recv_pending_eq;
        final_row.chain_send_round = recv_round;
        final_row.chain_send_tidx = w_qbase;
        final_row.chain_send_claim = recv_claim;
        final_row.chain_send_eq = recv_eq;
        final_row.opening_idx = WHIR_FINAL_ROOT_POSEIDON2_PERMS;
        final_row.height_group_rank = 0;
        final_row.cfr = cfr_limbs;
        final_row.event_value[..8].copy_from_slice(&self.iopp_oracles[rounds]);
        final_row.event_value[8] = self.query_pow_events[0];
        final_row.event_value[9] = self.query_pow_events[1];
        final_row.event_value[10] = self.query_pow_events[2];
        final_row.pow_sample_high = whir_pow_sample_high(
            "query",
            self.query_pow_events[2],
            WHIR_QUERY_POW_BITS,
            WHIR_QUERY_POW_HIGH_MAX,
        )?;
        final_row.final_root_poseidon2_inputs = final_root.inputs;
        final_row.final_root_poseidon2_outputs = final_root.outputs;
        final_row.query_init_mult = shape.num_queries as u32;
        final_row.chain_recv_mult = 1;
        final_row.chain_send_mult = 1;
        final_row.final_root_poseidon2_recv_mults = final_root.recv_mults;
        let mut final_root_state = final_root_seed_state(cfr_limbs);
        for step in 0..WHIR_FINAL_ROOT_POSEIDON2_PERMS {
            let mut perm_row =
                base_round_row(self.seed.proof_idx, shape, w_qbase, self.summary_id_base);
            perm_row.is_final_perm = true;
            perm_row.final_root_perm_step_flags[step] = true;
            perm_row.opening_idx = step;
            perm_row.height_group_rank = step + 1;
            perm_row.cfr = cfr_limbs;
            perm_row.final_root_poseidon2_input = final_root_state;
            perm_row.final_root_poseidon2_output = final_root.outputs[step];
            perm_row.final_root_poseidon2_recv_mult = final_root.recv_mults[step];
            final_root_state = final_root_next_state(
                cfr_limbs,
                shape.log_blowup,
                step,
                final_root_state,
                final_root.outputs[step],
            );
            rows.push(perm_row);
        }
        final_row.final_root_poseidon2_output = final_root_state;
        rows.push(final_row);

        let final_send = rows.last().copied().expect("final row just pushed");
        let pow = rows.first_mut().expect("pow row exists");
        pow.chain_recv_round = final_send.chain_send_round;
        pow.chain_recv_tidx = final_send.chain_send_tidx;
        pow.chain_recv_claim = final_send.chain_send_claim;
        pow.chain_recv_eq = final_send.chain_send_eq;
        pow.chain_recv_pending_is_merge = final_send.chain_send_pending_is_merge;
        pow.chain_recv_pending_beta = final_send.chain_send_pending_beta;
        pow.chain_recv_pending_eq = final_send.chain_send_pending_eq;

        Ok((rows, final_root))
    }
}

impl WhirQueryRoundControl {
    pub fn from_round_rows(
        shape: WhirSpecFoldShape,
        round_rows: &[RecursionWhirRoundRow],
    ) -> Result<Vec<Self>, WhirSpecFoldError> {
        let rows = round_rows.iter().copied().filter(|row| row.is_round).collect::<Vec<_>>();
        if rows.len() != shape.num_rounds {
            return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
                field: "round_rows",
                expected: shape.num_rounds,
                actual: rows.len(),
            });
        }

        rows.into_iter()
            .enumerate()
            .map(|(idx, row)| {
                if row.round != idx {
                    return Err(WhirSpecFoldError::QueryReplayRoundIndexMismatch {
                        expected: idx,
                        actual: row.round,
                    });
                }
                let is_merge = row.chain_recv_pending_is_merge;
                Ok(Self {
                    r_fold: row.r_fold,
                    is_merge,
                    is_assign: is_merge && row.round == 0,
                    merge_beta: row.chain_recv_pending_beta,
                    merge_eq: row.chain_recv_pending_eq,
                    emit_prep_seed: row.emit_prep_seed,
                    cfr: row.cfr,
                })
            })
            .collect()
    }
}

impl WhirQueryReplayInput {
    pub fn from_sibling_values(
        seed: WhirSpecFoldSeed,
        query_idx: usize,
        w_qbase: usize,
        query_sample_raw: F,
        query_sample: usize,
        controls: Vec<WhirQueryRoundControl>,
        siblings: Vec<[F; D_EF]>,
        leaf_sums_by_log_height: BTreeMap<usize, [F; D_EF]>,
    ) -> Result<Self, WhirSpecFoldError> {
        let rounds = seed.shape.num_rounds;
        if controls.len() != rounds {
            return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
                field: "controls",
                expected: rounds,
                actual: controls.len(),
            });
        }
        if siblings.len() != rounds {
            return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
                field: "siblings",
                expected: rounds,
                actual: siblings.len(),
            });
        }
        if query_sample >= (1usize << seed.shape.query_bits) {
            return Err(WhirSpecFoldError::QuerySampleOutOfRange {
                query_sample,
                query_bits: seed.shape.query_bits,
            });
        }
        Ok(Self {
            seed,
            query_idx,
            w_qbase,
            query_sample_raw,
            query_sample,
            controls,
            pair_source: WhirQueryPairSource::Siblings(siblings),
            leaf_sums_by_log_height,
        })
    }

    pub fn query_fold_rows(&self) -> Result<Vec<RecursionWhirQueryFoldRow>, WhirSpecFoldError> {
        query_fold_rows_from_parts(
            &self.seed,
            self.query_idx,
            self.w_qbase,
            self.query_sample_raw,
            self.query_sample,
            &self.controls,
            &self.pair_source,
            &self.leaf_sums_by_log_height,
        )
    }
}

pub(crate) fn query_fold_rows_from_sibling_values<I>(
    seed: &WhirSpecFoldSeed,
    query_idx: usize,
    w_qbase: usize,
    query_sample_raw: F,
    query_sample: usize,
    controls: &[WhirQueryRoundControl],
    siblings: I,
    leaf_sums_by_log_height: &BTreeMap<usize, [F; D_EF]>,
) -> Result<Vec<RecursionWhirQueryFoldRow>, WhirSpecFoldError>
where
    I: ExactSizeIterator<Item = [F; D_EF]>,
{
    let pair_source = WhirQueryPairSource::Siblings(siblings.collect());
    query_fold_rows_from_parts(
        seed,
        query_idx,
        w_qbase,
        query_sample_raw,
        query_sample,
        controls,
        &pair_source,
        leaf_sums_by_log_height,
    )
}

/// Validate the transcript sample and return only its high-band range authority.
///
/// The compact GPU QueryFold path needs these two tiny range candidates on the host, but performs
/// all leaf-sum selection and fold arithmetic on the device.
pub(crate) fn compact_query_sample_band(
    shape: WhirSpecFoldShape,
    query_sample_raw: F,
    query_sample: usize,
) -> Result<(usize, usize, usize), WhirSpecFoldError> {
    if query_sample >= (1usize << shape.query_bits) {
        return Err(WhirSpecFoldError::QuerySampleOutOfRange {
            query_sample,
            query_bits: shape.query_bits,
        });
    }
    let sample_band = sample_band_for_query_bits(shape.query_bits)
        .ok_or(WhirSpecFoldError::UnsupportedQueryBits { query_bits: shape.query_bits })?;
    let (high, _) = query_sample_parts(query_sample_raw, query_sample, sample_band)?;
    Ok((high, sample_band.high_max, sample_band.high_bits))
}

#[allow(clippy::too_many_arguments)]
fn query_fold_rows_from_parts(
    seed: &WhirSpecFoldSeed,
    query_idx: usize,
    w_qbase: usize,
    query_sample_raw: F,
    query_sample: usize,
    controls: &[WhirQueryRoundControl],
    pair_source: &WhirQueryPairSource,
    leaf_sums_by_log_height: &BTreeMap<usize, [F; D_EF]>,
) -> Result<Vec<RecursionWhirQueryFoldRow>, WhirSpecFoldError> {
    let shape = seed.shape;
    let rounds = shape.num_rounds;
    let sample_band = sample_band_for_query_bits(shape.query_bits)
        .ok_or(WhirSpecFoldError::UnsupportedQueryBits { query_bits: shape.query_bits })?;
    if controls.len() != rounds {
        return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
            field: "controls",
            expected: rounds,
            actual: controls.len(),
        });
    }
    let (pair_field, pair_count) = match pair_source {
        WhirQueryPairSource::Explicit(pairs) => ("pairs", pairs.len()),
        WhirQueryPairSource::Siblings(siblings) => ("siblings", siblings.len()),
    };
    if pair_count != rounds {
        return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
            field: pair_field,
            expected: rounds,
            actual: pair_count,
        });
    }
    if query_sample >= (1usize << shape.query_bits) {
        return Err(WhirSpecFoldError::QuerySampleOutOfRange {
            query_sample,
            query_bits: shape.query_bits,
        });
    }

    let (query_sample_high, query_sample_high_gap_inv) =
        query_sample_parts(query_sample_raw, query_sample, sample_band)?;
    let (twiddle_bytes, twiddle_values, twiddle_product_01, x0) =
        query_twiddle_seed(query_sample, shape.query_bits - 1);

    let inv2 = F::from_canonical_usize(2).inverse();
    let mut denominator_inv =
        if rounds == 0 { F::zero() } else { initial_query_fold_denominator_inverse(x0)? };
    let mut idx = query_sample;
    let mut x = x0;
    let mut acc = F::zero();
    let mut ipw = inv2;
    let mut folded = [F::zero(); D_EF];
    let mut round_rows = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let control = controls[round];
        let idx_bit = idx & 1 == 1;
        let merge_log_height = shape.query_bits - round;
        let leaf_sum = if control.is_merge {
            leaf_sums_by_log_height
                .get(&merge_log_height)
                .copied()
                .ok_or(WhirSpecFoldError::MissingQueryLeafSum { log_height: merge_log_height })?
        } else {
            [F::zero(); D_EF]
        };
        let (f0, f1) = match pair_source {
            WhirQueryPairSource::Explicit(pairs) => pairs[round],
            WhirQueryPairSource::Siblings(siblings) => {
                let selected = if control.is_assign {
                    limbs_to_ext(leaf_sum)
                } else if control.is_merge {
                    limbs_to_ext(control.merge_eq) * limbs_to_ext(folded) +
                        limbs_to_ext(control.merge_beta) * limbs_to_ext(leaf_sum)
                } else {
                    limbs_to_ext(folded)
                };
                let selected = ext_limbs(&selected);
                if idx_bit {
                    (siblings[round], selected)
                } else {
                    (selected, siblings[round])
                }
            }
        };
        // Keep both openings and the merge inputs in the row; the AIR,
        // rather than tracegen, enforces their relationship.

        let x_ext = EF::from_base(x);
        let denom = EF::from_base(denominator_inv);
        let f0_ext = limbs_to_ext(f0);
        let f1_ext = limbs_to_ext(f1);
        let r_fold = limbs_to_ext(control.r_fold);
        let folded_out = (x_ext * (f0_ext + f1_ext) + r_fold * (f0_ext - f1_ext)) * denom;
        let folded_out_limbs = ext_limbs(&folded_out);

        let next_idx = idx >> 1;
        let next_idx_bit = next_idx & 1 == 1;
        let x_sq = x * x;
        let sign = if next_idx_bit { F::zero() - F::one() } else { F::one() };
        let next_x = sign * x_sq;
        let next_denominator_inv = next_query_fold_denominator_inverse(denominator_inv, sign);
        let next_acc = if next_idx_bit { acc + ipw } else { acc };
        let next_ipw = ipw * inv2;
        round_rows.push(RecursionWhirQueryFoldRow {
            proof_idx: seed.proof_idx,
            is_round: true,
            query_idx,
            cursor: round,
            query_bits: shape.query_bits,
            r_rounds: rounds,
            idx: F::from_canonical_usize(idx),
            idx_bit,
            x,
            acc,
            ipw,
            folded,
            f0,
            f1,
            chain_send_cursor: round + 1,
            chain_send_idx: F::from_canonical_usize(next_idx),
            chain_send_idx_bit: next_idx_bit,
            chain_send_x: next_x,
            chain_send_acc: next_acc,
            chain_send_ipw: next_ipw,
            chain_send_folded: folded_out_limbs,
            r_fold: control.r_fold,
            is_merge: control.is_merge,
            is_assign: control.is_assign,
            merge_beta: control.merge_beta,
            merge_eq: control.merge_eq,
            emit_prep_seed: control.emit_prep_seed,
            cfr: control.cfr,
            leaf_sum,
            ..Default::default()
        });

        idx = next_idx;
        x = next_x;
        acc = next_acc;
        ipw = next_ipw;
        folded = folded_out_limbs;
        denominator_inv = next_denominator_inv;
    }

    let cfr = controls.last().map(|control| control.cfr).unwrap_or([F::zero(); D_EF]);
    // The cycle-closing row binds `folded` to `cfr` in AIR.

    let seed_row = RecursionWhirQueryFoldRow {
        proof_idx: seed.proof_idx,
        is_seed: true,
        query_idx,
        cursor: rounds,
        w_qbase,
        query_bits: shape.query_bits,
        r_rounds: rounds,
        query_sample: F::from_canonical_usize(query_sample),
        query_sample_raw,
        query_sample_high,
        query_sample_shift: sample_band.shift,
        query_sample_high_max: sample_band.high_max,
        query_sample_high_bits: sample_band.high_bits,
        query_sample_high_gap_inv,
        idx: F::from_canonical_usize(idx),
        idx_bit: idx & 1 == 1,
        idx_tail_bit0: (idx >> 1) & 1 == 1,
        idx_tail_bit1: (idx >> 2) & 1 == 1,
        x,
        acc,
        ipw,
        folded,
        chain_send_cursor: 0,
        chain_send_idx: F::from_canonical_usize(query_sample),
        chain_send_idx_bit: query_sample & 1 == 1,
        chain_send_x: x0,
        chain_send_acc: F::zero(),
        chain_send_ipw: inv2,
        chain_send_folded: [F::zero(); D_EF],
        cfr,
        twiddle_bytes,
        twiddle_values,
        twiddle_product_01,
        ..Default::default()
    };

    let mut rows = Vec::with_capacity(rounds + 1);
    rows.push(seed_row);
    rows.extend(round_rows);
    Ok(rows)
}

#[cfg(test)]
fn query_pairs_from_siblings_oracle(
    shape: WhirSpecFoldShape,
    query_sample: usize,
    controls: &[WhirQueryRoundControl],
    siblings: &[[F; D_EF]],
    leaf_sums_by_log_height: &BTreeMap<usize, [F; D_EF]>,
) -> Result<Vec<([F; D_EF], [F; D_EF])>, WhirSpecFoldError> {
    let rounds = shape.num_rounds;
    if controls.len() != rounds {
        return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
            field: "controls",
            expected: rounds,
            actual: controls.len(),
        });
    }
    if siblings.len() != rounds {
        return Err(WhirSpecFoldError::QueryReplayRoundCountMismatch {
            field: "siblings",
            expected: rounds,
            actual: siblings.len(),
        });
    }
    if query_sample >= (1usize << shape.query_bits) {
        return Err(WhirSpecFoldError::QuerySampleOutOfRange {
            query_sample,
            query_bits: shape.query_bits,
        });
    }

    let (_, _, _, x0) = query_twiddle_seed(query_sample, shape.query_bits - 1);
    let mut idx = query_sample;
    let mut x = x0;
    let mut folded = [F::zero(); D_EF];
    let mut pairs = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let control = controls[round];
        let merge_log_height = shape.query_bits - round;
        let leaf_sum = if control.is_merge {
            leaf_sums_by_log_height
                .get(&merge_log_height)
                .copied()
                .ok_or(WhirSpecFoldError::MissingQueryLeafSum { log_height: merge_log_height })?
        } else {
            [F::zero(); D_EF]
        };
        let selected = if control.is_assign {
            limbs_to_ext(leaf_sum)
        } else if control.is_merge {
            limbs_to_ext(control.merge_eq) * limbs_to_ext(folded) +
                limbs_to_ext(control.merge_beta) * limbs_to_ext(leaf_sum)
        } else {
            limbs_to_ext(folded)
        };
        let selected_limbs = ext_limbs(&selected);
        let idx_bit = idx & 1 == 1;
        let pair = if idx_bit {
            (siblings[round], selected_limbs)
        } else {
            (selected_limbs, siblings[round])
        };
        pairs.push(pair);

        let x_ext = EF::from_base(x);
        let denom = (EF::from_base(F::two()) * x_ext)
            .try_inverse()
            .ok_or(WhirSpecFoldError::ZeroQueryTwiddle { round })?;
        let f0_ext = limbs_to_ext(pair.0);
        let f1_ext = limbs_to_ext(pair.1);
        let r_fold = limbs_to_ext(control.r_fold);
        let folded_out = (x_ext * (f0_ext + f1_ext) + r_fold * (f0_ext - f1_ext)) * denom;

        let next_idx = idx >> 1;
        let next_idx_bit = next_idx & 1 == 1;
        let x_sq = x * x;
        idx = next_idx;
        x = if next_idx_bit { F::zero() - x_sq } else { x_sq };
        folded = ext_limbs(&folded_out);
    }

    // The final folded value remains a recursive constraint, not a host gate.
    Ok(pairs)
}

fn query_sample_parts(
    query_sample_raw: F,
    query_sample: usize,
    sample_band: WhirSampleBandConfig,
) -> Result<(usize, F), WhirSpecFoldError> {
    let raw = query_sample_raw.as_canonical_u32() as usize;
    let raw_low = raw & (sample_band.shift - 1);
    if raw_low != query_sample {
        return Err(WhirSpecFoldError::QuerySampleLowMismatch {
            expected: raw_low,
            actual: query_sample,
        });
    }
    let high = raw >> sample_band.query_bits;
    if high > sample_band.high_max || (high == sample_band.high_max && query_sample != 0) {
        return Err(WhirSpecFoldError::QuerySampleOutOfRange {
            query_sample,
            query_bits: sample_band.query_bits,
        });
    }
    let high_gap = sample_band.high_max - high;
    let high_gap_inv =
        if high_gap == 0 { F::zero() } else { F::from_canonical_usize(high_gap).inverse() };
    Ok((high, high_gap_inv))
}

fn initial_query_fold_denominator_inverse(x: F) -> Result<F, WhirSpecFoldError> {
    (F::two() * x).try_inverse().ok_or(WhirSpecFoldError::ZeroQueryTwiddle { round: 0 })
}

fn next_query_fold_denominator_inverse(current: F, sign: F) -> F {
    F::two() * sign * current * current
}

fn whir_pow_sample_high(
    phase: &'static str,
    sample: F,
    bits: usize,
    high_max: usize,
) -> Result<usize, WhirSpecFoldError> {
    let raw = sample.as_canonical_u32() as usize;
    // Low-bit zero is a proof-of-work constraint represented in the row.
    let high = raw >> bits;
    if high > high_max {
        return Err(WhirSpecFoldError::PowSampleOutOfRange { phase, high, max: high_max });
    }
    Ok(high)
}

fn query_twiddle_seed(
    query_sample: usize,
    bits: usize,
) -> ([u8; WHIR_TWIDDLE_TABLES], [F; WHIR_TWIDDLE_TABLES], F, F) {
    let pair_index = query_sample >> 1;
    let exponent = reverse_bits_len(pair_index, bits) << (23 - bits);
    let twiddle_bytes = core::array::from_fn(|idx| ((exponent >> (8 * idx)) & 0xff) as u8);
    let twiddle_values = core::array::from_fn(|table_id| {
        twiddle_value(table_id, usize::from(twiddle_bytes[table_id]))
    });
    let twiddle_product_01 = twiddle_values[0] * twiddle_values[1];
    let seed = twiddle_product_01 * twiddle_values[2];
    (twiddle_bytes, twiddle_values, twiddle_product_01, seed)
}

fn reverse_bits_len(value: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for idx in 0..bits {
        reversed <<= 1;
        reversed |= (value >> idx) & 1;
    }
    reversed
}

fn base_round_row(
    proof_idx: usize,
    shape: WhirSpecFoldShape,
    w_qbase: usize,
    summary_id_base: usize,
) -> RecursionWhirRoundRow {
    RecursionWhirRoundRow {
        proof_idx,
        role_id: shape.role_id,
        num_queries: shape.num_queries,
        batching_bits: shape.batching_bits,
        query_bits: shape.query_bits,
        log_blowup: shape.log_blowup,
        r_rounds: shape.num_rounds,
        c_chips: shape.c_chips,
        num_public_values: shape.num_public_values,
        w_qbase,
        summary_id_base,
        ..Default::default()
    }
}

fn round_event_stride(
    round: usize,
    rounds: usize,
    group_by_height: &BTreeMap<usize, (usize, [F; D_EF])>,
) -> usize {
    let merge_height = rounds - round - 1;
    20 + usize::from(round > 0) * 8 + usize::from(group_by_height.contains_key(&merge_height)) * 5
}

fn one_ext_limbs() -> [F; D_EF] {
    core::array::from_fn(|idx| if idx == 0 { F::one() } else { F::zero() })
}

fn final_root_seed_state(combined_f_r: [F; D_EF]) -> [F; POSEIDON2_WIDTH] {
    let mut state = [F::zero(); POSEIDON2_WIDTH];
    for (idx, lane) in state.iter_mut().enumerate().take(WHIR_FINAL_ROOT_DIGEST_LANES) {
        *lane = final_codeword_limb(combined_f_r, idx);
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
            next[0] = final_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES);
            next[1] = final_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES + 1);
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

impl WhirBatchRlc {
    pub fn batch_eval_rows(
        &self,
        proof_idx: usize,
        alpha_tidx: usize,
        role_id: usize,
        num_queries: usize,
        batching_bits: usize,
        log_blowup: usize,
        static_chip_ids: &[usize],
        publish_opened_eval: bool,
    ) -> Vec<RecursionWhirBatchEvalRow> {
        self.batch_eval_rows_with_authority(
            proof_idx,
            alpha_tidx,
            role_id,
            num_queries,
            batching_bits,
            log_blowup,
            static_chip_ids,
            move |_, _, _, _| u32::from(publish_opened_eval),
            |_| 0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_eval_rows_with_authority<M, P>(
        &self,
        proof_idx: usize,
        alpha_tidx: usize,
        role_id: usize,
        num_queries: usize,
        batching_bits: usize,
        log_blowup: usize,
        static_chip_ids: &[usize],
        opened_eval_mult: M,
        pow_seed_count: P,
    ) -> Vec<RecursionWhirBatchEvalRow>
    where
        M: Fn(usize, usize, usize, usize) -> u32,
        P: Fn(usize) -> u32,
    {
        let mut rows = Vec::new();

        if let Some(last_step) = self.steps.last() {
            let first_step = self.steps.first().expect("last step exists");
            let last_count = segment_element_count(last_step.batch_id, last_step.width);
            rows.push(RecursionWhirBatchEvalRow {
                proof_idx,
                is_start: true,
                role_id,
                role_num_queries: num_queries,
                role_batching_bits: batching_bits,
                role_log_blowup: log_blowup,
                cursor: 0,
                chain_recv_cursor: self.steps.len(),
                chain_recv_log_height: last_step.log_height,
                chain_recv_batch_id: last_step.batch_id,
                chain_recv_batch_pos: last_step.batch_pos,
                chain_recv_value_idx: last_step.value_idx,
                chain_recv_segment_element_count: last_count,
                chain_send_cursor: 0,
                alpha_tidx,
                alpha: self.alpha,
                pow_in: last_step.alpha_power_out,
                acc_in: last_step.acc_out,
                group_base_in: last_step.group_base_out,
                pow_out: ext_limbs(&EF::one()),
                acc_out: ext_limbs(&EF::zero()),
                group_base_out: ext_limbs(&EF::zero()),
                log_height: first_step.log_height + 1,
                segment_element_count: 0,
                role_config_recv_mult: 1,
                chain_recv_mult: 1,
                chain_send_mult: 1,
                ..Default::default()
            });
        }

        let mut next_step = 0;
        let mut prev_chain_log_height = self.steps.first().map_or(0, |step| step.log_height + 1);
        let mut prev_chain_batch_id = 0usize;
        let mut prev_chain_batch_pos = 0usize;
        let mut prev_chain_value_idx = 0usize;
        let mut prev_chain_count = 0usize;
        let mut prev_group_log_height = self.steps.first().map_or(0, |step| step.log_height + 1);
        for segment in &self.segments {
            let segment_steps = &self.steps[next_step..next_step + segment.element_count];
            if segment_steps.is_empty() {
                rows.push(batch_dim_only_row(
                    proof_idx,
                    alpha_tidx,
                    self.alpha,
                    segment,
                    static_chip_ids,
                ));
                continue;
            }

            let segment_count = segment_element_count(segment.batch_id, segment.width);
            let is_group_start = segment.log_height != prev_group_log_height;
            let group_log_height_gap = if is_group_start {
                prev_group_log_height
                    .checked_sub(segment.log_height + 1)
                    .expect("batch RLC group heights must be strictly descending")
            } else {
                0
            };
            for (local_idx, step) in segment_steps.iter().enumerate() {
                let is_segment_end = local_idx + 1 == segment_steps.len();
                let row_is_group_start = is_group_start && local_idx == 0;
                rows.push(batch_value_row(
                    proof_idx,
                    alpha_tidx,
                    log_blowup,
                    self.alpha,
                    step,
                    static_chip_ids,
                    opened_eval_mult(step.batch_id, step.batch_pos, step.chip_idx, step.value_idx),
                    segment_count,
                    is_segment_end,
                    row_is_group_start,
                    if row_is_group_start { group_log_height_gap } else { 0 },
                    if row_is_group_start {
                        pow_seed_count(step.log_height + log_blowup)
                    } else {
                        0
                    },
                    ChainSegmentState {
                        log_height: prev_chain_log_height,
                        batch_id: prev_chain_batch_id,
                        batch_pos: prev_chain_batch_pos,
                        value_idx: prev_chain_value_idx,
                        segment_element_count: prev_chain_count,
                    },
                ));
                prev_chain_log_height = step.log_height;
                prev_chain_batch_id = step.batch_id;
                prev_chain_batch_pos = step.batch_pos;
                prev_chain_value_idx = step.value_idx;
                prev_chain_count = segment_count;
            }
            prev_group_log_height = segment.log_height;
            next_step += segment.element_count;
        }
        debug_assert_eq!(next_step, self.steps.len());

        rows
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChainSegmentState {
    log_height: usize,
    batch_id: usize,
    batch_pos: usize,
    value_idx: usize,
    segment_element_count: usize,
}

fn segment_element_count(batch_id: usize, width: usize) -> usize {
    if batch_id == WHIR_BATCH_PERMUTATION {
        width / D_EF
    } else {
        width
    }
}

fn batch_dim_only_row(
    proof_idx: usize,
    alpha_tidx: usize,
    alpha: [F; D_EF],
    segment: &WhirBatchRlcSegment,
    static_chip_ids: &[usize],
) -> RecursionWhirBatchEvalRow {
    RecursionWhirBatchEvalRow {
        proof_idx,
        alpha_tidx,
        alpha,
        log_height: segment.log_height,
        batch_id: segment.batch_id,
        batch_pos: segment.batch_pos,
        chip_idx: segment.chip_idx,
        static_chip_id: static_chip_id(static_chip_ids, segment.chip_idx),
        width: segment.width,
        segment_element_count: segment_element_count(segment.batch_id, segment.width),
        is_perm_batch: segment.batch_id == WHIR_BATCH_PERMUTATION,
        batch_dim_recv_mult: 1,
        ..Default::default()
    }
}

fn batch_value_row(
    proof_idx: usize,
    alpha_tidx: usize,
    log_blowup: usize,
    alpha: [F; D_EF],
    step: &WhirBatchRlcStep,
    static_chip_ids: &[usize],
    opened_eval_send_mult: u32,
    segment_element_count: usize,
    is_segment_end: bool,
    is_group_start: bool,
    group_log_height_gap: usize,
    pow_seed_cnt: u32,
    chain_recv: ChainSegmentState,
) -> RecursionWhirBatchEvalRow {
    RecursionWhirBatchEvalRow {
        proof_idx,
        // Record-side mirror of the baked role blowup so residual keys are
        // self-contained on every row.
        role_log_blowup: log_blowup,
        is_group_end: step.is_group_end,
        cursor: step.cursor,
        chain_recv_cursor: step.cursor,
        chain_send_cursor: step.cursor + 1,
        chain_recv_log_height: chain_recv.log_height,
        chain_recv_batch_id: chain_recv.batch_id,
        chain_recv_batch_pos: chain_recv.batch_pos,
        chain_recv_value_idx: chain_recv.value_idx,
        chain_recv_segment_element_count: chain_recv.segment_element_count,
        alpha_tidx,
        alpha,
        pow_in: step.alpha_power,
        acc_in: step.acc_in,
        group_base_in: step.group_base_in,
        pow_out: step.alpha_power_out,
        acc_out: step.acc_out,
        group_base_out: step.group_base_out,
        value: step.value,
        log_height: step.log_height,
        batch_id: step.batch_id,
        batch_pos: step.batch_pos,
        chip_idx: step.chip_idx,
        static_chip_id: static_chip_id(static_chip_ids, step.chip_idx),
        width: step.width,
        value_idx: step.value_idx,
        segment_element_count,
        is_value: true,
        is_segment_start: step.is_segment_start,
        is_segment_end,
        is_first_value: step.cursor == 0,
        is_group_start,
        is_perm_batch: step.batch_id == WHIR_BATCH_PERMUTATION,
        group_log_height_gap,
        pow_seed_cnt,
        batch_dim_recv_mult: u32::from(step.is_segment_start),
        group_claim_send_mult: u32::from(step.is_group_end),
        opened_eval_send_mult,
        chain_recv_mult: 1,
        chain_send_mult: 1,
        ..Default::default()
    }
}

fn static_chip_id(static_chip_ids: &[usize], chip_idx: usize) -> usize {
    static_chip_ids
        .get(chip_idx)
        .copied()
        .expect("WHIR opened matrix chip_idx must have a proof-shape static chip id")
}

impl WhirFinalRootSponge {
    pub fn from_combined_f_r(
        combined_f_r: [F; D_EF],
        log_blowup: usize,
        poseidon2_output: &impl RecursionPoseidon2Output,
    ) -> Self {
        debug_assert_eq!(D_EF, 5);
        debug_assert_eq!(POSEIDON2_WIDTH, 16);

        assert!(
            (1..=3).contains(&log_blowup),
            "unsupported WHIR final-root log_blowup {log_blowup}"
        );

        let mut inputs = [[F::zero(); POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS];
        let mut outputs = [[F::zero(); POSEIDON2_WIDTH]; WHIR_FINAL_ROOT_POSEIDON2_PERMS];

        inputs[0][..WHIR_FINAL_ROOT_DIGEST_LANES].copy_from_slice(&core::array::from_fn::<
            F,
            WHIR_FINAL_ROOT_DIGEST_LANES,
            _,
        >(|idx| {
            final_codeword_limb(combined_f_r, idx)
        }));
        outputs[0] = poseidon2_output.permute_output(inputs[0]);

        inputs[1] = outputs[0];
        inputs[1][0] = final_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES);
        inputs[1][1] = final_codeword_limb(combined_f_r, WHIR_FINAL_ROOT_DIGEST_LANES + 1);
        outputs[1] = poseidon2_output.permute_output(inputs[1]);

        let mut root_perm = 1;
        let leaf_digest: [F; WHIR_FINAL_ROOT_DIGEST_LANES] =
            core::array::from_fn(|idx| outputs[1][idx]);
        if log_blowup >= 2 {
            inputs[2][..WHIR_FINAL_ROOT_DIGEST_LANES].copy_from_slice(&leaf_digest);
            inputs[2][WHIR_FINAL_ROOT_DIGEST_LANES..].copy_from_slice(&leaf_digest);
            outputs[2] = poseidon2_output.permute_output(inputs[2]);
            root_perm = 2;
        }
        if log_blowup >= 3 {
            let level1_digest: [F; WHIR_FINAL_ROOT_DIGEST_LANES] =
                core::array::from_fn(|idx| outputs[2][idx]);
            inputs[3][..WHIR_FINAL_ROOT_DIGEST_LANES].copy_from_slice(&level1_digest);
            inputs[3][WHIR_FINAL_ROOT_DIGEST_LANES..].copy_from_slice(&level1_digest);
            outputs[3] = poseidon2_output.permute_output(inputs[3]);
            root_perm = 3;
        }

        let num_perms = root_perm + 1;
        let digest = core::array::from_fn(|idx| outputs[root_perm][idx]);
        Self { inputs, outputs, num_perms, digest }
    }

    pub fn round_fields(&self) -> WhirFinalRootRowFields {
        WhirFinalRootRowFields {
            inputs: self.inputs,
            outputs: self.outputs,
            recv_mults: match self.num_perms {
                2 => [1, 1, 0, 0],
                3 => [2, 2, 1, 0],
                4 => [4, 4, 2, 1],
                _ => panic!("unsupported WHIR final-root permutation count {}", self.num_perms),
            },
            digest: self.digest,
        }
    }
}

fn final_codeword_limb(combined_f_r: [F; D_EF], idx: usize) -> F {
    combined_f_r[idx % D_EF]
}

fn opened_matrix(
    batch_id: usize,
    batch_pos: usize,
    chip_idx: usize,
    dim: Dimensions,
    values: &[EF],
    flattened_ext_width: bool,
) -> Result<WhirOpenedMatrix, WhirSpecFoldError> {
    if dim.height == 0 || !dim.height.is_power_of_two() {
        return Err(WhirSpecFoldError::NonPowerOfTwoHeight {
            batch_id,
            batch_pos,
            height: dim.height,
        });
    }
    let expected_width = if flattened_ext_width { values.len() * D_EF } else { values.len() };
    if dim.width != expected_width {
        return Err(WhirSpecFoldError::OpenedWidthMismatch {
            batch_id,
            batch_pos,
            chip_idx,
            expected_width: dim.width,
            actual_values: values.len(),
        });
    }
    Ok(WhirOpenedMatrix {
        batch_id,
        batch_pos,
        chip_idx,
        width: dim.width,
        log_height: dim.height.trailing_zeros() as usize,
        values: values.iter().map(ext_limbs).collect(),
    })
}

pub(crate) fn ext_limbs(value: &EF) -> [F; D_EF] {
    let limbs = value.as_base_slice();
    debug_assert_eq!(limbs.len(), D_EF);
    core::array::from_fn(|idx| limbs[idx])
}

pub(crate) fn limbs_to_ext(limbs: [F; D_EF]) -> EF {
    EF::from_base_slice(&limbs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::EF, system_dt::record::RecursionSumcheckRoundRecord};
    use dt_stark::{
        koalabear_poseidon2::koala_bear_poseidon2::{
            my_perm, ChallengeMmcs, MyCompress, MyHash, ValMmcs,
        },
        sumcheck::proof::{SCChipOpenedValues, SCShardOpenedValues},
        SCAirOpenedValues,
    };
    use p3_commit::Mmcs;
    use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn opening_point_uses_reversed_batch_round_challenges_not_eq_challenges() {
        let num_rounds = 3;
        let mut batch = RecursionBatchConstraintRecord {
            num_rounds,
            eq_challenges: (0..num_rounds)
                .map(|idx| [F::from_canonical_usize(100 + idx); D_EF])
                .collect(),
            rounds: (0..num_rounds)
                .map(|round_idx| RecursionSumcheckRoundRecord {
                    round_idx,
                    challenge: [F::from_canonical_usize(10 + round_idx); D_EF],
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        batch.c_chips = 1;

        let seed = WhirSpecFoldSeed::from_batch(
            0,
            WhirSpecFoldShape {
                role_id: 0,
                num_rounds,
                c_chips: 1,
                num_public_values: 0,
                num_queries: 261,
                batching_bits: 10,
                query_bits: 4,
                log_blowup: 1,
                w0_tidx: 0,
            },
            &batch,
        )
        .expect("shape matches");

        assert_eq!(seed.opening_point[0], [F::from_canonical_usize(12); D_EF]);
        assert_eq!(seed.opening_point[1], [F::from_canonical_usize(11); D_EF]);
        assert_eq!(seed.opening_point[2], [F::from_canonical_usize(10); D_EF]);
        assert_ne!(seed.opening_point[0], batch.eq_challenges[0]);
    }

    #[test]
    fn final_root_sponge_matches_extension_mmcs_commitment() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let combined_f_r = core::array::from_fn(|idx| F::from_canonical_usize(31 + idx * 7));
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));

        for (log_blowup, expected_perms, expected_mults) in
            [(1, 2, [1, 1, 0, 0]), (2, 3, [2, 2, 1, 0]), (3, 4, [4, 4, 2, 1])]
        {
            let sponge =
                WhirFinalRootSponge::from_combined_f_r(combined_f_r, log_blowup, &poseidon2_memo);
            let row_fields = sponge.round_fields();
            let codeword_len = 1usize << log_blowup;
            let codeword = vec![EF::from_base_slice(&combined_f_r); codeword_len];
            let (commitment, _) = mmcs.commit_matrix(RowMajorMatrix::new(codeword, 2));

            assert_eq!(sponge.num_perms, expected_perms);
            assert_eq!(row_fields.recv_mults, expected_mults);
            assert_eq!(row_fields.inputs, sponge.inputs);
            assert_eq!(row_fields.outputs, sponge.outputs);
            assert_eq!(commitment, sponge.digest);
        }
    }

    #[test]
    fn opened_matrices_follow_whir_batch_order_and_width_rules() {
        let ext = |base: usize| {
            EF::from_base_slice(&core::array::from_fn::<F, D_EF, _>(|idx| {
                F::from_canonical_usize(base + idx)
            }))
        };
        let chip = |preprocessed: Vec<EF>, main: Vec<EF>, permutation: Vec<EF>, log_height| {
            SCChipOpenedValues {
                preprocessed: SCAirOpenedValues { local: preprocessed },
                main: SCAirOpenedValues { local: main },
                permutation: SCAirOpenedValues { local: permutation },
                local_cumulative_sum: EF::zero(),
                log_height,
                _field: core::marker::PhantomData,
            }
        };
        let opened = SCShardOpenedValues {
            chips: vec![
                chip(vec![ext(10)], vec![ext(20), ext(30)], vec![ext(40)], 4),
                chip(vec![], vec![ext(50)], vec![], 3),
            ],
            _field: core::marker::PhantomData,
        };
        let dimensions = vec![
            vec![Dimensions { width: 1, height: 16 }],
            vec![Dimensions { width: 2, height: 16 }, Dimensions { width: 1, height: 8 }],
            vec![Dimensions { width: D_EF, height: 16 }, Dimensions { width: 0, height: 8 }],
        ];

        let matrices = WhirOpenedMatrices::from_child_openings(&dimensions, &opened)
            .expect("fixture dimensions match opened values")
            .matrices;

        assert_eq!(
            matrices
                .iter()
                .map(|m| (m.batch_id, m.batch_pos, m.chip_idx, m.width))
                .collect::<Vec<_>>(),
            vec![
                (WHIR_BATCH_PREPROCESSED, 0, 0, 1),
                (WHIR_BATCH_MAIN, 0, 0, 2),
                (WHIR_BATCH_MAIN, 1, 1, 1),
                (WHIR_BATCH_PERMUTATION, 0, 0, D_EF),
                (WHIR_BATCH_PERMUTATION, 1, 1, 0),
            ]
        );
        assert_eq!(matrices[0].values[0][0], F::from_canonical_usize(10));
        assert_eq!(matrices[3].values[0][4], F::from_canonical_usize(44));
        assert!(matrices[4].values.is_empty());
    }

    #[test]
    fn prep_first_height_must_be_prep_max_for_record_path() {
        let opened = WhirOpenedMatrices {
            matrices: vec![
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PREPROCESSED,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: 1,
                    log_height: 3,
                    values: vec![ext_limbs(&EF::one())],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PREPROCESSED,
                    batch_pos: 1,
                    chip_idx: 1,
                    width: 1,
                    log_height: 5,
                    values: vec![ext_limbs(&EF::one())],
                },
            ],
        };

        assert_eq!(
            opened.assert_prep_first_height_is_max(),
            Err(WhirSpecFoldError::PrepFirstHeightNotMax {
                first_log_height: 3,
                max_log_height: 5,
            })
        );
    }

    #[test]
    fn batch_rlc_replays_height_order_and_alpha_schedule() {
        let ext = |base: usize| {
            EF::from_base_slice(&core::array::from_fn::<F, D_EF, _>(|idx| {
                F::from_canonical_usize(base + idx)
            }))
        };
        let opened = WhirOpenedMatrices {
            matrices: vec![
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PREPROCESSED,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: 1,
                    log_height: 4,
                    values: vec![ext_limbs(&ext(10))],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_MAIN,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: 2,
                    log_height: 4,
                    values: vec![ext_limbs(&ext(20)), ext_limbs(&ext(30))],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_MAIN,
                    batch_pos: 1,
                    chip_idx: 1,
                    width: 1,
                    log_height: 3,
                    values: vec![ext_limbs(&ext(50))],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PERMUTATION,
                    batch_pos: 0,
                    chip_idx: 0,
                    width: D_EF,
                    log_height: 4,
                    values: vec![ext_limbs(&ext(40))],
                },
                WhirOpenedMatrix {
                    batch_id: WHIR_BATCH_PERMUTATION,
                    batch_pos: 1,
                    chip_idx: 1,
                    width: 0,
                    log_height: 3,
                    values: vec![],
                },
            ],
        };
        let alpha = ext_limbs(&EF::from_canonical_usize(3));
        let rlc = WhirBatchRlc::from_opened_matrices(&opened, alpha);
        let alpha_ext = limbs_to_ext(alpha);

        assert_eq!(rlc.groups.iter().map(|group| group.log_height).collect::<Vec<_>>(), vec![4, 3]);
        assert_eq!(
            rlc.steps
                .iter()
                .map(|step| (step.log_height, step.batch_id, step.batch_pos, step.value_idx))
                .collect::<Vec<_>>(),
            vec![
                (4, WHIR_BATCH_PREPROCESSED, 0, 0),
                (4, WHIR_BATCH_MAIN, 0, 0),
                (4, WHIR_BATCH_MAIN, 0, 1),
                (4, WHIR_BATCH_PERMUTATION, 0, 0),
                (3, WHIR_BATCH_MAIN, 1, 0),
            ]
        );
        assert_eq!(
            rlc.segments
                .iter()
                .map(|segment| {
                    (
                        segment.log_height,
                        segment.batch_id,
                        segment.batch_pos,
                        segment.first_cursor,
                        segment.element_count,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (4, WHIR_BATCH_PREPROCESSED, 0, 0, 1),
                (4, WHIR_BATCH_MAIN, 0, 1, 2),
                (4, WHIR_BATCH_PERMUTATION, 0, 3, 1),
                (3, WHIR_BATCH_MAIN, 1, 4, 1),
                (3, WHIR_BATCH_PERMUTATION, 1, 5, 0),
            ]
        );

        let h4_expected = ext(10) +
            alpha_ext * ext(20) +
            alpha_ext * alpha_ext * ext(30) +
            alpha_ext * alpha_ext * alpha_ext * ext(40);
        let h3_expected = alpha_ext * alpha_ext * alpha_ext * alpha_ext * ext(50);

        assert_eq!(rlc.groups[0].claim, ext_limbs(&h4_expected));
        assert_eq!(rlc.groups[1].claim, ext_limbs(&h3_expected));
        assert_eq!(
            rlc.steps[4].alpha_power,
            ext_limbs(&(alpha_ext * alpha_ext * alpha_ext * alpha_ext))
        );
        let h4_prefix = h4_expected;
        assert_eq!(rlc.steps[3].acc_out, ext_limbs(&h4_prefix));
        assert_eq!(rlc.steps[4].acc_in, ext_limbs(&h4_prefix));
        assert_eq!(rlc.steps[4].group_base_in, ext_limbs(&h4_prefix));
        assert_eq!(rlc.steps[4].acc_out, ext_limbs(&(h4_prefix + h3_expected)));
        assert_eq!(
            ext_limbs(
                &(limbs_to_ext(rlc.steps[4].acc_out) - limbs_to_ext(rlc.steps[4].group_base_in))
            ),
            rlc.groups[1].claim
        );
        assert!(rlc.steps[0].is_segment_start);
        assert!(!rlc.steps[2].is_segment_start);
        assert!(rlc.steps[3].is_segment_start);
        assert!(rlc.steps[0].is_group_start);
        assert!(rlc.steps[3].is_group_end);
        assert!(rlc.steps[4].is_group_start && rlc.steps[4].is_group_end);

        let leaf_openings = vec![
            vec![vec![F::from_canonical_usize(100)]],
            vec![
                vec![F::from_canonical_usize(200), F::from_canonical_usize(300)],
                vec![F::from_canonical_usize(500)],
            ],
            vec![(400..400 + D_EF).map(F::from_canonical_usize).collect::<Vec<_>>(), vec![]],
        ];
        let leaf_sums =
            rlc.query_leaf_sums(&leaf_openings, 1).expect("query leaf openings match shape");
        let leaf_h4_expected = EF::from_canonical_usize(100) +
            alpha_ext * EF::from_canonical_usize(200) +
            alpha_ext * alpha_ext * EF::from_canonical_usize(300) +
            alpha_ext * alpha_ext * alpha_ext * ext(400);
        let leaf_h3_expected =
            alpha_ext * alpha_ext * alpha_ext * alpha_ext * EF::from_canonical_usize(500);
        assert_eq!(leaf_sums.get(&5), Some(&ext_limbs(&leaf_h4_expected)));
        assert_eq!(leaf_sums.get(&4), Some(&ext_limbs(&leaf_h3_expected)));

        let mut wrong_width = leaf_openings.clone();
        wrong_width[2][0].pop();
        assert_eq!(
            rlc.query_leaf_sums(&wrong_width, 1),
            Err(WhirSpecFoldError::QueryOpeningWidthMismatch {
                batch_id: WHIR_BATCH_PERMUTATION,
                batch_pos: 0,
                expected_width: D_EF,
                actual_width: D_EF - 1,
            })
        );

        let rows = rlc.batch_eval_rows(2, 99, 0, 261, 10, 1, &[7, 8], false);
        assert_eq!(rows.len(), 7);
        assert!(rows[0].is_start);
        assert_eq!(rows[0].role_id, 0);
        assert_eq!(rows[0].role_num_queries, 261);
        assert_eq!(rows[0].role_batching_bits, 10);
        assert_eq!(rows[0].role_log_blowup, 1);
        assert_eq!(rows[0].role_config_recv_mult, 1);
        assert_eq!(rows[0].chain_recv_cursor, rlc.steps.len());
        assert_eq!(rows[0].chain_send_cursor, 0);
        assert_eq!(rows[0].pow_out, ext_limbs(&EF::one()));
        assert_eq!(rows[0].acc_out, ext_limbs(&EF::zero()));

        assert!(rows[1].is_value);
        assert_eq!(rows[1].static_chip_id, 7);
        assert_eq!(rows[1].batch_dim_recv_mult, 1);
        assert_eq!(rows[1].opened_eval_send_mult, 0);
        assert_eq!(rows[3].batch_dim_recv_mult, 0);
        assert_eq!(rows[4].group_claim_send_mult, 1);
        assert_eq!(
            ext_limbs(&(limbs_to_ext(rows[4].acc_out) - limbs_to_ext(rows[4].group_base_in))),
            rlc.groups[0].claim
        );
        assert_eq!(rows[5].group_claim_send_mult, 1);
        assert_eq!(
            ext_limbs(&(limbs_to_ext(rows[5].acc_out) - limbs_to_ext(rows[5].group_base_in))),
            rlc.groups[1].claim
        );

        let dim_only = rows.last().expect("zero-width segment row is emitted");
        assert!(!dim_only.is_value);
        assert_eq!(dim_only.batch_id, WHIR_BATCH_PERMUTATION);
        assert_eq!(dim_only.batch_pos, 1);
        assert_eq!(dim_only.static_chip_id, 8);
        assert_eq!(dim_only.width, 0);
        assert_eq!(dim_only.batch_dim_recv_mult, 1);
        assert_eq!(dim_only.chain_recv_mult, 0);
        assert_eq!(dim_only.chain_send_mult, 0);
        assert!(
            batch_eval_chain_residual(&rows).is_empty(),
            "B eval-chain rows must internally balance"
        );

        let mut tampered = rows.clone();
        tampered[2].chain_recv_cursor += 1;
        assert!(
            !batch_eval_chain_residual(&tampered).is_empty(),
            "cursor tampering must leave a WhirEvalChain residual"
        );

        // Per-instance builds (one per height group), no cycle row.
        let pows = rlc.group_start_pows(1);
        let (leaf_rows, leaf_ext_rows) = rlc
            .leaf_group_stream_rows(2, 5, 3, &leaf_openings, 1, pows[&5])
            .expect("D rows follow the same query-opening shape");
        assert_eq!(leaf_rows.len(), 2);
        assert_eq!(leaf_ext_rows.len(), 1);

        assert_eq!(leaf_rows[0].cursor, 0);
        assert!(leaf_rows[0].is_unit_start);
        assert!(!leaf_rows[0].is_unit_end);
        assert_eq!(leaf_rows[0].log_height, 5);
        assert_eq!(leaf_rows[0].idx, 3);
        assert_eq!(leaf_rows[0].pow_in, ext_limbs(&pows[&5]));
        assert_eq!(leaf_rows[0].values[0], F::from_canonical_usize(100));
        assert_eq!(
            leaf_rows[0].chunk_mask,
            [true, false, false, false, false, false, false, false]
        );
        assert_eq!(leaf_rows[0].unit_key, whir_unit_key(WHIR_INPUT_PREPROCESSED_PATH_SLOT, 5));

        assert_eq!(leaf_rows[1].cursor, 1);
        assert_eq!(leaf_rows[1].values[0], F::from_canonical_usize(200));
        assert_eq!(leaf_rows[1].values[1], F::from_canonical_usize(300));
        assert_eq!(leaf_rows[1].unit_key, whir_unit_key(WHIR_INPUT_MAIN_PATH_SLOT, 5));

        assert_eq!(leaf_ext_rows[0].cursor, 2);
        assert!(leaf_ext_rows[0].is_unit_end);
        assert_eq!(leaf_ext_rows[0].log_height, 5);
        assert_eq!(leaf_ext_rows[0].idx, 3);
        assert_eq!(leaf_ext_rows[0].value_blocks[0][0], F::from_canonical_usize(400));
        assert_eq!(leaf_ext_rows[0].value_blocks[0][4], F::from_canonical_usize(404));
        assert_eq!(
            leaf_ext_rows[0].chunk_masks[0],
            [true, true, true, true, true, false, false, false]
        );
        assert_eq!(leaf_ext_rows[0].unit_key, whir_unit_key(WHIR_INPUT_PERMUTATION_PATH_SLOT, 5));
        assert!(
            leaf_chain_residual(&leaf_rows, &leaf_ext_rows).is_empty(),
            "D leaf-chain rows must internally balance across D1/D2"
        );

        let (h4_rows, h4_ext_rows) = rlc
            .leaf_group_stream_rows(2, 4, 1, &leaf_openings, 1, pows[&4])
            .expect("second height group builds independently");
        assert_eq!(h4_rows.len(), 1);
        assert!(h4_ext_rows.is_empty());
        assert!(h4_rows[0].is_unit_start && h4_rows[0].is_unit_end);
        assert_eq!(h4_rows[0].log_height, 4);
        assert_eq!(h4_rows[0].idx, 1);
        assert_eq!(h4_rows[0].pow_in, ext_limbs(&pows[&4]));
        assert_eq!(h4_rows[0].values[0], F::from_canonical_usize(500));
        assert_eq!(h4_rows[0].acc_out, ext_limbs(&leaf_h3_expected));

        let mut tampered_ext = leaf_ext_rows.clone();
        tampered_ext[0].chain_recv_cursor += 1;
        assert!(
            !leaf_chain_residual(&leaf_rows, &tampered_ext).is_empty(),
            "D2 cursor tampering must leave a WhirLeafChain residual"
        );
    }

    #[test]
    fn round_replay_builds_a_cycle_with_lagged_merge_and_final_root() {
        let poseidon2_memo = RecursionPoseidon2Memo::default();
        let shape = WhirSpecFoldShape {
            role_id: 0,
            num_rounds: 2,
            c_chips: 1,
            num_public_values: 0,
            num_queries: 7,
            batching_bits: 10,
            query_bits: 3,
            log_blowup: 1,
            w0_tidx: 100,
        };
        let z_round_1 = ext(2);
        let z_round_0 = ext(5);
        let seed = WhirSpecFoldSeed {
            proof_idx: 3,
            shape,
            opening_point: vec![ext_limbs(&z_round_1), ext_limbs(&z_round_0)],
        };
        let tallest_claim = ext(7);
        let lower_claim = ext(5);
        let r0 = ext(3);
        let r1 = ext(6);
        let folded0 = ext(11);
        let beta1 = ext(4);
        let round1_claim = folded0 + beta1 * lower_claim;
        let eq1 = eq_factor(z_round_1, r1);
        let final_cfr = ext(9);
        let final_claim = final_cfr * eq1;
        let coeffs0 = quadratic_coeffs_for_claim_and_fold(tallest_claim, folded0, r0);
        let coeffs1 = quadratic_coeffs_for_claim_and_fold(round1_claim, final_claim, r1);
        let final_digest = WhirFinalRootSponge::from_combined_f_r(
            ext_limbs(&final_cfr),
            shape.log_blowup,
            &poseidon2_memo,
        )
        .digest;

        let input = WhirRoundReplayInput {
            seed,
            summary_id_base: 128,
            group_claims: vec![
                WhirBatchRlcGroup {
                    log_height: 2,
                    claim: ext_limbs(&tallest_claim),
                    first_cursor: 0,
                    element_count: 1,
                },
                WhirBatchRlcGroup {
                    log_height: 1,
                    claim: ext_limbs(&lower_claim),
                    first_cursor: 1,
                    element_count: 1,
                },
            ],
            sumcheck_coeffs: vec![coeffs0, coeffs1],
            r_folds: vec![ext_limbs(&r0), ext_limbs(&r1)],
            merge_betas_by_height: BTreeMap::from([(1, ext_limbs(&beta1))]),
            iopp_oracles: vec![digest(10), digest(20), final_digest],
            batching_pow_events: [
                F::from_canonical_usize(31),
                F::from_canonical_usize(32),
                F::from_canonical_usize(3usize << WHIR_BATCHING_POW_BITS),
            ],
            query_pow_events: [
                F::from_canonical_usize(41),
                F::from_canonical_usize(42),
                F::from_canonical_usize(2usize << WHIR_QUERY_POW_BITS),
            ],
            prep_seed_round: Some(1),
        };

        let mut bad_batch_pow = input.clone();
        bad_batch_pow.batching_pow_events[2] = F::one();
        let bad_batch_rows =
            bad_batch_pow.round_rows(&poseidon2_memo).expect("PoW validity is an AIR constraint");
        assert_eq!(bad_batch_rows[0].event_value[2], F::one());
        assert_eq!(bad_batch_rows[0].pow_sample_high, 0);

        let mut bad_query_pow = input.clone();
        bad_query_pow.query_pow_events[2] = F::one();
        let bad_query_rows =
            bad_query_pow.round_rows(&poseidon2_memo).expect("PoW validity is an AIR constraint");
        let bad_query_final = bad_query_rows.last().expect("final row exists");
        assert_eq!(bad_query_final.event_value[10], F::one());
        assert_eq!(bad_query_final.pow_sample_high, 0);

        let mut bad_final_root = input.clone();
        bad_final_root.iopp_oracles[shape.num_rounds][0] += F::one();
        let bad_root_rows = bad_final_root
            .round_rows(&poseidon2_memo)
            .expect("claimed and computed roots are constrained by AIR");
        let bad_root_final = bad_root_rows.last().expect("final row exists");
        assert_ne!(
            &bad_root_final.event_value[..DIGEST_SIZE],
            &bad_root_final.final_root_poseidon2_output[..DIGEST_SIZE]
        );

        let rows = input.round_rows(&poseidon2_memo).expect("fixture is internally consistent");
        assert!(rows.iter().all(|row| row.summary_id_base == 128));
        assert_eq!(rows.len(), shape.num_rounds + 3 + WHIR_FINAL_ROOT_POSEIDON2_PERMS);
        let final_idx = rows.len() - 1;
        let first_final_perm = final_idx - WHIR_FINAL_ROOT_POSEIDON2_PERMS;
        assert!(rows[0].is_pow_batch);
        assert!(rows[1].is_preamble);
        assert!(rows[2].is_round && rows[2].round == 0);
        assert!(rows[3].is_round && rows[3].round == 1);
        assert!(rows[final_idx].is_final);
        for step in 0..WHIR_FINAL_ROOT_POSEIDON2_PERMS {
            let row = rows[first_final_perm + step];
            assert!(row.is_final_perm);
            assert!(row.final_root_perm_step_flags[step]);
        }
        assert_eq!(rows[0].pow_sample_high, 3);
        assert_eq!(rows[final_idx].pow_sample_high, 2);

        assert!(rows[2].chain_recv_pending_is_merge);
        assert_eq!(rows[2].chain_recv_pending_beta, [F::zero(); D_EF]);
        assert!(rows[2].is_merge);
        assert_eq!(rows[2].chain_send_pending_beta, ext_limbs(&beta1));
        assert!(rows[3].chain_recv_pending_is_merge);
        assert_eq!(rows[3].chain_recv_pending_beta, ext_limbs(&beta1));
        assert!(rows[3].emit_prep_seed);
        assert_eq!(rows[2].cfr, ext_limbs(&final_cfr));
        assert_eq!(rows[3].cfr, ext_limbs(&final_cfr));
        assert_eq!(rows[final_idx].cfr, ext_limbs(&final_cfr));
        assert_eq!(rows[final_idx].event_value[..8], final_digest);
        assert_eq!(rows[first_final_perm].final_root_poseidon2_recv_mult, 1);
        assert_eq!(rows[first_final_perm + 1].final_root_poseidon2_recv_mult, 1);
        assert_eq!(rows[first_final_perm + 2].final_root_poseidon2_recv_mult, 0);
        assert_eq!(rows[first_final_perm + 3].final_root_poseidon2_recv_mult, 0);
        let controls =
            WhirQueryRoundControl::from_round_rows(shape, &rows).expect("round controls derive");
        assert_eq!(controls.len(), shape.num_rounds);
        assert!(controls[0].is_assign);
        assert!(controls[0].is_merge);
        assert_eq!(controls[0].merge_beta, [F::zero(); D_EF]);
        assert!(controls[1].is_merge);
        assert!(!controls[1].is_assign);
        assert_eq!(controls[1].merge_beta, ext_limbs(&beta1));
        assert_eq!(controls[1].cfr, ext_limbs(&final_cfr));
        assert!(
            round_chain_residual(&rows).is_empty(),
            "A round-chain rows must internally balance"
        );

        let mut tampered = rows.clone();
        tampered[3].chain_recv_pending_beta[0] += F::one();
        assert!(
            !round_chain_residual(&tampered).is_empty(),
            "pending merge tampering must leave a WhirRoundChain residual"
        );
    }

    #[test]
    fn query_replay_builds_seed_round_cycle_and_binds_raw_sample_low_bits() {
        let rounds = 21;
        let shape = WhirSpecFoldShape {
            role_id: 0,
            num_rounds: rounds,
            c_chips: 1,
            num_public_values: 0,
            num_queries: 261,
            batching_bits: 10,
            query_bits: crate::whir_dt::columns::WHIR_CORE_QUERY_SAMPLE_BITS,
            log_blowup: 1,
            w0_tidx: 100,
        };
        let seed = WhirSpecFoldSeed {
            proof_idx: 3,
            shape,
            opening_point: vec![ext_limbs(&EF::zero()); rounds],
        };
        let cfr = ext_limbs(&ext(70));
        let mut controls = vec![
            WhirQueryRoundControl {
                r_fold: ext_limbs(&EF::from_canonical_usize(3)),
                cfr,
                ..empty_query_control()
            };
            rounds
        ];
        controls[0].is_merge = true;
        controls[0].is_assign = true;
        controls[2].emit_prep_seed = true;
        let pairs = vec![(cfr, cfr); rounds];

        let input = WhirQueryReplayInput {
            seed: seed.clone(),
            query_idx: 7,
            w_qbase: 500,
            query_sample_raw: F::from_canonical_usize(5),
            query_sample: 5,
            controls: controls.clone(),
            pair_source: WhirQueryPairSource::Explicit(pairs.clone()),
            leaf_sums_by_log_height: BTreeMap::from([(rounds + shape.log_blowup, cfr)]),
        };

        let rows = input.query_fold_rows().expect("fixture is internally consistent");
        assert_eq!(rows.len(), rounds + 1);
        assert!(rows[0].is_seed);
        assert_eq!(rows[0].query_sample, F::from_canonical_usize(5));
        assert_eq!(rows[0].query_sample_raw, F::from_canonical_usize(5));
        assert_eq!(rows[0].query_sample_high, 0);
        assert_eq!(rows[0].query_bits, shape.query_bits);
        assert_eq!(rows[0].r_rounds, rounds);
        assert_eq!(rows[0].query_sample_shift, 1usize << shape.query_bits);
        assert_eq!(rows[0].query_sample_high_bits, 9);
        assert_eq!(rows[0].cursor, rounds);
        assert_eq!(rows[0].idx, F::zero());
        assert_eq!(rows[0].folded, cfr);
        assert_eq!(rows[0].chain_send_idx, F::from_canonical_usize(5));
        assert!(rows[1].is_round && rows[1].is_assign);
        assert_eq!(rows[1].query_bits, shape.query_bits);
        assert_eq!(rows[1].r_rounds, rounds);
        assert_eq!(rows[1].idx, F::from_canonical_usize(5));
        assert!(rows[1].idx_bit);
        assert_eq!(rows[1].chain_send_idx, F::from_canonical_usize(2));
        assert!(!rows[1].chain_send_idx_bit);
        assert_eq!(rows[2].chain_send_idx, F::one());
        assert!(rows[2].chain_send_idx_bit);
        assert_eq!(rows.last().expect("round exists").chain_send_folded, cfr);
        assert!(
            query_chain_residual(&rows).is_empty(),
            "C query-chain rows must internally balance"
        );

        let sibling_input = WhirQueryReplayInput::from_sibling_values(
            seed,
            7,
            500,
            F::from_canonical_usize(5),
            5,
            controls.clone(),
            vec![cfr; rounds],
            BTreeMap::from([(rounds + shape.log_blowup, cfr)]),
        )
        .expect("sibling values reconstruct the same query rows");
        assert_eq!(
            query_pairs_from_siblings_oracle(
                shape,
                5,
                &controls,
                &vec![cfr; rounds],
                &BTreeMap::from([(rounds + shape.log_blowup, cfr)]),
            )
            .expect("independent pair oracle"),
            pairs
        );
        assert_eq!(sibling_input.query_fold_rows().expect("sibling-derived rows are valid"), rows);

        let mut bad_siblings = vec![cfr; rounds];
        bad_siblings[0][0] += F::one();
        let bad_sibling_input = WhirQueryReplayInput::from_sibling_values(
            input.seed.clone(),
            7,
            500,
            F::from_canonical_usize(5),
            5,
            controls,
            bad_siblings,
            BTreeMap::from([(rounds + shape.log_blowup, cfr)]),
        )
        .expect("the final fold equality is an AIR constraint");
        let bad_sibling_rows = bad_sibling_input.query_fold_rows().expect("rows materialize");
        assert_ne!(bad_sibling_rows[0].folded, bad_sibling_rows[0].cfr);

        let mut tampered = rows.clone();
        tampered[2].chain_send_idx_bit = false;
        assert!(
            !query_chain_residual(&tampered).is_empty(),
            "lookahead-bit tampering must leave a WhirQueryChain residual"
        );

        let mut raw_mismatch = input.clone();
        raw_mismatch.query_sample_raw = F::from_canonical_usize(6);
        assert_eq!(
            raw_mismatch.query_fold_rows(),
            Err(WhirSpecFoldError::QuerySampleLowMismatch { expected: 6, actual: 5 })
        );

        let mut pair_mismatch = input;
        let WhirQueryPairSource::Explicit(pairs) = &mut pair_mismatch.pair_source else {
            panic!("fixture uses explicit pairs");
        };
        pairs[0].1[0] += F::one();
        let pair_mismatch_rows =
            pair_mismatch.query_fold_rows().expect("selected-pair validity is an AIR constraint");
        assert_ne!(pair_mismatch_rows[1].f1, pair_mismatch_rows[1].leaf_sum);
    }

    #[test]
    fn query_replay_accepts_native_child_query_bits_21() {
        let rounds = 19;
        let shape = WhirSpecFoldShape {
            role_id: 1,
            num_rounds: rounds,
            c_chips: 1,
            num_public_values: 0,
            num_queries: 160,
            batching_bits: 10,
            query_bits: 21,
            log_blowup: 2,
            w0_tidx: 100,
        };
        let cfr = ext_limbs(&ext(70));
        let mut controls = vec![
            WhirQueryRoundControl {
                r_fold: ext_limbs(&EF::from_canonical_usize(3)),
                cfr,
                ..empty_query_control()
            };
            rounds
        ];
        controls[0].is_merge = true;
        controls[0].is_assign = true;
        let raw_high = 7usize;
        let raw_low = 5usize;
        let input = WhirQueryReplayInput {
            seed: WhirSpecFoldSeed {
                proof_idx: 3,
                shape,
                opening_point: vec![ext_limbs(&EF::zero()); rounds],
            },
            query_idx: 7,
            w_qbase: 500,
            query_sample_raw: F::from_canonical_usize((raw_high << shape.query_bits) + raw_low),
            query_sample: raw_low,
            controls,
            pair_source: WhirQueryPairSource::Explicit(vec![(cfr, cfr); rounds]),
            leaf_sums_by_log_height: BTreeMap::from([(rounds + shape.log_blowup, cfr)]),
        };

        let rows = input.query_fold_rows().expect("qb=21 replay is supported");
        assert_eq!(rows[0].query_bits, 21);
        assert_eq!(rows[0].r_rounds, rounds);
        assert_eq!(rows[0].query_sample_high, raw_high);
        assert_eq!(rows[0].query_sample_shift, 1usize << 21);
        assert_eq!(rows[0].query_sample_high_bits, 10);
        assert!(query_chain_residual(&rows).is_empty());
    }

    #[test]
    fn query_fold_denominator_inverse_recurrence_matches_per_round_inverse() {
        for (query_bits, query_sample, rounds) in [(5, 0, 4), (21, 5, 19), (22, 17, 21)] {
            let (_, _, _, mut x) = query_twiddle_seed(query_sample, query_bits - 1);
            let mut idx = query_sample;
            let mut denominator_inv =
                initial_query_fold_denominator_inverse(x).expect("twiddle seed is nonzero");
            for round in 0..rounds {
                let direct = (F::two() * x)
                    .try_inverse()
                    .unwrap_or_else(|| panic!("round {round} twiddle is nonzero"));
                assert_eq!(denominator_inv, direct, "round {round} inverse mismatch");

                let next_idx = idx >> 1;
                let sign = if next_idx & 1 == 1 { F::zero() - F::one() } else { F::one() };
                x = sign * x * x;
                denominator_inv = next_query_fold_denominator_inverse(denominator_inv, sign);
                idx = next_idx;
            }
        }

        assert_eq!(
            initial_query_fold_denominator_inverse(F::zero()),
            Err(WhirSpecFoldError::ZeroQueryTwiddle { round: 0 })
        );
    }

    fn empty_query_control() -> WhirQueryRoundControl {
        WhirQueryRoundControl {
            r_fold: [F::zero(); D_EF],
            is_merge: false,
            is_assign: false,
            merge_beta: [F::zero(); D_EF],
            merge_eq: [F::zero(); D_EF],
            emit_prep_seed: false,
            cfr: [F::zero(); D_EF],
        }
    }

    fn batch_eval_chain_residual(rows: &[RecursionWhirBatchEvalRow]) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::<Vec<u32>, i64>::new();
        for row in rows {
            apply_residual(
                &mut residual,
                batch_eval_chain_recv_key(row),
                -(row.chain_recv_mult as i64),
            );
            apply_residual(
                &mut residual,
                batch_eval_chain_send_key(row),
                row.chain_send_mult as i64,
            );
        }
        residual.retain(|_, value| *value != 0);
        residual
    }

    fn batch_eval_chain_recv_key(row: &RecursionWhirBatchEvalRow) -> Vec<u32> {
        let mut key = vec![row.proof_idx as u32, row.chain_recv_cursor as u32];
        key.extend(row.alpha.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.pow_in.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.acc_in.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.group_base_in.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn batch_eval_chain_send_key(row: &RecursionWhirBatchEvalRow) -> Vec<u32> {
        let mut key = vec![row.proof_idx as u32, row.chain_send_cursor as u32];
        key.extend(row.alpha.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.pow_out.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.acc_out.iter().map(|value| value.as_canonical_u32()));
        key.extend(row.group_base_out.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn apply_residual(residual: &mut BTreeMap<Vec<u32>, i64>, key: Vec<u32>, delta: i64) {
        *residual.entry(key).or_default() += delta;
    }

    fn round_chain_residual(rows: &[RecursionWhirRoundRow]) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::<Vec<u32>, i64>::new();
        for row in rows {
            apply_residual(&mut residual, round_chain_recv_key(row), -(row.chain_recv_mult as i64));
            apply_residual(&mut residual, round_chain_send_key(row), row.chain_send_mult as i64);
        }
        residual.retain(|_, value| *value != 0);
        residual
    }

    fn query_chain_residual(rows: &[RecursionWhirQueryFoldRow]) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::<Vec<u32>, i64>::new();
        for row in rows {
            apply_residual(&mut residual, query_chain_recv_key(row), -1);
            apply_residual(&mut residual, query_chain_send_key(row), 1);
        }
        residual.retain(|_, value| *value != 0);
        residual
    }

    fn query_chain_recv_key(row: &RecursionWhirQueryFoldRow) -> Vec<u32> {
        let mut key = vec![
            row.proof_idx as u32,
            row.query_idx as u32,
            row.cursor as u32,
            row.query_bits as u32,
            row.r_rounds as u32,
            row.idx.as_canonical_u32(),
            u32::from(row.idx_bit),
            row.x.as_canonical_u32(),
            row.acc.as_canonical_u32(),
            row.ipw.as_canonical_u32(),
        ];
        key.extend(row.folded.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn query_chain_send_key(row: &RecursionWhirQueryFoldRow) -> Vec<u32> {
        let mut key = vec![
            row.proof_idx as u32,
            row.query_idx as u32,
            row.chain_send_cursor as u32,
            row.query_bits as u32,
            row.r_rounds as u32,
            row.chain_send_idx.as_canonical_u32(),
            u32::from(row.chain_send_idx_bit),
            row.chain_send_x.as_canonical_u32(),
            row.chain_send_acc.as_canonical_u32(),
            row.chain_send_ipw.as_canonical_u32(),
        ];
        key.extend(row.chain_send_folded.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn leaf_chain_residual(
        base_rows: &[RecursionWhirLeafStreamRow],
        ext_rows: &[RecursionWhirLeafExtStreamRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::<Vec<u32>, i64>::new();
        // Boundary mults: instance-start rows do not recv, unit-end rows
        // do not send (linear per-instance chain).
        for row in base_rows {
            if !row.is_unit_start {
                apply_residual(&mut residual, leaf_base_chain_recv_key(row), -1);
            }
            if !row.is_unit_end {
                apply_residual(&mut residual, leaf_base_chain_send_key(row), 1);
            }
        }
        for row in ext_rows {
            if !row.is_unit_start {
                apply_residual(&mut residual, leaf_ext_chain_recv_key(row), -1);
            }
            if !row.is_unit_end {
                apply_residual(&mut residual, leaf_ext_chain_send_key(row), 1);
            }
        }
        residual.retain(|_, value| *value != 0);
        residual
    }

    fn leaf_base_chain_recv_key(row: &RecursionWhirLeafStreamRow) -> Vec<u32> {
        leaf_chain_key(
            row.proof_idx,
            row.idx,
            row.chain_recv_cursor,
            row.chain_recv_log_height,
            row.chain_recv_batch_id,
            row.alpha,
            row.pow_in,
            row.acc_in,
        )
    }

    fn leaf_base_chain_send_key(row: &RecursionWhirLeafStreamRow) -> Vec<u32> {
        leaf_chain_key(
            row.proof_idx,
            row.idx,
            row.chain_send_cursor,
            row.log_height,
            row.batch_id,
            row.alpha,
            row.pow_out,
            row.acc_out,
        )
    }

    fn leaf_ext_chain_recv_key(row: &RecursionWhirLeafExtStreamRow) -> Vec<u32> {
        leaf_chain_key(
            row.proof_idx,
            row.idx,
            row.chain_recv_cursor,
            row.chain_recv_log_height,
            row.chain_recv_batch_id,
            row.alpha,
            row.pow_in,
            row.acc_in,
        )
    }

    fn leaf_ext_chain_send_key(row: &RecursionWhirLeafExtStreamRow) -> Vec<u32> {
        leaf_chain_key(
            row.proof_idx,
            row.idx,
            row.chain_send_cursor,
            row.log_height,
            row.batch_id,
            row.alpha,
            row.pow_out,
            row.acc_out,
        )
    }

    fn leaf_chain_key(
        proof_idx: usize,
        query_idx: usize,
        cursor: usize,
        log_height: usize,
        batch_id: usize,
        alpha: [F; D_EF],
        pow: [F; D_EF],
        acc: [F; D_EF],
    ) -> Vec<u32> {
        let mut key = vec![
            proof_idx as u32,
            query_idx as u32,
            cursor as u32,
            log_height as u32,
            batch_id as u32,
        ];
        key.extend(alpha.iter().map(|value| value.as_canonical_u32()));
        key.extend(pow.iter().map(|value| value.as_canonical_u32()));
        key.extend(acc.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn round_chain_recv_key(row: &RecursionWhirRoundRow) -> Vec<u32> {
        round_chain_key(
            row.proof_idx,
            row.chain_recv_round,
            row.chain_recv_tidx,
            row.chain_recv_claim,
            row.chain_recv_eq,
            row.chain_recv_pending_is_merge,
            row.chain_recv_pending_beta,
            row.chain_recv_pending_eq,
        )
    }

    fn round_chain_send_key(row: &RecursionWhirRoundRow) -> Vec<u32> {
        round_chain_key(
            row.proof_idx,
            row.chain_send_round,
            row.chain_send_tidx,
            row.chain_send_claim,
            row.chain_send_eq,
            row.chain_send_pending_is_merge,
            row.chain_send_pending_beta,
            row.chain_send_pending_eq,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn round_chain_key(
        proof_idx: usize,
        round: usize,
        tidx: usize,
        claim: [F; D_EF],
        eq: [F; D_EF],
        pending_is_merge: bool,
        pending_beta: [F; D_EF],
        pending_eq: [F; D_EF],
    ) -> Vec<u32> {
        let mut key = vec![proof_idx as u32, round as u32, tidx as u32, pending_is_merge as u32];
        key.extend(claim.iter().map(|value| value.as_canonical_u32()));
        key.extend(eq.iter().map(|value| value.as_canonical_u32()));
        key.extend(pending_beta.iter().map(|value| value.as_canonical_u32()));
        key.extend(pending_eq.iter().map(|value| value.as_canonical_u32()));
        key
    }

    fn ext(value: usize) -> EF {
        EF::from_canonical_usize(value)
    }

    fn digest(base: usize) -> [F; 8] {
        core::array::from_fn(|idx| F::from_canonical_usize(base + idx))
    }

    fn eq_factor(z: EF, r: EF) -> EF {
        z * r + (EF::one() - z) * (EF::one() - r)
    }

    fn quadratic_coeffs_for_claim_and_fold(claim: EF, folded: EF, r: EF) -> [[F; D_EF]; 3] {
        let denom = EF::one() - r.double();
        let c0 = (folded - r * claim) * denom.try_inverse().expect("non-zero fixture denom");
        let c1 = claim - c0.double();
        [ext_limbs(&c0), ext_limbs(&c1), ext_limbs(&EF::zero())]
    }
}
