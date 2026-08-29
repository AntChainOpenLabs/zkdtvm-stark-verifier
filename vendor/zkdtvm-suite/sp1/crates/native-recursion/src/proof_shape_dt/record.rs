use std::collections::BTreeSet;

use crate::{
    child_views::{NativeChildRole, NativeChildViews, KOALABEAR_MAX_TRACE_LOG_HEIGHT},
    config::F,
    statement_dt::CORE_PV_SHARD,
    system_dt::{
        RecursionNativeChipMetadataRequest, RecursionProofShapeChip, RecursionProofShapeRecord,
        RecursionRecord,
    },
    whir_dt::WHIR_ROLE_CORE,
};
use dt_stark::{
    air::stable_air_id_v1,
    sumcheck::config::{MlCom, SCStarkGenericConfig},
    DIGEST_SIZE,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofShapeRecordError {
    UnsupportedValueField,
    MissingPermutationCommitment,
    PublicValueLengthMismatch {
        expected: usize,
        actual: usize,
    },
    StaticChipIdOutOfRange {
        chip_name: String,
        static_chip_id: usize,
    },
    LogHeightTooLarge {
        chip_name: String,
        log_height: usize,
    },
    ProofOrderViolation {
        chip_idx: usize,
        prev_log_height: usize,
        prev_static_chip_id: usize,
        log_height: usize,
        static_chip_id: usize,
    },
    DuplicateStaticChipId {
        chip_idx: usize,
        static_chip_id: usize,
    },
    RangeValueTooLarge {
        chip_idx: usize,
        range_val: usize,
    },
}

pub fn metadata_universe_from_view(
    role_id: usize,
    metadata: &crate::child_views::NativeChildMetadataView<'_>,
) -> Vec<RecursionNativeChipMetadataRequest> {
    let mut chips = metadata.chips.iter().collect::<Vec<_>>();
    chips.sort_by(|left, right| left.name.cmp(&right.name));
    chips
        .into_iter()
        .enumerate()
        .map(|(chip_id, chip)| RecursionNativeChipMetadataRequest {
            role_id,
            chip_id: chip_id + metadata.static_chip_id_offset,
            stable_air_id: stable_air_id_v1(&chip.name),
            prep_width: chip.preprocessed_width,
            main_width: chip.main_width,
            perm_width: chip.permutation_width,
            constraint_count: chip.constraint_count,
            gate_count: chip.gate_count,
            count: 0,
        })
        .collect()
}

pub fn record_proof_shape_from_views<SC>(
    record: &mut RecursionRecord,
    proof_idx: usize,
    views: &NativeChildViews<'_, SC>,
    publish_external: bool,
) -> Result<(), ProofShapeRecordError>
where
    SC: SCStarkGenericConfig<Val = F>,
    MlCom<SC>: AsRef<[F; DIGEST_SIZE]>,
{
    let role_id = role_id(views.layout.role());
    if views.layout.chips().len() > 256 {
        return Err(ProofShapeRecordError::StaticChipIdOutOfRange {
            chip_name: "<role-universe>".to_string(),
            static_chip_id: views.layout.chips().len(),
        });
    }

    let proof = views.proof.proof();
    let num_public_values = views.layout.num_observed_public_values();
    if proof.public_values.len() < num_public_values {
        return Err(ProofShapeRecordError::PublicValueLengthMismatch {
            expected: num_public_values,
            actual: proof.public_values.len(),
        });
    }

    let permutation_commit = proof
        .commitment
        .permutation_commit
        .as_ref()
        .ok_or(ProofShapeRecordError::MissingPermutationCommitment)?;

    let mut chips = Vec::with_capacity(views.proof.chip_count());
    for opened_chip in views.proof.ordered_chips() {
        let metadata = views
            .layout
            .find_chip(opened_chip.name)
            .expect("NativeChildViews validated metadata existence");
        let static_chip_id = views
            .layout
            .static_chip_id(opened_chip.name)
            .expect("verified layout contains every admitted chip");
        let log_height = opened_chip.opened_values.log_height;
        if log_height > KOALABEAR_MAX_TRACE_LOG_HEIGHT {
            return Err(ProofShapeRecordError::LogHeightTooLarge {
                chip_name: opened_chip.name.to_string(),
                log_height,
            });
        }
        chips.push(RecursionProofShapeChip {
            chip_idx: opened_chip.index,
            static_chip_id,
            stable_air_id: stable_air_id_v1(opened_chip.name),
            log_height,
            prep_width: metadata.preprocessed_width,
            main_width: metadata.main_width,
            perm_width: metadata.permutation_width,
            constraint_count: metadata.constraint_count,
            gate_count: metadata.gate_count,
        });
    }
    validate_sorted_shape(&chips)?;

    let proof_shape = RecursionProofShapeRecord {
        role_id,
        num_public_values,
        vk_commit: digest_from_commit(&views.vk.vk().commit),
        vk_meta: vk_meta(views),
        vk_meta_send_mults: vec![0; vk_meta_len(views.layout.role())],
        public_values: proof.public_values[..num_public_values].to_vec(),
        public_value_send_mults: vec![u32::from(publish_external); num_public_values],
        main_commit: digest_from_commit(&proof.commitment.main_commit),
        permutation_commit: digest_from_commit(permutation_commit),
        chips,
        publish_external,
        publish_whir_inputs: false,
        publish_terminal_summary: false,
    };

    for chip in &proof_shape.chips {
        record.native_chip_metadata.record_metadata(chip.metadata_request(role_id));
    }
    for range_val in proof_shape_range_values(&proof_shape) {
        record.range.record_range(range_val, 8);
    }
    // Segment-band range demands: per chip, its own id band; per chip AND the E5 row, the
    // chain-recv'd prev id band (E5's prev is the last chip).
    let mut prev_band = 0usize;
    for chip in &proof_shape.chips {
        let local = chip.static_chip_id % 128;
        record.range.record_range(local, 8);
        record.range.record_range(127 - local, 8);
        record.range.record_range(prev_band, 8);
        record.range.record_range(127 - prev_band, 8);
        prev_band = local;
    }
    record.range.record_range(prev_band, 8);
    record.range.record_range(127 - prev_band, 8);
    // Statement shard-range demands (lift machines only fire these lookups): the core
    // child's shard byte limbs, once per child on its scalar row.
    if role_id == WHIR_ROLE_CORE {
        use p3_field::PrimeField32;
        let shard = proof_shape.public_values[CORE_PV_SHARD].as_canonical_u32() as usize;
        record.range.record_range(shard % 256, 8);
        record.range.record_range(shard / 256, 8);
    }
    record.proof_record_mut(proof_idx).proof_shape = proof_shape;
    Ok(())
}

pub fn proof_shape_range_values(
    proof_shape: &RecursionProofShapeRecord,
) -> impl Iterator<Item = usize> + '_ {
    let mut prev_log_height = 25usize;
    let mut prev_static_chip_id = 0usize;
    proof_shape.chips.iter().map(move |chip| {
        let range_val = proof_shape_range_value(prev_log_height, prev_static_chip_id, chip);
        prev_log_height = chip.log_height;
        prev_static_chip_id = chip.static_chip_id;
        range_val
    })
}

pub(crate) fn proof_shape_range_value(
    prev_log_height: usize,
    prev_static_chip_id: usize,
    chip: &RecursionProofShapeChip,
) -> usize {
    if chip.log_height != prev_log_height {
        prev_log_height - chip.log_height - 1
    } else {
        chip.static_chip_id - prev_static_chip_id - 1
    }
}

fn validate_sorted_shape(chips: &[RecursionProofShapeChip]) -> Result<(), ProofShapeRecordError> {
    let mut prev_log_height = 25usize;
    let mut prev_static_chip_id = 0usize;
    let mut seen_static_chip_ids = BTreeSet::new();
    for chip in chips {
        if !seen_static_chip_ids.insert(chip.static_chip_id) {
            return Err(ProofShapeRecordError::DuplicateStaticChipId {
                chip_idx: chip.chip_idx,
                static_chip_id: chip.static_chip_id,
            });
        }
        let group_start = chip.log_height != prev_log_height;
        let ok = if group_start {
            prev_log_height > chip.log_height
        } else {
            chip.static_chip_id > prev_static_chip_id
        };
        if !ok {
            return Err(ProofShapeRecordError::ProofOrderViolation {
                chip_idx: chip.chip_idx,
                prev_log_height,
                prev_static_chip_id,
                log_height: chip.log_height,
                static_chip_id: chip.static_chip_id,
            });
        }
        let range_val = proof_shape_range_value(prev_log_height, prev_static_chip_id, chip);
        if range_val >= 256 {
            return Err(ProofShapeRecordError::RangeValueTooLarge {
                chip_idx: chip.chip_idx,
                range_val,
            });
        }
        prev_log_height = chip.log_height;
        prev_static_chip_id = chip.static_chip_id;
    }
    Ok(())
}

fn vk_meta<SC>(views: &NativeChildViews<'_, SC>) -> Vec<F>
where
    SC: SCStarkGenericConfig<Val = F>,
    MlCom<SC>: AsRef<[F; DIGEST_SIZE]>,
{
    let vk = views.vk.vk();
    let mut values = digest_from_commit(&vk.commit).to_vec();
    if views.layout.role() == NativeChildRole::Core {
        values.push(vk.pc_start);
        values.extend(
            dt_stark::global_d11::canonical_program_boundary_fields_v1::<F>(
                &vk.program_boundary,
            )
            .expect("validated VK program boundary"),
        );
    }
    debug_assert_eq!(values.len(), vk_meta_len(views.layout.role()));
    values
}

fn vk_meta_len(role: NativeChildRole) -> usize {
    if role == NativeChildRole::Core {
        crate::proof_shape_dt::bus::PROOF_SHAPE_CORE_VK_META_VALUE_COUNT
    } else {
        crate::proof_shape_dt::bus::PROOF_SHAPE_NATIVE_VK_META_VALUE_COUNT
    }
}

fn digest_from_commit<C>(commit: &C) -> [F; 8]
where
    C: AsRef<[F; DIGEST_SIZE]>,
{
    *commit.as_ref()
}

fn role_id(role: NativeChildRole) -> usize {
    match role {
        NativeChildRole::Core => 0,
        NativeChildRole::Compress => 1,
        NativeChildRole::Shrink => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dt_stark::air::InteractionScope;

    use crate::child_views::{NativeAirAuthority, NativeChildMetadataView, NativeChipMetadata};

    #[test]
    fn metadata_universe_uses_rust_string_ord_not_natural_sort() {
        let chips = vec![
            metadata_chip("chip_2", 2),
            metadata_chip("chip_10", 10),
            metadata_chip("Chip_1", 1),
        ];
        let view = NativeChildMetadataView {
            static_chip_id_offset: 0,
            role: NativeChildRole::Core,
            air_authority: NativeAirAuthority::PublicMetadata,
            num_observed_public_values: 0,
            contains_global_bus: false,
            chips: &chips,
        };

        let universe = metadata_universe_from_view(0, &view);
        assert_eq!(universe.len(), 3);
        assert_eq!(universe[0].chip_id, 0);
        assert_eq!(universe[0].prep_width, 1);
        assert_eq!(universe[1].chip_id, 1);
        assert_eq!(universe[1].prep_width, 10);
        assert_eq!(universe[2].chip_id, 2);
        assert_eq!(universe[2].prep_width, 2);
    }

    #[test]
    fn shape_range_values_match_desc_height_then_static_id_chain() {
        let shape = RecursionProofShapeRecord {
            chips: vec![shape_chip(0, 1, 4), shape_chip(1, 3, 4), shape_chip(2, 0, 2)],
            ..RecursionProofShapeRecord::default()
        };

        let values = proof_shape_range_values(&shape).collect::<Vec<_>>();
        assert_eq!(values, vec![20, 1, 1]);
        assert!(validate_sorted_shape(&shape.chips).is_ok());
    }

    #[test]
    fn duplicate_static_id_across_height_groups_is_rejected() {
        let chips = vec![shape_chip(0, 1, 4), shape_chip(1, 1, 2)];
        assert_eq!(
            validate_sorted_shape(&chips),
            Err(ProofShapeRecordError::DuplicateStaticChipId { chip_idx: 1, static_chip_id: 1 })
        );
    }

    #[test]
    fn increasing_height_is_rejected() {
        let chips = vec![shape_chip(0, 1, 4), shape_chip(1, 2, 5)];
        assert_eq!(
            validate_sorted_shape(&chips),
            Err(ProofShapeRecordError::ProofOrderViolation {
                chip_idx: 1,
                prev_log_height: 4,
                prev_static_chip_id: 1,
                log_height: 5,
                static_chip_id: 2,
            })
        );
    }

    #[test]
    fn equal_height_static_id_reverse_is_rejected() {
        let chips = vec![shape_chip(0, 3, 4), shape_chip(1, 2, 4)];
        assert_eq!(
            validate_sorted_shape(&chips),
            Err(ProofShapeRecordError::ProofOrderViolation {
                chip_idx: 1,
                prev_log_height: 4,
                prev_static_chip_id: 3,
                log_height: 4,
                static_chip_id: 2,
            })
        );
    }

    #[test]
    fn large_same_height_static_id_gap_is_rejected() {
        let chips = vec![shape_chip(0, 0, 4), shape_chip(1, 300, 4)];
        assert_eq!(
            validate_sorted_shape(&chips),
            Err(ProofShapeRecordError::RangeValueTooLarge { chip_idx: 1, range_val: 299 })
        );
    }

    fn metadata_chip(name: &str, prep_width: usize) -> NativeChipMetadata {
        NativeChipMetadata {
            name: name.to_string(),
            preprocessed_width: prep_width,
            main_width: prep_width + 100,
            permutation_width: 5,
            commit_scope: InteractionScope::Local,
            has_local_interactions: true,
            constraint_count: prep_width + 200,
            gate_count: prep_width + 200,
            logup_batch_size: 1,
            required_max_beta_power: 1,
        }
    }

    fn shape_chip(
        chip_idx: usize,
        static_chip_id: usize,
        log_height: usize,
    ) -> RecursionProofShapeChip {
        RecursionProofShapeChip {
            chip_idx,
            static_chip_id,
            stable_air_id: 43 + chip_idx as u32,
            log_height,
            prep_width: 1,
            main_width: 2,
            perm_width: 3,
            constraint_count: 4,
            gate_count: 4,
        }
    }
}
