//! Child proof, VK, metadata, config, and PCS-opening views.
//!
//! These views are the boundary between raw zkDTVM child proof objects and the
//! native-recursion transcript production and the sumcheck, WHIR, and PolyAIR
//! tracegen material builders. They validate layout and source ownership, but
//! they do not claim PCS or constraint soundness by themselves.

use dt_stark::{
    air::InteractionScope,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, SCStarkGenericConfig},
        keys::SCStarkVerifyingKey,
        proof::{
            SCChipOpenedValues, SCShardCommitment, SCShardOpenedValues, SCShardProof, SumcheckProof,
        },
    },
    Challenge, Val,
};
use p3_field::{AbstractExtensionField, Field};
use p3_matrix::Dimensions;

use crate::{
    config::D_EF, statement_dt::NATIVE_RECURSION_NUM_PV_ELTS,
    symbolic_expr_fixed_dt::RecursionChildRole, symbolic_ir_dt::RecursionPolyAirVerifierProgramDto,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeChildRole {
    Core,
    Compress,
    Shrink,
}

/// KoalaBear two-adic domain budget for child traces after WHIR/FRI blowup.
pub const KOALABEAR_MAX_TRACE_LOG_HEIGHT: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAirAuthority {
    RecursionVkConstant,
    PublicMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePcsBatchKind {
    Preprocessed,
    Main,
    Permutation,
}

#[derive(Debug, Clone, Copy)]
pub struct NativePcsBatchView<'a> {
    pub kind: NativePcsBatchKind,
    pub dimensions: &'a [Dimensions],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeChildViewError {
    Global146IdentityMismatch,
    EmptyChipOrdering,
    ChipOpeningLengthMismatch {
        chip_ordering: usize,
        opened_values: usize,
    },
    ChipOrderingIndexOutOfRange {
        source: &'static str,
        chip_name: String,
        index: usize,
        len: usize,
    },
    ChipOrderingDuplicateIndex {
        source: &'static str,
        first_chip_name: String,
        second_chip_name: String,
        index: usize,
    },
    VkChipInformationLengthMismatch {
        chip_ordering: usize,
        chip_information: usize,
    },
    VkChipInformationOrderMismatch {
        chip_name: String,
        expected_index: usize,
        actual_index: usize,
    },
    VkMissingConstraintCount {
        chip_name: String,
    },
    ProofChipMissingConstraintCount {
        chip_name: String,
    },
    ProofPreprocessedChipMissingFromVk {
        chip_name: String,
    },
    PublicValueLengthTooShort {
        required: usize,
        actual: usize,
    },
    PcsBatchCountMismatch {
        expected: usize,
        actual: usize,
        permutation_commit_present: bool,
    },
    PcsDimensionLengthMismatch {
        batch: NativePcsBatchKind,
        expected: usize,
        actual: usize,
    },
    DimensionWidthMismatch {
        batch: NativePcsBatchKind,
        chip_name: String,
        expected: usize,
        actual: usize,
    },
    DimensionHeightMismatch {
        batch: NativePcsBatchKind,
        chip_name: String,
        expected: usize,
        actual: usize,
    },
    LogHeightTooLarge {
        chip_name: String,
        log_height: usize,
    },
    KoalaBearTraceLogHeightExceeded {
        max_trace_log_height: usize,
        log_height: usize,
        log_blowup: usize,
    },
    PermutationCommitMissingButOpenedValuesPresent {
        chip_name: String,
    },
    PermutationCommitPresentButNoOpenedValues,
    MetadataChipMissing {
        chip_name: String,
    },
    DuplicateMetadataChip {
        chip_name: String,
    },
    RoleMismatch {
        metadata_role: NativeChildRole,
        verifier_config_role: NativeChildRole,
    },
    MetadataWidthMismatch {
        chip_name: String,
        column_kind: NativeChildColumnKind,
        expected: usize,
        actual: usize,
    },
    MetadataConstraintCountMismatch {
        chip_name: String,
        expected: usize,
        actual: usize,
    },
    NoLocalInteractionHasLocalCumulativeSum {
        chip_name: String,
    },
    InvalidNumSkipRounds,
    ChipLogHeightThresholdNotDivisible {
        chip_log_height_threshold: usize,
        num_skip_rounds: usize,
    },
    UnsupportedNonlinearRounds {
        chip_log_height_threshold: usize,
        num_rounds_nonlinear: usize,
    },
    MissingTranscriptBoundOpeningPoint,
}

pub type NativeChildViewResult<T> = Result<T, NativeChildViewError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeChildColumnKind {
    Preprocessed,
    Main,
    Permutation,
}

#[derive(Debug, Clone)]
pub struct NativeChipMetadata {
    pub name: String,
    pub preprocessed_width: usize,
    pub main_width: usize,
    pub permutation_width: usize,
    pub commit_scope: InteractionScope,
    pub has_local_interactions: bool,
    pub constraint_count: usize,
    /// Number of actual `air.eval()` gate roots. This differs from
    /// `constraint_count` when retained precomputations are present.
    pub gate_count: usize,
    pub logup_batch_size: usize,
    pub required_max_beta_power: usize,
}

#[derive(Debug, Clone)]
pub struct NativeChildMetadataView<'a> {
    pub role: NativeChildRole,
    pub air_authority: NativeAirAuthority,
    pub num_observed_public_values: usize,
    pub contains_global_bus: bool,
    /// Dual-segment replay support (M1 Part 3 / M2 mixed nodes): static chip ids of this
    /// child machine's universe are offset so distinct child-machine DAG segments coexist
    /// in one constraint program artifact.
    pub static_chip_id_offset: usize,
    pub chips: &'a [NativeChipMetadata],
}

/// Frozen, backend-neutral authority for one child-machine replay segment.
///
/// It is compiled together with the immutable constraint program at ladder/prover cold start.
/// Per-proof admission borrows this object and performs only proof-local shape checks plus one
/// bounded name-to-id lookup for each present chip; it never rebuilds or sorts machine metadata,
/// validates a full VK map, or walks the machine layout a second time.
#[derive(Debug)]
pub struct VerifiedChildLayout {
    role: NativeChildRole,
    air_authority: NativeAirAuthority,
    num_observed_public_values: usize,
    contains_global_bus: bool,
    static_chip_id_offset: usize,
    chips: Box<[NativeChipMetadata]>,
    static_chip_ids: Box<[usize]>,
    chip_index_by_name: hashbrown::HashMap<String, usize>,
}

impl VerifiedChildLayout {
    pub(crate) fn compile_all(
        program: &RecursionPolyAirVerifierProgramDto,
    ) -> Result<Box<[Self]>, String> {
        let role = match program.role {
            RecursionChildRole::Core => NativeChildRole::Core,
            RecursionChildRole::Compress => NativeChildRole::Compress,
            RecursionChildRole::Shrink => NativeChildRole::Shrink,
        };
        let (num_observed_public_values, contains_global_bus) = match role {
            NativeChildRole::Core => (dt_stark::air::DT_PROOF_NUM_PV_ELTS, true),
            NativeChildRole::Compress | NativeChildRole::Shrink => {
                (NATIVE_RECURSION_NUM_PV_ELTS, false)
            }
        };
        let mut bases = Vec::new();
        for chip in &program.chips {
            let base = chip.static_chip_id & !127;
            if bases.last().copied() != Some(base) {
                bases.push(base);
            }
        }
        let mut layouts = Vec::new();
        layouts
            .try_reserve_exact(bases.len())
            .map_err(|_| "verified child layout allocation rejected".to_string())?;
        for base in bases {
            let segment = program
                .chips
                .iter()
                .filter(|chip| chip.static_chip_id & !127 == base)
                .collect::<Vec<_>>();
            let mut chips = Vec::new();
            let mut static_chip_ids = Vec::new();
            let mut chip_index_by_name = hashbrown::HashMap::new();
            chips
                .try_reserve_exact(segment.len())
                .map_err(|_| "verified child metadata allocation rejected".to_string())?;
            static_chip_ids
                .try_reserve_exact(segment.len())
                .map_err(|_| "verified child id allocation rejected".to_string())?;
            chip_index_by_name
                .try_reserve(segment.len())
                .map_err(|_| "verified child name index allocation rejected".to_string())?;
            for chip in segment {
                let permutation_width = chip
                    .lookup_multiplicity_roots
                    .len()
                    .div_ceil(chip.logup_batch_size)
                    .checked_mul(D_EF)
                    .ok_or_else(|| {
                        format!("verified child permutation width overflow for {}", chip.chip_name)
                    })?;
                let index = chips.len();
                if chip_index_by_name.insert(chip.chip_name.clone(), index).is_some() {
                    return Err(format!(
                        "duplicate verified child chip name {} in segment {base}",
                        chip.chip_name
                    ));
                }
                static_chip_ids.push(chip.static_chip_id);
                chips.push(NativeChipMetadata {
                    name: chip.chip_name.clone(),
                    preprocessed_width: chip.widths.preprocessed,
                    main_width: chip.widths.main,
                    permutation_width,
                    commit_scope: chip.commit_scope,
                    // Native machines currently expose local interaction columns for every chip.
                    // This is also checked against the cold machine metadata below.
                    has_local_interactions: true,
                    constraint_count: chip.num_constraints_from_builder,
                    gate_count: chip.gate_roots.len(),
                    logup_batch_size: chip.logup_batch_size,
                    required_max_beta_power: chip
                        .derived_roots
                        .iter()
                        .filter_map(|root| match root {
                            crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::BetaPower {
                                power,
                            } => Some(*power),
                            _ => None,
                        })
                        .max()
                        .map(|power| {
                            power.checked_add(1).ok_or_else(|| {
                                format!(
                                    "verified child beta-power width overflow for {}",
                                    chip.chip_name
                                )
                            })
                        })
                        .transpose()?
                        .unwrap_or(0),
                });
            }
            layouts.push(Self {
                role,
                air_authority: NativeAirAuthority::PublicMetadata,
                num_observed_public_values,
                contains_global_bus,
                static_chip_id_offset: base,
                chips: chips.into_boxed_slice(),
                static_chip_ids: static_chip_ids.into_boxed_slice(),
                chip_index_by_name,
            });
        }
        Ok(layouts.into_boxed_slice())
    }

    pub const fn role(&self) -> NativeChildRole {
        self.role
    }

    pub const fn air_authority(&self) -> NativeAirAuthority {
        self.air_authority
    }

    pub const fn num_observed_public_values(&self) -> usize {
        self.num_observed_public_values
    }

    pub const fn contains_global_bus(&self) -> bool {
        self.contains_global_bus
    }

    pub const fn static_chip_id_offset(&self) -> usize {
        self.static_chip_id_offset
    }

    pub fn chips(&self) -> &[NativeChipMetadata] {
        &self.chips
    }

    pub fn find_chip(&self, name: &str) -> Option<&NativeChipMetadata> {
        self.chip_index_by_name.get(name).and_then(|&index| self.chips.get(index))
    }

    pub fn static_chip_id(&self, name: &str) -> Option<usize> {
        self.chip_index_by_name
            .get(name)
            .and_then(|&index| self.static_chip_ids.get(index))
            .copied()
    }

    /// One-time cold binding between the frozen program layout and the live recording machine.
    pub(crate) fn validate_machine_metadata(
        &self,
        machine: &[NativeChipMetadata],
    ) -> NativeChildViewResult<()> {
        if machine.len() != self.chips.len() {
            return Err(NativeChildViewError::MetadataChipMissing {
                chip_name: format!(
                    "<machine layout length {} != {}>",
                    machine.len(),
                    self.chips.len()
                ),
            });
        }
        for actual in machine {
            let expected = self.find_chip(&actual.name).ok_or_else(|| {
                NativeChildViewError::MetadataChipMissing { chip_name: actual.name.clone() }
            })?;
            for (kind, expected_width, actual_width) in [
                (
                    NativeChildColumnKind::Preprocessed,
                    expected.preprocessed_width,
                    actual.preprocessed_width,
                ),
                (NativeChildColumnKind::Main, expected.main_width, actual.main_width),
                (
                    NativeChildColumnKind::Permutation,
                    expected.permutation_width,
                    actual.permutation_width,
                ),
            ] {
                validate_metadata_width(&actual.name, kind, expected_width, actual_width)?;
            }
            if expected.constraint_count != actual.constraint_count {
                return Err(NativeChildViewError::MetadataConstraintCountMismatch {
                    chip_name: actual.name.clone(),
                    expected: expected.constraint_count,
                    actual: actual.constraint_count,
                });
            }
            if expected.commit_scope != actual.commit_scope ||
                expected.has_local_interactions != actual.has_local_interactions ||
                expected.logup_batch_size != actual.logup_batch_size ||
                expected.required_max_beta_power != actual.required_max_beta_power
            {
                return Err(NativeChildViewError::MetadataChipMissing {
                    chip_name: format!(
                        "{} static authority mismatch: expected scope={:?} local={} batch={} beta={}, actual scope={:?} local={} batch={} beta={}",
                        actual.name,
                        expected.commit_scope,
                        expected.has_local_interactions,
                        expected.logup_batch_size,
                        expected.required_max_beta_power,
                        actual.commit_scope,
                        actual.has_local_interactions,
                        actual.logup_batch_size,
                        actual.required_max_beta_power,
                    ),
                });
            }
        }
        Ok(())
    }

    /// One-time cold VK schema binding. The commitment and statement fields remain request data;
    /// only the immutable chip ordering/constraint schema is checked here.
    pub(crate) fn validate_vk<SC: SCStarkGenericConfig>(
        &self,
        vk: &SCStarkVerifyingKey<SC>,
    ) -> NativeChildViewResult<()> {
        NativeChildVkView::validate_full(vk)?;
        for chip in &self.chips {
            let actual = vk.constraints_map.get(&chip.name).copied().ok_or_else(|| {
                NativeChildViewError::VkMissingConstraintCount { chip_name: chip.name.clone() }
            })?;
            if actual != chip.constraint_count {
                return Err(NativeChildViewError::MetadataConstraintCountMismatch {
                    chip_name: chip.name.clone(),
                    expected: chip.constraint_count,
                    actual,
                });
            }
            if chip.preprocessed_width != 0 && !vk.chip_ordering.contains_key(&chip.name) {
                return Err(NativeChildViewError::ProofPreprocessedChipMissingFromVk {
                    chip_name: chip.name.clone(),
                });
            }
        }
        Ok(())
    }

    fn validate_proof<SC: SCStarkGenericConfig>(
        &self,
        proof: &NativeChildProofView<'_, SC>,
    ) -> NativeChildViewResult<NativeChildAdmissionEvents> {
        let mut events = NativeChildAdmissionEvents::default();
        proof.validate_public_values(self.num_observed_public_values)?;
        for chip in proof.ordered_chips() {
            events.bounded_name_lookups = events
                .bounded_name_lookups
                .checked_add(1)
                .expect("bounded child name lookup counter overflow");
            let metadata = self.find_chip(chip.name).ok_or_else(|| {
                NativeChildViewError::MetadataChipMissing { chip_name: chip.name.to_owned() }
            })?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Preprocessed,
                metadata.preprocessed_width,
                chip.opened_values.preprocessed.local.len(),
            )?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Main,
                metadata.main_width,
                chip.opened_values.main.local.len(),
            )?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Permutation,
                metadata.permutation_width,
                proof.permutation_dimension_width(chip.index),
            )?;
            if !metadata.has_local_interactions &&
                !chip.opened_values.local_cumulative_sum.is_zero()
            {
                return Err(NativeChildViewError::NoLocalInteractionHasLocalCumulativeSum {
                    chip_name: chip.name.to_owned(),
                });
            }
        }
        Ok(events)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWhirConfigView {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub grinding_bits_query: usize,
    pub grinding_bits_batching: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeChildVerifierConfigView {
    pub role: NativeChildRole,
    pub num_skip_rounds: usize,
    pub chip_log_height_threshold: usize,
    pub whir: NativeWhirConfigView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeVerifierRoundShape {
    pub num_rounds_linear: usize,
    pub num_rounds_nonlinear: usize,
    pub num_rounds: usize,
}

#[derive(Debug)]
pub struct NativeChildProofView<'a, SC: SCStarkGenericConfig> {
    proof: &'a SCShardProof<SC>,
    /// The proof's single validated chip order. All verifier/recorder views
    /// reuse this layout instead of sorting the hash map independently.
    ordered_chip_names: Vec<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct NativeChildVkView<'a, SC: SCStarkGenericConfig> {
    vk: &'a SCStarkVerifyingKey<SC>,
}

#[derive(Debug, Clone, Copy)]
pub struct NativePcsOpeningView<'a, SC: SCStarkGenericConfig> {
    proof: &'a SCShardProof<SC>,
    opening_point: Option<&'a [Challenge<SC>]>,
}

#[derive(Debug)]
pub struct NativeChildViews<'a, SC: SCStarkGenericConfig> {
    pub proof: NativeChildProofView<'a, SC>,
    pub vk: NativeChildVkView<'a, SC>,
    pub layout: &'a VerifiedChildLayout,
    pub verifier_config: &'a NativeChildVerifierConfigView,
    admission_events: NativeChildAdmissionEvents,
}

/// Event-backed proof-admission telemetry. The frozen-layout path has no operation capable of
/// incrementing the four forbidden static-work fields; the counts therefore express its
/// typestate, while `bounded_name_lookups` is incremented at the actual lookup site.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeChildAdmissionEvents {
    pub bounded_name_lookups: u64,
    pub vk_full_validation_calls: u64,
    pub metadata_full_rebuilds: u64,
    pub metadata_name_sorts: u64,
    pub machine_layout_second_passes: u64,
}

impl<'a, SC: SCStarkGenericConfig> NativeChildProofView<'a, SC> {
    pub fn new(proof: &'a SCShardProof<SC>) -> NativeChildViewResult<Self> {
        let ordered_chip_names = Self::validate_chip_ordering(proof)?;
        let view = Self { proof, ordered_chip_names };
        view.validate_pcs_layout()?;
        Ok(view)
    }

    pub fn proof(&self) -> &'a SCShardProof<SC> {
        self.proof
    }

    pub fn commitment(&self) -> &'a SCShardCommitment<MlCom<SC>> {
        &self.proof.commitment
    }

    pub fn opened_values(&self) -> &'a SCShardOpenedValues<Val<SC>, Challenge<SC>> {
        &self.proof.opened_values
    }

    pub fn opening_proof(&self) -> &'a MlPcsOpeningProof<SC> {
        &self.proof.opening_proof
    }

    pub fn sumcheck_proof(&self) -> &'a SumcheckProof<SC> {
        &self.proof.sumcheck_proof
    }

    pub fn dimensions(&self) -> &'a [Vec<Dimensions>] {
        &self.proof.dimensions
    }

    pub fn public_values(&self) -> &'a [Val<SC>] {
        &self.proof.public_values
    }

    pub fn has_permutation_commitment(&self) -> bool {
        self.proof.commitment.permutation_commit.is_some()
    }

    pub fn permutation_dimension_width(&self, chip_index: usize) -> usize {
        self.proof
            .dimensions
            .get(2)
            .and_then(|batch| batch.get(chip_index))
            .map(|dimension| dimension.width)
            .unwrap_or(0)
    }

    pub fn chip_count(&self) -> usize {
        self.proof.opened_values.chips.len()
    }

    pub fn ordered_chip_names(&self) -> &[&'a str] {
        &self.ordered_chip_names
    }

    pub fn ordered_chips(
        &self,
    ) -> impl ExactSizeIterator<Item = NativeOpenedChipView<'a, SC>> + '_ {
        self.ordered_chip_names.iter().copied().enumerate().map(|(index, name)| {
            NativeOpenedChipView {
                index,
                name,
                opened_values: &self.proof.opened_values.chips[index],
            }
        })
    }

    pub fn validate_public_values(&self, required: usize) -> NativeChildViewResult<()> {
        let actual = self.proof.public_values.len();
        if actual < required {
            return Err(NativeChildViewError::PublicValueLengthTooShort { required, actual });
        }
        Ok(())
    }

    pub fn verifier_round_log_height(&self) -> NativeChildViewResult<usize> {
        Ok(self.proof.opened_values.chips[0].log_height)
    }

    fn validate_chip_ordering(proof: &'a SCShardProof<SC>) -> NativeChildViewResult<Vec<&'a str>> {
        let opened_values = proof.opened_values.chips.len();
        let chip_ordering = proof.chip_ordering.len();
        if chip_ordering == 0 {
            return Err(NativeChildViewError::EmptyChipOrdering);
        }
        if chip_ordering != opened_values {
            return Err(NativeChildViewError::ChipOpeningLengthMismatch {
                chip_ordering,
                opened_values,
            });
        }
        validate_index_permutation(
            "proof.chip_ordering",
            proof.chip_ordering.iter().map(|(name, index)| (name.as_str(), *index)),
            chip_ordering,
        )?;
        let mut ordered = vec![None; chip_ordering];
        for (name, &index) in &proof.chip_ordering {
            ordered[index] = Some(name.as_str());
        }
        Ok(ordered
            .into_iter()
            .map(|name| name.expect("validated chip-order permutation is dense"))
            .collect())
    }

    fn validate_pcs_layout(&self) -> NativeChildViewResult<()> {
        let has_permutation_commitment = self.has_permutation_commitment();
        let expected_batches = if has_permutation_commitment { 3 } else { 2 };
        let actual_batches = self.proof.dimensions.len();
        if actual_batches != expected_batches {
            return Err(NativeChildViewError::PcsBatchCountMismatch {
                expected: expected_batches,
                actual: actual_batches,
                permutation_commit_present: has_permutation_commitment,
            });
        }

        let ordered_names = self.ordered_chip_names();
        let chips = &self.proof.opened_values.chips;
        let preprocessed_count =
            chips.iter().filter(|chip| !chip.preprocessed.local.is_empty()).count();

        validate_batch_len(
            NativePcsBatchKind::Preprocessed,
            preprocessed_count,
            self.proof.dimensions[0].len(),
        )?;
        validate_batch_len(NativePcsBatchKind::Main, chips.len(), self.proof.dimensions[1].len())?;

        let mut preprocessed_dim_index = 0;
        for (chip_index, (chip_name, opened)) in ordered_names.iter().zip(chips.iter()).enumerate()
        {
            let height = expected_height(*chip_name, opened.log_height)?;

            if !opened.preprocessed.local.is_empty() {
                let dim = self.proof.dimensions[0][preprocessed_dim_index];
                validate_dimension(
                    NativePcsBatchKind::Preprocessed,
                    chip_name,
                    opened.preprocessed.local.len(),
                    height,
                    dim,
                )?;
                preprocessed_dim_index += 1;
            }

            validate_dimension(
                NativePcsBatchKind::Main,
                chip_name,
                opened.main.local.len(),
                height,
                self.proof.dimensions[1][chip_index],
            )?;
        }

        if has_permutation_commitment {
            validate_batch_len(
                NativePcsBatchKind::Permutation,
                chips.len(),
                self.proof.dimensions[2].len(),
            )?;

            let mut any_permutation_opening = false;
            for (chip_index, (chip_name, opened)) in
                ordered_names.iter().zip(chips.iter()).enumerate()
            {
                let height = expected_height(chip_name, opened.log_height)?;
                any_permutation_opening |= !opened.permutation.local.is_empty();
                let permutation_width = opened.permutation.local.len() *
                    <Challenge<SC> as AbstractExtensionField<Val<SC>>>::D;
                validate_dimension(
                    NativePcsBatchKind::Permutation,
                    chip_name,
                    permutation_width,
                    height,
                    self.proof.dimensions[2][chip_index],
                )?;
            }
            if !any_permutation_opening {
                return Err(NativeChildViewError::PermutationCommitPresentButNoOpenedValues);
            }
        } else {
            for (chip_name, opened) in ordered_names.iter().zip(chips.iter()) {
                if !opened.permutation.local.is_empty() {
                    return Err(
                        NativeChildViewError::PermutationCommitMissingButOpenedValuesPresent {
                            chip_name: (*chip_name).to_owned(),
                        },
                    );
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NativeOpenedChipView<'a, SC: SCStarkGenericConfig> {
    pub index: usize,
    pub name: &'a str,
    pub opened_values: &'a SCChipOpenedValues<Val<SC>, Challenge<SC>>,
}

impl<'a, SC: SCStarkGenericConfig> NativeChildVkView<'a, SC> {
    /// Cold-start constructor retained for explicit authority tests. Production proof admission
    /// uses `from_verified_layout`, whose typestate proves the full map was not revalidated.
    pub fn new(vk: &'a SCStarkVerifyingKey<SC>) -> NativeChildViewResult<Self> {
        Self::validate_full(vk)?;
        Ok(Self { vk })
    }

    fn from_verified_layout(vk: &'a SCStarkVerifyingKey<SC>) -> Self {
        Self { vk }
    }

    pub fn vk(&self) -> &'a SCStarkVerifyingKey<SC> {
        self.vk
    }

    pub fn validate_proof(
        &self,
        proof: &NativeChildProofView<'_, SC>,
    ) -> NativeChildViewResult<()> {
        for chip in proof.ordered_chips() {
            if !self.vk.constraints_map.contains_key(chip.name) {
                return Err(NativeChildViewError::ProofChipMissingConstraintCount {
                    chip_name: chip.name.to_owned(),
                });
            }
            if !chip.opened_values.preprocessed.local.is_empty() &&
                !self.vk.chip_ordering.contains_key(chip.name)
            {
                return Err(NativeChildViewError::ProofPreprocessedChipMissingFromVk {
                    chip_name: chip.name.to_owned(),
                });
            }
        }
        Ok(())
    }

    fn validate_full(vk: &SCStarkVerifyingKey<SC>) -> NativeChildViewResult<()> {
        dt_stark::global_d11::validate_global146_identity(&vk.global146_identity)
            .map_err(|_| NativeChildViewError::Global146IdentityMismatch)?;
        let chip_ordering = vk.chip_ordering.len();
        let chip_information = vk.chip_information.len();
        if chip_ordering != chip_information {
            return Err(NativeChildViewError::VkChipInformationLengthMismatch {
                chip_ordering,
                chip_information,
            });
        }
        validate_index_permutation(
            "vk.chip_ordering",
            vk.chip_ordering.iter().map(|(name, index)| (name.as_str(), *index)),
            chip_ordering,
        )?;

        for (expected_index, (chip_name, _)) in vk.chip_information.iter().enumerate() {
            let Some(actual_index) = vk.chip_ordering.get(chip_name).copied() else {
                return Err(NativeChildViewError::VkChipInformationOrderMismatch {
                    chip_name: chip_name.clone(),
                    expected_index,
                    actual_index: usize::MAX,
                });
            };
            if expected_index != actual_index {
                return Err(NativeChildViewError::VkChipInformationOrderMismatch {
                    chip_name: chip_name.clone(),
                    expected_index,
                    actual_index,
                });
            }
            if !vk.constraints_map.contains_key(chip_name) {
                return Err(NativeChildViewError::VkMissingConstraintCount {
                    chip_name: chip_name.clone(),
                });
            }
        }
        Ok(())
    }
}

impl NativeChildMetadataView<'_> {
    pub fn validate<SC: SCStarkGenericConfig>(
        &self,
        proof: &NativeChildProofView<'_, SC>,
        vk: &NativeChildVkView<'_, SC>,
    ) -> NativeChildViewResult<()> {
        proof.validate_public_values(self.num_observed_public_values)?;
        vk.validate_proof(proof)?;
        self.validate_unique_chip_names()?;

        for chip in proof.ordered_chips() {
            let metadata = self.find_chip(chip.name).ok_or_else(|| {
                NativeChildViewError::MetadataChipMissing { chip_name: chip.name.to_owned() }
            })?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Preprocessed,
                metadata.preprocessed_width,
                chip.opened_values.preprocessed.local.len(),
            )?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Main,
                metadata.main_width,
                chip.opened_values.main.local.len(),
            )?;
            validate_metadata_width(
                chip.name,
                NativeChildColumnKind::Permutation,
                metadata.permutation_width,
                proof.permutation_dimension_width(chip.index),
            )?;

            let vk_constraint_count =
                vk.vk.constraints_map.get(chip.name).copied().ok_or_else(|| {
                    NativeChildViewError::ProofChipMissingConstraintCount {
                        chip_name: chip.name.to_owned(),
                    }
                })?;
            if vk_constraint_count != metadata.constraint_count {
                return Err(NativeChildViewError::MetadataConstraintCountMismatch {
                    chip_name: chip.name.to_owned(),
                    expected: metadata.constraint_count,
                    actual: vk_constraint_count,
                });
            }

            if !metadata.has_local_interactions &&
                !chip.opened_values.local_cumulative_sum.is_zero()
            {
                return Err(NativeChildViewError::NoLocalInteractionHasLocalCumulativeSum {
                    chip_name: chip.name.to_owned(),
                });
            }
        }

        Ok(())
    }

    pub fn find_chip(&self, name: &str) -> Option<&NativeChipMetadata> {
        self.chips.iter().find(|chip| chip.name == name)
    }

    fn validate_unique_chip_names(&self) -> NativeChildViewResult<()> {
        let mut names = self.chips.iter().map(|chip| chip.name.as_str()).collect::<Vec<_>>();
        names.sort_unstable();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                return Err(NativeChildViewError::DuplicateMetadataChip {
                    chip_name: pair[0].to_owned(),
                });
            }
        }
        Ok(())
    }
}

impl NativeChildVerifierConfigView {
    pub fn round_shape(
        &self,
        verifier_log_height: usize,
    ) -> NativeChildViewResult<NativeVerifierRoundShape> {
        if self.num_skip_rounds == 0 {
            return Err(NativeChildViewError::InvalidNumSkipRounds);
        }
        if self.chip_log_height_threshold % self.num_skip_rounds != 0 {
            return Err(NativeChildViewError::ChipLogHeightThresholdNotDivisible {
                chip_log_height_threshold: self.chip_log_height_threshold,
                num_skip_rounds: self.num_skip_rounds,
            });
        }

        let num_rounds_linear = verifier_log_height.saturating_sub(self.chip_log_height_threshold);
        let num_rounds_nonlinear =
            verifier_log_height.min(self.chip_log_height_threshold) / self.num_skip_rounds;
        Ok(NativeVerifierRoundShape {
            num_rounds_linear,
            num_rounds_nonlinear,
            num_rounds: num_rounds_linear + num_rounds_nonlinear,
        })
    }

    pub fn validate_first_sound_path(
        &self,
        verifier_log_height: usize,
    ) -> NativeChildViewResult<()> {
        let round_shape = self.round_shape(verifier_log_height)?;
        // NATIVE_REC_TODO_DELETE: remove after nonlinear skip-round sumcheck and terminal eq
        // support.
        if self.chip_log_height_threshold != 0 || round_shape.num_rounds_nonlinear != 0 {
            return Err(NativeChildViewError::UnsupportedNonlinearRounds {
                chip_log_height_threshold: self.chip_log_height_threshold,
                num_rounds_nonlinear: round_shape.num_rounds_nonlinear,
            });
        }
        if verifier_log_height.saturating_add(self.whir.log_blowup) > KOALABEAR_MAX_TRACE_LOG_HEIGHT
        {
            return Err(NativeChildViewError::KoalaBearTraceLogHeightExceeded {
                max_trace_log_height: KOALABEAR_MAX_TRACE_LOG_HEIGHT,
                log_height: verifier_log_height,
                log_blowup: self.whir.log_blowup,
            });
        }
        Ok(())
    }
}

impl<'a, SC: SCStarkGenericConfig> NativePcsOpeningView<'a, SC> {
    pub fn layout_only(proof: &NativeChildProofView<'a, SC>) -> Self {
        Self { proof: proof.proof, opening_point: None }
    }

    pub fn with_transcript_bound_opening_point(
        proof: &NativeChildProofView<'a, SC>,
        opening_point: &'a [Challenge<SC>],
    ) -> Self {
        Self { proof: proof.proof, opening_point: Some(opening_point) }
    }

    pub fn opening_point(&self) -> NativeChildViewResult<&'a [Challenge<SC>]> {
        self.opening_point.ok_or(NativeChildViewError::MissingTranscriptBoundOpeningPoint)
    }

    pub fn opening_proof(&self) -> &'a MlPcsOpeningProof<SC> {
        &self.proof.opening_proof
    }

    pub fn dimensions(&self) -> &'a [Vec<Dimensions>] {
        &self.proof.dimensions
    }

    pub fn commitment(&self) -> &'a SCShardCommitment<MlCom<SC>> {
        &self.proof.commitment
    }

    pub fn dimension_batches(&self) -> Vec<NativePcsBatchView<'a>> {
        let mut batches = Vec::with_capacity(self.proof.dimensions.len());
        batches.push(NativePcsBatchView {
            kind: NativePcsBatchKind::Preprocessed,
            dimensions: &self.proof.dimensions[0],
        });
        batches.push(NativePcsBatchView {
            kind: NativePcsBatchKind::Main,
            dimensions: &self.proof.dimensions[1],
        });
        if self.proof.commitment.permutation_commit.is_some() {
            batches.push(NativePcsBatchView {
                kind: NativePcsBatchKind::Permutation,
                dimensions: &self.proof.dimensions[2],
            });
        }
        batches
    }
}

impl<'a, SC: SCStarkGenericConfig> NativeChildViews<'a, SC> {
    pub fn new(
        proof: &'a SCShardProof<SC>,
        vk: &'a SCStarkVerifyingKey<SC>,
        layout: &'a VerifiedChildLayout,
        verifier_config: &'a NativeChildVerifierConfigView,
    ) -> NativeChildViewResult<Self> {
        Self::from_proof_view(NativeChildProofView::new(proof)?, vk, layout, verifier_config)
    }

    pub fn from_proof_view(
        proof: NativeChildProofView<'a, SC>,
        vk: &'a SCStarkVerifyingKey<SC>,
        layout: &'a VerifiedChildLayout,
        verifier_config: &'a NativeChildVerifierConfigView,
    ) -> NativeChildViewResult<Self> {
        if layout.role != verifier_config.role {
            return Err(NativeChildViewError::RoleMismatch {
                metadata_role: layout.role,
                verifier_config_role: verifier_config.role,
            });
        }
        // Full VK/program/machine binding is a cold-start operation. The only per-proof work here
        // is bounded validation of dynamic containers and present-chip metadata against the
        // immutable layout.
        let vk = NativeChildVkView::from_verified_layout(vk);
        let admission_events = layout.validate_proof(&proof)?;
        let verifier_log_height = proof.verifier_round_log_height()?;
        verifier_config.validate_first_sound_path(verifier_log_height)?;
        Ok(Self { proof, vk, layout, verifier_config, admission_events })
    }

    pub fn pcs_opening_layout(&self) -> NativePcsOpeningView<'a, SC> {
        NativePcsOpeningView::layout_only(&self.proof)
    }

    pub const fn admission_events(&self) -> NativeChildAdmissionEvents {
        self.admission_events
    }
}

fn validate_index_permutation<'a>(
    source: &'static str,
    entries: impl IntoIterator<Item = (&'a str, usize)>,
    len: usize,
) -> NativeChildViewResult<()> {
    let mut seen: Vec<Option<&'a str>> = vec![None; len];
    for (chip_name, index) in entries {
        if index >= len {
            return Err(NativeChildViewError::ChipOrderingIndexOutOfRange {
                source,
                chip_name: chip_name.to_owned(),
                index,
                len,
            });
        }
        if let Some(first_chip_name) = seen[index] {
            return Err(NativeChildViewError::ChipOrderingDuplicateIndex {
                source,
                first_chip_name: first_chip_name.to_owned(),
                second_chip_name: chip_name.to_owned(),
                index,
            });
        }
        seen[index] = Some(chip_name);
    }
    Ok(())
}

fn expected_height(chip_name: &str, log_height: usize) -> NativeChildViewResult<usize> {
    if log_height > KOALABEAR_MAX_TRACE_LOG_HEIGHT {
        return Err(NativeChildViewError::LogHeightTooLarge {
            chip_name: chip_name.to_owned(),
            log_height,
        });
    }
    let shift = u32::try_from(log_height).map_err(|_| NativeChildViewError::LogHeightTooLarge {
        chip_name: chip_name.to_owned(),
        log_height,
    })?;
    1usize.checked_shl(shift).ok_or_else(|| NativeChildViewError::LogHeightTooLarge {
        chip_name: chip_name.to_owned(),
        log_height,
    })
}

fn validate_batch_len(
    batch: NativePcsBatchKind,
    expected: usize,
    actual: usize,
) -> NativeChildViewResult<()> {
    if expected != actual {
        return Err(NativeChildViewError::PcsDimensionLengthMismatch { batch, expected, actual });
    }
    Ok(())
}

fn validate_dimension(
    batch: NativePcsBatchKind,
    chip_name: &str,
    expected_width: usize,
    expected_height: usize,
    actual: Dimensions,
) -> NativeChildViewResult<()> {
    if actual.width != expected_width {
        return Err(NativeChildViewError::DimensionWidthMismatch {
            batch,
            chip_name: chip_name.to_owned(),
            expected: expected_width,
            actual: actual.width,
        });
    }
    if actual.height != expected_height {
        return Err(NativeChildViewError::DimensionHeightMismatch {
            batch,
            chip_name: chip_name.to_owned(),
            expected: expected_height,
            actual: actual.height,
        });
    }
    Ok(())
}

fn validate_metadata_width(
    chip_name: &str,
    column_kind: NativeChildColumnKind,
    expected: usize,
    actual: usize,
) -> NativeChildViewResult<()> {
    if expected != actual {
        return Err(NativeChildViewError::MetadataWidthMismatch {
            chip_name: chip_name.to_owned(),
            column_kind,
            expected,
            actual,
        });
    }
    Ok(())
}
