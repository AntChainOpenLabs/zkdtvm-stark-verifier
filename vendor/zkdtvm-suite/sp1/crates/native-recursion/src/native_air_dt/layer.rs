use std::collections::{BTreeMap, BTreeSet};

use dt_stark::{
    air::{FullAir, InteractionScope, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;

use crate::{
    batch_constraint_dt::{BatchSumcheckAir, BatchTranscriptInputsAir},
    config::{RootSC, D_EF, F, SC},
    constraint_replay_dt::{
        ConstraintBetaLadderAir, ConstraintBoundaryAir, ConstraintChallengeAir,
        ConstraintDagEvalAir, ConstraintFoldAir, ConstraintProgramTableAir,
        ConstraintRootTableAir, ConstraintTerminalAir,
    },
    machine_dt::{NativeRecursionAssemblyError, NativeRecursionAssemblyResult},
    proof_shape_dt::{NativeChipMetadataAir, ProofShapeBinderAir},
    statement_boundary_air_dt::StatementBoundaryAir,
    statement_config_air_dt::StatementConfigAir,
    statement_dt::{
        NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
        STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
    },
    statement_hash_air_dt::{StatementDigestMode, StatementHashAir},
    symbolic_expr_fixed_dt::RecursionChildRole,
    system_dt::{
        RecordingStage, RecursionNativeProgram, RecursionRecord, RecursionStatementRole,
        StatementConfigRow,
    },
    validate::NativeValidateConfig,
    whir_dt::{whir_role_config, WhirBatchEvalAir, WhirRoundAir},
};

use super::{
    NativeAirFamily, NativeChildClass, NativeFinalReplayLayout, NativeProofConfigClass,
    NativeRecursionLayer,
};

const L1_ACCEPTED_CHILDREN: [NativeChildClass; 1] = [NativeChildClass::CoreShard];
const L23_ACCEPTED_CHILDREN: [NativeChildClass; 2] = [NativeChildClass::Lift, NativeChildClass::L2];
const L4_ACCEPTED_CHILDREN: [NativeChildClass; 1] = [NativeChildClass::L3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeLayerParams {
    pub layer: NativeRecursionLayer,
    pub child_role: RecursionChildRole,
    pub statement_role: RecursionStatementRole,
    pub num_child_public_values: usize,
    pub child_contains_global_bus: bool,
    pub accepted_child_classes: &'static [NativeChildClass],
    pub proof_config_class: NativeProofConfigClass,
    pub final_replay_layout: NativeFinalReplayLayout,
}

/// The 17 layer-specific AIR families with a distinct logical identity at each native
/// recursion layer. `ConstraintChallenge` is symbolic-sensitive even though its
/// accepted wire-order slot remains adjacent to the program-sensitive block.
#[derive(Debug, Clone)]
pub enum NativeLayerAirKind {
    // Symbolic-sensitive families.
    ProofShapeBinder(ProofShapeBinderAir),
    BatchTranscriptInputs(BatchTranscriptInputsAir),
    BatchSumcheck(BatchSumcheckAir),
    WhirRound(WhirRoundAir),
    WhirBatchEval(WhirBatchEvalAir),
    ConstraintTerminal(ConstraintTerminalAir),
    ConstraintBoundary(ConstraintBoundaryAir),
    StatementBoundary(StatementBoundaryAir),
    StatementHash(StatementHashAir),

    // Program-sensitive families, plus ConstraintChallenge in its accepted wire-order slot.
    NativeChipMetadata(NativeChipMetadataAir),
    ConstraintProgramTable(ConstraintProgramTableAir),
    ConstraintRootTable(ConstraintRootTableAir),
    ConstraintDagEval(ConstraintDagEvalAir),
    ConstraintFold(ConstraintFoldAir),
    ConstraintBetaLadder(ConstraintBetaLadderAir),
    ConstraintChallenge(ConstraintChallengeAir),
    StatementConfig(StatementConfigAir),
}

impl NativeLayerAirKind {
    pub fn all(
        program: &RecursionNativeProgram<F>,
        params: &NativeLayerParams,
    ) -> NativeRecursionAssemblyResult<Vec<Self>> {
        validate_program_matches_layer(program, params)?;
        let constraint_program = program.constraint_program.clone();
        let role_config = whir_role_config(child_role_id(params.child_role));
        Ok(vec![
            Self::ProofShapeBinder(ProofShapeBinderAir::new(
                params.num_child_public_values,
                role_config,
                params.child_contains_global_bus,
            )),
            Self::BatchTranscriptInputs(BatchTranscriptInputsAir::new(
                params.num_child_public_values,
                params.child_contains_global_bus,
            )),
            Self::BatchSumcheck(BatchSumcheckAir::new(
                params.num_child_public_values,
                params.child_contains_global_bus,
            )),
            Self::WhirRound(WhirRoundAir::new(role_config, params.num_child_public_values)),
            Self::WhirBatchEval(WhirBatchEvalAir::new(role_config)),
            Self::ConstraintTerminal(ConstraintTerminalAir::new(
                constraint_program.clone(),
                params.num_child_public_values,
                params.child_contains_global_bus,
            )),
            Self::ConstraintBoundary(ConstraintBoundaryAir::new(
                constraint_program.clone(),
                params.child_contains_global_bus,
            )),
            Self::StatementBoundary(StatementBoundaryAir::new(
                params.statement_role,
                params.num_child_public_values,
                program.statement_config.clone(),
            )),
            Self::StatementHash(StatementHashAir::for_child(
                StatementDigestMode::from_role(params.statement_role),
                params.num_child_public_values,
            )),
            Self::NativeChipMetadata(NativeChipMetadataAir::new(
                program.native_chip_metadata.clone(),
            )),
            Self::ConstraintProgramTable(ConstraintProgramTableAir::new(
                constraint_program.clone(),
            )),
            Self::ConstraintRootTable(ConstraintRootTableAir::new(constraint_program.clone())),
            Self::ConstraintDagEval(ConstraintDagEvalAir::new(constraint_program.clone())),
            Self::ConstraintFold(ConstraintFoldAir::new(constraint_program.clone())),
            Self::ConstraintBetaLadder(ConstraintBetaLadderAir::new(constraint_program.clone())),
            Self::ConstraintChallenge(ConstraintChallengeAir::new(
                constraint_program,
                params.num_child_public_values,
                params.child_contains_global_bus,
            )),
            Self::StatementConfig(StatementConfigAir::new(program.statement_config.clone())),
        ])
    }

    pub fn family(&self) -> NativeAirFamily {
        match self {
            Self::ProofShapeBinder(_) => NativeAirFamily::ProofShapeBinder,
            Self::BatchTranscriptInputs(_) => NativeAirFamily::BatchTranscriptInputs,
            Self::BatchSumcheck(_) => NativeAirFamily::BatchSumcheck,
            Self::WhirRound(_) => NativeAirFamily::WhirRound,
            Self::WhirBatchEval(_) => NativeAirFamily::WhirBatchEval,
            Self::ConstraintTerminal(_) => NativeAirFamily::ConstraintTerminal,
            Self::ConstraintBoundary(_) => NativeAirFamily::ConstraintBoundary,
            Self::StatementBoundary(_) => NativeAirFamily::Statement,
            Self::StatementHash(_) => NativeAirFamily::StatementHash,
            Self::NativeChipMetadata(_) => NativeAirFamily::NativeChipMetadata,
            Self::ConstraintProgramTable(_) => NativeAirFamily::ConstraintProgramTable,
            Self::ConstraintRootTable(_) => NativeAirFamily::ConstraintRootTable,
            Self::ConstraintDagEval(_) => NativeAirFamily::ConstraintDagEval,
            Self::ConstraintFold(_) => NativeAirFamily::ConstraintFold,
            Self::ConstraintBetaLadder(_) => NativeAirFamily::ConstraintBetaLadder,
            Self::ConstraintChallenge(_) => NativeAirFamily::ConstraintChallenge,
            Self::StatementConfig(_) => NativeAirFamily::StatementConfig,
        }
    }
}

fn child_role_id(role: RecursionChildRole) -> usize {
    match role {
        RecursionChildRole::Core => 0,
        RecursionChildRole::Compress => 1,
        RecursionChildRole::Shrink => 2,
    }
}

macro_rules! dispatch_layer_air {
    ($self:expr, $air:ident => $body:expr) => {
        match $self {
            NativeLayerAirKind::ProofShapeBinder($air) => $body,
            NativeLayerAirKind::BatchTranscriptInputs($air) => $body,
            NativeLayerAirKind::BatchSumcheck($air) => $body,
            NativeLayerAirKind::WhirRound($air) => $body,
            NativeLayerAirKind::WhirBatchEval($air) => $body,
            NativeLayerAirKind::ConstraintTerminal($air) => $body,
            NativeLayerAirKind::ConstraintBoundary($air) => $body,
            NativeLayerAirKind::StatementBoundary($air) => $body,
            NativeLayerAirKind::StatementHash($air) => $body,
            NativeLayerAirKind::NativeChipMetadata($air) => $body,
            NativeLayerAirKind::ConstraintProgramTable($air) => $body,
            NativeLayerAirKind::ConstraintRootTable($air) => $body,
            NativeLayerAirKind::ConstraintDagEval($air) => $body,
            NativeLayerAirKind::ConstraintFold($air) => $body,
            NativeLayerAirKind::ConstraintBetaLadder($air) => $body,
            NativeLayerAirKind::ConstraintChallenge($air) => $body,
            NativeLayerAirKind::StatementConfig($air) => $body,
        }
    };
}

impl BaseAir<F> for NativeLayerAirKind {
    fn width(&self) -> usize {
        dispatch_layer_air!(self, air => BaseAir::<F>::width(air))
    }
}

impl<AB> FullAir<AB> for NativeLayerAirKind
where
    AB: dt_stark::air::FullAirBuilder<F = F>,
{
    fn width(&self) -> usize {
        dispatch_layer_air!(self, air => FullAir::<AB>::width(air))
    }

    fn num_public_values(&self) -> usize {
        dispatch_layer_air!(self, air => FullAir::<AB>::num_public_values(air))
    }

    fn required_max_beta_power(&self) -> usize {
        dispatch_layer_air!(self, air => FullAir::<AB>::required_max_beta_power(air))
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        dispatch_layer_air!(self, air => FullAir::<AB>::reserved_poly(air))
    }

    fn precompute_lc(&self, builder: &mut AB) {
        dispatch_layer_air!(self, air => FullAir::<AB>::precompute_lc(air, builder))
    }

    fn eval(&self, builder: &mut AB) {
        dispatch_layer_air!(self, air => FullAir::<AB>::eval(air, builder))
    }

    fn lookup(&self, builder: &mut AB) {
        dispatch_layer_air!(self, air => FullAir::<AB>::lookup(air, builder))
    }

    fn global(&self) -> bool {
        dispatch_layer_air!(self, air => FullAir::<AB>::global(air))
    }
}

impl MachineAir<F> for NativeLayerAirKind {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        dispatch_layer_air!(self, air => MachineAir::<F>::name(air))
    }

    fn preprocessed_width(&self) -> usize {
        dispatch_layer_air!(self, air => MachineAir::<F>::preprocessed_width(air))
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        dispatch_layer_air!(self, air =>
            MachineAir::<F>::preprocessed_num_rows(air, program, instrs_len)
        )
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        dispatch_layer_air!(self, air =>
            MachineAir::<F>::generate_preprocessed_trace(air, program)
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        dispatch_layer_air!(self, air => MachineAir::<F>::num_rows(air, input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        dispatch_layer_air!(self, air => MachineAir::<F>::generate_trace(air, input, output))
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        dispatch_layer_air!(self, air =>
            MachineAir::<F>::generate_dependencies(air, input, output)
        )
    }

    fn included(&self, record: &Self::Record) -> bool {
        dispatch_layer_air!(self, air => MachineAir::<F>::included(air, record))
    }

    fn commit_scope(&self) -> InteractionScope {
        dispatch_layer_air!(self, air => MachineAir::<F>::commit_scope(air))
    }

    fn local_only(&self) -> bool {
        dispatch_layer_air!(self, air => MachineAir::<F>::local_only(air))
    }

    fn padding_row(&self) -> Vec<F> {
        dispatch_layer_air!(self, air => MachineAir::<F>::padding_row(air))
    }
}

const LAYER_PARAMS: [NativeLayerParams; 4] = [
    NativeLayerParams {
        layer: NativeRecursionLayer::L1Lift,
        child_role: RecursionChildRole::Core,
        statement_role: RecursionStatementRole::Lift,
        num_child_public_values: dt_stark::air::DT_PROOF_NUM_PV_ELTS,
        child_contains_global_bus: true,
        accepted_child_classes: &L1_ACCEPTED_CHILDREN,
        proof_config_class: NativeProofConfigClass::Compress,
        final_replay_layout: NativeFinalReplayLayout::SingleBase0,
    },
    NativeLayerParams {
        layer: NativeRecursionLayer::L2Reduce,
        child_role: RecursionChildRole::Compress,
        statement_role: RecursionStatementRole::ReduceL2,
        num_child_public_values: NATIVE_RECURSION_NUM_PV_ELTS,
        child_contains_global_bus: false,
        accepted_child_classes: &L23_ACCEPTED_CHILDREN,
        proof_config_class: NativeProofConfigClass::Compress,
        final_replay_layout: NativeFinalReplayLayout::DualBase0Base128,
    },
    NativeLayerParams {
        layer: NativeRecursionLayer::L3Reduce,
        child_role: RecursionChildRole::Compress,
        statement_role: RecursionStatementRole::ReduceL3,
        num_child_public_values: NATIVE_RECURSION_NUM_PV_ELTS,
        child_contains_global_bus: false,
        accepted_child_classes: &L23_ACCEPTED_CHILDREN,
        proof_config_class: NativeProofConfigClass::Shrink,
        final_replay_layout: NativeFinalReplayLayout::DualBase0Base128,
    },
    NativeLayerParams {
        layer: NativeRecursionLayer::L4Root,
        child_role: RecursionChildRole::Shrink,
        statement_role: RecursionStatementRole::RootShrink,
        num_child_public_values: NATIVE_RECURSION_NUM_PV_ELTS,
        child_contains_global_bus: false,
        accepted_child_classes: &L4_ACCEPTED_CHILDREN,
        proof_config_class: NativeProofConfigClass::RootShrink,
        final_replay_layout: NativeFinalReplayLayout::SingleBase0,
    },
];

impl NativeRecursionLayer {
    pub const fn params(self) -> &'static NativeLayerParams {
        match self {
            Self::L1Lift => &LAYER_PARAMS[0],
            Self::L2Reduce => &LAYER_PARAMS[1],
            Self::L3Reduce => &LAYER_PARAMS[2],
            Self::L4Root => &LAYER_PARAMS[3],
        }
    }
}

pub fn validate_statement_config(
    statement_role: RecursionStatementRole,
    statement_config: &[StatementConfigRow],
) -> NativeRecursionAssemblyResult<()> {
    let valid = match statement_role {
        RecursionStatementRole::Lift => statement_config.is_empty(),
        RecursionStatementRole::ReduceL2 => {
            statement_config.len() == 1 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_LIFT
        }
        RecursionStatementRole::ReduceL3 => {
            statement_config.len() == 2 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_LIFT &&
                statement_config[1].class_id == STATEMENT_CONFIG_CLASS_BAKED_L2
        }
        RecursionStatementRole::RootShrink => {
            statement_config.len() == 1 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_L3
        }
    };
    if !valid {
        return Err(invalid_program(format!(
            "invalid StatementConfig classes for {statement_role:?}: {:?}",
            statement_config.iter().map(|row| row.class_id).collect::<Vec<_>>()
        )));
    }
    Ok(())
}

pub fn validate_program_matches_layer<Fld>(
    program: &RecursionNativeProgram<Fld>,
    params: &NativeLayerParams,
) -> NativeRecursionAssemblyResult<()> {
    let actual_layer = program.layer()?;
    if actual_layer != params.layer ||
        program.role != params.child_role ||
        program.statement_role != params.statement_role ||
        program.num_child_public_values != params.num_child_public_values ||
        program.child_contains_global_bus != params.child_contains_global_bus ||
        program.constraint_program.role != params.child_role
    {
        return Err(invalid_program(format!(
            "program does not match {:?}: layer={actual_layer:?} role={:?} statement={:?} num_child_public_values={} child_contains_global_bus={} constraint_role={:?}",
            params.layer,
            program.role,
            program.statement_role,
            program.num_child_public_values,
            program.child_contains_global_bus,
            program.constraint_program.role,
        )));
    }
    validate_statement_config(params.statement_role, &program.statement_config)
}

pub trait NativeLayerProofConfig: NativeValidateConfig {
    fn native_proof_config_class(&self) -> NativeRecursionAssemblyResult<NativeProofConfigClass>;
}

impl NativeLayerProofConfig for SC {
    fn native_proof_config_class(&self) -> NativeRecursionAssemblyResult<NativeProofConfigClass> {
        match self.whir_stage_name() {
            "compress" => Ok(NativeProofConfigClass::Compress),
            "shrink" => Ok(NativeProofConfigClass::Shrink),
            stage => Err(NativeRecursionAssemblyError::Validation(format!(
                "unsupported native recursion Poseidon2 proof-config stage {stage:?}"
            ))),
        }
    }
}

impl NativeLayerProofConfig for RootSC {
    fn native_proof_config_class(&self) -> NativeRecursionAssemblyResult<NativeProofConfigClass> {
        Ok(NativeProofConfigClass::RootShrink)
    }
}

pub fn validate_proof_config_for_layer<C: NativeLayerProofConfig>(
    config: &C,
    params: &NativeLayerParams,
) -> NativeRecursionAssemblyResult<()> {
    let actual = config.native_proof_config_class()?;
    if actual != params.proof_config_class {
        return Err(NativeRecursionAssemblyError::Validation(format!(
            "proof config {actual:?} does not match {:?}, which requires {:?}",
            params.layer, params.proof_config_class
        )));
    }
    Ok(())
}

pub fn validate_recording_stage_for_layer(
    stage: RecordingStage,
    params: &NativeLayerParams,
) -> NativeRecursionAssemblyResult<()> {
    let actual = match stage {
        RecordingStage::Compress => NativeProofConfigClass::Compress,
        RecordingStage::Shrink => NativeProofConfigClass::Shrink,
        RecordingStage::Core => {
            return Err(invalid_program(
                "native recording machines record Compress or Shrink proofs, not Core",
            ));
        }
    };
    if actual != params.proof_config_class {
        return Err(NativeRecursionAssemblyError::Validation(format!(
            "recording stage {stage:?} does not match {:?}, which requires {:?}",
            params.layer, params.proof_config_class
        )));
    }
    Ok(())
}

pub fn validate_final_replay_layout<Fld>(
    program: &RecursionNativeProgram<Fld>,
) -> NativeRecursionAssemblyResult<()> {
    let layer = program.layer()?;
    validate_program_matches_layer(program, layer.params())?;
    let expected_bases: &[usize] = match layer.params().final_replay_layout {
        NativeFinalReplayLayout::SingleBase0 => &[0],
        NativeFinalReplayLayout::DualBase0Base128 => &[0, 128],
    };
    validate_replay_segments(program, expected_bases)
}

pub fn validate_l2_bootstrap_layout<Fld>(
    program: &RecursionNativeProgram<Fld>,
) -> NativeRecursionAssemblyResult<()> {
    if program.layer()? != NativeRecursionLayer::L2Reduce {
        return Err(invalid_program(
            "the single-segment bootstrap layout is valid only for an L2-shaped program",
        ));
    }
    validate_program_matches_layer(program, NativeRecursionLayer::L2Reduce.params())?;
    validate_replay_segments(program, &[0])
}

fn validate_replay_segments<Fld>(
    program: &RecursionNativeProgram<Fld>,
    expected_bases: &[usize],
) -> NativeRecursionAssemblyResult<()> {
    const NATIVE_CHIPS_PER_SEGMENT: usize = NativeAirFamily::ALL.len();

    let chips = &program.constraint_program.chips;
    if chips.is_empty() {
        return Err(invalid_program("final constraint-program universe is empty"));
    }
    if !chips.windows(2).all(|pair| pair[0].static_chip_id < pair[1].static_chip_id) {
        return Err(invalid_program(
            "constraint-program chips must be strictly ordered by static_chip_id",
        ));
    }

    let mut program_ids_by_base = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut names_by_base = BTreeMap::<usize, BTreeSet<&str>>::new();
    for chip in chips {
        let base = replay_segment_base(chip.static_chip_id)?;
        if !names_by_base.entry(base).or_default().insert(&chip.chip_name) {
            return Err(invalid_program(format!(
                "duplicate chip name {:?} in replay segment {base}",
                chip.chip_name
            )));
        }
        program_ids_by_base.entry(base).or_default().insert(chip.static_chip_id);
    }

    let actual_bases = program_ids_by_base.keys().copied().collect::<Vec<_>>();
    if actual_bases != expected_bases {
        return Err(invalid_program(format!(
            "replay segments {actual_bases:?} do not match required segments {expected_bases:?}"
        )));
    }
    let layer = program.layer()?;
    for (&base, ids) in &program_ids_by_base {
        let segment_len = if layer == NativeRecursionLayer::L1Lift {
            ids.len()
        } else {
            NATIVE_CHIPS_PER_SEGMENT
        };
        let expected = (base..base + segment_len).collect::<BTreeSet<_>>();
        if *ids != expected {
            return Err(invalid_program(format!(
                "replay segment {base} is partial or malformed: ids={ids:?}"
            )));
        }
    }

    let expected_role_id = match program.constraint_program.role {
        RecursionChildRole::Core => 0,
        RecursionChildRole::Compress => 1,
        RecursionChildRole::Shrink => 2,
    };
    let chips_by_id =
        chips.iter().map(|chip| (chip.static_chip_id, chip)).collect::<BTreeMap<_, _>>();
    let mut metadata_ids = BTreeSet::new();
    for metadata in &program.native_chip_metadata {
        replay_segment_base(metadata.chip_id)?;
        if !metadata_ids.insert(metadata.chip_id) {
            return Err(invalid_program(format!(
                "duplicate native metadata static chip id {}",
                metadata.chip_id
            )));
        }
        if metadata.role_id != expected_role_id {
            return Err(invalid_program(format!(
                "native metadata chip {} has role_id {}, expected {}",
                metadata.chip_id, metadata.role_id, expected_role_id
            )));
        }
        let chip = chips_by_id.get(&metadata.chip_id).ok_or_else(|| {
            invalid_program(format!(
                "native metadata chip {} has no constraint-program entry",
                metadata.chip_id
            ))
        })?;
        if chip.logup_batch_size == 0 {
            return Err(invalid_program(format!(
                "constraint-program chip {} has zero logup batch size",
                metadata.chip_id
            )));
        }
        let expected_perm_width =
            chip.lookup_multiplicity_roots.len().div_ceil(chip.logup_batch_size) * D_EF;
        if metadata.prep_width != chip.widths.preprocessed ||
            metadata.main_width != chip.widths.main ||
            metadata.perm_width != expected_perm_width ||
            metadata.constraint_count != chip.num_constraints_from_builder
        {
            return Err(invalid_program(format!(
                "native metadata and constraint-program entry disagree for static chip id {}",
                metadata.chip_id
            )));
        }
    }
    let program_ids = chips_by_id.keys().copied().collect::<BTreeSet<_>>();
    if metadata_ids != program_ids {
        return Err(invalid_program(format!(
            "native metadata/program child universes differ: metadata={metadata_ids:?} program={program_ids:?}"
        )));
    }
    Ok(())
}

fn replay_segment_base(static_chip_id: usize) -> NativeRecursionAssemblyResult<usize> {
    let base = static_chip_id & !127;
    if !matches!(base, 0 | 128) {
        return Err(invalid_program(format!(
            "static chip id {static_chip_id} is outside the supported base-0/base-128 layout"
        )));
    }
    Ok(base)
}

fn invalid_program(message: impl Into<String>) -> NativeRecursionAssemblyError {
    NativeRecursionAssemblyError::InvalidProgram(message.into())
}

#[cfg(test)]
mod tests {
    use dt_stark::air::InteractionScope;
    use p3_field::AbstractField;

    use crate::{
        config::{DIGEST_SIZE, F},
        symbolic_expr_adapter_dt::RecursionOpMix,
        symbolic_ir_dt::{
            RecursionD0CostLedger, RecursionPolyAirChipIr, RecursionPolyAirDerivedRoot,
            RecursionPolyAirVerifierProgram, RecursionPolyAirWidths,
        },
        system_dt::RecursionNativeChipMetadataRequest,
    };

    use super::*;

    fn statement_config(layer: NativeRecursionLayer) -> Vec<StatementConfigRow> {
        let row = |class_id| StatementConfigRow { class_id, digest: [F::zero(); DIGEST_SIZE] };
        match layer {
            NativeRecursionLayer::L1Lift => Vec::new(),
            NativeRecursionLayer::L2Reduce => vec![row(STATEMENT_CONFIG_CLASS_BAKED_LIFT)],
            NativeRecursionLayer::L3Reduce => {
                vec![row(STATEMENT_CONFIG_CLASS_BAKED_LIFT), row(STATEMENT_CONFIG_CLASS_BAKED_L2)]
            }
            NativeRecursionLayer::L4Root => vec![row(STATEMENT_CONFIG_CLASS_BAKED_L3)],
        }
    }

    fn empty_cost_ledger() -> RecursionD0CostLedger {
        RecursionD0CostLedger {
            node_count: 0,
            op_mix: RecursionOpMix::default(),
            gate_count: 0,
            precompute_root_count: 0,
            derived_root_count: 2,
            expected_node_bus_rows: 0,
            expected_wide_unroll_rows: 1,
            expected_wide_unroll_width: 0,
            internal_recursion_interactions_node_bus: 0,
            internal_recursion_interactions_wide_unroll: 0,
        }
    }

    fn test_program(
        layer: NativeRecursionLayer,
        segment_bases: &[usize],
    ) -> RecursionNativeProgram<F> {
        let params = layer.params();
        let role_id = match params.child_role {
            RecursionChildRole::Core => 0,
            RecursionChildRole::Compress => 1,
            RecursionChildRole::Shrink => 2,
        };
        let mut chips = Vec::new();
        let mut metadata = Vec::new();
        for &base in segment_bases {
            for local_id in 0..NativeAirFamily::ALL.len() {
                let static_chip_id = base + local_id;
                chips.push(RecursionPolyAirChipIr {
                    static_chip_id,
                    chip_name: format!("TestFamily{local_id}"),
                    widths: RecursionPolyAirWidths {
                        preprocessed: 1,
                        main: 2,
                        public: params.num_child_public_values,
                    },
                    commit_scope: InteractionScope::Local,
                    logup_batch_size: 2,
                    reserved_poly: Vec::new(),
                    derived_roots: vec![
                        RecursionPolyAirDerivedRoot::BetaPower { power: 0 },
                        RecursionPolyAirDerivedRoot::BetaSeptix,
                    ],
                    gate_roots: Vec::new(),
                    lookup_multiplicity_roots: Vec::new(),
                    node_table: Vec::new(),
                    num_constraints_from_builder: 0,
                    cost_ledger: empty_cost_ledger(),
                });
                metadata.push(RecursionNativeChipMetadataRequest {
                    role_id,
                    chip_id: static_chip_id,
                    stable_air_id: dt_stark::air::stable_air_id_v1(&format!(
                        "TestFamily{local_id}"
                    )),
                    prep_width: 1,
                    main_width: 2,
                    perm_width: 0,
                    constraint_count: 0,
                    gate_count: 0,
                    count: 0,
                });
            }
        }
        RecursionNativeProgram::new_with_roles(
            params.child_role,
            params.statement_role,
            params.num_child_public_values,
            params.child_contains_global_bus,
            metadata,
            RecursionPolyAirVerifierProgram::try_new(
                crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
                params.child_role,
                [F::zero(); DIGEST_SIZE],
                chips,
                0,
            )
            .expect("layer test constraint program"),
            statement_config(layer),
        )
    }

    #[test]
    fn native_layer_role_pairs_and_params_match_the_canonical_table() {
        let child_roles =
            [RecursionChildRole::Core, RecursionChildRole::Compress, RecursionChildRole::Shrink];
        let statement_roles = [
            RecursionStatementRole::Lift,
            RecursionStatementRole::ReduceL2,
            RecursionStatementRole::ReduceL3,
            RecursionStatementRole::RootShrink,
        ];
        for child in child_roles {
            for statement in statement_roles {
                let actual = NativeRecursionLayer::from_roles(child, statement);
                let expected = NativeRecursionLayer::ALL.into_iter().find(|layer| {
                    layer.params().child_role == child && layer.params().statement_role == statement
                });
                assert_eq!(actual.ok(), expected, "pair {child:?}/{statement:?}");
            }
        }

        let l1 = NativeRecursionLayer::L1Lift.params();
        assert_eq!(l1.num_child_public_values, dt_stark::air::DT_PROOF_NUM_PV_ELTS);
        assert!(l1.child_contains_global_bus);
        assert_eq!(l1.accepted_child_classes, &[NativeChildClass::CoreShard]);
        assert_eq!(l1.proof_config_class, NativeProofConfigClass::Compress);
        assert_eq!(l1.final_replay_layout, NativeFinalReplayLayout::SingleBase0);

        for layer in [NativeRecursionLayer::L2Reduce, NativeRecursionLayer::L3Reduce] {
            let params = layer.params();
            assert_eq!(params.num_child_public_values, NATIVE_RECURSION_NUM_PV_ELTS);
            assert!(!params.child_contains_global_bus);
            assert_eq!(
                params.accepted_child_classes,
                &[NativeChildClass::Lift, NativeChildClass::L2]
            );
            assert_eq!(params.final_replay_layout, NativeFinalReplayLayout::DualBase0Base128);
        }
        assert_eq!(
            NativeRecursionLayer::L2Reduce.params().proof_config_class,
            NativeProofConfigClass::Compress
        );
        assert_eq!(
            NativeRecursionLayer::L3Reduce.params().proof_config_class,
            NativeProofConfigClass::Shrink
        );

        let l4 = NativeRecursionLayer::L4Root.params();
        assert_eq!(l4.accepted_child_classes, &[NativeChildClass::L3]);
        assert_eq!(l4.proof_config_class, NativeProofConfigClass::RootShrink);
        assert_eq!(l4.final_replay_layout, NativeFinalReplayLayout::SingleBase0);
    }

    #[test]
    fn native_layer_proof_and_recording_configs_fail_closed() {
        let compress = SC::compressed();
        let shrink = SC::shrink();
        let root = RootSC::default();

        assert!(validate_proof_config_for_layer(&compress, NativeRecursionLayer::L1Lift.params())
            .is_ok());
        assert!(validate_proof_config_for_layer(
            &compress,
            NativeRecursionLayer::L2Reduce.params()
        )
        .is_ok());
        assert!(validate_proof_config_for_layer(&shrink, NativeRecursionLayer::L3Reduce.params())
            .is_ok());
        assert!(
            validate_proof_config_for_layer(&root, NativeRecursionLayer::L4Root.params()).is_ok()
        );

        assert!(validate_proof_config_for_layer(&shrink, NativeRecursionLayer::L2Reduce.params())
            .is_err());
        assert!(validate_proof_config_for_layer(
            &compress,
            NativeRecursionLayer::L3Reduce.params()
        )
        .is_err());
        assert!(validate_proof_config_for_layer(&root, NativeRecursionLayer::L3Reduce.params())
            .is_err());
        assert!(validate_proof_config_for_layer(
            &SC::default(),
            NativeRecursionLayer::L1Lift.params()
        )
        .is_err());

        assert!(validate_recording_stage_for_layer(
            RecordingStage::Compress,
            NativeRecursionLayer::L1Lift.params()
        )
        .is_ok());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Compress,
            NativeRecursionLayer::L2Reduce.params()
        )
        .is_ok());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Shrink,
            NativeRecursionLayer::L3Reduce.params()
        )
        .is_ok());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Compress,
            NativeRecursionLayer::L3Reduce.params()
        )
        .is_err());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Shrink,
            NativeRecursionLayer::L2Reduce.params()
        )
        .is_err());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Shrink,
            NativeRecursionLayer::L4Root.params()
        )
        .is_err());
        assert!(validate_recording_stage_for_layer(
            RecordingStage::Core,
            NativeRecursionLayer::L1Lift.params()
        )
        .is_err());
    }

    #[test]
    fn native_layer_final_and_bootstrap_layouts_accept_only_complete_universes() {
        let l1 = test_program(NativeRecursionLayer::L1Lift, &[0]);
        let l2 = test_program(NativeRecursionLayer::L2Reduce, &[0, 128]);
        let l3 = test_program(NativeRecursionLayer::L3Reduce, &[0, 128]);
        let l4 = test_program(NativeRecursionLayer::L4Root, &[0]);
        assert!(validate_final_replay_layout(&l1).is_ok());
        assert!(validate_final_replay_layout(&l2).is_ok());
        assert!(validate_final_replay_layout(&l3).is_ok());
        assert!(validate_final_replay_layout(&l4).is_ok());

        let bootstrap = test_program(NativeRecursionLayer::L2Reduce, &[0]);
        assert!(validate_l2_bootstrap_layout(&bootstrap).is_ok());
        assert!(validate_final_replay_layout(&bootstrap).is_err());
        assert!(validate_l2_bootstrap_layout(&l1).is_err());
    }

    #[test]
    fn native_layer_layout_rejects_empty_partial_stray_and_unsupported_segments() {
        let mut empty = test_program(NativeRecursionLayer::L1Lift, &[0]);
        let mut empty_dto = empty.constraint_program.to_dto();
        empty_dto.chips.clear();
        empty.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(empty_dto)
            .expect("empty frozen program is rejected by layer validation");
        empty.native_chip_metadata.clear();
        assert!(validate_final_replay_layout(&empty).is_err());

        let mut partial = test_program(NativeRecursionLayer::L2Reduce, &[0, 128]);
        let mut partial_dto = partial.constraint_program.to_dto();
        partial_dto.chips.pop();
        partial.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(partial_dto)
            .expect("partial segment remains structurally bounded");
        partial.native_chip_metadata.pop();
        assert!(validate_final_replay_layout(&partial).is_err());

        let stray = test_program(NativeRecursionLayer::L1Lift, &[0, 128]);
        assert!(validate_final_replay_layout(&stray).is_err());

        let unsupported = test_program(NativeRecursionLayer::L1Lift, &[0]);
        let mut unsupported_dto = unsupported.constraint_program.to_dto();
        let last_chip = NativeAirFamily::ALL.len() - 1;
        unsupported_dto.chips[last_chip].static_chip_id = 256;
        assert!(RecursionPolyAirVerifierProgram::try_from_dto(unsupported_dto).is_err());
    }

    #[test]
    fn native_layer_layout_rejects_metadata_order_and_collision_mismatches() {
        let mut metadata_mismatch = test_program(NativeRecursionLayer::L1Lift, &[0]);
        metadata_mismatch.native_chip_metadata[7].main_width += 1;
        assert!(validate_final_replay_layout(&metadata_mismatch).is_err());

        let unsorted = test_program(NativeRecursionLayer::L1Lift, &[0]);
        let mut unsorted_dto = unsorted.constraint_program.to_dto();
        unsorted_dto.chips.swap(4, 5);
        assert!(RecursionPolyAirVerifierProgram::try_from_dto(unsorted_dto).is_err());

        let duplicate = test_program(NativeRecursionLayer::L1Lift, &[0]);
        let mut duplicate_dto = duplicate.constraint_program.to_dto();
        duplicate_dto.chips[5].static_chip_id = duplicate_dto.chips[4].static_chip_id;
        assert!(RecursionPolyAirVerifierProgram::try_from_dto(duplicate_dto).is_err());

        let collision = test_program(NativeRecursionLayer::L2Reduce, &[0, 128]);
        let mut collision_dto = collision.constraint_program.to_dto();
        let second_segment = NativeAirFamily::ALL.len();
        collision_dto.chips[second_segment].static_chip_id = second_segment - 1;
        assert!(RecursionPolyAirVerifierProgram::try_from_dto(collision_dto).is_err());
    }
}
