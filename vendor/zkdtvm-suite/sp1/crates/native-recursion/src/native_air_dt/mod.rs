//! Typed identity and semantic-layer contracts for the native recursion AIR
//! registry.
//!
//! The shared families are:
//! `TranscriptSponge`, `MerklePath`, `Poseidon2Permute`, `ProofHeightSet`,
//! `WhirTwiddleTable`, `WhirSampleBand`, `WhirQueryFold`, `WhirLeafStream`,
//! `WhirLeafExtStream`, `Range8`, and `Range21`.
//!
//! The symbolic-sensitive layer families are:
//! `ProofShapeBinder`, `BatchTranscriptInputs`, `BatchSumcheck`, `WhirRound`,
//! `WhirBatchEval`, `ConstraintTerminal`, `ConstraintBoundary`, `ConstraintChallenge`, `Statement`,
//! and `StatementHash`.
//!
//! The program-sensitive layer families are:
//! `NativeChipMetadata`, `ConstraintProgramTable`, `ConstraintRootTable`,
//! `ConstraintDagEval`, `ConstraintFold`, `ConstraintBetaLadder`,
//! and `StatementConfig`.

use std::collections::{BTreeSet, HashSet};

use dt_stark::{
    air::{FullAir, InteractionScope, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;

use crate::{
    config::F,
    machine_dt::{NativeRecursionAssemblyError, NativeRecursionAssemblyResult},
    system_dt::{RecursionNativeProgram, RecursionRecord},
};

mod identity;
mod layer;
mod shared;

pub use identity::{
    NativeAirFamily, NativeAirId, NativeChildClass, NativeFinalReplayLayout,
    NativeProofConfigClass, NativeRecursionLayer, LAYER_AIR_FAMILIES, NATIVE_AIR_REGISTRY_VERSION,
    PROGRAM_SENSITIVE_AIR_FAMILIES, SHARED_AIR_FAMILIES, SYMBOLIC_SENSITIVE_AIR_FAMILIES,
};
pub(crate) use layer::{
    validate_final_replay_layout, validate_l2_bootstrap_layout, validate_program_matches_layer,
    validate_proof_config_for_layer, validate_recording_stage_for_layer, validate_statement_config,
};
pub use layer::{NativeLayerAirKind, NativeLayerParams, NativeLayerProofConfig};
pub use shared::NativeSharedAir;

/// The stable top-level AIR type used by every native recursion machine.
///
/// A machine contains all 11 shared variants plus the 17 layer variants from
/// exactly one of `L1`/`L2`/`L3`/`L4`.
#[derive(Debug, Clone)]
pub enum NativeRecursionAir {
    Shared(NativeSharedAir),
    L1(NativeLayerAirKind),
    L2(NativeLayerAirKind),
    L3(NativeLayerAirKind),
    L4(NativeLayerAirKind),
}

impl NativeRecursionAir {
    pub fn family(&self) -> NativeAirFamily {
        match self {
            Self::Shared(air) => air.family(),
            Self::L1(air) | Self::L2(air) | Self::L3(air) | Self::L4(air) => air.family(),
        }
    }

    pub fn layer(&self) -> Option<NativeRecursionLayer> {
        match self {
            Self::Shared(_) => None,
            Self::L1(_) => Some(NativeRecursionLayer::L1Lift),
            Self::L2(_) => Some(NativeRecursionLayer::L2Reduce),
            Self::L3(_) => Some(NativeRecursionLayer::L3Reduce),
            Self::L4(_) => Some(NativeRecursionLayer::L4Root),
        }
    }

    pub fn air_id(&self) -> NativeAirId {
        NativeAirId { family: self.family(), layer: self.layer() }
    }

    pub fn all(program: &RecursionNativeProgram<F>) -> NativeRecursionAssemblyResult<Vec<Self>> {
        validate_native_recursion_program(program)?;
        let layer = program.layer()?;
        let params = layer.params();

        let mut registry = NativeSharedAir::all().into_iter().map(Self::Shared).collect::<Vec<_>>();
        registry.extend(
            NativeLayerAirKind::all(program, params)?
                .into_iter()
                .map(|kind| wrap_layer_air(layer, kind)),
        );
        validate_native_registry(program, registry.iter())?;
        Ok(registry)
    }
}

pub fn wrap_layer_air(layer: NativeRecursionLayer, kind: NativeLayerAirKind) -> NativeRecursionAir {
    match layer {
        NativeRecursionLayer::L1Lift => NativeRecursionAir::L1(kind),
        NativeRecursionLayer::L2Reduce => NativeRecursionAir::L2(kind),
        NativeRecursionLayer::L3Reduce => NativeRecursionAir::L3(kind),
        NativeRecursionLayer::L4Root => NativeRecursionAir::L4(kind),
    }
}

pub fn validate_native_recursion_program(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    let params = program.layer()?.params();
    validate_program_matches_layer(program, params)?;
    if !program
        .constraint_program
        .chips
        .windows(2)
        .all(|pair| pair[0].static_chip_id < pair[1].static_chip_id)
    {
        return Err(NativeRecursionAssemblyError::InvalidProgram(
            "constraint program chips are not sorted by static_chip_id".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_native_registry<'a>(
    program: &RecursionNativeProgram<F>,
    registry: impl IntoIterator<Item = &'a NativeRecursionAir>,
) -> NativeRecursionAssemblyResult<()> {
    validate_native_recursion_program(program)?;
    let expected_layer = program.layer()?;
    validate_native_registry_entries(
        expected_layer,
        registry.into_iter().map(|air| (air.air_id(), MachineAir::<F>::name(air))),
    )
}

fn validate_native_registry_entries(
    expected_layer: NativeRecursionLayer,
    entries: impl IntoIterator<Item = (NativeAirId, String)>,
) -> NativeRecursionAssemblyResult<()> {
    let entries = entries.into_iter().collect::<Vec<_>>();
    if entries.len() != NativeAirFamily::ALL.len() {
        return Err(registry_error(format!(
            "native registry has {} AIRs, expected {}",
            entries.len(),
            NativeAirFamily::ALL.len()
        )));
    }

    let shared_families = SHARED_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
    let layer_families = LAYER_AIR_FAMILIES.into_iter().collect::<BTreeSet<_>>();
    let expected_families = NativeAirFamily::ALL.into_iter().collect::<BTreeSet<_>>();
    let mut families = BTreeSet::new();
    let mut names = HashSet::new();
    let mut shared_count = 0;
    let mut layer_count = 0;

    for (id, name) in entries {
        if !families.insert(id.family) {
            return Err(registry_error(format!(
                "native registry contains duplicate family {:?}",
                id.family
            )));
        }
        if !names.insert(name.clone()) {
            return Err(registry_error(format!(
                "native registry contains duplicate wire name {name:?}"
            )));
        }
        if shared_families.contains(&id.family) {
            if id.layer.is_some() {
                return Err(registry_error(format!(
                    "shared family {:?} unexpectedly carries layer {:?}",
                    id.family, id.layer
                )));
            }
            shared_count += 1;
        } else if layer_families.contains(&id.family) {
            if id.layer != Some(expected_layer) {
                return Err(registry_error(format!(
                    "layer family {:?} carries {:?}, expected {:?}",
                    id.family, id.layer, expected_layer
                )));
            }
            layer_count += 1;
        } else {
            return Err(registry_error(format!(
                "family {:?} is absent from the canonical AIR taxonomy",
                id.family
            )));
        }
    }

    if shared_count != SHARED_AIR_FAMILIES.len() || layer_count != LAYER_AIR_FAMILIES.len() {
        return Err(registry_error(format!(
            "native registry split is {shared_count} shared + {layer_count} layer, expected {} + {}",
            SHARED_AIR_FAMILIES.len(),
            LAYER_AIR_FAMILIES.len()
        )));
    }
    if families != expected_families {
        return Err(registry_error(format!(
            "native registry family set differs from the canonical set: {families:?}"
        )));
    }
    Ok(())
}

fn registry_error(message: impl Into<String>) -> NativeRecursionAssemblyError {
    NativeRecursionAssemblyError::Validation(message.into())
}

macro_rules! dispatch_native_air {
    ($self:expr, $air:ident => $body:expr) => {
        match $self {
            NativeRecursionAir::Shared($air) => $body,
            NativeRecursionAir::L1($air) |
            NativeRecursionAir::L2($air) |
            NativeRecursionAir::L3($air) |
            NativeRecursionAir::L4($air) => $body,
        }
    };
}

impl BaseAir<F> for NativeRecursionAir {
    fn width(&self) -> usize {
        dispatch_native_air!(self, air => BaseAir::<F>::width(air))
    }
}

impl<AB> FullAir<AB> for NativeRecursionAir
where
    AB: dt_stark::air::FullAirBuilder<F = F>,
{
    fn width(&self) -> usize {
        dispatch_native_air!(self, air => FullAir::<AB>::width(air))
    }

    fn num_public_values(&self) -> usize {
        dispatch_native_air!(self, air => FullAir::<AB>::num_public_values(air))
    }

    fn required_max_beta_power(&self) -> usize {
        dispatch_native_air!(self, air => FullAir::<AB>::required_max_beta_power(air))
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        dispatch_native_air!(self, air => FullAir::<AB>::reserved_poly(air))
    }

    fn precompute_lc(&self, builder: &mut AB) {
        dispatch_native_air!(self, air => FullAir::<AB>::precompute_lc(air, builder))
    }

    fn eval(&self, builder: &mut AB) {
        dispatch_native_air!(self, air => FullAir::<AB>::eval(air, builder))
    }

    fn lookup(&self, builder: &mut AB) {
        dispatch_native_air!(self, air => FullAir::<AB>::lookup(air, builder))
    }

    fn global(&self) -> bool {
        dispatch_native_air!(self, air => FullAir::<AB>::global(air))
    }
}

impl MachineAir<F> for NativeRecursionAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        self.air_id().wire_name().to_string()
    }

    fn preprocessed_width(&self) -> usize {
        dispatch_native_air!(self, air => MachineAir::<F>::preprocessed_width(air))
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        dispatch_native_air!(self, air =>
            MachineAir::<F>::preprocessed_num_rows(air, program, instrs_len)
        )
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        dispatch_native_air!(self, air =>
            MachineAir::<F>::generate_preprocessed_trace(air, program)
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        dispatch_native_air!(self, air => MachineAir::<F>::num_rows(air, input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        dispatch_native_air!(self, air => MachineAir::<F>::generate_trace(air, input, output))
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        dispatch_native_air!(self, air =>
            MachineAir::<F>::generate_dependencies(air, input, output)
        )
    }

    fn included(&self, record: &Self::Record) -> bool {
        dispatch_native_air!(self, air => MachineAir::<F>::included(air, record))
    }

    fn commit_scope(&self) -> InteractionScope {
        dispatch_native_air!(self, air => MachineAir::<F>::commit_scope(air))
    }

    fn local_only(&self) -> bool {
        dispatch_native_air!(self, air => MachineAir::<F>::local_only(air))
    }

    fn padding_row(&self) -> Vec<F> {
        dispatch_native_air!(self, air => MachineAir::<F>::padding_row(air))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use p3_field::AbstractField;

    use super::*;
    use crate::{
        config::{DIGEST_SIZE, D_EF},
        statement_dt::{
            STATEMENT_CONFIG_CLASS_BAKED_L2, STATEMENT_CONFIG_CLASS_BAKED_L3,
            STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        },
        symbolic_ir_dt::RecursionPolyAirVerifierProgram,
        system_dt::StatementConfigRow,
    };

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

    fn program(layer: NativeRecursionLayer) -> RecursionNativeProgram<F> {
        let params = layer.params();
        RecursionNativeProgram::new_with_roles(
            params.child_role,
            params.statement_role,
            params.num_child_public_values,
            params.child_contains_global_bus,
            Vec::new(),
            RecursionPolyAirVerifierProgram::try_new(
                crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
                params.child_role,
                [F::zero(); DIGEST_SIZE],
                Vec::new(),
                0,
            )
            .expect("empty registry test constraint program"),
            statement_config(layer),
        )
    }

    fn registry_entries(registry: &[NativeRecursionAir]) -> Vec<(NativeAirId, String)> {
        registry.iter().map(|air| (air.air_id(), MachineAir::<F>::name(air))).collect()
    }

    #[test]
    fn native_layer_registries_are_complete_homogeneous_and_layer_qualified() {
        let expected_shared_names = [
            (NativeAirFamily::TranscriptSponge, "NativeTranscriptSponge"),
            (NativeAirFamily::MerklePath, "NativeMerklePath"),
            (NativeAirFamily::Poseidon2Permute, "NativePoseidon2Permute"),
            (NativeAirFamily::ProofHeightSet, "NativeProofHeightSet"),
            (NativeAirFamily::WhirTwiddleTable, "WhirTwiddleTable"),
            (NativeAirFamily::WhirSampleBand, "WhirSampleBand"),
            (NativeAirFamily::WhirQueryFold, "WhirQueryFold"),
            (NativeAirFamily::WhirLeafStream, "WhirLeafStream"),
            (NativeAirFamily::WhirLeafExtStream, "WhirLeafExtStream"),
            (NativeAirFamily::Range8, "NativeRangeChecker8"),
            (NativeAirFamily::Range21, "NativeRangeChecker21"),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
        let layer_tokens = [
            (NativeRecursionLayer::L1Lift, "L1"),
            (NativeRecursionLayer::L2Reduce, "L2"),
            (NativeRecursionLayer::L3Reduce, "L3"),
            (NativeRecursionLayer::L4Root, "L4"),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
        let layer_suffixes = [
            (NativeAirFamily::ProofShapeBinder, "ProofShapeBinder"),
            (NativeAirFamily::BatchTranscriptInputs, "BatchTranscriptInputs"),
            (NativeAirFamily::BatchSumcheck, "BatchSumcheck"),
            (NativeAirFamily::WhirRound, "WhirRound"),
            (NativeAirFamily::WhirBatchEval, "WhirBatchEval"),
            (NativeAirFamily::ConstraintTerminal, "ConstraintTerminal"),
            (NativeAirFamily::ConstraintBoundary, "ConstraintBoundary"),
            (NativeAirFamily::Statement, "Statement"),
            (NativeAirFamily::StatementHash, "StatementHash"),
            (NativeAirFamily::NativeChipMetadata, "NativeChipMetadata"),
            (NativeAirFamily::ConstraintProgramTable, "ConstraintProgramTable"),
            (NativeAirFamily::ConstraintRootTable, "ConstraintRootTable"),
            (NativeAirFamily::ConstraintDagEval, "ConstraintDagEval"),
            (NativeAirFamily::ConstraintFold, "ConstraintFold"),
            (NativeAirFamily::ConstraintBetaLadder, "ConstraintBetaLadder"),
            (NativeAirFamily::ConstraintChallenge, "ConstraintChallenge"),
            (NativeAirFamily::StatementConfig, "StatementConfig"),
        ]
        .into_iter()
        .collect::<BTreeMap<_, _>>();

        for layer in NativeRecursionLayer::ALL {
            let program = program(layer);
            let registry = NativeRecursionAir::all(&program).expect("valid registry");
            validate_native_registry(&program, registry.iter()).expect("registry validation");

            assert_eq!(registry.len(), 28);
            assert_eq!(registry.iter().filter(|air| air.layer().is_none()).count(), 11);
            assert_eq!(registry.iter().filter(|air| air.layer() == Some(layer)).count(), 17);
            assert_eq!(
                registry.iter().map(NativeRecursionAir::family).collect::<BTreeSet<_>>(),
                NativeAirFamily::ALL.into_iter().collect()
            );
            assert_eq!(
                registry
                    .iter()
                    .map(|air| MachineAir::<F>::name(air))
                    .collect::<BTreeSet<_>>()
                    .len(),
                28
            );
            for air in &registry {
                let expected = if air.layer().is_none() {
                    expected_shared_names[&air.family()].to_string()
                } else {
                    format!("Native{}{}", layer_tokens[&layer], layer_suffixes[&air.family()])
                };
                assert_eq!(MachineAir::<F>::name(air), expected, "{layer:?} {:?}", air.family());
            }
        }

        let bootstrap = program(NativeRecursionLayer::L2Reduce);
        let mut final_l2 = bootstrap.clone();
        let mut final_dto = final_l2.constraint_program.to_dto();
        final_dto.artifact_digest[0] = F::one();
        final_l2.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(final_dto)
            .expect("final registry test constraint program");
        let bootstrap_ids = NativeRecursionAir::all(&bootstrap)
            .unwrap()
            .iter()
            .map(NativeRecursionAir::air_id)
            .collect::<Vec<_>>();
        let final_ids = NativeRecursionAir::all(&final_l2)
            .unwrap()
            .iter()
            .map(NativeRecursionAir::air_id)
            .collect::<Vec<_>>();
        assert_eq!(bootstrap_ids, final_ids);
    }

    #[test]
    fn native_layer_wire_registries_exclude_all_legacy_layer_names() {
        let legacy_layer_names = [
            "NativeProofShapeBinder",
            "NativeBatchTranscriptInputs",
            "NativeBatchSumcheck",
            "WhirRound",
            "WhirBatchEval",
            "NativeConstraintTerminal",
            "NativeStatement",
            "NativeStatementHash",
            "NativeChipMetadata",
            "NativeConstraintProgramTable",
            "NativeConstraintRootTable",
            "NativeConstraintDagEval",
            "NativeConstraintFold",
            "NativeBetaLadder",
            "NativeConstraintChallenge",
            "NativeStatementConfig",
        ];

        for layer in NativeRecursionLayer::ALL {
            let names = NativeRecursionAir::all(&program(layer))
                .expect("valid registry")
                .iter()
                .map(|air| MachineAir::<F>::name(air))
                .collect::<BTreeSet<_>>();
            for legacy in legacy_layer_names {
                assert!(
                    !names.contains(legacy),
                    "{layer:?} registry revived legacy layer name {legacy:?}"
                );
            }
        }
    }

    #[test]
    fn native_layer_registry_validator_rejects_mixed_missing_duplicate_and_tagged_shared() {
        let layer = NativeRecursionLayer::L2Reduce;
        let registry = NativeRecursionAir::all(&program(layer)).unwrap();
        let canonical = registry_entries(&registry);

        let mut mixed = canonical.clone();
        mixed.iter_mut().find(|(id, _)| id.layer.is_some()).unwrap().0.layer =
            Some(NativeRecursionLayer::L3Reduce);
        assert!(validate_native_registry_entries(layer, mixed).is_err());

        let mut missing = canonical.clone();
        missing.pop();
        assert!(validate_native_registry_entries(layer, missing).is_err());

        let mut duplicate = canonical.clone();
        let last = duplicate.len() - 1;
        duplicate[last].0 = duplicate[last - 1].0;
        assert!(validate_native_registry_entries(layer, duplicate).is_err());

        let mut tagged_shared = canonical.clone();
        tagged_shared.iter_mut().find(|(id, _)| id.layer.is_none()).unwrap().0.layer = Some(layer);
        assert!(validate_native_registry_entries(layer, tagged_shared).is_err());

        let mut duplicate_name = canonical;
        let last = duplicate_name.len() - 1;
        duplicate_name[last].1 = duplicate_name[last - 1].1.clone();
        assert!(validate_native_registry_entries(layer, duplicate_name).is_err());
    }

    #[test]
    fn native_layer_outer_forwarding_matches_inner_shared_and_layer_airs() {
        let program = program(NativeRecursionLayer::L2Reduce);
        let record = RecursionRecord::default();

        let shared = NativeSharedAir::all()
            .into_iter()
            .find(|air| air.family() == NativeAirFamily::Range8)
            .unwrap();
        let shared_outer = NativeRecursionAir::Shared(shared.clone());
        assert_eq!(BaseAir::<F>::width(&shared), BaseAir::<F>::width(&shared_outer));
        assert_eq!(shared.num_rows(&record), shared_outer.num_rows(&record));
        assert_eq!(shared.included(&record), shared_outer.included(&record));
        assert_eq!(shared.padding_row(), shared_outer.padding_row());
        let mut shared_direct_output = RecursionRecord::default();
        let mut shared_outer_output = RecursionRecord::default();
        shared.generate_dependencies(&record, &mut shared_direct_output);
        shared_outer.generate_dependencies(&record, &mut shared_outer_output);
        assert_eq!(shared_direct_output, shared_outer_output);
        assert_eq!(
            bincode::serialize(&shared.generate_trace(&record, &mut RecursionRecord::default()))
                .unwrap(),
            bincode::serialize(
                &shared_outer.generate_trace(&record, &mut RecursionRecord::default())
            )
            .unwrap()
        );

        let layer = NativeLayerAirKind::all(&program, NativeRecursionLayer::L2Reduce.params())
            .unwrap()
            .into_iter()
            .find(|air| air.family() == NativeAirFamily::StatementConfig)
            .unwrap();
        let layer_outer = NativeRecursionAir::L2(layer.clone());
        assert_eq!(BaseAir::<F>::width(&layer), BaseAir::<F>::width(&layer_outer));
        assert_eq!(layer.preprocessed_width(), layer_outer.preprocessed_width());
        assert_eq!(
            layer.preprocessed_num_rows(&program, 0),
            layer_outer.preprocessed_num_rows(&program, 0)
        );
        assert_eq!(layer.num_rows(&record), layer_outer.num_rows(&record));
        assert_eq!(layer.included(&record), layer_outer.included(&record));
        assert_eq!(layer.padding_row(), layer_outer.padding_row());
        assert_eq!(
            bincode::serialize(&layer.generate_preprocessed_trace(&program)).unwrap(),
            bincode::serialize(&layer_outer.generate_preprocessed_trace(&program)).unwrap()
        );
        assert_eq!(
            bincode::serialize(&layer.generate_trace(&record, &mut RecursionRecord::default()))
                .unwrap(),
            bincode::serialize(
                &layer_outer.generate_trace(&record, &mut RecursionRecord::default())
            )
            .unwrap()
        );
    }

    #[test]
    fn native_layer_registry_types_remain_the_machine_alias_authority() {
        fn assert_machine(
            _: &polyair::SCStarkMachine<crate::config::SC, NativeRecursionAir, D_EF>,
        ) {
        }
        let program = program(NativeRecursionLayer::L1Lift);
        let machine = crate::machine_dt::native_recursion_machine(&program).unwrap();
        assert_machine(&machine);
    }
}
