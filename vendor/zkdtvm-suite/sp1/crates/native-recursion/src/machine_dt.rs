use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::Instant;

use dt_core_machine::{
    riscv::riscv_polyair::RiscvPolyAir,
    utils::prove_polyair::{POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD, POLYAIR_NUM_SKIP_ROUNDS},
};
use dt_stark::{
    air::{derive_active_shape_v1, observe_active_shape_v1, MachineAir, PairCol},
    global_d11::observe_program_global_metadata_v2,
    koalabear_poseidon2::koala_bear_poseidon2::{
        compressed_fri_config, default_fri_config, shrink_fri_config,
    },
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
        trace::CompressedMatrix,
    },
    MachineRecord, StarkGenericConfig,
};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::Matrix;
use pcs::basefold::mlpcs::MlPCS;
use polyair::prover::SCMachineProver;

use crate::{
    batch_constraint_dt::record_batch_constraint_materials_from_views,
    child_views::{
        NativeAirAuthority, NativeChildMetadataView, NativeChildRole,
        NativeChildVerifierConfigView, NativeChildViews, NativeChipMetadata, NativeWhirConfigView,
    },
    config::{RootSC, DIGEST_SIZE, D_EF, EF, F, SC},
    constraint_replay_dt::{
        annotate_child_constraint_replay_publications, constraint_challenge_rows,
        constraint_replay_bus_residual_report, constraint_terminal_rows,
    },
    native_air_dt::{
        validate_native_registry, validate_program_matches_layer, validate_proof_config_for_layer,
        validate_recording_stage_for_layer, validate_statement_config, NativeLayerProofConfig,
        NativeRecursionLayer,
    },
    proof_shape_dt::{metadata_universe_from_view, record_proof_shape_from_views},
    statement_boundary_air_dt::{
        annotate_child_statement_publications, statement_part_b_bus_residual_report,
        statement_rows_cached,
    },
    statement_dt::NATIVE_RECURSION_NUM_PV_ELTS,
    statement_hash_air_dt::{
        statement_hash_bus_residual_report, statement_hash_rows_cached, StatementDigestMode,
    },
    symbolic_expr_fixed_dt::{RecursionChildRole, RecursionFixedSymbolicChip},
    symbolic_ir_dt::{
        RecursionPolyAirChipIr, RecursionPolyAirDerivedRoot, RecursionPolyAirVerifierProgram,
        RecursionPolyAirVerifierProgramDto,
    },
    system_dt::{
        BuildingRecord, FinalizedRecord, RecordingSC, RecordingStage, RecursionNativeProgram,
        RecursionRecord, RecursionRecordProfileSnapshot, RecursionStatementRole,
        ReplayCompatibleProofConfig, StatementConfigRow,
    },
    tracegen_backend::{
        PreparedRecord, TracegenAdmission, TracegenInput, TracegenReductionSummary,
        TracegenWorkspace,
    },
    transcript_dt::{
        poseidon2::RecursionPoseidon2MemoSnapshot, sponge::trace::transcript_sponge_rows_cached,
    },
    validate::{check_provider_pools, finalize_provider_requests_at_source},
    whir_dt::{
        attach_whir_tracegen_materials, materialize_whir_tracegen_sources,
        prepare_whir_tracegen_materials, whir_bus_residual_report,
    },
};

pub use crate::native_air_dt::NativeRecursionAir;

pub type CoreRecordingMachine = polyair::SCStarkMachine<RecordingSC, RiscvPolyAir<F>, D_EF>;
pub type CoreRecordingChip = polyair::Chip<RiscvPolyAir<F>, F, D_EF>;
pub type NativeRecordingMachine = polyair::SCStarkMachine<RecordingSC, NativeRecursionAir, D_EF>;
pub type NativeRecursionMachine = polyair::SCStarkMachine<SC, NativeRecursionAir, D_EF>;
pub type NativeRecursionProver<P = CpuNativeProver> = <P as NativeProverProvider>::SCProver;
/// The root_shrink (L4) machine/prover over the SHA256-hashed [`RootSC`]. Only
/// the final proof's own PCS/transcript differ; the circuit content is the
/// same shrink-shaped recursion machine.
pub type NativeRootMachine = polyair::SCStarkMachine<RootSC, NativeRecursionAir, D_EF>;
pub type NativeRootProver<P = CpuNativeProver> = <P as NativeProverProvider>::RootProver;

/// GPU injection point for native recursion provers.
pub trait NativeProverProvider {
    /// Prover for compress/shrink (Poseidon2 SC).
    type SCProver: SCMachineProver<SC, NativeRecursionAir, D_EF>;
    /// Prover for root_shrink (SHA256 RootSC).
    type RootProver: SCMachineProver<RootSC, NativeRecursionAir, D_EF>;
}

/// Default CPU prover.
pub struct CpuNativeProver;
impl NativeProverProvider for CpuNativeProver {
    type SCProver = polyair::prover::SumcheckProver<SC, NativeRecursionAir, D_EF>;
    type RootProver = polyair::prover::SumcheckProver<RootSC, NativeRecursionAir, D_EF>;
}

const NATIVE_RECURSION_NUM_SKIP_ROUNDS: usize = 1;
const NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD: usize = 0;
pub(crate) const NATIVE_ROOT_SHRINK_DEGREE_FLOOR: usize = 3;
/// The L3 shrink prover and its recording mirror share the degree-3 floor.
pub(crate) const NATIVE_SHRINK_DEGREE_FLOOR: usize = 3;
const FINAL_RESIDUALS_ENV: &str = "DT_NATIVE_RECURSION_FINAL_RESIDUALS";
const INTERMEDIATE_RESIDUALS_ENV: &str = "DT_NATIVE_RECURSION_INTERMEDIATE_RESIDUALS";

/// Unforgeable outside this module; it makes the canonical finalizer the only constructor of a
/// [`FinalizedRecord`] even though the wrapper itself lives with the record DTO.
pub(crate) struct FinalizationSeal(());

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProveRecursionTimings {
    pub record_generation: u64,
    pub record_profile: RecursionRecordProfileSnapshot,
    pub poseidon2_memo: RecursionPoseidon2MemoSnapshot,
    pub planned_chip_log_heights: Vec<(String, u8)>,
    pub row_count_admission_ms: u128,
    pub trace_plan_fold_ms: u128,
    pub tracegen_ms: u128,
    pub budget_ms: u128,
    pub pool_gate_ms: u128,
    pub commit_ms: u128,
    pub commit_profile: BTreeMap<String, u128>,
    pub open_ms: u128,
    pub open_profile: BTreeMap<String, u128>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RecursionTraceCost {
    pub chip: String,
    pub height: usize,
    pub stored_height: usize,
    pub width: usize,
    pub perm_width: usize,
    pub interactions: usize,
    pub constraints: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProveRecursionMetrics {
    pub timings: ProveRecursionTimings,
    pub trace_costs: Vec<RecursionTraceCost>,
}

#[derive(Debug)]
pub enum NativeRecursionAssemblyError {
    InvalidProgram(String),
    Record(String),
    BusResidual(String),
    Validation(String),
    Prove(String),
    Verify(String),
}

impl fmt::Display for NativeRecursionAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProgram(message) => {
                write!(f, "invalid native recursion program: {message}")
            }
            Self::Record(message) => write!(f, "native recursion record build failed: {message}"),
            Self::BusResidual(message) => {
                write!(f, "native recursion bus residual is non-empty: {message}")
            }
            Self::Validation(message) => write!(f, "native recursion validation failed: {message}"),
            Self::Prove(message) => write!(f, "native recursion prove failed: {message}"),
            Self::Verify(message) => write!(f, "native recursion verify failed: {message}"),
        }
    }
}

impl Error for NativeRecursionAssemblyError {}

pub type NativeRecursionAssemblyResult<T> = Result<T, NativeRecursionAssemblyError>;

#[cfg(test)]
pub(crate) fn native_recursion_machine(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionMachine> {
    native_recursion_machine_with_config(program, SC::compressed())
}

pub(crate) fn native_recursion_machine_with_config(
    program: &RecursionNativeProgram<F>,
    config: SC,
) -> NativeRecursionAssemblyResult<NativeRecursionMachine> {
    let params = program.layer()?.params();
    validate_program_matches_layer(program, params)?;
    validate_proof_config_for_layer(&config, params)?;
    let chips = NativeRecursionAir::all(program)?
        .into_iter()
        .map(polyair::Chip::<NativeRecursionAir, F, D_EF>::new)
        .collect();
    let machine = polyair::SCStarkMachine::new(config, chips, NATIVE_RECURSION_NUM_PV_ELTS, false);
    debug_assert_eq!(
        machine.num_pv_elts(),
        NATIVE_RECURSION_NUM_PV_ELTS,
        "phase-2 M0 native recursion statement public-value width changed"
    );
    Ok(machine)
}

#[cfg(test)]
pub(crate) fn native_recursion_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    let machine = native_recursion_machine(program)?;
    Ok(polyair::prover::SumcheckProver { machine })
}

pub(crate) fn native_recursion_prover_with_config(
    program: &RecursionNativeProgram<F>,
    config: SC,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    native_recursion_prover_with_config_and_provider::<CpuNativeProver>(program, config)
}

pub(crate) fn native_recursion_prover_with_config_and_provider<P: NativeProverProvider>(
    program: &RecursionNativeProgram<F>,
    config: SC,
) -> NativeRecursionAssemblyResult<NativeRecursionProver<P>> {
    let machine = native_recursion_machine_with_config(program, config)?;
    Ok(<<P as NativeProverProvider>::SCProver as SCMachineProver<SC, NativeRecursionAir, D_EF>>::new(machine))
}

pub(crate) fn native_root_shrink_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRootProver> {
    native_root_shrink_prover_with_provider::<CpuNativeProver>(program)
}

pub(crate) fn native_root_shrink_prover_with_provider<P: NativeProverProvider>(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRootProver<P>> {
    let machine = native_root_verifier_machine(program, RootSC::default())?;
    print_chip_batch_profile("root_shrink", machine.chips());
    Ok(<<P as NativeProverProvider>::RootProver as SCMachineProver<
        RootSC,
        NativeRecursionAir,
        D_EF,
    >>::new(machine))
}

/// Builds the canonical provider-free L4 verifier machine from its frozen program and an explicit
/// root proof configuration. No proving key, prover provider, thread pool, or GPU resource is
/// constructed by this path.
pub fn native_root_verifier_machine(
    program: &RecursionNativeProgram<F>,
    config: RootSC,
) -> NativeRecursionAssemblyResult<NativeRootMachine> {
    let params = program.layer()?.params();
    validate_program_matches_layer(program, params)?;
    validate_proof_config_for_layer(&config, params)?;
    let chips: Vec<polyair::Chip<NativeRecursionAir, F, D_EF>> = NativeRecursionAir::all(program)?
        .into_iter()
        .map(|air| {
            polyair::Chip::<NativeRecursionAir, F, D_EF>::new_with_degree_floor(
                air,
                NATIVE_ROOT_SHRINK_DEGREE_FLOOR,
            )
        })
        .collect();
    Ok(polyair::SCStarkMachine::new(config, chips, NATIVE_RECURSION_NUM_PV_ELTS, false))
}

/// Prints one per-chip profile line per machine construction: degree, logup batch,
/// lookup count, perm width, committed width, and the number of MAIN columns actually
/// referenced (`r`). This exposes the exact per-chip batch profile selected by
/// the configured degree floor.
pub(crate) fn print_chip_batch_profile(
    label: &str,
    chips: &[polyair::Chip<NativeRecursionAir, F, D_EF>],
) {
    use p3_air::BaseAir;
    if !crate::debug_prints_enabled() {
        return;
    }
    let summary = chips
        .iter()
        .map(|chip| {
            format!(
                "{}=d{}/b{}/l{}/p{}/w{}/r{}",
                MachineAir::<F>::name(chip),
                chip.degree,
                chip.logup_batch_size(),
                chip.num_lookup(),
                chip.perm_width(),
                BaseAir::<F>::width(chip),
                referenced_main_columns(chip),
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!("native_chip_batch_profile machine={label} {summary}");
}

/// Counts the MAIN columns a chip's symbolic constraint system actually reads —
/// gate constraints, lookup multiplicities, and precomputed LCs, with `ReservedPoly`
/// slots resolved through the reserve list. Uses a zero-allocation recursive walker
/// (`iter_all_var`'s chained collects go quadratic on large expression trees).
fn referenced_main_columns(chip: &polyair::Chip<NativeRecursionAir, F, D_EF>) -> usize {
    use polyair::symbolic::{SymbolicExpression, SymbolicVar};
    // The expression graph is an Rc-shared DAG; memoize on node identity or
    // the walk is exponential on shared subtrees (the iter_all_var failure
    // mode, caught by perf as a multi-minute allocation storm).
    fn walk(
        expr: &SymbolicExpression<SymbolicVar, F, D_EF>,
        reserved: &[PairCol],
        referenced: &mut std::collections::BTreeSet<usize>,
        seen: &mut std::collections::HashSet<*const SymbolicExpression<SymbolicVar, F, D_EF>>,
    ) {
        if !seen.insert(expr as *const _) {
            return;
        }
        match expr {
            SymbolicExpression::VARiable(var) => match var {
                SymbolicVar::Main(idx) => {
                    referenced.insert(*idx);
                }
                SymbolicVar::ReservedPoly(idx, _) => {
                    if let Some(PairCol::Main(main_idx)) = reserved.get(*idx) {
                        referenced.insert(*main_idx);
                    }
                }
                _ => {}
            },
            SymbolicExpression::Constant(_) | SymbolicExpression::ConstantExt(_) => {}
            SymbolicExpression::Add { x, y, .. } |
            SymbolicExpression::Sub { x, y, .. } |
            SymbolicExpression::Mul { x, y, .. } => {
                walk(x, reserved, referenced, seen);
                walk(y, reserved, referenced, seen);
            }
            SymbolicExpression::Neg { x, .. } => walk(x, reserved, referenced, seen),
        }
    }
    let builder = &chip.symbolic_builder;
    let reserved = chip.reserved_poly();
    let mut referenced = std::collections::BTreeSet::new();
    let mut seen = std::collections::HashSet::new();
    for expr in &builder.gate {
        walk(expr, reserved, &mut referenced, &mut seen);
    }
    for lookup in &builder.lookup_infos {
        walk(&lookup.multiplicity, reserved, &mut referenced, &mut seen);
    }
    for expr in &builder.precomputed_lc_output {
        walk(expr, reserved, &mut referenced, &mut seen);
    }
    referenced.len()
}

/// The generic prover type used by the recursion prove/verify helpers below:
/// both the Poseidon2 [`SC`] (lift/L2/L3) and the SHA256 [`RootSC`] (L4)
/// configs instantiate it. The AIR is always the concrete
/// [`NativeRecursionAir`], so all AIR-side bounds normalize to concrete impls
/// once `Val = F` / `Challenge = EF` are pinned.
pub type NativeProverFor<C> = polyair::prover::SumcheckProver<C, NativeRecursionAir, D_EF>;

#[cfg(test)]
pub(crate) fn prove_recursion<C, PROV>(
    prover: &PROV,
    pk: &SCStarkProvingKey<C>,
    device_pk: &PROV::DeviceProvingKey,
    record: FinalizedRecord,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<SCMachineProof<C>>
where
    C: NativeLayerProofConfig,
    PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
    MlCom<C>: Send + Sync,
    MlPcsProverData<C>: Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    prove_recursion_with_metrics(prover, pk, device_pk, record, program).map(|(proof, _)| proof)
}

/// Descriptor-only S6 node seal. Exact row admission and every final matrix allocation happen in
/// [`TraceBundle::materialize`], after tracegen has consumed the mode-specific providers.
pub(crate) struct FinalizedNode {
    input: TracegenInput,
    admit_ms: u128,
}

impl FinalizedNode {
    pub(crate) fn admit<C, PROV>(
        prover: &PROV,
        record: FinalizedRecord,
        program: &RecursionNativeProgram<F>,
    ) -> NativeRecursionAssemblyResult<Self>
    where
        C: NativeLayerProofConfig,
        PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
    {
        if !record.matches_program(program) {
            return Err(NativeRecursionAssemblyError::Validation(
                "finalized recursion record was paired with a different program authority"
                    .to_string(),
            ));
        }
        let admit_start = Instant::now();
        let raw = record.record();
        let params = program.layer()?.params();
        validate_program_matches_layer(program, params)?;
        validate_native_registry(program, prover.machine().chips.iter().map(|chip| &chip.air))?;
        validate_proof_config_for_layer(prover.config(), params)?;
        validate_record_matches_program(raw, program)?;
        let safe_admission_start = Instant::now();
        let prepared = PreparedRecord::seal(record, program.layer()?)
            .map_err(NativeRecursionAssemblyError::Validation)?;
        let bounds = prepared.bounds();
        let counts = prepared.counts();
        let input =
            TracegenInput::new(prepared).map_err(NativeRecursionAssemblyError::Validation)?;
        TracegenAdmission::admit(&input).map_err(NativeRecursionAssemblyError::Validation)?;
        let safe_admission_ms = safe_admission_start.elapsed().as_millis();
        let raw = input.record().record();
        let mut structural_counters = vec![
            ("full_poseidon2_witness_rows_during_prepare", 0),
            ("full_poseidon2_witness_bytes_during_prepare", 0),
            ("final_air_row_cells_during_prepare", 0),
            ("parent_provider_rehash_entries_during_seal", 0),
            ("provider_events_inspected_during_seal", 0),
            ("bytes_copied_during_seal", 0),
            (
                "tracegen_input_descriptor_bytes",
                u64::try_from(bounds.descriptor_bytes).expect("descriptor bytes exceed u64"),
            ),
            (
                "matrix_count_safe_upper_bound",
                u64::try_from(bounds.matrix_count_upper_bound)
                    .expect("matrix count upper bound exceeds u64"),
            ),
            (
                "prepared_family_count",
                u64::try_from(counts.family_count).expect("family count exceeds u64"),
            ),
        ];
        structural_counters.push(("locally_reduced_provider_segments", 0));
        let admit_ms = raw.profile.publish_tracegen_input_batch_and_seal(
            admit_start,
            &[
                ("safe_upper_bound_pre_admission", safe_admission_ms),
                ("descriptor_and_semantic_seal", safe_admission_ms),
            ],
            &structural_counters,
        );
        Ok(Self { input, admit_ms })
    }
}

#[derive(Debug)]
pub struct PreparedRecursionTracegenRecord {
    pub record: RecursionRecord,
    pub plan: crate::validate::ExactTracePlan,
    pub authority: crate::TracegenAuthorityHandle,
    pub record_generation: u64,
    pub pool_gate_ms: u128,
    pub tracegen_preparation_ms: u128,
}

/// Control/artifact lane prepared without expanding compact WHIR sources on
/// the host. Device Pass A applies the single complete-key provider semantics
/// and supplies the heavy-family exact counts before this record can
/// receive an [`crate::validate::ExactTracePlan`].
#[derive(Debug)]
pub struct PreparedCompactRecursionTracegenRecord {
    pub record: RecursionRecord,
    pub authority: crate::TracegenAuthorityHandle,
    pub record_generation: u64,
    pub pool_gate_ms: u128,
    pub tracegen_preparation_ms: u128,
}

pub fn prepare_recursion_tracegen_record_compact_with_timing<C, PROV>(
    prover: &PROV,
    record: FinalizedRecord,
    program: &RecursionNativeProgram<F>,
    mut mark: impl FnMut(&str),
) -> NativeRecursionAssemblyResult<PreparedCompactRecursionTracegenRecord>
where
    C: NativeLayerProofConfig + SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>,
    PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
{
    mark("enter_prepare_recursion_tracegen_record_compact");
    let record_generation = record.generation();
    let node = FinalizedNode::admit(prover, record, program)?;
    mark("FinalizedNode::admit");
    let pool_gate_ms = node.admit_ms;
    let tracegen_start = Instant::now();
    let FinalizedNode { input, .. } = node;
    let mut workspace = input.into_workspace().map_err(NativeRecursionAssemblyError::Validation)?;
    mark("TracegenInput::into_workspace");
    if workspace.record().poseidon2_tracegen.retained_rows() != 0 ||
        workspace.record().poseidon2_tracegen.generated_rows() != 0
    {
        return Err(NativeRecursionAssemblyError::Validation(
            "Poseidon2 full-witness cache was populated before compact tracegen".to_string(),
        ));
    }
    if workspace.record().proof_records.iter().any(|proof| {
        proof.whir_source.is_none() || !proof.whir.is_empty() || proof.merkle_path.row_count() != 0
    }) {
        return Err(NativeRecursionAssemblyError::Validation(
            "compact tracegen requires one unmaterialized WHIR source per proof".to_string(),
        ));
    }
    mark("validate_compact_whir_authority");

    finalize_provider_requests_at_source(
        workspace.record_mut(),
        StatementDigestMode::from_role(program.statement_role),
    );
    mark("finalize_base_provider_requests_at_source");
    workspace.validate().map_err(NativeRecursionAssemblyError::Validation)?;
    let _base_provider_seal =
        workspace.seal_provider_inputs().map_err(NativeRecursionAssemblyError::Validation)?;
    mark("seal_base_provider_inputs");

    workspace.install_transcript_owner().map_err(NativeRecursionAssemblyError::Validation)?;
    mark("install_transcript_owner");
    let raw = workspace.record();
    let constraint_started = Instant::now();
    let constraint_counts = crate::constraint_replay_dt::trace::prepare_constraint_authority(
        raw,
        &program.constraint_program,
    );
    raw.profile.add_record_split(
        "tracegen.compact.constraint_authority",
        constraint_started.elapsed().as_millis(),
    );
    mark("prepare_constraint_authority");
    let statement_started = Instant::now();
    let statement_rows =
        statement_rows_cached(raw, program.statement_role, &program.statement_config);
    let statement_hash_rows =
        statement_hash_rows_cached(raw, StatementDigestMode::from_role(program.statement_role));
    let transcript_rows = transcript_sponge_rows_cached(raw);
    raw.profile.add_record_split(
        "tracegen.compact.statement_transcript_authority",
        statement_started.elapsed().as_millis(),
    );
    raw.profile.set_structural_counters([
        ("host_whir_exact_rows", 0),
        ("host_merkle_exact_rows", 0),
        ("host_poseidon_full_witness_rows", 0),
        ("host_provider_rehash_entries_after_dispatch", 0),
        ("compact_source_owner_count", 1),
        ("compact_source_batch_dispatch_count", 1),
        ("cpu_authority_artifact_owner_count", 1),
        ("cpu_authority_artifact_batch_dispatch_count", 1),
        ("constraint_statement_device_derivations", 0),
        ("provider_unregistered_or_double_publish", 0),
        (
            "unpadded_rows_by_family.constraint_dag",
            u64::try_from(constraint_counts.dag).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.constraint_fold",
            u64::try_from(constraint_counts.fold).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.constraint_challenge",
            u64::try_from(constraint_counts.challenge).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.constraint_beta_ladder",
            u64::try_from(constraint_counts.beta_ladder).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.constraint_terminal",
            u64::try_from(constraint_counts.terminal).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.statement",
            u64::try_from(statement_rows.len()).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.statement_hash",
            u64::try_from(statement_hash_rows.len()).unwrap_or(u64::MAX),
        ),
        (
            "unpadded_rows_by_family.transcript_sponge",
            u64::try_from(transcript_rows.len()).unwrap_or(u64::MAX),
        ),
    ]);
    mark("prepare_statement_and_transcript_exact_rows");
    check_provider_pools(raw)
        .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
    mark("check_base_provider_pools");

    let authority = workspace.authority_handle();
    let tracegen_preparation_ms = tracegen_start.elapsed().as_millis();
    mark("prepare_recursion_tracegen_record_compact_complete");
    Ok(PreparedCompactRecursionTracegenRecord {
        record: workspace.into_record(),
        authority,
        record_generation,
        pool_gate_ms,
        tracegen_preparation_ms,
    })
}

struct PreparedTracegenWorkspace {
    workspace: TracegenWorkspace,
    plan: crate::validate::ExactTracePlan,
    tracegen_start: Instant,
}

pub fn prepare_recursion_tracegen_record_with_timing<C>(
    prover: &NativeProverFor<C>,
    record: FinalizedRecord,
    program: &RecursionNativeProgram<F>,
    mut mark: impl FnMut(&str),
) -> NativeRecursionAssemblyResult<PreparedRecursionTracegenRecord>
where
    C: NativeLayerProofConfig + SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>,
    NativeProverFor<C>:
        SCMachineProver<C, NativeRecursionAir, D_EF, DeviceProvingKey = SCStarkProvingKey<C>>,
{
    mark("enter_prepare_recursion_tracegen_record");
    let record_generation = record.generation();
    mark("read_record_generation");
    let node = FinalizedNode::admit(prover, record, program)?;
    mark("FinalizedNode::admit");
    let pool_gate_ms = node.admit_ms;
    let prepared = prepare_tracegen_workspace_from_node(prover, node, program, &mut mark)?;
    let tracegen_preparation_ms = prepared.tracegen_start.elapsed().as_millis();
    let authority = prepared.workspace.authority_handle();
    mark("prepare_recursion_tracegen_record_complete");
    Ok(PreparedRecursionTracegenRecord {
        record: prepared.workspace.into_record(),
        plan: prepared.plan,
        authority,
        record_generation,
        pool_gate_ms,
        tracegen_preparation_ms,
    })
}

fn prepare_tracegen_workspace_from_node<C, PROV>(
    prover: &PROV,
    node: FinalizedNode,
    program: &RecursionNativeProgram<F>,
    mark: &mut impl FnMut(&str),
) -> NativeRecursionAssemblyResult<PreparedTracegenWorkspace>
where
    C: NativeLayerProofConfig + SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>,
    PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
{
    let tracegen_start = Instant::now();
    let FinalizedNode { input, .. } = node;
    let mut workspace = input.into_workspace().map_err(NativeRecursionAssemblyError::Validation)?;
    mark("TracegenInput::into_workspace");
    let authority_telemetry_start = Instant::now();
    record_preparation_event_telemetry(workspace.record());
    workspace.record().profile.add_record_split(
        "tracegen.authority_telemetry",
        authority_telemetry_start.elapsed().as_millis(),
    );
    mark("record_preparation_event_telemetry");
    if workspace.record().poseidon2_tracegen.retained_rows() != 0 ||
        workspace.record().poseidon2_tracegen.generated_rows() != 0
    {
        return Err(NativeRecursionAssemblyError::Validation(
            "Poseidon2 full-witness cache was populated before tracegen".to_string(),
        ));
    }
    mark("validate_empty_poseidon2_tracegen_cache");
    let expansion_start = Instant::now();
    materialize_whir_tracegen_sources(workspace.record_mut()).map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("WHIR tracegen expansion: {err:?}"))
    })?;
    mark("materialize_whir_tracegen_sources");
    let raw = workspace.record_mut();
    finalize_provider_requests_at_source(
        raw,
        StatementDigestMode::from_role(program.statement_role),
    );
    mark("finalize_provider_requests_at_source");
    let provider_input = provider_telemetry(raw);
    mark("provider_telemetry_input");
    publish_provider_input_telemetry(raw, &provider_input);
    mark("publish_provider_input_telemetry");
    let fused_witness_ms = u128::from(raw.poseidon2_tracegen.generation_nanos()) / 1_000_000;
    raw.profile.add_record_split("tracegen.full_poseidon2_witness", fused_witness_ms);
    raw.profile.set_structural_counter(
        "poseidon2_full_witness_rows_fused_with_derived_expansion",
        raw.poseidon2_tracegen.generated_rows(),
    );
    mark("record_poseidon2_tracegen_witness_profile");
    if run_final_residuals() {
        timed_residual_assert(raw, program, "tracegen_materialized")?;
    }
    mark("final_residuals_after_tracegen_materialized");
    raw.profile
        .add_record_split("tracegen.semantic_row_expansion", expansion_start.elapsed().as_millis());
    mark("record_semantic_row_expansion_profile");
    workspace.validate().map_err(NativeRecursionAssemblyError::Validation)?;
    mark("TracegenWorkspace::validate_after_expansion");
    if !workspace.record().provider_requests_finalized {
        return Err(NativeRecursionAssemblyError::Validation(
            "providers were not finalized by tracegen".to_string(),
        ));
    }
    mark("validate_provider_requests_finalized");
    let provider_seal_start = Instant::now();
    let provider_input_seal =
        workspace.seal_provider_inputs().map_err(NativeRecursionAssemblyError::Validation)?;
    let profile = &workspace.record().profile;
    profile.add_record_split(
        "provider_input_descriptor_seal",
        provider_seal_start.elapsed().as_millis(),
    );
    profile.set_structural_counter(
        "provider_input_seal_segments",
        u64::try_from(provider_input_seal.segment_count)
            .expect("provider input seal count exceeds u64"),
    );
    profile.set_structural_counter(
        "provider_input_seal_entries",
        u64::try_from(provider_input_seal.entry_count)
            .expect("provider input seal entry count exceeds u64"),
    );
    profile.set_structural_counter(
        "provider_input_seal_retained_bytes",
        u64::try_from(provider_input_seal.retained_bytes)
            .expect("provider input seal byte count exceeds u64"),
    );
    mark("seal_provider_inputs");
    let reduce_start = Instant::now();
    let stats = workspace
        .record_mut()
        .reduce_provider_inputs()
        .map_err(NativeRecursionAssemblyError::Validation)?;
    let profile = &workspace.record().profile;
    profile.add_record_split(
        "tracegen.provider_enumeration_reduce",
        reduce_start.elapsed().as_millis(),
    );
    profile.set_structural_counter("tracegen_provider_reduce_passes", u64::from(stats.passes));
    profile.set_structural_counter(
        "provider_unique_count",
        u64::try_from(stats.unique_entries).expect("provider unique count exceeds u64"),
    );
    profile.set_structural_counter(
        "provider_duplicate_entries",
        u64::try_from(stats.duplicate_entries).expect("provider duplicate count exceeds u64"),
    );
    let provider_output = provider_telemetry(workspace.record());
    let exact_multiplicity_sum = provider_input
        .family_multiplicity_units
        .iter()
        .try_fold(0u64, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            NativeRecursionAssemblyError::Validation(
                "provider multiplicity sum overflow".to_string(),
            )
        })?;
    let reduction_summary = TracegenReductionSummary::new(
        &workspace,
        provider_input_seal,
        u64::try_from(stats.raw_entries).expect("raw provider count exceeds u64"),
        u64::try_from(stats.unique_entries).expect("unique provider count exceeds u64"),
        u64::try_from(stats.duplicate_entries).expect("duplicate provider count exceeds u64"),
        exact_multiplicity_sum,
        stats.passes,
    );
    reduction_summary
        .validate(&workspace, provider_input_seal)
        .map_err(NativeRecursionAssemblyError::Validation)?;
    profile.set_structural_counter("provider_prereduce_passes", 0);
    profile.set_structural_counter("raw_provider_entries", provider_input.entries);
    profile.set_structural_counter("provider_exact_multiplicity_sum", exact_multiplicity_sum);
    profile.set_structural_counter("provider_reducer_input_bytes", provider_input.bytes);
    profile.set_structural_counter("provider_reducer_output_bytes", provider_output.bytes);
    profile.set_structural_counter(
        "provider_reducer_temp_bytes_upper_bound",
        provider_input
            .bytes
            .checked_add(provider_output.bytes)
            .expect("provider reducer byte counter overflow"),
    );
    profile.set_structural_counter("provider_rows", provider_output.entries);
    profile.set_structural_counter(
        "provider_padded_height",
        provider_padded_height(workspace.record()),
    );
    tracing::info!(
        raw_provider_entries = stats.raw_entries,
        provider_unique_count = stats.unique_entries,
        provider_duplicate_entries = stats.duplicate_entries,
        tracegen_provider_reduce_passes = stats.passes,
        "native recursion provider reduction"
    );
    mark("provider_reduction");

    let dependency_started = Instant::now();
    let transcript_owner_started = Instant::now();
    workspace.install_transcript_owner().map_err(NativeRecursionAssemblyError::Validation)?;
    let transcript_owner_elapsed = transcript_owner_started.elapsed();
    mark("install_transcript_owner");
    let raw = workspace.record();
    let constraint_started = Instant::now();
    let constraint_counts = crate::constraint_replay_dt::trace::prepare_constraint_authority(
        raw,
        &program.constraint_program,
    );
    let constraint_elapsed = constraint_started.elapsed();
    mark("prepare_constraint_authority");
    let statement_start = Instant::now();
    let statement_rows =
        statement_rows_cached(raw, program.statement_role, &program.statement_config);
    let statement_hash_mode = StatementDigestMode::from_role(program.statement_role);
    let statement_hash_rows = statement_hash_rows_cached(raw, statement_hash_mode);
    let transcript_rows = transcript_sponge_rows_cached(raw);
    let statement_elapsed = statement_start.elapsed();
    let dependency_elapsed = dependency_started.elapsed();
    mark("prepare_statement_and_transcript_exact_rows");
    let constraint_us = u64::try_from(constraint_elapsed.as_micros()).unwrap_or(u64::MAX);
    let preflight_max_us = raw.profile.structural_counter("preflight.total_max_us").unwrap_or(0);
    let combined_preflight_dependency_us = preflight_max_us.saturating_add(constraint_us);
    raw.profile.add_record_split(
        "tracegen.sequential_constraint_derivation",
        constraint_elapsed.as_millis(),
    );
    raw.profile.add_record_split(
        "tracegen.constraint_dependency_heavy_construction",
        constraint_elapsed.as_millis(),
    );
    raw.profile.add_record_split(
        "tracegen.transcript_exact_row_owner_transition",
        transcript_owner_elapsed.as_millis(),
    );
    raw.profile.add_record_split("tracegen.statement_exact_rows", statement_elapsed.as_millis());
    raw.profile.add_record_split(
        "tracegen.post_seal_exact_artifact_derivation",
        dependency_elapsed.as_millis(),
    );
    let constraint_dynamic_bytes =
        raw.profile.structural_counter("constraint_dynamic_artifact_bytes").unwrap_or(0);
    let transcript_artifact_bytes = transcript_rows
        .len()
        .saturating_mul(core::mem::size_of::<crate::system_dt::SpecSpongeBlock>());
    let statement_artifact_bytes =
        statement_rows
            .len()
            .saturating_mul(core::mem::size_of::<
                crate::statement_boundary_air_dt::StatementBoundaryRow,
            >())
            .saturating_add(statement_hash_rows.len().saturating_mul(core::mem::size_of::<
                crate::statement_hash_air_dt::StatementHashRow,
            >()));
    let workspace_artifact_bytes = u64::try_from(
        transcript_artifact_bytes.saturating_add(statement_artifact_bytes).saturating_add(
            core::mem::size_of::<crate::system_dt::record::TracegenWorkspaceArtifacts>(),
        ),
    )
    .unwrap_or(u64::MAX)
    .saturating_add(constraint_dynamic_bytes);
    raw.profile.set_structural_counters([
        ("constraint.pre_seal_derivation_us", 0),
        ("constraint.post_seal_derivation_us", constraint_us),
        ("constraint.combined_sequential_derivation_us", constraint_us),
        ("pre_seal_constraint_derivation_us", 0),
        ("post_seal_sequential_constraint_derivation_us", constraint_us),
        ("combined_preflight_and_constraint_derivation_us", combined_preflight_dependency_us),
        (
            "transcript_exact_row_owner_transition_us",
            u64::try_from(transcript_owner_elapsed.as_micros()).unwrap_or(u64::MAX),
        ),
        (
            "statement_post_seal_exact_rows_us",
            u64::try_from(statement_elapsed.as_micros()).unwrap_or(u64::MAX),
        ),
        (
            "post_seal_exact_artifact_derivation_us",
            u64::try_from(dependency_elapsed.as_micros()).unwrap_or(u64::MAX),
        ),
        (
            "transcript_workspace_artifact_bytes",
            u64::try_from(transcript_artifact_bytes).unwrap_or(u64::MAX),
        ),
        (
            "statement_workspace_artifact_bytes",
            u64::try_from(statement_artifact_bytes).unwrap_or(u64::MAX),
        ),
        ("tracegen_workspace_artifact_retained_bytes", workspace_artifact_bytes),
        ("statement_static_bytes_duplicated_per_case", 0),
        ("statement_config_authority_bytes_per_case", core::mem::size_of::<u64>() as u64),
        (
            "statement_authoritative_rows_or_events",
            u64::try_from(statement_rows.len() + statement_hash_rows.len())
                .expect("statement authority row count exceeds u64"),
        ),
        (
            "transcript_authoritative_rows_or_events",
            u64::try_from(transcript_rows.len()).expect("transcript row count exceeds u64"),
        ),
        (
            "unpadded_rows_by_family.statement",
            u64::try_from(statement_rows.len()).expect("statement row count exceeds u64"),
        ),
        (
            "unpadded_rows_by_family.statement_hash",
            u64::try_from(statement_hash_rows.len()).expect("statement hash row count exceeds u64"),
        ),
        (
            "unpadded_rows_by_family.transcript_sponge",
            u64::try_from(transcript_rows.len()).expect("transcript row count exceeds u64"),
        ),
        (
            "tracegen_workspace_artifact_slots_initialized",
            u64::try_from(raw.tracegen_artifacts.initialized_entries())
                .expect("artifact slot count exceeds u64"),
        ),
        (
            "transcript_exact_row_owner_count",
            u64::from(raw.tracegen_artifacts.transcript_sponge.get().is_some()),
        ),
        (
            "statement_case_owner_count",
            u64::from(
                raw.tracegen_artifacts.statement.get().is_some() &&
                    raw.tracegen_artifacts.statement_hash.get().is_some(),
            ),
        ),
    ]);
    if let Some(peak_rss_kb) = peak_rss_kb() {
        raw.profile.set_structural_counter("peak_tracegen_workspace_rss_kb", peak_rss_kb);
    }
    for (family, rows) in [
        ("constraint_dag", constraint_counts.dag),
        ("constraint_fold", constraint_counts.fold),
        ("constraint_challenge", constraint_counts.challenge),
        ("constraint_beta_ladder", constraint_counts.beta_ladder),
        ("constraint_terminal", constraint_counts.terminal),
    ] {
        raw.profile.set_structural_counter(
            format!("unpadded_rows_by_family.{family}"),
            u64::try_from(rows).expect("constraint authority row count exceeds u64"),
        );
    }
    mark("record_exact_artifact_profile");
    let admission_start = Instant::now();
    let plan = crate::validate::exact_pre_trace_gate(
        prover,
        raw,
        prover.config().mlpcs_stack_log_height_hint(),
    )
    .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
    raw.profile.add_record_split("tracegen.exact_admission", admission_start.elapsed().as_millis());
    raw.profile.add_record_split("tracegen.exact_row_count_admission", plan.row_count_admission_ms);
    mark("exact_pre_trace_gate");
    check_provider_pools(raw)
        .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
    raw.profile.set_structural_counter("constraint_matrix_population_bytes", 0);
    mark("check_provider_pools");
    Ok(PreparedTracegenWorkspace { workspace, plan, tracegen_start })
}

/// S6 terminal typestate: the generated traces of one admitted node, already
/// asserted equal to the node's [`ExactTracePlan`] per chip and bidirectional.
/// Move-only and not `Clone`: the traces leave exactly once, by value, into
/// commit — there is no host-matrix round trip after this point.
pub(crate) struct TraceBundle {
    workspace: TracegenWorkspace,
    traces: Vec<(String, CompressedMatrix<F>)>,
    plan: crate::validate::ExactTracePlan,
    tracegen_ms: u128,
    match_ms: u128,
}

impl TraceBundle {
    fn materialize<C, PROV>(
        prover: &PROV,
        node: FinalizedNode,
        program: &RecursionNativeProgram<F>,
    ) -> NativeRecursionAssemblyResult<Self>
    where
        C: NativeLayerProofConfig
            + SCStarkGenericConfig
            + StarkGenericConfig<Val = F, Challenge = EF>,
        PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
    {
        let mut no_timing = |_step: &str| {};
        let prepared = prepare_tracegen_workspace_from_node(prover, node, program, &mut no_timing)?;
        let PreparedTracegenWorkspace { workspace, plan, tracegen_start } = prepared;
        let raw = workspace.record();
        let matrix_start = Instant::now();
        // Reset the request-local accumulator immediately before the seven constraint generators
        // add their actual allocated main-matrix byte counts (one add per realized matrix).
        let traces = prover.generate_traces(raw);
        raw.profile.add_record_split(
            "tracegen.padding_and_matrix_population",
            matrix_start.elapsed().as_millis(),
        );
        let tracegen_ms = tracegen_start.elapsed().as_millis();
        raw.profile.add_record_split("tracegen.total", tracegen_ms);
        let match_start = Instant::now();
        crate::validate::check_traces_match_plan(&plan, &traces)
            .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
        Ok(Self {
            workspace,
            traces,
            plan,
            tracegen_ms,
            match_ms: match_start.elapsed().as_millis(),
        })
    }

    /// Surrender the workspace and traces together, by value, exactly once at commit.
    fn into_parts(self) -> (TracegenWorkspace, Vec<(String, CompressedMatrix<F>)>) {
        (self.workspace, self.traces)
    }
}

pub(crate) fn prove_recursion_with_metrics<C, PROV>(
    prover: &PROV,
    pk: &SCStarkProvingKey<C>,
    device_pk: &PROV::DeviceProvingKey,
    record: FinalizedRecord,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<(SCMachineProof<C>, ProveRecursionMetrics)>
where
    C: NativeLayerProofConfig,
    PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
    MlCom<C>: Send + Sync,
    MlPcsProverData<C>: Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    let record_generation = record.generation();
    let node = FinalizedNode::admit(prover, record, program)?;
    let pool_gate_ms = node.admit_ms;
    let bundle = TraceBundle::materialize(prover, node, program)?;
    let planned_chip_log_heights = bundle
        .plan
        .chips
        .iter()
        .map(|chip| {
            debug_assert!(chip.total_height.is_power_of_two());
            (chip.chip.clone(), chip.total_height.ilog2() as u8)
        })
        .collect();

    let tracegen_ms = bundle.tracegen_ms;
    let row_count_admission_ms = bundle.plan.row_count_admission_ms;
    let trace_plan_fold_ms = bundle.plan.plan_fold_ms;
    let traces = &bundle.traces;

    // Opt-in trace digests (DT_NATIVE_D12_TRACE_DIGEST=1): one stable digest per
    // chip trace per prove. Diffing two identical runs on these lines localizes
    // any run-to-run trace nondeterminism to a (machine, chip) before the
    // transcript ever sees it.
    if std::env::var("DT_NATIVE_D12_TRACE_DIGEST").is_ok() {
        for (name, trace) in traces.iter() {
            let mut acc: u64 = 0xcbf29ce484222325;
            for value in &trace.main.values {
                acc ^= u64::from(value.as_canonical_u32());
                acc = acc.wrapping_mul(0x100000001b3);
            }
            println!(
                "native_d12_trace_digest role={:?} chip={} stored={} digest={acc:016x}",
                prover.machine().contains_global_bus,
                name,
                trace.stored_height(),
            );
        }
    }

    let trace_costs = traces
        .iter()
        .map(|(name, trace)| {
            let chip = prover
                .machine()
                .chips
                .iter()
                .find(|chip| chip.name() == *name)
                .expect("generated trace chip must exist in native recursion machine");
            RecursionTraceCost {
                chip: name.clone(),
                height: trace.total_height,
                stored_height: trace.stored_height(),
                width: trace.main.width(),
                perm_width: chip.perm_width(),
                interactions: chip.num_lookup(),
                constraints: chip.num_alpha,
            }
        })
        .collect::<Vec<_>>();

    // Planned == realized already ran inside TraceBundle::materialize (per
    // chip, bidirectional, against the admission plan).
    let budget_ms = bundle.match_ms;

    if bundle.workspace.record().statement_public_values.is_none() {
        return Err(NativeRecursionAssemblyError::Validation(
            "native recursion statement public values missing".to_string(),
        ));
    }

    let mut challenger = prover.config().mlchallenger();
    pk.observe_into(&mut challenger);
    pcs::whir::profile::reset();
    let commit_start = Instant::now();
    // The bundle is surrendered by value: commit is the one consumer of the
    // generated matrices (no host round trip after this line).
    let (workspace, traces) = bundle.into_parts();
    let raw = workspace.record();
    let shard_data =
        prover.commit_with_pcs_stack_log_height(raw, traces, pk.preprocessed_pcs_stack_log_height);
    let commit_ms = commit_start.elapsed().as_millis();
    let commit_profile = pcs::whir::profile::take();

    pcs::whir::profile::reset();
    let open_start = Instant::now();
    let shard = prover
        .open(
            device_pk,
            shard_data,
            &mut challenger,
            NATIVE_RECURSION_NUM_SKIP_ROUNDS,
            NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD,
        )
        .map_err(|err| NativeRecursionAssemblyError::Prove(err.to_string()))?;
    let open_ms = open_start.elapsed().as_millis();
    let open_profile = pcs::whir::profile::take();
    let poseidon2_memo = raw.poseidon2_memo.snapshot();
    let record_profile = raw.profile.snapshot();

    Ok((
        SCMachineProof { shard_proofs: vec![shard] },
        ProveRecursionMetrics {
            timings: ProveRecursionTimings {
                record_generation,
                record_profile,
                poseidon2_memo,
                planned_chip_log_heights,
                row_count_admission_ms,
                trace_plan_fold_ms,
                tracegen_ms,
                budget_ms,
                pool_gate_ms,
                commit_ms,
                commit_profile,
                open_ms,
                open_profile,
            },
            trace_costs,
        },
    ))
}

pub fn verify_recursion<C, PROV>(
    prover: &PROV,
    vk: &SCStarkVerifyingKey<C>,
    proof: &SCMachineProof<C>,
) -> NativeRecursionAssemblyResult<()>
where
    C: SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>,
    PROV: SCMachineProver<C, NativeRecursionAir, D_EF>,
    C::MlChallenger: Clone,
{
    verify_recursion_machine(prover.machine(), vk, proof)
}

/// Verifies one native-recursion machine proof without constructing or retaining a prover.
pub fn verify_recursion_machine<C>(
    machine: &polyair::SCStarkMachine<C, NativeRecursionAir, D_EF>,
    vk: &SCStarkVerifyingKey<C>,
    proof: &SCMachineProof<C>,
) -> NativeRecursionAssemblyResult<()>
where
    C: SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>,
    C::MlChallenger: Clone,
{
    let mut challenger = machine.config().mlchallenger();
    machine
        .verify(
            vk,
            proof,
            &mut challenger,
            NATIVE_RECURSION_NUM_SKIP_ROUNDS,
            NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD,
        )
        .map_err(|err| NativeRecursionAssemblyError::Verify(err.to_string()))
}

/// Verifies the single shard of a terminal L4 proof by reference.
///
/// This is the allocation-free companion to [`verify_recursion_machine`] for the root product:
/// it performs the same VK observation, chip selection, and PolyAir shard verification without
/// first cloning the (potentially large) shard into an [`SCMachineProof`]. Root machines never
/// carry the inter-shard Global bus; the statement-level Global interval is checked separately by
/// the native root verifier.
pub fn verify_root_recursion_shard(
    machine: &NativeRootMachine,
    vk: &SCStarkVerifyingKey<RootSC>,
    shard: &SCShardProof<RootSC>,
) -> NativeRecursionAssemblyResult<()> {
    if machine.contains_global_bus {
        return Err(NativeRecursionAssemblyError::Verify(
            "terminal root verifier machine unexpectedly contains the Global bus".to_string(),
        ));
    }
    if machine.global_boundary_registry != vk.owner_registry ||
        vk.owner_registry.validate().is_err()
    {
        return Err(NativeRecursionAssemblyError::Verify(
            "root machine and verification key owner registries differ".to_string(),
        ));
    }
    if shard.public_values.len() < machine.num_pv_elts() {
        return Err(NativeRecursionAssemblyError::Verify(format!(
            "root shard has {} public values, expected at least {}",
            shard.public_values.len(),
            machine.num_pv_elts(),
        )));
    }

    let mut challenger = machine.config().mlchallenger();
    vk.observe_into(&mut challenger);
    challenger.observe_slice(&shard.public_values[..machine.num_pv_elts()]);
    let chips = machine.shard_chips_ordered(&shard.chip_ordering).collect::<Vec<_>>();
    polyair::verifier::Verifier::<RootSC, NativeRecursionAir, D_EF>::verify_shard(
        machine.config(),
        vk,
        &chips,
        &mut challenger,
        shard,
        NATIVE_RECURSION_NUM_SKIP_ROUNDS,
        NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD,
        false,
    )
    .map_err(|err| NativeRecursionAssemblyError::Verify(err.to_string()))
}

pub fn core_recording_machine() -> CoreRecordingMachine {
    RiscvPolyAir::<F>::sc_machine::<RecordingSC, D_EF>(RecordingSC::for_stage(RecordingStage::Core))
}

pub fn native_recording_machine(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecordingMachine> {
    native_recording_machine_for_stage(program, RecordingStage::Compress)
}

/// Native recording machine at an explicit child-recording stage: Compress for lift/L2/L3
/// children, Shrink for L4's L3-at-shrink children. Core is not a native stage, and
/// root_shrink is rejected upstream (`RecordingStage::from_whir_stage`) — L4 proofs are
/// terminal artifacts, never recorded as children.
pub fn native_recording_machine_for_stage(
    program: &RecursionNativeProgram<F>,
    stage: RecordingStage,
) -> NativeRecursionAssemblyResult<NativeRecordingMachine> {
    let params = program.layer()?.params();
    validate_program_matches_layer(program, params)?;
    validate_recording_stage_for_layer(stage, params)?;
    // The Shrink recording stage mirrors the shrink-floored L3 prover.
    // Note: the recorded verify walk and the L4 program IR compiled from this machine
    // must see the same degree/batch profile, or the replayed constraint set diverges
    // from the actual L3 proof.
    let floor = if stage == RecordingStage::Shrink { NATIVE_SHRINK_DEGREE_FLOOR } else { 0 };
    let chips = NativeRecursionAir::all(program)?
        .into_iter()
        .map(|air| polyair::Chip::<NativeRecursionAir, F, D_EF>::new_with_degree_floor(air, floor))
        .collect();
    Ok(polyair::SCStarkMachine::new(
        RecordingSC::for_stage(stage),
        chips,
        NATIVE_RECURSION_NUM_PV_ELTS,
        false,
    ))
}

/// The L3 (shrink) prover carries the same degree-3 floor as the root. L4
/// derives its child transcript/replay width from the compiled L3 IR.
pub(crate) fn native_shrink_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    native_shrink_prover_with_provider::<CpuNativeProver>(program)
}

pub(crate) fn native_shrink_prover_with_provider<P: NativeProverProvider>(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver<P>> {
    let params = program.layer()?.params();
    validate_program_matches_layer(program, params)?;
    let config = SC::shrink();
    validate_proof_config_for_layer(&config, params)?;
    let chips: Vec<polyair::Chip<NativeRecursionAir, F, D_EF>> = NativeRecursionAir::all(program)?
        .into_iter()
        .map(|air| {
            polyair::Chip::<NativeRecursionAir, F, D_EF>::new_with_degree_floor(
                air,
                NATIVE_SHRINK_DEGREE_FLOOR,
            )
        })
        .collect();
    print_chip_batch_profile("shrink_l3", &chips);
    let machine = polyair::SCStarkMachine::new(config, chips, NATIVE_RECURSION_NUM_PV_ELTS, false);
    Ok(<<P as NativeProverProvider>::SCProver as SCMachineProver<SC, NativeRecursionAir, D_EF>>::new(machine))
}

pub fn native_child_verifier_config() -> NativeChildVerifierConfigView {
    native_child_verifier_config_for_role(NativeChildRole::Core)
}

pub fn native_child_verifier_config_for_role(
    role: NativeChildRole,
) -> NativeChildVerifierConfigView {
    let fri = match role {
        NativeChildRole::Core => default_fri_config(),
        NativeChildRole::Compress => compressed_fri_config(),
        NativeChildRole::Shrink => shrink_fri_config(),
    };
    NativeChildVerifierConfigView {
        role,
        num_skip_rounds: POLYAIR_NUM_SKIP_ROUNDS,
        chip_log_height_threshold: POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD,
        whir: NativeWhirConfigView {
            log_blowup: fri.log_blowup,
            num_queries: fri.num_queries,
            grinding_bits_query: fri.grinding_bits_query,
            grinding_bits_batching: fri.grinding_bits_batching,
        },
    }
}

pub fn build_native_recursion_program<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
    statement_role: RecursionStatementRole,
    child_role: RecursionChildRole,
    num_child_public_values: usize,
    child_contains_global_bus: bool,
    statement_config: Vec<StatementConfigRow>,
) -> NativeRecursionAssemblyResult<RecursionNativeProgram<F>>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    validate_role_matrix(child_role, num_child_public_values, child_contains_global_bus)?;
    validate_statement_role(child_role, statement_role)?;
    validate_statement_config(statement_role, &statement_config)?;
    if machine.num_pv_elts() != num_child_public_values {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "child machine num_pv_elts={} expected {} for {:?}",
            machine.num_pv_elts(),
            num_child_public_values,
            child_role
        )));
    }
    if machine.contains_global_bus != child_contains_global_bus {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "child machine contains_global_bus={} expected {} for {:?}",
            machine.contains_global_bus, child_contains_global_bus, child_role
        )));
    }

    let fixed_chips = compile_constraint_segment(machine, 0)?;
    let max_required_beta_power = segment_max_beta_power(&fixed_chips);
    let constraint_program = RecursionPolyAirVerifierProgramDto {
        version: crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
        role: child_role,
        artifact_digest: [F::zero(); DIGEST_SIZE],
        chips: fixed_chips,
        max_required_beta_power,
    };
    let native_role = native_child_role(child_role);
    let native_chip_metadata = segment_metadata_universe(machine, native_role, 0);
    RecursionNativeProgram::try_from_constraint_dto(
        child_role,
        statement_role,
        num_child_public_values,
        child_contains_global_bus,
        native_chip_metadata,
        constraint_program,
        statement_config,
        child_role == RecursionChildRole::Compress &&
            statement_role == RecursionStatementRole::ReduceL2,
    )
    .map_err(NativeRecursionAssemblyError::InvalidProgram)
}

fn compile_constraint_segment<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
    static_chip_id_offset: usize,
) -> NativeRecursionAssemblyResult<Vec<RecursionPolyAirChipIr>>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let static_ids = proof_shape_static_chip_id_map(machine);
    let mut fixed_chips = Vec::new();
    for chip in &machine.chips {
        let name = chip.name();
        let Some(static_chip_id) = static_ids.get(&name).copied() else {
            return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                "missing static chip id for {name}"
            )));
        };
        let fixed = RecursionFixedSymbolicChip::from_symbolic_builder(
            static_chip_id + static_chip_id_offset,
            name.clone(),
            chip.commit_scope(),
            chip.logup_batch_size(),
            chip.num_alpha,
            &chip.symbolic_builder,
        )
        .map_err(|err| {
            NativeRecursionAssemblyError::InvalidProgram(format!(
                "fixed symbolic compile failed for {name}: {err:?}"
            ))
        })?;
        let ir = RecursionPolyAirChipIr::compile(&fixed).map_err(|err| {
            NativeRecursionAssemblyError::InvalidProgram(format!(
                "symbolic IR compile failed for {name}: {err:?}"
            ))
        })?;
        fixed_chips.push(ir);
    }
    fixed_chips.sort_by_key(|chip| chip.static_chip_id);
    Ok(fixed_chips)
}

fn segment_max_beta_power(chips: &[RecursionPolyAirChipIr]) -> usize {
    chips
        .iter()
        .flat_map(|chip| chip.derived_roots.iter())
        .filter_map(|root| match root {
            RecursionPolyAirDerivedRoot::BetaPower { power } => Some(*power),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn segment_metadata_universe<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
    native_role: NativeChildRole,
    static_chip_id_offset: usize,
) -> Vec<crate::system_dt::RecursionNativeChipMetadataRequest>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let metadata = native_metadata_from_machine(machine);
    let metadata_view = NativeChildMetadataView {
        role: native_role,
        air_authority: NativeAirAuthority::PublicMetadata,
        num_observed_public_values: machine.num_pv_elts(),
        contains_global_bus: machine.contains_global_bus,
        static_chip_id_offset,
        chips: &metadata,
    };
    metadata_universe_from_view(role_id(native_role), &metadata_view)
}

/// Static chip id offset of the second (reduce-child) replay segment in a dual-segment
/// mixed program. Per-machine universes stay within 256 ids, so one stride keeps both
/// segments inside the proof-shape id budget.
pub const MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET: usize = 128;

/// Builds the dual-segment ReduceL2 program for mixed nodes: segment A = the lift
/// child machine's replay universe at offset 0, segment B = the reduce child
/// machine's universe at the mixed segment offset.
pub fn build_mixed_reduce_program(
    lift_child_machine: &NativeRecordingMachine,
    reduce_child_machine: &NativeRecordingMachine,
    statement_config: Vec<StatementConfigRow>,
) -> NativeRecursionAssemblyResult<RecursionNativeProgram<F>> {
    build_dual_segment_reduce_program(
        lift_child_machine,
        reduce_child_machine,
        RecursionStatementRole::ReduceL2,
        statement_config,
    )
}

/// The canonical dual-segment reduce builder: L2 and L3 machines share the replay
/// segment set {u1@0 (lift-child universe), u2@128 (ReduceL2-child universe)} and
/// differ only in statement role + config. The two-pass L2 fixed point holds only when
/// the bootstrap and final programs derive the same child unipoly evaluation length.
/// [`validate_l2_bootstrap_fixed_point`] recompiles and checks that agreement during
/// uncached ladder construction.
pub fn build_dual_segment_reduce_program(
    lift_child_machine: &NativeRecordingMachine,
    reduce_child_machine: &NativeRecordingMachine,
    statement_role: RecursionStatementRole,
    statement_config: Vec<StatementConfigRow>,
) -> NativeRecursionAssemblyResult<RecursionNativeProgram<F>> {
    if !matches!(
        statement_role,
        RecursionStatementRole::ReduceL2 | RecursionStatementRole::ReduceL3
    ) {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "dual-segment reduce programs are ReduceL2/ReduceL3 only, got {statement_role:?}"
        )));
    }
    validate_statement_config(statement_role, &statement_config)?;
    for (label, machine) in
        [("lift-child", lift_child_machine), ("reduce-child", reduce_child_machine)]
    {
        if machine.num_pv_elts() != NATIVE_RECURSION_NUM_PV_ELTS || machine.contains_global_bus {
            return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                "mixed {label} machine is not a native ({NATIVE_RECURSION_NUM_PV_ELTS}, cgb=false) machine"
            )));
        }
    }
    let mut chips = compile_constraint_segment(lift_child_machine, 0)?;
    chips.extend(compile_constraint_segment(
        reduce_child_machine,
        MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
    )?);
    chips.sort_by_key(|chip| chip.static_chip_id);
    let max_required_beta_power = segment_max_beta_power(&chips);
    let constraint_program = RecursionPolyAirVerifierProgramDto {
        version: crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
        role: RecursionChildRole::Compress,
        artifact_digest: [F::zero(); DIGEST_SIZE],
        chips,
        max_required_beta_power,
    };
    let mut native_chip_metadata =
        segment_metadata_universe(lift_child_machine, NativeChildRole::Compress, 0);
    native_chip_metadata.extend(segment_metadata_universe(
        reduce_child_machine,
        NativeChildRole::Compress,
        MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET,
    ));
    RecursionNativeProgram::try_from_constraint_dto(
        RecursionChildRole::Compress,
        statement_role,
        NATIVE_RECURSION_NUM_PV_ELTS,
        false,
        native_chip_metadata,
        constraint_program,
        statement_config,
        false,
    )
    .map_err(NativeRecursionAssemblyError::InvalidProgram)
}

/// Recompiles the final L2 machine through the same segment compiler used to embed u2,
/// then checks that the bootstrap-produced segment is exactly the final machine's segment.
///
/// This is intentionally called only while building an uncached ladder. In particular, it
/// must not become part of cache loading or per-proof admission.
pub(crate) fn validate_l2_bootstrap_fixed_point(
    final_l2_machine: &NativeRecordingMachine,
    l2_program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    let final_u2 =
        compile_constraint_segment(final_l2_machine, MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET)?;
    let embedded_u2 = l2_program
        .constraint_program
        .chips
        .iter()
        .filter(|chip| chip.static_chip_id >= MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET)
        .collect::<Vec<_>>();
    let final_u2_bytes = bincode::serialize(&final_u2).map_err(|err| {
        NativeRecursionAssemblyError::InvalidProgram(format!(
            "serialize recompiled final L2 segment: {err}"
        ))
    })?;
    let embedded_u2_bytes = bincode::serialize(&embedded_u2).map_err(|err| {
        NativeRecursionAssemblyError::InvalidProgram(format!(
            "serialize embedded bootstrap L2 segment: {err}"
        ))
    })?;
    if final_u2_bytes != embedded_u2_bytes {
        return Err(NativeRecursionAssemblyError::InvalidProgram(
            "L2 bootstrap fixed-point mismatch in embedded constraint segment".to_string(),
        ));
    }
    Ok(())
}

/// Builds the L4 (root_shrink) program: single replay segment {u3@0} = the
/// ReduceL3-child universe; children are L3 proofs at the shrink config.
/// Note: L4 stays single-segment — root_shrink never verifies a lift directly.
pub fn build_root_shrink_program(
    l3_child_machine: &NativeRecordingMachine,
    statement_config: Vec<StatementConfigRow>,
) -> NativeRecursionAssemblyResult<RecursionNativeProgram<F>> {
    build_native_recursion_program(
        l3_child_machine,
        RecursionStatementRole::RootShrink,
        RecursionChildRole::Shrink,
        NATIVE_RECURSION_NUM_PV_ELTS,
        false,
        statement_config,
    )
}

pub fn build_core_native_recursion_program(
    machine: &CoreRecordingMachine,
) -> NativeRecursionAssemblyResult<RecursionNativeProgram<F>> {
    build_native_recursion_program(
        machine,
        RecursionStatementRole::Lift,
        RecursionChildRole::Core,
        dt_stark::air::DT_PROOF_NUM_PV_ELTS,
        true,
        Vec::new(),
    )
}

pub(crate) fn record_core_proof_shard<ChildSC>(
    machine: &CoreRecordingMachine,
    vk: &SCStarkVerifyingKey<ChildSC>,
    shard: SCShardProof<ChildSC>,
    proof_idx: usize,
    seed_challenger: <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<BuildingRecord>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
{
    record_child_proof_shard(machine, vk, shard, proof_idx, seed_challenger, program)
}

pub fn record_native_proof_shard<ChildSC>(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<ChildSC>,
    shard: SCShardProof<ChildSC>,
    proof_idx: usize,
    seed_challenger: <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<BuildingRecord>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
{
    record_child_proof_shard(machine, vk, shard, proof_idx, seed_challenger, program)
}

/// Records a native child whose replay universe lives at a non-zero static-chip-id offset
/// of a dual-segment (mixed) program.
pub fn record_native_proof_shard_in_segment<ChildSC>(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<ChildSC>,
    shard: SCShardProof<ChildSC>,
    proof_idx: usize,
    seed_challenger: <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
) -> NativeRecursionAssemblyResult<BuildingRecord>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
{
    record_child_proof_shard_with_offset(
        machine,
        vk,
        shard,
        proof_idx,
        seed_challenger,
        program,
        static_chip_id_offset,
    )
}

fn record_child_proof_shard<ChildSC, A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<ChildSC>,
    shard: SCShardProof<ChildSC>,
    proof_idx: usize,
    seed_challenger: <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<BuildingRecord>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
    A: MachineAir<F>,
{
    record_child_proof_shard_with_offset(machine, vk, shard, proof_idx, seed_challenger, program, 0)
}

/// Walk only the sequential Fiat--Shamir schedule needed to produce tracegen
/// material. Structural checks prevent indexing malformed containers; no
/// proof identity, opening, root, constraint, or PoW validity is checked.
fn produce_child_transcript_materials<ChildSC>(
    challenger: &mut <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    shard: &SCShardProof<ChildSC>,
    views: &NativeChildViews<'_, ChildSC>,
    query_grinding_bits: usize,
) -> Result<(), String>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
{
    let verifier_log_height =
        views.proof.verifier_round_log_height().map_err(|err| format!("round shape: {err:?}"))?;
    let round_shape = views
        .verifier_config
        .round_shape(verifier_log_height)
        .map_err(|err| format!("round shape: {err:?}"))?;
    if round_shape.num_rounds_nonlinear != 0 {
        return Err("nonlinear child sumcheck rounds are unsupported".to_string());
    }
    let num_rounds = round_shape.num_rounds;
    if shard.sumcheck_proof.unipolys.len() != num_rounds {
        return Err(format!(
            "child sumcheck round count: expected {num_rounds}, got {}",
            shard.sumcheck_proof.unipolys.len()
        ));
    }
    if shard
        .sumcheck_proof
        .unipolys
        .iter()
        .any(|unipoly| unipoly.evals.len() != crate::batch_constraint_dt::BATCH_SUMCHECK_EVALS)
    {
        return Err(format!(
            "child sumcheck eval width must be {}",
            crate::batch_constraint_dt::BATCH_SUMCHECK_EVALS
        ));
    }

    challenger.observe(shard.commitment.main_commit.clone());
    let active_shape = derive_active_shape_v1(views.proof.ordered_chips().map(|chip| {
        let metadata =
            views.layout.find_chip(chip.name).expect("validated child chip has fixed metadata");
        (chip.name.to_string(), metadata.main_width, chip.opened_values.log_height)
    }))
    .map_err(|err| format!("active shape: {err:?}"))?;
    observe_active_shape_v1::<F, _>(challenger, &active_shape);
    let _: EF = challenger.sample_ext_element();
    let _: EF = challenger.sample_ext_element();
    if let Some(permutation) = shard.commitment.permutation_commit.as_ref() {
        challenger.observe(permutation.clone());
    }
    for opening in &shard.opened_values.chips {
        challenger.observe_ext_element(opening.local_cumulative_sum);
    }
    let _: EF = challenger.sample_ext_element();
    for _ in 0..num_rounds {
        let _: EF = challenger.sample_ext_element();
    }
    for unipoly in &shard.sumcheck_proof.unipolys {
        for eval in &unipoly.evals {
            challenger.observe_ext_element(*eval);
        }
        let _: EF = challenger.sample_ext_element();
    }

    let proof = views.proof.opening_proof();
    if proof.grinding_batching_witness.len() != 2 || proof.grinding_query_witness.len() != 2 {
        return Err("WHIR PoW witness containers must each contain two elements".to_string());
    }
    if proof.sumcheck_transcript.uni_polys.len() != num_rounds ||
        proof.sumcheck_transcript.uni_polys.iter().any(|poly| poly.coeffs.len() != 3)
    {
        return Err(format!("WHIR sumcheck shape must be {num_rounds} degree-2 rounds"));
    }
    if proof.iopp_oracles.len() != num_rounds + 1 {
        return Err(format!(
            "WHIR oracle count: expected {}, got {}",
            num_rounds + 1,
            proof.iopp_oracles.len()
        ));
    }

    let mut height_groups = BTreeSet::new();
    for (batch_idx, batch) in shard.dimensions.iter().enumerate() {
        for (matrix_idx, dimension) in batch.iter().enumerate() {
            if dimension.height == 0 || !dimension.height.is_power_of_two() {
                return Err(format!(
                    "non-power-of-two dimension at batch {batch_idx}, matrix {matrix_idx}"
                ));
            }
            height_groups.insert(dimension.height.trailing_zeros() as usize);
        }
    }

    let _: EF = challenger.sample_ext_element();
    challenger.observe(proof.grinding_batching_witness[0]);
    challenger.observe(proof.grinding_batching_witness[1]);
    let _ = challenger.sample_bits(views.verifier_config.whir.grinding_bits_batching);
    challenger.observe(proof.iopp_oracles[0].clone());
    for (round_idx, poly) in proof.sumcheck_transcript.uni_polys.iter().enumerate() {
        if round_idx > 0 {
            challenger.observe(proof.iopp_oracles[round_idx].clone());
        }
        for coefficient in &poly.coeffs {
            challenger.observe_ext_element(*coefficient);
        }
        let _: EF = challenger.sample_ext_element();
        let merge_height = num_rounds - round_idx - 1;
        if height_groups.contains(&merge_height) {
            let _: EF = challenger.sample_ext_element();
        }
    }
    challenger.observe(proof.iopp_oracles[num_rounds].clone());
    challenger.observe(proof.grinding_query_witness[0]);
    challenger.observe(proof.grinding_query_witness[1]);
    let _ = challenger.sample_bits(query_grinding_bits);
    let query_bits = num_rounds + views.verifier_config.whir.log_blowup;
    for _ in 0..views.verifier_config.whir.num_queries {
        let _ = challenger.sample_bits(query_bits);
    }
    Ok(())
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[allow(clippy::too_many_arguments)]
fn record_child_proof_shard_with_offset<ChildSC, A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<ChildSC>,
    shard: SCShardProof<ChildSC>,
    proof_idx: usize,
    seed_challenger: <RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
) -> NativeRecursionAssemblyResult<BuildingRecord>
where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs:
        MlPCS<Commitment = MlCom<RecordingSC>, BatchProof = MlPcsOpeningProof<RecordingSC>>,
    A: MachineAir<F>,
{
    let prepare_start = Instant::now();
    seed_challenger.record().profile.mark_prepare_started(prepare_start);
    let static_context_start = Instant::now();
    if !program.constraint_program.has_matching_constraint_static_plan() {
        return Err(NativeRecursionAssemblyError::InvalidProgram(
            "recording requires an eagerly installed constraint static plan".to_string(),
        ));
    }
    if machine.num_pv_elts() != program.num_child_public_values ||
        machine.contains_global_bus != program.child_contains_global_bus
    {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "recording machine ({}, {}) does not match program {:?} ({}, {})",
            machine.num_pv_elts(),
            machine.contains_global_bus,
            program.role,
            program.num_child_public_values,
            program.child_contains_global_bus
        )));
    }
    let static_context_us = elapsed_us(static_context_start);
    let static_chip_id_offset_value = static_chip_id_offset;
    let view_start = Instant::now();
    let proof_view_start = Instant::now();
    let proof_view = crate::child_views::NativeChildProofView::new(&shard).map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("NativeChildProofView: {err:?}"))
    })?;
    let proof_view_us = elapsed_us(proof_view_start);
    let child_role = native_child_role(program.role);
    let layout = program
        .constraint_program
        .verified_child_layout(static_chip_id_offset_value)
        .ok_or_else(|| {
            NativeRecursionAssemblyError::InvalidProgram(format!(
                "missing verified child layout at static chip offset {static_chip_id_offset_value}"
            ))
        })?;
    if layout.role() != child_role ||
        layout.num_observed_public_values() != machine.num_pv_elts() ||
        layout.contains_global_bus() != machine.contains_global_bus
    {
        return Err(NativeRecursionAssemblyError::InvalidProgram(
            "verified child layout does not match recording role/machine context".to_string(),
        ));
    }
    let verifier_config = native_child_verifier_config_for_role(child_role);
    let view_binding_start = Instant::now();
    let views = NativeChildViews::from_proof_view(proof_view, vk, layout, &verifier_config)
        .map_err(|err| {
            NativeRecursionAssemblyError::Record(format!("NativeChildViews: {err:?}"))
        })?;
    let view_binding_us = elapsed_us(view_binding_start);
    let view_ms = view_start.elapsed().as_millis();

    // Record only the sequential transcript producer. All regular arithmetic
    // and cryptographic identities are materialized and constrained later.
    let transcript_start = Instant::now();
    let mut challenger = seed_challenger.into_for_proof(proof_idx);
    let transcript_memo_before = challenger.record().poseidon2_memo.snapshot();
    challenger.observe_slice(&shard.public_values[..machine.num_pv_elts()]);
    produce_child_transcript_materials(
        &mut challenger,
        &shard,
        &views,
        machine.config.whir_grinding_bits_query(),
    )
    .map_err(|err| NativeRecursionAssemblyError::Record(format!("transcript producer: {err}")))?;
    let transcript_walk_elapsed = transcript_start.elapsed();
    let transcript_walk_ms = transcript_walk_elapsed.as_millis();
    let transcript_walk_us = u64::try_from(transcript_walk_elapsed.as_micros()).unwrap_or(u64::MAX);

    let transcript_finalize_start = Instant::now();
    challenger.finish_transcript_capture().map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("transcript sponge capture: {err}"))
    })?;
    let transcript_capture_elapsed = transcript_finalize_start.elapsed();
    let transcript_capture_ms = transcript_capture_elapsed.as_millis();
    let transcript_capture_us =
        u64::try_from(transcript_capture_elapsed.as_micros()).unwrap_or(u64::MAX);
    let mut record = challenger.take_record();
    let transcript_memo_after = record.poseidon2_memo.snapshot();
    let transcript_memo_hits =
        transcript_memo_after.hits.saturating_sub(transcript_memo_before.hits);
    let transcript_memo_misses =
        transcript_memo_after.misses.saturating_sub(transcript_memo_before.misses);
    let assembly_start = Instant::now();
    let proof_shape_start = Instant::now();
    record_proof_shape_from_views(&mut record, proof_idx, &views, true)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("proof_shape: {err:?}")))?;
    record.proof_record_mut(proof_idx).proof_shape.publish_whir_inputs = true;
    record.proof_record_mut(proof_idx).proof_shape.publish_terminal_summary = true;
    let proof_shape_us = elapsed_us(proof_shape_start);
    let batch_start = Instant::now();
    record_batch_constraint_materials_from_views(&mut record, proof_idx, &views, true, true)
        .map_err(|err| {
            NativeRecursionAssemblyError::Record(format!("batch_constraint: {err:?}"))
        })?;
    let batch_us = elapsed_us(batch_start);
    let whir_start = Instant::now();
    let whir_header = prepare_whir_tracegen_materials(&record, proof_idx, &views, true)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("whir: {err:?}")))?;
    let admission_events = views.admission_events();
    drop(views);
    let SCShardProof { opening_proof, opened_values, dimensions, .. } = shard;
    attach_whir_tracegen_materials(
        &mut record,
        proof_idx,
        whir_header,
        opening_proof,
        opened_values,
        dimensions,
    )
    .map_err(|err| NativeRecursionAssemblyError::Record(format!("whir: {err:?}")))?;
    let proof_record = record
        .proof_records
        .iter_mut()
        .find(|proof| proof.proof_idx == proof_idx)
        .expect("recorded proof exists after WHIR source capture");
    annotate_child_constraint_replay_publications(proof_record, &program.constraint_program);
    annotate_child_statement_publications(proof_record);
    let whir_capture_us = elapsed_us(whir_start);
    // ProofShapeBinder is the sole native-metadata consumer. The authoritative
    // proof-shape capture above already recorded one request for every chip row.
    let pool_start = Instant::now();
    let pool_us = elapsed_us(pool_start);
    let assembly_us = elapsed_us(assembly_start);
    let semantic_prepare_us = elapsed_us(prepare_start);
    let record_splits = [
        ("child_view_validation", view_ms),
        ("transcript_producer_walk", transcript_walk_ms),
        ("transcript_first_pass_capture", transcript_capture_ms),
        (
            "fiat_shamir_and_transcript_capture",
            transcript_walk_ms.saturating_add(transcript_capture_ms),
        ),
        ("proof_shape_capture", u128::from(proof_shape_us) / 1_000),
        ("sumcheck_first_pass_capture", u128::from(batch_us) / 1_000),
        ("whir_material_capture", u128::from(whir_capture_us) / 1_000),
        ("metadata_pool_bookkeeping", u128::from(pool_us) / 1_000),
        ("whir_assembly", u128::from(assembly_us) / 1_000),
        ("semantic_event_construction", u128::from(assembly_us) / 1_000),
        ("raw_provider_publication", u128::from(pool_us) / 1_000),
        ("provider_publication_raw_append", u128::from(pool_us) / 1_000),
    ];
    let structural_counters = [
        ("preflight.static_context_binding_us", static_context_us),
        ("preflight.static_context_binding_max_us", static_context_us),
        ("preflight.proof_view_validation_us", proof_view_us),
        ("preflight.proof_view_validation_max_us", proof_view_us),
        ("preflight.view_binding_validation_us", view_binding_us),
        ("preflight.view_binding_validation_max_us", view_binding_us),
        ("preflight.transcript_producer_us", transcript_walk_us),
        ("preflight.transcript_producer_max_us", transcript_walk_us),
        ("preflight.transcript_finalize_us", transcript_capture_us),
        ("preflight.transcript_finalize_max_us", transcript_capture_us),
        ("preflight.proof_shape_capture_us", proof_shape_us),
        ("preflight.proof_shape_capture_max_us", proof_shape_us),
        ("preflight.batch_capture_us", batch_us),
        ("preflight.batch_capture_max_us", batch_us),
        ("preflight.whir_source_attach_us", whir_capture_us),
        ("preflight.whir_source_attach_max_us", whir_capture_us),
        ("preflight.metadata_provider_us", pool_us),
        ("preflight.metadata_provider_max_us", pool_us),
        ("per_proof_layout_name_lookups", admission_events.bounded_name_lookups),
        ("per_proof_vk_full_validation_calls", admission_events.vk_full_validation_calls),
        ("per_proof_metadata_full_rebuilds", admission_events.metadata_full_rebuilds),
        ("per_proof_metadata_name_sorts", admission_events.metadata_name_sorts),
        ("per_proof_machine_layout_second_passes", admission_events.machine_layout_second_passes),
    ];
    let building = BuildingRecord::from_record(record);
    building.record().profile.publish_proof_batch_and_seal(
        prepare_start,
        semantic_prepare_us,
        &record_splits,
        &[("transcript_producer_walk", transcript_memo_hits, transcript_memo_misses)],
        &structural_counters,
    );
    Ok(building)
}

pub fn observe_replay_vk<ChildSC>(
    vk: &SCStarkVerifyingKey<ChildSC>,
    challenger: &mut <RecordingSC as SCStarkGenericConfig>::MlChallenger,
) where
    ChildSC: ReplayCompatibleProofConfig,
    ChildSC::Mlpcs: MlPCS<Commitment = MlCom<RecordingSC>>,
{
    // `SCStarkVerifyingKey::observe_into` observes only these fields. Observe them directly so a
    // proof config can remain nominally `ChildSC`; rebuilding a `RecordingSC` VK would clone all
    // chip metadata/maps even though Fiat--Shamir never reads them.
    challenger.observe(F::from_canonical_u32(0x3156_4b47));
    challenger.observe(F::one());
    challenger.observe(vk.commit.clone());
    observe_program_global_metadata_v2::<F, _>(
        challenger,
        vk.pc_start,
        &vk.program_boundary,
        &vk.owner_registry,
    )
    .expect("validated replay VK Global metadata");
}

pub fn merge_child_proof_shard_records(
    shard_records: Vec<BuildingRecord>,
) -> NativeRecursionAssemblyResult<BuildingRecord> {
    let mut slots = ProofSlotAssembler::new(shard_records.len())?;
    for record in shard_records {
        slots.admit(record)?;
    }
    slots.finish()
}

/// Dense exact-once proof-slot admission. Worker completion order is not an
/// ordering authority; records are merged only by their planner-assigned
/// `proof_idx`. Cancellation permanently closes the assembler to late work.
struct ProofSlotAssembler {
    slots: Vec<Option<BuildingRecord>>,
    cancelled: bool,
}

impl ProofSlotAssembler {
    fn new(required: usize) -> NativeRecursionAssemblyResult<Self> {
        if required == 0 {
            return Err(NativeRecursionAssemblyError::Record(
                "cannot merge an empty child-record set".to_string(),
            ));
        }
        Ok(Self { slots: (0..required).map(|_| None).collect(), cancelled: false })
    }

    fn admit(&mut self, record: BuildingRecord) -> NativeRecursionAssemblyResult<()> {
        let admission_start = Instant::now();
        if self.cancelled {
            return Err(NativeRecursionAssemblyError::Record(
                "proof completion arrived after node cancellation".to_string(),
            ));
        }
        let proof_records = &record.record().proof_records;
        if proof_records.len() != 1 {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "one proof slot must contain exactly one proof segment, got {}",
                proof_records.len()
            )));
        }
        let proof_idx = proof_records[0].proof_idx;
        let required = self.slots.len();
        let slot = self.slots.get_mut(proof_idx).ok_or_else(|| {
            NativeRecursionAssemblyError::Record(format!(
                "proof slot {proof_idx} is outside planned range 0..{required}"
            ))
        })?;
        if slot.is_some() {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "duplicate proof slot admission for proof_idx={proof_idx}"
            )));
        }
        let admission_ms = admission_start.elapsed().as_millis();
        record.record().profile.add_record_split("node_slot_admission", admission_ms);
        record.record().profile.add_record_split("proof_slot_admission", admission_ms);
        *slot = Some(record);
        Ok(())
    }

    #[cfg(test)]
    fn cancel(&mut self) {
        self.cancelled = true;
        self.slots.iter_mut().for_each(|slot| *slot = None);
    }

    fn finish(self) -> NativeRecursionAssemblyResult<BuildingRecord> {
        if self.cancelled {
            return Err(NativeRecursionAssemblyError::Record(
                "cancelled proof node cannot be sealed".to_string(),
            ));
        }
        if let Some(missing) = self.slots.iter().position(Option::is_none) {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "required proof slot {missing} was not completed"
            )));
        }
        let mut ordered = self.slots.into_iter().map(Option::unwrap);
        let mut record = ordered.next().expect("non-empty slots checked at construction");
        let append_start = Instant::now();
        for mut child in ordered {
            record.append(&mut child);
        }
        record
            .record()
            .profile
            .add_record_split("combined.final_append", append_start.elapsed().as_millis());
        Ok(record)
    }
}

/// The sole Building → Finalized transition for the native proving route.
///
/// All layer-specific scalar inputs must be installed on `record` before this call. Publication,
/// statement derivation, provider registration, nonce registration, and the generation seal each
/// run exactly once here; [`prove_recursion_with_metrics`] accepts only the returned typestate.
pub fn finalize_building_record(
    mut record: BuildingRecord,
    program: &RecursionNativeProgram<F>,
    _residual_label: &'static str,
) -> NativeRecursionAssemblyResult<FinalizedRecord> {
    let proof_count = record.record().proof_records.len();
    if proof_count == 0 {
        return Err(NativeRecursionAssemblyError::Record(format!(
            "final record proof indices must be dense and ordered: record is empty"
        )));
    }
    if let Some((expected, actual)) =
        record.record().proof_records.iter().enumerate().find_map(|(expected, proof)| {
            (proof.proof_idx != expected).then_some((expected, proof.proof_idx))
        })
    {
        return Err(NativeRecursionAssemblyError::Record(format!(
            "final record proof indices must be dense and ordered: expected {expected}, got {actual}"
        )));
    }
    if record.record().provider_requests_finalized {
        return Err(NativeRecursionAssemblyError::Record(
            "cannot finalize or merge a record whose provider requests are already sealed"
                .to_string(),
        ));
    }

    let finalize_start = Instant::now();
    let raw = record.record_mut();
    if let Some(proof) =
        raw.proof_records.iter().find(|proof| proof.transcript.sponge_blocks.is_empty())
    {
        return Err(NativeRecursionAssemblyError::Record(format!(
            "proof {} is missing source-captured transcript sponge rows",
            proof.proof_idx
        )));
    }
    let statement_public_values_start = Instant::now();
    raw.refresh_statement_public_values(program)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("statement: {err}")))?;
    let statement_public_values_us = elapsed_us(statement_public_values_start);
    assert_machine_record_fully_published(raw)?;
    raw.register_nonces(&());

    // The finalized record retains only statement control. Exact statement and hash rows are
    // installed once by the post-seal TracegenWorkspace and reused by exact admission + matrix
    // generation.
    raw.profile.set_structural_counters([
        ("statement.case.public_values_us", statement_public_values_us),
        ("statement.case.control_rows_us", 0),
        ("statement.case.hash_rows_us", 0),
        ("statement.case.total_us", statement_public_values_us),
        (
            "statement.static_config_rows",
            u64::try_from(program.statement_config.len()).unwrap_or(u64::MAX),
        ),
        ("statement.dynamic_case_owner_count", 1),
        ("statement.cross_case_dynamic_cache_entries", 0),
        ("statement_pre_seal_exact_rows", 0),
    ]);

    raw.profile.set_structural_counter("constraint.pre_seal_derivation_us", 0);
    raw.profile.set_structural_counter("duplicate_statement_control_replay_calls", 0);
    raw.profile.set_structural_counter("full_poseidon2_witness_rows_during_prepare", 0);
    raw.profile.set_structural_counter("padded_matrix_cells_during_prepare", 0);
    raw.profile.set_structural_counter("record_final_matrix_allocations", 0);
    raw.profile.set_structural_counter("merkle_candidate_rows", 0);
    raw.profile.set_structural_counter("merkle_union_passes", 0);
    raw.profile.set_structural_counter("post_row_publication_patch_passes", 0);
    raw.profile.set_structural_counter("row_to_provider_rescans", 0);
    raw.profile.set_structural_counter("active_proof_linear_lookups", 0);
    raw.profile.set_structural_counter("seed_prefix_deep_clones", 0);
    raw.profile.set_structural_counter("completion_order_reorder_copies", 0);
    if let Some(peak_rss_kb) = peak_rss_kb() {
        raw.profile.set_structural_counter("peak_preparation_rss_kb", peak_rss_kb);
    }
    raw.profile.add_record_split("segment_finalization", finalize_start.elapsed().as_millis());
    raw.profile.add_record_split("combined.finalize_once", finalize_start.elapsed().as_millis());
    Ok(FinalizedRecord::from_record(record.into_record(), program, FinalizationSeal(())))
}

#[derive(Clone, Copy)]
struct ProviderTelemetry {
    segments: u64,
    entries: u64,
    bytes: u64,
    family_segments: [u64; 4],
    family_entries: [u64; 4],
    family_bytes: [u64; 4],
    family_multiplicity_units: [u64; 4],
}

fn provider_telemetry(record: &RecursionRecord) -> ProviderTelemetry {
    let families = record.provider_input_layout().families;
    let request_sizes = [
        core::mem::size_of::<crate::system_dt::RecursionNativeChipMetadataRequest>(),
        core::mem::size_of::<crate::system_dt::RecursionPoseidon2Request>(),
        core::mem::size_of::<crate::system_dt::RecursionRangeRequest>(),
        core::mem::size_of::<crate::system_dt::RecursionPowerRequest>(),
    ];
    let family_segments = families.map(|family| {
        u64::try_from(family.segment_count).expect("provider segment count exceeds u64")
    });
    let family_entries = families
        .map(|family| u64::try_from(family.entry_count).expect("provider entry count exceeds u64"));
    let family_bytes = core::array::from_fn(|idx| {
        family_entries[idx]
            .checked_mul(u64::try_from(request_sizes[idx]).expect("request size exceeds u64"))
            .expect("provider retained byte counter overflow")
    });
    let family_multiplicity_units = [
        record.native_chip_metadata.total_count(),
        record.poseidon2.total_count(),
        record.range.total_count(),
        record.pow.total_count(),
    ];
    ProviderTelemetry {
        segments: family_segments.iter().copied().sum(),
        entries: family_entries.iter().copied().sum(),
        bytes: family_bytes.iter().copied().sum(),
        family_segments,
        family_entries,
        family_bytes,
        family_multiplicity_units,
    }
}

fn publish_provider_input_telemetry(record: &RecursionRecord, telemetry: &ProviderTelemetry) {
    const FAMILY_NAMES: [&str; 4] = ["metadata", "poseidon2", "range", "power"];
    let profile = &record.profile;
    profile.set_structural_counter("provider_segment_count", telemetry.segments);
    profile.set_structural_counter("provider_input_entries", telemetry.entries);
    profile.set_structural_counter("provider_input_retained_bytes", telemetry.bytes);
    for (idx, family) in FAMILY_NAMES.iter().enumerate() {
        profile.set_structural_counter(
            format!("{family}_provider_segments"),
            telemetry.family_segments[idx],
        );
        profile.set_structural_counter(
            format!("{family}_provider_entries"),
            telemetry.family_entries[idx],
        );
        profile.set_structural_counter(
            format!("{family}_provider_retained_bytes"),
            telemetry.family_bytes[idx],
        );
        profile.set_structural_counter(
            format!("{family}_provider_multiplicity_units"),
            telemetry.family_multiplicity_units[idx],
        );
    }
    profile.set_structural_counter("raw_provider_segments", telemetry.segments);
    profile.set_structural_counter("proof_local_provider_map_allocations", 0);
    profile.set_structural_counter("proof_local_provider_dedup_lookups", 0);
}

fn provider_padded_height(record: &RecursionRecord) -> u64 {
    let metadata = record.native_chip_metadata.unique_count().max(1).next_power_of_two();
    let poseidon2 = crate::transcript_dt::poseidon2::Poseidon2PermuteTraceGenerator::trace_height(
        &record.poseidon2,
    );
    let range8 =
        crate::primitives_dt::range::RangeCheckerTraceGenerator::<8>::trace_height(&record.range);
    let range21 =
        crate::primitives_dt::range::RangeCheckerTraceGenerator::<21>::trace_height(&record.range);
    u64::try_from(metadata + poseidon2 + range8 + range21)
        .expect("provider padded height exceeds u64")
}

fn record_preparation_event_telemetry(record: &RecursionRecord) {
    let mut transcript_events = 0usize;
    let mut transcript_bytes = 0usize;
    let mut merkle_events = 0usize;
    let mut merkle_bytes = 0usize;
    let mut constraint_events = 0usize;
    let mut constraint_bytes = 0usize;
    let mut proof_shape_events = 0usize;
    let mut proof_shape_bytes = 0usize;
    let mut batch_events = 0usize;
    let mut batch_bytes = 0usize;
    let mut whir_sources = 0usize;
    let mut whir_source_bytes = 0usize;

    for proof in &record.proof_records {
        transcript_events += proof.transcript.events.len() + proof.transcript.bits_events.len();
        transcript_bytes += proof.transcript.events.capacity() *
            core::mem::size_of::<crate::system_dt::RecursionTranscriptEvent>() +
            proof.transcript.bits_events.capacity() *
                core::mem::size_of::<crate::system_dt::RecursionTranscriptBitsEvent>();
        merkle_events += proof.merkle_path.row_count();
        merkle_bytes += proof.merkle_path.row_count() *
            core::mem::size_of::<crate::system_dt::RecursionMerklePathRow>();
        constraint_events += proof.constraints.events.len();
        constraint_bytes += proof.constraints.events.capacity() *
            core::mem::size_of::<crate::system_dt::RecursionConstraintEvent>();
        proof_shape_events += proof.proof_shape.chips.len() + proof.proof_shape.public_values.len();
        proof_shape_bytes += proof.proof_shape.chips.capacity() *
            core::mem::size_of::<crate::system_dt::RecursionProofShapeChip>() +
            proof.proof_shape.public_values.capacity() * core::mem::size_of::<F>() +
            proof.proof_shape.public_value_send_mults.capacity() * core::mem::size_of::<u32>();
        batch_events += proof.batch_constraint.cum_sums.len() +
            proof.batch_constraint.eq_challenges.len() +
            proof.batch_constraint.rounds.len();
        batch_bytes += proof.batch_constraint.cum_sums.capacity() *
            core::mem::size_of::<crate::system_dt::RecursionBatchCumSumRecord>() +
            proof.batch_constraint.eq_challenges.capacity() * core::mem::size_of::<[F; 5]>() +
            proof.batch_constraint.rounds.capacity() *
                core::mem::size_of::<crate::system_dt::RecursionSumcheckRoundRecord>();
        if let Some(source) = &proof.whir_source {
            whir_sources += 1;
            whir_source_bytes += core::mem::size_of_val(source) +
                source.opened_eval_publications.capacity() *
                    core::mem::size_of::<crate::system_dt::RecursionWhirOpenedEvalPublication>(
                    ) +
                source.input_roots.capacity() * core::mem::size_of::<[F; DIGEST_SIZE]>() +
                source.opened_values.chips.capacity() *
                    core::mem::size_of::<dt_stark::sumcheck::proof::SCChipOpenedValues<F, EF>>(
                    ) +
                source.dimensions.iter().map(Vec::capacity).sum::<usize>() *
                    core::mem::size_of::<p3_matrix::Dimensions>();
            for query in &source.opening_proof.query_openings.per_query {
                for opening in query {
                    merkle_events += opening.opening_proof.len();
                    merkle_bytes +=
                        opening.opening_proof.capacity() * core::mem::size_of::<[F; DIGEST_SIZE]>();
                }
            }
            for query in &source.opening_proof.iopp_queries {
                for opening in &query.commit_phase_openings {
                    merkle_events += opening.opening_proof.len();
                    merkle_bytes +=
                        opening.opening_proof.capacity() * core::mem::size_of::<[F; DIGEST_SIZE]>();
                }
            }
        }
    }

    let transcript_rows = record.tracegen_artifacts.transcript_sponge.get().map_or_else(
        || record.proof_records.iter().map(|proof| proof.transcript.sponge_blocks.len()).sum(),
        |rows| rows.len(),
    );
    transcript_events = transcript_events.saturating_add(transcript_rows);
    transcript_bytes = transcript_bytes.saturating_add(
        transcript_rows.saturating_mul(core::mem::size_of::<crate::system_dt::SpecSpongeBlock>()),
    );

    let profile = &record.profile;
    for (family, events, bytes) in [
        ("transcript", transcript_events, transcript_bytes),
        ("merkle", merkle_events, merkle_bytes),
        ("constraint", constraint_events, constraint_bytes),
        ("proof_shape", proof_shape_events, proof_shape_bytes),
        ("batch", batch_events, batch_bytes),
        ("whir_source", whir_sources, whir_source_bytes),
    ] {
        profile.set_structural_counter(
            format!("prepare_{family}_event_count"),
            u64::try_from(events).expect("preparation event count exceeds u64"),
        );
        profile.set_structural_counter(
            format!("prepare_{family}_retained_bytes"),
            u64::try_from(bytes).expect("preparation retained bytes exceed u64"),
        );
        profile.set_structural_counter(
            format!("authoritative_bytes_by_family.{family}"),
            u64::try_from(bytes).expect("authoritative bytes exceed u64"),
        );
    }
    profile.set_structural_counter(
        "transcript_authoritative_rows_or_events",
        u64::try_from(transcript_events).expect("transcript authority count exceeds u64"),
    );
    profile.set_structural_counter(
        "merkle_authoritative_rows_or_events",
        u64::try_from(merkle_events).expect("Merkle authority count exceeds u64"),
    );
}

fn peak_rss_kb() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

pub(crate) fn timed_residual_assert(
    record: &RecursionRecord,
    program: &RecursionNativeProgram<F>,
    label: &'static str,
) -> NativeRecursionAssemblyResult<()> {
    let start = Instant::now();
    let result = assert_native_recursion_record_residuals(record, program);
    record.profile.add_record_split(format!("residual:{label}"), start.elapsed().as_millis());
    result
}

pub(crate) fn run_intermediate_residuals() -> bool {
    cfg!(test) || env_flag(INTERMEDIATE_RESIDUALS_ENV)
}

pub(crate) fn run_final_residuals() -> bool {
    cfg!(test) || env_flag(FINAL_RESIDUALS_ENV) || run_intermediate_residuals()
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

pub fn assert_machine_record_fully_published(
    record: &RecursionRecord,
) -> NativeRecursionAssemblyResult<()> {
    for proof in &record.proof_records {
        if !proof.proof_shape.publish_external ||
            !proof.proof_shape.publish_whir_inputs ||
            !proof.proof_shape.publish_terminal_summary
        {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "proof {} has half-published proof-shape switches",
                proof.proof_idx
            )));
        }
        if !proof.batch_constraint.publish_opening_point ||
            !proof.batch_constraint.publish_terminal_outputs
        {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "proof {} has half-published batch-constraint switches",
                proof.proof_idx
            )));
        }
        let has_opened_eval = proof.whir_source.as_ref().is_some_and(|source| {
            source.publish_opened_eval && !source.opened_values.chips.is_empty()
        }) || proof
            .whir
            .batch_eval_rows
            .iter()
            .any(|row| row.is_value && row.opened_eval_send_mult != 0);
        if !has_opened_eval {
            return Err(NativeRecursionAssemblyError::Record(format!(
                "proof {} has no opened-eval publications",
                proof.proof_idx
            )));
        }
    }
    Ok(())
}

pub fn assert_native_recursion_record_residuals(
    record: &RecursionRecord,
    native_program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    let program = &native_program.constraint_program;
    let mut report = whir_bus_residual_report(record);
    apply_constraint_terminal_consumers(record, program, &mut report);
    if let Some(transcript) = report.remove("1007 TranscriptEvent") {
        if !transcript.is_empty() {
            return Err(NativeRecursionAssemblyError::BusResidual(format!(
                "1007 TranscriptEvent: {} residual keys",
                transcript.len()
            )));
        }
    }
    if !report.is_empty() {
        return Err(NativeRecursionAssemblyError::BusResidual(format_report(&report)));
    }
    let report = constraint_replay_bus_residual_report(record, program);
    if !report.is_empty() {
        return Err(NativeRecursionAssemblyError::BusResidual(format_report(&report)));
    }
    let report = statement_part_b_bus_residual_report(record, native_program);
    if !report.is_empty() {
        return Err(NativeRecursionAssemblyError::BusResidual(format_report(&report)));
    }
    let report = statement_hash_bus_residual_report(
        record,
        StatementDigestMode::from_role(native_program.statement_role),
    );
    if !report.is_empty() {
        return Err(NativeRecursionAssemblyError::BusResidual(format_report(&report)));
    }
    Ok(())
}

pub fn native_metadata_from_machine<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
) -> Vec<NativeChipMetadata>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    machine
        .chips
        .iter()
        .map(|chip| NativeChipMetadata {
            name: chip.name(),
            preprocessed_width: chip.symbolic_builder.preprocessed.len(),
            main_width: chip.symbolic_builder.main.len(),
            permutation_width: chip.perm_width() * D_EF,
            commit_scope: chip.commit_scope(),
            has_local_interactions: true,
            constraint_count: chip.num_alpha,
            gate_count: chip.symbolic_builder.gate.len(),
            logup_batch_size: chip.logup_batch_size(),
            required_max_beta_power: chip.required_max_beta_power(),
        })
        .collect()
}

pub fn native_metadata_for_shard<MachineSC, ProofSC, A>(
    machine: &polyair::SCStarkMachine<MachineSC, A, D_EF>,
    shard: &SCShardProof<ProofSC>,
) -> NativeRecursionAssemblyResult<Vec<NativeChipMetadata>>
where
    MachineSC: SCStarkGenericConfig<Val = F>,
    ProofSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let proof_view = crate::child_views::NativeChildProofView::new(shard).map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("NativeChildProofView: {err:?}"))
    })?;
    native_metadata_for_proof_view(machine, &proof_view)
}

fn native_metadata_for_proof_view<MachineSC, ProofSC, A>(
    machine: &polyair::SCStarkMachine<MachineSC, A, D_EF>,
    proof_view: &crate::child_views::NativeChildProofView<'_, ProofSC>,
) -> NativeRecursionAssemblyResult<Vec<NativeChipMetadata>>
where
    MachineSC: SCStarkGenericConfig<Val = F>,
    ProofSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let mut metadata = native_metadata_from_machine(machine);
    for opened_chip in proof_view.ordered_chips() {
        let chip =
            metadata.iter_mut().find(|chip| chip.name == opened_chip.name).ok_or_else(|| {
                NativeRecursionAssemblyError::Record(format!(
                    "machine metadata missing opened chip {}",
                    opened_chip.name
                ))
            })?;
        chip.preprocessed_width = opened_chip.opened_values.preprocessed.local.len();
        chip.main_width = opened_chip.opened_values.main.local.len();
        chip.permutation_width = proof_view.permutation_dimension_width(opened_chip.index);
    }
    Ok(metadata)
}

pub fn proof_shape_static_chip_id_map<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
) -> BTreeMap<String, usize>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let mut names = machine.chips.iter().map(|chip| chip.name()).collect::<Vec<_>>();
    names.sort_unstable();
    names.into_iter().enumerate().map(|(idx, name)| (name, idx)).collect()
}

fn validate_role_matrix(
    role: RecursionChildRole,
    num_child_public_values: usize,
    child_contains_global_bus: bool,
) -> NativeRecursionAssemblyResult<()> {
    let matching = NativeRecursionLayer::ALL
        .into_iter()
        .map(NativeRecursionLayer::params)
        .filter(|params| params.child_role == role)
        .collect::<Vec<_>>();
    let valid = !matching.is_empty() &&
        matching.iter().all(|params| {
            params.num_child_public_values == num_child_public_values &&
                params.child_contains_global_bus == child_contains_global_bus
        });
    if !valid {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "invalid role matrix for {:?}: num_child_public_values={} child_contains_global_bus={}",
            role, num_child_public_values, child_contains_global_bus
        )));
    }
    Ok(())
}

fn validate_statement_role(
    child_role: RecursionChildRole,
    statement_role: RecursionStatementRole,
) -> NativeRecursionAssemblyResult<()> {
    NativeRecursionLayer::from_roles(child_role, statement_role).map(|_| ())
}

fn native_child_role(role: RecursionChildRole) -> NativeChildRole {
    match role {
        RecursionChildRole::Core => NativeChildRole::Core,
        RecursionChildRole::Compress => NativeChildRole::Compress,
        RecursionChildRole::Shrink => NativeChildRole::Shrink,
    }
}

fn role_id(role: NativeChildRole) -> usize {
    match role {
        NativeChildRole::Core => 0,
        NativeChildRole::Compress => 1,
        NativeChildRole::Shrink => 2,
    }
}

fn validate_record_matches_program(
    record: &RecursionRecord,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    let params = program.layer()?.params();
    let expected_role_id = role_id(native_child_role(params.child_role));
    for proof in &record.proof_records {
        if proof.proof_shape.is_empty() {
            continue;
        }
        if proof.proof_shape.role_id != expected_role_id ||
            proof.proof_shape.num_public_values != params.num_child_public_values
        {
            return Err(NativeRecursionAssemblyError::Validation(format!(
                "proof {} role matrix mismatch: role_id={} num_public_values={} expected role_id={} num_public_values={}",
                proof.proof_idx,
                proof.proof_shape.role_id,
                proof.proof_shape.num_public_values,
                expected_role_id,
                params.num_child_public_values
            )));
        }
    }
    Ok(())
}

fn apply_constraint_terminal_consumers(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
    report: &mut BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>,
) {
    if let Some(residual) = report.get_mut("1007 TranscriptEvent") {
        for row in constraint_challenge_rows(record, program) {
            // e6 local-sum transcript band; the fixed D5 E9 stride is irrelevant here.
            let layout = crate::batch_constraint_dt::BatchTranscriptLayout::new(
                row.num_public_values,
                row.c_chips,
                0,
                record
                    .proof_records
                    .iter()
                    .find(|proof| proof.proof_idx == row.proof_idx)
                    .is_some_and(|proof| proof.proof_shape.role_id == 0),
            );
            for (offset, value) in row.lcs_limbs.into_iter().enumerate() {
                apply_report_residual(
                    residual,
                    vec![
                        row.proof_idx as u32,
                        (layout.e6_tidx(row.chip_idx) + offset) as u32,
                        0,
                        value.as_canonical_u32(),
                    ],
                    -1,
                );
            }
        }
        residual.retain(|_, value| *value != 0);
    }

    if let Some(residual) = report.get_mut("1009 BatchDim") {
        for binder_row in crate::proof_shape_dt::proof_shape_binder_rows(record) {
            if let crate::proof_shape_dt::trace::ProofShapeBinderRow::Chip {
                proof_idx,
                chip,
                publish_batch_dim: true,
                ..
            } = binder_row
            {
                apply_report_residual(
                    residual,
                    vec![
                        proof_idx as u32,
                        crate::proof_shape_dt::PROOF_SHAPE_BATCH_MAIN as u32,
                        chip.chip_idx as u32,
                        chip.chip_idx as u32,
                        chip.static_chip_id as u32,
                        chip.main_width as u32,
                        chip.log_height as u32,
                    ],
                    1,
                );
            }
        }
        for row in constraint_challenge_rows(record, program) {
            apply_report_residual(
                residual,
                vec![
                    row.proof_idx as u32,
                    crate::proof_shape_dt::PROOF_SHAPE_BATCH_MAIN as u32,
                    row.chip_idx as u32,
                    row.chip_idx as u32,
                    row.static_chip_id as u32,
                    row.main_width as u32,
                    row.log_height as u32,
                ],
                -1,
            );
        }
        residual.retain(|_, value| *value != 0);
    }

    if let Some(residual) = report.get_mut("1017 BatchOpeningPoint") {
        for row in constraint_terminal_rows(record, program) {
            if row.opening_point_recv_mult {
                let mut key = vec![row.proof_idx as u32, row.opening_idx as u32];
                key.extend(row.opening_point.into_iter().map(|value| value.as_canonical_u32()));
                apply_report_residual(residual, key, -1);
            }
        }
        residual.retain(|_, value| *value != 0);
    }

    if let Some(residual) = report.get_mut("1022 ProofShapeSummary") {
        for row in constraint_terminal_rows(record, program) {
            if row.summary_recv_mult {
                apply_report_residual(
                    residual,
                    vec![
                        row.proof_idx as u32,
                        row.num_rounds as u32,
                        row.c_chips as u32,
                        row.num_public_values as u32,
                        row.summary_id_base as u32,
                    ],
                    -1,
                );
            }
        }
        residual.retain(|_, value| *value != 0);
    }
    report.retain(|_, residual| !residual.is_empty());
}

fn apply_report_residual(residual: &mut BTreeMap<Vec<u32>, i64>, key: Vec<u32>, delta: i64) {
    *residual.entry(key).or_insert(0) += delta;
}

fn format_report(report: &BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>) -> String {
    report
        .iter()
        .map(|(name, residual)| {
            let sample = residual
                .iter()
                .take(3)
                .map(|(key, value)| format!("{key:?}=>{value}"))
                .collect::<Vec<_>>()
                .join("; ");
            format!("{name}: {} residual keys sample=[{sample}]", residual.len())
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use p3_field::AbstractField;

    use super::*;
    use crate::{
        config::POSEIDON2_WIDTH,
        native_air_dt::{
            validate_final_replay_layout, NativeAirFamily, NativeRecursionLayer,
            LAYER_AIR_FAMILIES, PROGRAM_SENSITIVE_AIR_FAMILIES, SHARED_AIR_FAMILIES,
        },
        statement_dt::{
            STATEMENT_CONFIG_CLASS_BAKED_L2, STATEMENT_CONFIG_CLASS_BAKED_L3,
            STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        },
        symbolic_expr_fixed_dt::RecursionFixedSymbolicChip,
        system_dt::{RecursionProofRecord, RecursionProofShapeChip, RecursionProofShapeRecord},
        validate::set_budget_log_height_override_for_test,
    };

    struct BudgetOverrideGuard;

    impl Drop for BudgetOverrideGuard {
        fn drop(&mut self) {
            set_budget_log_height_override_for_test(None);
        }
    }

    fn evaluator_statement_config(layer: NativeRecursionLayer) -> Vec<StatementConfigRow> {
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

    fn evaluator_program(layer: NativeRecursionLayer) -> RecursionNativeProgram<F> {
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
            .expect("empty evaluator constraint program"),
            evaluator_statement_config(layer),
        )
    }

    fn symbolic_ir(
        air: &NativeRecursionAir,
        static_chip_id: usize,
        chip_name: &str,
    ) -> RecursionPolyAirChipIr {
        let chip = polyair::Chip::<NativeRecursionAir, F, D_EF>::new(air.clone());
        let fixed = RecursionFixedSymbolicChip::from_symbolic_builder(
            static_chip_id,
            chip_name.to_string(),
            chip.commit_scope(),
            chip.logup_batch_size(),
            chip.num_alpha,
            &chip.symbolic_builder,
        )
        .expect("symbolic snapshot");
        RecursionPolyAirChipIr::compile(&fixed).expect("symbolic IR")
    }

    fn normalized_symbolic_ir(air: &NativeRecursionAir) -> Vec<u8> {
        bincode::serialize(&symbolic_ir(air, 0, "NormalizedAir")).expect("serialize symbolic IR")
    }

    fn evaluator_profiles(
        program: &RecursionNativeProgram<F>,
    ) -> BTreeMap<NativeAirFamily, Vec<u8>> {
        NativeRecursionAir::all(program)
            .expect("semantic program")
            .into_iter()
            .map(|air| (air.family(), normalized_symbolic_ir(&air)))
            .collect()
    }

    fn assert_profile_group(
        profiles: &BTreeMap<NativeRecursionLayer, BTreeMap<NativeAirFamily, Vec<u8>>>,
        family: NativeAirFamily,
        equal_groups: &[&[NativeRecursionLayer]],
    ) {
        for group in equal_groups {
            let first = &profiles[&group[0]][&family];
            for layer in &group[1..] {
                assert_eq!(first, &profiles[layer][&family], "{family:?} at {layer:?}");
            }
        }
        for left in 0..equal_groups.len() {
            for right in left + 1..equal_groups.len() {
                assert_ne!(
                    profiles[&equal_groups[left][0]][&family],
                    profiles[&equal_groups[right][0]][&family],
                    "{family:?} groups {left}/{right}"
                );
            }
        }
    }

    #[test]
    fn native_layer_evaluator_profiles_match_the_documented_groups() {
        const L1: &[NativeRecursionLayer] = &[NativeRecursionLayer::L1Lift];
        const L2: &[NativeRecursionLayer] = &[NativeRecursionLayer::L2Reduce];
        const L3: &[NativeRecursionLayer] = &[NativeRecursionLayer::L3Reduce];
        const L4: &[NativeRecursionLayer] = &[NativeRecursionLayer::L4Root];
        const L23: &[NativeRecursionLayer] =
            &[NativeRecursionLayer::L2Reduce, NativeRecursionLayer::L3Reduce];
        const L234: &[NativeRecursionLayer] = &[
            NativeRecursionLayer::L2Reduce,
            NativeRecursionLayer::L3Reduce,
            NativeRecursionLayer::L4Root,
        ];
        const ALL: &[NativeRecursionLayer] = &NativeRecursionLayer::ALL;

        let profiles = NativeRecursionLayer::ALL
            .into_iter()
            .map(|layer| (layer, evaluator_profiles(&evaluator_program(layer))))
            .collect::<BTreeMap<_, _>>();

        for family in SHARED_AIR_FAMILIES {
            assert_profile_group(&profiles, family, &[ALL]);
        }
        assert_profile_group(&profiles, NativeAirFamily::ProofShapeBinder, &[L1, L23, L4]);
        for family in [
            NativeAirFamily::BatchTranscriptInputs,
            NativeAirFamily::BatchSumcheck,
            NativeAirFamily::ConstraintTerminal,
            NativeAirFamily::ConstraintChallenge,
        ] {
            assert_profile_group(&profiles, family, &[L1, L234]);
        }
        assert_profile_group(&profiles, NativeAirFamily::ConstraintBoundary, &[ALL]);
        for family in [NativeAirFamily::WhirRound, NativeAirFamily::WhirBatchEval] {
            assert_profile_group(&profiles, family, &[L1, L23, L4]);
        }
        assert_profile_group(&profiles, NativeAirFamily::Statement, &[L1, L2, L3, L4]);
        assert_profile_group(&profiles, NativeAirFamily::StatementHash, &[L1, L23, L4]);
        for family in PROGRAM_SENSITIVE_AIR_FAMILIES {
            assert_profile_group(&profiles, family, &[ALL]);
        }

        let mut bootstrap = evaluator_program(NativeRecursionLayer::L2Reduce);
        let seed_air = NativeRecursionAir::all(&bootstrap).expect("bootstrap AIRs")[0].clone();
        let mut bootstrap_dto = bootstrap.constraint_program.to_dto();
        bootstrap_dto.chips = vec![symbolic_ir(&seed_air, 0, "BootstrapPayload")];
        bootstrap_dto.max_required_beta_power = segment_max_beta_power(&bootstrap_dto.chips);
        bootstrap.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(bootstrap_dto)
            .expect("bootstrap evaluator constraint program");
        let mut final_l2 = bootstrap.clone();
        let mut final_dto = final_l2.constraint_program.to_dto();
        final_dto.chips[0].chip_name = "FinalPayload".to_string();
        final_dto.artifact_digest[0] = F::one();
        final_l2.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(final_dto)
            .expect("final evaluator constraint program");
        assert_eq!(
            evaluator_profiles(&bootstrap),
            evaluator_profiles(&final_l2),
            "bootstrap and final L2 parent symbolic universes must be identical"
        );

        let mut wide_dto = bootstrap.constraint_program.to_dto();
        wide_dto.chips[0].logup_batch_size = 4;
        let error = RecursionPolyAirVerifierProgram::try_from_dto(wide_dto)
            .expect_err("the shipped native product must reject non-binary lookup batches");
        assert!(
            matches!(
                error,
                crate::symbolic_ir_dt::RecursionPolyAirProgramError::InvalidProgram(ref message)
                    if message.contains("requires 2")
            ),
            "unexpected batch-size validation error: {error:?}"
        );
    }

    #[test]
    fn l2_bootstrap_fixed_point_accepts_the_frozen_batch_two_product() {
        let lift = evaluator_program(NativeRecursionLayer::L1Lift);
        let lift_machine = native_recording_machine(&lift).expect("lift recording machine");

        let mut bootstrap = evaluator_program(NativeRecursionLayer::L2Reduce);
        let seed_air = NativeRecursionAir::all(&bootstrap).expect("bootstrap AIRs")[0].clone();
        let mut bootstrap_dto = bootstrap.constraint_program.to_dto();
        bootstrap_dto.chips = vec![symbolic_ir(&seed_air, 0, "BootstrapPayload")];
        bootstrap_dto.max_required_beta_power = segment_max_beta_power(&bootstrap_dto.chips);
        bootstrap.constraint_program = RecursionPolyAirVerifierProgram::try_from_dto(bootstrap_dto)
            .expect("bootstrap fixed-point test constraint program");
        let bootstrap_machine =
            native_recording_machine(&bootstrap).expect("bootstrap L2 recording machine");

        let final_l2 = build_dual_segment_reduce_program(
            &lift_machine,
            &bootstrap_machine,
            RecursionStatementRole::ReduceL2,
            evaluator_statement_config(NativeRecursionLayer::L2Reduce),
        )
        .expect("final L2 program");
        let final_l2_machine =
            native_recording_machine(&final_l2).expect("final L2 recording machine");

        validate_l2_bootstrap_fixed_point(&final_l2_machine, &final_l2)
            .expect("the frozen batch-two L2 program must be its bootstrap fixed point");
    }

    #[test]
    fn serialized_l2_bootstrap_is_not_accepted_as_a_final_ladder_program() {
        let lift = evaluator_program(NativeRecursionLayer::L1Lift);
        let lift_machine = native_recording_machine(&lift).expect("lift recording machine");
        let bootstrap = build_native_recursion_program(
            &lift_machine,
            RecursionStatementRole::ReduceL2,
            RecursionChildRole::Compress,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false,
            evaluator_statement_config(NativeRecursionLayer::L2Reduce),
        )
        .expect("live bootstrap builder accepts its explicit one-segment typestate");
        crate::native_air_dt::validate_l2_bootstrap_layout(&bootstrap).expect("bootstrap layout");

        let encoded = bincode::serialize(&bootstrap).expect("serialize bootstrap program");
        let decoded = bincode::deserialize::<RecursionNativeProgram<F>>(&encoded);
        let message =
            decoded.expect_err("serialized ladder programs require the final layout").to_string();
        assert!(message.contains("replay segments"), "{message}");
    }

    #[test]
    fn native_layer_static_ids_follow_alphabetical_names_and_segment_rebasing() {
        fn assert_segment(
            program: &RecursionNativeProgram<F>,
            base: usize,
            source_machine: &NativeRecordingMachine,
            layer_token: &str,
        ) {
            let alphabetical = proof_shape_static_chip_id_map(source_machine);
            assert_eq!(alphabetical.len(), NativeAirFamily::ALL.len());

            let segment = program
                .constraint_program
                .chips
                .iter()
                .filter(|chip| chip.static_chip_id & !127 == base)
                .collect::<Vec<_>>();
            assert_eq!(segment.len(), NativeAirFamily::ALL.len());
            assert_eq!(
                segment.iter().map(|chip| chip.static_chip_id).collect::<BTreeSet<_>>(),
                (base..base + NativeAirFamily::ALL.len()).collect()
            );

            let mut qualified_count = 0;
            for chip in segment {
                assert_eq!(
                    chip.static_chip_id,
                    base + alphabetical[&chip.chip_name],
                    "alphabetical static ID drift for {}",
                    chip.chip_name
                );
                assert!(
                    chip.gate_roots.iter().all(|root| root.static_chip_id == chip.static_chip_id),
                    "rebasing missed a gate root for {}",
                    chip.chip_name
                );
                if chip.chip_name.starts_with("NativeL") {
                    qualified_count += 1;
                    assert!(
                        chip.chip_name.starts_with(&format!("Native{layer_token}")),
                        "unexpected layer-qualified name {}",
                        chip.chip_name
                    );
                }
            }
            assert_eq!(qualified_count, LAYER_AIR_FAMILIES.len());

            let metadata_ids = program
                .native_chip_metadata
                .iter()
                .filter(|metadata| metadata.chip_id & !127 == base)
                .map(|metadata| metadata.chip_id)
                .collect::<BTreeSet<_>>();
            assert_eq!(metadata_ids, (base..base + NativeAirFamily::ALL.len()).collect());
        }

        let lift_parent = evaluator_program(NativeRecursionLayer::L1Lift);
        let lift_machine = native_recording_machine(&lift_parent).expect("L1 recording machine");
        let l2_bootstrap = evaluator_program(NativeRecursionLayer::L2Reduce);
        let l2_bootstrap_machine =
            native_recording_machine(&l2_bootstrap).expect("bootstrap L2 recording machine");

        let final_l2 = build_dual_segment_reduce_program(
            &lift_machine,
            &l2_bootstrap_machine,
            RecursionStatementRole::ReduceL2,
            evaluator_statement_config(NativeRecursionLayer::L2Reduce),
        )
        .expect("final L2 program");
        validate_final_replay_layout(&final_l2).expect("final L2 layout");
        assert_segment(&final_l2, 0, &lift_machine, "L1");
        assert_segment(&final_l2, MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET, &l2_bootstrap_machine, "L2");

        let final_l2_machine =
            native_recording_machine(&final_l2).expect("final L2 recording machine");
        let l3 = build_dual_segment_reduce_program(
            &lift_machine,
            &final_l2_machine,
            RecursionStatementRole::ReduceL3,
            evaluator_statement_config(NativeRecursionLayer::L3Reduce),
        )
        .expect("L3 program");
        validate_final_replay_layout(&l3).expect("final L3 layout");
        assert_segment(&l3, 0, &lift_machine, "L1");
        assert_segment(&l3, MIXED_SEGMENT_STATIC_CHIP_ID_OFFSET, &final_l2_machine, "L2");

        let l3_shrink_machine = native_recording_machine_for_stage(&l3, RecordingStage::Shrink)
            .expect("L3 shrink recording machine");
        let l4 = build_root_shrink_program(
            &l3_shrink_machine,
            evaluator_statement_config(NativeRecursionLayer::L4Root),
        )
        .expect("L4 program");
        validate_final_replay_layout(&l4).expect("final L4 layout");
        assert_segment(&l4, 0, &l3_shrink_machine, "L3");

        assert_eq!(
            final_l2
                .constraint_program
                .chips
                .iter()
                .map(|chip| chip.static_chip_id)
                .collect::<BTreeSet<_>>()
                .len(),
            2 * NativeAirFamily::ALL.len(),
            "dual replay segments must not collide"
        );
    }

    #[test]
    fn native_layer_machine_constructors_enforce_proof_config_pairings() {
        let l1 = evaluator_program(NativeRecursionLayer::L1Lift);
        let l2 = evaluator_program(NativeRecursionLayer::L2Reduce);
        let l3 = evaluator_program(NativeRecursionLayer::L3Reduce);
        let l4 = evaluator_program(NativeRecursionLayer::L4Root);

        assert!(native_recursion_machine_with_config(&l1, SC::compressed()).is_ok());
        assert!(native_recursion_machine_with_config(&l2, SC::compressed()).is_ok());
        assert!(native_shrink_prover(&l3).is_ok());
        assert!(native_root_shrink_prover(&l4).is_ok());

        assert!(native_recursion_machine_with_config(&l2, SC::shrink()).is_err());
        assert!(native_recursion_machine_with_config(&l3, SC::compressed()).is_err());
        assert!(native_shrink_prover(&l2).is_err());
        assert!(native_root_shrink_prover(&l3).is_err());
        assert!(native_recording_machine_for_stage(&l2, RecordingStage::Shrink).is_err());
        assert!(native_recording_machine_for_stage(&l3, RecordingStage::Compress).is_err());
    }

    #[test]
    fn native_layer_cold_start_validates_config_and_registry_before_dynamic_admission() {
        fn finalized(program: &RecursionNativeProgram<F>) -> FinalizedRecord {
            let mut record = RecursionRecord::default();
            record.proof_records.push(RecursionProofRecord::default());
            assert!(record.proof_records[0].proof_shape.is_empty());
            record.mark_provider_requests_finalized();
            FinalizedRecord::from_record(record, program, FinalizationSeal(()))
        }

        let program = evaluator_program(NativeRecursionLayer::L2Reduce);
        let err = match native_recursion_machine_with_config(&program, SC::shrink()) {
            Err(err) => err,
            Ok(_) => panic!("cold prover construction must reject a wrong proof config"),
        };
        assert!(err.to_string().contains("proof config"), "unexpected cold-start error: {err}");

        let mut incomplete_registry = NativeRecursionAir::all(&program).unwrap();
        incomplete_registry.pop();
        let err =
            crate::native_air_dt::validate_native_registry(&program, incomplete_registry.iter())
                .expect_err("cold registry construction must reject an incomplete registry");
        let expected_count = NativeAirFamily::ALL.len() - 1;
        assert!(
            err.to_string().contains(&format!("registry has {expected_count} AIRs")),
            "unexpected cold-start error: {err}"
        );

        let prover = native_recursion_prover(&program).expect("validated cold prover");
        FinalizedNode::admit(&prover, finalized(&program), &program)
            .expect("dynamic admission trusts the cold-validated static prover authority");
    }

    /// The mock non-host consumer: like commit, it takes the bundle by value.
    /// The matrices it receives must be the very allocations tracegen produced
    /// — a copy anywhere in the S6 path would change the buffer address.
    #[test]
    fn trace_bundle_surrenders_the_same_allocation() {
        let matrix = CompressedMatrix::from_full_matrix_no_padding(
            p3_matrix::dense::RowMajorMatrix::new(vec![F::zero(); 8], 2),
        );
        let data_ptr = matrix.main.values.as_ptr();
        let program = evaluator_program(NativeRecursionLayer::L2Reduce);
        let finalized = FinalizedRecord::from_record(
            RecursionRecord::default(),
            &program,
            FinalizationSeal(()),
        );
        let input = TracegenInput::new(
            PreparedRecord::seal(finalized, NativeRecursionLayer::L2Reduce)
                .expect("prepared mock record"),
        )
        .expect("mock tracegen input");
        assert_eq!(input.record().record().tracegen_artifacts.initialized_entries(), 0);
        let workspace = input.into_workspace().expect("mock tracegen workspace");
        assert_eq!(
            workspace
                .record()
                .profile
                .structural_counter("tracegen_workspace_derivation_owner_count"),
            Some(1)
        );
        assert_eq!(
            workspace.record().profile.structural_counter("sealed_semantic_mutation_paths"),
            Some(0)
        );
        assert_eq!(
            workspace.record().profile.structural_counter("production_dynamic_row_cache_entries"),
            Some(0)
        );
        let bundle = TraceBundle {
            workspace,
            traces: vec![("mock".to_string(), matrix)],
            plan: crate::validate::ExactTracePlan {
                chips: Vec::new(),
                row_count_admission_ms: 0,
                plan_fold_ms: 0,
            },
            tracegen_ms: 0,
            match_ms: 0,
        };
        let (_, traces) = bundle.into_parts();
        assert_eq!(
            traces[0].1.main.values.as_ptr(),
            data_ptr,
            "trace matrices must move through the bundle, never copy"
        );
    }

    #[test]
    fn prove_recursion_rejects_realized_budget_overflow() {
        set_budget_log_height_override_for_test(Some(63));
        let _guard = BudgetOverrideGuard;

        let recording_machine = core_recording_machine();
        let program =
            build_core_native_recursion_program(&recording_machine).expect("core program");
        let prover = native_recursion_prover(&program).expect("native recursion prover");
        let (pk, _) = prover.setup(&program);

        let mut record = RecursionRecord::default();
        record.poseidon2.record_poseidon2_count([F::zero(); POSEIDON2_WIDTH], 1);
        let record = FinalizedRecord::from_record(record, &program, FinalizationSeal(()));

        let err = match prove_recursion(&prover, &pk, &pk, record, &program) {
            Ok(_) => panic!("budget overflow must stop prove_recursion before prove"),
            Err(err) => err,
        };
        match err {
            NativeRecursionAssemblyError::Validation(message) => {
                assert!(
                    message.contains("interaction budget validation failed"),
                    "unexpected validation message: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn role_matrix_accepts_only_frozen_child_shapes() {
        assert!(validate_role_matrix(
            RecursionChildRole::Core,
            dt_stark::air::DT_PROOF_NUM_PV_ELTS,
            true
        )
        .is_ok());
        assert!(validate_role_matrix(
            RecursionChildRole::Compress,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false
        )
        .is_ok());
        assert!(validate_role_matrix(
            RecursionChildRole::Shrink,
            NATIVE_RECURSION_NUM_PV_ELTS,
            false
        )
        .is_ok());

        assert!(validate_role_matrix(RecursionChildRole::Core, NATIVE_RECURSION_NUM_PV_ELTS, true)
            .is_err());
        assert!(validate_role_matrix(
            RecursionChildRole::Compress,
            dt_stark::air::DT_PROOF_NUM_PV_ELTS,
            false,
        )
        .is_err());
        assert!(validate_role_matrix(
            RecursionChildRole::Compress,
            NATIVE_RECURSION_NUM_PV_ELTS,
            true
        )
        .is_err());
    }

    #[test]
    fn native_child_verifier_config_dispatches_by_role() {
        let core = native_child_verifier_config_for_role(NativeChildRole::Core);
        let compress = native_child_verifier_config_for_role(NativeChildRole::Compress);
        let shrink = native_child_verifier_config_for_role(NativeChildRole::Shrink);
        assert_eq!(core.whir.log_blowup, default_fri_config().log_blowup);
        assert_eq!(core.whir.num_queries, default_fri_config().num_queries);
        assert_eq!(compress.whir.log_blowup, compressed_fri_config().log_blowup);
        assert_eq!(compress.whir.num_queries, compressed_fri_config().num_queries);
        assert_eq!(shrink.whir.log_blowup, shrink_fri_config().log_blowup);
        assert_eq!(shrink.whir.num_queries, shrink_fri_config().num_queries);
        assert_ne!(compress.whir.log_blowup, core.whir.log_blowup);
        assert_ne!(compress.whir.num_queries, core.whir.num_queries);
    }

    #[test]
    fn statement_role_matrix_rejects_mismatched_layers() {
        assert!(
            validate_statement_role(RecursionChildRole::Core, RecursionStatementRole::Lift).is_ok()
        );
        assert!(validate_statement_role(
            RecursionChildRole::Compress,
            RecursionStatementRole::ReduceL2
        )
        .is_ok());
        assert!(validate_statement_role(
            RecursionChildRole::Compress,
            RecursionStatementRole::ReduceL3
        )
        .is_ok());
        assert!(validate_statement_role(
            RecursionChildRole::Shrink,
            RecursionStatementRole::RootShrink
        )
        .is_ok());

        assert!(validate_statement_role(
            RecursionChildRole::Core,
            RecursionStatementRole::ReduceL2
        )
        .is_err());
        assert!(validate_statement_role(
            RecursionChildRole::Shrink,
            RecursionStatementRole::ReduceL2
        )
        .is_err());
    }

    #[test]
    fn record_role_preflight_rejects_pv_count_mutation() {
        let recording_machine = core_recording_machine();
        let program =
            build_core_native_recursion_program(&recording_machine).expect("core program");

        let mut record = RecursionRecord::default();
        record.proof_records.push(RecursionProofRecord {
            proof_idx: 0,
            proof_shape: RecursionProofShapeRecord {
                role_id: role_id(NativeChildRole::Core),
                num_public_values: NATIVE_RECURSION_NUM_PV_ELTS,
                chips: vec![RecursionProofShapeChip { chip_idx: 0, ..Default::default() }],
                ..Default::default()
            },
            ..Default::default()
        });

        let err = validate_record_matches_program(&record, &program).unwrap_err();
        match err {
            NativeRecursionAssemblyError::Validation(message) => {
                assert!(
                    message.contains("role matrix mismatch"),
                    "unexpected validation message: {message}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    fn slot_record(proof_idx: usize) -> BuildingRecord {
        let mut record = RecursionRecord::default();
        record.proof_records.push(RecursionProofRecord { proof_idx, ..Default::default() });
        BuildingRecord::from_record(record)
    }

    #[test]
    fn proof_slots_publish_in_planner_order_not_completion_order() {
        let merged =
            merge_child_proof_shard_records(vec![slot_record(2), slot_record(0), slot_record(1)])
                .expect("out-of-order completion must seal by dense proof_idx");
        assert_eq!(
            merged.record().proof_records.iter().map(|proof| proof.proof_idx).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn proof_slots_reject_duplicate_missing_and_late_completion() {
        let duplicate = merge_child_proof_shard_records(vec![slot_record(0), slot_record(0)])
            .expect_err("duplicate slot must fail closed")
            .to_string();
        assert!(duplicate.contains("duplicate proof slot"), "{duplicate}");

        let mut missing = ProofSlotAssembler::new(2).expect("two slots");
        missing.admit(slot_record(0)).expect("first slot");
        let missing = missing.finish().expect_err("missing slot must fail closed").to_string();
        assert!(missing.contains("slot 1 was not completed"), "{missing}");

        let mut cancelled = ProofSlotAssembler::new(1).expect("one slot");
        cancelled.cancel();
        let late = cancelled
            .admit(slot_record(0))
            .expect_err("late completion after cancellation must fail")
            .to_string();
        assert!(late.contains("after node cancellation"), "{late}");
    }
}
