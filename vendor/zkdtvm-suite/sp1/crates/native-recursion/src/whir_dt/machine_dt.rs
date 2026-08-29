use std::{collections::BTreeMap, error::Error, fmt};

use dt_core_machine::{
    riscv::riscv_polyair::RiscvPolyAir,
    utils::prove_polyair::{POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD, POLYAIR_NUM_SKIP_ROUNDS},
};
use dt_stark::{
    air::{FullAir, InteractionScope, MachineAir, PairCol},
    koalabear_poseidon2::koala_bear_poseidon2::{
        compressed_fri_config, default_fri_config, shrink_fri_config,
    },
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
    },
    MachineRecord,
};
use p3_air::BaseAir;
use p3_challenger::CanObserve;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::Matrix;
use polyair::prover::SCMachineProver;

use crate::{
    batch_constraint_dt::{
        record_batch_constraint_from_views, BatchSumcheckAir, BatchTranscriptInputsAir,
    },
    child_views::{
        NativeAirAuthority, NativeChildMetadataView, NativeChildRole,
        NativeChildVerifierConfigView, NativeChildViews, NativeChipMetadata, NativeWhirConfigView,
    },
    config::{DIGEST_SIZE, D_EF, EF, F, SC},
    constraint_replay_dt::{
        annotate_constraint_replay_publications, constraint_challenge_rows,
        constraint_replay_bus_residual_report, constraint_terminal_rows, ConstraintBetaLadderAir,
        ConstraintBoundaryAir, ConstraintChallengeAir, ConstraintDagEvalAir, ConstraintFoldAir,
        ConstraintProgramTableAir, ConstraintRootTableAir, ConstraintTerminalAir,
    },
    primitives_dt::{bus::RangeCheckerBus, range::RangeCheckerAir},
    proof_shape_dt::{
        metadata_universe_from_view, record_proof_shape_from_views, NativeChipMetadataAir,
        ProofHeightSetAir, ProofShapeBinderAir,
    },
    statement_boundary_air_dt::{
        annotate_statement_publications, statement_part_b_bus_residual_report, StatementBoundaryAir,
    },
    statement_config_air_dt::StatementConfigAir,
    statement_dt::{
        NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
        STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
    },
    statement_hash_air_dt::{
        statement_hash_bus_residual_report, StatementDigestMode, StatementHashAir,
    },
    symbolic_expr_fixed_dt::{RecursionChildRole, RecursionFixedSymbolicChip},
    symbolic_ir_dt::{
        RecursionPolyAirChipIr, RecursionPolyAirDerivedRoot, RecursionPolyAirVerifierProgram,
    },
    system_dt::{
        record::RecursionRecordProfile, RecordingSC, RecordingStage, RecursionNativeProgram,
        RecursionRecord, RecursionRecordProfileSnapshot, RecursionStatementRole,
        StatementConfigRow,
    },
    transcript_dt::{
        bus::Poseidon2PermuteBus,
        merkle_path::MerklePathAir,
        poseidon2::{
            poseidon2_permute_cache_snapshot, Poseidon2PermuteAir, Poseidon2PermuteCacheSnapshot,
        },
        sponge::TranscriptSpongeAir,
    },
    validate::{
        check_provider_pools, check_trace_shapes_and_budget, prepare_recursion_record_with_profile,
        PrepareRecursionProfile,
    },
    whir_dt::{
        record_whir_from_views, whir_bus_residual_report, whir_role_config, WhirBatchEvalAir,
        WhirLeafExtStreamAir, WhirLeafStreamAir, WhirQueryFoldAir, WhirRoundAir, WhirSampleBandAir,
        WhirTwiddleTableAir,
    },
    Instant,
};

pub type CoreRecordingMachine = polyair::SCStarkMachine<RecordingSC, RiscvPolyAir<F>, D_EF>;
pub type CoreRecordingChip = polyair::Chip<RiscvPolyAir<F>, F, D_EF>;
pub type NativeRecordingMachine = polyair::SCStarkMachine<RecordingSC, NativeRecursionAir, D_EF>;
pub type NativeRecursionMachine = polyair::SCStarkMachine<SC, NativeRecursionAir, D_EF>;
pub type NativeRecursionProver = polyair::prover::SumcheckProver<SC, NativeRecursionAir, D_EF>;

const NATIVE_RECURSION_NUM_SKIP_ROUNDS: usize = 1;
const NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD: usize = 0;
pub(crate) const NATIVE_ROOT_SHRINK_DEGREE_FLOOR: usize = 5;
/// K3 arm 2: the L3 shrink prover (and its recording mirror) share the
/// degree-5 floor.
pub(crate) const NATIVE_SHRINK_DEGREE_FLOOR: usize = 5;
const ASSERT_PARALLEL_RECORD_EQ_ENV: &str = "DT_NATIVE_RECURSION_ASSERT_PARALLEL_RECORD_EQ";
const FINAL_RESIDUALS_ENV: &str = "DT_NATIVE_RECURSION_FINAL_RESIDUALS";
const INTERMEDIATE_RESIDUALS_ENV: &str = "DT_NATIVE_RECURSION_INTERMEDIATE_RESIDUALS";

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProveRecursionTimings {
    pub prepare_ms: u128,
    pub record_profile: RecursionRecordProfileSnapshot,
    pub prepare_profile: PrepareRecursionProfile,
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
    ChildVerifier(String),
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
            Self::ChildVerifier(message) => write!(f, "child verifier replay failed: {message}"),
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

#[derive(Debug, Clone)]
pub enum NativeRecursionAir {
    TranscriptSponge(TranscriptSpongeAir),
    MerklePath(MerklePathAir),
    Poseidon2Permute(Poseidon2PermuteAir),
    NativeChipMetadata(NativeChipMetadataAir),
    ProofShapeBinder(ProofShapeBinderAir),
    ProofHeightSet(ProofHeightSetAir),
    BatchTranscriptInputs(BatchTranscriptInputsAir),
    BatchSumcheck(BatchSumcheckAir),
    WhirTwiddleTable(WhirTwiddleTableAir),
    WhirSampleBand(WhirSampleBandAir),
    WhirRound(WhirRoundAir),
    WhirBatchEval(WhirBatchEvalAir),
    WhirQueryFold(WhirQueryFoldAir),
    WhirLeafStream(WhirLeafStreamAir),
    WhirLeafExtStream(WhirLeafExtStreamAir),
    ConstraintProgramTable(ConstraintProgramTableAir),
    ConstraintRootTable(ConstraintRootTableAir),
    ConstraintDagEval(ConstraintDagEvalAir),
    ConstraintFold(ConstraintFoldAir),
    ConstraintBetaLadder(ConstraintBetaLadderAir),
    ConstraintChallenge(ConstraintChallengeAir),
    ConstraintTerminal(ConstraintTerminalAir),
    ConstraintBoundary(ConstraintBoundaryAir),
    StatementBoundary(StatementBoundaryAir),
    StatementConfig(StatementConfigAir),
    StatementHash(StatementHashAir),
    Range7(RangeCheckerAir<7>),
    Range8(RangeCheckerAir<8>),
    Range9(RangeCheckerAir<9>),
    Range10(RangeCheckerAir<10>),
    Range11(RangeCheckerAir<11>),
    Range12(RangeCheckerAir<12>),
    Range21(RangeCheckerAir<21>),
}

impl NativeRecursionAir {
    pub fn all(program: &RecursionNativeProgram<F>) -> NativeRecursionAssemblyResult<Vec<Self>> {
        validate_native_recursion_program(program)?;
        let constraint_program = program.constraint_program.clone();
        let num_pv = program.num_child_public_values;
        let role_config = whir_role_config(role_id(native_child_role(program.role)));
        Ok(vec![
            Self::TranscriptSponge(TranscriptSpongeAir::default()),
            Self::MerklePath(MerklePathAir::default()),
            Self::Poseidon2Permute(Poseidon2PermuteAir::new(Poseidon2PermuteBus::new())),
            Self::NativeChipMetadata(NativeChipMetadataAir::new(
                program.native_chip_metadata.clone(),
            )),
            Self::ProofShapeBinder(ProofShapeBinderAir::new(
                num_pv,
                role_config,
                program.child_contains_global_bus,
            )),
            Self::ProofHeightSet(ProofHeightSetAir::default()),
            Self::BatchTranscriptInputs(BatchTranscriptInputsAir::new(
                num_pv,
                program.child_contains_global_bus,
            )),
            Self::BatchSumcheck(BatchSumcheckAir::new(num_pv, program.child_contains_global_bus)),
            Self::WhirTwiddleTable(WhirTwiddleTableAir::default()),
            Self::WhirSampleBand(WhirSampleBandAir::default()),
            Self::WhirRound(WhirRoundAir::new(role_config, num_pv)),
            Self::WhirBatchEval(WhirBatchEvalAir::new(role_config)),
            Self::WhirQueryFold(WhirQueryFoldAir::default()),
            Self::WhirLeafStream(WhirLeafStreamAir::default()),
            Self::WhirLeafExtStream(WhirLeafExtStreamAir::default()),
            Self::ConstraintProgramTable(ConstraintProgramTableAir::new(
                constraint_program.clone(),
            )),
            Self::ConstraintRootTable(ConstraintRootTableAir::new(constraint_program.clone())),
            Self::ConstraintDagEval(ConstraintDagEvalAir::new(constraint_program.clone())),
            Self::ConstraintFold(ConstraintFoldAir::new(constraint_program.clone())),
            Self::ConstraintBetaLadder(ConstraintBetaLadderAir::new(constraint_program.clone())),
            Self::ConstraintChallenge(ConstraintChallengeAir::new(
                constraint_program.clone(),
                num_pv,
                program.child_contains_global_bus,
            )),
            Self::ConstraintTerminal(ConstraintTerminalAir::new(
                constraint_program.clone(),
                num_pv,
                program.child_contains_global_bus,
            )),
            Self::ConstraintBoundary(ConstraintBoundaryAir::new(
                constraint_program,
                program.child_contains_global_bus,
            )),
            Self::StatementBoundary(StatementBoundaryAir::new(
                program.statement_role,
                num_pv,
                program.statement_config.clone(),
            )),
            Self::StatementConfig(StatementConfigAir::new(program.statement_config.clone())),
            Self::StatementHash(StatementHashAir::for_child(
                StatementDigestMode::from_role(program.statement_role),
                num_pv,
            )),
            Self::Range7(RangeCheckerAir::<7>::new(RangeCheckerBus::new())),
            Self::Range8(RangeCheckerAir::<8>::new(RangeCheckerBus::new())),
            Self::Range9(RangeCheckerAir::<9>::new(RangeCheckerBus::new())),
            Self::Range10(RangeCheckerAir::<10>::new(RangeCheckerBus::new())),
            Self::Range11(RangeCheckerAir::<11>::new(RangeCheckerBus::new())),
            Self::Range12(RangeCheckerAir::<12>::new(RangeCheckerBus::new())),
            Self::Range21(RangeCheckerAir::<21>::new(RangeCheckerBus::new())),
        ])
    }
}

macro_rules! dispatch_air {
    ($self:expr, $air:ident => $body:expr) => {
        match $self {
            NativeRecursionAir::TranscriptSponge($air) => $body,
            NativeRecursionAir::MerklePath($air) => $body,
            NativeRecursionAir::Poseidon2Permute($air) => $body,
            NativeRecursionAir::NativeChipMetadata($air) => $body,
            NativeRecursionAir::ProofShapeBinder($air) => $body,
            NativeRecursionAir::ProofHeightSet($air) => $body,
            NativeRecursionAir::BatchTranscriptInputs($air) => $body,
            NativeRecursionAir::BatchSumcheck($air) => $body,
            NativeRecursionAir::WhirTwiddleTable($air) => $body,
            NativeRecursionAir::WhirSampleBand($air) => $body,
            NativeRecursionAir::WhirRound($air) => $body,
            NativeRecursionAir::WhirBatchEval($air) => $body,
            NativeRecursionAir::WhirQueryFold($air) => $body,
            NativeRecursionAir::WhirLeafStream($air) => $body,
            NativeRecursionAir::WhirLeafExtStream($air) => $body,
            NativeRecursionAir::ConstraintProgramTable($air) => $body,
            NativeRecursionAir::ConstraintRootTable($air) => $body,
            NativeRecursionAir::ConstraintDagEval($air) => $body,
            NativeRecursionAir::ConstraintFold($air) => $body,
            NativeRecursionAir::ConstraintBetaLadder($air) => $body,
            NativeRecursionAir::ConstraintChallenge($air) => $body,
            NativeRecursionAir::ConstraintTerminal($air) => $body,
            NativeRecursionAir::ConstraintBoundary($air) => $body,
            NativeRecursionAir::StatementBoundary($air) => $body,
            NativeRecursionAir::StatementConfig($air) => $body,
            NativeRecursionAir::StatementHash($air) => $body,
            NativeRecursionAir::Range7($air) => $body,
            NativeRecursionAir::Range8($air) => $body,
            NativeRecursionAir::Range9($air) => $body,
            NativeRecursionAir::Range10($air) => $body,
            NativeRecursionAir::Range11($air) => $body,
            NativeRecursionAir::Range12($air) => $body,
            NativeRecursionAir::Range21($air) => $body,
        }
    };
}

impl BaseAir<F> for NativeRecursionAir {
    fn width(&self) -> usize {
        dispatch_air!(self, air => BaseAir::<F>::width(air))
    }
}

impl<AB> FullAir<AB> for NativeRecursionAir
where
    AB: dt_stark::air::FullAirBuilder<F = F>,
{
    fn width(&self) -> usize {
        dispatch_air!(self, air => FullAir::<AB>::width(air))
    }

    fn num_public_values(&self) -> usize {
        dispatch_air!(self, air => FullAir::<AB>::num_public_values(air))
    }

    fn required_max_beta_power(&self) -> usize {
        dispatch_air!(self, air => FullAir::<AB>::required_max_beta_power(air))
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        dispatch_air!(self, air => FullAir::<AB>::reserved_poly(air))
    }

    fn precompute_lc(&self, builder: &mut AB) {
        dispatch_air!(self, air => FullAir::<AB>::precompute_lc(air, builder))
    }

    fn eval(&self, builder: &mut AB) {
        dispatch_air!(self, air => FullAir::<AB>::eval(air, builder))
    }

    fn lookup(&self, builder: &mut AB) {
        dispatch_air!(self, air => FullAir::<AB>::lookup(air, builder))
    }

    fn global(&self) -> bool {
        dispatch_air!(self, air => FullAir::<AB>::global(air))
    }
}

impl MachineAir<F> for NativeRecursionAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        dispatch_air!(self, air => MachineAir::<F>::name(air))
    }

    fn preprocessed_width(&self) -> usize {
        dispatch_air!(self, air => MachineAir::<F>::preprocessed_width(air))
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        dispatch_air!(self, air => MachineAir::<F>::preprocessed_num_rows(air, program, instrs_len))
    }

    fn generate_preprocessed_trace(
        &self,
        program: &Self::Program,
    ) -> Option<dt_stark::sumcheck::trace::CompressedMatrix<F>> {
        dispatch_air!(self, air => MachineAir::<F>::generate_preprocessed_trace(air, program))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        dispatch_air!(self, air => MachineAir::<F>::num_rows(air, input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> dt_stark::sumcheck::trace::CompressedMatrix<F> {
        dispatch_air!(self, air => MachineAir::<F>::generate_trace(air, input, output))
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        dispatch_air!(self, air => MachineAir::<F>::generate_dependencies(air, input, output))
    }

    fn included(&self, record: &Self::Record) -> bool {
        dispatch_air!(self, air => MachineAir::<F>::included(air, record))
    }

    fn commit_scope(&self) -> InteractionScope {
        dispatch_air!(self, air => MachineAir::<F>::commit_scope(air))
    }

    fn local_only(&self) -> bool {
        dispatch_air!(self, air => MachineAir::<F>::local_only(air))
    }

    fn padding_row(&self) -> Vec<F> {
        dispatch_air!(self, air => MachineAir::<F>::padding_row(air))
    }
}

pub fn native_recursion_machine(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionMachine> {
    native_recursion_machine_with_config(program, SC::default())
}

pub fn native_recursion_machine_with_config(
    program: &RecursionNativeProgram<F>,
    config: SC,
) -> NativeRecursionAssemblyResult<NativeRecursionMachine> {
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

pub fn native_recursion_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    let machine = native_recursion_machine(program)?;
    Ok(polyair::prover::SumcheckProver { machine })
}

pub fn native_recursion_prover_with_config(
    program: &RecursionNativeProgram<F>,
    config: SC,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    let machine = native_recursion_machine_with_config(program, config)?;
    Ok(polyair::prover::SumcheckProver { machine })
}

pub fn native_root_shrink_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
    let chips: Vec<polyair::Chip<NativeRecursionAir, F, D_EF>> = NativeRecursionAir::all(program)?
        .into_iter()
        .map(|air| {
            polyair::Chip::<NativeRecursionAir, F, D_EF>::new_with_degree_floor(
                air,
                NATIVE_ROOT_SHRINK_DEGREE_FLOOR,
            )
        })
        .collect();
    print_chip_batch_profile("root_shrink", &chips);
    let machine =
        polyair::SCStarkMachine::new(SC::root_shrink(), chips, NATIVE_RECURSION_NUM_PV_ELTS, false);
    Ok(polyair::prover::SumcheckProver { machine })
}

/// K3 census: per-chip degree / logup batch / lookup count / perm width, one
/// line per machine construction — pins whether the degree floor reached every
/// chip (perm halving is machine-wide only if batch=4 shows on every
/// lookup-bearing chip). The `r` component is the D-27 census: distinct MAIN
/// columns actually referenced by constraints/lookups/precomputes (reserved
/// slots mapped through the chip's reserve list) vs the committed width.
pub(crate) fn print_chip_batch_profile(
    label: &str,
    chips: &[polyair::Chip<NativeRecursionAir, F, D_EF>],
) {
    use p3_air::BaseAir;
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

/// D-27 census: the set of MAIN columns a chip's symbolic constraint system
/// actually reads — gate constraints, lookup multiplicities, and precomputed
/// LCs, with `ReservedPoly` slots resolved through the reserve list. Uses a
/// zero-allocation recursive walker (`iter_all_var`'s chained collects go
/// quadratic on large expression trees — measured as a multi-minute hang).
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

pub fn prove_recursion(
    prover: &NativeRecursionProver,
    pk: &SCStarkProvingKey<SC>,
    record: RecursionRecord,
) -> NativeRecursionAssemblyResult<SCMachineProof<SC>> {
    prove_recursion_with_metrics(prover, pk, record).map(|(proof, _)| proof)
}

pub fn prove_recursion_with_metrics(
    prover: &NativeRecursionProver,
    pk: &SCStarkProvingKey<SC>,
    record: RecursionRecord,
) -> NativeRecursionAssemblyResult<(SCMachineProof<SC>, ProveRecursionMetrics)> {
    validate_record_matches_machine_role(prover, &record)?;

    let prepare_start = Instant::now();
    let (record, prepare_profile) = prepare_recursion_record_with_profile(prover, record);
    let prepare_ms = prepare_start.elapsed().as_millis();
    let record_profile = record.profile.snapshot();

    let tracegen_start = Instant::now();
    let traces = prover.generate_traces(&record);
    let tracegen_ms = tracegen_start.elapsed().as_millis();

    // D-12 origin tagging (opt-in via DT_NATIVE_D12_TRACE_DIGEST=1): one
    // stable digest per chip trace per prove. Two identical runs diffed on
    // these lines localize any run-to-run trace nondeterminism (the source of
    // the ±2 KB proof-byte jitter) to a (machine, chip) before the transcript
    // ever sees it.
    if crate::env_var("DT_NATIVE_D12_TRACE_DIGEST").is_ok() {
        for (name, trace) in &traces {
            let mut acc: u64 = 0xcbf29ce484222325;
            for value in &trace.main.values {
                acc ^= u64::from(value.as_canonical_u32());
                acc = acc.wrapping_mul(0x100000001b3);
            }
            println!(
                "native_d12_trace_digest role={:?} chip={} stored={} digest={acc:016x}",
                prover.machine.contains_global_bus,
                name,
                trace.stored_height(),
            );
        }
    }

    let trace_costs = traces
        .iter()
        .map(|(name, trace)| {
            let chip = prover
                .machine
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

    // R-M2-5 stack-height pin: when the suite JSON (or an authorized override) fixes an
    // explicit stack log height H for this stage, the stacked open requires every main
    // matrix to fit under H — the upstream record-max warn is not a gate, this is.
    if let Some(stack_h) = prover.config().mlpcs_stack_log_height_hint() {
        let tallest = trace_costs
            .iter()
            .map(|cost| cost.height.next_power_of_two().trailing_zeros() as usize)
            .max()
            .unwrap_or(0);
        if tallest > stack_h {
            return Err(NativeRecursionAssemblyError::Validation(format!(
                "tallest main matrix 2^{tallest} exceeds the pinned stack_log_height H = {stack_h}; \
                 the H freeze must be reopened (R-M2-5)"
            )));
        }
    }

    let budget_start = Instant::now();
    check_trace_shapes_and_budget(prover, &record, &traces)
        .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
    let budget_ms = budget_start.elapsed().as_millis();

    let pool_gate_start = Instant::now();
    check_provider_pools(&record)
        .map_err(|err| NativeRecursionAssemblyError::Validation(err.to_string()))?;
    let pool_gate_ms = pool_gate_start.elapsed().as_millis();

    if record.statement_public_values.is_none() {
        return Err(NativeRecursionAssemblyError::Validation(
            "native recursion statement public values missing".to_string(),
        ));
    }

    let mut challenger = prover.config().mlchallenger();
    pk.observe_into(&mut challenger);
    pcs::whir::profile::reset();
    let commit_start = Instant::now();
    let shard_data = prover.commit_with_pcs_stack_log_height(
        &record,
        traces,
        pk.preprocessed_pcs_stack_log_height,
    );
    let commit_ms = commit_start.elapsed().as_millis();
    let commit_profile = pcs::whir::profile::take();

    pcs::whir::profile::reset();
    let open_start = Instant::now();
    let shard = prover
        .open(
            pk,
            shard_data,
            &mut challenger,
            NATIVE_RECURSION_NUM_SKIP_ROUNDS,
            NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD,
        )
        .map_err(|err| NativeRecursionAssemblyError::Prove(err.to_string()))?;
    let open_ms = open_start.elapsed().as_millis();
    let open_profile = pcs::whir::profile::take();

    Ok((
        SCMachineProof { shard_proofs: vec![shard] },
        ProveRecursionMetrics {
            timings: ProveRecursionTimings {
                prepare_ms,
                record_profile,
                prepare_profile,
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

pub fn verify_recursion(
    prover: &NativeRecursionProver,
    vk: &SCStarkVerifyingKey<SC>,
    proof: &SCMachineProof<SC>,
) -> NativeRecursionAssemblyResult<()> {
    let mut challenger = prover.config().mlchallenger();
    prover
        .machine()
        .verify(
            vk,
            proof,
            &mut challenger,
            NATIVE_RECURSION_NUM_SKIP_ROUNDS,
            NATIVE_RECURSION_CHIP_LOG_HEIGHT_THRESHOLD,
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
    if stage == RecordingStage::Core {
        return Err(NativeRecursionAssemblyError::InvalidProgram(
            "native recording machines record at Compress or Shrink, not Core".to_string(),
        ));
    }
    // K3 arm 2: the Shrink recording stage mirrors the shrink-floored L3
    // prover — the recorded verify walk AND the L4 program IR compiled from
    // this machine must both see batch=4 chips or the replayed constraint
    // set diverges from the actual L3 proof.
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

/// K3 arm 2: the L3 (shrink) prover carries the same degree floor as the
/// root — logup batch 4 halves the shrink proof's perm openings, which is
/// what the L4 transcript/replay rows are priced in.
pub fn native_shrink_prover(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<NativeRecursionProver> {
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
    let machine =
        polyair::SCStarkMachine::new(SC::shrink(), chips, NATIVE_RECURSION_NUM_PV_ELTS, false);
    Ok(polyair::prover::SumcheckProver { machine })
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
    let constraint_program = RecursionPolyAirVerifierProgram {
        version: crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
        role: child_role,
        artifact_digest: [F::zero(); DIGEST_SIZE],
        chips: fixed_chips,
        max_required_beta_power,
    };
    let native_role = native_child_role(child_role);
    let native_chip_metadata = segment_metadata_universe(machine, native_role, 0);
    Ok(RecursionNativeProgram::new_with_roles(
        child_role,
        statement_role,
        num_child_public_values,
        child_contains_global_bus,
        native_chip_metadata,
        constraint_program,
        statement_config,
    ))
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

/// Builds the dual-segment ReduceL2 program for MIXED nodes (M1 trace/S1e obligation,
/// M2 mainline): segment A = the lift child machine's replay universe at offset 0,
/// segment B = the reduce child machine's universe at the mixed segment offset.
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

/// The canonical dual-segment reduce builder (R-M2-4): L2 and L3 machines share the
/// replay segment set {u1@0 (lift-child universe), u2@128 (ReduceL2-child universe)} and
/// differ only in statement role + config. Two-pass non-circularity holds because chip
/// DAGs depend on constructor constants, never preprocessed content, so u2 compiled from
/// ANY ReduceL2-shaped instance equals the final machine's u2.
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
    let constraint_program = RecursionPolyAirVerifierProgram {
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
    Ok(RecursionNativeProgram::new_with_roles(
        RecursionChildRole::Compress,
        statement_role,
        NATIVE_RECURSION_NUM_PV_ELTS,
        false,
        native_chip_metadata,
        constraint_program,
        statement_config,
    ))
}

/// Builds the L4 (root_shrink) program (R-M2-4): single replay segment {u3@0} = the
/// ReduceL3-child universe; children are L3 proofs at the shrink config. L4
/// single-segment is a design invariant — root_shrink never verifies a lift directly.
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

pub fn record_core_proof_shard(
    machine: &CoreRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    shard: &SCShardProof<RecordingSC>,
    proof_idx: usize,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shard(machine, vk, shard, proof_idx, seed_challenger, program)
}

pub fn record_native_proof_shard(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    shard: &SCShardProof<RecordingSC>,
    proof_idx: usize,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shard(machine, vk, shard, proof_idx, seed_challenger, program)
}

/// Records a native child whose replay universe lives at a non-zero static-chip-id offset
/// of a dual-segment (mixed) program.
pub fn record_native_proof_shard_in_segment(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    shard: &SCShardProof<RecordingSC>,
    proof_idx: usize,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shard_with_offset(
        machine,
        vk,
        shard,
        proof_idx,
        seed_challenger,
        program,
        static_chip_id_offset,
        false,
    )
}

fn record_child_proof_shard<A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    shard: &SCShardProof<RecordingSC>,
    proof_idx: usize,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord>
where
    A: MachineAir<F>,
    A: for<'a> FullAir<polyair::precompute::PrecomputeRowBuilder<'a, F, EF, EF>>,
    A: for<'a> FullAir<polyair::verifier::SumcheckVerifierConstraintFolder<'a, F, EF>>,
{
    record_child_proof_shard_with_offset(
        machine,
        vk,
        shard,
        proof_idx,
        seed_challenger,
        program,
        0,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_child_proof_shard_with_offset<A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    shard: &SCShardProof<RecordingSC>,
    proof_idx: usize,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
    program: &RecursionNativeProgram<F>,
    static_chip_id_offset: usize,
    refresh_first_proof_statement: bool,
) -> NativeRecursionAssemblyResult<RecursionRecord>
where
    A: MachineAir<F>,
    A: for<'a> FullAir<polyair::precompute::PrecomputeRowBuilder<'a, F, EF, EF>>,
    A: for<'a> FullAir<polyair::verifier::SumcheckVerifierConstraintFolder<'a, F, EF>>,
{
    validate_native_recursion_program(program)?;
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
    let static_chip_id_offset_value = static_chip_id_offset;
    let fs_memo_before = poseidon2_permute_cache_snapshot();
    let fs_start = Instant::now();
    let chips = machine.shard_chips_ordered(&shard.chip_ordering).collect::<Vec<_>>();
    let mut challenger = seed_challenger.fork_for_proof(proof_idx);
    challenger.observe_slice(&shard.public_values[..machine.num_pv_elts()]);
    polyair::verifier::Verifier::<RecordingSC, A, D_EF>::verify_shard(
        &machine.config,
        vk,
        &chips,
        &mut challenger,
        shard,
        POLYAIR_NUM_SKIP_ROUNDS,
        POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD,
        machine.contains_global_bus,
    )
    .map_err(|err| NativeRecursionAssemblyError::ChildVerifier(format!("{err}")))?;
    let verify_ms = fs_start.elapsed().as_millis();

    let mut record = challenger.take_record();
    record.profile.add_record_split("verify", verify_ms);
    record.profile.add_record_split(format!("child[{proof_idx}].fs_verify_walk"), verify_ms);
    record_poseidon2_memo_delta(
        &record.profile,
        format!("child[{proof_idx}].fs_verify_walk"),
        fs_memo_before,
    );
    let assembly_start = Instant::now();
    let view_start = Instant::now();
    let metadata_chips = native_metadata_for_shard(machine, shard)?;
    let child_role = native_child_role(program.role);
    let metadata = NativeChildMetadataView {
        role: child_role,
        air_authority: NativeAirAuthority::PublicMetadata,
        num_observed_public_values: machine.num_pv_elts(),
        contains_global_bus: machine.contains_global_bus,
        static_chip_id_offset: static_chip_id_offset_value,
        chips: &metadata_chips,
    };
    let verifier_config = native_child_verifier_config_for_role(child_role);
    let views = NativeChildViews::new(shard, vk, &metadata, &verifier_config).map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("NativeChildViews: {err:?}"))
    })?;
    record.profile.add_record_split(
        format!("child[{proof_idx}].view_build"),
        view_start.elapsed().as_millis(),
    );
    let proof_shape_start = Instant::now();
    record_proof_shape_from_views(&mut record, proof_idx, &views, true)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("proof_shape: {err:?}")))?;
    record.proof_record_mut(proof_idx).proof_shape.publish_whir_inputs = true;
    record.proof_record_mut(proof_idx).proof_shape.publish_terminal_summary = true;
    record.profile.add_record_split(
        format!("child[{proof_idx}].proof_shape_rows"),
        proof_shape_start.elapsed().as_millis(),
    );
    let batch_start = Instant::now();
    record_batch_constraint_from_views(&mut record, proof_idx, &views, true, true).map_err(
        |err| NativeRecursionAssemblyError::Record(format!("batch_constraint: {err:?}")),
    )?;
    record.profile.add_record_split(
        format!("child[{proof_idx}].batch_constraint_rows"),
        batch_start.elapsed().as_millis(),
    );
    let whir_start = Instant::now();
    record_whir_from_views(&mut record, proof_idx, &views, true)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("whir: {err:?}")))?;
    record.profile.add_record_split(
        format!("child[{proof_idx}].whir_total"),
        whir_start.elapsed().as_millis(),
    );
    let segment_start = Instant::now();
    {
        // R-M2-1: stamp the child's segment id-base onto its whir round rows (the 1022
        // recv payload backing); the binder band gates bind the same value in-circuit.
        let base = record
            .proof_records
            .iter()
            .find(|proof| proof.proof_idx == proof_idx)
            .map(|proof| proof.proof_shape.segment_id_base())
            .unwrap_or(0);
        for row in &mut record.proof_record_mut(proof_idx).whir.round_rows {
            row.summary_id_base = base;
        }
    }
    record.profile.add_record_split(
        format!("child[{proof_idx}].segment_row_patch"),
        segment_start.elapsed().as_millis(),
    );
    // ProofShapeBinder is the sole native-metadata consumer. Proof-shape
    // recording already published one metadata request for every chip row.
    let pool_start = Instant::now();
    record.profile.add_record_split(
        format!("child[{proof_idx}].metadata_pool_bookkeeping"),
        pool_start.elapsed().as_millis(),
    );
    let publish_start = Instant::now();
    annotate_constraint_replay_publications(&mut record, &program.constraint_program);
    assert_machine_record_fully_published(&record)?;
    record.profile.add_record_split(
        format!("child[{proof_idx}].publication_bookkeeping"),
        publish_start.elapsed().as_millis(),
    );
    record.profile.add_record_split("whir_assembly", assembly_start.elapsed().as_millis());
    if proof_idx == 0 && refresh_first_proof_statement && run_intermediate_residuals() {
        let residual_start = Instant::now();
        record
            .refresh_statement_public_values(program)
            .map_err(|err| NativeRecursionAssemblyError::Record(format!("statement: {err}")))?;
        annotate_statement_publications(&mut record);
        timed_residual_assert(&record, program, "first_proof")?;
        record.profile.add_record_split(
            format!("child[{proof_idx}].intermediate_residuals"),
            residual_start.elapsed().as_millis(),
        );
    }
    Ok(record)
}

pub fn record_core_proof_shards(
    machine: &CoreRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shards(machine, vk, proof, shard_indices, program, true)
}

pub fn record_native_proof_shards(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shards(machine, vk, proof, shard_indices, program, true)
}

pub fn record_native_proof_shards_pending_finalization(
    machine: &NativeRecordingMachine,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    record_child_proof_shards(machine, vk, proof, shard_indices, program, false)
}

fn record_child_proof_shards<A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
    assert_final_residual: bool,
) -> NativeRecursionAssemblyResult<RecursionRecord>
where
    A: MachineAir<F>,
    A: for<'a> FullAir<polyair::precompute::PrecomputeRowBuilder<'a, F, EF, EF>>,
    A: for<'a> FullAir<polyair::verifier::SumcheckVerifierConstraintFolder<'a, F, EF>>,
{
    let mut seed_challenger = machine.config.mlchallenger();
    vk.observe_into(&mut seed_challenger);
    let shard_records = record_child_proof_shards_parallel(
        machine,
        vk,
        proof,
        shard_indices,
        program,
        &seed_challenger,
    )?;
    let record = finalize_child_proof_shard_records(shard_records, program, assert_final_residual)?;
    if should_assert_parallel_record_eq() {
        let sequential_records = record_child_proof_shards_sequential(
            machine,
            vk,
            proof,
            shard_indices,
            program,
            &seed_challenger,
        )?;
        let sequential =
            finalize_child_proof_shard_records(sequential_records, program, assert_final_residual)?;
        assert_recursion_record_eq("child-shards", &record, &sequential);
    }
    Ok(record)
}

fn record_child_proof_shards_parallel<A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
) -> NativeRecursionAssemblyResult<Vec<RecursionRecord>>
where
    A: MachineAir<F>,
    A: for<'a> FullAir<polyair::precompute::PrecomputeRowBuilder<'a, F, EF, EF>>,
    A: for<'a> FullAir<polyair::verifier::SumcheckVerifierConstraintFolder<'a, F, EF>>,
{
    std::thread::scope(|scope| -> NativeRecursionAssemblyResult<Vec<RecursionRecord>> {
        let mut handles = Vec::with_capacity(shard_indices.len());
        for (proof_idx, shard_index) in shard_indices.iter().copied().enumerate() {
            let shard = proof.shard_proofs.get(shard_index).ok_or_else(|| {
                NativeRecursionAssemblyError::Record(format!(
                    "shard index {shard_index} out of range {}",
                    proof.shard_proofs.len()
                ))
            })?;
            let seed = seed_challenger.clone();
            handles.push(scope.spawn(move || {
                record_child_proof_shard(machine, vk, shard, proof_idx, &seed, program).map_err(
                    |err| {
                        NativeRecursionAssemblyError::Record(format!(
                            "child proof_idx={proof_idx} shard_index={shard_index}: {err}"
                        ))
                    },
                )
            }));
        }

        let mut records = Vec::with_capacity(handles.len());
        for handle in handles {
            let record = handle.join().map_err(|_| {
                NativeRecursionAssemblyError::Record(
                    "parallel child shard recorder panicked".to_string(),
                )
            })??;
            records.push(record);
        }
        Ok(records)
    })
}

fn record_child_proof_shards_sequential<A>(
    machine: &polyair::SCStarkMachine<RecordingSC, A, D_EF>,
    vk: &SCStarkVerifyingKey<RecordingSC>,
    proof: &SCMachineProof<RecordingSC>,
    shard_indices: &[usize],
    program: &RecursionNativeProgram<F>,
    seed_challenger: &<RecordingSC as SCStarkGenericConfig>::MlChallenger,
) -> NativeRecursionAssemblyResult<Vec<RecursionRecord>>
where
    A: MachineAir<F>,
    A: for<'a> FullAir<polyair::precompute::PrecomputeRowBuilder<'a, F, EF, EF>>,
    A: for<'a> FullAir<polyair::verifier::SumcheckVerifierConstraintFolder<'a, F, EF>>,
{
    let mut records = Vec::with_capacity(shard_indices.len());
    for (proof_idx, shard_index) in shard_indices.iter().copied().enumerate() {
        let shard = proof.shard_proofs.get(shard_index).ok_or_else(|| {
            NativeRecursionAssemblyError::Record(format!(
                "shard index {shard_index} out of range {}",
                proof.shard_proofs.len()
            ))
        })?;
        records.push(
            record_child_proof_shard(machine, vk, shard, proof_idx, seed_challenger, program)
                .map_err(|err| {
                    NativeRecursionAssemblyError::Record(format!(
                        "child proof_idx={proof_idx} shard_index={shard_index}: {err}"
                    ))
                })?,
        );
    }
    Ok(records)
}

fn finalize_child_proof_shard_records(
    shard_records: Vec<RecursionRecord>,
    program: &RecursionNativeProgram<F>,
    assert_final_residual: bool,
) -> NativeRecursionAssemblyResult<RecursionRecord> {
    let mut record = RecursionRecord::default();
    let append_start = Instant::now();
    for mut shard_record in shard_records {
        record.append(&mut shard_record);
    }
    record.profile.add_record_split("combined.final_append", append_start.elapsed().as_millis());
    let publish_start = Instant::now();
    annotate_constraint_replay_publications(&mut record, &program.constraint_program);
    record
        .refresh_statement_public_values(program)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("statement: {err}")))?;
    annotate_statement_publications(&mut record);
    assert_machine_record_fully_published(&record)?;
    record
        .profile
        .add_record_split("combined.final_publication", publish_start.elapsed().as_millis());
    if (assert_final_residual && run_final_residuals()) || run_intermediate_residuals() {
        let residual_start = Instant::now();
        timed_residual_assert(&record, program, "combined_record")?;
        record
            .profile
            .add_record_split("combined.final_residuals", residual_start.elapsed().as_millis());
    }
    Ok(record)
}

fn record_poseidon2_memo_delta(
    profile: &RecursionRecordProfile,
    label: impl Into<String>,
    before: Poseidon2PermuteCacheSnapshot,
) {
    let after = poseidon2_permute_cache_snapshot();
    profile.add_poseidon2_memo_delta(
        label,
        after.hits.saturating_sub(before.hits),
        after.misses.saturating_sub(before.misses),
    );
}

/// Installs the ReduceL2 threaded vk_root input on a recorded reduce record and rebuilds
/// the statement + publication annotations on top of it (R-S8 threaded-self slot).
pub fn set_statement_vk_root(
    record: &mut RecursionRecord,
    vk_root: [F; DIGEST_SIZE],
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    record.statement_vk_root = vk_root;
    annotate_constraint_replay_publications(record, &program.constraint_program);
    record
        .refresh_statement_public_values(program)
        .map_err(|err| NativeRecursionAssemblyError::Record(format!("statement: {err}")))?;
    annotate_statement_publications(record);
    if run_final_residuals() {
        timed_residual_assert(record, program, "set_statement_vk_root")?;
    }
    Ok(())
}

pub fn timed_residual_assert(
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

pub fn run_final_residuals() -> bool {
    cfg!(test) || env_flag(FINAL_RESIDUALS_ENV) || run_intermediate_residuals()
}

fn env_flag(name: &str) -> bool {
    match crate::env_var(name) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

pub(crate) fn should_assert_parallel_record_eq() -> bool {
    match crate::env_var(ASSERT_PARALLEL_RECORD_EQ_ENV) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

pub(crate) fn assert_recursion_record_eq(
    label: &str,
    parallel: &RecursionRecord,
    sequential: &RecursionRecord,
) {
    if parallel != sequential {
        panic!(
            "{label} parallel record mismatch: parallel_fp={} sequential_fp={} parallel_stats={:?} sequential_stats={:?}",
            parallel.compute_cache_fingerprint(),
            sequential.compute_cache_fingerprint(),
            parallel.stats(),
            sequential.stats()
        );
    }
    eprintln!("{label} parallel record dev gate passed");
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
        let has_opened_eval = proof
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

pub fn native_metadata_for_shard<ChildSC, A>(
    machine: &polyair::SCStarkMachine<ChildSC, A, D_EF>,
    shard: &SCShardProof<RecordingSC>,
) -> NativeRecursionAssemblyResult<Vec<NativeChipMetadata>>
where
    ChildSC: SCStarkGenericConfig<Val = F>,
    A: MachineAir<F>,
{
    let mut metadata = native_metadata_from_machine(machine);
    let proof_view = crate::child_views::NativeChildProofView::new(shard).map_err(|err| {
        NativeRecursionAssemblyError::Record(format!("NativeChildProofView: {err:?}"))
    })?;
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
        chip.permutation_width = shard.dimensions[2][opened_chip.index].width;
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

fn validate_native_recursion_program(
    program: &RecursionNativeProgram<F>,
) -> NativeRecursionAssemblyResult<()> {
    validate_role_matrix(
        program.role,
        program.num_child_public_values,
        program.child_contains_global_bus,
    )?;
    validate_statement_role(program.role, program.statement_role)?;
    validate_statement_config(program.statement_role, &program.statement_config)?;
    if program.constraint_program.role != program.role {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "constraint program role {:?} does not match child role {:?}",
            program.constraint_program.role, program.role
        )));
    }
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

fn validate_role_matrix(
    role: RecursionChildRole,
    num_child_public_values: usize,
    child_contains_global_bus: bool,
) -> NativeRecursionAssemblyResult<()> {
    let (expected_num_pv, expected_cgb) = match role {
        RecursionChildRole::Core => (dt_stark::air::DT_PROOF_NUM_PV_ELTS, true),
        RecursionChildRole::Compress | RecursionChildRole::Shrink => {
            (NATIVE_RECURSION_NUM_PV_ELTS, false)
        }
    };
    if num_child_public_values != expected_num_pv || child_contains_global_bus != expected_cgb {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "invalid role matrix for {:?}: num_child_public_values={} child_contains_global_bus={} expected ({expected_num_pv}, {expected_cgb})",
            role, num_child_public_values, child_contains_global_bus
        )));
    }
    Ok(())
}

fn validate_statement_config(
    statement_role: RecursionStatementRole,
    statement_config: &[StatementConfigRow],
) -> NativeRecursionAssemblyResult<()> {
    match statement_role {
        RecursionStatementRole::Lift => {
            if !statement_config.is_empty() {
                return Err(NativeRecursionAssemblyError::InvalidProgram(
                    "lift machines carry no StatementConfig rows".to_string(),
                ));
            }
        }
        RecursionStatementRole::ReduceL2 => {
            // M1: exactly one baked class (BAKED_LIFT); M2's L3 machine adds BAKED_L2.
            let valid = statement_config.len() == 1 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_LIFT;
            if !valid {
                return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                    "ReduceL2 requires exactly one BAKED_LIFT StatementConfig row, got {:?}",
                    statement_config.iter().map(|row| row.class_id).collect::<Vec<_>>()
                )));
            }
        }
        RecursionStatementRole::ReduceL3 => {
            // R-M2-3: exactly {class 0 = vk_lift digest, class 1 = vk_L2 digest}.
            let valid = statement_config.len() == 2 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_LIFT &&
                statement_config[1].class_id == STATEMENT_CONFIG_CLASS_BAKED_L2;
            if !valid {
                return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                    "ReduceL3 requires the BAKED_LIFT + BAKED_L2 StatementConfig rows, got {:?}",
                    statement_config.iter().map(|row| row.class_id).collect::<Vec<_>>()
                )));
            }
        }
        RecursionStatementRole::RootShrink => {
            // R-M2-3: exactly {class 2 = vk_L3 digest}.
            let valid = statement_config.len() == 1 &&
                statement_config[0].class_id == STATEMENT_CONFIG_CLASS_BAKED_L3;
            if !valid {
                return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
                    "RootShrink requires exactly one BAKED_L3 StatementConfig row, got {:?}",
                    statement_config.iter().map(|row| row.class_id).collect::<Vec<_>>()
                )));
            }
        }
    }
    Ok(())
}

fn validate_statement_role(
    child_role: RecursionChildRole,
    statement_role: RecursionStatementRole,
) -> NativeRecursionAssemblyResult<()> {
    let valid = matches!(
        (child_role, statement_role),
        (RecursionChildRole::Core, RecursionStatementRole::Lift) |
            (RecursionChildRole::Compress, RecursionStatementRole::ReduceL2) |
            (RecursionChildRole::Compress, RecursionStatementRole::ReduceL3) |
            (RecursionChildRole::Shrink, RecursionStatementRole::RootShrink)
    );
    if !valid {
        return Err(NativeRecursionAssemblyError::InvalidProgram(format!(
            "invalid statement role {:?} for child role {:?}",
            statement_role, child_role
        )));
    }
    Ok(())
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

pub fn validate_record_matches_machine_role(
    prover: &NativeRecursionProver,
    record: &RecursionRecord,
) -> NativeRecursionAssemblyResult<()> {
    let (role, num_public_values) = machine_child_role_config(&prover.machine)?;
    let expected_role_id = role_id(native_child_role(role));
    for proof in &record.proof_records {
        if proof.proof_shape.is_empty() {
            continue;
        }
        if proof.proof_shape.role_id != expected_role_id ||
            proof.proof_shape.num_public_values != num_public_values
        {
            return Err(NativeRecursionAssemblyError::Validation(format!(
                "proof {} role matrix mismatch: role_id={} num_public_values={} expected role_id={} num_public_values={}",
                proof.proof_idx,
                proof.proof_shape.role_id,
                proof.proof_shape.num_public_values,
                expected_role_id,
                num_public_values
            )));
        }
    }
    Ok(())
}

fn machine_child_role_config(
    machine: &NativeRecursionMachine,
) -> NativeRecursionAssemblyResult<(RecursionChildRole, usize)> {
    let role = machine
        .chips
        .iter()
        .find_map(|chip| match &chip.air {
            NativeRecursionAir::ConstraintProgramTable(air) => Some(air.program.role),
            _ => None,
        })
        .ok_or_else(|| {
            NativeRecursionAssemblyError::InvalidProgram(
                "native recursion machine missing constraint program table".to_string(),
            )
        })?;
    let num_public_values = machine
        .chips
        .iter()
        .find_map(|chip| match &chip.air {
            NativeRecursionAir::ProofShapeBinder(air) => Some(air.num_public_values),
            _ => None,
        })
        .ok_or_else(|| {
            NativeRecursionAssemblyError::InvalidProgram(
                "native recursion machine missing proof-shape binder".to_string(),
            )
        })?;
    Ok((role, num_public_values))
}

fn apply_constraint_terminal_consumers(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
    report: &mut BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>,
) {
    if let Some(residual) = report.get_mut("1007 TranscriptEvent") {
        for row in constraint_challenge_rows(record, program) {
            if let crate::constraint_replay_dt::ConstraintChallengeRow::Gsr {
                proof_idx,
                chip_idx,
                num_public_values,
                c_chips,
                gcs_limbs,
                lcs_limbs,
                ..
            } = row
            {
                // e2/e6 tidx only — the E9 stride (4th arg) is irrelevant here.
                let layout = crate::batch_constraint_dt::BatchTranscriptLayout::new(
                    num_public_values,
                    c_chips,
                    0,
                    crate::batch_constraint_dt::BATCH_SUMCHECK_EVALS,
                );
                for (offset, value) in gcs_limbs.into_iter().enumerate() {
                    apply_report_residual(
                        residual,
                        vec![
                            proof_idx as u32,
                            (layout.e2_tidx(chip_idx) + offset) as u32,
                            0,
                            value.as_canonical_u32(),
                        ],
                        -1,
                    );
                }
                for (offset, value) in lcs_limbs.into_iter().enumerate() {
                    apply_report_residual(
                        residual,
                        vec![
                            proof_idx as u32,
                            (layout.e6_tidx(chip_idx) + offset) as u32,
                            0,
                            value.as_canonical_u32(),
                        ],
                        -1,
                    );
                }
            }
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
    use p3_field::AbstractField;

    use super::*;
    use crate::{
        config::POSEIDON2_WIDTH,
        system_dt::{RecursionProofRecord, RecursionProofShapeChip, RecursionProofShapeRecord},
        validate::set_budget_log_height_override_for_test,
    };

    struct BudgetOverrideGuard;

    impl Drop for BudgetOverrideGuard {
        fn drop(&mut self) {
            set_budget_log_height_override_for_test(None);
        }
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
        record.refresh_cache_fingerprint();

        let err = match prove_recursion(&prover, &pk, record) {
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
        let prover = native_recursion_prover(&program).expect("native recursion prover");

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

        let err = validate_record_matches_machine_role(&prover, &record).unwrap_err();
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
}
