use core::{borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{ChallengeExtension, FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    batch_constraint_dt::{
        BatchOpeningPointBus, BatchSumcheckClaimChainBus, SumcheckOutBus, SUMCHECK_OUT_EQ,
        SUMCHECK_OUT_PERM_ALPHA, SUMCHECK_OUT_PERM_BETA,
    },
    config::{D_EF, F},
    constraint_replay_dt::{
        bus::{
            BetaLadderChainBus, ConstraintChallengeBus, ConstraintEqChainBus,
            ConstraintFoldChainBus, ConstraintFoldPlanChainBus, ConstraintHeightInverseBus,
            ConstraintNodeValueBus, ConstraintProgramBus, ConstraintRootTableBus,
        },
        columns::{
            ConstraintBetaLadderCols, ConstraintBoundaryCols,
            ConstraintBoundaryDenominatorCols, ConstraintBoundaryPackedCols,
            ConstraintBoundaryPrecomputedCols, ConstraintBoundaryReservedCols,
            ConstraintChallengeCols, ConstraintChallengeDenominatorCols,
            ConstraintChallengePrecomputedCols, ConstraintChallengeReservedCols,
            ConstraintDagEvalCols, ConstraintDagEvalReservedCols, ConstraintFoldCols,
            ConstraintFoldDenominatorCols, ConstraintFoldPackedCols, ConstraintFoldPrecomputedCols,
            ConstraintFoldReservedCols, ConstraintProgramPreprocessedCols,
            ConstraintRootTablePreprocessedCols, ConstraintTerminalCols,
            ConstraintTerminalColsNarrow, ConstraintTerminalDenominatorsNarrowCols,
            ConstraintTerminalDenominatorsWideCols, ConstraintTerminalLcsDenominator,
            ConstraintTerminalOuterDenominators, ConstraintTerminalPackedCommonCols,
            ConstraintTerminalPackedStateCols, ConstraintTerminalPrecomputedNarrowCols,
            ConstraintTerminalPrecomputedWideCols, ConstraintTerminalReservedNarrowCols,
            ConstraintTerminalReservedWideCols, CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT,
            CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS, CONSTRAINT_CHAIN_LIMBS,
            CONSTRAINT_CHALLENGE_BETA_POWER, CONSTRAINT_CHALLENGE_BETA_SEPTIX,
            CONSTRAINT_CHALLENGE_IS_FIRST, CONSTRAINT_CHALLENGE_IS_LAST, CONSTRAINT_CHALLENGE_LCS,
            CONSTRAINT_CHALLENGE_PERM_ALPHA, CONSTRAINT_CHALLENGE_STATE_LCS,
            CONSTRAINT_FOLD_BATCH_SIZE, CONSTRAINT_FOLD_ROOT_SLOTS, CONSTRAINT_GLOBAL_CHAIN_BLOCKS,
            CONSTRAINT_LEAF_BETA_POWER, CONSTRAINT_LEAF_BETA_SEPTIX, CONSTRAINT_LEAF_IS_FIRST_ROW,
            CONSTRAINT_LEAF_IS_LAST_ROW, CONSTRAINT_LEAF_MAIN, CONSTRAINT_LEAF_PERM_ALPHA,
            CONSTRAINT_LEAF_PRECOMPUTED, CONSTRAINT_LEAF_PREPROCESSED, CONSTRAINT_LEAF_PUBLIC,
            CONSTRAINT_LEAF_RESERVED_POLY, CONSTRAINT_MAX_BETA_POWERS, CONSTRAINT_OP_ADD,
            CONSTRAINT_OP_CONST, CONSTRAINT_OP_FUSED, CONSTRAINT_OP_LEAF, CONSTRAINT_OP_MUL,
            CONSTRAINT_OP_SUB, CONSTRAINT_ROOT_MULTIPLICITY, CONSTRAINT_ROOT_PRECOMPUTE_DENOM,
            CONSTRAINT_TERMINAL_LCS_LIMBS, CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT,
            NUM_CONSTRAINT_BETA_LADDER_COLS, NUM_CONSTRAINT_BOUNDARY_COLS,
            NUM_CONSTRAINT_CHALLENGE_COLS,
            NUM_CONSTRAINT_DAG_EVAL_COLS, NUM_CONSTRAINT_DAG_EVAL_RESERVED_COLS,
            NUM_CONSTRAINT_FOLD_COLS, NUM_CONSTRAINT_PROGRAM_COLS,
            NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS, NUM_CONSTRAINT_ROOT_TABLE_COLS,
            NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS, NUM_CONSTRAINT_TERMINAL_NARROW_COLS,
        },
        trace::{
            ConstraintBetaLadderTraceGenerator, ConstraintBoundaryTraceGenerator,
            ConstraintChallengeTraceGenerator,
            ConstraintDagEvalTraceGenerator, ConstraintFoldTraceGenerator,
            ConstraintProgramTraceGenerator, ConstraintRootTableTraceGenerator,
            ConstraintTerminalTraceGenerator,
        },
    },
    proof_shape_dt::{
        ProofShapeBatchDimBus, ProofShapeChipMetaBus, ProofShapeGlobalPackedBus,
        ProofShapeSummaryBus, ProofShapeValuesBus, PROOF_SHAPE_BATCH_MAIN,
        PROOF_SHAPE_BATCH_PERMUTATION, PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES,
    },
    symbolic_ir_dt::RecursionPolyAirVerifierProgram,
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::sponge::TranscriptEventBus,
    whir_dt::WhirOpenedEvalBus,
};

#[derive(Debug, Clone)]
pub struct ConstraintProgramTableAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub program_bus: ConstraintProgramBus,
}

impl ConstraintProgramTableAir {
    pub fn new(program: RecursionPolyAirVerifierProgram) -> Self {
        Self { program, program_bus: ConstraintProgramBus::new() }
    }
}

impl BaseAir<F> for ConstraintProgramTableAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_PROGRAM_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintProgramTableAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_PROGRAM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.program_bus.required_max_beta_power_floor()
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_CONSTRAINT_PROGRAM_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominator = {
            let prep = builder.preprocessed();
            let local: &ConstraintProgramPreprocessedCols<AB::VarMaybeExt> = prep.borrow();
            program_denominator(&self.program_bus, builder, local)
        };
        builder.retain_precomputed(denominator);
    }

    fn eval(&self, _builder: &mut AB) {}

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        builder.send(local[NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS].clone());
    }
}

impl MachineAir<F> for ConstraintProgramTableAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        "NativeConstraintProgramTable".to_string()
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(ConstraintProgramTraceGenerator::trace_height(&self.program))
    }

    fn preprocessed_width(&self) -> usize {
        NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(ConstraintProgramTraceGenerator::trace_height(&program.constraint_program))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Some(ConstraintProgramTraceGenerator::generate_preprocessed_trace(
            &program.constraint_program,
        ))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintProgramTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintRootTableAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub root_bus: ConstraintRootTableBus,
    pub height_bus: ConstraintHeightInverseBus,
}

impl ConstraintRootTableAir {
    pub fn new(program: RecursionPolyAirVerifierProgram) -> Self {
        Self {
            program,
            root_bus: ConstraintRootTableBus::new(),
            height_bus: ConstraintHeightInverseBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintRootTableAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_ROOT_TABLE_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintRootTableAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_ROOT_TABLE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.root_bus
            .required_max_beta_power_floor()
            .max(self.height_bus.required_max_beta_power_floor())
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS)
            .map(PairCol::Prep)
            .chain((0..NUM_CONSTRAINT_ROOT_TABLE_COLS).map(PairCol::Main))
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let prep = builder.preprocessed();
            let local: &ConstraintRootTablePreprocessedCols<AB::VarMaybeExt> = prep.borrow();
            [
                root_table_denominator(&self.root_bus, builder, local),
                self.height_bus.denominator(
                    builder,
                    local.root_ord.clone(),
                    local.node_idx.clone(),
                ),
            ]
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        let root_kind = local
            [core::mem::offset_of!(ConstraintRootTablePreprocessedCols<u8>, root_kind)]
        .clone();
        let height_mult = local[NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS +
            core::mem::offset_of!(
                crate::constraint_replay_dt::columns::ConstraintRootTableCols<u8>,
                height_mult
            )]
        .clone();
        builder.assert_zero(
            height_mult *
                (root_kind -
                    AB::VarMaybeExt::from(AB::F::from_canonical_usize(
                        crate::constraint_replay_dt::columns::CONSTRAINT_ROOT_HEIGHT_INVERSE,
                    ))),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();
        builder.send(local[NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS].clone());
        builder.send(local[NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS + 1].clone());
    }
}

impl MachineAir<F> for ConstraintRootTableAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        "NativeConstraintRootTable".to_string()
    }

    fn num_rows(&self, _input: &Self::Record) -> Option<usize> {
        Some(ConstraintRootTableTraceGenerator::trace_height(&self.program))
    }

    fn preprocessed_width(&self) -> usize {
        NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(ConstraintRootTableTraceGenerator::trace_height(&program.constraint_program))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        Some(ConstraintRootTableTraceGenerator::generate_preprocessed_trace(
            &program.constraint_program,
        ))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintRootTableTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintDagEvalAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub program_bus: ConstraintProgramBus,
    pub node_bus: ConstraintNodeValueBus,
    pub challenge_bus: ConstraintChallengeBus,
    pub opened_eval_bus: WhirOpenedEvalBus,
    pub proof_values_bus: ProofShapeValuesBus,
}

impl ConstraintDagEvalAir {
    pub fn new(program: RecursionPolyAirVerifierProgram) -> Self {
        Self {
            program,
            program_bus: ConstraintProgramBus::new(),
            node_bus: ConstraintNodeValueBus::new(),
            challenge_bus: ConstraintChallengeBus::new(),
            opened_eval_bus: WhirOpenedEvalBus::new(),
            proof_values_bus: ProofShapeValuesBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintDagEvalAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_DAG_EVAL_COLS
    }
}

fn dag_main_indices<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|index| start + index)
}

fn dag_reserved_main_indices() -> ConstraintDagEvalReservedCols<usize> {
    ConstraintDagEvalReservedCols {
        chip_idx: core::mem::offset_of!(ConstraintDagEvalCols<u8>, chip_idx),
        static_chip_id: core::mem::offset_of!(ConstraintDagEvalCols<u8>, static_chip_id),
        is_const: core::mem::offset_of!(ConstraintDagEvalCols<u8>, is_const),
        is_add: core::mem::offset_of!(ConstraintDagEvalCols<u8>, is_add),
        is_sub: core::mem::offset_of!(ConstraintDagEvalCols<u8>, is_sub),
        is_mul: core::mem::offset_of!(ConstraintDagEvalCols<u8>, is_mul),
        is_fused: core::mem::offset_of!(ConstraintDagEvalCols<u8>, is_fused),
        lhs_idx: core::mem::offset_of!(ConstraintDagEvalCols<u8>, lhs_idx),
        rhs_idx: core::mem::offset_of!(ConstraintDagEvalCols<u8>, rhs_idx),
        aux: core::mem::offset_of!(ConstraintDagEvalCols<u8>, aux),
        fanout: core::mem::offset_of!(ConstraintDagEvalCols<u8>, fanout),
        leaf_flags: dag_main_indices(core::mem::offset_of!(ConstraintDagEvalCols<u8>, leaf_flags)),
        value_0: core::mem::offset_of!(ConstraintDagEvalCols<u8>, value),
        opened_batch_pos: core::mem::offset_of!(ConstraintDagEvalCols<u8>, opened_batch_pos),
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintDagEvalAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_DAG_EVAL_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.program_bus.required_max_beta_power_floor(),
            self.node_bus.required_max_beta_power_floor(),
            self.challenge_bus.required_max_beta_power_floor(),
            self.opened_eval_bus.required_max_beta_power_floor(),
            self.proof_values_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let columns: Vec<PairCol> =
            dag_reserved_main_indices().as_slice().iter().copied().map(PairCol::Main).collect();
        debug_assert_eq!(columns.len(), NUM_CONSTRAINT_DAG_EVAL_RESERVED_COLS);
        columns
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let (denominators, packed_values) = {
            let main = builder.main();
            let local: &ConstraintDagEvalCols<AB::VarMaybeExt> = main.borrow();
            (
                dag_denominators(self, builder, local),
                [
                    AB::pack_ext_limbs(&local.value),
                    AB::pack_ext_limbs(&local.lhs_value),
                    AB::pack_ext_limbs(&local.rhs_value),
                    AB::pack_ext_limbs(&local.third_value),
                ],
            )
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
        for packed in packed_values {
            builder.retain_precomputed(packed);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintDagEvalReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let precomputed = builder.precomputed();
        let precomputed_binding = precomputed.row_slice(0);
        let packed = &precomputed_binding.deref()[CONSTRAINT_DAG_LOOKUP_COUNT..
            CONSTRAINT_DAG_LOOKUP_COUNT + CONSTRAINT_DAG_PACKED_COUNT];

        let mut leaf_sum = AB::zero_maybe();
        for flag in &local.leaf_flags {
            assert_bool(builder, flag.clone());
            leaf_sum = leaf_sum + flag.clone();
        }
        let is_valid = leaf_sum.clone() +
            local.is_const.clone() +
            local.is_add.clone() +
            local.is_sub.clone() +
            local.is_mul.clone() +
            local.is_fused.clone();
        for flag in [
            local.is_const.clone(),
            local.is_add.clone(),
            local.is_sub.clone(),
            local.is_mul.clone(),
            local.is_fused.clone(),
        ] {
            assert_bool(builder, flag);
        }
        assert_bool(builder, is_valid.clone());

        let opened_mult = local.leaf_flags[CONSTRAINT_LEAF_PREPROCESSED].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_MAIN].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_RESERVED_POLY].clone();
        builder.assert_zero((AB::one_maybe() - is_valid) * local.fanout.clone());

        constrain_node_value(builder, local, packed);
        constrain_opened_source(builder, local, opened_mult);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintDagEvalReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        let operand_pair_mult = local.is_add.clone() +
            local.is_sub.clone() +
            local.is_mul.clone() +
            local.is_fused.clone();
        let opened_mult = local.leaf_flags[CONSTRAINT_LEAF_PREPROCESSED].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_MAIN].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_RESERVED_POLY].clone();
        let challenge_mult = local.leaf_flags[CONSTRAINT_LEAF_PERM_ALPHA].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_BETA_POWER].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_BETA_SEPTIX].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_IS_FIRST_ROW].clone() +
            local.leaf_flags[CONSTRAINT_LEAF_IS_LAST_ROW].clone();
        let lhs_or_precomputed_mult =
            operand_pair_mult.clone() + local.leaf_flags[CONSTRAINT_LEAF_PRECOMPUTED].clone();
        let is_valid =
            local.leaf_flags.iter().cloned().fold(AB::zero_maybe(), |sum, flag| sum + flag) +
                local.is_const.clone() +
                local.is_add.clone() +
                local.is_sub.clone() +
                local.is_mul.clone() +
                local.is_fused.clone();

        // Order matches dag_denominators: program recv, self node send,
        // lhs-or-precomputed/rhs/third node recvs, opened/public/challenge leaf recvs.
        builder.recv(is_valid);
        builder.send(local.fanout.clone());
        builder.recv(lhs_or_precomputed_mult);
        builder.recv(operand_pair_mult);
        builder.recv(local.is_fused.clone());
        builder.recv(opened_mult);
        builder.recv(local.leaf_flags[CONSTRAINT_LEAF_PUBLIC].clone());
        builder.recv(challenge_mult);
    }
}

const CONSTRAINT_DAG_LOOKUP_COUNT: usize = 8;
const CONSTRAINT_DAG_PACKED_COUNT: usize = 4;

impl MachineAir<F> for ConstraintDagEvalAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        // Display name only; wire identity comes from NativeAirId::wire_name.
        "NativeConstraintDagEval".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintDagEvalTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintDagEvalTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintFoldAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub root_bus: ConstraintRootTableBus,
    pub node_bus: ConstraintNodeValueBus,
    pub opened_eval_bus: WhirOpenedEvalBus,
    pub challenge_bus: ConstraintChallengeBus,
    pub height_bus: ConstraintHeightInverseBus,
    pub chip_meta_bus: ProofShapeChipMetaBus,
    pub plan_chain_bus: ConstraintFoldPlanChainBus,
    pub fold_chain_bus: ConstraintFoldChainBus,
}

impl ConstraintFoldAir {
    pub fn new(program: RecursionPolyAirVerifierProgram) -> Self {
        Self {
            program,
            root_bus: ConstraintRootTableBus::new(),
            node_bus: ConstraintNodeValueBus::new(),
            opened_eval_bus: WhirOpenedEvalBus::new(),
            challenge_bus: ConstraintChallengeBus::new(),
            height_bus: ConstraintHeightInverseBus::new(),
            chip_meta_bus: ProofShapeChipMetaBus::new(),
            plan_chain_bus: ConstraintFoldPlanChainBus::new(),
            fold_chain_bus: ConstraintFoldChainBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintFoldAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_FOLD_COLS
    }
}

fn fold_main_indices<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|index| start + index)
}

fn fold_reserved_main_indices() -> ConstraintFoldReservedCols<usize> {
    ConstraintFoldReservedCols {
        is_skip: core::mem::offset_of!(ConstraintFoldCols<u8>, is_skip),
        is_gate: core::mem::offset_of!(ConstraintFoldCols<u8>, is_gate),
        is_batch: core::mem::offset_of!(ConstraintFoldCols<u8>, is_batch),
        height_inverse: core::mem::offset_of!(ConstraintFoldCols<u8>, root_nodes),
        batch_count: core::mem::offset_of!(ConstraintFoldCols<u8>, batch_count),
        multiplicity_signs: fold_main_indices(core::mem::offset_of!(
            ConstraintFoldCols<u8>,
            multiplicity_signs
        )),
        batch_has_second: core::mem::offset_of!(ConstraintFoldCols<u8>, batch_has_second),
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintFoldAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_FOLD_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.root_bus.required_max_beta_power_floor(),
            self.node_bus.required_max_beta_power_floor(),
            self.opened_eval_bus.required_max_beta_power_floor(),
            self.challenge_bus.required_max_beta_power_floor(),
            self.height_bus.required_max_beta_power_floor(),
            self.chip_meta_bus.required_max_beta_power_floor(),
            self.plan_chain_bus.required_max_beta_power_floor(),
            self.fold_chain_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        fold_reserved_main_indices().as_slice().iter().copied().map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let precomputed = {
            let main = builder.main();
            let local: &ConstraintFoldCols<AB::VarMaybeExt> = main.borrow();
            fold_precomputed(self, builder, local)
        };
        for value in precomputed.as_slice() {
            builder.retain_precomputed(value.clone());
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintFoldReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let precomputed = builder.precomputed();
        let precomputed_binding = precomputed.row_slice(0);
        let precomputed: &ConstraintFoldPrecomputedCols<AB::VarExt> =
            precomputed_binding.deref().borrow();

        for flag in [
            local.is_skip.clone(),
            local.is_gate.clone(),
            local.is_batch.clone(),
            local.batch_has_second.clone(),
        ] {
            assert_bool(builder, flag);
        }
        let is_valid = fold_is_valid::<AB>(local);
        assert_bool(builder, is_valid.clone());
        builder.assert_zero(
            local.batch_has_second.clone() * (AB::one_maybe() - local.is_batch.clone()),
        );

        constrain_fold_value(builder, local, &precomputed.packed);
        constrain_fold_chain(builder, local, &precomputed.packed);
        builder.assert_zero_ext(precomputed.packed.gate_position.clone() * local.is_gate.clone());
        builder.assert_zero_ext(precomputed.packed.batch_position.clone() * local.is_batch.clone());
        builder.assert_zero_ext(precomputed.packed.skip_position.clone() * local.is_skip.clone());
        builder.assert_zero_ext(precomputed.packed.skip_height.clone() * local.is_skip.clone());
        builder.assert_zero_ext(
            precomputed.packed.non_skip_successor.clone() *
                (local.is_gate.clone() + local.is_batch.clone()),
        );
        builder.assert_zero_ext(precomputed.packed.skip_successor.clone() * local.is_skip.clone());
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintFoldReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let root_mults = fold_node_multiplicities::<AB>(local);
        for mult in &root_mults {
            builder.recv(mult.clone());
        }
        for mult in &root_mults {
            builder.recv(mult.clone());
        }
        builder.recv(local.is_batch.clone());
        builder.recv(local.is_skip.clone());
        builder.recv(local.is_skip.clone());
        let is_valid = fold_is_valid::<AB>(local);
        builder.recv(is_valid.clone());
        builder.recv(is_valid.clone());
        builder.send(is_valid.clone());
        builder.recv(is_valid.clone());
        builder.send(is_valid);
    }
}

impl MachineAir<F> for ConstraintFoldAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        // Display name only; wire identity comes from NativeAirId::wire_name.
        "NativeConstraintFold".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintFoldTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintFoldTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintBetaLadderAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub challenge_bus: ConstraintChallengeBus,
    pub ladder_bus: BetaLadderChainBus,
    pub sumcheck_out_bus: SumcheckOutBus,
}

impl ConstraintBetaLadderAir {
    pub fn new(program: RecursionPolyAirVerifierProgram) -> Self {
        Self {
            program,
            challenge_bus: ConstraintChallengeBus::new(),
            ladder_bus: BetaLadderChainBus::new(),
            sumcheck_out_bus: SumcheckOutBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintBetaLadderAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_BETA_LADDER_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintBetaLadderAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_BETA_LADDER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.challenge_bus.required_max_beta_power_floor(),
            self.ladder_bus.required_max_beta_power_floor(),
            self.sumcheck_out_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_CONSTRAINT_BETA_LADDER_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &ConstraintBetaLadderCols<AB::VarMaybeExt> = main.borrow();
            beta_ladder_denominators(self, builder, local)
        };
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintBetaLadderCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        constrain_beta_ladder_row(builder, local);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintBetaLadderCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        builder.recv(local.challenges_recv_mult.clone());
        builder.recv(local.challenges_recv_mult.clone());
        builder.recv(local.is_valid.clone() - local.is_seed.clone());
        builder.send(local.is_valid.clone() - local.is_last.clone());
        builder.send(local.serve_mult.clone());
        builder.send(local.alpha_serve_mult.clone());
        builder.send(local.septix_serve_mult.clone());
    }
}

impl MachineAir<F> for ConstraintBetaLadderAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        // Display name only; wire identity comes from NativeAirId::wire_name.
        "NativeBetaLadder".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintBetaLadderTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintBetaLadderTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintChallengeAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub num_public_values: usize,
    pub seed_prefix_limbs: usize,
    pub challenge_bus: ConstraintChallengeBus,
    pub eq_chain_bus: ConstraintEqChainBus,
    pub transcript_event_bus: TranscriptEventBus,
    pub batch_dim_bus: ProofShapeBatchDimBus,
    pub fold_plan_chain_bus: ConstraintFoldPlanChainBus,
}

impl ConstraintChallengeAir {
    pub fn new(
        program: RecursionPolyAirVerifierProgram,
        num_public_values: usize,
        child_contains_global_bus: bool,
    ) -> Self {
        Self {
            program,
            num_public_values,
            seed_prefix_limbs: crate::batch_constraint_dt::columns::batch_seed_prefix_limbs(
                child_contains_global_bus,
            ),
            challenge_bus: ConstraintChallengeBus::new(),
            eq_chain_bus: ConstraintEqChainBus::new(),
            transcript_event_bus: TranscriptEventBus::new(),
            batch_dim_bus: ProofShapeBatchDimBus::new(),
            fold_plan_chain_bus: ConstraintFoldPlanChainBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintChallengeAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_CHALLENGE_COLS
    }
}

fn challenge_reserved_main_indices() -> ConstraintChallengeReservedCols<usize> {
    ConstraintChallengeReservedCols {
        is_valid: core::mem::offset_of!(ConstraintChallengeCols<u8>, is_valid),
        selector_first_send_mult: core::mem::offset_of!(
            ConstraintChallengeCols<u8>,
            selector_first_send_mult
        ),
        selector_last_send_mult: core::mem::offset_of!(
            ConstraintChallengeCols<u8>,
            selector_last_send_mult
        ),
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintChallengeAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_CHALLENGE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.challenge_bus.required_max_beta_power_floor(),
            self.eq_chain_bus.required_max_beta_power_floor(),
            self.transcript_event_bus.required_max_beta_power_floor(),
            self.batch_dim_bus.required_max_beta_power_floor(),
            self.fold_plan_chain_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        challenge_reserved_main_indices().as_slice().iter().copied().map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let precomputed = {
            let main = builder.main();
            let local: &ConstraintChallengeCols<AB::VarMaybeExt> = main.borrow();
            challenge_precomputed(self, builder, local)
        };
        for value in precomputed.as_slice() {
            builder.retain_precomputed(value.clone());
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintChallengeReservedCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        constrain_selector_row(builder, local);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintChallengeReservedCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();

        // Order matches `ConstraintChallengeDenominatorCols`.
        for _ in 0..CONSTRAINT_TERMINAL_LCS_LIMBS {
            builder.recv(local.is_valid.clone());
        }
        builder.recv(local.is_valid.clone());
        builder.send(local.is_valid.clone() + local.is_valid.clone());
        builder.send(local.selector_first_send_mult.clone());
        builder.send(local.selector_last_send_mult.clone());
        builder.recv(local.is_valid.clone());
        builder.recv(local.is_valid.clone());
    }
}

impl MachineAir<F> for ConstraintChallengeAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        "NativeConstraintChallenge".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintChallengeTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintChallengeTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintTerminalAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub num_public_values: usize,
    pub child_contains_global_bus: bool,
    pub summary_bus: ProofShapeSummaryBus,
    pub proof_values_bus: ProofShapeValuesBus,
    pub opening_point_bus: BatchOpeningPointBus,
    pub sumcheck_claim_chain_bus: BatchSumcheckClaimChainBus,
    pub sumcheck_out_bus: SumcheckOutBus,
    pub challenge_bus: ConstraintChallengeBus,
    pub fold_chain_bus: ConstraintFoldChainBus,
    pub fold_plan_chain_bus: ConstraintFoldPlanChainBus,
    pub eq_chain_bus: ConstraintEqChainBus,
}

impl ConstraintTerminalAir {
    pub fn new(
        program: RecursionPolyAirVerifierProgram,
        num_public_values: usize,
        child_contains_global_bus: bool,
    ) -> Self {
        Self {
            program,
            num_public_values,
            child_contains_global_bus,
            summary_bus: ProofShapeSummaryBus::new(),
            proof_values_bus: ProofShapeValuesBus::new(),
            opening_point_bus: BatchOpeningPointBus::new(),
            sumcheck_claim_chain_bus: BatchSumcheckClaimChainBus::new(),
            sumcheck_out_bus: SumcheckOutBus::new(),
            challenge_bus: ConstraintChallengeBus::new(),
            fold_chain_bus: ConstraintFoldChainBus::new(),
            fold_plan_chain_bus: ConstraintFoldPlanChainBus::new(),
            eq_chain_bus: ConstraintEqChainBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintTerminalAir {
    fn width(&self) -> usize {
        terminal_width(self.child_contains_global_bus)
    }
}

/// Every role commits the narrow replay thread. Core-only state/global
/// boundary witnesses live in `ConstraintBoundaryAir` at one row per child.
pub const fn terminal_width(_child_contains_global_bus: bool) -> usize {
    NUM_CONSTRAINT_TERMINAL_NARROW_COLS
}

#[cfg(test)]
const TERMINAL_WIDE_LOOKUP_PREFIX: usize = 51;
#[cfg(test)]
const TERMINAL_NARROW_LOOKUP_PREFIX: usize = 16;

fn terminal_main_indices<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|index| start + index)
}

/// Semantic map from the compact wide reserved row to the historical main
/// trace. The typed initializer and the typed reserved row share one field
/// order; tests pin each field to its original main index.
fn terminal_reserved_wide_main_indices() -> ConstraintTerminalReservedWideCols<usize> {
    ConstraintTerminalReservedWideCols {
        is_seed: core::mem::offset_of!(ConstraintTerminalCols<u8>, is_seed),
        is_eq_step: core::mem::offset_of!(ConstraintTerminalCols<u8>, is_eq_step),
        is_lcs_step: core::mem::offset_of!(ConstraintTerminalCols<u8>, is_lcs_step),
        is_final: core::mem::offset_of!(ConstraintTerminalCols<u8>, is_final),
        public_values: terminal_main_indices(core::mem::offset_of!(
            ConstraintTerminalCols<u8>,
            public_values
        )),
        state_chain_send_mult: core::mem::offset_of!(
            ConstraintTerminalCols<u8>,
            state_chain_send_mult
        ),
        state_clock_changed: core::mem::offset_of!(ConstraintTerminalCols<u8>, state_clock_changed),
        state_clock_delta_inverse: core::mem::offset_of!(
            ConstraintTerminalCols<u8>,
            state_clock_delta_inverse
        ),
        eq_chain_send_mult: core::mem::offset_of!(ConstraintTerminalCols<u8>, eq_chain_send_mult),
    }
}

fn terminal_reserved_narrow_main_indices() -> ConstraintTerminalReservedNarrowCols<usize> {
    ConstraintTerminalReservedNarrowCols {
        is_seed: core::mem::offset_of!(ConstraintTerminalColsNarrow<u8>, is_seed),
        is_eq_step: core::mem::offset_of!(ConstraintTerminalColsNarrow<u8>, is_eq_step),
        is_lcs_step: core::mem::offset_of!(ConstraintTerminalColsNarrow<u8>, is_lcs_step),
        is_final: core::mem::offset_of!(ConstraintTerminalColsNarrow<u8>, is_final),
        state_chain_send_mult: core::mem::offset_of!(
            ConstraintTerminalColsNarrow<u8>,
            state_chain_send_mult
        ),
        eq_chain_send_mult: core::mem::offset_of!(
            ConstraintTerminalColsNarrow<u8>,
            eq_chain_send_mult
        ),
    }
}

struct ConstraintTerminalReservedCommon<'a, T> {
    is_seed: &'a T,
    is_eq_step: &'a T,
    is_lcs_step: &'a T,
    is_final: &'a T,
    state_chain_send_mult: &'a T,
    eq_chain_send_mult: &'a T,
}

fn terminal_reserved_common_wide<T>(
    local: &ConstraintTerminalReservedWideCols<T>,
) -> ConstraintTerminalReservedCommon<'_, T> {
    ConstraintTerminalReservedCommon {
        is_seed: &local.is_seed,
        is_eq_step: &local.is_eq_step,
        is_lcs_step: &local.is_lcs_step,
        is_final: &local.is_final,
        state_chain_send_mult: &local.state_chain_send_mult,
        eq_chain_send_mult: &local.eq_chain_send_mult,
    }
}

fn terminal_reserved_common_narrow<T>(
    local: &ConstraintTerminalReservedNarrowCols<T>,
) -> ConstraintTerminalReservedCommon<'_, T> {
    ConstraintTerminalReservedCommon {
        is_seed: &local.is_seed,
        is_eq_step: &local.is_eq_step,
        is_lcs_step: &local.is_lcs_step,
        is_final: &local.is_final,
        state_chain_send_mult: &local.state_chain_send_mult,
        eq_chain_send_mult: &local.eq_chain_send_mult,
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintTerminalAir {
    fn width(&self) -> usize {
        terminal_width(self.child_contains_global_bus)
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.summary_bus.required_max_beta_power_floor(),
            self.proof_values_bus.required_max_beta_power_floor(),
            self.opening_point_bus.required_max_beta_power_floor(),
            self.sumcheck_claim_chain_bus.required_max_beta_power_floor(),
            self.sumcheck_out_bus.required_max_beta_power_floor(),
            self.challenge_bus.required_max_beta_power_floor(),
            self.fold_chain_bus.required_max_beta_power_floor(),
            self.fold_plan_chain_bus.required_max_beta_power_floor(),
            self.eq_chain_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        terminal_reserved_narrow_main_indices()
            .as_slice()
            .iter()
            .copied()
            .map(PairCol::Main)
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let precomputed = {
            let local: &ConstraintTerminalColsNarrow<AB::VarMaybeExt> = builder.main().borrow();
            terminal_precomputed_narrow(self, builder, local)
        };
        for value in precomputed.as_slice() {
            builder.retain_precomputed(value.clone());
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let precomputed = builder.precomputed();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintTerminalReservedNarrowCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();
        let precomputed_binding = precomputed.row_slice(0);
        let packed: &ConstraintTerminalPrecomputedNarrowCols<AB::VarExt> =
            precomputed_binding.deref().borrow();
        let common = terminal_reserved_common_narrow(local);
        constrain_terminal_common(builder, &common);
        constrain_terminal_eq(builder, &common, &packed.common);
        constrain_terminal_state_narrow(
            builder,
            &common,
            &packed.common,
            self.child_contains_global_bus,
        );
        constrain_terminal_final(builder, &common, &packed.common);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintTerminalReservedNarrowCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();
        let common = terminal_reserved_common_narrow(local);
        terminal_lookup_head(builder, &common);
        terminal_lookup_lcs_state(builder, &common, self.child_contains_global_bus);
    }
}

fn constrain_terminal_common<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
) {
    let kind_sum = local.is_seed.clone() +
        local.is_eq_step.clone() +
        local.is_lcs_step.clone() +
        local.is_final.clone();
    for flag in [
        local.is_seed.clone(),
        local.is_eq_step.clone(),
        local.is_lcs_step.clone(),
        local.is_final.clone(),
    ] {
        assert_bool(builder, flag);
    }
    assert_bool(builder, kind_sum);
    builder.assert_eq(
        local.state_chain_send_mult.clone(),
        local.is_seed.clone() + local.is_lcs_step.clone(),
    );
    builder.assert_zero(
        local.eq_chain_send_mult.clone() *
            (AB::one_maybe() - local.is_seed.clone() - local.is_eq_step.clone()),
    );
}

fn terminal_lookup_head<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
) {
    builder.recv(local.is_seed.clone());
    builder.recv(local.is_eq_step.clone());
    builder.recv(local.is_eq_step.clone());
    builder.recv(local.is_final.clone());
    builder.recv(local.is_final.clone());
    builder.recv(local.is_final.clone());
    builder.recv(local.is_eq_step.clone() + local.is_final.clone());
    builder.send(local.eq_chain_send_mult.clone());
}

fn terminal_lookup_lcs_state<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
    child_contains_global_bus: bool,
) {
    builder.recv(local.is_lcs_step.clone());
    let reduce_final = if child_contains_global_bus {
        AB::zero_maybe()
    } else {
        local.is_final.clone()
    };
    // Core Boundary consumes the last LCS-chain message directly.  Reduce has
    // no Boundary chip, so its final row remains the chain sink and pins zero.
    builder.recv(local.is_lcs_step.clone() + reduce_final);
    builder.send(local.state_chain_send_mult.clone());
}

impl MachineAir<F> for ConstraintTerminalAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) {}

    fn name(&self) -> String {
        "NativeConstraintTerminal".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintTerminalTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintTerminalTraceGenerator::generate_trace_compressed_narrow(input, &self.program)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

/// Core-only boundary checker split out of the long, narrow Terminal replay.
/// It has one active row per published child proof instead of repeating the
/// 77 public values and reciprocal witnesses on every Terminal row.
#[derive(Debug, Clone)]
pub struct ConstraintBoundaryAir {
    pub program: RecursionPolyAirVerifierProgram,
    pub enabled: bool,
    pub proof_values_bus: ProofShapeValuesBus,
    pub global_packed_bus: ProofShapeGlobalPackedBus,
    pub challenge_bus: ConstraintChallengeBus,
}

impl ConstraintBoundaryAir {
    pub fn new(program: RecursionPolyAirVerifierProgram, enabled: bool) -> Self {
        Self {
            program,
            enabled,
            proof_values_bus: ProofShapeValuesBus::new(),
            global_packed_bus: ProofShapeGlobalPackedBus::new(),
            challenge_bus: ConstraintChallengeBus::new(),
        }
    }
}

impl BaseAir<F> for ConstraintBoundaryAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_BOUNDARY_COLS
    }
}

fn constraint_boundary_reserved_main_indices() -> ConstraintBoundaryReservedCols<usize> {
    ConstraintBoundaryReservedCols {
        is_valid: core::mem::offset_of!(ConstraintBoundaryCols<u8>, is_valid),
        public_values: terminal_main_indices(core::mem::offset_of!(
            ConstraintBoundaryCols<u8>,
            public_values
        )),
        state_clock_changed: core::mem::offset_of!(
            ConstraintBoundaryCols<u8>,
            state_clock_changed
        ),
        state_clock_delta_inverse: core::mem::offset_of!(
            ConstraintBoundaryCols<u8>,
            state_clock_delta_inverse
        ),
    }
}

fn constraint_boundary_precomputed<AB: FullAirBuilder>(
    air: &ConstraintBoundaryAir,
    builder: &AB,
    local: &ConstraintBoundaryCols<AB::VarMaybeExt>,
) -> ConstraintBoundaryPrecomputedCols<AB::VarExt> {
    ConstraintBoundaryPrecomputedCols {
        denominators: ConstraintBoundaryDenominatorCols {
            public_values: core::array::from_fn(|index| {
                air.proof_values_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    const_maybe::<AB>(PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES),
                    const_maybe::<AB>(TERMINAL_PV_IDXS[index]),
                    local.public_values[index].clone(),
                )
            }),
            global_packed: core::array::from_fn(|row| {
                let shape_idx_base = 48 + 8 * row;
                let values = core::array::from_fn(|column| {
                    let pv_idx = shape_idx_base + column;
                    match pv_idx {
                        48..=50 => local.public_values[pv_idx - 42].clone(),
                        // PublicValues::empty is protocol padding and is authenticated as zero.
                        51 => AB::zero_maybe(),
                        52..=119 => local.public_values[9 + pv_idx - 52].clone(),
                        _ => unreachable!("fixed packed Global row is outside core public values"),
                    }
                });
                air.global_packed_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    const_maybe::<AB>(shape_idx_base),
                    &values,
                )
            }),
            perm_alpha: air.challenge_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_PERM_ALPHA),
                AB::zero_maybe(),
                AB::zero_maybe(),
                local.perm_alpha.clone(),
            ),
            beta_powers: core::array::from_fn(|power| {
                air.challenge_bus.denominator(
                    builder,
                    local.proof_idx.clone(),
                    const_maybe::<AB>(CONSTRAINT_CHALLENGE_BETA_POWER),
                    const_maybe::<AB>(power + 1),
                    AB::zero_maybe(),
                    local.beta_powers[power].clone(),
                )
            }),
            state_lcs: air.challenge_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_STATE_LCS),
                local.c_chips.clone(),
                AB::zero_maybe(),
                local.state_lcs.clone(),
            ),
        },
        packed: ConstraintBoundaryPackedCols {
            state_lcs: AB::pack_ext_limbs(&local.state_lcs),
            state: ConstraintTerminalPackedStateCols {
                perm_alpha: AB::pack_ext_limbs(&local.perm_alpha),
                beta_powers: core::array::from_fn(|index| {
                    AB::pack_ext_limbs(&local.beta_powers[index])
                }),
                state_transition_recv_inverse: AB::pack_ext_limbs(
                    &local.state_transition_recv_inverse,
                ),
                state_transition_send_inverse: AB::pack_ext_limbs(
                    &local.state_transition_send_inverse,
                ),
                init_address_recv_inverse: AB::pack_ext_limbs(&local.init_address_recv_inverse),
                init_address_send_inverse: AB::pack_ext_limbs(&local.init_address_send_inverse),
                finalize_address_recv_inverse: AB::pack_ext_limbs(
                    &local.finalize_address_recv_inverse,
                ),
                finalize_address_send_inverse: AB::pack_ext_limbs(
                    &local.finalize_address_send_inverse,
                ),
                global_chain_source_inverse: AB::pack_ext_limbs(
                    &local.global_chain_source_inverse,
                ),
                global_chain_sink_inverse: AB::pack_ext_limbs(&local.global_chain_sink_inverse),
            },
        },
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ConstraintBoundaryAir {
    fn width(&self) -> usize {
        NUM_CONSTRAINT_BOUNDARY_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        self.proof_values_bus
            .required_max_beta_power_floor()
            .max(self.global_packed_bus.required_max_beta_power_floor())
            .max(self.challenge_bus.required_max_beta_power_floor())
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        constraint_boundary_reserved_main_indices()
            .as_slice()
            .iter()
            .copied()
            .map(PairCol::Main)
            .collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let precomputed = {
            let local: &ConstraintBoundaryCols<AB::VarMaybeExt> = builder.main().borrow();
            constraint_boundary_precomputed(self, builder, local)
        };
        for value in precomputed.as_slice() {
            builder.retain_precomputed(value.clone());
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintBoundaryReservedCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();
        let precomputed = builder.precomputed();
        let packed_binding = precomputed.row_slice(0);
        let packed: &ConstraintBoundaryPrecomputedCols<AB::VarExt> =
            packed_binding.deref().borrow();
        assert_bool(builder, local.is_valid.clone());
        constrain_state_imbalance(
            builder,
            local.is_valid.clone(),
            &local.public_values,
            local.state_clock_changed.clone(),
            local.state_clock_delta_inverse.clone(),
            packed.packed.state_lcs.clone(),
            &packed.packed.state,
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &ConstraintBoundaryReservedCols<AB::VarMaybeExt> =
            local_binding.deref().borrow();
        for _ in 0..CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT {
            builder.recv(local.is_valid.clone());
        }
        for _ in 0..CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS {
            builder.recv(local.is_valid.clone());
        }
        builder.recv(local.is_valid.clone());
        for _ in 0..CONSTRAINT_CHAIN_LIMBS {
            builder.recv(local.is_valid.clone());
        }
        builder.recv(local.is_valid.clone());
    }
}

impl MachineAir<F> for ConstraintBoundaryAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeConstraintBoundary".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(ConstraintBoundaryTraceGenerator::trace_height(input, &self.program))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ConstraintBoundaryTraceGenerator::generate_trace_compressed(input, &self.program)
    }

    fn included(&self, record: &Self::Record) -> bool {
        self.enabled && !record.proof_records.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

const TERMINAL_PV_START_PC: usize = 40;
const TERMINAL_PV_NEXT_PC: usize = 41;
const TERMINAL_PV_EXECUTION_SHARD: usize = 44;
const TERMINAL_PV_PREVIOUS_INIT_ADDR: usize = 45;
const TERMINAL_PV_LAST_INIT_ADDR: usize = 46;
const TERMINAL_PV_PREVIOUS_FINALIZE_ADDR: usize = 47;
const TERMINAL_PV_LAST_FINALIZE_ADDR: usize = 48;
const TERMINAL_PV_START_CLK: usize = 49;
const TERMINAL_PV_EXIT_CLK: usize = 50;
pub(crate) const TERMINAL_PV_IDXS: [usize; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT] = [
    TERMINAL_PV_START_PC,
    TERMINAL_PV_NEXT_PC,
    TERMINAL_PV_EXECUTION_SHARD,
    TERMINAL_PV_PREVIOUS_INIT_ADDR,
    TERMINAL_PV_LAST_INIT_ADDR,
    TERMINAL_PV_PREVIOUS_FINALIZE_ADDR,
    TERMINAL_PV_LAST_FINALIZE_ADDR,
    TERMINAL_PV_START_CLK,
    TERMINAL_PV_EXIT_CLK,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
];
const PV_COL_START_PC: usize = 0;
const PV_COL_NEXT_PC: usize = 1;
const PV_COL_EXECUTION_SHARD: usize = 2;
const PV_COL_PREVIOUS_INIT_ADDR: usize = 3;
const PV_COL_LAST_INIT_ADDR: usize = 4;
const PV_COL_PREVIOUS_FINALIZE_ADDR: usize = 5;
const PV_COL_LAST_FINALIZE_ADDR: usize = 6;
const PV_COL_START_CLK: usize = 7;
const PV_COL_EXIT_CLK: usize = 8;
const PV_COL_GLOBAL_HAS: usize = 9;
const PV_COL_GLOBAL_COUNT: usize = 10;
const PV_COL_GLOBAL_START: usize = 11;
const PV_COL_GLOBAL_END: usize = 44;

fn program_denominator<AB: FullAirBuilder>(
    bus: &ConstraintProgramBus,
    builder: &AB,
    local: &ConstraintProgramPreprocessedCols<AB::VarMaybeExt>,
) -> AB::VarExt {
    bus.denominator(
        builder,
        local.static_chip_id.clone(),
        local.node_idx.clone(),
        local.op_code.clone(),
        local.lhs_idx.clone(),
        local.rhs_idx.clone(),
        local.third_idx.clone(),
        local.aux.clone(),
        local.leaf_kind.clone(),
        local.fanout.clone(),
    )
}

fn dag_op_code_expr<AB: FullAirBuilder>(
    local: &ConstraintDagEvalCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    let is_leaf = local.leaf_flags.iter().cloned().fold(AB::zero_maybe(), |sum, flag| sum + flag);
    is_leaf * const_maybe::<AB>(CONSTRAINT_OP_LEAF) +
        local.is_const.clone() * const_maybe::<AB>(CONSTRAINT_OP_CONST) +
        local.is_add.clone() * const_maybe::<AB>(CONSTRAINT_OP_ADD) +
        local.is_sub.clone() * const_maybe::<AB>(CONSTRAINT_OP_SUB) +
        local.is_mul.clone() * const_maybe::<AB>(CONSTRAINT_OP_MUL) +
        local.is_fused.clone() * const_maybe::<AB>(CONSTRAINT_OP_FUSED)
}

fn dag_leaf_kind_expr<AB: FullAirBuilder>(
    local: &ConstraintDagEvalCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    let mut leaf_kind = AB::zero_maybe();
    for (idx, flag) in local.leaf_flags.iter().enumerate() {
        leaf_kind = leaf_kind + flag.clone() * const_maybe::<AB>(idx);
    }
    leaf_kind
}

fn root_table_denominator<AB: FullAirBuilder>(
    bus: &ConstraintRootTableBus,
    builder: &AB,
    local: &ConstraintRootTablePreprocessedCols<AB::VarMaybeExt>,
) -> AB::VarExt {
    bus.denominator(
        builder,
        local.static_chip_id.clone(),
        local.root_kind.clone(),
        local.root_ord.clone(),
        local.node_idx.clone(),
        local.sign.clone(),
    )
}

struct ConstraintTerminalMainCommon<'a, T> {
    proof_idx: &'a T,
    is_eq_step: &'a T,
    is_lcs_step: &'a T,
    num_rounds: &'a T,
    c_chips: &'a T,
    round_idx: &'a T,
    opening_idx: &'a T,
    chip_idx: &'a T,
    opening_point: &'a [T; D_EF],
    eq_challenge: &'a [T; D_EF],
    eq_factor: &'a [T; D_EF],
    eq_in: &'a [T; D_EF],
    eq_out: &'a [T; D_EF],
    first_prefix_in: &'a [T; D_EF],
    first_prefix_out: &'a [T; D_EF],
    last_prefix_in: &'a [T; D_EF],
    last_prefix_out: &'a [T; D_EF],
    fold_cursor: &'a T,
    alpha: &'a [T; D_EF],
    main_eval: &'a [T; D_EF],
    perm_eval: &'a [T; D_EF],
    last_claim: &'a [T; D_EF],
    lcs: &'a [T; D_EF],
    state_lcs_in: &'a [T; D_EF],
    state_lcs_out: &'a [T; D_EF],
    summary_id_base: &'a T,
}

fn terminal_main_common_wide<T>(
    local: &ConstraintTerminalCols<T>,
) -> ConstraintTerminalMainCommon<'_, T> {
    ConstraintTerminalMainCommon {
        proof_idx: &local.proof_idx,
        is_eq_step: &local.is_eq_step,
        is_lcs_step: &local.is_lcs_step,
        num_rounds: &local.num_rounds,
        c_chips: &local.c_chips,
        round_idx: &local.round_idx,
        opening_idx: &local.opening_idx,
        chip_idx: &local.chip_idx,
        opening_point: &local.opening_point,
        eq_challenge: &local.eq_challenge,
        eq_factor: &local.eq_factor,
        eq_in: &local.eq_in,
        eq_out: &local.eq_out,
        first_prefix_in: &local.first_prefix_in,
        first_prefix_out: &local.first_prefix_out,
        last_prefix_in: &local.last_prefix_in,
        last_prefix_out: &local.last_prefix_out,
        fold_cursor: &local.fold_cursor,
        alpha: &local.alpha,
        main_eval: &local.main_eval,
        perm_eval: &local.perm_eval,
        last_claim: &local.last_claim,
        lcs: &local.lcs,
        state_lcs_in: &local.state_lcs_in,
        state_lcs_out: &local.state_lcs_out,
        summary_id_base: &local.summary_id_base,
    }
}

fn terminal_main_common_narrow<T>(
    local: &ConstraintTerminalColsNarrow<T>,
) -> ConstraintTerminalMainCommon<'_, T> {
    ConstraintTerminalMainCommon {
        proof_idx: &local.proof_idx,
        is_eq_step: &local.is_eq_step,
        is_lcs_step: &local.is_lcs_step,
        num_rounds: &local.num_rounds,
        c_chips: &local.c_chips,
        round_idx: &local.round_idx,
        opening_idx: &local.opening_idx,
        chip_idx: &local.chip_idx,
        opening_point: &local.opening_point,
        eq_challenge: &local.eq_challenge,
        eq_factor: &local.eq_factor,
        eq_in: &local.eq_in,
        eq_out: &local.eq_out,
        first_prefix_in: &local.first_prefix_in,
        first_prefix_out: &local.first_prefix_out,
        last_prefix_in: &local.last_prefix_in,
        last_prefix_out: &local.last_prefix_out,
        fold_cursor: &local.fold_cursor,
        alpha: &local.alpha,
        main_eval: &local.main_eval,
        perm_eval: &local.perm_eval,
        last_claim: &local.last_claim,
        lcs: &local.lcs,
        state_lcs_in: &local.state_lcs_in,
        state_lcs_out: &local.state_lcs_out,
        summary_id_base: &local.summary_id_base,
    }
}

fn terminal_outer_denominators<AB: FullAirBuilder>(
    air: &ConstraintTerminalAir,
    builder: &AB,
    local: &ConstraintTerminalMainCommon<'_, AB::VarMaybeExt>,
) -> ConstraintTerminalOuterDenominators<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    ConstraintTerminalOuterDenominators {
        summary: air.summary_bus.denominator(
            builder,
            proof_idx.clone(),
            local.num_rounds.clone(),
            local.c_chips.clone(),
            const_maybe::<AB>(air.num_public_values),
            local.summary_id_base.clone(),
        ),
        opening_point: air.opening_point_bus.denominator(
            builder,
            proof_idx.clone(),
            local.opening_idx.clone(),
            local.opening_point.clone(),
        ),
        eq: air.sumcheck_out_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(SUMCHECK_OUT_EQ),
            local.opening_idx.clone(),
            local.eq_challenge.clone(),
        ),
        fold_plan_chain: air.fold_plan_chain_bus.denominator(
            builder,
            proof_idx.clone(),
            local.fold_cursor.clone(),
            AB::zero_maybe(),
            AB::zero_maybe(),
        ),
        last_claim: air.sumcheck_claim_chain_bus.denominator(
            builder,
            proof_idx.clone(),
            local.num_rounds.clone(),
            local.num_rounds.clone(),
            local.c_chips.clone(),
            local.last_claim.clone(),
        ),
        fold_chain: air.fold_chain_bus.denominator(
            builder,
            proof_idx.clone(),
            local.fold_cursor.clone(),
            local.alpha.clone(),
            local.main_eval.clone(),
            local.perm_eval.clone(),
            core::array::from_fn(|_| AB::zero_maybe()),
        ),
        eq_chain_recv: air.eq_chain_bus.denominator(
            builder,
            proof_idx.clone(),
            local.round_idx.clone(),
            local.eq_in.clone(),
            local.first_prefix_in.clone(),
            local.last_prefix_in.clone(),
        ),
        eq_chain_send: air.eq_chain_bus.denominator(
            builder,
            proof_idx,
            local.round_idx.clone() + local.is_eq_step.clone(),
            local.eq_out.clone(),
            local.first_prefix_out.clone(),
            local.last_prefix_out.clone(),
        ),
    }
}

fn terminal_lcs_denominator<AB: FullAirBuilder>(
    air: &ConstraintTerminalAir,
    builder: &AB,
    local: &ConstraintTerminalMainCommon<'_, AB::VarMaybeExt>,
) -> ConstraintTerminalLcsDenominator<AB::VarExt> {
    ConstraintTerminalLcsDenominator {
        lcs: air.challenge_bus.denominator(
            builder,
            local.proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_LCS),
            local.chip_idx.clone(),
            AB::zero_maybe(),
            local.lcs.clone(),
        ),
    }
}

fn terminal_packed_common<AB: FullAirBuilder>(
    local: &ConstraintTerminalMainCommon<'_, AB::VarMaybeExt>,
) -> ConstraintTerminalPackedCommonCols<AB::VarExt> {
    ConstraintTerminalPackedCommonCols {
        opening_point: AB::pack_ext_limbs(local.opening_point),
        eq_challenge: AB::pack_ext_limbs(local.eq_challenge),
        eq_factor: AB::pack_ext_limbs(local.eq_factor),
        eq_in: AB::pack_ext_limbs(local.eq_in),
        eq_out: AB::pack_ext_limbs(local.eq_out),
        first_prefix_in: AB::pack_ext_limbs(local.first_prefix_in),
        first_prefix_out: AB::pack_ext_limbs(local.first_prefix_out),
        last_prefix_in: AB::pack_ext_limbs(local.last_prefix_in),
        last_prefix_out: AB::pack_ext_limbs(local.last_prefix_out),
        main_eval: AB::pack_ext_limbs(local.main_eval),
        last_claim_minus_perm_eval: AB::pack_ext_limbs(local.last_claim) -
            AB::pack_ext_limbs(local.perm_eval),
        lcs: AB::pack_ext_limbs(local.lcs),
        state_lcs_in: AB::pack_ext_limbs(local.state_lcs_in),
        state_lcs_out: AB::pack_ext_limbs(local.state_lcs_out),
    }
}

fn terminal_precomputed_narrow<AB: FullAirBuilder>(
    air: &ConstraintTerminalAir,
    builder: &AB,
    local: &ConstraintTerminalColsNarrow<AB::VarMaybeExt>,
) -> ConstraintTerminalPrecomputedNarrowCols<AB::VarExt> {
    let common = terminal_main_common_narrow(local);
    let send_cursor = common.opening_idx.clone() + common.is_lcs_step.clone();
    ConstraintTerminalPrecomputedNarrowCols {
        denominators: ConstraintTerminalDenominatorsNarrowCols {
            outer: terminal_outer_denominators(air, builder, &common),
            lcs: terminal_lcs_denominator(air, builder, &common),
            state_lcs_in: air.challenge_bus.denominator(
                builder,
                common.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_STATE_LCS),
                common.opening_idx.clone(),
                AB::zero_maybe(),
                common.state_lcs_in.clone(),
            ),
            state_lcs_out: air.challenge_bus.denominator(
                builder,
                common.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_STATE_LCS),
                send_cursor,
                AB::zero_maybe(),
                common.state_lcs_out.clone(),
            ),
        },
        common: terminal_packed_common::<AB>(&common),
    }
}

fn terminal_precomputed_wide<AB: FullAirBuilder>(
    air: &ConstraintTerminalAir,
    builder: &AB,
    local: &ConstraintTerminalCols<AB::VarMaybeExt>,
) -> ConstraintTerminalPrecomputedWideCols<AB::VarExt> {
    let common = terminal_main_common_wide(local);
    let send_cursor = common.opening_idx.clone() + common.is_lcs_step.clone();
    let denominators = ConstraintTerminalDenominatorsWideCols {
        outer: terminal_outer_denominators(air, builder, &common),
        public_values: core::array::from_fn(|index| {
            air.proof_values_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES),
                const_maybe::<AB>(TERMINAL_PV_IDXS[index]),
                local.public_values[index].clone(),
            )
        }),
        perm_alpha: air.challenge_bus.denominator(
            builder,
            local.proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_PERM_ALPHA),
            AB::zero_maybe(),
            AB::zero_maybe(),
            local.perm_alpha.clone(),
        ),
        beta_powers: core::array::from_fn(|power| {
            air.challenge_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_BETA_POWER),
                const_maybe::<AB>(power + 1),
                AB::zero_maybe(),
                local.beta_powers[power].clone(),
            )
        }),
        lcs: terminal_lcs_denominator(air, builder, &common),
        state_lcs_in: air.challenge_bus.denominator(
            builder,
            local.proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_STATE_LCS),
            local.opening_idx.clone(),
            AB::zero_maybe(),
            local.state_lcs_in.clone(),
        ),
        state_lcs_out: air.challenge_bus.denominator(
            builder,
            local.proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_STATE_LCS),
            send_cursor.clone(),
            AB::zero_maybe(),
            local.state_lcs_out.clone(),
        ),
    };
    ConstraintTerminalPrecomputedWideCols {
        denominators,
        common: terminal_packed_common::<AB>(&common),
        state: ConstraintTerminalPackedStateCols {
            perm_alpha: AB::pack_ext_limbs(&local.perm_alpha),
            beta_powers: core::array::from_fn(|index| {
                AB::pack_ext_limbs(&local.beta_powers[index])
            }),
            state_transition_recv_inverse: AB::pack_ext_limbs(&local.state_transition_recv_inverse),
            state_transition_send_inverse: AB::pack_ext_limbs(&local.state_transition_send_inverse),
            init_address_recv_inverse: AB::pack_ext_limbs(&local.init_address_recv_inverse),
            init_address_send_inverse: AB::pack_ext_limbs(&local.init_address_send_inverse),
            finalize_address_recv_inverse: AB::pack_ext_limbs(&local.finalize_address_recv_inverse),
            finalize_address_send_inverse: AB::pack_ext_limbs(&local.finalize_address_send_inverse),
            global_chain_source_inverse: AB::pack_ext_limbs(&local.global_chain_source_inverse),
            global_chain_sink_inverse: AB::pack_ext_limbs(&local.global_chain_sink_inverse),
        },
    }
}

fn constrain_terminal_eq<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
    packed: &ConstraintTerminalPackedCommonCols<AB::VarExt>,
) {
    let z = packed.opening_point.clone();
    let eq_challenge = packed.eq_challenge.clone();
    let one = AB::pack_ext_limbs(&[AB::one_maybe()]);
    let qz = eq_challenge.clone() * z.clone();
    let factor = qz.clone() + qz - eq_challenge - z.clone() + one.clone();
    let eq_expected = packed.eq_in.clone() * packed.eq_factor.clone();
    let first_expected = packed.first_prefix_in.clone() * (one.clone() - z.clone());
    let last_expected = packed.last_prefix_in.clone() * z;

    builder.assert_zero_ext((packed.eq_out.clone() - one.clone()) * local.is_seed.clone());
    builder
        .assert_zero_ext((packed.first_prefix_out.clone() - one.clone()) * local.is_seed.clone());
    builder.assert_zero_ext((packed.last_prefix_out.clone() - one) * local.is_seed.clone());
    builder.assert_zero_ext((packed.eq_factor.clone() - factor) * local.is_eq_step.clone());
    builder.assert_zero_ext((packed.eq_out.clone() - eq_expected) * local.is_eq_step.clone());
    builder.assert_zero_ext(
        (packed.first_prefix_out.clone() - first_expected) * local.is_eq_step.clone(),
    );
    builder.assert_zero_ext(
        (packed.last_prefix_out.clone() - last_expected) * local.is_eq_step.clone(),
    );
}

fn constrain_terminal_final<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
    packed: &ConstraintTerminalPackedCommonCols<AB::VarExt>,
) {
    builder.assert_zero_ext(
        (packed.last_claim_minus_perm_eval.clone() -
            packed.main_eval.clone() * packed.eq_in.clone()) *
            local.is_final.clone(),
    );
}

/// Reduce-role state check: local-sum rows accumulate and the final row pins the sum to zero.
/// Note: the seed anchor is load-bearing — without it the seed payload is
/// free witness and a malicious prover could offset a nonzero lcs total.
fn constrain_terminal_state_narrow<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedCommon<'_, AB::VarMaybeExt>,
    packed: &ConstraintTerminalPackedCommonCols<AB::VarExt>,
    child_contains_global_bus: bool,
) {
    builder.assert_zero_ext(packed.state_lcs_out.clone() * local.is_seed.clone());
    builder.assert_zero_ext(
        (packed.state_lcs_out.clone() - packed.state_lcs_in.clone() - packed.lcs.clone()) *
            local.is_lcs_step.clone(),
    );
    if child_contains_global_bus {
        builder.assert_zero_ext(
            (packed.state_lcs_out.clone() - packed.state_lcs_in.clone()) *
                local.is_final.clone(),
        );
    } else {
        builder.assert_zero_ext(packed.state_lcs_in.clone() * local.is_final.clone());
    }
}

fn constrain_terminal_state<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintTerminalReservedWideCols<AB::VarMaybeExt>,
    common: &ConstraintTerminalPackedCommonCols<AB::VarExt>,
    packed: &ConstraintTerminalPackedStateCols<AB::VarExt>,
) {
    builder.assert_zero_ext(
        (common.state_lcs_out.clone() - common.state_lcs_in.clone() - common.lcs.clone()) *
            local.is_lcs_step.clone(),
    );
    builder.assert_zero_ext(
        (common.state_lcs_out.clone() - common.state_lcs_in.clone()) * local.is_final.clone(),
    );
    constrain_state_imbalance(
        builder,
        local.is_final.clone(),
        &local.public_values,
        local.state_clock_changed.clone(),
        local.state_clock_delta_inverse.clone(),
        common.state_lcs_in.clone(),
        packed,
    );
}

fn constrain_state_imbalance<AB: FullAirBuilder>(
    builder: &mut AB,
    is_valid: AB::VarMaybeExt,
    public_values: &[AB::VarMaybeExt; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    state_clock_changed: AB::VarMaybeExt,
    state_clock_delta_inverse: AB::VarMaybeExt,
    state_lcs: AB::VarExt,
    packed: &ConstraintTerminalPackedStateCols<AB::VarExt>,
) {
    // The inner lookup challenges are sampled after the inner main commitment.
    // Each fingerprint below is affine in that committed-after challenge, so
    // forcing the four boundary inverses and two count-gated chain inverses
    // adds at most 6/|EF| honest proof failure probability.
    let clock_delta = public_values[PV_COL_START_CLK].clone() -
        public_values[PV_COL_EXIT_CLK].clone();
    builder.assert_zero(
        clock_delta.clone() * state_clock_delta_inverse - state_clock_changed.clone(),
    );
    builder.assert_zero(clock_delta * (AB::one_maybe() - state_clock_changed.clone()));

    let alpha = packed.perm_alpha.clone();
    let beta = packed.beta_powers[0].clone();
    let beta2 = packed.beta_powers[1].clone();
    let beta3 = packed.beta_powers[2].clone();
    let state_kind = AB::pack_ext_limbs(&[const_maybe::<AB>(InteractionKind::State as usize)]);
    let addr_kind =
        AB::pack_ext_limbs(&[const_maybe::<AB>(InteractionKind::MemoryGlobalAddr as usize)]);
    let shard_term = beta.clone() * public_values[PV_COL_EXECUTION_SHARD].clone();
    let recv_state = alpha.clone() +
        state_kind.clone() +
        shard_term.clone() +
        beta2.clone() * public_values[PV_COL_START_CLK].clone() +
        beta3.clone() * public_values[PV_COL_START_PC].clone();
    let send_state = alpha.clone() +
        state_kind +
        shard_term +
        beta2.clone() * public_values[PV_COL_EXIT_CLK].clone() +
        beta3 * public_values[PV_COL_NEXT_PC].clone();
    let base_init = alpha.clone() + addr_kind.clone();
    let recv_init =
        base_init.clone() + beta2.clone() * public_values[PV_COL_PREVIOUS_INIT_ADDR].clone();
    let send_init = base_init + beta2.clone() * public_values[PV_COL_LAST_INIT_ADDR].clone();
    let base_fin = alpha.clone() + addr_kind + beta;
    let recv_fin = base_fin.clone() +
        beta2.clone() * public_values[PV_COL_PREVIOUS_FINALIZE_ADDR].clone();
    let send_fin =
        base_fin + beta2.clone() * public_values[PV_COL_LAST_FINALIZE_ADDR].clone();
    let clock_changed = AB::pack_ext_limbs(&[state_clock_changed.clone()]);
    let is_final = AB::pack_ext_limbs(&[is_valid.clone()]);
    for (fingerprint, inverse) in [
        (recv_state, packed.state_transition_recv_inverse.clone()),
        (send_state, packed.state_transition_send_inverse.clone()),
    ] {
        builder.assert_zero_ext(fingerprint * inverse - clock_changed.clone());
    }
    for (fingerprint, inverse) in [
        (recv_init, packed.init_address_recv_inverse.clone()),
        (send_init, packed.init_address_send_inverse.clone()),
        (recv_fin, packed.finalize_address_recv_inverse.clone()),
        (send_fin, packed.finalize_address_send_inverse.clone()),
    ] {
        builder.assert_zero_ext(fingerprint * inverse - is_final.clone());
    }
    let has = public_values[PV_COL_GLOBAL_HAS].clone();
    assert_bool(builder, has.clone());
    let global_kind =
        AB::pack_ext_limbs(&[const_maybe::<AB>(InteractionKind::GlobalProjectiveChainV2 as usize)]);
    let source_blocks: [AB::VarExt; CONSTRAINT_GLOBAL_CHAIN_BLOCKS] =
        core::array::from_fn(|block| {
            let payload: [AB::VarMaybeExt; D_EF] = core::array::from_fn(|limb| {
                let offset = block * D_EF + limb;
                if offset == 0 {
                    AB::zero_maybe()
                } else if offset <= 33 {
                    public_values[PV_COL_GLOBAL_START + offset - 1].clone()
                } else {
                    AB::zero_maybe()
                }
            });
            AB::pack_ext_limbs(&payload)
        });
    let sink_blocks: [AB::VarExt; CONSTRAINT_GLOBAL_CHAIN_BLOCKS] = core::array::from_fn(|block| {
        let payload: [AB::VarMaybeExt; D_EF] = core::array::from_fn(|limb| {
            let offset = block * D_EF + limb;
            if offset == 0 {
                public_values[PV_COL_GLOBAL_COUNT].clone()
            } else if offset <= 33 {
                public_values[PV_COL_GLOBAL_END + offset - 1].clone()
            } else {
                AB::zero_maybe()
            }
        });
        AB::pack_ext_limbs(&payload)
    });
    let source = source_blocks.iter().enumerate().fold(
        alpha.clone() + global_kind.clone(),
        |fingerprint, (index, block)| {
            fingerprint + packed.beta_powers[index].clone() * block.clone()
        },
    );
    let sink =
        sink_blocks.iter().enumerate().fold(alpha + global_kind, |fingerprint, (index, block)| {
            fingerprint + packed.beta_powers[index].clone() * block.clone()
        });
    let active_global = is_final.clone() * has.clone();
    builder.assert_zero_ext(
        source * packed.global_chain_source_inverse.clone() - active_global.clone(),
    );
    builder
        .assert_zero_ext(sink * packed.global_chain_sink_inverse.clone() - active_global.clone());
    builder.assert_zero_ext(
        packed.global_chain_source_inverse.clone() * (AB::one_maybe() - has.clone()),
    );
    builder.assert_zero_ext(
        packed.global_chain_sink_inverse.clone() * (AB::one_maybe() - has.clone()),
    );

    let contribution = (packed.state_transition_send_inverse.clone() -
        packed.state_transition_recv_inverse.clone()) *
        state_clock_changed +
        packed.init_address_send_inverse.clone() -
        packed.init_address_recv_inverse.clone() +
        packed.finalize_address_send_inverse.clone() -
        packed.finalize_address_recv_inverse.clone() +
        (packed.global_chain_sink_inverse.clone() - packed.global_chain_source_inverse.clone()) *
            has;
    builder.assert_zero_ext((state_lcs - contribution) * is_valid);
}

fn dag_denominators<AB: FullAirBuilder>(
    air: &ConstraintDagEvalAir,
    builder: &AB,
    local: &ConstraintDagEvalCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    vec![
        air.program_bus.denominator(
            builder,
            local.static_chip_id.clone(),
            local.node_idx.clone(),
            dag_op_code_expr::<AB>(local),
            local.lhs_idx.clone(),
            local.rhs_idx.clone(),
            local.third_idx.clone(),
            local.aux.clone(),
            dag_leaf_kind_expr::<AB>(local),
            local.fanout.clone(),
        ),
        air.node_bus.denominator(
            builder,
            proof_idx.clone(),
            local.chip_idx.clone(),
            local.static_chip_id.clone(),
            local.node_idx.clone(),
            local.value.clone(),
        ),
        air.node_bus.denominator(
            builder,
            proof_idx.clone(),
            local.chip_idx.clone(),
            local.static_chip_id.clone(),
            local.lhs_idx.clone(),
            local.lhs_value.clone(),
        ),
        air.node_bus.denominator(
            builder,
            proof_idx.clone(),
            local.chip_idx.clone(),
            local.static_chip_id.clone(),
            local.rhs_idx.clone(),
            local.rhs_value.clone(),
        ),
        air.node_bus.denominator(
            builder,
            proof_idx.clone(),
            local.chip_idx.clone(),
            local.static_chip_id.clone(),
            local.third_idx.clone(),
            local.third_value.clone(),
        ),
        air.opened_eval_bus.denominator(
            builder,
            proof_idx.clone(),
            local.lhs_idx.clone(),
            local.opened_batch_pos.clone(),
            local.chip_idx.clone(),
            local.rhs_idx.clone(),
            local.value[0].clone(),
            local.value[1].clone(),
            local.value[2].clone(),
            local.value[3].clone(),
            local.value[4].clone(),
        ),
        air.proof_values_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES),
            local.lhs_idx.clone(),
            local.value[0].clone(),
        ),
        air.challenge_bus.denominator(
            builder,
            proof_idx,
            local.rhs_idx.clone(),
            local.third_idx.clone(),
            AB::zero_maybe(),
            local.value.clone(),
        ),
    ]
}

fn fold_is_valid<AB: FullAirBuilder>(
    local: &ConstraintFoldReservedCols<AB::VarMaybeExt>,
) -> AB::VarMaybeExt {
    local.is_skip.clone() + local.is_gate.clone() + local.is_batch.clone()
}

fn fold_node_multiplicities<AB: FullAirBuilder>(
    local: &ConstraintFoldReservedCols<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; CONSTRAINT_FOLD_ROOT_SLOTS] {
    [
        local.is_gate.clone() + local.is_batch.clone(),
        local.batch_has_second.clone(),
        local.is_batch.clone(),
        local.batch_has_second.clone(),
    ]
}

fn fold_root_kind<AB: FullAirBuilder>(
    local: &ConstraintFoldCols<AB::VarMaybeExt>,
    slot: usize,
) -> AB::VarMaybeExt {
    match slot {
        0 => AB::mul_base(
            local.is_batch.clone(),
            AB::F::from_canonical_usize(CONSTRAINT_ROOT_PRECOMPUTE_DENOM),
        ),
        1 => const_maybe::<AB>(CONSTRAINT_ROOT_PRECOMPUTE_DENOM),
        2 | 3 => const_maybe::<AB>(CONSTRAINT_ROOT_MULTIPLICITY),
        _ => unreachable!("ConstraintFold has four root slots"),
    }
}

fn fold_root_ord<AB: FullAirBuilder>(
    local: &ConstraintFoldCols<AB::VarMaybeExt>,
    slot: usize,
) -> AB::VarMaybeExt {
    local.root_ord.clone() +
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(usize::from(slot % 2 == 1)))
}

fn fold_root_sign<AB: FullAirBuilder>(
    local: &ConstraintFoldCols<AB::VarMaybeExt>,
    slot: usize,
) -> AB::VarMaybeExt {
    match slot {
        0 | 1 => AB::one_maybe(),
        2 | 3 => local.multiplicity_signs[slot - CONSTRAINT_FOLD_BATCH_SIZE].clone(),
        _ => unreachable!("ConstraintFold has four root slots"),
    }
}

fn fold_precomputed<AB: FullAirBuilder>(
    air: &ConstraintFoldAir,
    builder: &AB,
    local: &ConstraintFoldCols<AB::VarMaybeExt>,
) -> ConstraintFoldPrecomputedCols<AB::VarExt> {
    let chip_idx = local.remaining_chips.clone() - AB::one_maybe();
    let root_table = core::array::from_fn(|slot| {
        air.root_bus.denominator(
            builder,
            local.static_chip_id.clone(),
            fold_root_kind::<AB>(local, slot),
            fold_root_ord::<AB>(local, slot),
            local.root_nodes[slot].clone(),
            fold_root_sign::<AB>(local, slot),
        )
    });
    let node_value = core::array::from_fn(|slot| {
        air.node_bus.denominator(
            builder,
            local.proof_idx.clone(),
            chip_idx.clone(),
            local.static_chip_id.clone(),
            local.root_nodes[slot].clone(),
            local.root_values[slot].clone(),
        )
    });
    let is_valid = local.is_skip.clone() + local.is_gate.clone() + local.is_batch.clone();
    let previous_cursor = local.cursor.clone() - is_valid.clone();
    let batch_ord = AB::mul_base(local.root_ord.clone(), AB::F::two().inverse());
    ConstraintFoldPrecomputedCols {
        denominators: ConstraintFoldDenominatorCols {
            root_table,
            node_value,
            permutation: air.opened_eval_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(PROOF_SHAPE_BATCH_PERMUTATION),
                chip_idx.clone(),
                chip_idx.clone(),
                batch_ord,
                local.perm_value[0].clone(),
                local.perm_value[1].clone(),
                local.perm_value[2].clone(),
                local.perm_value[3].clone(),
                local.perm_value[4].clone(),
            ),
            lcs: air.challenge_bus.denominator(
                builder,
                local.proof_idx.clone(),
                const_maybe::<AB>(CONSTRAINT_CHALLENGE_LCS),
                chip_idx.clone(),
                AB::zero_maybe(),
                local.perm_value.clone(),
            ),
            height_inverse: air.height_bus.denominator(
                builder,
                local.root_ord.clone(),
                local.root_nodes[0].clone(),
            ),
            chip_meta: air.chip_meta_bus.denominator(
                builder,
                local.proof_idx.clone(),
                chip_idx,
                local.static_chip_id.clone(),
                local.log_height.clone(),
                local.gate_count.clone(),
                local.batch_count.clone(),
            ),
            plan_chain_recv: air.plan_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                previous_cursor.clone(),
                local.remaining_chips.clone(),
                local.local_ord.clone(),
            ),
            plan_chain_send: air.plan_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.cursor.clone(),
                local.remaining_chips.clone() - local.is_skip.clone(),
                local.chain_send_local_ord.clone(),
            ),
            fold_chain_recv: air.fold_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                previous_cursor,
                local.alpha.clone(),
                local.acc_in.clone(),
                local.pacc_in.clone(),
                local.perm_sum_in.clone(),
            ),
            fold_chain_send: air.fold_chain_bus.denominator(
                builder,
                local.proof_idx.clone(),
                local.cursor.clone(),
                local.alpha.clone(),
                local.acc_out.clone(),
                local.pacc_out.clone(),
                local.perm_sum_out.clone(),
            ),
        },
        packed: ConstraintFoldPackedCols {
            alpha: AB::pack_ext_limbs(&local.alpha),
            acc_in: AB::pack_ext_limbs(&local.acc_in),
            acc_out: AB::pack_ext_limbs(&local.acc_out),
            pacc_in: AB::pack_ext_limbs(&local.pacc_in),
            pacc_out: AB::pack_ext_limbs(&local.pacc_out),
            perm_delta: AB::pack_ext_limbs(&core::array::from_fn::<_, D_EF, _>(|limb| {
                local.perm_sum_out[limb].clone() - local.perm_sum_in[limb].clone()
            })),
            perm_sum_out: AB::pack_ext_limbs(&local.perm_sum_out),
            root_values: core::array::from_fn(|slot| AB::pack_ext_limbs(&local.root_values[slot])),
            perm_value: AB::pack_ext_limbs(&local.perm_value),
            gate_position: AB::pack_ext_limbs(&[local.local_ord.clone() - local.root_ord.clone()]),
            batch_position: AB::pack_ext_limbs(&[AB::mul_base(
                local.local_ord.clone() - local.gate_count.clone(),
                AB::F::two(),
            ) - local.root_ord.clone()]),
            skip_position: AB::pack_ext_limbs(&[local.local_ord.clone() -
                local.gate_count.clone() -
                local.batch_count.clone()]),
            skip_height: AB::pack_ext_limbs(&[local.root_ord.clone() - local.log_height.clone()]),
            non_skip_successor: AB::pack_ext_limbs(&[local.chain_send_local_ord.clone() -
                local.local_ord.clone() -
                AB::one_maybe()]),
            skip_successor: AB::pack_ext_limbs(&[local.chain_send_local_ord.clone()]),
        },
    }
}

fn constrain_fold_value<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintFoldReservedCols<AB::VarMaybeExt>,
    packed: &ConstraintFoldPackedCols<AB::VarExt>,
) {
    let root_mults = fold_node_multiplicities::<AB>(local);
    let one = AB::pack_ext_limbs(&[AB::one_maybe()]);
    for slot in 0..CONSTRAINT_FOLD_BATCH_SIZE {
        let denominator_inactive = AB::one_maybe() -
            root_mults[slot].clone() -
            if slot == 1 { local.is_skip.clone() } else { AB::zero_maybe() };
        builder.assert_zero_ext(
            (packed.root_values[slot].clone() - one.clone()) * denominator_inactive,
        );
        builder.assert_zero_ext(
            packed.root_values[CONSTRAINT_FOLD_BATCH_SIZE + slot].clone() *
                (AB::one_maybe() - root_mults[CONSTRAINT_FOLD_BATCH_SIZE + slot].clone()),
        );
    }
    builder.assert_zero_ext(packed.perm_value.clone() * local.is_gate.clone());
    builder.assert_zero_ext(
        (packed.perm_value.clone() - packed.root_values[1].clone() * local.batch_count.clone()) *
            local.is_skip.clone(),
    );

    let d0 = packed.root_values[0].clone();
    let d1 = packed.root_values[1].clone();
    let m0 = packed.root_values[CONSTRAINT_FOLD_BATCH_SIZE].clone() *
        local.multiplicity_signs[0].clone();
    let m1 = packed.root_values[CONSTRAINT_FOLD_BATCH_SIZE + 1].clone() *
        local.multiplicity_signs[1].clone();
    // Exact PolyAir batch-two residual. The tail default d1=1,m1=0
    // specializes it to m0-d0*p without a branch. Skip rows borrow d1 for
    // LCS/batch_count; their pinned d0=1, m0=m1=0 state is cancelled by the
    // degree-three term below so the residual stays zero without raising the
    // shipped Fold AIR above degree three.
    let batch_value = d1.clone() * m0 +
        d0.clone() * (m1 - d1.clone() * packed.perm_value.clone()) +
        d1 * packed.perm_value.clone() * local.is_skip.clone();
    let gate_value = d0 * local.is_gate.clone();
    let value = gate_value + batch_value;
    builder.assert_zero_ext(
        packed.acc_out.clone() - (packed.acc_in.clone() * packed.alpha.clone() + value),
    );
}

fn constrain_fold_chain<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintFoldReservedCols<AB::VarMaybeExt>,
    packed: &ConstraintFoldPackedCols<AB::VarExt>,
) {
    builder.assert_zero_ext(packed.perm_delta.clone() * local.is_gate.clone());
    builder.assert_zero_ext(
        (packed.perm_delta.clone() - packed.perm_value.clone()) * local.is_batch.clone(),
    );
    builder.assert_zero_ext(packed.perm_sum_out.clone() * local.is_skip.clone());
    let skip_correction = AB::pack_ext_limbs(&[AB::zero_maybe()]) -
        packed.perm_delta.clone() -
        packed.perm_value.clone() * local.height_inverse.clone();
    builder.assert_zero_ext(
        packed.pacc_out.clone() -
            (packed.pacc_in.clone() * packed.alpha.clone() +
                skip_correction * local.is_skip.clone()),
    );
}

fn challenge_precomputed<AB: FullAirBuilder>(
    air: &ConstraintChallengeAir,
    builder: &AB,
    local: &ConstraintChallengeCols<AB::VarMaybeExt>,
) -> ConstraintChallengePrecomputedCols<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    let lcs_events = core::array::from_fn(|limb| {
        air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            lcs_tidx::<AB>(
                air.seed_prefix_limbs,
                air.num_public_values,
                local.c_chips.clone(),
                local.chip_idx.clone(),
                limb,
            ),
            AB::zero_maybe(),
            local.lcs_limbs[limb].clone(),
        )
    });
    let eq_chain = air.eq_chain_bus.denominator(
        builder,
        proof_idx.clone(),
        local.log_height.clone(),
        local.selector_eq_acc.clone(),
        local.selector_first.clone(),
        local.selector_last.clone(),
    );
    let lcs = air.challenge_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(CONSTRAINT_CHALLENGE_LCS),
        local.chip_idx.clone(),
        AB::zero_maybe(),
        local.lcs_limbs.clone(),
    );
    let is_first = air.challenge_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(CONSTRAINT_CHALLENGE_IS_FIRST),
        local.static_chip_id.clone(),
        AB::zero_maybe(),
        local.selector_first.clone(),
    );
    let is_last = air.challenge_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(CONSTRAINT_CHALLENGE_IS_LAST),
        local.static_chip_id.clone(),
        AB::zero_maybe(),
        local.selector_last.clone(),
    );
    let batch_dim_main = air.batch_dim_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(PROOF_SHAPE_BATCH_MAIN),
        local.chip_idx.clone(),
        local.chip_idx.clone(),
        local.static_chip_id.clone(),
        local.main_width.clone(),
        local.log_height.clone(),
    );
    let fold_plan_source = air.fold_plan_chain_bus.denominator(
        builder,
        proof_idx,
        AB::zero_maybe(),
        local.c_chips.clone(),
        AB::zero_maybe(),
    );
    ConstraintChallengePrecomputedCols {
        denominators: ConstraintChallengeDenominatorCols {
            lcs_events,
            eq_chain,
            lcs,
            is_first,
            is_last,
            batch_dim_main,
            fold_plan_source,
        },
    }
}

fn beta_ladder_denominators<AB: FullAirBuilder>(
    air: &ConstraintBetaLadderAir,
    builder: &AB,
    local: &ConstraintBetaLadderCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    let septix_value = beta_septix_value::<AB>(local.power.clone(), local.beta.clone());
    vec![
        air.sumcheck_out_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(SUMCHECK_OUT_PERM_ALPHA),
            AB::zero_maybe(),
            local.prev_power_or_alpha.clone(),
        ),
        air.sumcheck_out_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(SUMCHECK_OUT_PERM_BETA),
            AB::zero_maybe(),
            local.beta.clone(),
        ),
        air.ladder_bus.denominator(
            builder,
            proof_idx.clone(),
            local.power_idx.clone() - AB::one_maybe(),
            local.prev_power_or_alpha.clone(),
            local.beta.clone(),
        ),
        air.ladder_bus.denominator(
            builder,
            proof_idx.clone(),
            local.power_idx.clone(),
            local.power.clone(),
            local.beta.clone(),
        ),
        air.challenge_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_BETA_POWER),
            local.power_idx.clone(),
            AB::zero_maybe(),
            local.power.clone(),
        ),
        air.challenge_bus.denominator(
            builder,
            proof_idx.clone(),
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_PERM_ALPHA),
            AB::zero_maybe(),
            AB::zero_maybe(),
            local.prev_power_or_alpha.clone(),
        ),
        air.challenge_bus.denominator(
            builder,
            proof_idx,
            const_maybe::<AB>(CONSTRAINT_CHALLENGE_BETA_SEPTIX),
            AB::zero_maybe(),
            AB::zero_maybe(),
            septix_value,
        ),
    ]
}

fn constrain_node_value<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintDagEvalReservedCols<AB::VarMaybeExt>,
    packed: &[AB::VarExt],
) {
    builder.assert_zero(
        local.is_fused.clone() * local.aux.clone() * (AB::one_maybe() - local.aux.clone()),
    );

    let value = packed[0].clone();
    let lhs = packed[1].clone();
    let rhs = packed[2].clone();
    let third = packed[3].clone();
    let product = lhs.clone() * rhs.clone();
    let aux_third = third.clone() * local.aux.clone();
    let packed_const = AB::pack_ext_limbs(&[local.aux.clone(), local.lhs_idx.clone()]);

    let residual = (value.clone() - lhs.clone() - rhs.clone()) * local.is_add.clone() +
        (value.clone() - lhs.clone() + rhs) * local.is_sub.clone() +
        (value.clone() - product.clone()) * local.is_mul.clone() +
        (value.clone() - product - third + aux_third.clone() + aux_third) *
            local.is_fused.clone() +
        (value.clone() - packed_const) * local.is_const.clone() +
        (value + (-local.value_0.clone())) * local.leaf_flags[CONSTRAINT_LEAF_PUBLIC].clone();
    builder.assert_zero_ext(residual);
}

fn constrain_opened_source<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintDagEvalReservedCols<AB::VarMaybeExt>,
    opened_mult: AB::VarMaybeExt,
) {
    let main = local.leaf_flags[CONSTRAINT_LEAF_MAIN].clone();
    let reserved = local.leaf_flags[CONSTRAINT_LEAF_RESERVED_POLY].clone();
    let reserved_is_main = reserved.clone() * local.lhs_idx.clone();
    builder.assert_zero(
        (main + reserved_is_main) * (local.opened_batch_pos.clone() - local.chip_idx.clone()),
    );
    builder.assert_zero((AB::one_maybe() - opened_mult) * local.opened_batch_pos.clone());
}

fn beta_septix_value<AB: FullAirBuilder>(
    power: [AB::VarMaybeExt; D_EF],
    beta: [AB::VarMaybeExt; D_EF],
) -> [AB::VarMaybeExt; D_EF] {
    core::array::from_fn(|limb| {
        let constant = if limb == 0 { const_maybe::<AB>(5) } else { AB::zero_maybe() };
        power[limb].clone() - beta[limb].clone() * const_maybe::<AB>(3) - constant
    })
}

fn constrain_beta_ladder_row<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintBetaLadderCols<AB::VarMaybeExt>,
) {
    assert_bool(builder, local.is_valid.clone());
    assert_bool(builder, local.is_seed.clone());
    assert_bool(builder, local.is_last.clone());
    builder.assert_zero(local.is_seed.clone() * (AB::one_maybe() - local.is_valid.clone()));
    builder.assert_zero(local.is_last.clone() * (AB::one_maybe() - local.is_valid.clone()));
    builder.assert_zero(local.is_seed.clone() * local.power_idx.clone());
    for limb in 0..D_EF {
        let seed = if limb == 0 { AB::one_maybe() } else { AB::zero_maybe() };
        builder.assert_zero(local.is_seed.clone() * (local.power[limb].clone() - seed));
    }
    builder.assert_zero(
        local.is_last.clone() *
            (local.power_idx.clone() - const_maybe::<AB>(CONSTRAINT_MAX_BETA_POWERS - 1)),
    );

    let chain_mult = local.is_valid.clone() - local.is_seed.clone();
    let expected_power = ChallengeExtension(local.prev_power_or_alpha.clone()) *
        ChallengeExtension(local.beta.clone());
    for limb in 0..D_EF {
        builder.assert_zero(
            chain_mult.clone() * (local.power[limb].clone() - expected_power.0[limb].clone()),
        );
    }
    builder.assert_zero(local.serve_mult.clone() * (AB::one_maybe() - local.is_valid.clone()));
    assert_bool(builder, local.challenges_recv_mult.clone());
    builder.assert_zero(
        local.challenges_recv_mult.clone() * (AB::one_maybe() - local.is_seed.clone()),
    );
    builder.assert_zero(local.alpha_serve_mult.clone() * (AB::one_maybe() - local.is_seed.clone()));
    builder.assert_zero(
        local.septix_serve_mult.clone() * (local.power_idx.clone() - const_maybe::<AB>(7)),
    );
    builder
        .assert_zero(local.septix_serve_mult.clone() * (AB::one_maybe() - local.is_valid.clone()));
}

fn constrain_selector_row<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &ConstraintChallengeReservedCols<AB::VarMaybeExt>,
) {
    builder.assert_zero(
        local.selector_first_send_mult.clone() * (AB::one_maybe() - local.is_valid.clone()),
    );
    builder.assert_zero(
        local.selector_last_send_mult.clone() * (AB::one_maybe() - local.is_valid.clone()),
    );
}

fn lcs_tidx<AB: FullAirBuilder>(
    seed_prefix_limbs: usize,
    num_public_values: usize,
    c_chips: AB::VarMaybeExt,
    chip_idx: AB::VarMaybeExt,
    limb: usize,
) -> AB::VarMaybeExt {
    const_maybe::<AB>(
        seed_prefix_limbs +
            num_public_values +
            crate::batch_constraint_dt::columns::BATCH_COMMITMENT_LIMBS +
            crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_HEADER_LIMBS,
    ) + c_chips *
        const_maybe::<AB>(crate::batch_constraint_dt::columns::BATCH_ACTIVE_SHAPE_ENTRY_LIMBS) +
        const_maybe::<AB>(
            crate::batch_constraint_dt::columns::BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS,
        ) +
        chip_idx * const_maybe::<AB>(D_EF) +
        const_maybe::<AB>(limb)
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}
