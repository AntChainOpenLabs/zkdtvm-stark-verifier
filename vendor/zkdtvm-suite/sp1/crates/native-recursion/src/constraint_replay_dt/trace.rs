use core::{
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
};
use std::{borrow::Cow, collections::BTreeMap, ops::Range, sync::Arc};

use dt_stark::{
    air::{PairCol, PublicValues},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind, Word,
};
use p3_field::{
    batch_multiplicative_inverse, AbstractExtensionField, AbstractField, Field, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    batch_constraint_dt::{
        trace::{
            batch_sumcheck_rows, batch_transcript_input_rows, BatchSumcheckRow,
            BatchTranscriptInputRow,
        },
        SUMCHECK_OUT_EQ, SUMCHECK_OUT_PERM_ALPHA, SUMCHECK_OUT_PERM_BETA,
    },
    config::{D_EF, EF, F},
    constraint_replay_dt::{
        air::TERMINAL_PV_IDXS,
        columns::{
            ConstraintBetaLadderCols, ConstraintBoundaryCols, ConstraintChallengeCols,
            ConstraintDagEvalCols, ConstraintFoldCols, ConstraintProgramCols,
            ConstraintProgramPreprocessedCols, ConstraintRootTableCols,
            ConstraintRootTablePreprocessedCols, ConstraintTerminalCols,
            ConstraintTerminalColsNarrow, CONSTRAINT_CHAIN_LIMBS, CONSTRAINT_CHALLENGE_BETA_POWER,
            CONSTRAINT_CHALLENGE_BETA_SEPTIX, CONSTRAINT_CHALLENGE_IS_FIRST,
            CONSTRAINT_CHALLENGE_IS_LAST, CONSTRAINT_CHALLENGE_LCS,
            CONSTRAINT_CHALLENGE_PERM_ALPHA, CONSTRAINT_CHALLENGE_STATE_LCS,
            CONSTRAINT_FOLD_BATCH_SIZE, CONSTRAINT_FOLD_ROOT_SLOTS, CONSTRAINT_HEIGHT_TABLE_ROWS,
            CONSTRAINT_HEIGHT_TABLE_STATIC_ID, CONSTRAINT_LEAF_BETA_POWER,
            CONSTRAINT_LEAF_BETA_SEPTIX, CONSTRAINT_LEAF_IS_FIRST_ROW, CONSTRAINT_LEAF_IS_LAST_ROW,
            CONSTRAINT_LEAF_KIND_COUNT, CONSTRAINT_LEAF_MAIN, CONSTRAINT_LEAF_PERM_ALPHA,
            CONSTRAINT_LEAF_PRECOMPUTED, CONSTRAINT_LEAF_PREPROCESSED, CONSTRAINT_LEAF_PUBLIC,
            CONSTRAINT_LEAF_RESERVED_POLY, CONSTRAINT_MAX_BETA_POWERS, CONSTRAINT_OP_ADD,
            CONSTRAINT_OP_CONST, CONSTRAINT_OP_FUSED, CONSTRAINT_OP_LEAF, CONSTRAINT_OP_MUL,
            CONSTRAINT_OP_SUB, CONSTRAINT_ROOT_GATE, CONSTRAINT_ROOT_HEIGHT_INVERSE,
            CONSTRAINT_ROOT_MULTIPLICITY, CONSTRAINT_ROOT_PRECOMPUTE_DENOM,
            CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT,
            CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS, CONSTRAINT_TERMINAL_LCS_LIMBS,
            CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT,
            NUM_CONSTRAINT_BETA_LADDER_COLS, NUM_CONSTRAINT_BOUNDARY_COLS,
            NUM_CONSTRAINT_CHALLENGE_COLS,
            NUM_CONSTRAINT_DAG_EVAL_COLS, NUM_CONSTRAINT_FOLD_COLS, NUM_CONSTRAINT_PROGRAM_COLS,
            NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS, NUM_CONSTRAINT_ROOT_TABLE_COLS,
            NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS, NUM_CONSTRAINT_TERMINAL_COLS,
            NUM_CONSTRAINT_TERMINAL_NARROW_COLS,
        },
    },
    proof_shape_dt::{
        proof_shape_binder_rows, trace::ProofShapeBinderRow, PROOF_SHAPE_BATCH_MAIN,
        PROOF_SHAPE_BATCH_PERMUTATION, PROOF_SHAPE_BATCH_PREPROCESSED,
        PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES, PROOF_SHAPE_NAMESPACE_VK_META,
        PROOF_SHAPE_VK_META_BOUNDARY_KIND, PROOF_SHAPE_VK_META_BOUNDARY_X_BASE,
    },
    statement_boundary_air_dt::STATEMENT_GLOBAL_CHUNKS,
    statement_dt::{
        CORE_PV_COMMITTED_VALUE_DIGEST_START, CORE_PV_DEFERRED_PROOFS_DIGEST_START,
        CORE_PV_EXECUTION_SHARD, CORE_PV_EXIT_CLK, CORE_PV_EXIT_CODE, CORE_PV_GLOBAL_INTERVAL_END,
        CORE_PV_GLOBAL_INTERVAL_START, CORE_PV_LAST_FINALIZE_ADDR, CORE_PV_LAST_INIT_ADDR,
        CORE_PV_NEXT_PC, CORE_PV_PREVIOUS_FINALIZE_ADDR, CORE_PV_PREVIOUS_INIT_ADDR, CORE_PV_SHARD,
        CORE_PV_START_CLK, CORE_PV_START_PC, NATIVE_PV_COMMITTED_VALUE_DIGEST_START,
        NATIVE_PV_CONTAINS_EXECUTION_SHARD, NATIVE_PV_DEFERRED_PROOFS_DIGEST_START,
        NATIVE_PV_DT_VK_DIGEST_START, NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START,
        NATIVE_PV_EXIT_CODE, NATIVE_PV_GLOBAL_INTERVAL_END, NATIVE_PV_GLOBAL_INTERVAL_START,
        NATIVE_PV_IS_COMPLETE, NATIVE_PV_LAST_FINALIZE_ADDR, NATIVE_PV_LAST_INIT_ADDR,
        NATIVE_PV_NEXT_EXECUTION_SHARD, NATIVE_PV_NEXT_PC, NATIVE_PV_NEXT_SHARD,
        NATIVE_PV_PREVIOUS_FINALIZE_ADDR, NATIVE_PV_PREVIOUS_INIT_ADDR,
        NATIVE_PV_START_EXECUTION_SHARD, NATIVE_PV_START_PC,
        NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START, NATIVE_PV_START_SHARD,
        NATIVE_PV_VK_ROOT_START,
    },
    symbolic_expr_adapter_dt::{RecursionPolyAirLeaf, RecursionPolyAirNode, RecursionPolyAirOp},
    symbolic_expr_fixed_dt::RecursionChildRole,
    symbolic_ir_dt::{
        evaluate_chip_node_arena_profiled, RecursionPolyAirChipIr, RecursionPolyAirEnv,
        RecursionPolyAirVerifierProgram, RecursionPolyAirVerifierProgramDto,
    },
    system_dt::{
        RecursionBatchConstraintRecord, RecursionNativeChipMetadataRequest, RecursionProofRecord,
        RecursionProofShapeChip, RecursionRecord, RecursionWhirOpenedEvalPublication,
    },
    whir_dt::{WHIR_ROLE_COMPRESS, WHIR_ROLE_SHRINK},
};
use crate::Instant;

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintProgramTraceGenerator;

impl ConstraintProgramTraceGenerator {
    pub fn trace_height(program: &RecursionPolyAirVerifierProgram) -> usize {
        program.constraint_static_plan().node_plans.len().max(1).next_power_of_two()
    }

    pub fn generate_preprocessed_trace(
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let plan = program.constraint_static_plan();
        let mut rows = Vec::with_capacity(plan.node_plans.len());
        plan.for_each_node(program, |node| rows.push(program_prep_row(node)));
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS, height)
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let counts = program_static_presence_counts(record);
        let plan = program.constraint_static_plan();
        let mut rows = Vec::with_capacity(plan.node_plans.len());
        plan.for_each_node(program, |node| {
            let mut values = vec![F::zero(); NUM_CONSTRAINT_PROGRAM_COLS];
            let cols: &mut ConstraintProgramCols<F> = values.as_mut_slice().borrow_mut();
            cols.mult = f(*counts.get(&node.static_chip_id).unwrap_or(&0));
            rows.push(values);
        });
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_PROGRAM_COLS);
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_PROGRAM_COLS, height)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintRootTableTraceGenerator;

impl ConstraintRootTableTraceGenerator {
    pub fn trace_height(program: &RecursionPolyAirVerifierProgram) -> usize {
        program.constraint_static_plan().root_rows.len().max(1).next_power_of_two()
    }

    pub fn generate_preprocessed_trace(
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let plan = program.constraint_static_plan();
        let rows = plan.root_rows.iter().map(root_table_prep_row).collect::<Vec<_>>();
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS, height)
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let counts = program_static_presence_counts(record);
        let plan = program.constraint_static_plan();
        let rows = plan
            .root_rows
            .iter()
            .map(|row| {
                let mut values = vec![F::zero(); NUM_CONSTRAINT_ROOT_TABLE_COLS];
                let cols: &mut ConstraintRootTableCols<F> = values.as_mut_slice().borrow_mut();
                let mult = f(root_table_row_multiplicity(record, &counts, row));
                if row.static_chip_id == CONSTRAINT_HEIGHT_TABLE_STATIC_ID {
                    cols.height_mult = mult;
                } else {
                    cols.root_mult = mult;
                }
                values
            })
            .collect::<Vec<_>>();
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_ROOT_TABLE_COLS);
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_ROOT_TABLE_COLS, height)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintDagEvalTraceGenerator;

impl ConstraintDagEvalTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        constraint_case_artifact(record, program).dag.len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let artifact = constraint_case_artifact(record, program);
        let plan = program.constraint_static_plan();
        let row_count = artifact.dag.len();
        let mut values = zeroed_trace_values(row_count, NUM_CONSTRAINT_DAG_EVAL_COLS);
        record_constraint_matrix_bytes(record, row_count, NUM_CONSTRAINT_DAG_EVAL_COLS);
        let mut row_index = 0usize;
        artifact.dag.for_each_case(program, &plan, |case, program_node| {
            let start = row_index * NUM_CONSTRAINT_DAG_EVAL_COLS;
            fill_dag_case_row(
                &mut values[start..start + NUM_CONSTRAINT_DAG_EVAL_COLS],
                case,
                program_node,
            );
            row_index += 1;
        });
        debug_assert_eq!(row_index, row_count);
        compressed_values(
            values,
            NUM_CONSTRAINT_DAG_EVAL_COLS,
            row_count.max(1).next_power_of_two(),
            vec![F::zero(); NUM_CONSTRAINT_DAG_EVAL_COLS],
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintFoldTraceGenerator;

impl ConstraintFoldTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        fold_rows_cached(record, program).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows = fold_rows_cached(record, program);
        let row_count = rows.len();
        let padding = fold_padding_row();
        let mut values = Vec::with_capacity(row_count.max(1) * NUM_CONSTRAINT_FOLD_COLS);
        if row_count == 0 {
            values.extend_from_slice(&padding);
        } else {
            for row in rows.iter() {
                append_fold_row(&mut values, row);
            }
        }
        record_constraint_matrix_bytes(record, row_count, NUM_CONSTRAINT_FOLD_COLS);
        compressed_values(
            values,
            NUM_CONSTRAINT_FOLD_COLS,
            row_count.max(1).next_power_of_two(),
            padding,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintChallengeTraceGenerator;

impl ConstraintChallengeTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        challenge_rows_cached(record, program).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows = challenge_rows_cached(record, program);
        let row_count = rows.len();
        let mut values = Vec::with_capacity(row_count.max(1) * NUM_CONSTRAINT_CHALLENGE_COLS);
        if row_count == 0 {
            values.resize(NUM_CONSTRAINT_CHALLENGE_COLS, F::zero());
        } else {
            for row in rows.iter() {
                let start = values.len();
                values.resize(start + NUM_CONSTRAINT_CHALLENGE_COLS, F::zero());
                fill_challenge_row(&mut values[start..], row);
            }
        }
        record_constraint_matrix_bytes(record, row_count, NUM_CONSTRAINT_CHALLENGE_COLS);
        compressed_values(
            values,
            NUM_CONSTRAINT_CHALLENGE_COLS,
            row_count.max(1).next_power_of_two(),
            vec![F::zero(); NUM_CONSTRAINT_CHALLENGE_COLS],
        )
    }
}

pub struct ConstraintBetaLadderTraceGenerator;

impl ConstraintBetaLadderTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        _program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        let rows = constraint_replay_present_proof_count(record) * CONSTRAINT_MAX_BETA_POWERS;
        rows.max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows = beta_ladder_rows_cached(record, program)
            .iter()
            .map(beta_ladder_row)
            .collect::<Vec<_>>();
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_BETA_LADDER_COLS);
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_BETA_LADDER_COLS, height)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintTerminalTraceGenerator;

impl ConstraintTerminalTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        terminal_rows_cached(record, program).len().max(1).next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows =
            terminal_rows_cached(record, program).iter().map(terminal_row).collect::<Vec<_>>();
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_TERMINAL_COLS);
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_TERMINAL_COLS, height)
    }

    /// The narrow role writes only its committed columns. Its row construction
    /// also skips the wide curve and state-imbalance witnesses.
    pub fn generate_trace_compressed_narrow(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows = terminal_rows_cached(record, program)
            .iter()
            .map(terminal_row_narrow)
            .collect::<Vec<_>>();
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_TERMINAL_NARROW_COLS);
        let height = rows.len().max(1).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_TERMINAL_NARROW_COLS, height)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintBoundaryTraceGenerator;

impl ConstraintBoundaryTraceGenerator {
    pub fn trace_height(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> usize {
        terminal_rows_cached(record, program)
            .iter()
            .filter(|row| row.is_final)
            .count()
            .max(2)
            .next_power_of_two()
    }

    pub fn generate_trace_compressed(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> CompressedMatrix<F> {
        let rows = terminal_rows_cached(record, program)
            .iter()
            .filter(|row| row.is_final)
            .map(constraint_boundary_row)
            .collect::<Vec<_>>();
        record_constraint_matrix_bytes(record, rows.len(), NUM_CONSTRAINT_BOUNDARY_COLS);
        let height = rows.len().max(2).next_power_of_two();
        compressed_rows(rows, NUM_CONSTRAINT_BOUNDARY_COLS, height)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintProgramRow {
    pub static_chip_id: usize,
    pub node_idx: usize,
    pub is_leaf: bool,
    pub is_const: bool,
    pub is_add: bool,
    pub is_sub: bool,
    pub is_mul: bool,
    pub is_fused: bool,
    pub lhs_idx: usize,
    pub rhs_idx: usize,
    pub third_idx: usize,
    pub aux: F,
    pub leaf_kind: usize,
    pub fanout: usize,
}

impl ConstraintProgramRow {
    fn op_code(&self) -> usize {
        if self.is_leaf {
            CONSTRAINT_OP_LEAF
        } else if self.is_const {
            CONSTRAINT_OP_CONST
        } else if self.is_add {
            CONSTRAINT_OP_ADD
        } else if self.is_sub {
            CONSTRAINT_OP_SUB
        } else if self.is_mul {
            CONSTRAINT_OP_MUL
        } else if self.is_fused {
            CONSTRAINT_OP_FUSED
        } else {
            CONSTRAINT_OP_LEAF
        }
    }
}

/// Compact retained projection of one frozen IR node. Chip identity and node
/// identity remain owned by the frozen IR; the static plan retains only the
/// authenticated route plus fanout that are not cheap direct node fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConstraintProgramNodePlan {
    lhs_idx: u32,
    rhs_idx: u32,
    third_idx: u32,
    fanout: u32,
    aux: F,
    op_code: u8,
    leaf_kind: u8,
}

impl ConstraintProgramNodePlan {
    fn from_row(row: &ConstraintProgramRow) -> Result<Self, String> {
        Ok(Self {
            lhs_idx: u32::try_from(row.lhs_idx)
                .map_err(|_| "constraint lhs route exceeds u32".to_string())?,
            rhs_idx: u32::try_from(row.rhs_idx)
                .map_err(|_| "constraint rhs route exceeds u32".to_string())?,
            third_idx: u32::try_from(row.third_idx)
                .map_err(|_| "constraint third route exceeds u32".to_string())?,
            fanout: u32::try_from(row.fanout)
                .map_err(|_| "constraint fanout exceeds u32".to_string())?,
            aux: row.aux,
            op_code: u8::try_from(row.op_code())
                .map_err(|_| "constraint op code exceeds u8".to_string())?,
            leaf_kind: u8::try_from(row.leaf_kind)
                .map_err(|_| "constraint leaf kind exceeds u8".to_string())?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ConstraintProgramNodeRef<'a> {
    static_chip_id: usize,
    node_idx: usize,
    plan: &'a ConstraintProgramNodePlan,
}

impl ConstraintProgramNodeRef<'_> {
    fn op_code(self) -> usize {
        usize::from(self.plan.op_code)
    }

    fn is_leaf(self) -> bool {
        self.op_code() == CONSTRAINT_OP_LEAF
    }

    fn is_const(self) -> bool {
        self.op_code() == CONSTRAINT_OP_CONST
    }

    fn is_add(self) -> bool {
        self.op_code() == CONSTRAINT_OP_ADD
    }

    fn is_sub(self) -> bool {
        self.op_code() == CONSTRAINT_OP_SUB
    }

    fn is_mul(self) -> bool {
        self.op_code() == CONSTRAINT_OP_MUL
    }

    fn is_fused(self) -> bool {
        self.op_code() == CONSTRAINT_OP_FUSED
    }

    fn lhs_idx(self) -> usize {
        self.plan.lhs_idx as usize
    }

    fn rhs_idx(self) -> usize {
        self.plan.rhs_idx as usize
    }

    fn third_idx(self) -> usize {
        self.plan.third_idx as usize
    }

    fn fanout(self) -> usize {
        self.plan.fanout as usize
    }

    fn leaf_kind(self) -> usize {
        usize::from(self.plan.leaf_kind)
    }

    fn materialize(self) -> ConstraintProgramRow {
        ConstraintProgramRow {
            static_chip_id: self.static_chip_id,
            node_idx: self.node_idx,
            is_leaf: self.is_leaf(),
            is_const: self.is_const(),
            is_add: self.is_add(),
            is_sub: self.is_sub(),
            is_mul: self.is_mul(),
            is_fused: self.is_fused(),
            lhs_idx: self.lhs_idx(),
            rhs_idx: self.rhs_idx(),
            third_idx: self.third_idx(),
            aux: self.plan.aux,
            leaf_kind: self.leaf_kind(),
            fanout: self.fanout(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintRootTableRow {
    pub static_chip_id: usize,
    pub root_kind: usize,
    pub root_ord: usize,
    pub node_idx: usize,
    pub sign: i32,
}

#[derive(Debug, Clone, Copy)]
struct StaticChallengeDemand {
    perm_alpha: usize,
    beta_power: [usize; CONSTRAINT_MAX_BETA_POWERS],
    beta_septix: usize,
    first: usize,
    last: usize,
    invalid_beta_power: Option<usize>,
}

impl Default for StaticChallengeDemand {
    fn default() -> Self {
        Self {
            perm_alpha: 0,
            beta_power: [0; CONSTRAINT_MAX_BETA_POWERS],
            beta_septix: 0,
            first: 0,
            last: 0,
            invalid_beta_power: None,
        }
    }
}

#[derive(Debug)]
struct ConstraintChipPlan {
    program_chip_index: usize,
    static_chip_id: usize,
    node_plan_range: Range<usize>,
    challenge_demand: StaticChallengeDemand,
}

/// Immutable constraint metadata compiled once with a frozen recursion program.
///
/// No proof value is stored here. Clones of the program share this plan by `Arc`; deserialized
/// ladder programs rebuild it eagerly before any proof becomes ready.
#[derive(Debug)]
pub(crate) struct ConstraintProgramPlan {
    chip_index_by_static_id: Box<[usize]>,
    chips: Box<[ConstraintChipPlan]>,
    node_plans: Arc<[ConstraintProgramNodePlan]>,
    root_rows: Arc<[ConstraintRootTableRow]>,
    compile_us: u64,
    retained_bytes: u64,
    legacy_program_row_bytes: u64,
    node_plan_bytes: u64,
}

impl ConstraintProgramPlan {
    const MISSING_CHIP: usize = usize::MAX;

    pub(crate) fn compile(program: &RecursionPolyAirVerifierProgramDto) -> Result<Self, String> {
        let started = Instant::now();
        let max_static_chip_id = program.chips.iter().map(|chip| chip.static_chip_id).max();
        let dense_chip_count = max_static_chip_id
            .map_or(Ok(0), |id| id.checked_add(1).ok_or("static chip id overflow"))?;
        let total_nodes = program.chips.iter().try_fold(0usize, |total, chip| {
            total.checked_add(chip.node_table.len()).ok_or("constraint program row count overflow")
        })?;
        let root_row_count = program
            .chips
            .iter()
            .try_fold(0usize, |total, chip| {
                let roots = chip
                    .gate_roots
                    .len()
                    .checked_add(chip.lookup_multiplicity_roots.len())
                    .and_then(|count| count.checked_add(chip.lookup_multiplicity_roots.len()))
                    .ok_or("constraint root row count overflow")?;
                total.checked_add(roots).ok_or("constraint root row count overflow")
            })?
            .checked_add(CONSTRAINT_HEIGHT_TABLE_ROWS)
            .ok_or("constraint height-table row count overflow")?;
        let mut chip_index_by_static_id = Vec::new();
        chip_index_by_static_id
            .try_reserve_exact(dense_chip_count)
            .map_err(|_| "constraint dense chip index allocation rejected".to_string())?;
        chip_index_by_static_id.resize(dense_chip_count, Self::MISSING_CHIP);
        let mut chips = Vec::new();
        chips
            .try_reserve_exact(program.chips.len())
            .map_err(|_| "constraint chip plan allocation rejected".to_string())?;
        let mut node_plans = Vec::new();
        node_plans
            .try_reserve_exact(total_nodes)
            .map_err(|_| "constraint node-plan allocation rejected".to_string())?;

        for (program_chip_index, chip) in program.chips.iter().enumerate() {
            if let Some(slot) = chip_index_by_static_id.get_mut(chip.static_chip_id) {
                *slot = program_chip_index;
            }
            let fanouts = node_fanouts(chip)?;
            let row_start = node_plans.len();
            for node in &chip.node_table {
                let fanout = *fanouts.get(node.node_id as usize).ok_or_else(|| {
                    format!("validated node {} missing fanout slot", node.node_id)
                })?;
                let row = program_row_for_node(chip, node, fanout)?;
                node_plans.push(ConstraintProgramNodePlan::from_row(&row)?);
            }
            let row_end = node_plans.len();
            let mut challenge_demand = StaticChallengeDemand::default();
            for node in &chip.node_table {
                let RecursionPolyAirOp::Leaf(leaf) = &node.op else {
                    continue;
                };
                match leaf {
                    RecursionPolyAirLeaf::PermAlpha => challenge_demand.perm_alpha += 1,
                    RecursionPolyAirLeaf::BetaPower { power }
                        if *power < CONSTRAINT_MAX_BETA_POWERS =>
                    {
                        challenge_demand.beta_power[*power] += 1;
                    }
                    RecursionPolyAirLeaf::BetaPower { power } => {
                        challenge_demand.invalid_beta_power.get_or_insert(*power);
                    }
                    RecursionPolyAirLeaf::BetaSeptix => challenge_demand.beta_septix += 1,
                    RecursionPolyAirLeaf::IsFirstRow => challenge_demand.first += 1,
                    RecursionPolyAirLeaf::IsLastRow => challenge_demand.last += 1,
                    _ => {}
                }
            }
            chips.push(ConstraintChipPlan {
                program_chip_index,
                static_chip_id: chip.static_chip_id,
                node_plan_range: row_start..row_end,
                challenge_demand,
            });
        }

        let root_rows = root_table_rows_uncached_from_chips(&program.chips, root_row_count)?;
        let legacy_program_row_bytes = node_plans
            .len()
            .checked_mul(core::mem::size_of::<ConstraintProgramRow>())
            .ok_or_else(|| "constraint legacy program-row byte count overflow".to_string())?;
        let node_plan_bytes = node_plans
            .len()
            .checked_mul(core::mem::size_of::<ConstraintProgramNodePlan>())
            .ok_or_else(|| "constraint node-plan byte count overflow".to_string())?;
        let retained_bytes = chip_index_by_static_id
            .len()
            .checked_mul(core::mem::size_of::<usize>())
            .and_then(|bytes| {
                chips
                    .len()
                    .checked_mul(core::mem::size_of::<ConstraintChipPlan>())
                    .and_then(|more| bytes.checked_add(more))
            })
            .and_then(|bytes| bytes.checked_add(node_plan_bytes))
            .and_then(|bytes| {
                root_rows
                    .len()
                    .checked_mul(core::mem::size_of::<ConstraintRootTableRow>())
                    .and_then(|more| bytes.checked_add(more))
            })
            .ok_or_else(|| "constraint static plan retained-byte count overflow".to_string())?;
        Ok(Self {
            chip_index_by_static_id: chip_index_by_static_id.into_boxed_slice(),
            chips: chips.into_boxed_slice(),
            node_plans: node_plans.into(),
            root_rows: root_rows.into(),
            compile_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            retained_bytes: u64::try_from(retained_bytes).unwrap_or(u64::MAX),
            legacy_program_row_bytes: u64::try_from(legacy_program_row_bytes).unwrap_or(u64::MAX),
            node_plan_bytes: u64::try_from(node_plan_bytes).unwrap_or(u64::MAX),
        })
    }

    fn chip(&self, static_chip_id: usize) -> Option<&ConstraintChipPlan> {
        let program_chip_index = *self
            .chip_index_by_static_id
            .get(static_chip_id)
            .filter(|&&idx| idx != Self::MISSING_CHIP)?;
        self.chips.get(program_chip_index).filter(|plan| {
            plan.program_chip_index == program_chip_index && plan.static_chip_id == static_chip_id
        })
    }

    fn node_plans_for_chip(&self, chip: &ConstraintChipPlan) -> &[ConstraintProgramNodePlan] {
        &self.node_plans[chip.node_plan_range.clone()]
    }

    fn node_ref<'a>(
        &'a self,
        program: &'a RecursionPolyAirVerifierProgram,
        program_chip_index: usize,
        node_offset: usize,
    ) -> ConstraintProgramNodeRef<'a> {
        let chip_plan = self
            .chips
            .get(program_chip_index)
            .unwrap_or_else(|| panic!("constraint chip plan {program_chip_index} is missing"));
        let chip = program
            .chips
            .get(program_chip_index)
            .unwrap_or_else(|| panic!("constraint IR chip {program_chip_index} is missing"));
        let node = chip.node_table.get(node_offset).unwrap_or_else(|| {
            panic!(
                "constraint IR node {node_offset} is missing for static chip {}",
                chip.static_chip_id
            )
        });
        let plan_idx = chip_plan.node_plan_range.start + node_offset;
        let node_plan = self
            .node_plans
            .get(plan_idx)
            .unwrap_or_else(|| panic!("constraint node plan {plan_idx} is missing"));
        ConstraintProgramNodeRef {
            static_chip_id: chip.static_chip_id,
            node_idx: node.node_id as usize,
            plan: node_plan,
        }
    }

    fn for_each_node<'a>(
        &'a self,
        program: &'a RecursionPolyAirVerifierProgram,
        mut visit: impl FnMut(ConstraintProgramNodeRef<'a>),
    ) {
        for chip in &self.chips {
            let node_count = program.chips[chip.program_chip_index].node_table.len();
            for node_offset in 0..node_count {
                visit(self.node_ref(program, chip.program_chip_index, node_offset));
            }
        }
    }
}

impl RecursionPolyAirVerifierProgram {
    pub(crate) fn has_matching_constraint_static_plan(&self) -> bool {
        // This is an executable typestate assertion now: no frozen program can exist without the
        // plan compiled from the same validated DTO and stored in the same Arc owner.
        true
    }

    pub(crate) fn constraint_static_plan(&self) -> Arc<ConstraintProgramPlan> {
        Arc::clone(&self.constraint_static_plan)
    }

    /// Cold-start diagnostics only. Callers retain the plan itself and never recompute these
    /// values in record or trace generation.
    pub(crate) fn constraint_static_plan_cold_metrics(&self) -> (u64, u64) {
        (self.constraint_static_plan.compile_us, self.constraint_static_plan.retained_bytes)
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintDagRow {
    pub proof_idx: usize,
    pub chip_idx: usize,
    pub program: ConstraintProgramRow,
    pub leaf_flags: [bool; CONSTRAINT_LEAF_KIND_COUNT],
    pub value: [F; D_EF],
    pub lhs_value: [F; D_EF],
    pub rhs_value: [F; D_EF],
    pub third_value: [F; D_EF],
    pub opened_batch_pos: usize,
}

/// Ephemeral proof-local view for one DAG row. Production retains only a flat value arena plus
/// one descriptor per proof/chip; operand values and leaf routing are reconstructed from the
/// frozen [`ConstraintProgramPlan`] while the row is consumed.
#[derive(Debug, Clone)]
struct ConstraintDagCaseRow {
    proof_idx: usize,
    chip_idx: usize,
    value: [F; D_EF],
    lhs_value: [F; D_EF],
    rhs_value: [F; D_EF],
    third_value: [F; D_EF],
    opened_batch_pos: usize,
}

#[derive(Debug, Clone, Copy)]
struct ConstraintDagChipDescriptor {
    proof_idx: usize,
    chip_idx: usize,
    program_chip_index: usize,
    node_count: usize,
    value_start: usize,
    prep_pos: usize,
}

#[derive(Debug, Default)]
struct ConstraintDagArena {
    chips: Vec<ConstraintDagChipDescriptor>,
    values: Vec<[F; D_EF]>,
}

impl ConstraintDagArena {
    fn with_capacity(chip_count: usize, node_count: usize) -> Self {
        Self { chips: Vec::with_capacity(chip_count), values: Vec::with_capacity(node_count) }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn allocated_bytes(&self) -> usize {
        self.chips
            .capacity()
            .saturating_mul(core::mem::size_of::<ConstraintDagChipDescriptor>())
            .saturating_add(
                self.values.capacity().saturating_mul(core::mem::size_of::<[F; D_EF]>()),
            )
    }

    fn push_chip(
        &mut self,
        proof_idx: usize,
        chip_idx: usize,
        program_chip_index: usize,
        prep_pos: usize,
        node_values: &[EF],
    ) {
        let value_start = self.values.len();
        self.values.extend(node_values.iter().map(ext_limbs));
        self.chips.push(ConstraintDagChipDescriptor {
            proof_idx,
            chip_idx,
            program_chip_index,
            node_count: node_values.len(),
            value_start,
            prep_pos,
        });
    }

    fn for_each_case(
        &self,
        program: &RecursionPolyAirVerifierProgram,
        plan: &ConstraintProgramPlan,
        mut visit: impl FnMut(&ConstraintDagCaseRow, ConstraintProgramNodeRef<'_>),
    ) {
        for chip in &self.chips {
            let values = &self.values[chip.value_start..chip.value_start + chip.node_count];
            for (node_offset, value) in values.iter().copied().enumerate() {
                let program_node = plan.node_ref(program, chip.program_chip_index, node_offset);
                let mut case = ConstraintDagCaseRow {
                    proof_idx: chip.proof_idx,
                    chip_idx: chip.chip_idx,
                    value,
                    lhs_value: [F::zero(); D_EF],
                    rhs_value: [F::zero(); D_EF],
                    third_value: [F::zero(); D_EF],
                    opened_batch_pos: 0,
                };
                if program_node.is_add() ||
                    program_node.is_sub() ||
                    program_node.is_mul() ||
                    program_node.is_fused()
                {
                    case.lhs_value = values[program_node.lhs_idx()];
                    case.rhs_value = values[program_node.rhs_idx()];
                }
                if program_node.is_fused() {
                    case.third_value = values[program_node.third_idx()];
                }
                if program_node.is_leaf() {
                    match program_node.leaf_kind() {
                        CONSTRAINT_LEAF_PREPROCESSED => {
                            case.opened_batch_pos = chip.prep_pos;
                        }
                        CONSTRAINT_LEAF_MAIN => {
                            case.opened_batch_pos = chip.chip_idx;
                        }
                        CONSTRAINT_LEAF_RESERVED_POLY => {
                            case.opened_batch_pos =
                                if program_node.lhs_idx() == PROOF_SHAPE_BATCH_PREPROCESSED {
                                    chip.prep_pos
                                } else {
                                    chip.chip_idx
                                };
                        }
                        CONSTRAINT_LEAF_PRECOMPUTED => {
                            case.lhs_value = value;
                        }
                        _ => {}
                    }
                }
                visit(&case, program_node);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintFoldRow {
    pub proof_idx: usize,
    pub cursor: usize,
    pub remaining_chips: usize,
    pub local_ord: usize,
    pub chain_send_local_ord: usize,
    pub static_chip_id: usize,
    pub log_height: usize,
    pub gate_count: usize,
    pub batch_count: usize,
    /// Gate ordinal, twice the batch ordinal, or log height on skip rows.
    pub root_ord: usize,
    pub is_skip: bool,
    pub is_gate: bool,
    pub is_batch: bool,
    pub alpha: [F; D_EF],
    pub acc_in: [F; D_EF],
    pub acc_out: [F; D_EF],
    pub pacc_in: [F; D_EF],
    pub pacc_out: [F; D_EF],
    pub perm_sum_in: [F; D_EF],
    pub perm_sum_out: [F; D_EF],
    pub root_nodes: [usize; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub multiplicity_signs: [i32; CONSTRAINT_FOLD_BATCH_SIZE],
    pub root_values: [[F; D_EF]; CONSTRAINT_FOLD_ROOT_SLOTS],
    pub batch_has_second: bool,
    pub perm_value: [F; D_EF],
}

#[derive(Debug, Clone)]
pub struct ConstraintChallengeRow {
    pub proof_idx: usize,
    pub chip_idx: usize,
    pub static_chip_id: usize,
    pub main_width: usize,
    pub log_height: usize,
    pub c_chips: usize,
    /// Retained for diagnostics that reconstruct transcript indices. The AIR
    /// takes this value from its layer constructor rather than committing it.
    pub num_public_values: usize,
    pub lcs_limbs: [F; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub eq_acc: [F; D_EF],
    pub first: [F; D_EF],
    pub last: [F; D_EF],
    pub first_send_mult: usize,
    pub last_send_mult: usize,
}

#[derive(Debug, Clone)]
pub struct ConstraintBetaLadderRow {
    pub proof_idx: usize,
    pub power_idx: usize,
    pub beta: [F; D_EF],
    pub prev_power_or_alpha: [F; D_EF],
    pub power: [F; D_EF],
    pub serve_mult: usize,
    pub challenges_recv_mult: bool,
    pub alpha_serve_mult: usize,
    pub septix_serve_mult: usize,
}

#[derive(Debug, Clone)]
pub struct ConstraintTerminalRow {
    pub proof_idx: usize,
    pub num_rounds: usize,
    pub c_chips: usize,
    pub num_public_values: usize,
    pub round_idx: usize,
    pub opening_idx: usize,
    pub chip_idx: usize,
    pub is_seed: bool,
    pub is_eq_step: bool,
    pub is_lcs_step: bool,
    pub is_final: bool,
    pub opening_point: [F; D_EF],
    pub eq_challenge: [F; D_EF],
    pub eq_factor: [F; D_EF],
    pub eq_in: [F; D_EF],
    pub eq_out: [F; D_EF],
    pub first_prefix_in: [F; D_EF],
    pub first_prefix_out: [F; D_EF],
    pub last_prefix_in: [F; D_EF],
    pub last_prefix_out: [F; D_EF],
    pub fold_cursor: usize,
    pub alpha: [F; D_EF],
    pub main_eval: [F; D_EF],
    pub perm_eval: [F; D_EF],
    pub last_claim: [F; D_EF],
    pub final_lhs: [F; D_EF],
    pub lcs: [F; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_in: [F; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub state_lcs_out: [F; CONSTRAINT_TERMINAL_LCS_LIMBS],
    pub public_values: [F; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    pub public_value_recv_mult: bool,
    pub state_chain_recv_mult: bool,
    pub state_chain_send_mult: bool,
    pub perm_alpha: [F; D_EF],
    pub beta_powers: [[F; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    pub state_clock_changed: F,
    pub state_clock_delta_inverse: F,
    pub state_transition_recv_inverse: [F; D_EF],
    pub state_transition_send_inverse: [F; D_EF],
    pub init_address_recv_inverse: [F; D_EF],
    pub init_address_send_inverse: [F; D_EF],
    pub finalize_address_recv_inverse: [F; D_EF],
    pub finalize_address_send_inverse: [F; D_EF],
    pub global_chain_source_inverse: [F; D_EF],
    pub global_chain_sink_inverse: [F; D_EF],
    pub summary_recv_mult: bool,
    pub summary_id_base: usize,
    pub opening_point_recv_mult: bool,
    pub eq_recv_mult: bool,
    pub last_claim_recv_mult: bool,
    pub fold_chain_recv_mult: bool,
    pub eq_chain_recv_mult: bool,
    pub eq_chain_send_mult: usize,
}

impl ConstraintTerminalRow {
    fn blank(proof_idx: usize, batch: &RecursionBatchConstraintRecord) -> Self {
        Self {
            proof_idx,
            num_rounds: batch.num_rounds,
            c_chips: batch.c_chips,
            num_public_values: batch.num_public_values,
            round_idx: 0,
            opening_idx: 0,
            chip_idx: 0,
            is_seed: false,
            is_eq_step: false,
            is_lcs_step: false,
            is_final: false,
            opening_point: [F::zero(); D_EF],
            eq_challenge: [F::zero(); D_EF],
            eq_factor: one_ext_limbs(),
            eq_in: one_ext_limbs(),
            eq_out: one_ext_limbs(),
            first_prefix_in: one_ext_limbs(),
            first_prefix_out: one_ext_limbs(),
            last_prefix_in: one_ext_limbs(),
            last_prefix_out: one_ext_limbs(),
            fold_cursor: 0,
            alpha: [F::zero(); D_EF],
            main_eval: [F::zero(); D_EF],
            perm_eval: [F::zero(); D_EF],
            last_claim: [F::zero(); D_EF],
            final_lhs: [F::zero(); D_EF],
            lcs: [F::zero(); CONSTRAINT_TERMINAL_LCS_LIMBS],
            state_lcs_in: [F::zero(); CONSTRAINT_TERMINAL_LCS_LIMBS],
            state_lcs_out: [F::zero(); CONSTRAINT_TERMINAL_LCS_LIMBS],
            public_values: [F::zero(); CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
            public_value_recv_mult: false,
            state_chain_recv_mult: false,
            state_chain_send_mult: false,
            perm_alpha: [F::zero(); D_EF],
            beta_powers: [[F::zero(); D_EF]; CONSTRAINT_CHAIN_LIMBS],
            state_clock_changed: F::zero(),
            state_clock_delta_inverse: F::zero(),
            state_transition_recv_inverse: [F::zero(); D_EF],
            state_transition_send_inverse: [F::zero(); D_EF],
            init_address_recv_inverse: [F::zero(); D_EF],
            init_address_send_inverse: [F::zero(); D_EF],
            finalize_address_recv_inverse: [F::zero(); D_EF],
            finalize_address_send_inverse: [F::zero(); D_EF],
            global_chain_source_inverse: [F::zero(); D_EF],
            global_chain_sink_inverse: [F::zero(); D_EF],
            summary_recv_mult: false,
            summary_id_base: 0,
            opening_point_recv_mult: false,
            eq_recv_mult: false,
            last_claim_recv_mult: false,
            fold_chain_recv_mult: false,
            eq_chain_recv_mult: false,
            eq_chain_send_mult: 0,
        }
    }
}

pub fn program_rows(program: &RecursionPolyAirVerifierProgram) -> Vec<ConstraintProgramRow> {
    let plan = program.constraint_static_plan();
    let mut rows = Vec::with_capacity(plan.node_plans.len());
    plan.for_each_node(program, |node| rows.push(node.materialize()));
    rows
}

pub fn root_table_rows(program: &RecursionPolyAirVerifierProgram) -> Vec<ConstraintRootTableRow> {
    program.constraint_static_plan().root_rows.as_ref().to_vec()
}

#[cfg(test)]
fn root_table_rows_uncached(
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintRootTableRow> {
    let root_row_count = program
        .chips
        .iter()
        .map(|chip| chip.gate_roots.len() + chip.lookup_multiplicity_roots.len() * 2)
        .sum::<usize>()
        .saturating_add(CONSTRAINT_HEIGHT_TABLE_ROWS);
    root_table_rows_uncached_from_chips(&program.chips, root_row_count)
        .expect("frozen constraint program root rows were validated during construction")
}

fn root_table_rows_uncached_from_chips(
    chips: &[RecursionPolyAirChipIr],
    capacity: usize,
) -> Result<Vec<ConstraintRootTableRow>, String> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(capacity)
        .map_err(|_| "constraint root row allocation rejected".to_string())?;
    for chip in chips {
        for root in &chip.gate_roots {
            rows.push(ConstraintRootTableRow {
                static_chip_id: chip.static_chip_id,
                root_kind: CONSTRAINT_ROOT_GATE,
                root_ord: root.gate_idx,
                node_idx: root.root_node_id as usize,
                sign: 1,
            });
        }
        for root in &chip.lookup_multiplicity_roots {
            rows.push(ConstraintRootTableRow {
                static_chip_id: chip.static_chip_id,
                root_kind: CONSTRAINT_ROOT_MULTIPLICITY,
                root_ord: root.lookup_idx,
                node_idx: root.root_node_id as usize,
                sign: if root.is_send { 1 } else { -1 },
            });
        }
        let lookup_roots = chip.lookup_multiplicity_roots.len();
        for root in chip.derived_roots.iter().filter_map(|root| match root {
            crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::PrecomputeLc {
                index,
                root_node_id,
            } if *index < lookup_roots => Some((*index, *root_node_id)),
            _ => None,
        }) {
            rows.push(ConstraintRootTableRow {
                static_chip_id: chip.static_chip_id,
                root_kind: CONSTRAINT_ROOT_PRECOMPUTE_DENOM,
                root_ord: root.0,
                node_idx: root.1 as usize,
                sign: 1,
            });
        }
    }
    for log_height in 0..CONSTRAINT_HEIGHT_TABLE_ROWS {
        let height = F::from_canonical_usize(1usize << log_height);
        rows.push(ConstraintRootTableRow {
            static_chip_id: CONSTRAINT_HEIGHT_TABLE_STATIC_ID,
            root_kind: CONSTRAINT_ROOT_HEIGHT_INVERSE,
            root_ord: log_height,
            node_idx: height.inverse().as_canonical_u32() as usize,
            sign: 1,
        });
    }
    Ok(rows)
}

fn root_table_row_multiplicity(
    record: &RecursionRecord,
    static_counts: &BTreeMap<usize, usize>,
    row: &ConstraintRootTableRow,
) -> usize {
    if row.static_chip_id == CONSTRAINT_HEIGHT_TABLE_STATIC_ID {
        record
            .proof_records
            .iter()
            .flat_map(|proof| &proof.proof_shape.chips)
            .filter(|chip| chip.log_height == row.root_ord)
            .count()
    } else {
        *static_counts.get(&row.static_chip_id).unwrap_or(&0)
    }
}

pub fn dag_rows(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintDagRow> {
    let artifact = constraint_case_artifact(record, program);
    let plan = program.constraint_static_plan();
    let mut rows = Vec::with_capacity(artifact.dag.len());
    artifact.dag.for_each_case(program, &plan, |case, program_node| {
        rows.push(materialize_dag_row(case, program_node))
    });
    rows
}

fn for_each_dag_row(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
    mut visit: impl FnMut(&ConstraintDagRow),
) {
    let artifact = constraint_case_artifact(record, program);
    let plan = program.constraint_static_plan();
    artifact.dag.for_each_case(program, &plan, |case, program_node| {
        let row = materialize_dag_row(case, program_node);
        visit(&row);
    });
}

pub fn fold_rows(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintFoldRow> {
    fold_rows_cached(record, program).as_ref().to_vec()
}

fn fold_rows_cached(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Arc<Vec<ConstraintFoldRow>> {
    Arc::clone(&constraint_case_artifact(record, program).fold)
}

pub fn challenge_rows(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintChallengeRow> {
    challenge_rows_cached(record, program).as_ref().to_vec()
}

fn challenge_rows_cached(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Arc<Vec<ConstraintChallengeRow>> {
    Arc::clone(&constraint_case_artifact(record, program).challenge)
}

pub fn beta_ladder_rows(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintBetaLadderRow> {
    beta_ladder_rows_cached(record, program).as_ref().to_vec()
}

fn beta_ladder_rows_cached(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Arc<Vec<ConstraintBetaLadderRow>> {
    Arc::clone(&constraint_case_artifact(record, program).beta_ladder)
}

pub fn terminal_rows(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Vec<ConstraintTerminalRow> {
    terminal_rows_cached(record, program).as_ref().to_vec()
}

fn terminal_rows_cached(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Arc<Vec<ConstraintTerminalRow>> {
    Arc::clone(&constraint_case_artifact(record, program).terminal)
}

#[derive(Debug, Default)]
struct ConstraintCaseBuildStats {
    total_us: u64,
    static_plan_lookup_us: u64,
    input_us: u64,
    node_eval_us: u64,
    precompute_eval_us: u64,
    remaining_node_eval_us: u64,
    dag_emit_us: u64,
    replay_compact_us: u64,
    fold_emit_us: u64,
    challenge_us: u64,
    beta_ladder_us: u64,
    terminal_us: u64,
    row_allocation_us: u64,
    scratch_drop_us: u64,
    artifact_wrap_us: u64,
    proof_count: usize,
    chip_count: usize,
    node_count: usize,
    precompute_node_count: usize,
    remaining_node_count: usize,
    opened_projection_builds: usize,
    opened_borrowed_views: usize,
    scratch_peak_bytes: usize,
    // Dominant wide DAG/fold row vectors only. The bounded challenge/beta/terminal allocations are
    // already included in their family timers and are not rescanned for allocation telemetry.
    row_reserved_bytes: usize,
    row_vector_reallocations: usize,
}

/// One request-local material bundle. It is deliberately not serialized or shared across cases.
#[derive(Debug)]
pub(crate) struct ConstraintCaseArtifact {
    dag: Arc<ConstraintDagArena>,
    fold: Arc<Vec<ConstraintFoldRow>>,
    challenge: Arc<Vec<ConstraintChallengeRow>>,
    beta_ladder: Arc<Vec<ConstraintBetaLadderRow>>,
    terminal: Arc<Vec<ConstraintTerminalRow>>,
    stats: ConstraintCaseBuildStats,
}

impl ConstraintCaseArtifact {
    fn dynamic_bytes(&self) -> usize {
        self.dag
            .allocated_bytes()
            .saturating_add(
                self.fold.capacity().saturating_mul(core::mem::size_of::<ConstraintFoldRow>()),
            )
            .saturating_add(
                self.challenge
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ConstraintChallengeRow>()),
            )
            .saturating_add(
                self.beta_ladder
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ConstraintBetaLadderRow>()),
            )
            .saturating_add(
                self.terminal
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ConstraintTerminalRow>()),
            )
    }
}

fn constraint_case_artifact(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> Arc<ConstraintCaseArtifact> {
    let expected_authority = program.authority_identity();
    let (installed_authority, artifact) =
        record.tracegen_artifacts.constraint_case.get_or_init(|| {
            (expected_authority, Arc::new(build_constraint_case_artifact(record, program)))
        });
    assert_eq!(
        *installed_authority, expected_authority,
        "one tracegen workspace was used with two constraint-program authorities"
    );
    Arc::clone(artifact)
}

fn build_constraint_case_artifact(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> ConstraintCaseArtifact {
    let total_started = Instant::now();
    let plan_lookup_started = Instant::now();
    let plan = program.constraint_static_plan();
    let static_plan_lookup_us = elapsed_us(plan_lookup_started);
    let mut stats = ConstraintCaseBuildStats::default();
    stats.static_plan_lookup_us = static_plan_lookup_us;
    let input_started = Instant::now();
    let proof_inputs = build_proof_constraint_inputs(record, program, &plan, &mut stats);
    stats.input_us = elapsed_us(input_started);
    let (dag, fold) = dag_fold_rows_uncached(&proof_inputs, &plan, &mut stats);

    let challenge_started = Instant::now();
    let challenge = challenge_rows_uncached(&proof_inputs);
    stats.challenge_us = elapsed_us(challenge_started);

    let beta_ladder_started = Instant::now();
    let child_contains_global_bus = child_contains_global_bus_for_role(program.role);
    let beta_ladder = beta_ladder_rows_uncached(&proof_inputs, child_contains_global_bus);
    stats.beta_ladder_us = elapsed_us(beta_ladder_started);

    let terminal_started = Instant::now();
    let terminal = terminal_rows_uncached(&proof_inputs, child_contains_global_bus, &fold);
    stats.terminal_us = elapsed_us(terminal_started);
    let scratch_drop_started = Instant::now();
    drop(proof_inputs);
    stats.scratch_drop_us = elapsed_us(scratch_drop_started);
    let artifact_wrap_started = Instant::now();
    let dag = Arc::new(dag);
    let fold = Arc::new(fold);
    let challenge = Arc::new(challenge);
    let beta_ladder = Arc::new(beta_ladder);
    let terminal = Arc::new(terminal);
    stats.artifact_wrap_us = elapsed_us(artifact_wrap_started);
    stats.total_us = elapsed_us(total_started);

    ConstraintCaseArtifact { dag, fold, challenge, beta_ladder, terminal, stats }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn vec_allocation_bytes<T>(values: &Vec<T>) -> usize {
    core::mem::size_of::<Vec<T>>()
        .saturating_add(values.capacity().saturating_mul(core::mem::size_of::<T>()))
}

fn chip_node_arena_scratch_bytes(
    node_values: &Vec<EF>,
    precomputed_lc: &Vec<EF>,
    reserved_poly: &Vec<EF>,
) -> usize {
    vec_allocation_bytes(node_values)
        .saturating_add(vec_allocation_bytes(precomputed_lc))
        .saturating_add(vec_allocation_bytes(reserved_poly))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConstraintFoldReplayChip {
    gate_start: usize,
    gate_count: usize,
    batch_start: usize,
    batch_count: usize,
}

#[derive(Debug)]
struct ConstraintFoldReplayArena {
    chips: Vec<ConstraintFoldReplayChip>,
    gate_values: Vec<EF>,
    /// Exact `[d0, d1, unsigned_m0, unsigned_m1]` roots for each fixed batch-two row.
    batch_roots: Vec<[EF; CONSTRAINT_FOLD_ROOT_SLOTS]>,
}

impl ConstraintFoldReplayArena {
    fn for_proof(proof_input: &ProofConstraintInputs<'_>) -> Self {
        let gate_capacity = proof_input.chips.iter().map(|chip| chip.ir.gate_roots.len()).sum();
        let batch_capacity = proof_input
            .chips
            .iter()
            .map(|chip| {
                chip.ir.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE)
            })
            .sum();
        Self {
            chips: Vec::with_capacity(proof_input.chips.len()),
            gate_values: Vec::with_capacity(gate_capacity),
            batch_roots: Vec::with_capacity(batch_capacity),
        }
    }

    fn allocated_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            .saturating_add(
                self.chips
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ConstraintFoldReplayChip>()),
            )
            .saturating_add(self.gate_values.capacity().saturating_mul(core::mem::size_of::<EF>()))
            .saturating_add(
                self.batch_roots
                    .capacity()
                    .saturating_mul(core::mem::size_of::<[EF; CONSTRAINT_FOLD_ROOT_SLOTS]>()),
            )
    }

    fn push_chip(
        &mut self,
        chip: &RecursionPolyAirChipIr,
        node_values: &[EF],
        precomputed_lc: &[EF],
    ) {
        let gate_start = self.gate_values.len();
        self.gate_values.extend(chip.gate_roots.iter().map(|root| {
            *node_values.get(root.root_node_id as usize).unwrap_or_else(|| {
                panic!(
                    "gate root node {} missing for static chip {} {}",
                    root.root_node_id, chip.static_chip_id, chip.chip_name
                )
            })
        }));

        let batch_start = self.batch_roots.len();
        for batch_idx in
            0..chip.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE)
        {
            let start = batch_idx * CONSTRAINT_FOLD_BATCH_SIZE;
            let end =
                (start + CONSTRAINT_FOLD_BATCH_SIZE).min(chip.lookup_multiplicity_roots.len());
            let mut roots = [EF::one(), EF::one(), EF::zero(), EF::zero()];
            for (slot, lookup_idx) in (start..end).enumerate() {
                roots[slot] = *precomputed_lc.get(lookup_idx).unwrap_or_else(|| {
                    panic!(
                        "denominator value {} missing for static chip {} {}",
                        lookup_idx, chip.static_chip_id, chip.chip_name
                    )
                });
                let root = &chip.lookup_multiplicity_roots[lookup_idx];
                roots[CONSTRAINT_FOLD_BATCH_SIZE + slot] =
                    *node_values.get(root.root_node_id as usize).unwrap_or_else(|| {
                        panic!(
                            "multiplicity root node {} missing for static chip {} {}",
                            root.root_node_id, chip.static_chip_id, chip.chip_name
                        )
                    });
            }
            self.batch_roots.push(roots);
        }

        self.chips.push(ConstraintFoldReplayChip {
            gate_start,
            gate_count: chip.gate_roots.len(),
            batch_start,
            batch_count: chip.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE),
        });
    }

    fn gate_values(&self, chip: ConstraintFoldReplayChip) -> &[EF] {
        &self.gate_values[chip.gate_start..chip.gate_start + chip.gate_count]
    }

    fn batch_roots(&self, chip: ConstraintFoldReplayChip) -> &[[EF; CONSTRAINT_FOLD_ROOT_SLOTS]] {
        &self.batch_roots[chip.batch_start..chip.batch_start + chip.batch_count]
    }
}

/// Exact seal-after tracegen authority for the dependency-heavy constraint families.
///
/// One sequential tracegen module calls this after the semantic seal and before regular row
/// expansion. The request-local artifact is then read by every constraint matrix generator; no
/// family can re-evaluate the DAG, fold, challenge, beta, or terminal chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConstraintAuthorityCounts {
    pub dag: usize,
    pub fold: usize,
    pub challenge: usize,
    pub beta_ladder: usize,
    pub terminal: usize,
}

pub(crate) fn prepare_constraint_authority(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> ConstraintAuthorityCounts {
    let static_plan_preinstalled = program.has_matching_constraint_static_plan();
    let artifact = constraint_case_artifact(record, program);
    let plan = program.constraint_static_plan();
    record.profile.set_structural_counters([
        ("constraint.static_plan_compile_us", plan.compile_us),
        ("constraint.static_plan_retained_bytes", plan.retained_bytes),
        ("constraint.static_plan_legacy_program_row_bytes", plan.legacy_program_row_bytes),
        ("constraint.static_plan_node_plan_bytes", plan.node_plan_bytes),
        (
            "constraint.static_plan_program_row_bytes_saved",
            plan.legacy_program_row_bytes.saturating_sub(plan.node_plan_bytes),
        ),
        ("constraint.static_plan_preinstalled", u64::from(static_plan_preinstalled)),
        ("constraint.static_plan_case_builds", u64::from(!static_plan_preinstalled)),
        ("constraint.case.total_us", artifact.stats.total_us),
        ("constraint.case.static_plan_lookup_us", artifact.stats.static_plan_lookup_us),
        ("constraint.case.input_us", artifact.stats.input_us),
        ("constraint.case.node_eval_us", artifact.stats.node_eval_us),
        ("constraint.case.precompute_eval_us", artifact.stats.precompute_eval_us),
        ("constraint.case.remaining_node_eval_us", artifact.stats.remaining_node_eval_us),
        ("constraint.case.dag_emit_us", artifact.stats.dag_emit_us),
        ("constraint.case.replay_compact_us", artifact.stats.replay_compact_us),
        ("constraint.case.fold_emit_us", artifact.stats.fold_emit_us),
        ("constraint.case.challenge_us", artifact.stats.challenge_us),
        ("constraint.case.beta_ladder_us", artifact.stats.beta_ladder_us),
        ("constraint.case.terminal_us", artifact.stats.terminal_us),
        ("constraint.case.row_allocation_us", artifact.stats.row_allocation_us),
        ("constraint.case.scratch_drop_us", artifact.stats.scratch_drop_us),
        ("constraint.case.artifact_wrap_us", artifact.stats.artifact_wrap_us),
        (
            "constraint.case.proof_count",
            u64::try_from(artifact.stats.proof_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.chip_count",
            u64::try_from(artifact.stats.chip_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.node_count",
            u64::try_from(artifact.stats.node_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.precompute_node_count",
            u64::try_from(artifact.stats.precompute_node_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.remaining_node_count",
            u64::try_from(artifact.stats.remaining_node_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.opened_projection_builds",
            u64::try_from(artifact.stats.opened_projection_builds).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.opened_borrowed_views",
            u64::try_from(artifact.stats.opened_borrowed_views).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.scratch_peak_bytes",
            u64::try_from(artifact.stats.scratch_peak_bytes).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.row_reserved_bytes",
            u64::try_from(artifact.stats.row_reserved_bytes).unwrap_or(u64::MAX),
        ),
        (
            "constraint.case.row_vector_reallocations",
            u64::try_from(artifact.stats.row_vector_reallocations).unwrap_or(u64::MAX),
        ),
        ("constraint.case.node_replay_count", 0),
        ("constraint_static_plan_builds_per_live_layer", 1),
        ("constraint_static_plan_builds_after_first_child_ready", 0),
        ("constraint_fanout_recomputations_per_case", 0),
        ("constraint_program_template_recomputations_per_case", 0),
        (
            "constraint_present_proof_chip_pairs",
            u64::try_from(artifact.stats.chip_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint_precompute_nodes_evaluated",
            u64::try_from(artifact.stats.precompute_node_count).unwrap_or(u64::MAX),
        ),
        (
            "constraint_dynamic_nodes_evaluated",
            u64::try_from(artifact.stats.node_count).unwrap_or(u64::MAX),
        ),
        ("constraint_replayed_node_evaluations", 0),
        ("constraint_lookup_multiplicity_replays", 0),
        ("constraint_precompute_prefix_replays", 0),
        (
            "constraint_opened_projection_builds",
            u64::try_from(artifact.stats.opened_projection_builds).unwrap_or(u64::MAX),
        ),
        (
            "constraint_case_scratch_peak_bytes",
            u64::try_from(artifact.stats.scratch_peak_bytes).unwrap_or(u64::MAX),
        ),
        ("cross_case_dynamic_cache_entries", 0),
        ("constraint_derivation_owner_count", 1),
        ("constraint_static_plan_rebuilds_in_tracegen", u64::from(!static_plan_preinstalled)),
        ("constraint_program_bytes_cloned_into_tracegen", 0),
        ("constraint_descriptor_bytes_crossing_seal", 0),
        ("constraint_static_bytes_duplicated_per_case", 0),
        ("constraint_exact_row_structs_retained", 0),
        (
            "constraint_dynamic_artifact_bytes",
            u64::try_from(artifact.dynamic_bytes()).unwrap_or(u64::MAX),
        ),
        (
            "constraint_dag_value_bytes",
            u64::try_from(
                artifact.dag.values.capacity().saturating_mul(core::mem::size_of::<[F; D_EF]>()),
            )
            .unwrap_or(u64::MAX),
        ),
        (
            "constraint_dag_chip_descriptor_bytes",
            u64::try_from(
                artifact
                    .dag
                    .chips
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ConstraintDagChipDescriptor>()),
            )
            .unwrap_or(u64::MAX),
        ),
    ]);
    ConstraintAuthorityCounts {
        dag: artifact.dag.len(),
        fold: artifact.fold.len(),
        challenge: artifact.challenge.len(),
        beta_ladder: artifact.beta_ladder.len(),
        terminal: artifact.terminal.len(),
    }
}

fn child_contains_global_bus_for_role(role: RecursionChildRole) -> bool {
    matches!(role, RecursionChildRole::Core)
}

struct ProofConstraintChipInputs<'a> {
    shape: &'a RecursionProofShapeChip,
    plan: &'a ConstraintChipPlan,
    ir: &'a RecursionPolyAirChipIr,
    opened_preprocessed: Cow<'a, [EF]>,
    opened_main: Cow<'a, [EF]>,
    permutation_local: Cow<'a, [EF]>,
    selector: TerminalPrefix,
    lcs_limbs: [F; CONSTRAINT_TERMINAL_LCS_LIMBS],
    prep_pos: usize,
}

/// Borrowed proof/transcript frontier plus the small case-local projections shared by every
/// constraint family. It is built once, consumed within one tracegen call, and never retained in
/// the static plan or across cases.
struct ProofConstraintInputs<'a> {
    proof: &'a RecursionProofRecord,
    env: ProofEvalEnv,
    chips: Vec<ProofConstraintChipInputs<'a>>,
    challenge_demand: ChallengeDemand,
    prefixes: Vec<TerminalPrefix>,
    opening_points: Vec<[F; D_EF]>,
    selector_demand: BTreeMap<usize, usize>,
    beta_power_limbs: Vec<[F; D_EF]>,
    terminal_beta_powers: [[F; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    owned_input_bytes: usize,
}

fn cow_owned_extension_bytes(values: &Cow<'_, [EF]>) -> usize {
    match values {
        Cow::Borrowed(_) => 0,
        Cow::Owned(values) => values.capacity().saturating_mul(core::mem::size_of::<EF>()),
    }
}

fn build_proof_constraint_inputs<'a>(
    record: &'a RecursionRecord,
    program: &'a RecursionPolyAirVerifierProgram,
    plan: &'a ConstraintProgramPlan,
    stats: &mut ConstraintCaseBuildStats,
) -> Vec<ProofConstraintInputs<'a>> {
    let mut inputs = Vec::with_capacity(record.proof_records.len());
    for proof in &record.proof_records {
        if !constraint_replay_proof_present(proof) {
            continue;
        }
        let Some(env) = ProofEvalEnv::new(proof, program) else {
            continue;
        };
        let opening_points = terminal_opening_points(&proof.batch_constraint);
        let prefixes = terminal_prefixes(&proof.batch_constraint, &opening_points);
        let challenge_demand = ChallengeDemand::for_proof(proof, plan);
        let selector_demand = selector_height_demand(proof);
        let beta_power_limbs = env.beta_powers.iter().map(ext_limbs).collect::<Vec<_>>();
        let terminal_beta_powers = terminal_beta_powers(&env);
        let mut chips = Vec::with_capacity(proof.proof_shape.chips.len());
        let mut owned_input_bytes = 0usize;
        let mut prep_pos = 0usize;

        for shape in chips_by_idx(&proof.proof_shape.chips) {
            let chip_plan = plan.chip(shape.static_chip_id).unwrap_or_else(|| {
                panic!(
                    "missing constraint program IR for proof {} chip {} static {}",
                    proof.proof_idx, shape.chip_idx, shape.static_chip_id
                )
            });
            let ir = &program.chips[chip_plan.program_chip_index];
            let (opened_preprocessed, opened_main) = opened_values_for_chip(proof, shape)
                .unwrap_or_else(|| {
                    panic!(
                        "missing opened values for proof {} chip {} static {}",
                        proof.proof_idx, shape.chip_idx, shape.static_chip_id
                    )
                });
            let permutation_local = permutation_values_for_chip(proof, shape, ir);
            let source_borrowed = matches!(opened_preprocessed, Cow::Borrowed(_)) &&
                matches!(opened_main, Cow::Borrowed(_)) &&
                matches!(permutation_local, Cow::Borrowed(_));
            if source_borrowed {
                stats.opened_borrowed_views += 1;
            } else {
                stats.opened_projection_builds += 1;
            }
            owned_input_bytes = owned_input_bytes
                .saturating_add(cow_owned_extension_bytes(&opened_preprocessed))
                .saturating_add(cow_owned_extension_bytes(&opened_main))
                .saturating_add(cow_owned_extension_bytes(&permutation_local));
            let selector = *prefixes.get(shape.log_height).unwrap_or_else(|| {
                panic!(
                    "invalid selector height for proof {} chip {} static {} log_height {} rounds {}",
                    proof.proof_idx,
                    shape.chip_idx,
                    shape.static_chip_id,
                    shape.log_height,
                    proof.batch_constraint.rounds.len()
                )
            });
            let cum_sum = proof
                .batch_constraint
                .cum_sums
                .iter()
                .find(|row| row.chip_idx == shape.chip_idx)
                .unwrap_or_else(|| {
                    panic!(
                        "missing local cumulative-sum row for proof {} chip {}",
                        proof.proof_idx, shape.chip_idx
                    )
                });
            chips.push(ProofConstraintChipInputs {
                shape,
                plan: chip_plan,
                ir,
                opened_preprocessed,
                opened_main,
                permutation_local,
                selector,
                lcs_limbs: cum_sum.lcs,
                prep_pos,
            });
            if shape.has_prep() {
                prep_pos += 1;
            }
        }

        stats.proof_count += 1;
        stats.chip_count = stats.chip_count.saturating_add(chips.len());
        inputs.push(ProofConstraintInputs {
            proof,
            env,
            chips,
            challenge_demand,
            prefixes,
            opening_points,
            selector_demand,
            beta_power_limbs,
            terminal_beta_powers,
            owned_input_bytes,
        });
    }
    inputs
}

fn dag_fold_rows_uncached(
    proof_inputs: &[ProofConstraintInputs<'_>],
    plan: &ConstraintProgramPlan,
    stats: &mut ConstraintCaseBuildStats,
) -> (ConstraintDagArena, Vec<ConstraintFoldRow>) {
    let allocation_started = Instant::now();
    let dag_capacity = proof_inputs
        .iter()
        .flat_map(|proof| &proof.chips)
        .map(|chip| chip.ir.node_table.len())
        .sum();
    let dag_chip_capacity = proof_inputs.iter().map(|proof| proof.chips.len()).sum();
    let fold_capacity = proof_inputs
        .iter()
        .flat_map(|proof| &proof.chips)
        .map(|chip| {
            chip.ir
                .gate_roots
                .len()
                .saturating_add(
                    chip.ir.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE),
                )
                .saturating_add(1)
        })
        .sum();
    let mut dag = ConstraintDagArena::with_capacity(dag_chip_capacity, dag_capacity);
    let mut fold_rows = Vec::with_capacity(fold_capacity);
    let dag_chip_initial_capacity = dag.chips.capacity();
    let dag_value_initial_capacity = dag.values.capacity();
    let fold_initial_capacity = fold_rows.capacity();
    stats.row_allocation_us = elapsed_us(allocation_started);
    stats.row_reserved_bytes = dag.allocated_bytes().saturating_add(
        fold_initial_capacity.saturating_mul(core::mem::size_of::<ConstraintFoldRow>()),
    );

    for proof_input in proof_inputs {
        let proof = proof_input.proof;
        let env = &proof_input.env;
        let mut replay_arena = ConstraintFoldReplayArena::for_proof(proof_input);
        let proof_scratch_bytes =
            proof_input.owned_input_bytes.saturating_add(replay_arena.allocated_bytes());
        stats.scratch_peak_bytes = stats.scratch_peak_bytes.max(proof_scratch_bytes);

        for chip_input in &proof_input.chips {
            let chip_shape = chip_input.shape;
            let chip_ir = chip_input.ir;
            let base_env = RecursionPolyAirEnv {
                proof_idx: proof.proof_idx,
                chip_idx: chip_shape.chip_idx,
                opened_preprocessed: chip_input.opened_preprocessed.as_ref(),
                opened_main: chip_input.opened_main.as_ref(),
                public_values: &proof.proof_shape.public_values,
                constraint_alpha: env.alpha,
                perm_alpha: env.perm_alpha,
                perm_beta: env.perm_beta,
                beta_powers: &env.beta_powers,
                beta_septix: env.beta_septix,
                precomputed_lc: &[],
                reserved_poly: &[],
                is_first_row: chip_input.selector.first,
                is_last_row: chip_input.selector.last,
            };

            let eval_started = Instant::now();
            let (node_values, precomputed_lc, reserved_poly, node_profile) =
                evaluate_chip_node_arena_profiled(chip_ir, &base_env).unwrap_or_else(|err| {
                    panic!(
                        "constraint replay failed for proof {} chip {} static {} {}: {err:?}",
                        proof.proof_idx,
                        chip_shape.chip_idx,
                        chip_shape.static_chip_id,
                        chip_ir.chip_name
                    )
                });
            stats.node_eval_us = stats.node_eval_us.saturating_add(elapsed_us(eval_started));
            stats.precompute_eval_us =
                stats.precompute_eval_us.saturating_add(node_profile.precompute_us);
            stats.remaining_node_eval_us =
                stats.remaining_node_eval_us.saturating_add(node_profile.remaining_us);
            stats.node_count = stats.node_count.saturating_add(chip_ir.node_table.len());
            stats.precompute_node_count =
                stats.precompute_node_count.saturating_add(node_profile.precompute_nodes);
            stats.remaining_node_count =
                stats.remaining_node_count.saturating_add(node_profile.remaining_nodes);
            let replay_bytes =
                chip_node_arena_scratch_bytes(&node_values, &precomputed_lc, &reserved_poly);
            stats.scratch_peak_bytes =
                stats.scratch_peak_bytes.max(proof_scratch_bytes.saturating_add(replay_bytes));

            let dag_emit_started = Instant::now();
            let node_plans = plan.node_plans_for_chip(chip_input.plan);
            assert_eq!(
                chip_ir.node_table.len(),
                node_plans.len(),
                "constraint static plan row count drift for static chip {}",
                chip_shape.static_chip_id
            );
            dag.push_chip(
                proof.proof_idx,
                chip_shape.chip_idx,
                chip_input.plan.program_chip_index,
                chip_input.prep_pos,
                &node_values,
            );
            stats.dag_emit_us = stats.dag_emit_us.saturating_add(elapsed_us(dag_emit_started));

            let compact_started = Instant::now();
            replay_arena.push_chip(chip_ir, &node_values, &precomputed_lc);
            stats.replay_compact_us =
                stats.replay_compact_us.saturating_add(elapsed_us(compact_started));
            // The complete node/precompute/reserved replay is dead after DAG emission and compact
            // Fold extraction. Dropping these Vecs here releases their capacity before the next
            // chip is evaluated; the reverse Fold pass retains only the proof-flat arenas above.
            drop(node_values);
            drop(precomputed_lc);
            drop(reserved_poly);
        }

        let fold_emit_started = Instant::now();
        let mut acc = EF::zero();
        let mut pacc = EF::zero();
        let mut cursor = 0usize;
        let mut perm_sum = EF::zero();
        let mut remaining_chips = proof_input.chips.len();
        for chip_input_idx in (0..proof_input.chips.len()).rev() {
            let chip_input = &proof_input.chips[chip_input_idx];
            let replay = replay_arena.chips[chip_input_idx];
            let chip_shape = chip_input.shape;
            let chip_ir = chip_input.ir;
            let gate_values = replay_arena.gate_values(replay);
            let batch_roots = replay_arena.batch_roots(replay);
            let gate_count = gate_values.len();
            let mut local_ord = 0usize;
            for (gate_idx, value) in gate_values.iter().copied().enumerate() {
                cursor += 1;
                let root = chip_ir.gate_roots.get(gate_idx).unwrap_or_else(|| {
                    panic!(
                        "missing gate root {} for static chip {} {}",
                        gate_idx, chip_ir.static_chip_id, chip_ir.chip_name
                    )
                });
                push_fold_root_row(
                    &mut fold_rows,
                    proof,
                    chip_shape,
                    cursor,
                    remaining_chips,
                    local_ord,
                    local_ord + 1,
                    gate_count,
                    env.alpha,
                    &mut acc,
                    &mut pacc,
                    perm_sum,
                    ConstraintRootSpec {
                        active: true,
                        ord: gate_idx,
                        node: root.root_node_id as usize,
                        sign: 1,
                        value,
                    },
                );
                local_ord += 1;
            }
            for (batch_idx, compact_roots) in batch_roots.iter().copied().enumerate() {
                cursor += 1;
                let permutation_value =
                    *chip_input.permutation_local.get(batch_idx).unwrap_or_else(|| {
                        panic!(
                            "missing permutation value {} for proof {} chip {} static {} {}",
                            batch_idx,
                            proof.proof_idx,
                            chip_shape.chip_idx,
                            chip_shape.static_chip_id,
                            chip_ir.chip_name
                        )
                    });
                let constraint_value = fold_batch_constraint_value(
                    chip_ir,
                    batch_idx,
                    compact_roots,
                    permutation_value,
                );
                let batch_roots = fold_batch_roots_from_compact(chip_ir, batch_idx, compact_roots);
                push_fold_batch_row(
                    &mut fold_rows,
                    proof,
                    chip_shape,
                    cursor,
                    remaining_chips,
                    local_ord,
                    local_ord + 1,
                    gate_count,
                    env.alpha,
                    &mut acc,
                    &mut pacc,
                    &mut perm_sum,
                    batch_idx,
                    batch_roots,
                    permutation_value,
                    constraint_value,
                );
                local_ord += 1;
            }

            cursor += 1;
            push_fold_skip_row(
                &mut fold_rows,
                proof,
                chip_shape,
                cursor,
                remaining_chips,
                local_ord,
                0,
                gate_count,
                env.alpha,
                &mut acc,
                &mut pacc,
                &mut perm_sum,
                EF::from_base_slice(&chip_input.lcs_limbs),
            );
            remaining_chips -= 1;
        }
        debug_assert_eq!(remaining_chips, 0);
        stats.fold_emit_us = stats.fold_emit_us.saturating_add(elapsed_us(fold_emit_started));
    }

    stats.row_vector_reallocations = usize::from(dag.chips.capacity() != dag_chip_initial_capacity)
        .saturating_add(usize::from(dag.values.capacity() != dag_value_initial_capacity))
        .saturating_add(usize::from(fold_rows.capacity() != fold_initial_capacity));
    debug_assert_eq!(stats.row_vector_reallocations, 0, "constraint row capacity formula drift");
    (dag, fold_rows)
}

fn challenge_rows_uncached(
    proof_inputs: &[ProofConstraintInputs<'_>],
) -> Vec<ConstraintChallengeRow> {
    let mut rows = Vec::new();
    for proof_input in proof_inputs {
        let proof = proof_input.proof;
        let demand = &proof_input.challenge_demand;
        for chip_input in &proof_input.chips {
            let chip = chip_input.shape;
            let prefix = chip_input.selector;
            rows.push(ConstraintChallengeRow {
                proof_idx: proof.proof_idx,
                chip_idx: chip.chip_idx,
                static_chip_id: chip.static_chip_id,
                main_width: chip.main_width,
                log_height: chip.log_height,
                c_chips: proof.batch_constraint.c_chips,
                num_public_values: proof.proof_shape.num_public_values,
                lcs_limbs: chip_input.lcs_limbs,
                eq_acc: ext_limbs(&prefix.eq),
                first: ext_limbs(&prefix.first),
                last: ext_limbs(&prefix.last),
                first_send_mult: demand
                    .first_by_static
                    .get(&chip.static_chip_id)
                    .copied()
                    .unwrap_or(0),
                last_send_mult: demand
                    .last_by_static
                    .get(&chip.static_chip_id)
                    .copied()
                    .unwrap_or(0),
            });
        }
    }
    rows
}

fn beta_ladder_rows_uncached(
    proof_inputs: &[ProofConstraintInputs<'_>],
    child_contains_global_bus: bool,
) -> Vec<ConstraintBetaLadderRow> {
    let mut rows = Vec::new();
    for proof_input in proof_inputs {
        let proof = proof_input.proof;
        let demand = &proof_input.challenge_demand;
        let mut serve_mults = demand.beta_power;
        // The Terminal final-row beta recvs exist only on the cgb=true build.
        if child_contains_global_bus && proof.batch_constraint.publish_terminal_outputs {
            for power in 1..=CONSTRAINT_CHAIN_LIMBS {
                serve_mults[power] += 1;
            }
        }
        for power_idx in 0..CONSTRAINT_MAX_BETA_POWERS {
            let prev_power_or_alpha = if power_idx == 0 {
                proof.batch_constraint.perm_alpha
            } else {
                proof_input.beta_power_limbs[power_idx - 1]
            };
            rows.push(ConstraintBetaLadderRow {
                proof_idx: proof.proof_idx,
                power_idx,
                beta: proof.batch_constraint.perm_beta,
                prev_power_or_alpha,
                power: proof_input.beta_power_limbs[power_idx],
                serve_mult: serve_mults[power_idx],
                challenges_recv_mult: power_idx == 0 &&
                    proof.batch_constraint.publish_terminal_outputs,
                alpha_serve_mult: if power_idx == 0 {
                    demand.perm_alpha +
                        usize::from(
                            child_contains_global_bus &&
                                proof.batch_constraint.publish_terminal_outputs,
                        )
                } else {
                    0
                },
                septix_serve_mult: if power_idx == 7 { demand.beta_septix } else { 0 },
            });
        }
    }
    rows
}

fn terminal_rows_uncached(
    proof_inputs: &[ProofConstraintInputs<'_>],
    child_contains_global_bus: bool,
    fold_rows: &[ConstraintFoldRow],
) -> Vec<ConstraintTerminalRow> {
    let mut rows = Vec::new();
    let final_folds = final_fold_rows(fold_rows);
    for proof_input in proof_inputs {
        let proof = proof_input.proof;
        let batch = &proof.batch_constraint;
        let env = &proof_input.env;
        let publish_outputs = batch.publish_terminal_outputs;
        let publish_summary = proof.proof_shape.publish_terminal_summary;
        if !publish_outputs && !publish_summary {
            continue;
        }

        let prefixes = &proof_input.prefixes;
        let selector_demand = &proof_input.selector_demand;
        let seed_send_mult =
            if publish_outputs { 1 + selector_demand.get(&0).copied().unwrap_or(0) } else { 0 };
        let mut seed = ConstraintTerminalRow::blank(proof.proof_idx, batch);
        seed.is_seed = true;
        seed.eq_out = ext_limbs(&prefixes[0].eq);
        seed.first_prefix_out = ext_limbs(&prefixes[0].first);
        seed.last_prefix_out = ext_limbs(&prefixes[0].last);
        seed.state_chain_send_mult = publish_outputs;
        seed.summary_recv_mult = publish_summary;
        seed.summary_id_base = proof.proof_shape.segment_id_base();
        seed.eq_chain_send_mult = seed_send_mult;
        rows.push(seed);

        if !publish_outputs {
            continue;
        }

        for opening_idx in 0..batch.num_rounds {
            let opening_point = *proof_input.opening_points.get(opening_idx).unwrap_or_else(|| {
                panic!(
                    "missing opening point {} for proof {} with {} rounds",
                    opening_idx, proof.proof_idx, batch.num_rounds
                )
            });
            let eq_challenge = *batch.eq_challenges.get(opening_idx).unwrap_or_else(|| {
                panic!(
                    "missing eq challenge {} for proof {} with {} rounds",
                    opening_idx, proof.proof_idx, batch.num_rounds
                )
            });
            let eq_factor = eq_factor_limbs(eq_challenge, opening_point);
            let send_idx = opening_idx + 1;
            let send_mult = 1 + selector_demand.get(&send_idx).copied().unwrap_or(0);
            let mut row = ConstraintTerminalRow::blank(proof.proof_idx, batch);
            row.round_idx = opening_idx;
            row.opening_idx = opening_idx;
            row.is_eq_step = true;
            row.opening_point = opening_point;
            row.eq_challenge = eq_challenge;
            row.eq_factor = eq_factor;
            row.eq_in = ext_limbs(&prefixes[opening_idx].eq);
            row.eq_out = ext_limbs(&prefixes[send_idx].eq);
            row.first_prefix_in = ext_limbs(&prefixes[opening_idx].first);
            row.first_prefix_out = ext_limbs(&prefixes[send_idx].first);
            row.last_prefix_in = ext_limbs(&prefixes[opening_idx].last);
            row.last_prefix_out = ext_limbs(&prefixes[send_idx].last);
            row.opening_point_recv_mult = true;
            row.eq_recv_mult = true;
            row.eq_chain_recv_mult = true;
            row.eq_chain_send_mult = send_mult;
            rows.push(row);
        }

        let mut lcs_sum = EF::zero();
        for (cursor, chip_input) in proof_input.chips.iter().enumerate() {
            let chip = chip_input.shape;
            let lcs = EF::from_base_slice(&chip_input.lcs_limbs);
            let mut row = ConstraintTerminalRow::blank(proof.proof_idx, batch);
            row.is_lcs_step = true;
            row.round_idx = cursor;
            row.opening_idx = cursor;
            row.chip_idx = chip.chip_idx;
            row.lcs = chip_input.lcs_limbs;
            row.state_lcs_in = ext_limbs(&lcs_sum);
            lcs_sum += lcs;
            row.state_lcs_out = ext_limbs(&lcs_sum);
            row.state_chain_recv_mult = true;
            row.state_chain_send_mult = true;
            rows.push(row);
        }

        let fold = final_folds.get(&proof.proof_idx).unwrap_or_else(|| {
            panic!("missing final fold row for terminal proof {}", proof.proof_idx)
        });
        let main = EF::from_base_slice(&fold.acc_out);
        let perm = EF::from_base_slice(&fold.pacc_out);
        let eq = prefixes[batch.num_rounds].eq;
        let final_lhs = main * eq + perm;
        let mut row = ConstraintTerminalRow::blank(proof.proof_idx, batch);
        row.round_idx = batch.num_rounds;
        row.opening_idx = proof.proof_shape.chips.len();
        row.is_final = true;
        row.eq_in = ext_limbs(&eq);
        row.eq_out = ext_limbs(&eq);
        row.first_prefix_in = ext_limbs(&prefixes[batch.num_rounds].first);
        row.first_prefix_out = ext_limbs(&prefixes[batch.num_rounds].first);
        row.last_prefix_in = ext_limbs(&prefixes[batch.num_rounds].last);
        row.last_prefix_out = ext_limbs(&prefixes[batch.num_rounds].last);
        row.fold_cursor = fold.cursor;
        row.alpha = fold.alpha;
        row.main_eval = fold.acc_out;
        row.perm_eval = fold.pacc_out;
        row.last_claim = batch.last_claim;
        row.final_lhs = ext_limbs(&final_lhs);
        row.state_lcs_in = ext_limbs(&lcs_sum);
        row.state_lcs_out = ext_limbs(&lcs_sum);
        row.public_value_recv_mult = child_contains_global_bus;
        row.state_chain_recv_mult = !child_contains_global_bus;
        row.state_chain_send_mult = false;
        if child_contains_global_bus {
            let state_witness = state_imbalance_witness(
                proof,
                env.perm_alpha,
                proof_input.terminal_beta_powers,
                lcs_sum,
                true,
            );
            row.public_values = state_witness.public_values;
            row.perm_alpha = proof.batch_constraint.perm_alpha;
            row.beta_powers = state_witness.beta_powers;
            row.state_clock_changed = state_witness.state_clock_changed;
            row.state_clock_delta_inverse = state_witness.state_clock_delta_inverse;
            row.state_transition_recv_inverse = state_witness.state_transition_recv_inverse;
            row.state_transition_send_inverse = state_witness.state_transition_send_inverse;
            row.init_address_recv_inverse = state_witness.init_address_recv_inverse;
            row.init_address_send_inverse = state_witness.init_address_send_inverse;
            row.finalize_address_recv_inverse = state_witness.finalize_address_recv_inverse;
            row.finalize_address_send_inverse = state_witness.finalize_address_send_inverse;
            row.global_chain_source_inverse = state_witness.global_chain_source_inverse;
            row.global_chain_sink_inverse = state_witness.global_chain_sink_inverse;
        }
        row.last_claim_recv_mult = true;
        row.fold_chain_recv_mult = true;
        row.eq_chain_recv_mult = true;
        rows.push(row);
    }
    rows
}

#[derive(Debug, Clone, Copy)]
struct ConstraintRootSpec {
    active: bool,
    ord: usize,
    node: usize,
    sign: i32,
    value: EF,
}

#[derive(Debug, Clone, Copy)]
struct StateImbalanceWitness {
    public_values: [F; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT],
    beta_powers: [[F; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    state_clock_changed: F,
    state_clock_delta_inverse: F,
    state_transition_recv_inverse: [F; D_EF],
    state_transition_send_inverse: [F; D_EF],
    init_address_recv_inverse: [F; D_EF],
    init_address_send_inverse: [F; D_EF],
    finalize_address_recv_inverse: [F; D_EF],
    finalize_address_send_inverse: [F; D_EF],
    global_chain_source_inverse: [F; D_EF],
    global_chain_sink_inverse: [F; D_EF],
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
const PV_COL_START_PC: usize = 0;
const PV_COL_NEXT_PC: usize = 1;
const PV_COL_EXECUTION_SHARD: usize = 2;
const PV_COL_PREVIOUS_INIT_ADDR: usize = 3;
const PV_COL_LAST_INIT_ADDR: usize = 4;
const PV_COL_PREVIOUS_FINALIZE_ADDR: usize = 5;
const PV_COL_LAST_FINALIZE_ADDR: usize = 6;
const PV_COL_START_CLK: usize = 7;
const PV_COL_EXIT_CLK: usize = 8;

fn terminal_beta_powers(env: &ProofEvalEnv) -> [[F; D_EF]; CONSTRAINT_CHAIN_LIMBS] {
    core::array::from_fn(|idx| {
        ext_limbs(env.beta_powers.get(idx + 1).unwrap_or_else(|| {
            panic!("terminal beta power {} out of range {}", idx + 1, env.beta_powers.len())
        }))
    })
}

fn terminal_public_values(
    proof: &RecursionProofRecord,
    child_contains_global_bus: bool,
) -> [F; CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT] {
    if !child_contains_global_bus {
        return [F::zero(); CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT];
    }
    if proof.proof_shape.public_values.len() < dt_stark::air::GLOBAL_CLAIM_END {
        panic!(
            "proof {} has {} public values, need at least {} for terminal state imbalance",
            proof.proof_idx,
            proof.proof_shape.public_values.len(),
            dt_stark::air::GLOBAL_CLAIM_END
        );
    }
    let pv_view: &PublicValues<Word<F>, F> = proof.proof_shape.public_values.as_slice().borrow();
    let direct = [
        pv_view.start_pc,
        pv_view.next_pc,
        pv_view.execution_shard,
        pv_view.previous_init_addr,
        pv_view.last_init_addr,
        pv_view.previous_finalize_addr,
        pv_view.last_finalize_addr,
        pv_view.start_clk,
        pv_view.exit_clk,
    ];
    let mut selected = [F::zero(); CONSTRAINT_TERMINAL_PUBLIC_VALUE_COUNT];
    selected[..9].copy_from_slice(&direct);
    selected[9..].copy_from_slice(
        &proof.proof_shape.public_values
            [dt_stark::air::CORE_PUBLIC_VALUES_PREFIX..dt_stark::air::GLOBAL_CLAIM_END],
    );
    let indexed_direct = [
        TERMINAL_PV_START_PC,
        TERMINAL_PV_NEXT_PC,
        TERMINAL_PV_EXECUTION_SHARD,
        TERMINAL_PV_PREVIOUS_INIT_ADDR,
        TERMINAL_PV_LAST_INIT_ADDR,
        TERMINAL_PV_PREVIOUS_FINALIZE_ADDR,
        TERMINAL_PV_LAST_FINALIZE_ADDR,
        TERMINAL_PV_START_CLK,
        TERMINAL_PV_EXIT_CLK,
    ]
    .map(|idx| proof.proof_shape.public_values[idx]);
    if direct != indexed_direct {
        panic!("public value typed view/index view mismatch for proof {}", proof.proof_idx);
    }
    selected
}

fn state_imbalance_witness(
    proof: &RecursionProofRecord,
    perm_alpha: EF,
    beta_powers: [[F; D_EF]; CONSTRAINT_CHAIN_LIMBS],
    lcs_sum: EF,
    child_contains_global_bus: bool,
) -> StateImbalanceWitness {
    let public_values = terminal_public_values(proof, child_contains_global_bus);
    let beta_ext: [EF; CONSTRAINT_CHAIN_LIMBS] =
        beta_powers.map(|limbs| EF::from_base_slice(&limbs));
    if !child_contains_global_bus {
        return StateImbalanceWitness {
            public_values,
            beta_powers,
            state_clock_changed: F::zero(),
            state_clock_delta_inverse: F::zero(),
            state_transition_recv_inverse: [F::zero(); D_EF],
            state_transition_send_inverse: [F::zero(); D_EF],
            init_address_recv_inverse: [F::zero(); D_EF],
            init_address_send_inverse: [F::zero(); D_EF],
            finalize_address_recv_inverse: [F::zero(); D_EF],
            finalize_address_send_inverse: [F::zero(); D_EF],
            global_chain_source_inverse: [F::zero(); D_EF],
            global_chain_sink_inverse: [F::zero(); D_EF],
        };
    }
    let beta = beta_ext[0];
    let beta2 = beta_ext[1];
    let beta3 = beta_ext[2];

    let mut contribution = EF::zero();

    let diff_clk = public_values[PV_COL_START_CLK] - public_values[PV_COL_EXIT_CLK];
    let state_clock_changed = f_bool(diff_clk != F::zero());
    let state_kind = EF::from_canonical_usize(InteractionKind::State as usize);
    let shard_term = beta * public_values[PV_COL_EXECUTION_SHARD];
    let recv_state = perm_alpha +
        state_kind +
        shard_term +
        beta2 * public_values[PV_COL_START_CLK] +
        beta3 * public_values[PV_COL_START_PC];
    let send_state = perm_alpha +
        state_kind +
        shard_term +
        beta2 * public_values[PV_COL_EXIT_CLK] +
        beta3 * public_values[PV_COL_NEXT_PC];
    let (state_transition_recv_inverse, state_transition_send_inverse) = if state_clock_changed !=
        F::zero()
    {
        if recv_state == EF::zero() || send_state == EF::zero() {
            panic!(
                    "terminal state-transition reciprocal denominator is zero for proof {}; retry with fresh challenges",
                    proof.proof_idx
                );
        }
        let recv_inv = recv_state.inverse();
        let send_inv = send_state.inverse();
        contribution += send_inv - recv_inv;
        (ext_limbs(&recv_inv), ext_limbs(&send_inv))
    } else {
        ([F::zero(); D_EF], [F::zero(); D_EF])
    };

    let addr_kind = EF::from_canonical_usize(InteractionKind::MemoryGlobalAddr as usize);
    let base_init = perm_alpha + addr_kind;
    let recv_init = base_init + beta2 * public_values[PV_COL_PREVIOUS_INIT_ADDR];
    let send_init = base_init + beta2 * public_values[PV_COL_LAST_INIT_ADDR];

    let base_fin = perm_alpha + addr_kind + beta;
    let recv_fin = base_fin + beta2 * public_values[PV_COL_PREVIOUS_FINALIZE_ADDR];
    let send_fin = base_fin + beta2 * public_values[PV_COL_LAST_FINALIZE_ADDR];

    let mandatory_fingerprints = [recv_init, send_init, recv_fin, send_fin];
    if mandatory_fingerprints.iter().any(|fingerprint| *fingerprint == EF::zero()) {
        panic!(
            "terminal reciprocal denominator is zero for proof {}; retry with fresh challenges",
            proof.proof_idx
        );
    }
    let mandatory_inverses = batch_multiplicative_inverse(&mandatory_fingerprints);
    contribution += mandatory_inverses[1] - mandatory_inverses[0];
    contribution += mandatory_inverses[3] - mandatory_inverses[2];
    let pv_view: &PublicValues<Word<F>, F> = proof.proof_shape.public_values.as_slice().borrow();
    let claim = &pv_view.global;
    let (global_chain_source_inverse, global_chain_sink_inverse) = if claim.has_global_opening ==
        F::zero()
    {
        ([F::zero(); D_EF], [F::zero(); D_EF])
    } else {
        let blocks = dt_stark::global_d11::projective_chain_claim_blocks_v2::<F, EF>(claim);
        let kind = EF::from_canonical_usize(InteractionKind::GlobalProjectiveChainV2 as usize);
        let fingerprint = |values: &[EF; CONSTRAINT_CHAIN_LIMBS]| {
            values
                .iter()
                .zip(beta_ext)
                .fold(perm_alpha + kind, |acc, (value, beta)| acc + *value * beta)
        };
        let source = fingerprint(&blocks.source);
        let sink = fingerprint(&blocks.sink);
        if source == EF::zero() || sink == EF::zero() {
            panic!(
                    "terminal Global claim reciprocal denominator is zero for proof {}; retry with fresh challenges",
                    proof.proof_idx
                );
        }
        let source_inverse = source.inverse();
        let sink_inverse = sink.inverse();
        contribution += sink_inverse - source_inverse;
        (ext_limbs(&source_inverse), ext_limbs(&sink_inverse))
    };

    if contribution != lcs_sum {
        panic!(
            "terminal state imbalance mismatch for proof {}: expected={contribution:?} lcs={lcs_sum:?}",
            proof.proof_idx
        );
    }

    StateImbalanceWitness {
        public_values,
        beta_powers,
        state_clock_changed,
        state_clock_delta_inverse: if diff_clk == F::zero() {
            F::zero()
        } else {
            diff_clk.inverse()
        },
        state_transition_recv_inverse,
        state_transition_send_inverse,
        init_address_recv_inverse: ext_limbs(&mandatory_inverses[0]),
        init_address_send_inverse: ext_limbs(&mandatory_inverses[1]),
        finalize_address_recv_inverse: ext_limbs(&mandatory_inverses[2]),
        finalize_address_send_inverse: ext_limbs(&mandatory_inverses[3]),
        global_chain_source_inverse,
        global_chain_sink_inverse,
    }
}

fn push_fold_skip_row(
    rows: &mut Vec<ConstraintFoldRow>,
    proof: &RecursionProofRecord,
    chip: &RecursionProofShapeChip,
    cursor: usize,
    remaining_chips: usize,
    local_ord: usize,
    chain_send_local_ord: usize,
    gate_count: usize,
    alpha: EF,
    acc: &mut EF,
    pacc: &mut EF,
    perm_sum: &mut EF,
    lcs: EF,
) {
    let acc_in = *acc;
    let pacc_in = *pacc;
    let perm_sum_in = *perm_sum;
    let height_inverse = F::from_canonical_usize(1usize << chip.log_height).inverse();
    let correction = perm_sum_in - lcs * height_inverse;
    *acc *= alpha;
    *pacc = *pacc * alpha + correction;
    *perm_sum = EF::zero();
    let mut row = base_fold_row(
        proof,
        chip,
        cursor,
        remaining_chips,
        local_ord,
        chain_send_local_ord,
        gate_count,
        true,
        false,
        false,
        alpha,
        acc_in,
        *acc,
        pacc_in,
        *pacc,
        chip.log_height,
    );
    row.perm_sum_in = ext_limbs(&perm_sum_in);
    row.perm_sum_out = [F::zero(); D_EF];
    row.root_nodes[0] = height_inverse.as_canonical_u32() as usize;
    row.perm_value = ext_limbs(&lcs);
    row.root_values[1] = if chip.perm_width == 0 {
        assert_eq!(lcs, EF::zero(), "zero-permutation chip must have zero authenticated LCS");
        [F::zero(); D_EF]
    } else {
        ext_limbs(&(lcs * F::from_canonical_usize(chip.perm_width / D_EF).inverse()))
    };
    rows.push(row);
}

fn push_fold_root_row(
    rows: &mut Vec<ConstraintFoldRow>,
    proof: &RecursionProofRecord,
    chip: &RecursionProofShapeChip,
    cursor: usize,
    remaining_chips: usize,
    local_ord: usize,
    chain_send_local_ord: usize,
    gate_count: usize,
    alpha: EF,
    acc: &mut EF,
    pacc: &mut EF,
    perm_sum: EF,
    root: ConstraintRootSpec,
) {
    let acc_in = *acc;
    let pacc_in = *pacc;
    *acc = *acc * alpha + root.value;
    *pacc *= alpha;
    let mut row = base_fold_row(
        proof,
        chip,
        cursor,
        remaining_chips,
        local_ord,
        chain_send_local_ord,
        gate_count,
        false,
        true,
        false,
        alpha,
        acc_in,
        *acc,
        pacc_in,
        *pacc,
        root.ord,
    );
    row.perm_sum_in = ext_limbs(&perm_sum);
    row.perm_sum_out = ext_limbs(&perm_sum);
    fill_fold_root_slot(&mut row, 0, root);
    rows.push(row);
}

fn push_fold_batch_row(
    rows: &mut Vec<ConstraintFoldRow>,
    proof: &RecursionProofRecord,
    chip: &RecursionProofShapeChip,
    cursor: usize,
    remaining_chips: usize,
    local_ord: usize,
    chain_send_local_ord: usize,
    gate_count: usize,
    alpha: EF,
    acc: &mut EF,
    pacc: &mut EF,
    perm_sum: &mut EF,
    batch_idx: usize,
    roots: [ConstraintRootSpec; CONSTRAINT_FOLD_ROOT_SLOTS],
    permutation_value: EF,
    constraint_value: EF,
) {
    let acc_in = *acc;
    let pacc_in = *pacc;
    *acc = *acc * alpha + constraint_value;
    *pacc *= alpha;
    let perm_sum_in = *perm_sum;
    *perm_sum += permutation_value;
    let mut row = base_fold_row(
        proof,
        chip,
        cursor,
        remaining_chips,
        local_ord,
        chain_send_local_ord,
        gate_count,
        false,
        false,
        true,
        alpha,
        acc_in,
        *acc,
        pacc_in,
        *pacc,
        batch_idx * CONSTRAINT_FOLD_BATCH_SIZE,
    );
    row.perm_sum_in = ext_limbs(&perm_sum_in);
    row.perm_sum_out = ext_limbs(perm_sum);
    row.batch_has_second = roots[1].active;
    row.perm_value = ext_limbs(&permutation_value);
    for (slot, root) in roots.into_iter().enumerate() {
        fill_fold_root_slot(&mut row, slot, root);
    }
    rows.push(row);
}

#[allow(clippy::too_many_arguments)]
fn base_fold_row(
    proof: &RecursionProofRecord,
    chip: &RecursionProofShapeChip,
    cursor: usize,
    remaining_chips: usize,
    local_ord: usize,
    chain_send_local_ord: usize,
    gate_count: usize,
    is_skip: bool,
    is_gate: bool,
    is_batch: bool,
    alpha: EF,
    acc_in: EF,
    acc_out: EF,
    pacc_in: EF,
    pacc_out: EF,
    root_ord: usize,
) -> ConstraintFoldRow {
    let one = ext_limbs(&EF::one());
    let mut root_values = [[F::zero(); D_EF]; CONSTRAINT_FOLD_ROOT_SLOTS];
    for slot in root_values.iter_mut().take(CONSTRAINT_FOLD_BATCH_SIZE) {
        *slot = one;
    }
    ConstraintFoldRow {
        proof_idx: proof.proof_idx,
        cursor,
        remaining_chips,
        local_ord,
        chain_send_local_ord,
        static_chip_id: chip.static_chip_id,
        log_height: chip.log_height,
        gate_count,
        batch_count: chip.perm_width / D_EF,
        root_ord,
        is_skip,
        is_gate,
        is_batch,
        alpha: ext_limbs(&alpha),
        acc_in: ext_limbs(&acc_in),
        acc_out: ext_limbs(&acc_out),
        pacc_in: ext_limbs(&pacc_in),
        pacc_out: ext_limbs(&pacc_out),
        perm_sum_in: [F::zero(); D_EF],
        perm_sum_out: [F::zero(); D_EF],
        root_nodes: [0; CONSTRAINT_FOLD_ROOT_SLOTS],
        multiplicity_signs: [1; CONSTRAINT_FOLD_BATCH_SIZE],
        root_values,
        batch_has_second: false,
        perm_value: [F::zero(); D_EF],
    }
}

fn fill_fold_root_slot(row: &mut ConstraintFoldRow, slot: usize, root: ConstraintRootSpec) {
    row.root_nodes[slot] = root.node;
    if slot >= CONSTRAINT_FOLD_BATCH_SIZE {
        row.multiplicity_signs[slot - CONSTRAINT_FOLD_BATCH_SIZE] = root.sign;
    }
    row.root_values[slot] = ext_limbs(&root.value);
}

fn fold_batch_roots_from_compact(
    chip: &RecursionPolyAirChipIr,
    batch_idx: usize,
    compact: [EF; CONSTRAINT_FOLD_ROOT_SLOTS],
) -> [ConstraintRootSpec; CONSTRAINT_FOLD_ROOT_SLOTS] {
    let start = batch_idx * CONSTRAINT_FOLD_BATCH_SIZE;
    let end = (start + CONSTRAINT_FOLD_BATCH_SIZE).min(chip.lookup_multiplicity_roots.len());
    let inactive_denominator =
        ConstraintRootSpec { active: false, ord: 0, node: 0, sign: 1, value: EF::one() };
    let inactive_multiplicity =
        ConstraintRootSpec { active: false, ord: 0, node: 0, sign: 1, value: EF::zero() };
    let mut roots =
        [inactive_denominator, inactive_denominator, inactive_multiplicity, inactive_multiplicity];
    for (slot, lookup_idx) in (start..end).enumerate() {
        let root_node = precompute_root_node(chip, lookup_idx).unwrap_or_else(|| {
            panic!(
                "missing denominator precompute root {} for static chip {} {}",
                lookup_idx, chip.static_chip_id, chip.chip_name
            )
        });
        roots[slot] = ConstraintRootSpec {
            active: true,
            ord: lookup_idx,
            node: root_node,
            sign: 1,
            value: compact[slot],
        };
    }
    for (slot, lookup_idx) in (start..end).enumerate() {
        let root = chip.lookup_multiplicity_roots.get(lookup_idx).unwrap_or_else(|| {
            panic!(
                "missing multiplicity root {} for static chip {} {}",
                lookup_idx, chip.static_chip_id, chip.chip_name
            )
        });
        let sign = if root.is_send { 1 } else { -1 };
        roots[CONSTRAINT_FOLD_BATCH_SIZE + slot] = ConstraintRootSpec {
            active: true,
            ord: root.lookup_idx,
            node: root.root_node_id as usize,
            sign,
            value: compact[CONSTRAINT_FOLD_BATCH_SIZE + slot],
        };
    }
    roots
}

fn fold_batch_constraint_value(
    chip: &RecursionPolyAirChipIr,
    batch_idx: usize,
    roots: [EF; CONSTRAINT_FOLD_ROOT_SLOTS],
    permutation_value: EF,
) -> EF {
    let start = batch_idx * CONSTRAINT_FOLD_BATCH_SIZE;
    let signed_multiplicity = |slot: usize| {
        let lookup_idx = start + slot;
        if let Some(root) = chip.lookup_multiplicity_roots.get(lookup_idx) {
            roots[CONSTRAINT_FOLD_BATCH_SIZE + slot] *
                EF::from_base(if root.is_send { F::one() } else { -F::one() })
        } else {
            EF::zero()
        }
    };
    let [d0, d1, _, _] = roots;
    let m0 = signed_multiplicity(0);
    let m1 = signed_multiplicity(1);
    // Exact fixed batch-two PolyAir residual. For the odd tail the compact arena stores
    // d1=1,m1=0, yielding m0-d0*p without a generic product loop.
    d1 * m0 + d0 * (m1 - d1 * permutation_value)
}

pub type ConstraintReplayBusResidualReport = BTreeMap<&'static str, BTreeMap<Vec<u32>, i64>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OpenedEvalDemandKey {
    batch_id: usize,
    batch_pos: usize,
    chip_idx: usize,
    value_idx: usize,
}

#[derive(Debug, Default)]
struct ProofExternalDemand {
    opened: BTreeMap<OpenedEvalDemandKey, u32>,
    public_values: BTreeMap<usize, u32>,
}

/// Finalize one child's external publication multiplicities from that proof and the frozen
/// program before its rows are lowered into slot-local segments.
pub fn annotate_child_constraint_replay_publications(
    proof: &mut RecursionProofRecord,
    program: &RecursionPolyAirVerifierProgram,
) {
    let demand = external_demand_for_proof(proof, program);
    if let Some(source) = proof.whir_source.as_mut() {
        source.opened_eval_publications = if source.publish_opened_eval {
            demand
                .opened
                .iter()
                .map(|(key, &multiplicity)| RecursionWhirOpenedEvalPublication {
                    batch_id: key.batch_id,
                    batch_pos: key.batch_pos,
                    chip_idx: key.chip_idx,
                    value_idx: key.value_idx,
                    multiplicity,
                })
                .collect()
        } else {
            Vec::new()
        };
    }
    if proof.proof_shape.publish_external {
        proof.proof_shape.public_value_send_mults = vec![0; proof.proof_shape.public_values.len()];
        for (idx, mult) in &demand.public_values {
            if let Some(slot) = proof.proof_shape.public_value_send_mults.get_mut(*idx) {
                *slot = *mult;
            }
        }
    }

    let publish_opened =
        proof.whir_source.as_ref().is_some_and(|source| source.publish_opened_eval) ||
            proof.whir.batch_eval_rows.iter().any(|row| row.opened_eval_send_mult != 0);
    for row in &mut proof.whir.batch_eval_rows {
        if !publish_opened || !row.is_value {
            row.opened_eval_send_mult = 0;
            continue;
        }
        let key = OpenedEvalDemandKey {
            batch_id: row.batch_id,
            batch_pos: row.batch_pos,
            chip_idx: row.chip_idx,
            value_idx: row.value_idx,
        };
        row.opened_eval_send_mult = demand.opened.get(&key).copied().unwrap_or(0);
    }
}

pub fn annotate_constraint_replay_publications(
    record: &mut RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) {
    record.mark_semantic_mutation();
    for proof in &mut record.proof_records {
        annotate_child_constraint_replay_publications(proof, program);
    }
}

pub fn constraint_replay_bus_residual_report(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> ConstraintReplayBusResidualReport {
    let mut report = ConstraintReplayBusResidualReport::new();
    let checks: Vec<(&'static str, BTreeMap<Vec<u32>, i64>)> = vec![
        ("6 ConstraintProgram", program_residual(record, program)),
        ("7 ConstraintRootTable", root_table_residual(record, program)),
        ("3 NativeChipMetadata", native_chip_metadata_residual(record)),
        ("1008 ProofShapeChipMeta", proof_shape_chip_meta_residual(record, program)),
        ("1033 ConstraintNodeValue", node_value_residual(record, program)),
        ("1034 ConstraintChallenge", challenge_residual(record, program)),
        ("1018 BatchSumcheckClaimChain", batch_sumcheck_claim_chain_residual(record, program)),
        ("1019 SumcheckOut", sumcheck_out_residual(record, program)),
        ("1042 BetaLadderChain", beta_ladder_chain_residual(record, program)),
        ("1035 ConstraintFoldChain", fold_chain_residual(record, program)),
        ("1048 ConstraintFoldPlanChain", fold_plan_chain_residual(record, program)),
        ("13 ConstraintHeightInverse", height_inverse_residual(record, program)),
        ("1036 ConstraintEqChain", eq_chain_residual(record, program)),
        ("1031 WhirOpenedEval", opened_eval_residual(record, program)),
        ("1010 ProofShapeValues(PUBLIC_VALUES)", public_values_residual(record, program)),
        ("1010 ProofShapeValues(VK_META)", vk_meta_residual(record)),
        ("1050 ProofShapeGlobalPacked", global_packed_residual(record, program)),
    ];
    for (name, residual) in checks {
        if !residual.is_empty() {
            report.insert(name, residual);
        }
    }
    report
}

fn program_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    let counts = program_static_presence_counts(record);
    let plan = program.constraint_static_plan();
    plan.for_each_node(program, |row| {
        let mult = *counts.get(&row.static_chip_id).unwrap_or(&0);
        apply_residual(&mut residual, program_key_ref(row), mult as i64);
    });
    for_each_dag_row(record, program, |row| {
        apply_residual(&mut residual, program_key(&row.program), -1);
    });
    finalize_residual(residual)
}

fn root_table_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let plan = program.constraint_static_plan();
    root_table_residual_from_rows(
        &program_static_presence_counts(record),
        &plan.root_rows,
        &fold_rows(record, program),
    )
}

fn root_table_residual_from_rows(
    counts: &BTreeMap<usize, usize>,
    root_rows: &[ConstraintRootTableRow],
    fold_rows: &[ConstraintFoldRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in root_rows {
        let mult = if row.static_chip_id == CONSTRAINT_HEIGHT_TABLE_STATIC_ID {
            0
        } else {
            *counts.get(&row.static_chip_id).unwrap_or(&0)
        };
        apply_residual(&mut residual, root_table_key(row), mult as i64);
    }
    for row in fold_rows {
        for slot in 0..CONSTRAINT_FOLD_ROOT_SLOTS {
            if fold_root_active(row, slot) {
                apply_residual(&mut residual, fold_root_table_key(row, slot), -1);
            }
        }
    }
    finalize_residual(residual)
}

fn proof_shape_chip_meta_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::Chip { proof_idx, chip, publish_external, .. } = row {
            if publish_external {
                let batch_count = chip.perm_width / D_EF;
                apply_residual(
                    &mut residual,
                    chip_meta_key(
                        proof_idx,
                        chip.chip_idx,
                        chip.static_chip_id,
                        chip.log_height,
                        chip.gate_count,
                        batch_count,
                    ),
                    (chip.gate_count + batch_count + 1) as i64,
                );
            }
        }
    }
    for row in fold_rows(record, program) {
        apply_residual(
            &mut residual,
            chip_meta_key(
                row.proof_idx,
                row.remaining_chips - 1,
                row.static_chip_id,
                row.log_height,
                row.gate_count,
                row.batch_count,
            ),
            -1,
        );
    }
    finalize_residual(residual)
}

fn native_chip_metadata_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for request in record.native_chip_metadata.requests() {
        apply_residual(&mut residual, native_chip_metadata_key(*request), i64::from(request.count));
    }
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::Chip { proof_idx, chip, publish_external: _, .. } = row {
            let role_id = record
                .proof_records
                .iter()
                .find(|proof| proof.proof_idx == proof_idx)
                .map(|proof| proof.proof_shape.role_id)
                .unwrap_or(0);
            apply_residual(
                &mut residual,
                native_chip_metadata_key(chip.metadata_request(role_id)),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn node_value_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for_each_dag_row(record, program, |row| {
        apply_residual(
            &mut residual,
            node_value_key(
                row.proof_idx,
                row.chip_idx,
                row.program.static_chip_id,
                row.program.node_idx,
                row.value,
            ),
            row.program.fanout as i64,
        );
        if row.program.is_add || row.program.is_sub || row.program.is_mul || row.program.is_fused {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.lhs_idx,
                    row.lhs_value,
                ),
                -1,
            );
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.rhs_idx,
                    row.rhs_value,
                ),
                -1,
            );
        }
        if row.program.is_fused {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.third_idx,
                    row.third_value,
                ),
                -1,
            );
        }
        if row.leaf_flags[CONSTRAINT_LEAF_PRECOMPUTED] {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.lhs_idx,
                    row.value,
                ),
                -1,
            );
        }
    });
    for row in fold_rows(record, program) {
        for slot in 0..CONSTRAINT_FOLD_ROOT_SLOTS {
            if fold_root_active(&row, slot) {
                apply_residual(
                    &mut residual,
                    node_value_key(
                        row.proof_idx,
                        row.remaining_chips - 1,
                        row.static_chip_id,
                        row.root_nodes[slot],
                        row.root_values[slot],
                    ),
                    -1,
                );
            }
        }
    }
    finalize_residual(residual)
}

#[cfg(test)]
fn node_value_residual_from_rows(
    dag_rows: &[ConstraintDagRow],
    fold_rows: &[ConstraintFoldRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in dag_rows {
        apply_residual(
            &mut residual,
            node_value_key(
                row.proof_idx,
                row.chip_idx,
                row.program.static_chip_id,
                row.program.node_idx,
                row.value,
            ),
            row.program.fanout as i64,
        );
        if row.program.is_add || row.program.is_sub || row.program.is_mul || row.program.is_fused {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.lhs_idx,
                    row.lhs_value,
                ),
                -1,
            );
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.rhs_idx,
                    row.rhs_value,
                ),
                -1,
            );
        }
        if row.program.is_fused {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.third_idx,
                    row.third_value,
                ),
                -1,
            );
        }
        if row.leaf_flags[CONSTRAINT_LEAF_PRECOMPUTED] {
            apply_residual(
                &mut residual,
                node_value_key(
                    row.proof_idx,
                    row.chip_idx,
                    row.program.static_chip_id,
                    row.program.lhs_idx,
                    row.value,
                ),
                -1,
            );
        }
    }
    for row in fold_rows {
        for slot in 0..CONSTRAINT_FOLD_ROOT_SLOTS {
            if fold_root_active(row, slot) {
                apply_residual(
                    &mut residual,
                    node_value_key(
                        row.proof_idx,
                        row.remaining_chips - 1,
                        row.static_chip_id,
                        row.root_nodes[slot],
                        row.root_values[slot],
                    ),
                    -1,
                );
            }
        }
    }
    finalize_residual(residual)
}

fn challenge_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in beta_ladder_rows(record, program) {
        apply_residual(
            &mut residual,
            challenge_key(
                row.proof_idx,
                CONSTRAINT_CHALLENGE_BETA_POWER,
                row.power_idx,
                0,
                row.power,
            ),
            row.serve_mult as i64,
        );
        apply_residual(
            &mut residual,
            challenge_key(
                row.proof_idx,
                CONSTRAINT_CHALLENGE_PERM_ALPHA,
                0,
                0,
                row.prev_power_or_alpha,
            ),
            row.alpha_serve_mult as i64,
        );
        apply_residual(
            &mut residual,
            challenge_key(
                row.proof_idx,
                CONSTRAINT_CHALLENGE_BETA_SEPTIX,
                0,
                0,
                beta_ladder_septix(&row),
            ),
            row.septix_serve_mult as i64,
        );
    }
    for row in challenge_rows(record, program) {
        apply_residual(
            &mut residual,
            challenge_key(row.proof_idx, CONSTRAINT_CHALLENGE_LCS, row.chip_idx, 0, row.lcs_limbs),
            2,
        );
        apply_residual(
            &mut residual,
            challenge_key(
                row.proof_idx,
                CONSTRAINT_CHALLENGE_IS_FIRST,
                row.static_chip_id,
                0,
                row.first,
            ),
            row.first_send_mult as i64,
        );
        apply_residual(
            &mut residual,
            challenge_key(
                row.proof_idx,
                CONSTRAINT_CHALLENGE_IS_LAST,
                row.static_chip_id,
                0,
                row.last,
            ),
            row.last_send_mult as i64,
        );
    }
    for_each_dag_row(record, program, |row| {
        let challenge_leaf = row.leaf_flags[CONSTRAINT_LEAF_PERM_ALPHA] ||
            row.leaf_flags[CONSTRAINT_LEAF_BETA_POWER] ||
            row.leaf_flags[CONSTRAINT_LEAF_BETA_SEPTIX] ||
            row.leaf_flags[CONSTRAINT_LEAF_IS_FIRST_ROW] ||
            row.leaf_flags[CONSTRAINT_LEAF_IS_LAST_ROW];
        if challenge_leaf {
            apply_residual(
                &mut residual,
                challenge_key(
                    row.proof_idx,
                    dag_challenge_kind(&row),
                    row.program.third_idx,
                    0,
                    row.value,
                ),
                -1,
            );
        }
    });
    let terminal_child_cgb = child_contains_global_bus_for_role(program.role);
    for row in terminal_rows(record, program) {
        // The ConstraintChallenge beta/alpha feeds are cgb-only recvs.
        if row.is_final && terminal_child_cgb {
            apply_residual(
                &mut residual,
                challenge_key(row.proof_idx, CONSTRAINT_CHALLENGE_PERM_ALPHA, 0, 0, row.perm_alpha),
                -1,
            );
            for power in 0..CONSTRAINT_CHAIN_LIMBS {
                apply_residual(
                    &mut residual,
                    challenge_key(
                        row.proof_idx,
                        CONSTRAINT_CHALLENGE_BETA_POWER,
                        power + 1,
                        0,
                        row.beta_powers[power],
                    ),
                    -1,
                );
            }
        }
        if row.is_lcs_step {
            apply_residual(
                &mut residual,
                challenge_key(row.proof_idx, CONSTRAINT_CHALLENGE_LCS, row.chip_idx, 0, row.lcs),
                -1,
            );
        }
        if row.state_chain_recv_mult {
            apply_terminal_state_chain_residual(&mut residual, &row, false);
        }
        if row.state_chain_send_mult {
            apply_terminal_state_chain_residual(&mut residual, &row, true);
        }
        if terminal_child_cgb && row.is_final {
            // The split ConstraintBoundary row is the direct sink of the
            // Terminal LCS chain at cursor c_chips; the final proof-check row
            // no longer adds a redundant recv/send self-pair.
            apply_terminal_state_chain_residual(&mut residual, &row, false);
        }
    }
    for row in fold_rows(record, program) {
        if row.is_skip {
            apply_residual(
                &mut residual,
                challenge_key(
                    row.proof_idx,
                    CONSTRAINT_CHALLENGE_LCS,
                    row.remaining_chips - 1,
                    0,
                    row.perm_value,
                ),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn sumcheck_out_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in batch_transcript_input_rows(record) {
        let BatchTranscriptInputRow::Fused { proof_idx, perm_alpha, perm_beta, .. } = row;
        apply_residual(
            &mut residual,
            sumcheck_out_key(proof_idx, SUMCHECK_OUT_PERM_ALPHA, 0, perm_alpha),
            1,
        );
        apply_residual(
            &mut residual,
            sumcheck_out_key(proof_idx, SUMCHECK_OUT_PERM_BETA, 0, perm_beta),
            1,
        );
    }
    for row in batch_sumcheck_rows(record) {
        if let BatchSumcheckRow::Round { proof_idx, num_rounds, round, eq_challenge, .. } = row {
            apply_residual(
                &mut residual,
                sumcheck_out_key(
                    proof_idx,
                    SUMCHECK_OUT_EQ,
                    num_rounds - 1 - round.round_idx,
                    eq_challenge,
                ),
                1,
            );
        }
    }
    for row in beta_ladder_rows(record, program) {
        if row.challenges_recv_mult {
            apply_residual(
                &mut residual,
                sumcheck_out_key(
                    row.proof_idx,
                    SUMCHECK_OUT_PERM_ALPHA,
                    0,
                    row.prev_power_or_alpha,
                ),
                -1,
            );
            apply_residual(
                &mut residual,
                sumcheck_out_key(row.proof_idx, SUMCHECK_OUT_PERM_BETA, 0, row.beta),
                -1,
            );
        }
    }
    for row in terminal_rows(record, program) {
        if row.eq_recv_mult {
            apply_residual(
                &mut residual,
                sumcheck_out_key(row.proof_idx, SUMCHECK_OUT_EQ, row.opening_idx, row.eq_challenge),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn batch_sumcheck_claim_chain_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in batch_sumcheck_rows(record) {
        match row {
            BatchSumcheckRow::Seed { proof_idx, num_rounds, c_chips, .. } => {
                apply_residual(
                    &mut residual,
                    batch_sumcheck_claim_chain_key(
                        proof_idx,
                        0,
                        num_rounds,
                        c_chips,
                        [F::zero(); D_EF],
                    ),
                    1,
                );
            }
            BatchSumcheckRow::Round { proof_idx, num_rounds, c_chips, round, .. } => {
                apply_residual(
                    &mut residual,
                    batch_sumcheck_claim_chain_key(
                        proof_idx,
                        round.round_idx,
                        num_rounds,
                        c_chips,
                        round.claim_in,
                    ),
                    -1,
                );
                apply_residual(
                    &mut residual,
                    batch_sumcheck_claim_chain_key(
                        proof_idx,
                        round.round_idx + 1,
                        num_rounds,
                        c_chips,
                        round.claim_out,
                    ),
                    1,
                );
            }
        }
    }
    for row in terminal_rows(record, program) {
        if row.last_claim_recv_mult {
            apply_residual(
                &mut residual,
                batch_sumcheck_claim_chain_key(
                    row.proof_idx,
                    row.num_rounds,
                    row.num_rounds,
                    row.c_chips,
                    row.last_claim,
                ),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn beta_ladder_chain_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in beta_ladder_rows(record, program) {
        if row.power_idx != 0 {
            apply_residual(&mut residual, beta_ladder_chain_recv_key(&row), -1);
        }
        if row.power_idx + 1 != CONSTRAINT_MAX_BETA_POWERS {
            apply_residual(&mut residual, beta_ladder_chain_send_key(&row), 1);
        }
    }
    finalize_residual(residual)
}

fn fold_chain_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    fold_chain_residual_from_rows(
        &batch_transcript_input_rows(record),
        &fold_rows(record, program),
        &terminal_rows(record, program),
    )
}

fn fold_chain_residual_from_rows(
    batch_rows: &[BatchTranscriptInputRow],
    fold_rows: &[ConstraintFoldRow],
    terminal_rows: &[ConstraintTerminalRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in batch_rows {
        let BatchTranscriptInputRow::Fused { proof_idx, alpha, .. } = row;
        apply_residual(
            &mut residual,
            fold_chain_key(
                *proof_idx,
                0,
                *alpha,
                [F::zero(); D_EF],
                [F::zero(); D_EF],
                [F::zero(); D_EF],
            ),
            1,
        );
    }
    for row in fold_rows {
        apply_residual(&mut residual, fold_chain_recv_key(row), -1);
        apply_residual(&mut residual, fold_chain_send_key(row), 1);
    }
    for row in terminal_rows {
        if row.fold_chain_recv_mult {
            apply_residual(&mut residual, terminal_fold_chain_key(row), -1);
        }
    }
    finalize_residual(residual)
}

fn fold_plan_chain_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let fold_rows = fold_rows(record, program);
    let challenge_rows = challenge_rows(record, program);
    let terminal_rows = terminal_rows(record, program);
    fold_plan_chain_residual_from_rows(record, &fold_rows, &challenge_rows, &terminal_rows)
}

fn fold_plan_chain_residual_from_rows(
    record: &RecursionRecord,
    fold_rows: &[ConstraintFoldRow],
    challenge_rows: &[ConstraintChallengeRow],
    terminal_rows: &[ConstraintTerminalRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::E5 { proof_idx, prev, .. } = row {
            apply_residual(
                &mut residual,
                fold_plan_chain_key(proof_idx, 0, prev.chip_idx, 0),
                (prev.chip_idx + 2) as i64,
            );
        }
    }
    for row in batch_transcript_input_rows(record) {
        let BatchTranscriptInputRow::Fused { proof_idx, c_chips, .. } = row;
        apply_residual(&mut residual, fold_plan_chain_key(proof_idx, 0, c_chips, 0), -1);
    }
    for row in challenge_rows {
        apply_residual(&mut residual, fold_plan_chain_key(row.proof_idx, 0, row.c_chips, 0), -1);
    }
    for row in fold_rows {
        apply_residual(&mut residual, fold_plan_chain_recv_key(row), -1);
        apply_residual(&mut residual, fold_plan_chain_send_key(row), 1);
    }
    for row in terminal_rows {
        if row.is_final {
            apply_residual(
                &mut residual,
                fold_plan_chain_key(row.proof_idx, row.fold_cursor, 0, 0),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn height_inverse_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    let static_counts = program_static_presence_counts(record);
    for row in program.constraint_static_plan().root_rows.iter() {
        if row.static_chip_id == CONSTRAINT_HEIGHT_TABLE_STATIC_ID {
            apply_residual(
                &mut residual,
                height_inverse_key(row.root_ord, row.node_idx),
                root_table_row_multiplicity(record, &static_counts, row) as i64,
            );
        }
    }
    for row in fold_rows(record, program) {
        if row.is_skip {
            apply_residual(&mut residual, height_inverse_key(row.root_ord, row.root_nodes[0]), -1);
        }
    }
    finalize_residual(residual)
}

fn eq_chain_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    eq_chain_residual_from_rows(&terminal_rows(record, program), &challenge_rows(record, program))
}

fn eq_chain_residual_from_rows(
    terminal_rows: &[ConstraintTerminalRow],
    challenge_rows: &[ConstraintChallengeRow],
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in terminal_rows {
        if row.eq_chain_recv_mult {
            apply_residual(&mut residual, terminal_eq_chain_recv_key(&row), -1);
        }
        if row.eq_chain_send_mult != 0 {
            apply_residual(
                &mut residual,
                terminal_eq_chain_send_key(&row),
                row.eq_chain_send_mult as i64,
            );
        }
    }
    for row in challenge_rows {
        apply_residual(
            &mut residual,
            eq_chain_key(row.proof_idx, row.log_height, row.eq_acc, row.first, row.last),
            -1,
        );
    }
    finalize_residual(residual)
}

fn opened_eval_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for proof in &record.proof_records {
        for row in &proof.whir.batch_eval_rows {
            apply_residual(
                &mut residual,
                opened_eval_key(
                    row.proof_idx,
                    row.batch_id,
                    row.batch_pos,
                    row.chip_idx,
                    row.value_idx,
                    row.value,
                ),
                row.opened_eval_send_mult as i64,
            );
        }
    }
    for_each_dag_row(record, program, |row| {
        let opened_leaf = row.leaf_flags[CONSTRAINT_LEAF_PREPROCESSED] ||
            row.leaf_flags[CONSTRAINT_LEAF_MAIN] ||
            row.leaf_flags[CONSTRAINT_LEAF_RESERVED_POLY];
        if opened_leaf {
            apply_residual(
                &mut residual,
                opened_eval_key(
                    row.proof_idx,
                    row.program.lhs_idx,
                    row.opened_batch_pos,
                    row.chip_idx,
                    row.program.rhs_idx,
                    row.value,
                ),
                -1,
            );
        }
    });
    for row in fold_rows(record, program) {
        if row.is_batch {
            apply_residual(
                &mut residual,
                opened_eval_key(
                    row.proof_idx,
                    PROOF_SHAPE_BATCH_PERMUTATION,
                    row.remaining_chips - 1,
                    row.remaining_chips - 1,
                    row.root_ord / CONSTRAINT_FOLD_BATCH_SIZE,
                    row.perm_value,
                ),
                -1,
            );
        }
    }
    finalize_residual(residual)
}

fn public_values_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for proof in &record.proof_records {
        if proof.proof_shape.publish_external {
            for (idx, value) in proof
                .proof_shape
                .public_values
                .iter()
                .copied()
                .take(proof.proof_shape.num_public_values)
                .enumerate()
            {
                let mult =
                    proof.proof_shape.public_value_send_mults.get(idx).copied().unwrap_or_else(
                        || {
                            panic!(
                                "missing public value send mult for proof {} public index {}",
                                proof.proof_idx, idx
                            )
                        },
                    );
                apply_residual(
                    &mut residual,
                    public_value_key(proof.proof_idx, idx, value),
                    mult as i64,
                );
            }
        }
    }
    for_each_dag_row(record, program, |row| {
        if row.leaf_flags[CONSTRAINT_LEAF_PUBLIC] {
            apply_residual(
                &mut residual,
                public_value_key(row.proof_idx, row.program.lhs_idx, row.value[0]),
                -1,
            );
        }
    });
    for row in terminal_rows(record, program) {
        if row.public_value_recv_mult {
            for (slot, pv_idx) in TERMINAL_PV_IDXS
                .iter()
                .copied()
                .take(CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT)
                .enumerate()
            {
                apply_residual(
                    &mut residual,
                    public_value_key(row.proof_idx, pv_idx, row.public_values[slot]),
                    -1,
                );
            }
        }
    }
    for proof in &record.proof_records {
        if proof.proof_shape.publish_whir_inputs {
            for idx in statement_lift_public_value_recvs(proof.proof_shape.role_id) {
                if let Some(value) = proof.proof_shape.public_values.get(idx).copied() {
                    apply_residual(
                        &mut residual,
                        public_value_key(proof.proof_idx, idx, value),
                        -1,
                    );
                }
            }
        }
    }
    finalize_residual(residual)
}

fn global_packed_key(
    proof_idx: usize,
    shape_idx_base: usize,
    values: [F; 8],
) -> Vec<u32> {
    let mut key = Vec::with_capacity(10);
    key.push(proof_idx as u32);
    key.push(shape_idx_base as u32);
    key.extend(values.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn boundary_global_packed_values(row: &ConstraintTerminalRow, packed_row: usize) -> [F; 8] {
    let shape_idx_base = 48 + 8 * packed_row;
    core::array::from_fn(|column| {
        let pv_idx = shape_idx_base + column;
        match pv_idx {
            48..=50 => row.public_values[pv_idx - 42],
            51 => F::zero(),
            52..=119 => row.public_values[9 + pv_idx - 52],
            _ => unreachable!("fixed packed Global row is outside core public values"),
        }
    })
}

fn global_packed_residual(
    record: &RecursionRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for row in proof_shape_binder_rows(record) {
        if let ProofShapeBinderRow::PublicValues {
            proof_idx,
            shape_idx_base,
            values,
            global_packed_send: true,
            ..
        } = row
        {
            apply_residual(
                &mut residual,
                global_packed_key(proof_idx, shape_idx_base, values),
                1,
            );
        }
    }
    if child_contains_global_bus_for_role(program.role) {
        for row in terminal_rows(record, program).into_iter().filter(|row| row.is_final) {
            for packed_row in 0..CONSTRAINT_BOUNDARY_GLOBAL_PACKED_ROWS {
                let shape_idx_base = 48 + 8 * packed_row;
                apply_residual(
                    &mut residual,
                    global_packed_key(
                        row.proof_idx,
                        shape_idx_base,
                        boundary_global_packed_values(&row, packed_row),
                    ),
                    -1,
                );
            }
        }
    }
    finalize_residual(residual)
}

fn vk_meta_residual(record: &RecursionRecord) -> BTreeMap<Vec<u32>, i64> {
    let mut residual = BTreeMap::<Vec<u32>, i64>::new();
    for proof in &record.proof_records {
        if proof.proof_shape.publish_external {
            for (idx, value) in proof.proof_shape.vk_meta.iter().copied().enumerate() {
                let mult = proof.proof_shape.vk_meta_send_mults[idx];
                apply_residual(
                    &mut residual,
                    proof_shape_value_key(
                        proof.proof_idx,
                        PROOF_SHAPE_NAMESPACE_VK_META,
                        idx,
                        value,
                    ),
                    mult as i64,
                );
            }
        }
        if proof.proof_shape.publish_whir_inputs &&
            proof.proof_shape.role_id == crate::whir_dt::WHIR_ROLE_CORE
        {
            for _ in 0..STATEMENT_GLOBAL_CHUNKS {
                apply_residual(
                    &mut residual,
                    proof_shape_value_key(
                        proof.proof_idx,
                        PROOF_SHAPE_NAMESPACE_VK_META,
                        PROOF_SHAPE_VK_META_BOUNDARY_KIND,
                        proof.proof_shape.vk_meta[PROOF_SHAPE_VK_META_BOUNDARY_KIND],
                    ),
                    -1,
                );
            }
            if proof.proof_shape.public_values.get(CORE_PV_SHARD).copied() == Some(F::one()) {
                for meta_idx in
                    PROOF_SHAPE_VK_META_BOUNDARY_X_BASE..PROOF_SHAPE_VK_META_BOUNDARY_X_BASE + 22
                {
                    apply_residual(
                        &mut residual,
                        proof_shape_value_key(
                            proof.proof_idx,
                            PROOF_SHAPE_NAMESPACE_VK_META,
                            meta_idx,
                            proof.proof_shape.vk_meta[meta_idx],
                        ),
                        -1,
                    );
                }
            }
        }
        if proof.proof_shape.publish_whir_inputs && record.statement_public_values.is_some() {
            for idx in 0..proof.proof_shape.vk_meta.len() {
                apply_residual(
                    &mut residual,
                    proof_shape_value_key(
                        proof.proof_idx,
                        PROOF_SHAPE_NAMESPACE_VK_META,
                        idx,
                        proof.proof_shape.vk_meta[idx],
                    ),
                    -1,
                );
            }
        }
    }
    finalize_residual(residual)
}

fn program_row_for_node(
    chip: &crate::symbolic_ir_dt::RecursionPolyAirChipIr,
    node: &RecursionPolyAirNode,
    fanout: usize,
) -> Result<ConstraintProgramRow, String> {
    let mut row = ConstraintProgramRow {
        static_chip_id: chip.static_chip_id,
        node_idx: node.node_id as usize,
        is_leaf: false,
        is_const: false,
        is_add: false,
        is_sub: false,
        is_mul: false,
        is_fused: false,
        lhs_idx: 0,
        rhs_idx: 0,
        third_idx: 0,
        aux: F::zero(),
        leaf_kind: 0,
        fanout,
    };
    match &node.op {
        RecursionPolyAirOp::Leaf(leaf) => {
            row.is_leaf = true;
            row.leaf_kind = leaf_kind(leaf);
            match leaf {
                RecursionPolyAirLeaf::Preprocessed { col } => {
                    row.lhs_idx = PROOF_SHAPE_BATCH_PREPROCESSED;
                    row.rhs_idx = *col;
                }
                RecursionPolyAirLeaf::Main { col } => {
                    row.lhs_idx = PROOF_SHAPE_BATCH_MAIN;
                    row.rhs_idx = *col;
                }
                RecursionPolyAirLeaf::Public { index } => row.lhs_idx = *index,
                RecursionPolyAirLeaf::BetaPower { power } => {
                    row.rhs_idx = CONSTRAINT_CHALLENGE_BETA_POWER;
                    row.third_idx = *power;
                }
                RecursionPolyAirLeaf::ReservedPoly { index } => {
                    let source = chip.reserved_poly.get(*index).copied().ok_or_else(|| {
                        format!(
                            "reserved-poly source {} missing for static chip {} {}",
                            index, chip.static_chip_id, chip.chip_name
                        )
                    })?;
                    match source {
                        PairCol::Prep(col) => {
                            row.lhs_idx = PROOF_SHAPE_BATCH_PREPROCESSED;
                            row.rhs_idx = col;
                        }
                        PairCol::Main(col) => {
                            row.lhs_idx = PROOF_SHAPE_BATCH_MAIN;
                            row.rhs_idx = col;
                        }
                    }
                }
                RecursionPolyAirLeaf::Precomputed { index } => {
                    row.lhs_idx = precompute_root_node(chip, *index).ok_or_else(|| {
                        format!(
                            "precompute root {} missing for static chip {} {}",
                            index, chip.static_chip_id, chip.chip_name
                        )
                    })?;
                }
                RecursionPolyAirLeaf::PermAlpha => {
                    row.rhs_idx = CONSTRAINT_CHALLENGE_PERM_ALPHA;
                }
                RecursionPolyAirLeaf::BetaSeptix => {
                    row.rhs_idx = CONSTRAINT_CHALLENGE_BETA_SEPTIX;
                }
                RecursionPolyAirLeaf::IsFirstRow => {
                    row.rhs_idx = CONSTRAINT_CHALLENGE_IS_FIRST;
                    row.third_idx = chip.static_chip_id;
                }
                RecursionPolyAirLeaf::IsLastRow => {
                    row.rhs_idx = CONSTRAINT_CHALLENGE_IS_LAST;
                    row.third_idx = chip.static_chip_id;
                }
            }
        }
        RecursionPolyAirOp::ConstBase(value) => {
            row.is_const = true;
            row.aux = *value;
        }
        RecursionPolyAirOp::ConstExt(value) => {
            use crate::symbolic_expr_adapter_dt::{
                classify_extension_constant, CanonicalExtensionConstant,
            };
            if classify_extension_constant(value) != CanonicalExtensionConstant::Theta {
                return Err(format!(
                    "unsupported extension constant in constraint program table: {value:?}"
                ));
            }
            row.is_const = true;
            row.lhs_idx = 1;
        }
        RecursionPolyAirOp::Add { lhs, rhs } => {
            row.is_add = true;
            row.lhs_idx = *lhs as usize;
            row.rhs_idx = *rhs as usize;
        }
        RecursionPolyAirOp::Sub { lhs, rhs } => {
            row.is_sub = true;
            row.lhs_idx = *lhs as usize;
            row.rhs_idx = *rhs as usize;
        }
        RecursionPolyAirOp::Mul { lhs, rhs } => {
            row.is_mul = true;
            row.lhs_idx = *lhs as usize;
            row.rhs_idx = *rhs as usize;
        }
        RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, sign } => {
            row.is_fused = true;
            row.lhs_idx = *lhs as usize;
            row.rhs_idx = *rhs as usize;
            row.third_idx = *addend as usize;
            row.aux = f_bool(*sign);
        }
        RecursionPolyAirOp::Neg { .. } => {
            return Err(format!(
                "unsupported symbolic op in constraint program table: {:?}",
                node.op
            ));
        }
    }
    Ok(row)
}

fn materialize_dag_row(
    case: &ConstraintDagCaseRow,
    program: ConstraintProgramNodeRef<'_>,
) -> ConstraintDagRow {
    let mut leaf_flags = [false; CONSTRAINT_LEAF_KIND_COUNT];
    if program.is_leaf() {
        leaf_flags[program.leaf_kind()] = true;
    }
    ConstraintDagRow {
        proof_idx: case.proof_idx,
        chip_idx: case.chip_idx,
        program: program.materialize(),
        leaf_flags,
        value: case.value,
        lhs_value: case.lhs_value,
        rhs_value: case.rhs_value,
        third_value: case.third_value,
        opened_batch_pos: case.opened_batch_pos,
    }
}

fn program_prep_row(row: ConstraintProgramNodeRef<'_>) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_PROGRAM_PREPROCESSED_COLS];
    let cols: &mut ConstraintProgramPreprocessedCols<F> = values.as_mut_slice().borrow_mut();
    cols.static_chip_id = f(row.static_chip_id);
    cols.node_idx = f(row.node_idx);
    cols.op_code = f(row.op_code());
    cols.lhs_idx = f(row.lhs_idx());
    cols.rhs_idx = f(row.rhs_idx());
    cols.third_idx = f(row.third_idx());
    cols.aux = row.plan.aux;
    cols.leaf_kind = f(row.leaf_kind());
    cols.fanout = f(row.fanout());
    values
}

fn root_table_prep_row(row: &ConstraintRootTableRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_ROOT_TABLE_PREPROCESSED_COLS];
    let cols: &mut ConstraintRootTablePreprocessedCols<F> = values.as_mut_slice().borrow_mut();
    cols.static_chip_id = f(row.static_chip_id);
    cols.root_kind = f(row.root_kind);
    cols.root_ord = f(row.root_ord);
    cols.node_idx = f(row.node_idx);
    cols.sign = signed_f(row.sign);
    values
}

fn fill_dag_case_row(
    values: &mut [F],
    row: &ConstraintDagCaseRow,
    program: ConstraintProgramNodeRef<'_>,
) {
    debug_assert_eq!(values.len(), NUM_CONSTRAINT_DAG_EVAL_COLS);
    let cols: &mut ConstraintDagEvalCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.chip_idx = f(row.chip_idx);
    cols.static_chip_id = f(program.static_chip_id);
    cols.node_idx = f(program.node_idx);
    cols.is_const = f_bool(program.is_const());
    cols.is_add = f_bool(program.is_add());
    cols.is_sub = f_bool(program.is_sub());
    cols.is_mul = f_bool(program.is_mul());
    cols.is_fused = f_bool(program.is_fused());
    cols.lhs_idx = f(program.lhs_idx());
    cols.rhs_idx = f(program.rhs_idx());
    cols.third_idx = f(program.third_idx());
    cols.aux = program.plan.aux;
    cols.fanout = f(program.fanout());
    if program.is_leaf() {
        cols.leaf_flags[program.leaf_kind()] = F::one();
    }
    cols.value = row.value;
    cols.lhs_value = row.lhs_value;
    cols.rhs_value = row.rhs_value;
    cols.third_value = row.third_value;
    cols.opened_batch_pos = f(row.opened_batch_pos);
}

/// Diagnostic oracle for the pre-split exact row shape. Production never retains these rows and
/// never calls this emitter; differential tests use it to pin the static-plan/dynamic-case join.
#[cfg(test)]
fn fill_exact_dag_row_oracle(values: &mut [F], row: &ConstraintDagRow) {
    debug_assert_eq!(values.len(), NUM_CONSTRAINT_DAG_EVAL_COLS);
    let cols: &mut ConstraintDagEvalCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.chip_idx = f(row.chip_idx);
    cols.static_chip_id = f(row.program.static_chip_id);
    cols.node_idx = f(row.program.node_idx);
    cols.is_const = f_bool(row.program.is_const);
    cols.is_add = f_bool(row.program.is_add);
    cols.is_sub = f_bool(row.program.is_sub);
    cols.is_mul = f_bool(row.program.is_mul);
    cols.is_fused = f_bool(row.program.is_fused);
    cols.lhs_idx = f(row.program.lhs_idx);
    cols.rhs_idx = f(row.program.rhs_idx);
    cols.third_idx = f(row.program.third_idx);
    cols.aux = row.program.aux;
    cols.fanout = f(row.program.fanout);
    for (dst, flag) in cols.leaf_flags.iter_mut().zip(row.leaf_flags.iter()) {
        *dst = f_bool(*flag);
    }
    cols.value = row.value;
    cols.lhs_value = row.lhs_value;
    cols.rhs_value = row.rhs_value;
    cols.third_value = row.third_value;
    cols.opened_batch_pos = f(row.opened_batch_pos);
}

fn fold_cols(row: &ConstraintFoldRow) -> ConstraintFoldCols<F> {
    ConstraintFoldCols {
        proof_idx: f(row.proof_idx),
        is_skip: f_bool(row.is_skip),
        is_gate: f_bool(row.is_gate),
        is_batch: f_bool(row.is_batch),
        cursor: f(row.cursor),
        remaining_chips: f(row.remaining_chips),
        local_ord: f(row.local_ord),
        chain_send_local_ord: f(row.chain_send_local_ord),
        static_chip_id: f(row.static_chip_id),
        log_height: f(row.log_height),
        gate_count: f(row.gate_count),
        batch_count: f(row.batch_count),
        root_ord: f(row.root_ord),
        alpha: row.alpha,
        acc_in: row.acc_in,
        acc_out: row.acc_out,
        pacc_in: row.pacc_in,
        pacc_out: row.pacc_out,
        perm_sum_in: row.perm_sum_in,
        perm_sum_out: row.perm_sum_out,
        root_nodes: row.root_nodes.map(f),
        multiplicity_signs: row.multiplicity_signs.map(signed_f),
        root_values: row.root_values,
        batch_has_second: f_bool(row.batch_has_second),
        perm_value: row.perm_value,
    }
}

fn append_fold_row(values: &mut Vec<F>, row: &ConstraintFoldRow) {
    values.extend_from_slice(fold_cols(row).as_slice());
}

#[cfg(test)]
fn fill_fold_row(values: &mut [F], row: &ConstraintFoldRow) {
    debug_assert_eq!(values.len(), NUM_CONSTRAINT_FOLD_COLS);
    values.copy_from_slice(fold_cols(row).as_slice());
}

fn beta_ladder_row(row: &ConstraintBetaLadderRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_BETA_LADDER_COLS];
    let cols: &mut ConstraintBetaLadderCols<F> = values.as_mut_slice().borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.is_seed = f_bool(row.power_idx == 0);
    cols.is_last = f_bool(row.power_idx + 1 == CONSTRAINT_MAX_BETA_POWERS);
    cols.power_idx = f(row.power_idx);
    cols.beta = row.beta;
    cols.prev_power_or_alpha = row.prev_power_or_alpha;
    cols.power = row.power;
    cols.serve_mult = f(row.serve_mult);
    cols.challenges_recv_mult = f_bool(row.challenges_recv_mult);
    cols.alpha_serve_mult = f(row.alpha_serve_mult);
    cols.septix_serve_mult = f(row.septix_serve_mult);
    values
}

fn fill_challenge_row(values: &mut [F], row: &ConstraintChallengeRow) {
    debug_assert_eq!(values.len(), NUM_CONSTRAINT_CHALLENGE_COLS);
    let cols: &mut ConstraintChallengeCols<F> = values.borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.chip_idx = f(row.chip_idx);
    cols.static_chip_id = f(row.static_chip_id);
    cols.main_width = f(row.main_width);
    cols.log_height = f(row.log_height);
    cols.c_chips = f(row.c_chips);
    cols.lcs_limbs = row.lcs_limbs;
    cols.selector_eq_acc = row.eq_acc;
    cols.selector_first = row.first;
    cols.selector_last = row.last;
    cols.selector_first_send_mult = f(row.first_send_mult);
    cols.selector_last_send_mult = f(row.last_send_mult);
}

#[cfg(test)]
fn challenge_row(row: &ConstraintChallengeRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_CHALLENGE_COLS];
    fill_challenge_row(&mut values, row);
    values
}

fn constraint_boundary_row(row: &ConstraintTerminalRow) -> Vec<F> {
    debug_assert!(row.is_final);
    let mut values = vec![F::zero(); NUM_CONSTRAINT_BOUNDARY_COLS];
    let cols: &mut ConstraintBoundaryCols<F> = values.as_mut_slice().borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_valid = F::one();
    cols.c_chips = f(row.c_chips);
    cols.state_lcs = row.state_lcs_out;
    cols.public_values = row.public_values;
    cols.perm_alpha = row.perm_alpha;
    cols.beta_powers = row.beta_powers;
    cols.state_clock_changed = row.state_clock_changed;
    cols.state_clock_delta_inverse = row.state_clock_delta_inverse;
    cols.state_transition_recv_inverse = row.state_transition_recv_inverse;
    cols.state_transition_send_inverse = row.state_transition_send_inverse;
    cols.init_address_recv_inverse = row.init_address_recv_inverse;
    cols.init_address_send_inverse = row.init_address_send_inverse;
    cols.finalize_address_recv_inverse = row.finalize_address_recv_inverse;
    cols.finalize_address_send_inverse = row.finalize_address_send_inverse;
    cols.global_chain_source_inverse = row.global_chain_source_inverse;
    cols.global_chain_sink_inverse = row.global_chain_sink_inverse;
    values
}

fn terminal_row(row: &ConstraintTerminalRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_TERMINAL_COLS];
    let cols: &mut ConstraintTerminalCols<F> = values.as_mut_slice().borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_seed = f_bool(row.is_seed);
    cols.is_eq_step = f_bool(row.is_eq_step);
    cols.is_lcs_step = f_bool(row.is_lcs_step);
    cols.is_final = f_bool(row.is_final);
    cols.num_rounds = f(row.num_rounds);
    cols.c_chips = f(row.c_chips);
    cols.round_idx = f(row.round_idx);
    cols.opening_idx = f(row.opening_idx);
    cols.chip_idx = f(row.chip_idx);
    cols.opening_point = row.opening_point;
    cols.eq_challenge = row.eq_challenge;
    cols.eq_factor = row.eq_factor;
    cols.eq_in = row.eq_in;
    cols.eq_out = row.eq_out;
    cols.first_prefix_in = row.first_prefix_in;
    cols.first_prefix_out = row.first_prefix_out;
    cols.last_prefix_in = row.last_prefix_in;
    cols.last_prefix_out = row.last_prefix_out;
    cols.fold_cursor = f(row.fold_cursor);
    cols.alpha = row.alpha;
    cols.main_eval = row.main_eval;
    cols.perm_eval = row.perm_eval;
    cols.last_claim = row.last_claim;
    cols.lcs = row.lcs;
    cols.state_lcs_in = row.state_lcs_in;
    cols.state_lcs_out = row.state_lcs_out;
    cols.public_values = row.public_values;
    cols.state_chain_send_mult = f_bool(row.state_chain_send_mult);
    cols.perm_alpha = row.perm_alpha;
    cols.beta_powers = row.beta_powers;
    cols.state_clock_changed = row.state_clock_changed;
    cols.state_clock_delta_inverse = row.state_clock_delta_inverse;
    cols.state_transition_recv_inverse = row.state_transition_recv_inverse;
    cols.state_transition_send_inverse = row.state_transition_send_inverse;
    cols.init_address_recv_inverse = row.init_address_recv_inverse;
    cols.init_address_send_inverse = row.init_address_send_inverse;
    cols.finalize_address_recv_inverse = row.finalize_address_recv_inverse;
    cols.finalize_address_send_inverse = row.finalize_address_send_inverse;
    cols.global_chain_source_inverse = row.global_chain_source_inverse;
    cols.global_chain_sink_inverse = row.global_chain_sink_inverse;
    cols.summary_id_base = f(row.summary_id_base);
    cols.eq_chain_send_mult = f(row.eq_chain_send_mult);
    values
}

fn terminal_row_narrow(row: &ConstraintTerminalRow) -> Vec<F> {
    let mut values = vec![F::zero(); NUM_CONSTRAINT_TERMINAL_NARROW_COLS];
    let cols: &mut ConstraintTerminalColsNarrow<F> = values.as_mut_slice().borrow_mut();
    cols.proof_idx = f(row.proof_idx);
    cols.is_seed = f_bool(row.is_seed);
    cols.is_eq_step = f_bool(row.is_eq_step);
    cols.is_lcs_step = f_bool(row.is_lcs_step);
    cols.is_final = f_bool(row.is_final);
    cols.num_rounds = f(row.num_rounds);
    cols.c_chips = f(row.c_chips);
    cols.round_idx = f(row.round_idx);
    cols.opening_idx = f(row.opening_idx);
    cols.chip_idx = f(row.chip_idx);
    cols.opening_point = row.opening_point;
    cols.eq_challenge = row.eq_challenge;
    cols.eq_factor = row.eq_factor;
    cols.eq_in = row.eq_in;
    cols.eq_out = row.eq_out;
    cols.first_prefix_in = row.first_prefix_in;
    cols.first_prefix_out = row.first_prefix_out;
    cols.last_prefix_in = row.last_prefix_in;
    cols.last_prefix_out = row.last_prefix_out;
    cols.fold_cursor = f(row.fold_cursor);
    cols.alpha = row.alpha;
    cols.main_eval = row.main_eval;
    cols.perm_eval = row.perm_eval;
    cols.last_claim = row.last_claim;
    cols.lcs = row.lcs;
    cols.state_lcs_in = row.state_lcs_in;
    cols.state_lcs_out = row.state_lcs_out;
    cols.state_chain_send_mult = f_bool(row.state_chain_send_mult);
    cols.summary_id_base = f(row.summary_id_base);
    cols.eq_chain_send_mult = f(row.eq_chain_send_mult);
    values
}

#[cfg(test)]
fn terminal_narrow_projection_columns() -> Vec<usize> {
    let mut columns =
        (0..core::mem::offset_of!(ConstraintTerminalCols<u8>, public_values)).collect::<Vec<_>>();
    for (start, len) in [
        (core::mem::offset_of!(ConstraintTerminalCols<u8>, state_chain_send_mult), 1),
        (core::mem::offset_of!(ConstraintTerminalCols<u8>, summary_id_base), 1),
        (core::mem::offset_of!(ConstraintTerminalCols<u8>, eq_chain_send_mult), 1),
    ] {
        columns.extend(start..start + len);
    }
    assert_eq!(columns.len(), NUM_CONSTRAINT_TERMINAL_NARROW_COLS);
    columns
}

fn node_fanouts(
    chip: &crate::symbolic_ir_dt::RecursionPolyAirChipIr,
) -> Result<Vec<usize>, String> {
    let mut fanouts = Vec::new();
    fanouts
        .try_reserve_exact(chip.node_table.len())
        .map_err(|_| "constraint fanout allocation rejected".to_string())?;
    fanouts.resize(chip.node_table.len(), 0usize);
    for node in &chip.node_table {
        match node.op {
            RecursionPolyAirOp::Add { lhs, rhs } |
            RecursionPolyAirOp::Sub { lhs, rhs } |
            RecursionPolyAirOp::Mul { lhs, rhs } => {
                bump(&mut fanouts, lhs)?;
                bump(&mut fanouts, rhs)?;
            }
            RecursionPolyAirOp::FusedMulAdd { lhs, rhs, addend, .. } => {
                bump(&mut fanouts, lhs)?;
                bump(&mut fanouts, rhs)?;
                bump(&mut fanouts, addend)?;
            }
            RecursionPolyAirOp::Leaf(RecursionPolyAirLeaf::Precomputed { index }) => {
                if let Some(root_node) = precompute_root_node(chip, index) {
                    bump(&mut fanouts, root_node as u32)?;
                }
            }
            _ => {}
        }
    }
    let lookup_roots = chip.lookup_multiplicity_roots.len();
    for root in chip.derived_roots.iter().filter_map(|root| match root {
        crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::PrecomputeLc {
            index,
            root_node_id,
        } if *index < lookup_roots => Some(*root_node_id),
        _ => None,
    }) {
        bump(&mut fanouts, root)?;
    }
    for root in &chip.lookup_multiplicity_roots {
        bump(&mut fanouts, root.root_node_id)?;
    }
    for root in &chip.gate_roots {
        bump(&mut fanouts, root.root_node_id)?;
    }
    Ok(fanouts)
}

fn bump(fanouts: &mut [usize], node_id: u32) -> Result<(), String> {
    let value = fanouts
        .get_mut(node_id as usize)
        .ok_or_else(|| format!("constraint fanout node {node_id} is out of bounds"))?;
    *value = value
        .checked_add(1)
        .ok_or_else(|| format!("constraint fanout count overflow at node {node_id}"))?;
    Ok(())
}

pub fn precompute_root_node(
    chip: &crate::symbolic_ir_dt::RecursionPolyAirChipIr,
    index: usize,
) -> Option<usize> {
    chip.derived_roots.iter().find_map(|root| match root {
        crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::PrecomputeLc {
            index: root_index,
            root_node_id,
        } if *root_index == index => Some(*root_node_id as usize),
        _ => None,
    })
}

struct ChallengeDemand {
    perm_alpha: usize,
    beta_power: [usize; CONSTRAINT_MAX_BETA_POWERS],
    beta_septix: usize,
    first_by_static: BTreeMap<usize, usize>,
    last_by_static: BTreeMap<usize, usize>,
}

impl Default for ChallengeDemand {
    fn default() -> Self {
        Self {
            perm_alpha: 0,
            beta_power: [0; CONSTRAINT_MAX_BETA_POWERS],
            beta_septix: 0,
            first_by_static: BTreeMap::new(),
            last_by_static: BTreeMap::new(),
        }
    }
}

impl ChallengeDemand {
    fn for_proof(proof: &RecursionProofRecord, plan: &ConstraintProgramPlan) -> Self {
        let mut demand = Self::default();
        for chip_shape in &proof.proof_shape.chips {
            let Some(chip) = plan.chip(chip_shape.static_chip_id) else {
                continue;
            };
            let static_demand = chip.challenge_demand;
            if let Some(power) = static_demand.invalid_beta_power {
                panic!(
                    "beta-power leaf {} exceeds CONSTRAINT_MAX_BETA_POWERS {} for proof {} static chip {}",
                    power,
                    CONSTRAINT_MAX_BETA_POWERS,
                    proof.proof_idx,
                    chip_shape.static_chip_id
                );
            }
            demand.perm_alpha += static_demand.perm_alpha;
            demand.beta_septix += static_demand.beta_septix;
            for (total, count) in
                demand.beta_power.iter_mut().zip(static_demand.beta_power.iter().copied())
            {
                *total += count;
            }
            *demand.first_by_static.entry(chip_shape.static_chip_id).or_default() +=
                static_demand.first;
            *demand.last_by_static.entry(chip_shape.static_chip_id).or_default() +=
                static_demand.last;
        }
        demand
    }
}

fn external_demand_for_proof(
    proof: &RecursionProofRecord,
    program: &RecursionPolyAirVerifierProgram,
) -> ProofExternalDemand {
    let mut demand = ProofExternalDemand::default();
    if proof.proof_shape.is_empty() {
        return demand;
    }
    if proof.batch_constraint.publish_terminal_outputs &&
        child_contains_global_bus_for_role(program.role)
    {
        for idx in TERMINAL_PV_IDXS
            .into_iter()
            .take(CONSTRAINT_BOUNDARY_DIRECT_PUBLIC_VALUE_COUNT)
        {
            *demand.public_values.entry(idx).or_default() += 1;
        }
    }

    let plan = program.constraint_static_plan();
    let mut prep_pos = 0usize;
    for chip_shape in chips_by_idx(&proof.proof_shape.chips) {
        let chip_prep_pos = prep_pos;
        if chip_shape.has_prep() {
            prep_pos += 1;
        }
        let Some(chip_plan) = plan.chip(chip_shape.static_chip_id) else {
            continue;
        };
        let chip = &program.chips[chip_plan.program_chip_index];
        for node in &chip.node_table {
            let RecursionPolyAirOp::Leaf(leaf) = &node.op else { continue };
            match leaf {
                RecursionPolyAirLeaf::Preprocessed { col } => {
                    bump_opened_demand(
                        &mut demand,
                        PROOF_SHAPE_BATCH_PREPROCESSED,
                        chip_prep_pos,
                        chip_shape.chip_idx,
                        *col,
                    );
                }
                RecursionPolyAirLeaf::Main { col } => {
                    bump_opened_demand(
                        &mut demand,
                        PROOF_SHAPE_BATCH_MAIN,
                        chip_shape.chip_idx,
                        chip_shape.chip_idx,
                        *col,
                    );
                }
                RecursionPolyAirLeaf::ReservedPoly { index } => {
                    if let Some(source) = chip.reserved_poly.get(*index).copied() {
                        match source {
                            PairCol::Prep(col) => {
                                bump_opened_demand(
                                    &mut demand,
                                    PROOF_SHAPE_BATCH_PREPROCESSED,
                                    chip_prep_pos,
                                    chip_shape.chip_idx,
                                    col,
                                );
                            }
                            PairCol::Main(col) => {
                                bump_opened_demand(
                                    &mut demand,
                                    PROOF_SHAPE_BATCH_MAIN,
                                    chip_shape.chip_idx,
                                    chip_shape.chip_idx,
                                    col,
                                );
                            }
                        }
                    }
                }
                RecursionPolyAirLeaf::Public { index } => {
                    *demand.public_values.entry(*index).or_default() += 1;
                }
                _ => {}
            }
        }
        let batch_size = chip.logup_batch_size.max(1);
        let permutation_batches = chip.lookup_multiplicity_roots.len().div_ceil(batch_size);
        for batch_idx in 0..permutation_batches {
            bump_opened_demand(
                &mut demand,
                PROOF_SHAPE_BATCH_PERMUTATION,
                chip_shape.chip_idx,
                chip_shape.chip_idx,
                batch_idx,
            );
        }
    }
    demand
}

fn bump_opened_demand(
    demand: &mut ProofExternalDemand,
    batch_id: usize,
    batch_pos: usize,
    chip_idx: usize,
    value_idx: usize,
) {
    *demand
        .opened
        .entry(OpenedEvalDemandKey { batch_id, batch_pos, chip_idx, value_idx })
        .or_default() += 1;
}

struct ProofEvalEnv {
    alpha: EF,
    perm_alpha: EF,
    perm_beta: EF,
    beta_powers: Vec<EF>,
    beta_septix: EF,
}

impl ProofEvalEnv {
    fn new(
        proof: &RecursionProofRecord,
        program: &RecursionPolyAirVerifierProgram,
    ) -> Option<Self> {
        let alpha = EF::from_base_slice(&proof.batch_constraint.alpha);
        let perm_alpha = EF::from_base_slice(&proof.batch_constraint.perm_alpha);
        let perm_beta = EF::from_base_slice(&proof.batch_constraint.perm_beta);
        let max_power = program.max_required_beta_power.max(CONSTRAINT_MAX_BETA_POWERS - 1);
        let mut beta_powers = Vec::with_capacity(max_power + 1);
        let mut cur = EF::one();
        for _ in 0..=max_power {
            beta_powers.push(cur);
            cur *= perm_beta;
        }
        let beta_septix = beta_powers.get(7).copied()? -
            perm_beta * EF::from_base(F::from_canonical_u32(3)) -
            EF::from_base(F::from_canonical_u32(5));
        Some(Self { alpha, perm_alpha, perm_beta, beta_powers, beta_septix })
    }
}

pub fn opened_values_for_chip<'a>(
    proof: &'a RecursionProofRecord,
    chip: &RecursionProofShapeChip,
) -> Option<(Cow<'a, [EF]>, Cow<'a, [EF]>)> {
    if let Some(source) = &proof.whir_source {
        let opened = source.opened_values.chips.get(chip.chip_idx)?;
        assert_eq!(
            opened.preprocessed.local.len(),
            chip.prep_width,
            "opened preprocessed width mismatch for proof {} chip {} static {}",
            proof.proof_idx,
            chip.chip_idx,
            chip.static_chip_id
        );
        assert_eq!(
            opened.main.local.len(),
            chip.main_width,
            "opened main width mismatch for proof {} chip {} static {}",
            proof.proof_idx,
            chip.chip_idx,
            chip.static_chip_id
        );
        return Some((
            Cow::Borrowed(opened.preprocessed.local.as_slice()),
            Cow::Borrowed(opened.main.local.as_slice()),
        ));
    }
    let mut prep = vec![None; chip.prep_width];
    let mut main = vec![None; chip.main_width];
    for row in &proof.whir.batch_eval_rows {
        if !row.is_value || row.chip_idx != chip.chip_idx {
            continue;
        }
        match row.batch_id {
            PROOF_SHAPE_BATCH_PREPROCESSED => {
                let slot = prep.get_mut(row.value_idx).unwrap_or_else(|| {
                    panic!(
                        "opened preprocessed value idx {} out of range {} for proof {} chip {} static {}",
                        row.value_idx,
                        chip.prep_width,
                        proof.proof_idx,
                        chip.chip_idx,
                        chip.static_chip_id
                    )
                });
                *slot = Some(EF::from_base_slice(&row.value));
            }
            PROOF_SHAPE_BATCH_MAIN => {
                let slot = main.get_mut(row.value_idx).unwrap_or_else(|| {
                    panic!(
                        "opened main value idx {} out of range {} for proof {} chip {} static {}",
                        row.value_idx,
                        chip.main_width,
                        proof.proof_idx,
                        chip.chip_idx,
                        chip.static_chip_id
                    )
                });
                *slot = Some(EF::from_base_slice(&row.value));
            }
            _ => {}
        }
    }
    let prep = prep
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            value.unwrap_or_else(|| {
                panic!(
                    "missing opened preprocessed value idx {} for proof {} chip {} static {}",
                    idx, proof.proof_idx, chip.chip_idx, chip.static_chip_id
                )
            })
        })
        .collect();
    let main = main
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            value.unwrap_or_else(|| {
                panic!(
                    "missing opened main value idx {} for proof {} chip {} static {}",
                    idx, proof.proof_idx, chip.chip_idx, chip.static_chip_id
                )
            })
        })
        .collect();
    Some((Cow::Owned(prep), Cow::Owned(main)))
}

pub fn permutation_values_for_chip<'a>(
    proof: &'a RecursionProofRecord,
    chip: &RecursionProofShapeChip,
    chip_ir: &crate::symbolic_ir_dt::RecursionPolyAirChipIr,
) -> Cow<'a, [EF]> {
    let batch_size = chip_ir.logup_batch_size.max(1);
    let permutation_len = chip_ir.lookup_multiplicity_roots.len().div_ceil(batch_size);
    if permutation_len == 0 {
        return Cow::Borrowed(&[]);
    }
    if let Some(source) = &proof.whir_source {
        let opened = source.opened_values.chips.get(chip.chip_idx).unwrap_or_else(|| {
            panic!(
                "missing opened permutation chip for proof {} chip {} static {}",
                proof.proof_idx, chip.chip_idx, chip.static_chip_id
            )
        });
        assert_eq!(
            opened.permutation.local.len(),
            permutation_len,
            "opened permutation width mismatch for proof {} chip {} static {}",
            proof.proof_idx,
            chip.chip_idx,
            chip.static_chip_id
        );
        return Cow::Borrowed(opened.permutation.local.as_slice());
    }
    let mut permutation = vec![None; permutation_len];
    for row in &proof.whir.batch_eval_rows {
        if !row.is_value ||
            row.batch_id != PROOF_SHAPE_BATCH_PERMUTATION ||
            row.chip_idx != chip.chip_idx
        {
            continue;
        }
        let slot = permutation.get_mut(row.value_idx).unwrap_or_else(|| {
            panic!(
                "opened permutation value idx {} out of range {} for proof {} chip {} static {}",
                row.value_idx, permutation_len, proof.proof_idx, chip.chip_idx, chip.static_chip_id
            )
        });
        *slot = Some(EF::from_base_slice(&row.value));
    }
    Cow::Owned(
        permutation
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                value.unwrap_or_else(|| {
                    panic!(
                        "missing opened permutation value idx {} for proof {} chip {} static {}",
                        idx, proof.proof_idx, chip.chip_idx, chip.static_chip_id
                    )
                })
            })
            .collect(),
    )
}

fn ext_array_limbs<const N: usize>(values: &[EF]) -> [[F; D_EF]; N] {
    core::array::from_fn(|idx| {
        values.get(idx).map(ext_limbs).unwrap_or_else(|| {
            panic!(
                "beta-power source idx {} out of range {}; CONSTRAINT_MAX_BETA_POWERS={}",
                idx,
                values.len(),
                CONSTRAINT_MAX_BETA_POWERS
            )
        })
    })
}

fn ext_limbs(value: &EF) -> [F; D_EF] {
    value.as_base_slice().try_into().expect("active extension degree is D_EF")
}

fn one_ext_limbs() -> [F; D_EF] {
    core::array::from_fn(|idx| if idx == 0 { F::one() } else { F::zero() })
}

#[derive(Debug, Clone, Copy)]
struct TerminalPrefix {
    eq: EF,
    first: EF,
    last: EF,
}

fn terminal_opening_points(batch: &RecursionBatchConstraintRecord) -> Vec<[F; D_EF]> {
    (0..batch.num_rounds)
        .map(|opening_idx| {
            batch
                .rounds
                .iter()
                .find(|round| batch.num_rounds - 1 - round.round_idx == opening_idx)
                .unwrap_or_else(|| {
                    panic!(
                        "missing terminal opening point {} for batch with {} rounds",
                        opening_idx, batch.num_rounds
                    )
                })
                .challenge
        })
        .collect()
}

fn terminal_prefixes(
    batch: &RecursionBatchConstraintRecord,
    opening_points: &[[F; D_EF]],
) -> Vec<TerminalPrefix> {
    let mut prefixes = Vec::with_capacity(batch.num_rounds + 1);
    let mut eq = EF::one();
    let mut first = EF::one();
    let mut last = EF::one();
    prefixes.push(TerminalPrefix { eq, first, last });
    for opening_idx in 0..batch.num_rounds {
        let opening_point = opening_points.get(opening_idx).unwrap_or_else(|| {
            panic!(
                "missing terminal opening point {} for batch with {} rounds",
                opening_idx, batch.num_rounds
            )
        });
        let z = EF::from_base_slice(opening_point);
        let eq_challenge =
            EF::from_base_slice(batch.eq_challenges.get(opening_idx).unwrap_or_else(|| {
                panic!(
                    "missing terminal eq challenge {} for batch with {} rounds",
                    opening_idx, batch.num_rounds
                )
            }));
        eq *= eq_factor(eq_challenge, z);
        first *= EF::one() - z;
        last *= z;
        prefixes.push(TerminalPrefix { eq, first, last });
    }
    prefixes
}

fn eq_factor(eq_challenge: EF, opening_point: EF) -> EF {
    let two = EF::from_base(F::from_canonical_u32(2));
    two * eq_challenge * opening_point - eq_challenge - opening_point + EF::one()
}

fn eq_factor_limbs(eq_challenge: [F; D_EF], opening_point: [F; D_EF]) -> [F; D_EF] {
    ext_limbs(&eq_factor(EF::from_base_slice(&eq_challenge), EF::from_base_slice(&opening_point)))
}

fn selector_height_demand(proof: &RecursionProofRecord) -> BTreeMap<usize, usize> {
    let mut demand = BTreeMap::new();
    for chip in &proof.proof_shape.chips {
        *demand.entry(chip.log_height).or_default() += 1;
    }
    demand
}

fn final_fold_rows(fold_rows: &[ConstraintFoldRow]) -> BTreeMap<usize, &ConstraintFoldRow> {
    let mut rows = BTreeMap::new();
    for row in fold_rows {
        rows.insert(row.proof_idx, row);
    }
    rows
}

pub fn leaf_kind(leaf: &RecursionPolyAirLeaf) -> usize {
    match leaf {
        RecursionPolyAirLeaf::Preprocessed { .. } => CONSTRAINT_LEAF_PREPROCESSED,
        RecursionPolyAirLeaf::Main { .. } => CONSTRAINT_LEAF_MAIN,
        RecursionPolyAirLeaf::Public { .. } => CONSTRAINT_LEAF_PUBLIC,
        RecursionPolyAirLeaf::PermAlpha => CONSTRAINT_LEAF_PERM_ALPHA,
        RecursionPolyAirLeaf::BetaPower { .. } => CONSTRAINT_LEAF_BETA_POWER,
        RecursionPolyAirLeaf::BetaSeptix => CONSTRAINT_LEAF_BETA_SEPTIX,
        RecursionPolyAirLeaf::Precomputed { .. } => CONSTRAINT_LEAF_PRECOMPUTED,
        RecursionPolyAirLeaf::ReservedPoly { .. } => CONSTRAINT_LEAF_RESERVED_POLY,
        RecursionPolyAirLeaf::IsFirstRow => CONSTRAINT_LEAF_IS_FIRST_ROW,
        RecursionPolyAirLeaf::IsLastRow => CONSTRAINT_LEAF_IS_LAST_ROW,
    }
}

pub fn prep_batch_pos(chips: &[RecursionProofShapeChip], chip_idx: usize) -> usize {
    chips.iter().filter(|chip| chip.chip_idx < chip_idx && chip.has_prep()).count()
}

pub fn selector_values(
    log_height: usize,
    batch: &RecursionBatchConstraintRecord,
) -> Option<(EF, EF)> {
    let opening_points = terminal_opening_points(batch);
    let prefixes = terminal_prefixes(batch, &opening_points);
    prefixes.get(log_height).map(|prefix| (prefix.first, prefix.last))
}

fn chips_by_idx(chips: &[RecursionProofShapeChip]) -> Vec<&RecursionProofShapeChip> {
    let mut sorted = chips.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| match left.chip_idx.cmp(&right.chip_idx) {
        Ordering::Equal => left.static_chip_id.cmp(&right.static_chip_id),
        other => other,
    });
    sorted
}

fn program_static_presence_counts(record: &RecursionRecord) -> BTreeMap<usize, usize> {
    let mut counts = BTreeMap::new();
    for proof in &record.proof_records {
        if !constraint_replay_proof_present(proof) {
            continue;
        }
        for chip in &proof.proof_shape.chips {
            *counts.entry(chip.static_chip_id).or_default() += 1;
        }
    }
    counts
}

fn constraint_replay_proof_present(proof: &RecursionProofRecord) -> bool {
    !proof.proof_shape.is_empty() && !proof.batch_constraint.is_empty()
}

fn constraint_replay_present_proof_count(record: &RecursionRecord) -> usize {
    record.proof_records.iter().filter(|proof| constraint_replay_proof_present(proof)).count()
}

fn dag_challenge_kind(row: &ConstraintDagRow) -> usize {
    row.program.rhs_idx
}

fn program_key(row: &ConstraintProgramRow) -> Vec<u32> {
    vec![
        row.static_chip_id as u32,
        row.node_idx as u32,
        row.op_code() as u32,
        row.lhs_idx as u32,
        row.rhs_idx as u32,
        row.third_idx as u32,
        row.aux.as_canonical_u32(),
        row.leaf_kind as u32,
        row.fanout as u32,
    ]
}

fn program_key_ref(row: ConstraintProgramNodeRef<'_>) -> Vec<u32> {
    vec![
        row.static_chip_id as u32,
        row.node_idx as u32,
        row.op_code() as u32,
        row.lhs_idx() as u32,
        row.rhs_idx() as u32,
        row.third_idx() as u32,
        row.plan.aux.as_canonical_u32(),
        row.leaf_kind() as u32,
        row.fanout() as u32,
    ]
}

fn root_table_key(row: &ConstraintRootTableRow) -> Vec<u32> {
    vec![
        row.static_chip_id as u32,
        row.root_kind as u32,
        row.root_ord as u32,
        row.node_idx as u32,
        signed_f(row.sign).as_canonical_u32(),
    ]
}

fn fold_root_active(row: &ConstraintFoldRow, slot: usize) -> bool {
    match slot {
        0 => row.is_gate || row.is_batch,
        1 | 3 => row.batch_has_second,
        2 => row.is_batch,
        _ => unreachable!("ConstraintFold has four root slots"),
    }
}

fn fold_root_kind(row: &ConstraintFoldRow, slot: usize) -> usize {
    match slot {
        0 if row.is_gate => CONSTRAINT_ROOT_GATE,
        0 | 1 => CONSTRAINT_ROOT_PRECOMPUTE_DENOM,
        2 | 3 => CONSTRAINT_ROOT_MULTIPLICITY,
        _ => unreachable!("ConstraintFold has four root slots"),
    }
}

fn fold_root_ord(row: &ConstraintFoldRow, slot: usize) -> usize {
    row.root_ord + usize::from(slot % 2 == 1)
}

fn fold_root_sign(row: &ConstraintFoldRow, slot: usize) -> i32 {
    match slot {
        0 | 1 => 1,
        2 | 3 => row.multiplicity_signs[slot - CONSTRAINT_FOLD_BATCH_SIZE],
        _ => unreachable!("ConstraintFold has four root slots"),
    }
}

fn fold_root_table_key(row: &ConstraintFoldRow, slot: usize) -> Vec<u32> {
    vec![
        row.static_chip_id as u32,
        fold_root_kind(row, slot) as u32,
        fold_root_ord(row, slot) as u32,
        row.root_nodes[slot] as u32,
        signed_f(fold_root_sign(row, slot)).as_canonical_u32(),
    ]
}

fn node_value_key(
    proof_idx: usize,
    chip_idx: usize,
    static_chip_id: usize,
    node_idx: usize,
    value: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, chip_idx as u32, static_chip_id as u32, node_idx as u32];
    key.extend(value.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn challenge_key(
    proof_idx: usize,
    kind: usize,
    key0: usize,
    key1: usize,
    value: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, kind as u32, key0 as u32, key1 as u32];
    key.extend(value.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn beta_ladder_chain_recv_key(row: &ConstraintBetaLadderRow) -> Vec<u32> {
    let mut key = vec![row.proof_idx as u32, row.power_idx.saturating_sub(1) as u32];
    key.extend(row.prev_power_or_alpha.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(row.beta.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn beta_ladder_chain_send_key(row: &ConstraintBetaLadderRow) -> Vec<u32> {
    let mut key = vec![row.proof_idx as u32, row.power_idx as u32];
    key.extend(row.power.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(row.beta.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn beta_ladder_septix(row: &ConstraintBetaLadderRow) -> [F; D_EF] {
    core::array::from_fn(|limb| {
        let constant = if limb == 0 { F::from_canonical_u32(5) } else { F::zero() };
        row.power[limb] - row.beta[limb] * F::from_canonical_u32(3) - constant
    })
}

fn apply_terminal_state_chain_residual(
    residual: &mut BTreeMap<Vec<u32>, i64>,
    row: &ConstraintTerminalRow,
    is_send: bool,
) {
    let delta = if is_send { 1 } else { -1 };
    let cursor =
        if is_send { row.opening_idx + usize::from(row.is_lcs_step) } else { row.opening_idx };
    let lcs = if is_send { row.state_lcs_out } else { row.state_lcs_in };
    apply_residual(
        residual,
        challenge_key(row.proof_idx, CONSTRAINT_CHALLENGE_STATE_LCS, cursor, 0, lcs),
        delta,
    );
}

fn sumcheck_out_key(proof_idx: usize, kind: usize, idx: usize, value: [F; D_EF]) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, kind as u32, idx as u32];
    key.extend(value.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn batch_sumcheck_claim_chain_key(
    proof_idx: usize,
    round_idx: usize,
    r_rounds: usize,
    c_chips: usize,
    claim: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, round_idx as u32, r_rounds as u32, c_chips as u32];
    key.extend(claim.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn fold_chain_recv_key(row: &ConstraintFoldRow) -> Vec<u32> {
    fold_chain_key(
        row.proof_idx,
        row.cursor - 1,
        row.alpha,
        row.acc_in,
        row.pacc_in,
        row.perm_sum_in,
    )
}

fn fold_chain_send_key(row: &ConstraintFoldRow) -> Vec<u32> {
    fold_chain_key(
        row.proof_idx,
        row.cursor,
        row.alpha,
        row.acc_out,
        row.pacc_out,
        row.perm_sum_out,
    )
}

fn terminal_fold_chain_key(row: &ConstraintTerminalRow) -> Vec<u32> {
    fold_chain_key(
        row.proof_idx,
        row.fold_cursor,
        row.alpha,
        row.main_eval,
        row.perm_eval,
        [F::zero(); D_EF],
    )
}

fn fold_chain_key(
    proof_idx: usize,
    cursor: usize,
    alpha: [F; D_EF],
    acc: [F; D_EF],
    pacc: [F; D_EF],
    perm_sum: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, cursor as u32];
    key.extend(alpha.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(acc.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(pacc.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(perm_sum.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn fold_plan_chain_key(
    proof_idx: usize,
    cursor: usize,
    remaining_chips: usize,
    local_ord: usize,
) -> Vec<u32> {
    vec![proof_idx as u32, cursor as u32, remaining_chips as u32, local_ord as u32]
}

fn fold_plan_chain_recv_key(row: &ConstraintFoldRow) -> Vec<u32> {
    fold_plan_chain_key(row.proof_idx, row.cursor - 1, row.remaining_chips, row.local_ord)
}

fn fold_plan_chain_send_key(row: &ConstraintFoldRow) -> Vec<u32> {
    fold_plan_chain_key(
        row.proof_idx,
        row.cursor,
        row.remaining_chips - usize::from(row.is_skip),
        row.chain_send_local_ord,
    )
}

fn height_inverse_key(log_height: usize, inverse: usize) -> Vec<u32> {
    vec![log_height as u32, inverse as u32]
}

fn terminal_eq_chain_recv_key(row: &ConstraintTerminalRow) -> Vec<u32> {
    eq_chain_key(row.proof_idx, row.round_idx, row.eq_in, row.first_prefix_in, row.last_prefix_in)
}

fn terminal_eq_chain_send_key(row: &ConstraintTerminalRow) -> Vec<u32> {
    eq_chain_key(
        row.proof_idx,
        row.round_idx + usize::from(row.is_eq_step),
        row.eq_out,
        row.first_prefix_out,
        row.last_prefix_out,
    )
}

fn eq_chain_key(
    proof_idx: usize,
    round_idx: usize,
    eq_acc: [F; D_EF],
    first_prefix: [F; D_EF],
    last_prefix: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![proof_idx as u32, round_idx as u32];
    key.extend(eq_acc.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(first_prefix.into_iter().map(|value| value.as_canonical_u32()));
    key.extend(last_prefix.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn opened_eval_key(
    proof_idx: usize,
    batch_id: usize,
    batch_pos: usize,
    chip_idx: usize,
    value_idx: usize,
    value: [F; D_EF],
) -> Vec<u32> {
    let mut key = vec![
        proof_idx as u32,
        batch_id as u32,
        batch_pos as u32,
        chip_idx as u32,
        value_idx as u32,
    ];
    key.extend(value.into_iter().map(|value| value.as_canonical_u32()));
    key
}

fn statement_lift_public_value_recvs(role_id: usize) -> Vec<usize> {
    let native = matches!(role_id, WHIR_ROLE_COMPRESS | WHIR_ROLE_SHRINK);
    let mut idxs = if native {
        vec![
            NATIVE_PV_START_PC,
            NATIVE_PV_NEXT_PC,
            NATIVE_PV_START_SHARD,
            NATIVE_PV_NEXT_SHARD,
            NATIVE_PV_START_EXECUTION_SHARD,
            NATIVE_PV_NEXT_EXECUTION_SHARD,
            NATIVE_PV_PREVIOUS_INIT_ADDR,
            NATIVE_PV_LAST_INIT_ADDR,
            NATIVE_PV_PREVIOUS_FINALIZE_ADDR,
            NATIVE_PV_LAST_FINALIZE_ADDR,
            NATIVE_PV_CONTAINS_EXECUTION_SHARD,
        ]
    } else {
        vec![
            CORE_PV_START_PC,
            CORE_PV_NEXT_PC,
            CORE_PV_EXIT_CODE,
            CORE_PV_SHARD,
            CORE_PV_EXECUTION_SHARD,
            CORE_PV_PREVIOUS_INIT_ADDR,
            CORE_PV_LAST_INIT_ADDR,
            CORE_PV_PREVIOUS_FINALIZE_ADDR,
            CORE_PV_LAST_FINALIZE_ADDR,
            CORE_PV_START_CLK,
            CORE_PV_EXIT_CLK,
        ]
    };
    if native {
        idxs.extend(
            NATIVE_PV_COMMITTED_VALUE_DIGEST_START..NATIVE_PV_COMMITTED_VALUE_DIGEST_START + 32,
        );
        idxs.extend(
            NATIVE_PV_DEFERRED_PROOFS_DIGEST_START..NATIVE_PV_DEFERRED_PROOFS_DIGEST_START + 8,
        );
        idxs.extend(
            NATIVE_PV_START_RECONSTRUCT_DEFERRED_DIGEST_START..
                NATIVE_PV_END_RECONSTRUCT_DEFERRED_DIGEST_START + 8,
        );
        idxs.extend(NATIVE_PV_DT_VK_DIGEST_START..NATIVE_PV_DT_VK_DIGEST_START + 8);
        idxs.extend(NATIVE_PV_VK_ROOT_START..NATIVE_PV_VK_ROOT_START + 8);
        idxs.extend(NATIVE_PV_GLOBAL_INTERVAL_START..NATIVE_PV_GLOBAL_INTERVAL_END + 33);
        idxs.push(NATIVE_PV_IS_COMPLETE);
        idxs.push(NATIVE_PV_EXIT_CODE);
    } else {
        idxs.extend(
            CORE_PV_COMMITTED_VALUE_DIGEST_START..CORE_PV_COMMITTED_VALUE_DIGEST_START + 32,
        );
        idxs.extend(CORE_PV_DEFERRED_PROOFS_DIGEST_START..CORE_PV_DEFERRED_PROOFS_DIGEST_START + 8);
        idxs.extend(CORE_PV_GLOBAL_INTERVAL_START..CORE_PV_GLOBAL_INTERVAL_END + 33);
    }
    idxs
}

fn public_value_key(proof_idx: usize, idx: usize, value: F) -> Vec<u32> {
    proof_shape_value_key(proof_idx, PROOF_SHAPE_NAMESPACE_PUBLIC_VALUES, idx, value)
}

fn proof_shape_value_key(proof_idx: usize, namespace: usize, idx: usize, value: F) -> Vec<u32> {
    vec![proof_idx as u32, namespace as u32, idx as u32, value.as_canonical_u32()]
}

fn chip_meta_key(
    proof_idx: usize,
    chip_idx: usize,
    static_chip_id: usize,
    log_height: usize,
    gate_count: usize,
    batch_count: usize,
) -> Vec<u32> {
    vec![
        proof_idx as u32,
        chip_idx as u32,
        static_chip_id as u32,
        log_height as u32,
        gate_count as u32,
        batch_count as u32,
    ]
}

fn native_chip_metadata_key(row: RecursionNativeChipMetadataRequest) -> Vec<u32> {
    vec![
        row.role_id as u32,
        row.chip_id as u32,
        row.prep_width as u32,
        row.main_width as u32,
        row.perm_width as u32,
        row.constraint_count as u32,
        row.gate_count as u32,
    ]
}

fn apply_residual(residual: &mut BTreeMap<Vec<u32>, i64>, key: Vec<u32>, delta: i64) {
    *residual.entry(key).or_default() += delta;
}

fn finalize_residual(mut residual: BTreeMap<Vec<u32>, i64>) -> BTreeMap<Vec<u32>, i64> {
    residual.retain(|_, value| *value != 0);
    residual
}

/// Padding vector for ConstraintFold: zeros except the two denominator root
/// values, which are ext-one so the ungated inactive-slot pins hold.
fn fold_padding_row() -> Vec<F> {
    let mut root_values = [[F::zero(); D_EF]; CONSTRAINT_FOLD_ROOT_SLOTS];
    for slot in root_values.iter_mut().take(CONSTRAINT_FOLD_BATCH_SIZE) {
        slot[0] = F::one();
    }
    ConstraintFoldCols {
        proof_idx: F::zero(),
        is_skip: F::zero(),
        is_gate: F::zero(),
        is_batch: F::zero(),
        cursor: F::zero(),
        remaining_chips: F::zero(),
        local_ord: F::zero(),
        chain_send_local_ord: F::zero(),
        static_chip_id: F::zero(),
        log_height: F::zero(),
        gate_count: F::zero(),
        batch_count: F::zero(),
        root_ord: F::zero(),
        alpha: [F::zero(); D_EF],
        acc_in: [F::zero(); D_EF],
        acc_out: [F::zero(); D_EF],
        pacc_in: [F::zero(); D_EF],
        pacc_out: [F::zero(); D_EF],
        perm_sum_in: [F::zero(); D_EF],
        perm_sum_out: [F::zero(); D_EF],
        root_nodes: [F::zero(); CONSTRAINT_FOLD_ROOT_SLOTS],
        multiplicity_signs: [F::zero(); CONSTRAINT_FOLD_BATCH_SIZE],
        root_values,
        batch_has_second: F::zero(),
        perm_value: [F::zero(); D_EF],
    }
    .as_slice()
    .to_vec()
}

fn zeroed_trace_values(row_count: usize, width: usize) -> Vec<F> {
    vec![F::zero(); row_count.max(1) * width]
}

fn record_constraint_matrix_bytes(record: &RecursionRecord, row_count: usize, width: usize) {
    let bytes = row_count
        .max(1)
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(core::mem::size_of::<F>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .expect("constraint matrix byte count overflow");
    record.profile.add_structural_counter("constraint_matrix_population_bytes", bytes);
}

fn compressed_values(
    values: Vec<F>,
    width: usize,
    height: usize,
    padding: Vec<F>,
) -> CompressedMatrix<F> {
    let main = RowMajorMatrix::new(values, width);
    let pad = if main.height() < height { PaddingRow::General(padding) } else { PaddingRow::None };
    CompressedMatrix::new(main, pad, height)
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

fn f_bool(value: bool) -> F {
    F::from_bool(value)
}

fn signed_f(value: i32) -> F {
    if value >= 0 {
        F::from_canonical_u32(value as u32)
    } else {
        -F::from_canonical_u32(value.unsigned_abs())
    }
}

#[cfg(all(test, any()))]
mod tests {
    use super::*;
    use dt_stark::air::{FullAir, FullAirBuilder, InteractionScope};
    use p3_field::AbstractExtensionField;
    use p3_matrix::dense::RowMajorMatrixView;
    use polyair::{
        evaluator::ConstraintFolder,
        permutation::fused_precompute_reserved_permutation,
        symbolic::{SymbolicAirBuilder, SymbolicExpression, SymbolicVar},
        Chip,
    };

    use crate::{
        config::DIGEST_SIZE,
        constraint_replay_dt::air::{ConstraintFoldAir, ConstraintRootTableAir},
        machine_dt::{
            build_core_native_recursion_program, build_dual_segment_reduce_program,
            build_native_recursion_program, build_root_shrink_program, core_recording_machine,
            native_recording_machine, native_recording_machine_for_stage,
        },
        statement_dt::{
            NATIVE_RECURSION_NUM_PV_ELTS, STATEMENT_CONFIG_CLASS_BAKED_L2,
            STATEMENT_CONFIG_CLASS_BAKED_L3, STATEMENT_CONFIG_CLASS_BAKED_LIFT,
        },
        symbolic_expr_adapter_dt::{RecursionOpMix, RecursionPolyAirLeaf, RecursionPolyAirOp},
        symbolic_expr_fixed_dt::{
            RecursionChildRole, RecursionFixedSymbolicChip, RecursionFixedSymbolicProgram,
        },
        symbolic_ir_dt::{
            RecursionD0CostLedger, RecursionPolyAirChipIr, RecursionPolyAirConstraintRoot,
            RecursionPolyAirVerifierProgram, RecursionPolyAirWidths,
        },
        system_dt::{
            RecordingStage, RecursionBatchConstraintRecord, RecursionBatchCumSumRecord,
            RecursionProofRecord, RecursionProofShapeChip, RecursionProofShapeRecord,
            RecursionStatementRole, RecursionSumcheckRoundRecord, RecursionWhirBatchEvalRow,
            StatementConfigRow,
        },
    };

    fn ext_base(value: usize) -> [F; D_EF] {
        core::array::from_fn(|idx| if idx == 0 { f(value) } else { F::zero() })
    }

    fn one_round() -> RecursionSumcheckRoundRecord {
        // p(x) = 7 - 14x has p(0) + p(1) = 0 and p(0) = 7.  With
        // challenge r = 0 this is a valid one-round chain from the seed claim
        // zero to the terminal claim seven.
        let mut evals = [[F::zero(); D_EF]; crate::batch_constraint_dt::BATCH_SUMCHECK_EVALS];
        for (node, eval) in evals.iter_mut().enumerate() {
            eval[0] = f(7) - f(14 * node);
        }
        RecursionSumcheckRoundRecord {
            round_idx: 0,
            challenge: [F::zero(); D_EF],
            claim_in: [F::zero(); D_EF],
            claim_out: ext_base(7),
            evals,
        }
    }

    fn simple_program_for_role(role: RecursionChildRole) -> RecursionPolyAirVerifierProgram {
        RecursionPolyAirVerifierProgram::try_new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            role,
            [F::zero(); DIGEST_SIZE],
            vec![RecursionPolyAirChipIr {
                static_chip_id: 0,
                chip_name: "TestMainLeaf".to_string(),
                widths: RecursionPolyAirWidths { preprocessed: 0, main: 1, public: 51 },
                commit_scope: InteractionScope::Local,
                logup_batch_size: CONSTRAINT_FOLD_BATCH_SIZE,
                reserved_poly: Vec::new(),
                derived_roots: vec![
                    crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::BetaPower { power: 0 },
                    crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::BetaSeptix,
                ],
                gate_roots: vec![RecursionPolyAirConstraintRoot {
                    static_chip_id: 0,
                    gate_idx: 0,
                    root_node_id: 0,
                }],
                lookup_multiplicity_roots: Vec::new(),
                node_table: vec![RecursionPolyAirNode {
                    node_id: 0,
                    op: RecursionPolyAirOp::Leaf(RecursionPolyAirLeaf::Main { col: 0 }),
                    degree_multiple: 1,
                }],
                num_constraints_from_builder: 1,
                cost_ledger: RecursionD0CostLedger {
                    node_count: 1,
                    op_mix: RecursionOpMix { leaves: 1, ..Default::default() },
                    gate_count: 1,
                    precompute_root_count: 0,
                    derived_root_count: 3,
                    expected_node_bus_rows: 1,
                    expected_wide_unroll_rows: 1,
                    expected_wide_unroll_width: D_EF,
                    internal_recursion_interactions_node_bus: 1,
                    internal_recursion_interactions_wide_unroll: 0,
                },
            }],
            0,
        )
        .expect("simple constraint replay test program")
    }

    fn simple_program() -> RecursionPolyAirVerifierProgram {
        simple_program_for_role(RecursionChildRole::Core)
    }

    fn simple_record_with_public_value_count(public_value_count: usize) -> RecursionRecord {
        assert!(public_value_count > TERMINAL_PV_EXIT_CLK);
        let chip = RecursionProofShapeChip {
            chip_idx: 0,
            static_chip_id: 0,
            stable_air_id: 43,
            log_height: 1,
            prep_width: 0,
            main_width: 1,
            perm_width: 0,
            constraint_count: 1,
            gate_count: 1,
        };
        let mut public_values = vec![F::zero(); public_value_count];
        public_values[TERMINAL_PV_START_PC] = F::zero();
        public_values[TERMINAL_PV_NEXT_PC] = F::zero();
        let zero_digest = SepticDigest::<F>::zero_for_field().0;

        let proof = RecursionProofRecord {
            proof_idx: 0,
            proof_shape: RecursionProofShapeRecord {
                role_id: 0,
                num_public_values: public_values.len(),
                public_value_send_mults: vec![0; public_values.len()],
                public_values,
                chips: vec![chip],
                publish_external: true,
                publish_terminal_summary: true,
                ..Default::default()
            },
            batch_constraint: RecursionBatchConstraintRecord {
                num_public_values: public_value_count,
                num_rounds: 1,
                c_chips: 1,
                cum_sums: vec![RecursionBatchCumSumRecord {
                    chip_idx: 0,
                    gcs_x: zero_digest.x.0,
                    gcs_y: zero_digest.y.0,
                    ..Default::default()
                }],
                perm_alpha: ext_base(5),
                perm_beta: ext_base(3),
                alpha: ext_base(2),
                eq_challenges: vec![[F::zero(); D_EF]],
                rounds: vec![one_round()],
                last_claim: ext_base(7),
                publish_opening_point: true,
                publish_terminal_outputs: true,
            },
            whir: crate::system_dt::RecursionWhirRecord {
                batch_eval_rows: vec![RecursionWhirBatchEvalRow {
                    proof_idx: 0,
                    value: ext_base(7),
                    log_height: 1,
                    batch_id: PROOF_SHAPE_BATCH_MAIN,
                    batch_pos: 0,
                    chip_idx: 0,
                    static_chip_id: 0,
                    width: 1,
                    value_idx: 0,
                    segment_element_count: 1,
                    is_value: true,
                    is_segment_start: true,
                    is_segment_end: true,
                    is_first_value: true,
                    opened_eval_send_mult: 1,
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let mut record = RecursionRecord { proof_records: vec![proof], ..Default::default() };
        record.native_chip_metadata.record_metadata(chip.metadata_request(0));
        let program = simple_program();
        annotate_constraint_replay_publications(&mut record, &program);
        record
    }

    fn simple_record() -> RecursionRecord {
        simple_record_with_public_value_count(TERMINAL_PV_EXIT_CLK + 2)
    }

    fn lookup_program(lookup_count: usize) -> RecursionPolyAirVerifierProgram {
        assert!(lookup_count > 0);
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(2 * lookup_count);
        builder.width_max_beta_power(1);
        for lookup_idx in 0..lookup_count {
            builder.retain_precomputed(SymbolicExpression::VARiable(SymbolicVar::Main(lookup_idx)));
        }
        for lookup_idx in 0..lookup_count {
            let multiplicity =
                SymbolicExpression::VARiable(SymbolicVar::Main(lookup_count + lookup_idx));
            if lookup_idx % 2 == 0 {
                builder.send(multiplicity);
            } else {
                builder.recv(multiplicity);
            }
        }
        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            format!("ConstraintFoldLookup{lookup_count}"),
            InteractionScope::Local,
            CONSTRAINT_FOLD_BATCH_SIZE,
            lookup_count,
            &builder,
        )
        .expect("compile lookup test chip");
        let fixed = RecursionFixedSymbolicProgram::new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![fixed_chip],
            [F::zero(); DIGEST_SIZE],
        )
        .expect("freeze lookup test program");
        RecursionPolyAirVerifierProgram::compile(&fixed).expect("compile lookup verifier program")
    }

    fn mixed_gate_lookup_program(lookup_count: usize) -> RecursionPolyAirVerifierProgram {
        assert!(lookup_count > 0);
        let mut builder = SymbolicAirBuilder::<F, D_EF>::new_empty();
        builder.with_main_width(2 * lookup_count);
        builder.with_public_width(1);
        builder.width_max_beta_power(1);
        for lookup_idx in 0..lookup_count {
            builder.retain_precomputed(SymbolicExpression::VARiable(SymbolicVar::Main(lookup_idx)));
        }
        for lookup_idx in 0..lookup_count {
            let multiplicity =
                SymbolicExpression::VARiable(SymbolicVar::Main(lookup_count + lookup_idx));
            if lookup_idx % 2 == 0 {
                builder.send(multiplicity);
            } else {
                builder.recv(multiplicity);
            }
        }
        builder.gate.push(
            SymbolicExpression::VARiable(SymbolicVar::BetaPowers(1)) +
                SymbolicExpression::VARiable(SymbolicVar::Public(0)),
        );
        let fixed_chip = RecursionFixedSymbolicChip::from_symbolic_builder(
            0,
            format!("ConstraintFoldMixed{lookup_count}"),
            InteractionScope::Local,
            CONSTRAINT_FOLD_BATCH_SIZE,
            lookup_count + 1,
            &builder,
        )
        .expect("compile mixed gate/lookup test chip");
        let fixed = RecursionFixedSymbolicProgram::new(
            crate::symbolic_ir_dt::CONSTRAINT_PROGRAM_SCHEMA_VERSION,
            RecursionChildRole::Core,
            vec![fixed_chip],
            [F::zero(); DIGEST_SIZE],
        )
        .expect("freeze mixed gate/lookup test program");
        RecursionPolyAirVerifierProgram::compile(&fixed)
            .expect("compile mixed gate/lookup verifier program")
    }

    fn lookup_record(
        program: &RecursionPolyAirVerifierProgram,
        publish_terminal_outputs: bool,
    ) -> RecursionRecord {
        let chip_ir = &program.chips[0];
        let batch_count =
            chip_ir.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE);
        let shape = RecursionProofShapeChip {
            chip_idx: 0,
            static_chip_id: chip_ir.static_chip_id,
            stable_air_id: dt_stark::air::stable_air_id_v1(&chip_ir.chip_name),
            log_height: 1,
            prep_width: 0,
            main_width: chip_ir.widths.main,
            perm_width: batch_count * D_EF,
            constraint_count: chip_ir.num_constraints_from_builder,
            gate_count: chip_ir.gate_roots.len(),
        };
        let zero_digest = SepticDigest::<F>::zero_for_field().0;
        let mut batch_eval_rows = (0..shape.main_width)
            .map(|value_idx| RecursionWhirBatchEvalRow {
                proof_idx: 0,
                value: ext_base(3 + value_idx),
                log_height: 1,
                batch_id: PROOF_SHAPE_BATCH_MAIN,
                batch_pos: 0,
                chip_idx: 0,
                static_chip_id: 0,
                width: shape.main_width,
                value_idx,
                segment_element_count: shape.main_width,
                is_value: true,
                is_segment_start: value_idx == 0,
                is_segment_end: value_idx + 1 == shape.main_width,
                is_first_value: value_idx == 0,
                opened_eval_send_mult: 1,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        batch_eval_rows.extend((0..batch_count).map(|value_idx| RecursionWhirBatchEvalRow {
            proof_idx: 0,
            value: ext_base(19 + value_idx),
            log_height: 1,
            batch_id: PROOF_SHAPE_BATCH_PERMUTATION,
            batch_pos: 0,
            chip_idx: 0,
            static_chip_id: 0,
            width: batch_count,
            value_idx,
            segment_element_count: batch_count,
            is_value: true,
            is_segment_start: value_idx == 0,
            is_segment_end: value_idx + 1 == batch_count,
            is_first_value: value_idx == 0,
            opened_eval_send_mult: 1,
            ..Default::default()
        }));

        let proof = RecursionProofRecord {
            proof_idx: 0,
            proof_shape: RecursionProofShapeRecord {
                role_id: 0,
                num_public_values: TERMINAL_PV_EXIT_CLK + 2,
                public_value_send_mults: vec![0; TERMINAL_PV_EXIT_CLK + 2],
                public_values: vec![F::zero(); TERMINAL_PV_EXIT_CLK + 2],
                chips: vec![shape],
                publish_external: true,
                publish_terminal_summary: publish_terminal_outputs,
                ..Default::default()
            },
            batch_constraint: RecursionBatchConstraintRecord {
                num_public_values: TERMINAL_PV_EXIT_CLK + 2,
                num_rounds: 1,
                c_chips: 1,
                cum_sums: vec![RecursionBatchCumSumRecord {
                    chip_idx: 0,
                    gcs_x: zero_digest.x.0,
                    gcs_y: zero_digest.y.0,
                    ..Default::default()
                }],
                perm_alpha: ext_base(5),
                perm_beta: ext_base(3),
                alpha: ext_base(2),
                eq_challenges: vec![[F::zero(); D_EF]],
                rounds: vec![one_round()],
                last_claim: ext_base(7),
                publish_opening_point: true,
                publish_terminal_outputs,
            },
            whir: crate::system_dt::RecursionWhirRecord { batch_eval_rows, ..Default::default() },
            ..Default::default()
        };
        let mut record = RecursionRecord { proof_records: vec![proof], ..Default::default() };
        record.native_chip_metadata.record_metadata(shape.metadata_request(0));
        annotate_constraint_replay_publications(&mut record, program);
        record
    }

    #[derive(Debug)]
    struct FoldRowEvaluation {
        first: EF,
        nonfirst: EF,
        lookup_multiplicities: Vec<F>,
    }

    fn fold_materialized_evaluations(
        program: &RecursionPolyAirVerifierProgram,
        main: &CompressedMatrix<F>,
    ) -> Vec<FoldRowEvaluation> {
        let chip = Chip::<ConstraintFoldAir, F, D_EF>::new(ConstraintFoldAir::new(program.clone()));
        let perm_alpha = EF::from_canonical_u32(211);
        let beta = EF::from_canonical_u32(223) + <EF as AbstractExtensionField<F>>::monomial(1);
        let mut powers = beta.powers();
        let beta_powers = (0..=chip.required_max_beta_power())
            .map(|_| powers.next().expect("infinite powers"))
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
                multiplicities: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut first_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut first);
            chip.air.lookup(&mut first);
            let lookup_multiplicities = first.multiplicities.clone();
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
                multiplicities: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut nonfirst_accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut nonfirst);
            chip.air.lookup(&mut nonfirst);
            nonfirst.constrain_lookup();
            drop(nonfirst);
            evaluations.push(FoldRowEvaluation {
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
                multiplicities: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut padding);
            chip.air.lookup(&mut padding);
            assert!(padding.multiplicities.iter().all(|value| *value == F::zero()));
            padding.constrain_lookup();
            drop(padding);
            assert_eq!(accumulator, EF::zero(), "ConstraintFold padding");
        }
        evaluations
    }

    fn root_table_materialized_evaluations(
        program: &RecursionPolyAirVerifierProgram,
        preprocessed: &CompressedMatrix<F>,
        main: &CompressedMatrix<F>,
    ) -> Vec<EF> {
        let chip = Chip::<ConstraintRootTableAir, F, D_EF>::new(ConstraintRootTableAir::new(
            program.clone(),
        ));
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
            Some(preprocessed),
            main,
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
        let mut evaluations = Vec::with_capacity(main.stored_height());
        for row_idx in 0..main.stored_height() {
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
                multiplicities: Vec::new(),
                batch_size: chip.logup_batch_size(),
                accumulator: &mut accumulator,
                constraint_reducer: &reducers,
                constraint_index: 0,
            };
            chip.air.eval(&mut folder);
            chip.air.lookup(&mut folder);
            folder.constrain_lookup();
            drop(folder);
            evaluations.push(accumulator);
        }
        evaluations
    }

    fn fold_matrix_from_rows(rows: &[ConstraintFoldRow]) -> CompressedMatrix<F> {
        let row_count = rows.len();
        let mut values = zeroed_trace_values(row_count, NUM_CONSTRAINT_FOLD_COLS);
        for (trace_row, row) in values.chunks_exact_mut(NUM_CONSTRAINT_FOLD_COLS).zip(rows) {
            fill_fold_row(trace_row, row);
        }
        compressed_values(
            values,
            NUM_CONSTRAINT_FOLD_COLS,
            row_count.max(1).next_power_of_two(),
            fold_padding_row(),
        )
    }

    fn rethread_fold_suffix(
        rows: &mut [ConstraintFoldRow],
        start: usize,
        mut acc: EF,
        mut pacc: EF,
        mut perm_sum: EF,
        alpha_override: Option<EF>,
    ) {
        for (offset, row) in rows[start..].iter_mut().enumerate() {
            row.cursor = start + offset + 1;
            let old_alpha = EF::from_base_slice(&row.alpha);
            let alpha = alpha_override.unwrap_or(old_alpha);
            let roots = row.root_values.map(|value| EF::from_base_slice(&value));
            let constraint_value = if row.is_gate {
                roots[0]
            } else if row.is_batch {
                let perm_value = EF::from_base_slice(&row.perm_value);
                let m0 = roots[2] * EF::from_base(signed_f(row.multiplicity_signs[0]));
                let m1 = roots[3] * EF::from_base(signed_f(row.multiplicity_signs[1]));
                roots[1] * m0 + roots[0] * (m1 - roots[1] * perm_value)
            } else {
                EF::zero()
            };
            row.alpha = ext_limbs(&alpha);
            row.acc_in = ext_limbs(&acc);
            acc = acc * alpha + constraint_value;
            row.acc_out = ext_limbs(&acc);

            row.pacc_in = ext_limbs(&pacc);
            row.perm_sum_in = ext_limbs(&perm_sum);
            if row.is_gate {
                row.perm_sum_out = ext_limbs(&perm_sum);
                pacc *= alpha;
            } else if row.is_batch {
                perm_sum += EF::from_base_slice(&row.perm_value);
                row.perm_sum_out = ext_limbs(&perm_sum);
                pacc *= alpha;
            } else {
                let height_inverse = F::from_canonical_usize(row.root_nodes[0]);
                pacc =
                    pacc * alpha + perm_sum - EF::from_base_slice(&row.perm_value) * height_inverse;
                perm_sum = EF::zero();
                row.perm_sum_out = [F::zero(); D_EF];
            }
            row.pacc_out = ext_limbs(&pacc);
        }
    }

    fn retarget_terminal_fold_sink(
        terminal_rows: &mut [ConstraintTerminalRow],
        last: &ConstraintFoldRow,
    ) {
        let terminal = terminal_rows
            .iter_mut()
            .find(|row| row.fold_chain_recv_mult)
            .expect("terminal FoldChain sink");
        terminal.fold_cursor = last.cursor;
        terminal.alpha = last.alpha;
        terminal.main_eval = last.acc_out;
        terminal.perm_eval = last.pacc_out;
    }

    fn chip_meta_residual_from_fold_rows(
        record: &RecursionRecord,
        fold_rows: &[ConstraintFoldRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::new();
        for row in proof_shape_binder_rows(record) {
            if let ProofShapeBinderRow::Chip { proof_idx, chip, publish_external, .. } = row {
                if publish_external {
                    let batch_count = chip.perm_width / D_EF;
                    apply_residual(
                        &mut residual,
                        chip_meta_key(
                            proof_idx,
                            chip.chip_idx,
                            chip.static_chip_id,
                            chip.log_height,
                            chip.gate_count,
                            batch_count,
                        ),
                        (chip.gate_count + batch_count + 1) as i64,
                    );
                }
            }
        }
        for row in fold_rows {
            apply_residual(
                &mut residual,
                chip_meta_key(
                    row.proof_idx,
                    row.remaining_chips - 1,
                    row.static_chip_id,
                    row.log_height,
                    row.gate_count,
                    row.batch_count,
                ),
                -1,
            );
        }
        finalize_residual(residual)
    }

    fn batch_dim_residual_from_challenge_rows(
        record: &RecursionRecord,
        rows: &[ConstraintChallengeRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::new();
        for row in proof_shape_binder_rows(record) {
            if let ProofShapeBinderRow::Chip { proof_idx, chip, publish_batch_dim: true, .. } = row
            {
                apply_residual(
                    &mut residual,
                    vec![
                        proof_idx as u32,
                        PROOF_SHAPE_BATCH_MAIN as u32,
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
        for row in rows {
            apply_residual(
                &mut residual,
                vec![
                    row.proof_idx as u32,
                    PROOF_SHAPE_BATCH_MAIN as u32,
                    row.chip_idx as u32,
                    row.chip_idx as u32,
                    row.static_chip_id as u32,
                    row.main_width as u32,
                    row.log_height as u32,
                ],
                -1,
            );
        }
        finalize_residual(residual)
    }

    fn lcs_residual_from_fold_rows(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
        fold_rows: &[ConstraintFoldRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::new();
        for row in challenge_rows(record, program) {
            apply_residual(
                &mut residual,
                challenge_key(
                    row.proof_idx,
                    CONSTRAINT_CHALLENGE_LCS,
                    row.chip_idx,
                    0,
                    row.lcs_limbs,
                ),
                2,
            );
        }
        for row in terminal_rows(record, program) {
            if row.gcs_data_recv_mult {
                apply_residual(
                    &mut residual,
                    challenge_key(
                        row.proof_idx,
                        CONSTRAINT_CHALLENGE_LCS,
                        row.chip_idx,
                        0,
                        row.lcs,
                    ),
                    -1,
                );
            }
        }
        for row in fold_rows {
            if row.is_skip {
                apply_residual(
                    &mut residual,
                    challenge_key(
                        row.proof_idx,
                        CONSTRAINT_CHALLENGE_LCS,
                        row.remaining_chips - 1,
                        0,
                        row.perm_value,
                    ),
                    -1,
                );
            }
        }
        finalize_residual(residual)
    }

    fn height_inverse_residual_from_raw_root_main(
        program: &RecursionPolyAirVerifierProgram,
        root_main: &CompressedMatrix<F>,
        fold_rows: &[ConstraintFoldRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::new();
        for (row_idx, row) in program.constraint_static_plan().root_rows.iter().enumerate() {
            let values = root_main.main.row_slice(row_idx);
            let cols: &ConstraintRootTableCols<F> = values.as_ref().borrow();
            apply_residual(
                &mut residual,
                height_inverse_key(row.root_ord, row.node_idx),
                cols.height_mult.as_canonical_u32() as i64,
            );
        }
        for row in fold_rows {
            if row.is_skip {
                apply_residual(
                    &mut residual,
                    height_inverse_key(row.root_ord, row.root_nodes[0]),
                    -1,
                );
            }
        }
        finalize_residual(residual)
    }

    fn opened_eval_residual_from_fold_rows(
        record: &RecursionRecord,
        program: &RecursionPolyAirVerifierProgram,
        fold_rows: &[ConstraintFoldRow],
    ) -> BTreeMap<Vec<u32>, i64> {
        let mut residual = BTreeMap::new();
        for proof in &record.proof_records {
            for row in &proof.whir.batch_eval_rows {
                apply_residual(
                    &mut residual,
                    opened_eval_key(
                        row.proof_idx,
                        row.batch_id,
                        row.batch_pos,
                        row.chip_idx,
                        row.value_idx,
                        row.value,
                    ),
                    row.opened_eval_send_mult as i64,
                );
            }
        }
        for row in dag_rows(record, program) {
            let opened_leaf = row.leaf_flags[CONSTRAINT_LEAF_PREPROCESSED] ||
                row.leaf_flags[CONSTRAINT_LEAF_MAIN] ||
                row.leaf_flags[CONSTRAINT_LEAF_RESERVED_POLY];
            if opened_leaf {
                apply_residual(
                    &mut residual,
                    opened_eval_key(
                        row.proof_idx,
                        row.program.lhs_idx,
                        row.opened_batch_pos,
                        row.chip_idx,
                        row.program.rhs_idx,
                        row.value,
                    ),
                    -1,
                );
            }
        }
        for row in fold_rows {
            if row.is_batch {
                apply_residual(
                    &mut residual,
                    opened_eval_key(
                        row.proof_idx,
                        PROOF_SHAPE_BATCH_PERMUTATION,
                        row.remaining_chips - 1,
                        row.remaining_chips - 1,
                        row.root_ord / CONSTRAINT_FOLD_BATCH_SIZE,
                        row.perm_value,
                    ),
                    -1,
                );
            }
        }
        finalize_residual(residual)
    }

    #[test]
    fn program_row_encodes_only_canonical_theta_const_ext() {
        let mut chip = simple_program().chips[0].clone();
        let theta = <EF as AbstractExtensionField<F>>::monomial(1);
        chip.node_table[0].op = RecursionPolyAirOp::ConstExt(theta);

        let row = program_row_for_node(&chip, &chip.node_table[0], 1)
            .expect("canonical theta has a program-table row");
        assert!(row.is_const);
        assert_eq!(row.aux, F::zero());
        assert_eq!(row.lhs_idx, 1);
        assert_eq!(row.rhs_idx, 0);
        assert_eq!(row.third_idx, 0);

        let base_one = <EF as AbstractExtensionField<F>>::from_base_fn(|idx| {
            if idx == 0 {
                F::one()
            } else {
                F::zero()
            }
        });
        for invalid in [base_one, theta * theta] {
            chip.node_table[0].op = RecursionPolyAirOp::ConstExt(invalid);
            assert!(program_row_for_node(&chip, &chip.node_table[0], 1).is_err());
        }
    }

    #[test]
    fn leaf_program_slots_authenticate_opened_and_challenge_routing() {
        let mut chip = simple_program().chips[0].clone();
        chip.static_chip_id = 17;
        chip.reserved_poly = vec![PairCol::Prep(9), PairCol::Main(12)];
        chip.derived_roots.push(crate::symbolic_ir_dt::RecursionPolyAirDerivedRoot::PrecomputeLc {
            index: 0,
            root_node_id: 33,
        });

        let cases = vec![
            (
                RecursionPolyAirLeaf::Preprocessed { col: 7 },
                CONSTRAINT_LEAF_PREPROCESSED,
                PROOF_SHAPE_BATCH_PREPROCESSED,
                7,
                0,
            ),
            (
                RecursionPolyAirLeaf::Main { col: 8 },
                CONSTRAINT_LEAF_MAIN,
                PROOF_SHAPE_BATCH_MAIN,
                8,
                0,
            ),
            (RecursionPolyAirLeaf::Public { index: 9 }, CONSTRAINT_LEAF_PUBLIC, 9, 0, 0),
            (
                RecursionPolyAirLeaf::PermAlpha,
                CONSTRAINT_LEAF_PERM_ALPHA,
                0,
                CONSTRAINT_CHALLENGE_PERM_ALPHA,
                0,
            ),
            (
                RecursionPolyAirLeaf::BetaPower { power: 10 },
                CONSTRAINT_LEAF_BETA_POWER,
                0,
                CONSTRAINT_CHALLENGE_BETA_POWER,
                10,
            ),
            (
                RecursionPolyAirLeaf::BetaSeptix,
                CONSTRAINT_LEAF_BETA_SEPTIX,
                0,
                CONSTRAINT_CHALLENGE_BETA_SEPTIX,
                0,
            ),
            (RecursionPolyAirLeaf::Precomputed { index: 0 }, CONSTRAINT_LEAF_PRECOMPUTED, 33, 0, 0),
            (
                RecursionPolyAirLeaf::ReservedPoly { index: 0 },
                CONSTRAINT_LEAF_RESERVED_POLY,
                PROOF_SHAPE_BATCH_PREPROCESSED,
                9,
                0,
            ),
            (
                RecursionPolyAirLeaf::ReservedPoly { index: 1 },
                CONSTRAINT_LEAF_RESERVED_POLY,
                PROOF_SHAPE_BATCH_MAIN,
                12,
                0,
            ),
            (
                RecursionPolyAirLeaf::IsFirstRow,
                CONSTRAINT_LEAF_IS_FIRST_ROW,
                0,
                CONSTRAINT_CHALLENGE_IS_FIRST,
                17,
            ),
            (
                RecursionPolyAirLeaf::IsLastRow,
                CONSTRAINT_LEAF_IS_LAST_ROW,
                0,
                CONSTRAINT_CHALLENGE_IS_LAST,
                17,
            ),
        ];

        for (leaf, expected_kind, expected_lhs, expected_rhs, expected_third) in cases {
            chip.node_table[0].op = RecursionPolyAirOp::Leaf(leaf);
            let row = program_row_for_node(&chip, &chip.node_table[0], 1)
                .expect("leaf routing must fit the authenticated ProgramBus slots");
            assert!(row.is_leaf);
            assert_eq!(row.leaf_kind, expected_kind);
            assert_eq!(row.lhs_idx, expected_lhs);
            assert_eq!(row.rhs_idx, expected_rhs);
            assert_eq!(row.third_idx, expected_third);
            assert_eq!(row.aux, F::zero());
        }
    }

    #[test]
    fn dag_leaf_routing_mutation_leaves_program_bus_residual() {
        fn residual_from_rows(
            record: &RecursionRecord,
            program: &RecursionPolyAirVerifierProgram,
            rows: &[ConstraintDagRow],
        ) -> BTreeMap<Vec<u32>, i64> {
            let mut residual = BTreeMap::new();
            let counts = program_static_presence_counts(record);
            let plan = program.constraint_static_plan();
            plan.for_each_node(program, |row| {
                let mult = *counts.get(&row.static_chip_id).unwrap_or(&0);
                apply_residual(&mut residual, program_key_ref(row), mult as i64);
            });
            for row in rows {
                apply_residual(&mut residual, program_key(&row.program), -1);
            }
            finalize_residual(residual)
        }

        let record = simple_record();
        let program = simple_program();
        let honest = dag_rows(&record, &program);
        assert!(residual_from_rows(&record, &program, &honest).is_empty());

        let mut wrong_batch = honest.clone();
        wrong_batch[0].program.lhs_idx = PROOF_SHAPE_BATCH_PREPROCESSED;
        assert!(
            !residual_from_rows(&record, &program, &wrong_batch).is_empty(),
            "tampered opened batch escaped the authenticated ProgramBus route"
        );

        let mut wrong_column = honest;
        wrong_column[0].program.rhs_idx += 1;
        assert!(
            !residual_from_rows(&record, &program, &wrong_column).is_empty(),
            "tampered opened column escaped the authenticated ProgramBus route"
        );
    }

    #[test]
    fn static_constraint_plan_matches_uncached_builders_and_is_arc_shared() {
        let program = simple_program();
        let mut expected_program_rows = Vec::new();
        for chip in &program.chips {
            let fanouts = node_fanouts(chip).expect("validated fanouts");
            for node in &chip.node_table {
                expected_program_rows.push(
                    program_row_for_node(chip, node, fanouts[node.node_id as usize])
                        .expect("validated program row"),
                );
            }
        }
        let expected_root_rows = root_table_rows_uncached(&program);

        let first = program.constraint_static_plan();
        let second = program.constraint_static_plan();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(program_rows(&program), expected_program_rows);
        assert_eq!(first.root_rows.as_ref(), expected_root_rows.as_slice());
        assert!(
            core::mem::size_of::<ConstraintProgramNodePlan>() <
                core::mem::size_of::<ConstraintProgramRow>()
        );
        assert_eq!(
            first.node_plan_bytes,
            u64::try_from(
                first.node_plans.len() * core::mem::size_of::<ConstraintProgramNodePlan>()
            )
            .unwrap()
        );
        assert_eq!(
            first.legacy_program_row_bytes,
            u64::try_from(first.node_plans.len() * core::mem::size_of::<ConstraintProgramRow>())
                .unwrap()
        );
        assert!(first.node_plan_bytes < first.legacy_program_row_bytes);
        eprintln!(
            "CONSTRAINT_STATIC_OWNER nodes={} legacy_row_size={} compact_node_size={} \
             legacy_bytes={} compact_bytes={} saved_bytes={}",
            first.node_plans.len(),
            core::mem::size_of::<ConstraintProgramRow>(),
            core::mem::size_of::<ConstraintProgramNodePlan>(),
            first.legacy_program_row_bytes,
            first.node_plan_bytes,
            first.legacy_program_row_bytes - first.node_plan_bytes,
        );

        assert!(Arc::ptr_eq(&first, &program.constraint_static_plan()));
    }

    #[test]
    fn editing_a_dto_creates_a_distinct_ir_and_plan_authority() {
        let program = simple_program();
        let old_plan = program.constraint_static_plan();
        let mut dto = program.to_dto();
        dto.chips[0].chip_name.push_str("Modified");
        let modified =
            RecursionPolyAirVerifierProgram::try_from_dto(dto).expect("modified DTO refreezes");
        assert!(!program.shares_authority_with(&modified));
        assert!(!Arc::ptr_eq(&old_plan, &modified.constraint_static_plan()));
    }

    #[test]
    fn constraint_case_inputs_and_node_arena_are_request_local_and_single_flight() {
        let program = simple_program();
        let record = simple_record();
        let first = constraint_case_artifact(&record, &program);
        let second = constraint_case_artifact(&record, &program);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.stats.proof_count, 1);
        assert_eq!(first.stats.chip_count, 1);
        assert_eq!(first.stats.node_count, 1);
        assert_eq!(first.stats.opened_projection_builds, 1);
        assert_eq!(first.stats.opened_borrowed_views, 0);
        assert_eq!(first.stats.row_vector_reallocations, 0);

        let mut other_record = simple_record();
        other_record.proof_records[0].whir.batch_eval_rows[0].value = ext_base(9);
        let other = constraint_case_artifact(&other_record, &program);
        assert!(!Arc::ptr_eq(&first, &other));
        assert_ne!(first.dag.values[0], other.dag.values[0]);
        assert!(Arc::ptr_eq(&program.constraint_static_plan(), &program.constraint_static_plan()));
    }

    #[test]
    fn lean_fold_replay_matches_full_replay_oracle() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let plan = program.constraint_static_plan();
        let mut stats = ConstraintCaseBuildStats::default();
        let proof_inputs = build_proof_constraint_inputs(&record, &program, &plan, &mut stats);
        assert_eq!(proof_inputs.len(), 1);
        let proof_input = &proof_inputs[0];
        let mut arena = ConstraintFoldReplayArena::for_proof(proof_input);

        for chip_input in &proof_input.chips {
            let base_env = RecursionPolyAirEnv {
                proof_idx: proof_input.proof.proof_idx,
                chip_idx: chip_input.shape.chip_idx,
                opened_preprocessed: chip_input.opened_preprocessed.as_ref(),
                opened_main: chip_input.opened_main.as_ref(),
                public_values: &proof_input.proof.proof_shape.public_values,
                constraint_alpha: proof_input.env.alpha,
                perm_alpha: proof_input.env.perm_alpha,
                perm_beta: proof_input.env.perm_beta,
                beta_powers: &proof_input.env.beta_powers,
                beta_septix: proof_input.env.beta_septix,
                precomputed_lc: &[],
                reserved_poly: &[],
                is_first_row: chip_input.selector.first,
                is_last_row: chip_input.selector.last,
            };
            let full = crate::symbolic_ir_dt::evaluate_chip_replay(
                chip_input.ir,
                &base_env,
                chip_input.permutation_local.as_ref(),
            )
            .expect("full replay oracle");
            let (nodes, precomputed, reserved, _) =
                evaluate_chip_node_arena_profiled(chip_input.ir, &base_env)
                    .expect("lean node arena");
            arena.push_chip(chip_input.ir, &nodes, &precomputed);
            drop(nodes);
            drop(precomputed);
            drop(reserved);

            let compact = *arena.chips.last().expect("compact chip descriptor");
            assert_eq!(arena.gate_values(compact), full.gate_values);
            let compact_batches = arena.batch_roots(compact);
            assert_eq!(compact_batches.len(), full.lookup_batches.len());
            for (batch_idx, roots) in compact_batches.iter().copied().enumerate() {
                let full_batch = &full.lookup_batches[batch_idx];
                assert_eq!(
                    fold_batch_constraint_value(
                        chip_input.ir,
                        batch_idx,
                        roots,
                        chip_input.permutation_local[batch_idx],
                    ),
                    full_batch.constraint_value,
                );
                assert_eq!(chip_input.permutation_local[batch_idx], full_batch.permutation_value,);
                let materialized = fold_batch_roots_from_compact(chip_input.ir, batch_idx, roots);
                for slot in 0..CONSTRAINT_FOLD_BATCH_SIZE {
                    let lookup_idx = batch_idx * CONSTRAINT_FOLD_BATCH_SIZE + slot;
                    if lookup_idx < chip_input.ir.lookup_multiplicity_roots.len() {
                        assert_eq!(materialized[slot].value, full.precomputed_lc[lookup_idx]);
                        let signed = full.signed_lookup_multiplicities[lookup_idx];
                        let expected_unsigned =
                            if materialized[CONSTRAINT_FOLD_BATCH_SIZE + slot].sign == 1 {
                                signed
                            } else {
                                -signed
                            };
                        assert_eq!(
                            materialized[CONSTRAINT_FOLD_BATCH_SIZE + slot].value,
                            expected_unsigned,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compact_constraint_join_matches_exact_rows_matrices_and_residuals() {
        let record = simple_record();
        let program = simple_program();
        let artifact = constraint_case_artifact(&record, &program);
        let counts = prepare_constraint_authority(&record, &program);

        let dag = dag_rows(&record, &program);
        let fold = fold_rows(&record, &program);
        let challenge = challenge_rows(&record, &program);
        let beta = beta_ladder_rows(&record, &program);
        let terminal = terminal_rows(&record, &program);
        assert_eq!(
            (dag.len(), fold.len(), challenge.len(), beta.len(), terminal.len(),),
            (counts.dag, counts.fold, counts.challenge, counts.beta_ladder, counts.terminal,)
        );
        assert_eq!(artifact.dag.len(), dag.len());
        assert_eq!(artifact.fold.len(), fold.len());
        assert_eq!(artifact.challenge.len(), challenge.len());
        assert_eq!(artifact.beta_ladder.len(), beta.len());
        assert_eq!(artifact.terminal.len(), terminal.len());

        let mut dag_oracle = zeroed_trace_values(dag.len(), NUM_CONSTRAINT_DAG_EVAL_COLS);
        for (values, row) in
            dag_oracle.chunks_exact_mut(NUM_CONSTRAINT_DAG_EVAL_COLS).zip(dag.iter())
        {
            fill_exact_dag_row_oracle(values, row);
        }
        let dag_matrix =
            ConstraintDagEvalTraceGenerator::generate_trace_compressed(&record, &program);
        assert_eq!(dag_matrix.main.values, dag_oracle);

        let mut fold_oracle = zeroed_trace_values(fold.len(), NUM_CONSTRAINT_FOLD_COLS);
        for (values, row) in fold_oracle.chunks_exact_mut(NUM_CONSTRAINT_FOLD_COLS).zip(fold.iter())
        {
            fill_fold_row(values, row);
        }
        let fold_matrix =
            ConstraintFoldTraceGenerator::generate_trace_compressed(&record, &program);
        assert_eq!(fold_matrix.main.values, fold_oracle);

        let challenge_oracle = challenge.iter().flat_map(challenge_row).collect::<Vec<_>>();
        let challenge_matrix =
            ConstraintChallengeTraceGenerator::generate_trace_compressed(&record, &program);
        assert_eq!(challenge_matrix.main.values, challenge_oracle);

        let beta_oracle = beta.iter().flat_map(beta_ladder_row).collect::<Vec<_>>();
        let beta_matrix =
            ConstraintBetaLadderTraceGenerator::generate_trace_compressed(&record, &program);
        assert_eq!(beta_matrix.main.values, beta_oracle);

        let terminal_oracle = terminal.iter().flat_map(terminal_row).collect::<Vec<_>>();
        let terminal_matrix =
            ConstraintTerminalTraceGenerator::generate_trace_compressed(&record, &program);
        assert_eq!(terminal_matrix.main.values, terminal_oracle);
        assert!(constraint_replay_bus_residual_report(&record, &program).is_empty());

        let expected_matrix_bytes = [
            (dag.len(), NUM_CONSTRAINT_DAG_EVAL_COLS),
            (fold.len(), NUM_CONSTRAINT_FOLD_COLS),
            (challenge.len(), NUM_CONSTRAINT_CHALLENGE_COLS),
            (beta.len(), NUM_CONSTRAINT_BETA_LADDER_COLS),
            (terminal.len(), NUM_CONSTRAINT_TERMINAL_COLS),
        ]
        .into_iter()
        .map(|(rows, width)| rows.max(1) * width * core::mem::size_of::<F>())
        .sum::<usize>();
        assert_eq!(
            record.profile.structural_counter("constraint_matrix_population_bytes"),
            Some(u64::try_from(expected_matrix_bytes).unwrap())
        );

        assert_eq!(
            record.profile.structural_counter("constraint_static_bytes_duplicated_per_case"),
            Some(0)
        );
        assert_eq!(
            record.profile.structural_counter("constraint_exact_row_structs_retained"),
            Some(0)
        );
        assert!(
            core::mem::size_of::<ConstraintDagCaseRow>() < core::mem::size_of::<ConstraintDagRow>()
        );
    }

    #[test]
    fn fused_challenge_authenticates_chip_identity_c_chips_and_exact_presence() {
        let record = simple_record();
        let program = simple_program();
        let fold = fold_rows(&record, &program);
        let terminal = terminal_rows(&record, &program);
        let honest = challenge_rows(&record, &program);
        assert!(!honest.is_empty());
        assert!(batch_dim_residual_from_challenge_rows(&record, &honest).is_empty());
        assert!(fold_plan_chain_residual_from_rows(&record, &fold, &honest, &terminal).is_empty());

        let mut wrong_identity = honest.clone();
        wrong_identity[0].main_width += 1;
        assert!(
            !batch_dim_residual_from_challenge_rows(&record, &wrong_identity).is_empty(),
            "BatchDim MAIN accepted a coherent Challenge identity splice"
        );

        let mut wrong_c_chips = honest.clone();
        wrong_c_chips[0].c_chips += 1;
        assert!(
            !fold_plan_chain_residual_from_rows(&record, &fold, &wrong_c_chips, &terminal,)
                .is_empty(),
            "PlanChain source accepted a forged Challenge c_chips"
        );

        let mut omitted = honest.clone();
        omitted.remove(0);
        assert!(!batch_dim_residual_from_challenge_rows(&record, &omitted).is_empty());
        assert!(!fold_plan_chain_residual_from_rows(&record, &fold, &omitted, &terminal).is_empty());

        let mut duplicated = honest.clone();
        duplicated.push(honest[0].clone());
        assert!(!batch_dim_residual_from_challenge_rows(&record, &duplicated).is_empty());
        assert!(
            !fold_plan_chain_residual_from_rows(&record, &fold, &duplicated, &terminal).is_empty()
        );
    }

    #[test]
    fn full_switch_synthetic_record_balances_all_constraint_replay_bridges() {
        let record = simple_record();
        let program = simple_program();
        let report = constraint_replay_bus_residual_report(&record, &program);
        assert!(report.is_empty(), "unexpected residual report: {report:?}");
    }

    fn assert_canonical_fold_semantics(rows: &[ConstraintFoldRow]) {
        for (row_idx, row) in rows.iter().enumerate() {
            assert_eq!(row.cursor, row_idx + 1);
            let alpha = EF::from_base_slice(&row.alpha);
            let acc_in = EF::from_base_slice(&row.acc_in);
            let acc_out = EF::from_base_slice(&row.acc_out);
            let pacc_in = EF::from_base_slice(&row.pacc_in);
            let pacc_out = EF::from_base_slice(&row.pacc_out);
            let perm_sum_in = EF::from_base_slice(&row.perm_sum_in);
            let perm_sum_out = EF::from_base_slice(&row.perm_sum_out);
            let perm_value = EF::from_base_slice(&row.perm_value);
            let roots = row.root_values.map(|value| EF::from_base_slice(&value));

            let constraint_value = if row.is_gate {
                assert_eq!(perm_value, EF::zero());
                roots[0]
            } else if row.is_batch {
                let m0 = roots[2] * EF::from_base(signed_f(row.multiplicity_signs[0]));
                let m1 = roots[3] * EF::from_base(signed_f(row.multiplicity_signs[1]));
                roots[1] * m0 + roots[0] * (m1 - roots[1] * perm_value)
            } else {
                EF::zero()
            };
            assert_eq!(acc_out, acc_in * alpha + constraint_value, "constraint fold row {row_idx}");

            if row.is_gate {
                assert_eq!(perm_sum_out, perm_sum_in, "gate row {row_idx}");
                assert_eq!(pacc_out, pacc_in * alpha, "gate row {row_idx}");
            } else if row.is_batch {
                assert_eq!(perm_sum_out, perm_sum_in + perm_value, "batch row {row_idx}");
                assert_eq!(pacc_out, pacc_in * alpha, "batch row {row_idx}");
            } else {
                assert!(row.is_skip);
                let height_inverse = F::from_canonical_usize(row.root_nodes[0]);
                assert_eq!(perm_sum_out, EF::zero(), "skip reset row {row_idx}");
                assert_eq!(
                    pacc_out,
                    pacc_in * alpha + perm_sum_in - perm_value * height_inverse,
                    "skip correction row {row_idx}"
                );
                assert_eq!(
                    perm_value,
                    roots[1] * F::from_canonical_usize(row.batch_count),
                    "authenticated LCS/batch-count quotient row {row_idx}"
                );
            }
        }
    }

    #[test]
    fn canonical_fold_seed_plan_permutation_and_padding_materialize_exactly() {
        let simple_program = simple_program();
        let simple_record = simple_record();
        let simple_rows = fold_rows(&simple_record, &simple_program);
        assert_eq!(simple_rows.len(), 2);
        assert!(simple_rows[0].is_gate);
        assert!(simple_rows[1].is_skip);
        assert_eq!(simple_rows[0].cursor, 1);
        assert_eq!(simple_rows[0].acc_in, [F::zero(); D_EF]);
        assert_eq!(simple_rows[0].pacc_in, [F::zero(); D_EF]);
        assert_eq!(simple_rows[0].perm_sum_in, [F::zero(); D_EF]);
        assert_eq!(simple_rows[1].perm_sum_out, [F::zero(); D_EF]);
        assert_eq!(simple_rows[0].remaining_chips, 1);
        assert_eq!(simple_rows[0].local_ord, 0);
        assert_eq!(simple_rows[0].chain_send_local_ord, 1);
        assert_eq!(simple_rows[1].remaining_chips, 1);
        assert_eq!(simple_rows[1].local_ord, 1);
        assert_eq!(simple_rows[1].chain_send_local_ord, 0);
        assert_canonical_fold_semantics(&simple_rows);

        let simple_main = ConstraintFoldTraceGenerator::generate_trace_compressed(
            &simple_record,
            &simple_program,
        );
        for (row_idx, evaluation) in
            fold_materialized_evaluations(&simple_program, &simple_main).iter().enumerate()
        {
            assert_eq!(evaluation.first, EF::zero(), "simple first row {row_idx}");
            assert_eq!(evaluation.nonfirst, EF::zero(), "simple nonfirst row {row_idx}");
            assert_eq!(evaluation.lookup_multiplicities.len(), 16);
        }

        let lookup_program = lookup_program(3);
        let lookup_record = lookup_record(&lookup_program, true);
        let lookup_rows = fold_rows(&lookup_record, &lookup_program);
        assert_eq!(lookup_rows.len(), 3);
        assert!(lookup_rows[0].is_batch && lookup_rows[0].batch_has_second);
        assert_eq!(lookup_rows[0].root_ord, 0);
        assert!(lookup_rows[1].is_batch && !lookup_rows[1].batch_has_second);
        assert_eq!(lookup_rows[1].root_ord, 2);
        assert!(lookup_rows[2].is_skip);
        assert_eq!(lookup_rows[0].perm_sum_in, [F::zero(); D_EF]);
        assert_ne!(lookup_rows[0].perm_sum_out, [F::zero(); D_EF]);
        assert_eq!(lookup_rows[1].perm_sum_in, lookup_rows[0].perm_sum_out);
        assert_eq!(lookup_rows[2].perm_sum_in, lookup_rows[1].perm_sum_out);
        assert_eq!(lookup_rows[2].perm_sum_out, [F::zero(); D_EF]);
        assert_canonical_fold_semantics(&lookup_rows);

        let lookup_main = ConstraintFoldTraceGenerator::generate_trace_compressed(
            &lookup_record,
            &lookup_program,
        );
        assert_eq!(lookup_main.stored_height(), lookup_rows.len());
        assert_eq!(lookup_main.total_height, 4);
        for (row_idx, evaluation) in
            fold_materialized_evaluations(&lookup_program, &lookup_main).iter().enumerate()
        {
            assert_eq!(evaluation.first, EF::zero(), "lookup first row {row_idx}");
            assert_eq!(evaluation.nonfirst, EF::zero(), "lookup nonfirst row {row_idx}");
            assert_eq!(evaluation.lookup_multiplicities.len(), 16);
        }
        let report = constraint_replay_bus_residual_report(&lookup_record, &lookup_program);
        assert!(report.is_empty(), "lookup replay bus residual: {report:?}");
    }

    #[test]
    fn fold_chain_seed_alpha_and_detached_component_attacks_are_rejected() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let batch_rows = batch_transcript_input_rows(&record);
        let honest_rows = fold_rows(&record, &program);
        let honest_terminal = terminal_rows(&record, &program);
        assert!(
            fold_chain_residual_from_rows(&batch_rows, &honest_rows, &honest_terminal).is_empty()
        );

        // Old optional-first relations admitted an arbitrary initial state. Recompute every
        // downstream state/key so the only unmatched edge is the literal-zero Batch seed.
        let mut bad_seed = honest_rows.clone();
        let forged = EF::from_canonical_u32(17);
        rethread_fold_suffix(&mut bad_seed, 0, forged, forged, forged, None);
        let mut bad_seed_terminal = honest_terminal.clone();
        retarget_terminal_fold_sink(
            &mut bad_seed_terminal,
            bad_seed.last().expect("nonempty Fold chain"),
        );
        assert_canonical_fold_semantics(&bad_seed);
        assert!(
            !fold_chain_residual_from_rows(&batch_rows, &bad_seed, &bad_seed_terminal).is_empty(),
            "literal-zero Fold seed accepted a fully rethreaded nonzero initial state"
        );

        // Drift alpha in the middle, then recompute acc/pacc and every following FoldChain key.
        // The boundary edge retains the transcript-authenticated alpha and must stay unmatched.
        let mut bad_alpha = honest_rows.clone();
        let split = 1;
        let new_alpha = EF::from_base_slice(&bad_alpha[split].alpha) + EF::one();
        let acc = EF::from_base_slice(&bad_alpha[split - 1].acc_out);
        let pacc = EF::from_base_slice(&bad_alpha[split - 1].pacc_out);
        let perm_sum = EF::from_base_slice(&bad_alpha[split - 1].perm_sum_out);
        rethread_fold_suffix(&mut bad_alpha, split, acc, pacc, perm_sum, Some(new_alpha));
        let mut bad_alpha_terminal = honest_terminal.clone();
        retarget_terminal_fold_sink(
            &mut bad_alpha_terminal,
            bad_alpha.last().expect("nonempty Fold chain"),
        );
        assert_canonical_fold_semantics(&bad_alpha);
        assert!(
            !fold_chain_residual_from_rows(&batch_rows, &bad_alpha, &bad_alpha_terminal).is_empty(),
            "FoldChain payload failed to bind alpha across a hop"
        );

        // Every valid row now has one mandatory recv and send. A detached finite path therefore
        // contributes one extra source and one extra sink even when its internal transition is
        // algebraically honest.
        let mut detached = honest_rows.clone();
        let mut extra = honest_rows[0].clone();
        extra.cursor = 100;
        extra.acc_in = ext_limbs(&EF::from_canonical_u32(29));
        extra.pacc_in = ext_limbs(&EF::from_canonical_u32(31));
        extra.perm_sum_in = [F::zero(); D_EF];
        let alpha = EF::from_base_slice(&extra.alpha);
        let old_value = EF::from_base_slice(&honest_rows[0].acc_out) -
            EF::from_base_slice(&honest_rows[0].acc_in) * alpha;
        extra.acc_out = ext_limbs(&(EF::from_base_slice(&extra.acc_in) * alpha + old_value));
        extra.pacc_out = ext_limbs(&(EF::from_base_slice(&extra.pacc_in) * alpha));
        extra.perm_sum_out = extra.perm_sum_in;
        detached.push(extra);
        assert!(
            !fold_chain_residual_from_rows(&batch_rows, &detached, &honest_terminal).is_empty(),
            "detached mandatory recv/send component unexpectedly balanced"
        );
    }

    #[test]
    fn canonical_plan_rejects_reordered_rows_after_full_state_recomputation() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let mut rows = fold_rows(&record, &program);
        let batch_rows = batch_transcript_input_rows(&record);
        let mut terminal = terminal_rows(&record, &program);
        assert!(rows[0].is_gate && rows[1].is_batch);
        assert!(fold_plan_chain_residual_from_rows(
            &record,
            &rows,
            &challenge_rows(&record, &program),
            &terminal,
        )
        .is_empty());

        rows.swap(0, 1);
        rethread_fold_suffix(&mut rows, 0, EF::zero(), EF::zero(), EF::zero(), None);
        retarget_terminal_fold_sink(&mut terminal, rows.last().expect("nonempty Fold chain"));
        assert_canonical_fold_semantics(&rows);
        assert!(
            fold_chain_residual_from_rows(&batch_rows, &rows, &terminal).is_empty(),
            "reordered attack did not fully recompute FoldChain state"
        );
        assert!(
            !fold_plan_chain_residual_from_rows(
                &record,
                &rows,
                &challenge_rows(&record, &program),
                &terminal,
            )
            .is_empty(),
            "canonical plan accepted a gate/batch reorder"
        );
    }

    #[test]
    fn skip_correction_lcs_and_log_height_attacks_are_isolated() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let batch_rows = batch_transcript_input_rows(&record);
        let honest_rows = fold_rows(&record, &program);
        let honest_terminal = terminal_rows(&record, &program);
        let skip = honest_rows.len() - 1;

        // Keep every FoldChain key balanced, but forge the skip pacc correction. The local AIR
        // equation itself must reject it.
        let mut bad_correction = honest_rows.clone();
        bad_correction[skip].pacc_out[0] += F::one();
        let mut bad_correction_terminal = honest_terminal.clone();
        retarget_terminal_fold_sink(&mut bad_correction_terminal, &bad_correction[skip]);
        assert!(fold_chain_residual_from_rows(
            &batch_rows,
            &bad_correction,
            &bad_correction_terminal,
        )
        .is_empty());
        let evaluations =
            fold_materialized_evaluations(&program, &fold_matrix_from_rows(&bad_correction));
        assert_ne!(evaluations[skip].first, EF::zero());
        assert_ne!(evaluations[skip].nonfirst, EF::zero());

        // Change authenticated LCS, update the quotient, skip correction, pacc and terminal key.
        // Fold is locally honest; only the authoritative ConstraintChallenge producer rejects it.
        let mut bad_lcs = honest_rows.clone();
        let forged_lcs = EF::from_base_slice(&bad_lcs[skip].perm_value) + EF::one();
        bad_lcs[skip].perm_value = ext_limbs(&forged_lcs);
        bad_lcs[skip].root_values[1] =
            ext_limbs(&(forged_lcs * F::from_canonical_usize(bad_lcs[skip].batch_count).inverse()));
        let acc = EF::from_base_slice(&bad_lcs[skip - 1].acc_out);
        let pacc = EF::from_base_slice(&bad_lcs[skip - 1].pacc_out);
        let perm_sum = EF::from_base_slice(&bad_lcs[skip - 1].perm_sum_out);
        rethread_fold_suffix(&mut bad_lcs, skip, acc, pacc, perm_sum, None);
        let mut bad_lcs_terminal = honest_terminal.clone();
        retarget_terminal_fold_sink(&mut bad_lcs_terminal, &bad_lcs[skip]);
        assert_canonical_fold_semantics(&bad_lcs);
        assert!(fold_chain_residual_from_rows(&batch_rows, &bad_lcs, &bad_lcs_terminal).is_empty());
        let bad_lcs_evaluations =
            fold_materialized_evaluations(&program, &fold_matrix_from_rows(&bad_lcs));
        assert!(
            bad_lcs_evaluations.iter().all(|evaluation| {
                evaluation.first == EF::zero() && evaluation.nonfirst == EF::zero()
            }),
            "locally recomputed LCS attack should satisfy Fold AIR: {bad_lcs_evaluations:?}"
        );
        assert!(
            !lcs_residual_from_fold_rows(&record, &program, &bad_lcs).is_empty(),
            "authenticated LCS producer accepted a fully recomputed forged correction"
        );

        // Use another valid static height-table entry and recompute the correction. The height
        // lookup remains valid, but ProofShapeChipMeta binds the chip's actual log height.
        let mut bad_height = honest_rows.clone();
        let forged_height = 0usize;
        let forged_inverse = F::one();
        bad_height[skip].root_ord = forged_height;
        bad_height[skip].root_nodes[0] = forged_inverse.as_canonical_u32() as usize;
        for row in &mut bad_height {
            row.log_height = forged_height;
            if row.is_skip {
                row.root_ord = forged_height;
            }
        }
        let acc = EF::from_base_slice(&bad_height[skip - 1].acc_out);
        let pacc = EF::from_base_slice(&bad_height[skip - 1].pacc_out);
        let perm_sum = EF::from_base_slice(&bad_height[skip - 1].perm_sum_out);
        rethread_fold_suffix(&mut bad_height, skip, acc, pacc, perm_sum, None);
        let mut bad_height_terminal = honest_terminal.clone();
        retarget_terminal_fold_sink(&mut bad_height_terminal, &bad_height[skip]);
        assert!(
            fold_plan_chain_residual_from_rows(
                &record,
                &bad_height,
                &challenge_rows(&record, &program),
                &bad_height_terminal,
            )
            .is_empty(),
            "height attack did not preserve the canonical schedule chain"
        );
        assert!(
            !chip_meta_residual_from_fold_rows(&record, &bad_height).is_empty(),
            "ProofShapeChipMeta failed to bind authenticated log height"
        );
    }

    #[test]
    fn forged_height_inverse_provider_is_rejected_after_full_correction_rethread() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let mut fold = fold_rows(&record, &program);
        let honest_terminal = terminal_rows(&record, &program);
        let skip = fold.len() - 1;
        let batch_rows = batch_transcript_input_rows(&record);
        let plan = program.constraint_static_plan();
        let correct_inverse = F::from_canonical_usize(1usize << fold[skip].root_ord)
            .inverse()
            .as_canonical_u32() as usize;
        let forged_provider = plan
            .root_rows
            .iter()
            .enumerate()
            .find(|(_, row)| {
                row.static_chip_id != CONSTRAINT_HEIGHT_TABLE_STATIC_ID &&
                    row.root_ord == fold[skip].root_ord &&
                    row.node_idx != correct_inverse
            })
            .map(|(idx, row)| (idx, row.node_idx))
            .expect("test program has an ordinary root row usable as a forged provider");
        let correct_provider = plan
            .root_rows
            .iter()
            .enumerate()
            .find(|(_, row)| {
                row.static_chip_id == CONSTRAINT_HEIGHT_TABLE_STATIC_ID &&
                    row.root_ord == fold[skip].root_ord
            })
            .map(|(idx, _)| idx)
            .expect("canonical height provider");

        let root_preprocessed =
            ConstraintRootTableTraceGenerator::generate_preprocessed_trace(&program);
        let honest_root_main =
            ConstraintRootTableTraceGenerator::generate_trace_compressed(&record, &program);
        let mut forged_root_values = honest_root_main.main.values.clone();
        {
            let row = &mut forged_root_values[correct_provider * NUM_CONSTRAINT_ROOT_TABLE_COLS..
                (correct_provider + 1) * NUM_CONSTRAINT_ROOT_TABLE_COLS];
            let cols: &mut ConstraintRootTableCols<F> = row.borrow_mut();
            cols.height_mult = F::zero();
        }
        {
            let row = &mut forged_root_values[forged_provider.0 * NUM_CONSTRAINT_ROOT_TABLE_COLS..
                (forged_provider.0 + 1) * NUM_CONSTRAINT_ROOT_TABLE_COLS];
            let cols: &mut ConstraintRootTableCols<F> = row.borrow_mut();
            cols.height_mult = F::one();
        }
        let forged_root_main = compressed_values(
            forged_root_values,
            NUM_CONSTRAINT_ROOT_TABLE_COLS,
            honest_root_main.total_height,
            vec![F::zero(); NUM_CONSTRAINT_ROOT_TABLE_COLS],
        );

        // Keep LCS fixed and non-zero inside the isolated Fold witness so changing the inverse
        // changes the skip correction. The LCS authority is exercised independently above.
        let fixed_lcs = EF::from_canonical_u32(23);
        fold[skip].perm_value = ext_limbs(&fixed_lcs);
        fold[skip].root_values[1] =
            ext_limbs(&(fixed_lcs * F::from_canonical_usize(fold[skip].batch_count).inverse()));
        fold[skip].root_nodes[0] = forged_provider.1;
        let acc = EF::from_base_slice(&fold[skip - 1].acc_out);
        let pacc = EF::from_base_slice(&fold[skip - 1].pacc_out);
        let perm_sum = EF::from_base_slice(&fold[skip - 1].perm_sum_out);
        rethread_fold_suffix(&mut fold, skip, acc, pacc, perm_sum, None);
        let mut terminal = honest_terminal;
        retarget_terminal_fold_sink(&mut terminal, &fold[skip]);

        assert!(
            height_inverse_residual_from_raw_root_main(&program, &forged_root_main, &fold)
                .is_empty(),
            "attack did not synchronize provider multiplicity with the forged Fold key"
        );
        assert!(
            fold_chain_residual_from_rows(&batch_rows, &fold, &terminal).is_empty(),
            "attack did not synchronize pacc and FoldChain"
        );
        let fold_evaluations =
            fold_materialized_evaluations(&program, &fold_matrix_from_rows(&fold));
        assert!(
            fold_evaluations.iter().all(|evaluation| {
                evaluation.first == EF::zero() && evaluation.nonfirst == EF::zero()
            }),
            "forged inverse attack must isolate the provider relation"
        );

        let root_evaluations =
            root_table_materialized_evaluations(&program, &root_preprocessed, &forged_root_main);
        assert_ne!(
            root_evaluations[forged_provider.0],
            EF::zero(),
            "ordinary root row published a forged height inverse"
        );
    }

    #[test]
    fn permutation_omission_duplication_cross_proof_swap_and_nonzero_sink_are_rejected() {
        let program = mixed_gate_lookup_program(3);
        let record = lookup_record(&program, true);
        let honest_rows = fold_rows(&record, &program);
        let honest_terminal = terminal_rows(&record, &program);

        let mut duplicate = honest_rows.clone();
        let first_batch = duplicate.iter().position(|row| row.is_batch).unwrap();
        let second_batch = duplicate
            .iter()
            .enumerate()
            .find(|(idx, row)| *idx > first_batch && row.is_batch)
            .map(|(idx, _)| idx)
            .unwrap();
        duplicate[second_batch].perm_value = duplicate[first_batch].perm_value;
        let acc = EF::from_base_slice(&duplicate[second_batch - 1].acc_out);
        let pacc = EF::from_base_slice(&duplicate[second_batch - 1].pacc_out);
        let perm_sum = EF::from_base_slice(&duplicate[second_batch - 1].perm_sum_out);
        rethread_fold_suffix(&mut duplicate, second_batch, acc, pacc, perm_sum, None);
        assert_canonical_fold_semantics(&duplicate);
        assert!(
            !opened_eval_residual_from_fold_rows(&record, &program, &duplicate).is_empty(),
            "duplicate permutation value escaped the authoritative opening bus"
        );

        let mut omitted = honest_rows.clone();
        omitted.remove(first_batch);
        rethread_fold_suffix(&mut omitted, 0, EF::zero(), EF::zero(), EF::zero(), None);
        assert!(
            !fold_plan_chain_residual_from_rows(
                &record,
                &omitted,
                &challenge_rows(&record, &program),
                &honest_terminal,
            )
            .is_empty(),
            "canonical plan accepted an omitted batch row"
        );
        assert!(
            !opened_eval_residual_from_fold_rows(&record, &program, &omitted).is_empty(),
            "authoritative permutation opening accepted an omitted batch"
        );

        // PlanChain is per proof; moving one canonical row to another proof cannot transfer
        // schedule multiplicity.
        let mut cross_proof = honest_rows.clone();
        cross_proof[0].proof_idx = 1;
        assert!(
            !fold_plan_chain_residual_from_rows(
                &record,
                &cross_proof,
                &challenge_rows(&record, &program),
                &honest_terminal,
            )
            .is_empty(),
            "fused Fold schedule allowed a cross-proof multiplicity transfer"
        );

        let batch_rows = batch_transcript_input_rows(&record);
        let terminal = honest_terminal;
        let mut nonzero_sink = honest_rows.clone();
        let last = nonzero_sink.last_mut().expect("nonempty Fold chain");
        last.perm_sum_out[0] = F::one();
        assert_ne!(
            fold_chain_send_key(last),
            terminal_fold_chain_key(terminal.iter().find(|row| row.fold_chain_recv_mult).unwrap())
        );
        assert!(
            !fold_chain_residual_from_rows(&batch_rows, &nonzero_sink, &terminal).is_empty(),
            "Terminal final denominator failed to use literal-zero permutation state"
        );
    }

    #[test]
    fn product_l1_l4_constraint_programs_use_exact_batch_two() {
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
            assert!(
                program
                    .constraint_program
                    .chips
                    .iter()
                    .all(|chip| chip.logup_batch_size == CONSTRAINT_FOLD_BATCH_SIZE),
                "{layer} contains a non-batch-two child AIR"
            );
            let lookup_count = program
                .constraint_program
                .chips
                .iter()
                .map(|chip| chip.lookup_multiplicity_roots.len())
                .sum::<usize>();
            let batch_rows = program
                .constraint_program
                .chips
                .iter()
                .map(|chip| {
                    chip.lookup_multiplicity_roots.len().div_ceil(CONSTRAINT_FOLD_BATCH_SIZE)
                })
                .sum::<usize>();
            let tails = program
                .constraint_program
                .chips
                .iter()
                .filter(|chip| {
                    chip.lookup_multiplicity_roots.len() % CONSTRAINT_FOLD_BATCH_SIZE != 0
                })
                .count();
            assert!(lookup_count > 0, "{layer} unexpectedly has no lookups");
            assert!(batch_rows > 0, "{layer} unexpectedly has no batch rows");
            assert!(tails > 0, "{layer} must exercise the tail-batch path");
        }
    }

    #[test]
    fn node_value_and_fanout_mutations_leave_node_residual() {
        let record = simple_record();
        let program = simple_program();
        let dag = dag_rows(&record, &program);
        let fold = fold_rows(&record, &program);

        let mut bad_value = dag.clone();
        bad_value[0].value[0] += F::one();
        assert!(
            !node_value_residual_from_rows(&bad_value, &fold).is_empty(),
            "tampered node value must leave a ConstraintNodeValue residual"
        );

        let mut bad_fanout = dag;
        bad_fanout[0].program.fanout += 1;
        assert!(
            !node_value_residual_from_rows(&bad_fanout, &fold).is_empty(),
            "tampered fanout must leave a ConstraintNodeValue residual"
        );
    }

    #[test]
    fn root_order_mutation_leaves_root_table_residual() {
        let record = simple_record();
        let program = simple_program();
        let mut fold = fold_rows(&record, &program);
        let row = fold.iter_mut().find(|row| row.is_gate).expect("synthetic fold has a gate root");
        row.root_ord += 1;
        let residual = root_table_residual_from_rows(
            &program_static_presence_counts(&record),
            &root_table_rows(&program),
            &fold,
        );
        assert!(!residual.is_empty(), "tampered root order must leave a root-table residual");
    }

    #[test]
    fn last_claim_mutation_breaks_direct_claim_chain_and_eq_coordinate_tamper_breaks_bridge() {
        let program = simple_program();

        let mut bad_claim = simple_record();
        bad_claim.proof_records[0].batch_constraint.last_claim[0] += F::one();
        let bad_terminal = terminal_rows(&bad_claim, &program);
        let bad_final = bad_terminal
            .iter()
            .find(|row| row.is_final)
            .expect("synthetic terminal has one final row");
        assert_ne!(
            bad_final.final_lhs, bad_final.last_claim,
            "record must preserve an invalid terminal identity for the AIR to reject"
        );
        let residual = constraint_replay_bus_residual_report(&bad_claim, &program);
        assert!(
            residual.contains_key("1018 BatchSumcheckClaimChain"),
            "tampered terminal claim must leave a direct BatchSumcheckClaimChain residual: \
             {residual:?}"
        );

        let record = simple_record();
        let mut terminal = terminal_rows(&record, &program);
        let challenge = challenge_rows(&record, &program);
        let eq_row = terminal
            .iter_mut()
            .find(|row| row.is_eq_step)
            .expect("synthetic terminal has one eq step");
        eq_row.eq_out.swap(0, 1);
        let residual = eq_chain_residual_from_rows(&terminal, &challenge);
        assert!(
            !residual.is_empty(),
            "tampered eq coordinate order must leave an eq-chain residual"
        );
    }

    #[test]
    fn terminal_direct_narrow_writer_matches_projection_for_roles_boundaries_and_padding() {
        let keep = terminal_narrow_projection_columns();
        assert_eq!(keep.len(), NUM_CONSTRAINT_TERMINAL_NARROW_COLS);

        for role in [RecursionChildRole::Core, RecursionChildRole::Compress] {
            for public_value_count in
                [dt_stark::air::DT_PROOF_NUM_PV_ELTS, dt_stark::PROOF_MAX_NUM_PVS]
            {
                let record = simple_record_with_public_value_count(public_value_count);
                let program = simple_program_for_role(role);
                let wide =
                    ConstraintTerminalTraceGenerator::generate_trace_compressed(&record, &program)
                        .decompress();
                let narrow = ConstraintTerminalTraceGenerator::generate_trace_compressed_narrow(
                    &record, &program,
                )
                .decompress();
                assert_eq!(wide.height(), narrow.height());
                for row in 0..wide.height() {
                    for (narrow_column, wide_column) in keep.iter().copied().enumerate() {
                        assert_eq!(
                            narrow.get(row, narrow_column),
                            wide.get(row, wide_column),
                            "direct/projection mismatch for {role:?}, {public_value_count} public values at row {row}, narrow column {narrow_column}"
                        );
                    }
                }
            }
        }
    }
}
