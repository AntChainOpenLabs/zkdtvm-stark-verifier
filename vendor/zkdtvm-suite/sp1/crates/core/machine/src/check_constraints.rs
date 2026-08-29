//! Lightweight row-level constraint checker for AIR chips.
//!
//! This module provides [`RowChecker`], a simplified [`AirBuilder`] that checks whether
//! concrete field values (from trace rows) satisfy a chip's eval constraints.
//!
//! Unlike [`DebugConstraintBuilder`](dt_stark::debug::DebugConstraintBuilder) in the stark crate,
//! this checker:
//! - Does **not** require a permutation trace, cumulative sums, or extension field types.
//! - **Collects** all constraint violations instead of panicking on the first one.
//! - Works on individual rows, making it easy to check padding rows or single event rows.
//! - Captures **constraint index** and **source location** for each failure, making it easy for
//!   AI/humans to locate the exact eval code that produced the violation.
//!
//! All interactions (send/receive) are no-ops — only pure arithmetic constraints
//! (`assert_zero`, `assert_eq`, `assert_bool`, etc.) are checked.

#![allow(clippy::print_stdout)]

use std::cell::RefCell;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
};

use dt_core_executor::ExecutionRecord;
use dt_stark::{
    air::{EmptyMessageBuilder, MachineAir},
    PROOF_MAX_NUM_PVS,
};

/// A constraint violation found during row checking.
#[derive(Debug, Clone)]
pub struct ConstraintFailure {
    /// Sequential index of this constraint within the eval (0-based).
    /// This counts every `assert_*` call, including those inside `when()` guards
    /// and sub-operations (e.g., `IsZeroOperation::eval`, `AddOperation::eval`).
    pub constraint_index: usize,
    /// A human-readable description of what was expected.
    pub kind: ConstraintKind,
    /// The left-hand side value (as a string representation of the field element).
    pub lhs: String,
    /// The right-hand side value (for assert_eq; empty for assert_zero/assert_one).
    pub rhs: String,
    /// Filtered source locations from the backtrace — only frames within
    /// `crates/core/machine/src/` are included. Typically shows the eval call chain.
    pub source_locations: Vec<String>,
}

/// The kind of constraint that was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    AssertZero,
    AssertOne,
    AssertEq,
    AssertBool,
}

impl std::fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssertZero => write!(f, "assert_zero"),
            Self::AssertOne => write!(f, "assert_one"),
            Self::AssertEq => write!(f, "assert_eq"),
            Self::AssertBool => write!(f, "assert_bool"),
        }
    }
}

impl std::fmt::Display for ConstraintFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}: ", self.constraint_index)?;
        match self.kind {
            ConstraintKind::AssertZero => {
                write!(f, "assert_zero failed: got {}", self.lhs)?;
            }
            ConstraintKind::AssertOne => {
                write!(f, "assert_one failed: got {}", self.lhs)?;
            }
            ConstraintKind::AssertEq => {
                write!(f, "assert_eq failed: {} != {}", self.lhs, self.rhs)?;
            }
            ConstraintKind::AssertBool => {
                write!(f, "assert_bool failed: {} is not 0 or 1", self.lhs)?;
            }
        }
        if !self.source_locations.is_empty() {
            for loc in &self.source_locations {
                write!(f, "\n    at {loc}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RowChecker: the lightweight AirBuilder
// ---------------------------------------------------------------------------

/// A lightweight AirBuilder that evaluates constraints on concrete field values
/// and collects all violations.
///
/// Interactions (send_byte, receive_instruction, etc.) are no-ops because this
/// type implements [`EmptyMessageBuilder`].
pub struct RowChecker<'a, F: Field> {
    main: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    preprocessed: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    public_values: &'a [F],
    is_first_row: F,
    is_last_row: F,
    is_transition: F,
    /// Sequential counter: incremented on every assert_* call.
    constraint_counter: RefCell<usize>,
    /// Collected constraint failures (interior mutability so `when`/`when_not` can record).
    failures: RefCell<Vec<ConstraintFailure>>,
}

impl<'a, F: Field> RowChecker<'a, F> {
    /// Get next constraint index and increment counter.
    fn next_index(&self) -> usize {
        let mut counter = self.constraint_counter.borrow_mut();
        let idx = *counter;
        *counter += 1;
        idx
    }
}

// --- AirBuilder impl ---

impl<'a, F: Field> AirBuilder for RowChecker<'a, F> {
    type F = F;
    type Expr = F;
    type Var = F;
    type M = VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>;

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("RowChecker only supports a window size of 2")
        }
    }

    fn main(&self) -> Self::M {
        self.main
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let idx = self.next_index();
        let val = x.into();
        if val != F::zero() {
            self.failures.borrow_mut().push(ConstraintFailure {
                constraint_index: idx,
                kind: ConstraintKind::AssertZero,
                lhs: format!("{val:?}"),
                rhs: String::new(),
                source_locations: capture_source_locations(),
            });
        }
    }

    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        let idx = self.next_index();
        let val = x.into();
        if val != F::one() {
            self.failures.borrow_mut().push(ConstraintFailure {
                constraint_index: idx,
                kind: ConstraintKind::AssertOne,
                lhs: format!("{val:?}"),
                rhs: String::new(),
                source_locations: capture_source_locations(),
            });
        }
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let idx = self.next_index();
        let lhs = x.into();
        let rhs = y.into();
        if lhs != rhs {
            self.failures.borrow_mut().push(ConstraintFailure {
                constraint_index: idx,
                kind: ConstraintKind::AssertEq,
                lhs: format!("{lhs:?}"),
                rhs: format!("{rhs:?}"),
                source_locations: capture_source_locations(),
            });
        }
    }

    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        let idx = self.next_index();
        let val = x.into();
        if val != F::zero() && val != F::one() {
            self.failures.borrow_mut().push(ConstraintFailure {
                constraint_index: idx,
                kind: ConstraintKind::AssertBool,
                lhs: format!("{val:?}"),
                rhs: String::new(),
                source_locations: capture_source_locations(),
            });
        }
    }
}

// --- Trait impls that complete the DTCoreAirBuilder hierarchy ---

// EmptyMessageBuilder makes all send/receive interactions no-ops.
// This single impl gives us (via blanket impls):
//   MessageBuilder -> BaseAirBuilder -> ByteAirBuilder, InstructionAirBuilder,
//   ExtensionAirBuilder, SepticExtensionAirBuilder, WordAirBuilder,
//   MemoryAirBuilder, ProgramAirBuilder
impl<F: Field> EmptyMessageBuilder for RowChecker<'_, F> {}

// PairBuilder provides the preprocessed trace (needed by ProgramChip, ByteChip, etc.)
impl<'a, F: Field> PairBuilder for RowChecker<'a, F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

// AirBuilderWithPublicValues completes the chain:
//   MachineAirBuilder -> DTAirBuilder -> DTCoreAirBuilder
impl<F: Field> AirBuilderWithPublicValues for RowChecker<'_, F> {
    type PublicVar = F;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Check whether a single row satisfies the chip's AIR constraints.
///
/// `row` is the current row (local), `next_row` is the next row.
/// `preprocessed_width` is the width of the preprocessed trace (0 if none).
/// Only pure arithmetic constraints are checked; interactions (send/receive) are skipped.
///
/// Returns an empty vec if all constraints pass.
pub fn check_row<F, A>(
    chip: &A,
    row: &[F],
    next_row: &[F],
    public_values: &[F],
    preprocessed_width: usize,
) -> Vec<ConstraintFailure>
where
    F: Field,
    A: for<'a> Air<RowChecker<'a, F>>,
{
    let prep_width = if preprocessed_width > 0 { preprocessed_width } else { row.len() };
    let prep = vec![F::zero(); prep_width];
    let mut checker = RowChecker {
        main: VerticalPair::new(
            RowMajorMatrixView::new_row(row),
            RowMajorMatrixView::new_row(next_row),
        ),
        preprocessed: VerticalPair::new(
            RowMajorMatrixView::new_row(&prep),
            RowMajorMatrixView::new_row(&prep),
        ),
        public_values,
        is_first_row: F::zero(),
        is_last_row: F::zero(),
        is_transition: F::one(),
        constraint_counter: RefCell::new(0),
        failures: RefCell::new(Vec::new()),
    };
    chip.eval(&mut checker);
    checker.failures.into_inner()
}

/// Check whether a padding row satisfies the chip's AIR constraints.
///
/// `padding_row` is the actual padding row content (which may contain non-zero values
/// for some chips). The same row is used as both the current and next row.
/// Useful for verifying that unconditional constraints (those without `when(is_real)` guard)
/// are safe for padding rows.
///
/// Returns an empty vec if all constraints pass.
pub fn check_padding<F, A>(
    chip: &A,
    padding_row: &[F],
    preprocessed_width: usize,
) -> Vec<ConstraintFailure>
where
    F: Field,
    A: for<'a> Air<RowChecker<'a, F>>,
{
    let pv = vec![F::zero(); PROOF_MAX_NUM_PVS];
    check_row(chip, padding_row, padding_row, &pv, preprocessed_width)
}

/// Check all rows of a trace matrix against the chip's AIR constraints.
///
/// `preprocessed_width` is the width of the preprocessed trace (0 if chip has no
/// preprocessed trace, in which case the main trace width is used as a safe fallback).
///
/// Returns a list of `(row_index, failure)` pairs for every constraint violation found.
pub fn check_trace<F, A>(
    chip: &A,
    trace: &p3_matrix::dense::RowMajorMatrix<F>,
    public_values: &[F],
    preprocessed_width: usize,
) -> Vec<(usize, ConstraintFailure)>
where
    F: Field,
    A: for<'a> Air<RowChecker<'a, F>>,
{
    check_trace_with_preprocessed(chip, trace, public_values, preprocessed_width, None)
}

/// Check all rows of a trace matrix against the chip's AIR constraints,
/// optionally providing a real preprocessed trace.
///
/// If `preprocessed_trace` is `Some`, actual preprocessed rows are used per row.
/// If `None`, all-zero preprocessed rows are used (safe for chips without preprocessed data).
///
/// Returns a list of `(row_index, failure)` pairs for every constraint violation found.
pub fn check_trace_with_preprocessed<F, A>(
    chip: &A,
    trace: &p3_matrix::dense::RowMajorMatrix<F>,
    public_values: &[F],
    preprocessed_width: usize,
    preprocessed_trace: Option<&p3_matrix::dense::RowMajorMatrix<F>>,
) -> Vec<(usize, ConstraintFailure)>
where
    F: Field,
    A: for<'a> Air<RowChecker<'a, F>>,
{
    use p3_matrix::Matrix;

    let height = trace.height();
    if height == 0 {
        return Vec::new();
    }

    let prep_width = if preprocessed_width > 0 { preprocessed_width } else { trace.width() };
    let default_prep = vec![F::zero(); prep_width];

    let mut all_failures = Vec::new();

    for i in 0..height {
        let i_next = (i + 1) % height;

        let local = trace.row_slice(i);
        let next = trace.row_slice(i_next);

        // Use real preprocessed trace if available, otherwise zeros.
        let (prep_local, prep_next);
        let preprocessed = if let Some(pt) = preprocessed_trace {
            prep_local = pt.row_slice(i);
            prep_next = pt.row_slice(i_next);
            VerticalPair::new(
                RowMajorMatrixView::new_row(&prep_local),
                RowMajorMatrixView::new_row(&prep_next),
            )
        } else {
            VerticalPair::new(
                RowMajorMatrixView::new_row(&default_prep),
                RowMajorMatrixView::new_row(&default_prep),
            )
        };

        let mut checker = RowChecker {
            main: VerticalPair::new(
                RowMajorMatrixView::new_row(&local),
                RowMajorMatrixView::new_row(&next),
            ),
            preprocessed,
            public_values,
            is_first_row: if i == 0 { F::one() } else { F::zero() },
            is_last_row: if i == height - 1 { F::one() } else { F::zero() },
            is_transition: if i == height - 1 { F::zero() } else { F::one() },
            constraint_counter: RefCell::new(0),
            failures: RefCell::new(Vec::new()),
        };

        chip.eval(&mut checker);

        for failure in checker.failures.into_inner() {
            all_failures.push((i, failure));
        }
    }

    all_failures
}

// ---------------------------------------------------------------------------
// Higher-level pipeline functions (generate_trace → check)
// ---------------------------------------------------------------------------

/// Generate a trace from an `ExecutionRecord` and check all rows against the chip's constraints.
///
/// This is a convenience function that chains `generate_trace` → `check_trace`.
/// Returns a list of `(row_index, failure)` pairs.
pub fn check_record<F, A>(chip: &A, record: &ExecutionRecord) -> Vec<(usize, ConstraintFailure)>
where
    F: Field,
    A: MachineAir<F, Record = ExecutionRecord> + for<'a> Air<RowChecker<'a, F>>,
{
    let mut output = ExecutionRecord::default();
    let trace = chip.generate_trace(record, &mut output).decompress();
    let pv = record.public_values.to_vec::<F>();
    let prep_width = chip.preprocessed_width();
    check_trace(chip, &trace, &pv, prep_width)
}

/// Check whether the chip's padding row satisfies the AIR constraints.
///
/// Uses the chip's [`MachineAir::padding_row()`] method to obtain the padding row,
/// then checks it against all constraints. This is the recommended way to verify
/// padding row validity.
///
/// Returns an empty vec if padding rows satisfy all constraints.
pub fn check_padding_via_chip<F, A>(chip: &A) -> Vec<ConstraintFailure>
where
    F: Field,
    A: MachineAir<F> + for<'a> Air<RowChecker<'a, F>>,
{
    let padding_row = chip.padding_row();
    let prep_width = chip.preprocessed_width();
    check_padding(chip, &padding_row, prep_width)
}

/// Generate a trace from an `ExecutionRecord` and return both the trace matrix and
/// all constraint failures. Useful when you want to inspect the trace alongside failures.
pub fn generate_and_check<F, A>(
    chip: &A,
    record: &ExecutionRecord,
) -> (RowMajorMatrix<F>, Vec<(usize, ConstraintFailure)>)
where
    F: Field,
    A: MachineAir<F, Record = ExecutionRecord> + for<'a> Air<RowChecker<'a, F>>,
{
    let mut output = ExecutionRecord::default();
    let trace = chip.generate_trace(record, &mut output).decompress();
    let pv = record.public_values.to_vec::<F>();
    let prep_width = chip.preprocessed_width();
    let failures = check_trace(chip, &trace, &pv, prep_width);
    (trace, failures)
}

// ---------------------------------------------------------------------------
// Check all chips
// ---------------------------------------------------------------------------

/// Result of checking a single chip's padding row.
#[derive(Debug)]
pub struct ChipPaddingResult {
    /// Name of the chip.
    pub chip_name: String,
    /// Constraint failures found on the padding row (empty if all constraints pass).
    pub failures: Vec<ConstraintFailure>,
}

/// Check **all** RiscvAir chips' padding rows against their eval constraints.
///
/// Iterates over every chip returned by `RiscvAir::get_airs_and_costs()`,
/// obtains each chip's `padding_row()`, then evaluates all AIR constraints
/// (with interactions as no-ops) on that padding row.
///
/// Returns a vec of results for each chip. Chips with no failures have an empty
/// `failures` vec.
pub fn check_all_chips_padding() -> Vec<ChipPaddingResult> {
    use crate::riscv::RiscvAir;
    use p3_baby_bear::BabyBear;

    type F = BabyBear;
    let (airs, _) = RiscvAir::<F>::get_airs_and_costs();

    airs.iter()
        .map(|air| {
            let name = air.name();
            let padding = air.padding_row();
            let prep_width = air.preprocessed_width();
            let failures = check_padding(air, &padding, prep_width);
            ChipPaddingResult { chip_name: name, failures }
        })
        .collect()
}

/// Result of checking a single chip's full trace (real + padding rows).
#[derive(Debug)]
pub struct ChipTraceResult {
    /// Name of the chip.
    pub chip_name: String,
    /// Total number of rows in the generated trace (real + padding).
    pub trace_height: usize,
    /// Constraint failures: `(row_index, failure)` for every violation found.
    pub failures: Vec<(usize, ConstraintFailure)>,
}

/// Generate traces and check **all** RiscvAir chips against their eval constraints
/// using a real `ExecutionRecord`.
///
/// For each chip, calls `generate_trace(record)` to produce the full trace (including
/// padding rows), then checks every row against the AIR constraints.
///
/// Returns a vec of results for each chip.
pub fn check_all_chips_trace(
    record: &ExecutionRecord,
    skip_chips: &[&str],
) -> Vec<ChipTraceResult> {
    use crate::riscv::RiscvAir;
    use p3_baby_bear::BabyBear;
    use p3_matrix::Matrix;

    type F = BabyBear;
    let (airs, _) = RiscvAir::<F>::get_airs_and_costs();
    let pv = record.public_values.to_vec::<F>();

    airs.iter()
        .map(|air| {
            let name = air.name();

            // Skip chips the caller doesn't want to check.
            if skip_chips.contains(&name.as_str()) {
                return ChipTraceResult { chip_name: name, trace_height: 0, failures: vec![] };
            }

            // Skip chips with no corresponding events — their trace would be
            // all padding, which is already covered by the dedicated padding check.
            if !air.included(record) {
                return ChipTraceResult { chip_name: name, trace_height: 0, failures: vec![] };
            }

            // Wrap generate_trace in catch_unwind to capture panics
            // (some chips may panic due to missing fields in deserialized records).
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut output = ExecutionRecord::default();
                let trace = air.generate_trace(record, &mut output).decompress();
                let height = trace.height();
                let prep_width = air.preprocessed_width();

                // Generate preprocessed trace for chips that have one (e.g. Byte, Program).
                let prep_trace = if prep_width > 0 {
                    air.generate_preprocessed_trace(&record.program).map(|c| c.decompress())
                } else {
                    None
                };

                let failures = check_trace_with_preprocessed(
                    air,
                    &trace,
                    &pv,
                    prep_width,
                    prep_trace.as_ref(),
                );
                (height, failures)
            }));

            match result {
                Ok((height, failures)) => {
                    ChipTraceResult { chip_name: name, trace_height: height, failures }
                }
                Err(panic_info) => {
                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    // Report the panic as a single synthetic failure at row 0.
                    ChipTraceResult {
                        chip_name: name,
                        trace_height: 0,
                        failures: vec![(
                            0,
                            ConstraintFailure {
                                constraint_index: 0,
                                kind: ConstraintKind::AssertZero,
                                lhs: format!("PANIC in generate_trace: {msg}"),
                                rhs: String::new(),
                                source_locations: vec![],
                            },
                        )],
                    }
                }
            }
        })
        .collect()
}

/// A convenience wrapper that checks all chips and returns only the ones with failures.
pub fn check_all_chips_padding_failures_only() -> Vec<ChipPaddingResult> {
    check_all_chips_padding().into_iter().filter(|r| !r.failures.is_empty()).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Capture a backtrace and extract only source locations within `crates/core/machine/src/`.
///
/// Returns a vec of strings like `"alu/divrem/mod.rs:477 (dt_core_machine::alu::divrem::...)"`.
/// At most 5 frames are returned to keep output concise.
fn capture_source_locations() -> Vec<String> {
    let bt = std::backtrace::Backtrace::force_capture();
    let bt_str = format!("{bt:#}");

    let mut locations = Vec::new();
    // Each frame in the backtrace looks like:
    //   N: <function_name>
    //              at /path/to/file.rs:LINE:COL
    // We look for lines containing "crates/core/machine/src/" in the "at" line.
    let lines: Vec<&str> = bt_str.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("at ") && trimmed.contains("crates/core/machine/src/") {
            // Extract the path portion after "crates/core/machine/src/"
            if let Some(pos) = trimmed.find("crates/core/machine/src/") {
                let rel_path = &trimmed[pos + "crates/core/machine/src/".len()..];
                // Also try to grab the function name from the preceding line
                let func_name = if i > 0 {
                    let prev = lines[i - 1].trim();
                    // The prev line is like "N: dt_core_machine::alu::divrem::..."
                    if let Some(colon_pos) = prev.find(": ") {
                        let func = &prev[colon_pos + 2..];
                        // Shorten: keep only the last 2-3 path segments
                        let parts: Vec<&str> = func.split("::").collect();
                        if parts.len() > 3 {
                            parts[parts.len() - 3..].join("::")
                        } else {
                            func.to_string()
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                if func_name.is_empty() {
                    locations.push(rel_path.to_string());
                } else {
                    locations.push(format!("{rel_path} ({func_name})"));
                }
            }
        }
    }

    // Filter out frames that are not useful for debugging constraints
    locations.retain(|loc| {
        !loc.starts_with("check_constraints.rs")   // RowChecker internals
        && !loc.starts_with("bin/")                 // binary entry point
        && !loc.starts_with("riscv/mod.rs") // RiscvAir enum delegation
    });

    // Deduplicate consecutive identical entries
    locations.dedup();

    // Cap at 5 frames to keep output manageable
    locations.truncate(5);
    locations
}

// ---------------------------------------------------------------------------
// Dependencies generation
// ---------------------------------------------------------------------------

/// Run `generate_dependencies` for all RiscvAir chips on a mutable `ExecutionRecord`.
///
/// This populates derived fields like `global_interaction_events` (from MemoryLocal,
/// MemoryGlobal, Syscall chips) and byte lookup events (from ALU, Global chips, etc.).
///
/// This replicates what `StarkMachine::generate_dependencies` does, but without requiring
/// a full `StarkMachine` instance.
pub fn run_generate_dependencies(record: &mut ExecutionRecord) {
    use crate::riscv::RiscvAir;
    use dt_stark::{air::MachineAir, MachineRecord};
    use p3_baby_bear::BabyBear;

    type F = BabyBear;
    let (airs, _) = RiscvAir::<F>::get_airs_and_costs();

    for air in &airs {
        let mut output = ExecutionRecord::default();
        air.generate_dependencies(record, &mut output);
        record.append(&mut output);
    }
}

/// Perform the full shard pipeline (defer + split + generate_dependencies) and return
/// all records ready for constraint checking.
///
/// Returns `(cpu_record, deferred_shards)` where:
/// - `cpu_record` is the original record with deferred events removed and dependencies generated.
/// - `deferred_shards` are the memory init/finalize shards with dependencies generated.
///
/// The `info_callback` is called with status messages during the pipeline.
pub fn prepare_all_records(
    mut record: ExecutionRecord,
    info_callback: impl Fn(&str),
) -> (ExecutionRecord, Vec<ExecutionRecord>) {
    use dt_stark::SplitOpts;

    // Step 1: defer — extract memory init/finalize and precompile events
    let mut deferred = record.defer();
    info_callback(&format!(
        "Deferred: {} init events, {} finalize events",
        deferred.global_memory_initialize_events.len(),
        deferred.global_memory_finalize_events.len(),
    ));

    // Step 2: split — create separate shards for memory events
    // Use a large threshold so all memory events fit in one shard.
    let split_opts = SplitOpts::new(1 << 20);
    let mut deferred_shards = deferred.split(true, None, split_opts);
    info_callback(&format!("Created {} deferred shard(s)", deferred_shards.len()));

    // Step 3: generate_dependencies on all records
    run_generate_dependencies(&mut record);
    info_callback(&format!(
        "CPU shard: {} global_interaction_events",
        record.global_interaction_events.len(),
    ));

    for (i, shard) in deferred_shards.iter_mut().enumerate() {
        run_generate_dependencies(shard);
        info_callback(&format!(
            "Deferred shard {}: {} init, {} finalize, {} global_interaction_events",
            i,
            shard.global_memory_initialize_events.len(),
            shard.global_memory_finalize_events.len(),
            shard.global_interaction_events.len(),
        ));
    }

    (record, deferred_shards)
}

// ---------------------------------------------------------------------------
// Binary entry point: see `src/bin/check_padding.rs`
// Run with: cargo run --bin check_padding
// ---------------------------------------------------------------------------

/// Diagnostic: verify JalrCols field offsets using pointer arithmetic.
pub fn verify_jalr_layout() {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    use crate::control_flow::jump::jalr::{JalrCols, NUM_JALR_COLS};

    println!("  NUM_JALR_COLS = {NUM_JALR_COLS}");
    println!("  size_of::<JalrCols<u8>>()       = {}", std::mem::size_of::<JalrCols<u8>>());
    println!("  size_of::<JalrCols<BabyBear>>()  = {}", std::mem::size_of::<JalrCols<BabyBear>>());

    // Method 1: use a function to compute field offsets (prevent optimization)
    #[inline(never)]
    fn compute_offsets() -> Vec<(String, usize)> {
        use crate::control_flow::jump::jalr::JalrCols;
        use p3_baby_bear::BabyBear;
        let cols = Box::new(JalrCols::<BabyBear>::default());
        let base = &*cols as *const _ as usize;
        let es = std::mem::size_of::<BabyBear>();
        let offsets = vec![
            ("cpu_state".into(), (&cols.cpu_state as *const _ as usize - base) / es),
            ("mem_ops".into(), (&cols.mem_ops as *const _ as usize - base) / es),
            ("mem_ops.op_a".into(), (&cols.mem_ops.op_a as *const _ as usize - base) / es),
            ("mem_ops.op_c_imm".into(), (&cols.mem_ops.op_c_imm as *const _ as usize - base) / es),
            ("add_op".into(), (&cols.add_op as *const _ as usize - base) / es),
            ("op_a_rng".into(), (&cols.op_a_range_checker as *const _ as usize - base) / es),
            ("npc_rng".into(), (&cols.next_pc_range_checker as *const _ as usize - base) / es),
            ("is_real".into(), (&cols.is_real as *const _ as usize - base) / es),
        ];
        offsets
    }
    let offsets = compute_offsets();
    println!("  Field offsets (BabyBear) via ptr (in elements):");
    for (name, off) in &offsets {
        println!("    {name:20} = {off}");
    }

    // Also check u8 layout
    #[inline(never)]
    fn compute_offsets_u8() -> Vec<(String, usize)> {
        use crate::control_flow::jump::jalr::JalrCols;
        let cols = Box::new(JalrCols::<u8>::default());
        let base = &*cols as *const _ as usize;
        let offsets = vec![
            ("cpu_state".into(), &cols.cpu_state as *const _ as usize - base),
            ("mem_ops".into(), &cols.mem_ops as *const _ as usize - base),
            ("add_op".into(), &cols.add_op as *const _ as usize - base),
            ("op_a_rng".into(), &cols.op_a_range_checker as *const _ as usize - base),
            ("is_real".into(), &cols.is_real as *const _ as usize - base),
        ];
        offsets
    }
    let offsets_u8 = compute_offsets_u8();
    println!("  Field offsets (u8) via ptr (in bytes/elements):");
    for (name, off) in &offsets_u8 {
        println!("    {name:20} = {off}");
    }

    // Check MemoryAccessCols layout
    #[inline(never)]
    fn check_memory_access_layout() -> Vec<(String, usize)> {
        use crate::memory::MemoryAccessCols;
        use p3_baby_bear::BabyBear;
        let cols = Box::new(MemoryAccessCols::<BabyBear>::default());
        let base = &*cols as *const _ as usize;
        let es = std::mem::size_of::<BabyBear>();
        let r = vec![
            ("value".into(), (&cols.value as *const _ as usize - base) / es),
            ("prev_shard".into(), (&cols.prev_shard as *const _ as usize - base) / es),
            ("prev_clk".into(), (&cols.prev_clk as *const _ as usize - base) / es),
            ("compare_clk".into(), (&cols.compare_clk as *const _ as usize - base) / es),
            ("diff_16bit".into(), (&cols.diff_16bit_limb as *const _ as usize - base) / es),
            ("diff_12bit".into(), (&cols.diff_12bit_limb as *const _ as usize - base) / es),
        ];
        r
    }
    let mac_offsets = check_memory_access_layout();
    println!("  MemoryAccessCols<BabyBear> field offsets:");
    for (name, off) in &mac_offsets {
        println!("    {name:20} = {off}");
    }

    // Check ITypeRegisterOp layout
    #[inline(never)]
    fn check_itype_layout() -> Vec<(String, usize)> {
        use crate::adapter::ITypeRegisterOp;
        use p3_baby_bear::BabyBear;
        let cols = Box::new(ITypeRegisterOp::<BabyBear>::default());
        let base = &*cols as *const _ as usize;
        let es = std::mem::size_of::<BabyBear>();
        let r = vec![
            ("op_a".into(), (&cols.op_a as *const _ as usize - base) / es),
            ("op_a_access".into(), (&cols.op_a_access as *const _ as usize - base) / es),
            ("op_a_zero".into(), (&cols.op_a_zero as *const _ as usize - base) / es),
            ("op_b".into(), (&cols.op_b as *const _ as usize - base) / es),
            ("op_b_access".into(), (&cols.op_b_access as *const _ as usize - base) / es),
            ("op_c_imm".into(), (&cols.op_c_imm as *const _ as usize - base) / es),
        ];
        r
    }
    let it_offsets = check_itype_layout();
    println!("  ITypeRegisterOp<BabyBear> field offsets:");
    for (name, off) in &it_offsets {
        println!("    {name:20} = {off}");
    }

    // Check SymbolicVariable size and JalrCols<SymbolicVariable> layout
    {
        use p3_uni_stark::SymbolicVariable;
        use std::mem::MaybeUninit;
        let sv_es = std::mem::size_of::<SymbolicVariable<BabyBear>>();
        println!("  sizeof(BabyBear)           = {}", std::mem::size_of::<BabyBear>());
        println!("  sizeof(SymbolicVariable<BabyBear>) = {sv_es}");
        println!("  sizeof(JalrCols<BabyBear>) = {}", std::mem::size_of::<JalrCols<BabyBear>>());
        println!(
            "  sizeof(JalrCols<SymbolicVariable<BabyBear>>) = {}",
            std::mem::size_of::<JalrCols<SymbolicVariable<BabyBear>>>()
        );

        // Use MaybeUninit to avoid needing Default
        let sv_cols: MaybeUninit<JalrCols<SymbolicVariable<BabyBear>>> = MaybeUninit::uninit();
        let sv_ptr = sv_cols.as_ptr();
        let sv_base = sv_ptr as usize;
        if sv_es > 0 {
            println!("  JalrCols<SymbolicVariable> field offsets (byte / element):");
            unsafe {
                let cpu_state_off = std::ptr::addr_of!((*sv_ptr).cpu_state) as usize - sv_base;
                let mem_ops_off = std::ptr::addr_of!((*sv_ptr).mem_ops) as usize - sv_base;
                let add_op_off = std::ptr::addr_of!((*sv_ptr).add_op) as usize - sv_base;
                let is_real_off = std::ptr::addr_of!((*sv_ptr).is_real) as usize - sv_base;
                println!("    cpu_state = {} / {}", cpu_state_off, cpu_state_off / sv_es);
                println!("    mem_ops   = {} / {}", mem_ops_off, mem_ops_off / sv_es);
                println!("    add_op    = {} / {}", add_op_off, add_op_off / sv_es);
                println!("    is_real   = {} / {}", is_real_off, is_real_off / sv_es);
            }
        }
    }

    // Method 2: array borrow verification
    use p3_field::PrimeField32;
    let mut values = vec![BabyBear::zero(); NUM_JALR_COLS];
    {
        use std::borrow::BorrowMut;
        let cols: &mut JalrCols<BabyBear> = values.as_mut_slice().borrow_mut();
        cols.cpu_state.shard = BabyBear::from_canonical_u32(111);
        cols.cpu_state.pc = BabyBear::from_canonical_u32(222);
        cols.mem_ops.op_a = BabyBear::from_canonical_u32(333);
        cols.mem_ops.op_a_zero = BabyBear::from_canonical_u32(444);
        cols.mem_ops.op_b = BabyBear::from_canonical_u32(555);
        cols.mem_ops.op_c_imm[0] = BabyBear::from_canonical_u32(666);
        cols.add_op.value[0] = BabyBear::from_canonical_u32(777);
        cols.is_real = BabyBear::from_canonical_u32(999);
    }

    println!("  Array borrow verification (non-zero vals):");
    for (i, v) in values.iter().enumerate() {
        let vu = v.as_canonical_u32();
        if vu != 0 {
            let label = match vu {
                111 => "cpu_state.shard",
                222 => "cpu_state.pc",
                333 => "mem_ops.op_a",
                444 => "mem_ops.op_a_zero",
                555 => "mem_ops.op_b",
                666 => "mem_ops.op_c_imm[0]",
                777 => "add_op.value[0]",
                999 => "is_real",
                _ => "???",
            };
            println!("    col[{i:2}] = {vu:4} ← {label}");
        }
    }
}

// ---------------------------------------------------------------------------
// Shard capacity analysis (corrected model)
// ---------------------------------------------------------------------------
//
// Key insight: MemoryLocal events are per UNIQUE ADDRESS touched in the shard,
// NOT per instruction. Each address (register or memory) that is accessed in
// the shard produces exactly 1 MemoryLocalEvent (with initial and final access).
//
// - Registers: at most 31 unique addresses (x1..x31, x0 is hardwired)
// - Memory addresses: depends on program; each load/store touches 1 address, but repeated accesses
//   to the same address share the same event.
//
// Global chip rows = 2 * unique_addresses (init + finalize per address).
// Global chip IS in the same shard as instruction chips (execution shard).

/// Result of shard capacity analysis.
#[derive(Debug, Clone)]
pub struct ShardCapacityResult {
    pub scenario: String,
    pub bottleneck: String,
    pub max_by_height: u64,
    pub max_by_cells: u64,
    pub effective_max: u64,
    pub details: Vec<(String, u64, u64)>, // (chip, padded_height, cells)
}

/// Parameters for the memory model in a shard.
#[derive(Debug, Clone, Copy)]
pub struct MemoryModel {
    /// Number of unique register addresses touched (max 31).
    pub unique_registers: u64,
    /// Number of unique memory addresses touched.
    /// For analysis, this can be a fixed count or a function of N.
    pub unique_mem_addrs: UniqueMemModel,
}

#[derive(Debug, Clone, Copy)]
pub enum UniqueMemModel {
    /// Fixed number of unique memory addresses regardless of N.
    Fixed(u64),
    /// Fraction of load/store instructions that touch new addresses.
    /// unique_mem = load_store_count * new_addr_fraction
    /// (accounts for locality/reuse).
    FractionOfLoadStore(f64),
}

fn load_costs() -> std::collections::HashMap<&'static str, u64> {
    let mut m = std::collections::HashMap::new();
    m.insert("Add", 47u64);
    m.insert("Addi", 47);
    m.insert("Sub", 47);
    m.insert("Mul", 83);
    m.insert("Bitwise", 41);
    m.insert("ShiftLeft", 68);
    m.insert("ShiftRight", 134);
    m.insert("DivRem", 146);
    m.insert("Lt", 53);
    m.insert("Auipc", 41);
    m.insert("Branch", 58);
    m.insert("Jal", 50);
    m.insert("Jalr", 50);
    m.insert("LoadByte", 93);
    m.insert("LoadHalf", 93);
    m.insert("LoadWord", 93);
    m.insert("StoreByte", 93);
    m.insert("StoreHalf", 93);
    m.insert("StoreWord", 93);
    m.insert("SyscallInstrs", 80);
    m.insert("SyscallCore", 22);
    m.insert("MemoryLocal", 100);
    m.insert("Global", 428);
    m.insert("Byte", 52);
    m.insert("Program", 31);
    m
}

fn compute_unique_addresses(n: u64, mix: &[(&str, f64)], mem_model: &MemoryModel) -> u64 {
    let load_store_types =
        ["LoadByte", "LoadHalf", "LoadWord", "StoreByte", "StoreHalf", "StoreWord"];
    let load_store_count: u64 = mix
        .iter()
        .filter(|(name, _)| load_store_types.contains(name))
        .map(|(_, frac)| (*frac * n as f64).ceil() as u64)
        .sum();

    let unique_mem = match mem_model.unique_mem_addrs {
        UniqueMemModel::Fixed(v) => v,
        UniqueMemModel::FractionOfLoadStore(frac) => (load_store_count as f64 * frac).ceil() as u64,
    };

    mem_model.unique_registers + unique_mem
}

/// Check if N instructions with the given mix fit within thresholds.
fn check_capacity_v2(
    n: u64,
    mix: &[(&str, f64)],
    costs: &std::collections::HashMap<&'static str, u64>,
    height_threshold: u64,
    available_cells: u64,
    mem_model: &MemoryModel,
) -> (bool, bool) {
    let mut max_height: u64 = 0;
    let mut total_cells: u64 = 0;

    for (name, frac) in mix {
        let count = (*frac * n as f64).ceil() as u64;
        if count == 0 {
            continue;
        }
        let padded = count.next_power_of_two();
        let width = costs.get(*name).copied().unwrap_or(0);
        total_cells += padded * width;
        max_height = max_height.max(padded);
    }

    let unique_addrs = compute_unique_addresses(n, mix, mem_model);

    // MemoryLocal: 1 event per unique address, packed 4 per row
    let mem_local_rows = unique_addrs.div_ceil(4).max(1);
    let mem_local_padded = mem_local_rows.next_power_of_two();
    total_cells += mem_local_padded * costs["MemoryLocal"];
    max_height = max_height.max(mem_local_padded);

    // Global: 2 rows per unique address (init + finalize)
    let global_rows = (2 * unique_addrs).max(1);
    let global_padded = global_rows.next_power_of_two();
    total_cells += global_padded * costs["Global"];
    max_height = max_height.max(global_padded);

    (max_height <= height_threshold, total_cells <= available_cells)
}

fn find_max_instructions(
    mix: &[(&str, f64)],
    costs: &std::collections::HashMap<&'static str, u64>,
    height_threshold: u64,
    available_cells: u64,
    mem_model: &MemoryModel,
) -> u64 {
    let mut lo: u64 = 1;
    let mut hi: u64 = 1 << 26;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let (h_ok, c_ok) =
            check_capacity_v2(mid, mix, costs, height_threshold, available_cells, mem_model);
        if h_ok && c_ok {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Analyze how many instructions a shard can hold under a given instruction mix
/// and memory model.
pub fn analyze_shard_capacity(
    scenario: &str,
    mix: &[(&str, f64)],
    height_threshold: u64,
    element_threshold: u64,
    program_size: u32,
    mem_model: &MemoryModel,
) -> ShardCapacityResult {
    let costs = load_costs();

    let byte_rows: u64 = 1 << 16;
    let program_rows = (program_size as u64).next_power_of_two();
    let fixed_cells = byte_rows * costs["Byte"] + program_rows * costs["Program"];
    let available_cells = element_threshold.saturating_sub(fixed_cells);

    let n = find_max_instructions(mix, &costs, height_threshold, available_cells, mem_model);

    // Determine bottleneck at n+1
    let (h_ok, _c_ok) =
        check_capacity_v2(n + 1, mix, &costs, height_threshold, available_cells, mem_model);
    let bottleneck_name = if !h_ok { "height_threshold" } else { "element_threshold" };

    // Build details at n
    let mut details = Vec::new();
    let mut total_cells_dynamic = 0u64;
    let mut max_h = 0u64;

    for (name, frac) in mix {
        let count = (*frac * n as f64).round() as u64;
        if count == 0 {
            continue;
        }
        let padded = count.next_power_of_two();
        let width = costs.get(*name).copied().unwrap_or(0);
        let cells = padded * width;
        details.push((name.to_string(), padded, cells));
        total_cells_dynamic += cells;
        max_h = max_h.max(padded);
    }

    let unique_addrs = compute_unique_addresses(n, mix, mem_model);
    let mem_local_rows = unique_addrs.div_ceil(4).max(1);
    let mem_local_padded = mem_local_rows.next_power_of_two();
    let mem_local_cells = mem_local_padded * costs["MemoryLocal"];
    details.push((
        format!("MemoryLocal ({unique_addrs} addrs)"),
        mem_local_padded,
        mem_local_cells,
    ));
    total_cells_dynamic += mem_local_cells;
    max_h = max_h.max(mem_local_padded);

    let global_rows = (2 * unique_addrs).max(1);
    let global_padded = global_rows.next_power_of_two();
    let global_cells = global_padded * costs["Global"];
    details.push((format!("Global (2*{unique_addrs})"), global_padded, global_cells));
    total_cells_dynamic += global_cells;
    max_h = max_h.max(global_padded);

    // Compute height-only and cells-only limits
    let max_by_height = find_max_instructions(mix, &costs, height_threshold, u64::MAX, mem_model);
    let max_by_cells = find_max_instructions(mix, &costs, u64::MAX, available_cells, mem_model);

    ShardCapacityResult {
        scenario: scenario.to_string(),
        bottleneck: format!(
            "{bottleneck_name} (max_chip_height={max_h}, total_dynamic_cells={total_cells_dynamic})"
        ),
        max_by_height,
        max_by_cells,
        effective_max: n,
        details,
    }
}

/// Print a comprehensive shard capacity analysis report.
pub fn print_shard_capacity_report() {
    use dt_stark::{SHARD_CELLS_THRESHOLD, SHARD_HEIGHT_THRESHOLD};

    let height_threshold = SHARD_HEIGHT_THRESHOLD; // 2^22
    let element_threshold = SHARD_CELLS_THRESHOLD; // 2^28 + 2^27
    let program_size = 4096u32;

    let costs = load_costs();
    let byte_rows: u64 = 1 << 16;
    let program_rows = (program_size as u64).next_power_of_two();
    let fixed_cells = byte_rows * costs["Byte"] + program_rows * costs["Program"];

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║              Shard Capacity Analysis Report (Corrected)               ║");
    println!("╠════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ height_threshold  = {:>12} (2^{})                              ║",
        height_threshold,
        (height_threshold as f64).log2() as u32
    );
    println!(
        "║ element_threshold = {:>12} ({:.1} M cells, 2^28+2^27)         ║",
        element_threshold,
        element_threshold as f64 / 1e6
    );
    println!("║ program_size      = {program_size:>12} instructions                        ║");
    println!("║ fixed_overhead    = {fixed_cells:>12} cells (Byte 2^16 + Program)         ║");
    println!(
        "║ available_cells   = {:>12} cells                              ║",
        element_threshold - fixed_cells
    );
    println!("╠════════════════════════════════════════════════════════════════════════╣");
    println!("║ KEY CORRECTION: MemoryLocal events = unique ADDRESSES touched,       ║");
    println!("║ NOT per-instruction. Registers ≤ 31; memory addrs depend on program. ║");
    println!("║ Global rows = 2 * unique_addresses (init + finalize per address).    ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    // Print per-instruction cost table
    println!("\n=== Per-Instruction Chip Costs ===\n");
    println!("  {:20} {:>8} {:>10}", "Chip", "Width", "Category");
    println!("  {}", "-".repeat(42));
    let mut sorted_costs: Vec<_> = costs.iter().collect();
    sorted_costs.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    for (name, width) in &sorted_costs {
        let cat = match **name {
            "Add" | "Addi" | "Sub" => "ALU-light",
            "Mul" | "DivRem" | "Bitwise" | "ShiftLeft" | "ShiftRight" | "Lt" => "ALU-heavy",
            "Branch" | "Jal" | "Jalr" | "Auipc" => "Control",
            "LoadByte" | "LoadHalf" | "LoadWord" | "StoreByte" | "StoreHalf" | "StoreWord" => {
                "Memory"
            }
            "Global" | "MemoryLocal" | "Byte" | "Program" => "Infra",
            _ => "Other",
        };
        println!("  {name:20} {width:>8} {cat:>10}");
    }

    // zkDTVM v5 reference: CPU chip height = N, height_limit = 2^21 → max ~2M instructions
    println!("\n=== zkDTVM v5.0.0 Reference ===");
    println!("  CPU chip height = instruction_count, height_limit = 2^21");
    println!("  → max ~2M instructions per shard (height-limited)");
    println!("  After refactoring: CPU chip removed, each instruction type has its own chip.");
    println!("  → The most common instruction type determines max height, which is much");
    println!("    smaller than N for distributed mixes.\n");

    let all_instrs = [
        "Add",
        "Addi",
        "Sub",
        "Mul",
        "Bitwise",
        "ShiftLeft",
        "ShiftRight",
        "DivRem",
        "Lt",
        "Branch",
        "Jal",
        "Jalr",
        "Auipc",
        "LoadByte",
        "LoadHalf",
        "LoadWord",
        "StoreByte",
        "StoreHalf",
        "StoreWord",
    ];
    let equal_frac = 1.0 / all_instrs.len() as f64;
    let avg_mix: Vec<(&str, f64)> = all_instrs.iter().map(|n| (*n, equal_frac)).collect();

    let typical_mix: Vec<(&str, f64)> = vec![
        ("Add", 0.20),
        ("Addi", 0.25),
        ("Sub", 0.15),
        ("LoadWord", 0.15),
        ("StoreWord", 0.10),
        ("LoadByte", 0.05),
        ("Branch", 0.05),
        ("Jal", 0.02),
        ("Jalr", 0.03),
    ];

    let worst_mix: Vec<(&str, f64)> = vec![("DivRem", 1.0)];

    // Memory models to test:
    // 1. Low memory pressure: 31 regs + 256 fixed mem addrs (tight loop over small data)
    let mem_low =
        MemoryModel { unique_registers: 31, unique_mem_addrs: UniqueMemModel::Fixed(256) };
    // 2. Medium memory pressure: 31 regs + 10% of load/stores touch new addresses
    let mem_medium = MemoryModel {
        unique_registers: 31,
        unique_mem_addrs: UniqueMemModel::FractionOfLoadStore(0.10),
    };
    // 3. High memory pressure: 31 regs + 50% of load/stores touch new addresses
    let mem_high = MemoryModel {
        unique_registers: 31,
        unique_mem_addrs: UniqueMemModel::FractionOfLoadStore(0.50),
    };
    // 4. Extreme: every load/store is a unique address
    let mem_extreme = MemoryModel {
        unique_registers: 31,
        unique_mem_addrs: UniqueMemModel::FractionOfLoadStore(1.0),
    };

    let scenarios: Vec<(&str, &[(&str, f64)], &MemoryModel)> = vec![
        // --- Scenario 1: Average case, low memory ---
        ("1a. Average mix, low memory (31 regs + 256 mem addrs)", &avg_mix, &mem_low),
        ("1b. Average mix, medium memory (10% new addrs per LS)", &avg_mix, &mem_medium),
        ("1c. Average mix, high memory (50% new addrs per LS)", &avg_mix, &mem_high),
        // --- Scenario 2: Worst case (DivRem) ---
        ("2a. Worst (100% DivRem), low memory", &worst_mix, &mem_low),
        // --- Scenario 3: Typical workload ---
        ("3a. Typical (60%ALU/30%Mem/10%Ctrl), low memory", &typical_mix, &mem_low),
        ("3b. Typical, medium memory (10% new)", &typical_mix, &mem_medium),
        ("3c. Typical, high memory (50% new)", &typical_mix, &mem_high),
        ("3d. Typical, extreme memory (100% new)", &typical_mix, &mem_extreme),
    ];

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "ANALYSIS RESULTS");
    println!("{}", "=".repeat(80));

    // Summary table first
    println!("\n{:60} {:>8} {:>10}", "Scenario", "Max N", "Bottleneck");
    println!("{}", "-".repeat(80));
    let mut results = Vec::new();
    for (label, mix, mem) in &scenarios {
        let r = analyze_shard_capacity(
            label,
            mix,
            height_threshold,
            element_threshold,
            program_size,
            mem,
        );
        let bn_short = if r.bottleneck.contains("height") { "Height" } else { "Cells" };
        println!("{:60} {:>8} {:>10}", label, r.effective_max, bn_short);
        results.push(r);
    }

    // Detailed breakdown for each scenario
    for result in &results {
        println!("\n{}", "-".repeat(80));
        println!("Scenario: {}", result.scenario);
        println!("{}", "-".repeat(80));
        println!("  max_by_height  = {:>10} instructions", result.max_by_height);
        println!("  max_by_cells   = {:>10} instructions", result.max_by_cells);
        println!("  effective_max  = {:>10} instructions", result.effective_max);
        println!("  bottleneck     = {}", result.bottleneck);
        println!("\n  {:25} {:>12} {:>14} {:>6}", "Chip", "PaddedHeight", "Cells", "Pct");
        println!("  {}", "-".repeat(60));
        let total: u64 = result.details.iter().map(|(_, _, c)| c).sum();
        for (chip, height, cells) in &result.details {
            let pct = if total > 0 { *cells as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  {chip:25} {height:>12} {cells:>14} {pct:>5.1}%");
        }
        println!("  {:25} {:>12} {:>14}", "TOTAL (dynamic)", "", total);
        println!("  {:25} {:>12} {:>14}", "Fixed (Byte+Prog)", "", fixed_cells);
        println!("  {:25} {:>12} {:>14}", "GRAND TOTAL", "", total + fixed_cells);
    }
}
