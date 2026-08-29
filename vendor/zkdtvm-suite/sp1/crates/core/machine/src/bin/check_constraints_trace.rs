//! Check all chips' real+padding trace constraints using a serialized ExecutionRecord.
//!
//! Usage (with source locations in constraint failures):
//!   cargo run --profile fast --bin check_constraints_trace -p dt-core-machine
//!
//! Also works in pure release mode (faster compile, but no source locations):
//!   cargo run --release --bin check_constraints_trace -p dt-core-machine
//!
//! Reads `execution_record.json` from the current directory, generates traces
//! for each chip, and checks all rows against AIR constraints.
//!
//! The tool performs the full shard pipeline:
//! 1. Load the original ExecutionRecord
//! 2. defer() + split() to create separate memory shards
//! 3. generate_dependencies() on all records
//! 4. Check constraints on CPU shard and deferred shards

use std::{collections::BTreeMap, fs};

use dt_core_executor::ExecutionRecord;
use dt_core_machine::check_constraints::{check_all_chips_trace, prepare_all_records};

/// Chips to skip (precompiles only — no test ELF events for these).
const SKIP_CHIPS: &[&str] = &[
    "ShaExtend",
    "ShaCompress",
    "EdAddAssign",
    "EdDecompress",
    "Secp256k1Decompress",
    "Secp256k1AddAssign",
    "Secp256k1DoubleAssign",
    "Secp256r1Decompress",
    "Secp256r1AddAssign",
    "Secp256r1DoubleAssign",
    "Bn254AddAssign",
    "Bn254DoubleAssign",
    "Bls12381AddAssign",
    "Bls12381DoubleAssign",
    "Uint256MulMod",
    "U256XU2048Mul",
    "Bls12381FpOpAssign",
    "Bls12381Fp2AddSubAssign",
    "Bls12381Fp2MulAssign",
    "Bn254FpOpAssign",
    "Bn254Fp2AddSubAssign",
    "Bn254Fp2MulAssign",
    "Bls12381Decompress",
    "KeccakPermute",
];

fn main() {
    let json_path = "execution_record.json";
    println!("Reading {json_path} ...");

    let json =
        fs::read_to_string(json_path).unwrap_or_else(|e| panic!("Failed to read {json_path}: {e}"));

    let records: Vec<ExecutionRecord> = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Failed to deserialize {json_path}: {e}"));

    println!("Loaded {} shard(s)", records.len());

    let record = records.into_iter().next().expect("no records");

    // --- Full pipeline: defer + split + generate_dependencies ---
    println!("\nPreparing records (defer + split + dependencies) ...");
    let (cpu_record, deferred_shards) = prepare_all_records(record, |msg| {
        println!("  {msg}");
    });

    // --- Check CPU shard ---
    println!("\n=== CPU Shard Trace Constraint Check ===\n");
    let cpu_results = check_all_chips_trace(&cpu_record, SKIP_CHIPS);
    let (mut pass_count, mut fail_count, mut skip_count, mut empty_count) =
        print_results(&cpu_results, SKIP_CHIPS);

    // --- Check deferred shards ---
    for (i, shard) in deferred_shards.iter().enumerate() {
        println!("\n=== Deferred Shard {i} Trace Constraint Check ===\n");
        let results = check_all_chips_trace(shard, SKIP_CHIPS);
        let (p, f, s, e) = print_results(&results, SKIP_CHIPS);
        pass_count += p;
        fail_count += f;
        skip_count += s;
        empty_count += e;
    }

    // --- Overall summary ---
    let total_records = 1 + deferred_shards.len();
    println!(
        "\n=== Overall Summary ({total_records} record(s)): {pass_count} passed, {fail_count} failed, {empty_count} empty, {skip_count} skipped ==="
    );

    if fail_count > 0 {
        std::process::exit(1);
    } else {
        println!("\nAll checked chips' traces satisfy eval constraints.");
    }
}

/// Print results for one record and return (pass, fail, skip, empty).
fn print_results(
    results: &[dt_core_machine::check_constraints::ChipTraceResult],
    skip_chips: &[&str],
) -> (usize, usize, usize, usize) {
    let mut pass_count = 0;
    let mut fail_count = 0;
    let mut skip_count = 0;
    let mut empty_count = 0;

    for result in results {
        if skip_chips.contains(&result.chip_name.as_str()) {
            skip_count += 1;
            println!("  [SKIP] {}", result.chip_name);
            continue;
        }

        if result.trace_height == 0 && result.failures.is_empty() {
            empty_count += 1;
            println!("  [EMPTY] {} (no events, covered by padding check)", result.chip_name);
            continue;
        }

        if result.failures.is_empty() {
            pass_count += 1;
            println!("  [PASS] {} (trace height: {})", result.chip_name, result.trace_height);
        } else {
            fail_count += 1;
            let total = result.failures.len();
            println!(
                "  [FAIL] {} — {} violation(s) (trace height: {})",
                result.chip_name, total, result.trace_height
            );

            let mut failing_rows: Vec<usize> = result.failures.iter().map(|(r, _)| *r).collect();
            failing_rows.sort();
            failing_rows.dedup();
            println!(
                "         failing rows: {} unique out of {} total",
                failing_rows.len(),
                result.trace_height
            );

            let mut by_constraint: BTreeMap<
                usize,
                (&dt_core_machine::check_constraints::ConstraintFailure, Vec<usize>),
            > = BTreeMap::new();
            for (row, f) in &result.failures {
                by_constraint
                    .entry(f.constraint_index)
                    .and_modify(|(_, rows)| rows.push(*row))
                    .or_insert((f, vec![*row]));
            }

            println!("         {} unique constraint(s) failing:", by_constraint.len());
            for (idx, (failure, rows)) in &by_constraint {
                let kind_str = match failure.kind {
                    dt_core_machine::check_constraints::ConstraintKind::AssertZero => {
                        format!("assert_zero failed: got {}", failure.lhs)
                    }
                    dt_core_machine::check_constraints::ConstraintKind::AssertOne => {
                        format!("assert_one failed: got {}", failure.lhs)
                    }
                    dt_core_machine::check_constraints::ConstraintKind::AssertEq => {
                        format!("assert_eq failed: {} != {}", failure.lhs, failure.rhs)
                    }
                    dt_core_machine::check_constraints::ConstraintKind::AssertBool => {
                        format!("assert_bool failed: {} is not 0 or 1", failure.lhs)
                    }
                };

                let row_summary = if rows.len() <= 5 {
                    rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
                } else {
                    let first: Vec<String> = rows[..3].iter().map(|r| r.to_string()).collect();
                    format!("{}, ... ({} rows total)", first.join(", "), rows.len())
                };

                println!("           #{idx}: {kind_str}");
                println!("               rows: [{row_summary}]");

                if !failure.source_locations.is_empty() {
                    for loc in &failure.source_locations {
                        println!("               at {loc}");
                    }
                } else {
                    println!("               (no source location — rebuild with: --profile fast)");
                }
            }
        }
    }

    println!(
        "\n  --- Record summary: {pass_count} passed, {fail_count} failed, {empty_count} empty, {skip_count} skipped ---"
    );

    (pass_count, fail_count, skip_count, empty_count)
}
