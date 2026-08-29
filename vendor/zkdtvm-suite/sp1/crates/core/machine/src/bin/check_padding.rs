//! Binary to check all RiscvAir chips' padding rows against their eval constraints.
//!
//! Run from the machine crate directory:
//!   cargo run --bin check_padding
//!
//! Output is designed for AI consumption: each failure includes a constraint index
//! and source location so the AI can quickly open the relevant eval code.
//!
//! To check with backtraces, build in debug mode (default for `cargo run`).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use dt_core_machine::check_constraints::check_all_chips_padding;

/// Chips to skip — these are known to require special handling (precompiles with
/// complex field-op padding, chips with preprocessed traces or next-row dependencies
/// not fully supported by single-row checking).
const SKIP_CHIPS: &[&str] = &[
    // Precompiles (field-op padding not yet fully correct)
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
    // Global (next-row accumulation dependency)
    "Global",
    // MemoryGlobal (next-row dependency)
    "MemoryGlobalInit",
    "MemoryGlobalFinalize",
    // Preprocessed-only chips (all-zero preprocessed row may not be valid)
    "Byte",
    "Program",
];

fn main() {
    let results = check_all_chips_padding();

    let mut pass_count = 0;
    let mut fail_count = 0;
    let mut skip_count = 0;
    let mut all_passed = true;

    println!("=== Padding Row Constraint Check ===\n");

    for result in &results {
        if SKIP_CHIPS.contains(&result.chip_name.as_str()) {
            skip_count += 1;
            println!("  [SKIP] {}", result.chip_name);
            continue;
        }

        if result.failures.is_empty() {
            pass_count += 1;
            println!("  [PASS] {}", result.chip_name);
        } else {
            fail_count += 1;
            all_passed = false;
            println!("  [FAIL] {} — {} violation(s):", result.chip_name, result.failures.len());
            for f in &result.failures {
                // Each failure prints as:
                //   #N: assert_* failed: <values>
                //       at <relative_path>:<line> (<function>)
                //       at <relative_path>:<line> (<function>)
                // The indentation uses 9 spaces to align under the "- " prefix.
                let display = format!("{f}");
                for (i, line) in display.lines().enumerate() {
                    if i == 0 {
                        println!("         {line}");
                    } else {
                        println!("             {line}");
                    }
                }
            }
        }
    }

    println!(
        "\n--- Summary: {} passed, {} failed, {} skipped, {} total ---",
        pass_count,
        fail_count,
        skip_count,
        results.len()
    );

    if !all_passed {
        let failed: Vec<_> = results
            .iter()
            .filter(|r| !r.failures.is_empty() && !SKIP_CHIPS.contains(&r.chip_name.as_str()))
            .map(|r| &r.chip_name)
            .collect();
        eprintln!("\nFAILED chip(s): {failed:?}");
        eprintln!("\nTo debug: open the source files shown in 'at' lines above.");
        eprintln!("The constraint index (#N) counts every assert_* call in eval order,");
        eprintln!("including those inside sub-operation evals (IsZeroOperation, etc.).");
        std::process::exit(1);
    } else {
        println!("\nAll checked chips' padding rows satisfy eval constraints.");
    }
}
