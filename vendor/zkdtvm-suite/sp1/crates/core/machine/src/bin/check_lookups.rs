//! Check Local-scope lookup (interaction) consistency for a single shard.
//!
//! Usage:
//!   cargo run --profile fast --bin check_lookups -p dt-core-machine
//!
//! Reads `execution_record.json` from the current directory, runs generate_dependencies
//! to populate byte lookups, then checks that all Local-scope send/receive interactions
//! balance within the shard.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs;

use dt_core_executor::ExecutionRecord;
use dt_core_machine::{
    check_constraints::prepare_all_records,
    check_lookups::{check_all_lookups_local, collect_chip_kind_summaries},
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};

/// Precompile chips to skip (no events in our test ELF).
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

    // Prepare records: defer + split + generate_dependencies
    println!("\nPreparing records ...");
    let (cpu_record, deferred_shards) = prepare_all_records(record, |msg| {
        println!("  {msg}");
    });

    // --- Per-chip byte lookup diagnostic ---
    println!("\n=== Per-Chip Byte Lookup: generate_dependencies vs eval ===\n");
    {
        use dt_stark::air::MachineAir;
        use p3_baby_bear::BabyBear;
        type F = BabyBear;
        let (chips, _) = dt_core_machine::riscv::RiscvAir::<F>::get_chips_and_costs();

        println!("  {:30} {:>12} {:>12} {:>12}", "Chip", "DepByteLU", "EvalByteSend", "Diff");
        println!("  {}", "-".repeat(70));

        let mut total_dep = 0i64;
        let mut total_eval = 0i64;

        for chip in &chips {
            let name = chip.name();
            if SKIP_CHIPS.contains(&name.as_str()) || !chip.included(&cpu_record) {
                continue;
            }
            if name == "Byte" {
                continue;
            } // skip ByteChip itself

            // 1) Count byte lookups from generate_dependencies
            let mut dep_output = dt_core_executor::ExecutionRecord::default();
            let dep_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                chip.generate_dependencies(&cpu_record, &mut dep_output);
            }));
            let dep_byte_count: i64 = if dep_result.is_ok() {
                dep_output.byte_lookups.values().map(|v| *v as i64).sum()
            } else {
                -1 // error
            };

            // 2) Count byte sends from eval (using the summary data)
            // We already have summaries below, but let's just compute it here too
            // by evaluating the interactions on the trace
            let eval_byte_count: i64 = {
                let trace_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut output = dt_core_executor::ExecutionRecord::default();
                    let main_trace = chip.generate_trace(&cpu_record, &mut output);
                    let prep_trace = if chip.preprocessed_width() > 0 {
                        chip.generate_preprocessed_trace(&cpu_record.program)
                            .map(|c| c.decompress())
                    } else {
                        None
                    };
                    (main_trace.decompress(), prep_trace)
                }));
                match trace_result {
                    Ok((main_trace, prep_trace)) => {
                        use dt_stark::air::InteractionScope;
                        use p3_field::PrimeField32;
                        use p3_matrix::Matrix;
                        let sends = chip.sends();
                        let height = main_trace.height();
                        let empty_prep = vec![F::zero(); 1];
                        let mut count: i64 = 0;
                        for row in 0..height {
                            let main_row = main_trace.row_slice(row);
                            let prep_row: Vec<F> = if let Some(pt) = prep_trace.as_ref() {
                                pt.row_slice(row).to_vec()
                            } else {
                                empty_prep.clone()
                            };
                            for interaction in sends.iter().filter(|i| {
                                i.scope == InteractionScope::Local &&
                                    i.kind == dt_stark::InteractionKind::Byte
                            }) {
                                let mult: F =
                                    interaction.multiplicity.apply::<F, F>(&prep_row, &main_row);
                                if !mult.is_zero() {
                                    count += mult.as_canonical_u32() as i64;
                                }
                            }
                        }
                        count
                    }
                    Err(_) => -1,
                }
            };

            let diff = dep_byte_count - eval_byte_count;
            if diff != 0 {
                println!(
                    "  {name:30} {dep_byte_count:>12} {eval_byte_count:>12} {diff:>12} ← MISMATCH"
                );
            } else {
                println!("  {name:30} {dep_byte_count:>12} {eval_byte_count:>12} {diff:>12}");
            }
            total_dep += dep_byte_count;
            total_eval += eval_byte_count;
        }
        println!("  {}", "-".repeat(70));
        println!(
            "  {:30} {:>12} {:>12} {:>12}",
            "TOTAL",
            total_dep,
            total_eval,
            total_dep - total_eval
        );
        println!(
            "  record.byte_lookups total: {}",
            cpu_record.byte_lookups.values().map(|v| *v as i64).sum::<i64>()
        );
    }

    // --- Per-chip, per-kind summary ---
    println!("\n=== CPU Shard: Per-Chip Memory/Program/Byte Summary ===\n");
    let summaries = collect_chip_kind_summaries(&cpu_record, SKIP_CHIPS);
    println!(
        "  {:30} {:10} {:>10} {:>10} {:>10} {:>10}",
        "Chip", "Kind", "SendTot", "RecvTot", "SendRows", "RecvRows"
    );
    println!("  {}", "-".repeat(80));
    for s in &summaries {
        if s.kind == "Memory" || s.kind == "Program" || s.kind == "Byte" || s.kind == "State" {
            println!(
                "  {:30} {:10} {:>10} {:>10} {:>10} {:>10}",
                s.chip_name, s.kind, s.total_send, s.total_recv, s.send_entries, s.recv_entries
            );
        }
    }

    // --- Diagnostic: verify layout ---
    println!("\n=== JalrCols Layout Verification ===");
    dt_core_machine::check_constraints::verify_jalr_layout();

    // --- Diagnostic: struct sizes ---
    println!("\n=== Struct sizes (u8) ===");
    println!(
        "  CPUState:         {}",
        std::mem::size_of::<dt_core_machine::adapter::CPUState<u8>>()
    );
    println!(
        "  ITypeRegisterOp:  {}",
        std::mem::size_of::<dt_core_machine::adapter::ITypeRegisterOp<u8>>()
    );
    println!(
        "  AddOperation:     {}",
        std::mem::size_of::<dt_core_machine::operations::AddOperation<u8>>()
    );
    println!(
        "  BabyBearWordRC:   {}",
        std::mem::size_of::<dt_core_machine::operations::BabyBearWordRangeChecker<u8>>()
    );
    println!(
        "  MemoryReadWriteCols:  {}",
        std::mem::size_of::<dt_core_machine::memory::MemoryReadWriteCols<u8>>()
    );
    println!(
        "  MemoryReadCols:       {}",
        std::mem::size_of::<dt_core_machine::memory::MemoryReadCols<u8>>()
    );
    println!(
        "  MemoryAccessCols:     {}",
        std::mem::size_of::<dt_core_machine::memory::MemoryAccessCols<u8>>()
    );
    println!("  Word<u8>:             {}", std::mem::size_of::<dt_stark::Word<u8>>());

    // --- Diagnostic: dump Jalr event[0] ITypeRecord ---
    println!("\n=== Diagnostic: Jalr event[0] ITypeRecord ===\n");
    if !cpu_record.jalr_events.is_empty() {
        let (rec, evt) = &cpu_record.jalr_events[0];
        println!(
            "  ITypeRecord: clk={}, op_a={}, op_b={}, op_c={}",
            rec.clk, rec.op_a, rec.op_b, rec.op_c
        );
        println!("  JumpEvent:   pc={}, b={}, c={}, next_pc={}", evt.pc, evt.b, evt.c, evt.next_pc);
        println!("  ITypeRecord.a = {:?}", rec.a);
        println!("  ITypeRecord.b = {:?}", rec.b);
    }

    // --- Diagnostic: dump Jalr row 0 via struct aliasing ---
    println!("\n=== Diagnostic: Jalr row 0 via struct ===\n");
    {
        use dt_stark::air::MachineAir;
        use p3_field::PrimeField32;
        use p3_matrix::Matrix;
        let (chips, _) =
            dt_core_machine::riscv::RiscvAir::<p3_baby_bear::BabyBear>::get_chips_and_costs();
        for chip in &chips {
            if chip.name() != "Jalr" {
                continue;
            }
            let mut output = dt_core_executor::ExecutionRecord::default();
            let trace = chip.generate_trace(&cpu_record, &mut output).decompress();
            let width = trace.width();
            let height = trace.height();
            println!(
                "  Jalr trace: width={}, height={}, chip.width()={}",
                width,
                height,
                chip.width()
            );

            let row0 = trace.row_slice(0);
            // Layout (manually computed from struct definitions):
            // CPUState: [shard, clk_16_28, clk_0_16, pc] = 4 cols
            // ITypeRegisterOp: [op_a, op_a_access(13), op_a_zero, op_b, op_b_access(9),
            // op_c_imm(4)] = 29 cols   op_a_access = MemoryReadWriteCols =
            // [prev_value(4), access(9)]   access = MemoryAccessCols = [value(4),
            // prev_shard, prev_clk, compare_clk, diff_16bit, diff_8bit] AddOperation:
            // [value(4)] = 4 cols op_a_range: [1], next_pc_range: [1], is_real: [1] = 3
            // cols Total: 4 + 29 + 4 + 3 = 40 ✓
            let v = |i: usize| -> u32 { row0[i].as_canonical_u32() };
            println!("  Layout analysis (known offsets):");
            println!("    [0] shard        = {}", v(0));
            println!("    [1] clk_16_28    = {}", v(1));
            println!("    [2] clk_0_16     = {}", v(2));
            println!("    [3] pc           = {}", v(3));
            println!("    [4] op_a         = {}", v(4));
            println!("    [5..9] prev_val  = [{},{},{},{}]", v(5), v(6), v(7), v(8));
            println!("    [9..13] value    = [{},{},{},{}]", v(9), v(10), v(11), v(12));
            println!("    [13] prev_shard  = {}", v(13));
            println!("    [14] prev_clk    = {}", v(14));
            println!("    [15] compare_clk = {}", v(15));
            println!("    [16] diff_16     = {}", v(16));
            println!("    [17] diff_8      = {}", v(17));
            println!("    [18] op_a_zero   = {}", v(18));
            println!("    [19] op_b        = {}", v(19));
            println!("    [20..24] b_value = [{},{},{},{}]", v(20), v(21), v(22), v(23));
            println!("    [24] b_prev_shd  = {}", v(24));
            println!("    [25] b_prev_clk  = {}", v(25));
            println!("    [26] b_cmp_clk   = {}", v(26));
            println!("    [27] b_diff16    = {}", v(27));
            println!("    [28] b_diff8     = {}", v(28));
            println!("    [29..33] c_imm   = [{},{},{},{}]", v(29), v(30), v(31), v(32));
            println!("    [33..37] add_val = [{},{},{},{}]", v(33), v(34), v(35), v(36));
            println!("    [37] a_range     = {}", v(37));
            println!("    [38] npc_range   = {}", v(38));
            println!("    [39] is_real     = {}", v(39));

            // Also print expected from event data
            let (rec, _evt) = &cpu_record.jalr_events[0];
            println!("\n  Expected from event data:");
            println!("    op_a={}, op_b={}, op_c={}", rec.op_a, rec.op_b, rec.op_c);
            let write_rec = match rec.a {
                dt_core_executor::events::MemoryRecordEnum::Write(w) => w,
                _ => panic!("expected write"),
            };
            println!(
                "    write: value={}, prev_value={}, shard={}, ts={}, prev_shard={}, prev_ts={}",
                write_rec.value,
                write_rec.prev_value,
                write_rec.shard,
                write_rec.timestamp,
                write_rec.prev_shard,
                write_rec.prev_timestamp
            );

            // Print raw col values for comparison
            println!("\n  Raw col[0..20]:");
            for i in 0..20 {
                println!("    col[{:2}] = {}", i, row0[i].as_canonical_u32());
            }
            break;
        }
    }

    // --- Full lookup balance check ---
    println!("\n=== CPU Shard: Local Lookup Check ===\n");
    let cpu_result = check_all_lookups_local(&cpu_record, SKIP_CHIPS);
    print_result(&cpu_result, "CPU Shard");

    // Check deferred shards
    for (i, shard) in deferred_shards.iter().enumerate() {
        println!("\n=== Deferred Shard {i}: Local Lookup Check ===\n");
        let result = check_all_lookups_local(shard, SKIP_CHIPS);
        print_result(&result, &format!("Deferred Shard {i}"));
    }
}

fn print_result(result: &dt_core_machine::check_lookups::LookupCheckResult, label: &str) {
    println!(
        "  {} unique (kind, values) entries, {} balanced, {} mismatched",
        result.total_entries,
        result.balanced,
        result.mismatches.len()
    );

    if result.mismatches.is_empty() {
        println!("  [OK] All Local-scope lookups balanced for {label}");
    } else {
        println!("\n  [MISMATCH] {} unbalanced entries for {}:\n", result.mismatches.len(), label);

        // Show first N mismatches per kind
        for kind_name in &["Memory", "Program", "Byte", "State"] {
            let kind_mismatches: Vec<_> = result
                .mismatches
                .iter()
                .filter(|e| format!("{:?}", e.kind) == *kind_name)
                .collect();
            if kind_mismatches.is_empty() {
                continue;
            }
            println!("  --- {} ({} mismatches) ---", kind_name, kind_mismatches.len());
            let max_show = if *kind_name == "Memory" { 100 } else { 10 };
            for (i, entry) in kind_mismatches.iter().enumerate().take(max_show) {
                let direction =
                    if entry.net_mult > 0 { "send > receive" } else { "receive > send" };
                println!(
                    "    #{}: values={:?}, net_mult={} ({})",
                    i + 1,
                    entry.values,
                    entry.net_mult,
                    direction
                );
                for src in &entry.sources {
                    let dir = if src.is_send { "send" } else { "recv" };
                    println!(
                        "         {} chip={}, row={}, mult={}",
                        dir, src.chip_name, src.row_index, src.mult
                    );
                }
            }
            if kind_mismatches.len() > max_show {
                println!("    ... and {} more", kind_mismatches.len() - max_show);
            }
            println!();
        }

        // Summary by kind
        println!("  Summary by InteractionKind:");
        let mut by_kind: std::collections::BTreeMap<String, (usize, i64)> =
            std::collections::BTreeMap::new();
        for entry in &result.mismatches {
            let kind_str = format!("{:?}", entry.kind);
            let e = by_kind.entry(kind_str).or_insert((0, 0));
            e.0 += 1;
            e.1 += entry.net_mult;
        }
        for (kind, (count, net)) in &by_kind {
            println!("    {kind}: {count} mismatches, total net_mult = {net}");
        }
    }
}
