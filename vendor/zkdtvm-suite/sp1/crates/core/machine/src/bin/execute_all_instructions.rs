//! Execute the all-instructions ELF, print execution statistics,
//! and serialize the ExecutionRecord to JSON.
//!
//! Usage: cargo run --release --bin execute_all_instructions -p dt-core-machine

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fs;

use dt_core_executor::{Executor, Program};
use dt_core_test_elf::ALL_INSTRUCTIONS_ELF;
use dt_stark::{DTCoreOpts, MachineRecord, ShardingThreshold, SplitOpts};

fn main() {
    let program = Program::from(ALL_INSTRUCTIONS_ELF).unwrap();
    println!("Program loaded: {} instructions", program.instructions.len());

    // Construct opts manually to avoid the 20 GB memory check in Default::default().
    let opts = DTCoreOpts {
        shard_size: 1 << 22,
        shard_batch_size: 1,
        sharding_threshold: ShardingThreshold {
            element_threshold: 1 << 28,
            height_threshold: 1 << 22,
        },
        split_opts: SplitOpts {
            combine_memory_threshold: 1 << 24,
            deferred: 1 << 15,
            keccak: 8 * (1 << 15) / 24,
            sha_extend: 32 * (1 << 15) / 48,
            sha_compress: 32 * (1 << 15) / 64,
            memory: 1 << 24,
        },
        trace_gen_workers: 1,
        checkpoints_channel_capacity: 128,
        records_and_traces_channel_capacity: 1,
    };

    let mut executor = Executor::new(program, opts);
    executor.run().unwrap();

    println!("\n=== Execution Results ===");
    println!("Number of shards: {}", executor.records.len());

    for (i, record) in executor.records.iter().enumerate() {
        println!("\n--- Shard {i} ---");

        let mut stats: Vec<_> = record.stats().into_iter().collect();
        stats.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, count) in &stats {
            println!("  {name:<45} {count}");
        }

        println!("\n  Total event types with non-zero count: {}", stats.len());
    }

    // Serialize all records to JSON.
    let output_path = "execution_record.json";
    println!("\nSerializing {} shard(s) to {} ...", executor.records.len(), output_path);

    let json =
        serde_json::to_string(&executor.records).expect("Failed to serialize ExecutionRecord");
    fs::write(output_path, &json).expect("Failed to write JSON file");

    let size_mb = json.len() as f64 / (1024.0 * 1024.0);
    println!("Done. Written {size_mb:.2} MB to {output_path}");
}
