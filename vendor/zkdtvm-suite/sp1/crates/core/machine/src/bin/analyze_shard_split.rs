use dt_core_executor::{Executor, Program};
use dt_stark::{
    DTCoreOpts, ShardingThreshold, SplitOpts, SHARD_CELLS_THRESHOLD, SHARD_HEIGHT_THRESHOLD,
};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive("info".parse().unwrap())
                .from_env_lossy(),
        )
        .init();

    let n: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);

    let elf_path = std::env::args().nth(2).unwrap_or_else(|| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../examples/target/elf-compilation/riscv32im-succinct-zkvm-elf/release/fibonacci-program")
    });

    println!("=== Shard Split Analysis: Fibonacci(n={n}) ===\n");

    let elf_bytes = std::fs::read(&elf_path)
        .unwrap_or_else(|e| panic!("Failed to read ELF at {elf_path}: {e}"));
    let program = Program::from(&elf_bytes).unwrap();
    println!("Program: {} instructions in ELF", program.instructions.len());

    let opts = DTCoreOpts {
        shard_size: 1 << 24,
        shard_batch_size: 1,
        sharding_threshold: ShardingThreshold {
            element_threshold: SHARD_CELLS_THRESHOLD,
            height_threshold: SHARD_HEIGHT_THRESHOLD,
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

    println!(
        "height_threshold  = {} (2^{})",
        opts.sharding_threshold.height_threshold,
        (opts.sharding_threshold.height_threshold as f64).log2() as u32
    );
    println!(
        "element_threshold = {} ({:.1}M cells, = 2^28 + 2^27)",
        opts.sharding_threshold.element_threshold,
        opts.sharding_threshold.element_threshold as f64 / 1e6
    );
    println!("shard_size (hard) = {} cycles\n", opts.shard_size * 4);

    let mut executor = Executor::new(program, opts);
    executor.write_stdin(&n);
    executor.run().unwrap();

    println!("\n{}", "=".repeat(80));
    println!("EXECUTION COMPLETE");
    println!("{}", "=".repeat(80));

    let num_shards = executor.records.len();
    let mut total_instructions = 0u64;

    for (i, record) in executor.records.iter().enumerate() {
        let cpu_events = record.cpu_events as u64;
        let local_mem = record.cpu_local_memory_access.len();

        let chips: Vec<(&str, usize)> = vec![
            ("Add", record.add_events.len()),
            ("Addi", record.addi_events.len()),
            ("Sub", record.sub_events.len()),
            ("Mul", record.mul_events.len()),
            ("Bitwise", record.bitwise_events.len()),
            ("ShiftLeft", record.shift_left_events.len()),
            ("ShiftRight", record.shift_right_events.len()),
            ("DivRem", record.divrem_events.len()),
            ("Lt", record.lt_events.len()),
            ("Branch", record.branch_events.len()),
            ("Jal", record.jal_events.len()),
            ("Jalr", record.jalr_events.len()),
            ("Auipc", record.auipc_events.len()),
            ("LoadByte", record.load_byte_events.len()),
            ("LoadHalf", record.load_half_events.len()),
            ("LoadWord", record.load_word_events.len()),
            ("StoreByte", record.store_byte_events.len()),
            ("StoreHalf", record.store_half_events.len()),
            ("StoreWord", record.store_word_events.len()),
            ("Syscall", record.syscall_events.len()),
        ];

        let mut sorted_chips = chips.clone();
        sorted_chips.sort_by(|a, b| b.1.cmp(&a.1));

        // Find max padded height (MemoryLocal: 1 entry per row)
        let max_padded: u64 = sorted_chips
            .iter()
            .map(|(_, c)| (*c as u64).next_power_of_two())
            .max()
            .unwrap_or(1)
            .max((local_mem as u64).next_power_of_two())
            .max((2 * local_mem as u64).next_power_of_two());

        println!("\n--- Shard {i} ---");
        println!("  instructions:      {cpu_events:>10}");
        println!("  unique addresses:  {local_mem:>10} (MemoryLocal events)");
        println!("  max padded height: {max_padded:>10}");

        println!("  top chips:");
        for (name, count) in sorted_chips.iter().take(8) {
            if *count > 0 {
                println!(
                    "    {:20} {:>10} -> padded {:>10}",
                    name,
                    count,
                    (*count as u64).next_power_of_two()
                );
            }
        }
        println!(
            "    {:20} {:>10} -> padded {:>10}",
            "MemoryLocal(rows)",
            local_mem,
            (local_mem as u64).next_power_of_two()
        );
        println!(
            "    {:20} {:>10} -> padded {:>10}",
            "Global(rows)",
            2 * local_mem,
            (2 * local_mem as u64).next_power_of_two()
        );

        total_instructions += cpu_events;
    }

    println!("\n{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));
    println!("  Total instructions:       {total_instructions:>10}");
    println!("  Total execution shards:   {num_shards:>10}");
    if num_shards > 0 {
        println!("  Avg instructions/shard:   {:>10}", total_instructions / num_shards as u64);
    }
    if num_shards > 1 {
        println!("  First shard instructions: {:>10}", executor.records[0].cpu_events);
        println!("  Last shard instructions:  {:>10}", executor.records.last().unwrap().cpu_events);
    }
    println!("{}", "=".repeat(80));
}
