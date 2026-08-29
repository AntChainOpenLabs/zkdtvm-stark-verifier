#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::mem::size_of;

use dt_core_machine::riscv::RiscvAir;
use dt_primitives::SCField;
use dt_stark::air::MachineAir;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;

fn main() {
    let (chips, _costs) = RiscvAir::<SCField>::get_chips_and_costs();

    println!(
        "{:30} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "Chip", "pre", "main", "perm", "total", "cost()"
    );
    println!("{}", "-".repeat(80));

    for chip in &chips {
        let name = MachineAir::<SCField>::name(chip);
        let preprocessed = MachineAir::<SCField>::preprocessed_width(chip);
        let main = <_ as BaseAir<SCField>>::width(chip);
        let perm = chip.permutation_width() * 4;
        let cost = chip.cost();
        let total = preprocessed + main + perm;
        println!("{name:30} {preprocessed:>5} {main:>5} {perm:>5} {total:>5} {cost:>5}");
    }

    println!("\n=== Column size analysis for key structs ===");

    type F = BabyBear;

    use dt_core_machine::memory::MemoryAccessCols;
    println!("MemoryAccessCols<F>:   {} fields", size_of::<MemoryAccessCols<F>>() / size_of::<F>());

    use dt_core_machine::memory::MemoryReadCols;
    println!("MemoryReadCols<F>:     {} fields", size_of::<MemoryReadCols<F>>() / size_of::<F>());

    use dt_core_machine::memory::MemoryReadWriteCols;
    println!(
        "MemoryReadWriteCols<F>:{} fields",
        size_of::<MemoryReadWriteCols<F>>() / size_of::<F>()
    );

    use dt_core_machine::adapter::CPUState;
    println!("CPUState<F>:           {} fields", size_of::<CPUState<F>>() / size_of::<F>());

    use dt_core_machine::adapter::RTypeRegisterOp;
    println!("RTypeRegisterOp<F>:    {} fields", size_of::<RTypeRegisterOp<F>>() / size_of::<F>());

    println!("\n=== Breakdown of RTypeRegisterOp ===");
    println!("  op_a:            1");
    println!(
        "  op_a_access (MemoryReadWriteCols): {}",
        size_of::<MemoryReadWriteCols<F>>() / size_of::<F>()
    );
    println!("  op_a_zero:       1");
    println!("  op_b:            1");
    println!(
        "  op_b_access (MemoryReadCols):     {}",
        size_of::<MemoryReadCols<F>>() / size_of::<F>()
    );
    println!("  op_c:            1");
    println!(
        "  op_c_access (MemoryReadCols):     {}",
        size_of::<MemoryReadCols<F>>() / size_of::<F>()
    );

    println!("\n=== Breakdown of MemoryAccessCols ===");
    println!("  value (Word):    4");
    println!("  prev_shard:      1");
    println!("  prev_clk:        1");
    println!("  compare_clk:     1");
    println!("  diff_16bit_limb: 1");
    println!("  diff_12bit_limb: 1");
    println!("  TOTAL:           9");

    println!("\n=== Preprocessed vs Main analysis ===");
    println!("Chips with preprocessed columns get their preprocessed width counted in cost().");
    println!("This is correct because preprocessed trace IS materialized per shard.");
    println!("Only Byte and Program have preprocessed columns.");
    println!("All instruction chips have preprocessed_width = 0.");

    println!("\n=== Double-counting check in estimate_riscv_cost ===");
    println!("estimate_riscv_cost computes: sum(padded_height * cost) for each chip");
    println!("where cost = preprocessed + main + permutation_width*4");
    println!(
        "This is compared to element_threshold = (1<<28) + (1<<27) = {}",
        (1u64 << 28) + (1u64 << 27)
    );
    println!();
    println!("Potential issue: Byte chip and Program chip's preprocessed trace is committed once");
    println!("for the entire program, BUT the cost is counted per-shard. If the element_threshold");
    println!("is meant to represent per-shard memory usage, this is CORRECT because each shard");
    println!("needs to re-commit the preprocessed trace.");
    println!();
    println!(
        "However, the preprocessed trace for Byte and Program is FIXED (same for all shards)."
    );
    println!("The prover could cache or share the preprocessed commitment across shards.");
    println!("In that case, counting preprocessed columns as part of the per-shard cost would be");
    println!("an OVERESTIMATE.");
}
