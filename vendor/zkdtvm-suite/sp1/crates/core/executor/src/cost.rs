use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_baby_bear::BabyBear;

use crate::{syscalls::SyscallCode, RiscvAirId};

const BYTE_NUM_ROWS: u64 = 1 << 16;
const MAX_PROGRAM_SIZE: u64 = 1 << 22;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_riscv_lde_size(
    num_events_per_air: EnumMap<RiscvAirId, u64>,
    costs_per_air: &HashMap<RiscvAirId, u64>,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&RiscvAirId::Byte];

    // Compute the program chip contribution.
    cells += MAX_PROGRAM_SIZE * costs_per_air[&RiscvAirId::Program];

    // Compute the add chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Add]).next_power_of_two() * costs_per_air[&RiscvAirId::Add];

    // Compute the addi chip contribution.
    cells += (num_events_per_air[RiscvAirId::Addi]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Addi];

    // Compute the sub chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Sub]).next_power_of_two() * costs_per_air[&RiscvAirId::Sub];

    // Compute the mul chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Mul]).next_power_of_two() * costs_per_air[&RiscvAirId::Mul];

    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[RiscvAirId::Bitwise]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Bitwise];

    // Compute the shift left chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftLeft]).next_power_of_two() *
        costs_per_air[&RiscvAirId::ShiftLeft];

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftRight]).next_power_of_two() *
        costs_per_air[&RiscvAirId::ShiftRight];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[RiscvAirId::DivRem]).next_power_of_two() *
        costs_per_air[&RiscvAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Lt]).next_power_of_two() * costs_per_air[&RiscvAirId::Lt];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[RiscvAirId::MemoryLocal]).next_power_of_two() *
        costs_per_air[&RiscvAirId::MemoryLocal];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[RiscvAirId::Branch]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Branch];

    // Compute the jal chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Jal]).next_power_of_two() * costs_per_air[&RiscvAirId::Jal];

    // Compute the jalr chip contribution.
    cells += (num_events_per_air[RiscvAirId::Jalr]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Jalr];

    // Compute the auipc chip contribution.
    cells += (num_events_per_air[RiscvAirId::Auipc]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Auipc];

    // Compute the load byte chip contribution.
    cells += (num_events_per_air[RiscvAirId::LoadByte]).next_power_of_two() *
        costs_per_air[&RiscvAirId::LoadByte];

    // Compute the load half chip contribution.
    cells += (num_events_per_air[RiscvAirId::LoadHalf]).next_power_of_two() *
        costs_per_air[&RiscvAirId::LoadHalf];

    // Compute the load word chip contribution.
    cells += (num_events_per_air[RiscvAirId::LoadWord]).next_power_of_two() *
        costs_per_air[&RiscvAirId::LoadWord];

    // Compute the store byte chip contribution.
    cells += (num_events_per_air[RiscvAirId::StoreByte]).next_power_of_two() *
        costs_per_air[&RiscvAirId::StoreByte];

    // Compute the store half chip contribution.
    cells += (num_events_per_air[RiscvAirId::StoreHalf]).next_power_of_two() *
        costs_per_air[&RiscvAirId::StoreHalf];

    // Compute the store word chip contribution.
    cells += (num_events_per_air[RiscvAirId::StoreWord]).next_power_of_two() *
        costs_per_air[&RiscvAirId::StoreWord];

    // Compute the syscall instruction chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallInstrs]).next_power_of_two() *
        costs_per_air[&RiscvAirId::SyscallInstrs];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallCore]).next_power_of_two() *
        costs_per_air[&RiscvAirId::SyscallCore];

    // Compute the global chip contribution.
    cells += (num_events_per_air[RiscvAirId::Global]).next_power_of_two() *
        costs_per_air[&RiscvAirId::Global];

    cells * ((core::mem::size_of::<BabyBear>() << 1) as u64)
}

/// Pads the event counts to account for the worst case jump in events across N cycles.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_rv32im_event_counts(
    mut event_counts: EnumMap<RiscvAirId, u64>,
    num_cycles: u64,
) -> EnumMap<RiscvAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        RiscvAirId::Add => *v += num_cycles,
        RiscvAirId::Addi => *v += num_cycles,
        RiscvAirId::Sub => *v += num_cycles,
        RiscvAirId::Mul => *v += 4 * num_cycles,
        RiscvAirId::Bitwise => *v += 3 * num_cycles,
        RiscvAirId::ShiftLeft => *v += num_cycles,
        RiscvAirId::ShiftRight => *v += num_cycles,
        RiscvAirId::DivRem => *v += 4 * num_cycles,
        RiscvAirId::Lt => *v += 2 * num_cycles,
        RiscvAirId::LoadByte => *v += num_cycles,
        RiscvAirId::LoadHalf => *v += num_cycles,
        RiscvAirId::LoadWord => *v += num_cycles,
        RiscvAirId::StoreByte => *v += num_cycles,
        RiscvAirId::StoreHalf => *v += num_cycles,
        RiscvAirId::StoreWord => *v += num_cycles,
        RiscvAirId::Auipc => *v += 3 * num_cycles,
        RiscvAirId::Branch => *v += 8 * num_cycles,
        RiscvAirId::Jal => *v += num_cycles,
        RiscvAirId::Jalr => *v += num_cycles,
        RiscvAirId::MemoryLocal => *v += 64 * num_cycles,
        RiscvAirId::SyscallInstrs => *v += num_cycles,
        RiscvAirId::SyscallCore => *v += 2 * num_cycles,
        RiscvAirId::Global => *v += 64 * num_cycles,
        _ => (),
    });
    event_counts
}

/// Pads the event counts for the fine-grained chip architecture without dependency generation.
///
/// ## Background
///
/// In the original architecture, executing a single RISC-V instruction can trigger events in
/// multiple chips due to **dependency generation** (see `dependencies.rs`). For example, a
/// single `DIV` instruction generates 2 `mul_events` and up to 1 `lt_event` as dependencies.
/// The `pad_rv32im_event_counts` function uses inflated coefficients to account for these
/// cross-chip dependencies.
///
/// In the **no-dependencies** architecture, each chip is self-contained: a single instruction
/// produces exactly **1 event** in its own chip and **0 dependency events** in other chips.
/// Therefore, the worst-case events per cycle for each chip is simply 1 (since at most 1
/// instruction executes per CPU cycle).
///
/// ## Coefficient Rationale
///
/// - **All instruction chips (Add, Addi, Sub, Mul, Bitwise, `ShiftLeft`, `ShiftRight`, `DivRem`,
///   Lt, `LoadByte`, `LoadHalf`, `LoadWord`, `StoreByte`, `StoreHalf`, `StoreWord`, Auipc, Branch,
///   Jal, Jalr, `SyscallInstrs`)**: coefficient = **1**. Each CPU cycle executes at most 1
///   instruction, which produces at most 1 event in the corresponding chip.
///
/// - **`SyscallCore`**: coefficient = **1**. Each ECALL instruction sends at most 1 syscall to the
///   core syscall handler.
///
/// - **`MemoryLocal`**: coefficient = **4**. Each instruction accesses at most 4 memory operands
///   (`op_a`, `op_b`, `op_c`, plus potential memory load/store), contributing to local memory
///   tracking rows.
///
/// - **Global**: coefficient = **2 * `MemoryLocal_coefficient` + 1 = 9**, rounded up to **10** for
///   safety. Each local memory access generates 2 global interaction events (read + write), plus
///   the instruction itself may generate 1 global event.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_rv32im_event_counts_no_dependencies(
    mut event_counts: EnumMap<RiscvAirId, u64>,
    num_cycles: u64,
) -> EnumMap<RiscvAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        RiscvAirId::Add |
        RiscvAirId::Addi |
        RiscvAirId::Sub |
        RiscvAirId::Mul |
        RiscvAirId::Bitwise |
        RiscvAirId::ShiftLeft |
        RiscvAirId::ShiftRight |
        RiscvAirId::DivRem |
        RiscvAirId::Lt |
        RiscvAirId::LoadByte |
        RiscvAirId::LoadHalf |
        RiscvAirId::LoadWord |
        RiscvAirId::StoreByte |
        RiscvAirId::StoreHalf |
        RiscvAirId::StoreWord |
        RiscvAirId::Auipc |
        RiscvAirId::Branch |
        RiscvAirId::Jal |
        RiscvAirId::Jalr |
        RiscvAirId::SyscallInstrs |
        RiscvAirId::SyscallCore => *v += num_cycles,

        RiscvAirId::MemoryLocal => *v += 4 * num_cycles,

        RiscvAirId::Global => *v += 10 * num_cycles,

        _ => (),
    });
    event_counts
}

/// The result of estimating the cost of a RISC-V execution record.
#[derive(Debug, Clone, Copy)]
pub struct RiscvCostEstimate {
    /// The total number of cells consumed (sum of `padded_height` * width for each chip).
    pub total_cells: u64,
    /// The maximum padded height across all chips.
    pub max_height: u64,
}

/// Estimates the total cells and max height for the new fine-grained chip architecture.
///
/// For each chip, the padded height is `next_power_of_two(num_events)` and the cost per row
/// is given by `costs_per_air`. The function returns both the total cells (sum of
/// `padded_height * cost_per_row` for all chips) and the maximum padded height.
///
/// This function covers the new split chips (Add, Addi, Sub, `LoadByte`, `LoadHalf`, `LoadWord`,
/// `StoreByte`, `StoreHalf`, `StoreWord`, Jal, Jalr) as well as the unchanged chips.
#[must_use]
pub fn estimate_riscv_cost(
    num_events_per_air: EnumMap<RiscvAirId, u64>,
    costs_per_air: &HashMap<RiscvAirId, u64>,
    program_size: u32,
) -> RiscvCostEstimate {
    let mut total_cells: u64 = 0;
    let mut max_height: u64 = 0;

    // Fixed-size chips: Byte and Program.
    if let Some(&cost) = costs_per_air.get(&RiscvAirId::Byte) {
        total_cells += BYTE_NUM_ROWS * cost;
        max_height = max_height.max(BYTE_NUM_ROWS);
    }
    if let Some(&cost) = costs_per_air.get(&RiscvAirId::Program) {
        let program_rows = (program_size as u64).next_power_of_two();
        total_cells += program_rows * cost;
        max_height = max_height.max(program_rows);
    }

    // All dynamic chips: compute padded_height = next_power_of_two(num_events),
    // then accumulate cells and track max height.
    let dynamic_chips = [
        // Fine-grained ALU chips.
        RiscvAirId::Add,
        RiscvAirId::Addi,
        RiscvAirId::Sub,
        // Unchanged ALU chips.
        RiscvAirId::Mul,
        RiscvAirId::Bitwise,
        RiscvAirId::ShiftLeft,
        RiscvAirId::ShiftRight,
        RiscvAirId::DivRem,
        RiscvAirId::Lt,
        // New fine-grained memory chips.
        RiscvAirId::LoadByte,
        RiscvAirId::LoadHalf,
        RiscvAirId::LoadWord,
        RiscvAirId::StoreByte,
        RiscvAirId::StoreHalf,
        RiscvAirId::StoreWord,
        // Control flow.
        RiscvAirId::Auipc,
        RiscvAirId::Branch,
        // New fine-grained jump chips.
        RiscvAirId::Jal,
        RiscvAirId::Jalr,
        // Memory and syscall.
        RiscvAirId::MemoryLocal,
        RiscvAirId::SyscallInstrs,
        RiscvAirId::SyscallCore,
        RiscvAirId::Global,
    ];

    for chip_id in dynamic_chips {
        if let Some(&cost) = costs_per_air.get(&chip_id) {
            let padded_height = num_events_per_air[chip_id].next_power_of_two();
            total_cells += padded_height * cost;
            max_height = max_height.max(padded_height);
        }
    }

    RiscvCostEstimate { total_cells, max_height }
}

/// Number of `MemoryLocal` entries packed per trace row.
/// After the 4→1 refactor, each trace row holds exactly 1 entry.
const MEM_LOCAL_ENTRIES_PER_ROW: usize = 1;

/// Returns the estimated total cells and max height that a single precompile event would
/// contribute to its precompile shard.
///
/// A precompile shard contains:
/// 1. The precompile chip itself
/// 2. `SyscallPrecompile` chip (1 row per event)
/// 3. `MemoryLocal` chip (1 row per local memory event)
/// 4. Global chip (≈ 2 * `local_mem_events` + 1 rows per event)
/// 5. Byte chip (fixed 2^16 — not counted here as it's constant)
/// 6. Program chip (fixed — not counted here as it's constant)
///
/// The returned tuple is `(cells_per_event, max_height_per_event)`:
/// - `cells_per_event`: the total variable cells consumed by one event across all chips in the
///   shard (not including fixed Byte/Program overhead).
/// - `max_height_per_event`: the maximum per-event row contribution across all chips.
///
/// These values are used by `compute_syscall_thresholds` to determine how many events of
/// this syscall can fit within a shard's cells/height budget. The caller should subtract
/// fixed overhead (Byte + Program) from the element threshold before dividing.
#[must_use]
pub fn cost_and_height_per_syscall(
    syscall_code: SyscallCode,
    costs: &HashMap<RiscvAirId, usize>,
) -> (usize, usize) {
    let Some(air_id) = syscall_code.as_air_id() else {
        return (0, 0); // Not a precompile syscall.
    };

    let rows_per_event = air_id.rows_per_event();
    let local_mem_per_event = air_id.local_mem_events_per_event();

    // 1. Precompile chip: rows_per_event rows per event.
    let precompile_width = costs.get(&air_id).copied().unwrap_or(0);
    let precompile_cells = rows_per_event * precompile_width;
    let precompile_height = rows_per_event;

    // 2. Dispatch chip: 1 row per event.
    let dispatch_id = precompile_dispatch_id(air_id);
    let syscall_precompile_width = costs.get(&dispatch_id).copied().unwrap_or(0);
    let syscall_precompile_cells = syscall_precompile_width;
    let syscall_precompile_height = 1;

    // 3. MemoryLocal chip: 1 row per local memory event (1 entry per row).
    let memory_local_width = costs.get(&RiscvAirId::MemoryLocal).copied().unwrap_or(0);
    let memory_local_rows = local_mem_per_event.div_ceil(MEM_LOCAL_ENTRIES_PER_ROW);
    let memory_local_cells = memory_local_rows * memory_local_width;
    let memory_local_height = memory_local_rows;

    // 4. Global chip: each local memory event generates 2 global interactions (read timestamp +
    //    write timestamp), plus 1 for the precompile syscall itself.
    let global_width = costs.get(&RiscvAirId::Global).copied().unwrap_or(0);
    let global_rows = 2 * local_mem_per_event + 1;
    let global_cells = global_rows * global_width;
    let global_height = global_rows;

    // Total cells and max height per event.
    let total_cells =
        precompile_cells + syscall_precompile_cells + memory_local_cells + global_cells;
    let max_height = precompile_height
        .max(syscall_precompile_height)
        .max(memory_local_height)
        .max(global_height);

    (total_cells, max_height)
}

/// Returns the dispatch chip `RiscvAirId` for a given precompile AIR.
///
/// SHA/Keccak precompiles use dedicated controller chips; all others use the
/// generic `SyscallPrecompile` dispatch chip.
#[must_use]
pub fn precompile_dispatch_id(air_id: RiscvAirId) -> RiscvAirId {
    match air_id {
        RiscvAirId::ShaExtend => RiscvAirId::ShaExtendController,
        RiscvAirId::ShaCompress => RiscvAirId::ShaCompressController,
        RiscvAirId::KeccakPermute => RiscvAirId::KeccakController,
        _ => RiscvAirId::SyscallPrecompile,
    }
}

/// Estimates the total **padded** cells for `n` events of a precompile in a shard.
///
/// Unlike `cost_and_height_per_syscall` which returns raw (unpadded) per-event costs,
/// this function computes the actual padded trace area by rounding each chip's height
/// to the next power of two. Used for memory usage estimation, not for limit computation.
///
/// The shard contains:
/// 1. Precompile chip: `next_power_of_two(n * rows_per_event) * width`
/// 2. Dispatch chip: `next_power_of_two(n) * width`
/// 3. MemoryLocal chip: `next_power_of_two(n * local_mem_per_event) * width`
/// 4. Global chip: `next_power_of_two(n * (2 * local_mem_per_event + 1)) * width`
/// 5. Byte chip: `2^16 * width` (fixed)
/// 6. Program chip: `2^14 * width` (conservative estimate)
#[must_use]
pub fn precompile_shard_padded_cells(
    syscall_code: SyscallCode,
    n: usize,
    costs: &HashMap<RiscvAirId, usize>,
) -> usize {
    let Some(air_id) = syscall_code.as_air_id() else {
        return 0;
    };

    let rows_per_event = air_id.rows_per_event();
    let local_mem_per_event = air_id.local_mem_events_per_event();

    let precompile_width = costs.get(&air_id).copied().unwrap_or(0);
    let precompile_cells = (n * rows_per_event).next_power_of_two() * precompile_width;

    let dispatch_id = precompile_dispatch_id(air_id);
    let dispatch_width = costs.get(&dispatch_id).copied().unwrap_or(0);
    let dispatch_cells = n.next_power_of_two() * dispatch_width;

    let mem_local_width = costs.get(&RiscvAirId::MemoryLocal).copied().unwrap_or(0);
    let mem_local_rows = n * local_mem_per_event.div_ceil(MEM_LOCAL_ENTRIES_PER_ROW);
    let mem_local_cells = mem_local_rows.next_power_of_two() * mem_local_width;

    let global_width = costs.get(&RiscvAirId::Global).copied().unwrap_or(0);
    let global_rows = n * (2 * local_mem_per_event + 1);
    let global_cells = global_rows.next_power_of_two() * global_width;

    let byte_width = costs.get(&RiscvAirId::Byte).copied().unwrap_or(0);
    let byte_cells = (1usize << 16) * byte_width;

    let program_width = costs.get(&RiscvAirId::Program).copied().unwrap_or(0);
    let program_cells = (1usize << 14) * program_width;

    precompile_cells + dispatch_cells + mem_local_cells + global_cells + byte_cells + program_cells
}
