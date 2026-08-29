use dt_stark::{PRECOMPILE_SHARD_CELLS_THRESHOLD, SHARD_HEIGHT_THRESHOLD};
use hashbrown::HashMap;
use strum::IntoEnumIterator;

use crate::{cost::cost_and_height_per_syscall, syscalls::SyscallCode, RiscvAirId};

/// Fixed shard overhead from Byte chip (2^16 rows × width).
const BYTE_FIXED_ROWS: usize = 1 << 16;

/// Conservative program chip overhead estimate (2^14 rows × width).
const PROGRAM_FIXED_ROWS: usize = 1 << 14;

/// Recomputes per-precompile limits using raw-cells formula.
///
/// This is a **maintenance tool** for verifying that the static limits in
/// `RiscvAirId::max_precompile_shard_events()` match the dynamic calculation.
/// It is NOT used in the hot path; record splitting uses the static limits
/// returned by `SyscallCode::precompile_shard_limit()`.
///
/// Formula per precompile:
/// ```text
/// area_limit   = (PRECOMPILE_SHARD_CELLS_THRESHOLD - byte_overhead - program_overhead)
///                / raw_cells_per_event
/// height_limit = SHARD_HEIGHT_THRESHOLD / max_height_per_event
/// limit        = min(area_limit, height_limit)
/// ```
#[must_use]
pub fn compute_precompile_limits(costs: &HashMap<RiscvAirId, usize>) -> HashMap<RiscvAirId, usize> {
    let budget = PRECOMPILE_SHARD_CELLS_THRESHOLD as usize;
    let height_threshold = SHARD_HEIGHT_THRESHOLD as usize;

    let byte_width = costs.get(&RiscvAirId::Byte).copied().unwrap_or(0);
    let byte_overhead = BYTE_FIXED_ROWS * byte_width;

    let program_width = costs.get(&RiscvAirId::Program).copied().unwrap_or(0);
    let program_overhead = PROGRAM_FIXED_ROWS * program_width;

    let effective_budget = budget.saturating_sub(byte_overhead + program_overhead);

    let mut limits = HashMap::new();

    for syscall_code in SyscallCode::iter() {
        let Some(air_id) = syscall_code.as_air_id() else { continue };

        let (cells_per_event, max_height_per_event) =
            cost_and_height_per_syscall(syscall_code, costs);

        if cells_per_event == 0 || max_height_per_event == 0 {
            continue;
        }

        let area_limit = effective_budget / cells_per_event;
        let height_limit = height_threshold / max_height_per_event;

        limits.insert(air_id, area_limit.min(height_limit).max(1));
    }

    limits
}

/// Verifies that the static limits in `RiscvAirId::max_precompile_shard_events()`
/// match the dynamically computed limits from `compute_precompile_limits`.
///
/// Panics with a descriptive message if any mismatch is found.
pub fn verify_static_precompile_limits(costs: &HashMap<RiscvAirId, usize>) {
    let dynamic = compute_precompile_limits(costs);

    for (air_id, dynamic_limit) in &dynamic {
        let static_limit = air_id.max_precompile_shard_events();
        assert_eq!(
            static_limit, *dynamic_limit,
            "Static limit mismatch for {air_id:?}: static={static_limit}, dynamic={dynamic_limit}",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rv32im_costs;

    #[test]
    fn test_static_precompile_limits_match() {
        let costs = rv32im_costs();
        verify_static_precompile_limits(&costs);
    }
}
