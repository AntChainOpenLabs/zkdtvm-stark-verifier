use hashbrown::HashMap;

use dt_core_executor::{events::PrecompileLocalMemory, ExecutionRecord, RiscvAirId};
use dt_stark::MachineRecord;

#[derive(Debug, Clone, Copy)]
pub enum ShardKind {
    PackedCore,
    Core,
    GlobalMemory,
    Precompile,
}

pub trait Shapeable {
    fn kind(&self) -> ShardKind;
    fn shard(&self) -> u32;
    fn log2_shard_size(&self) -> usize;
    fn debug_stats(&self) -> HashMap<String, usize>;
    fn core_heights(&self) -> Vec<(RiscvAirId, usize)>;
    fn memory_heights(&self) -> Vec<(RiscvAirId, usize)>;
    /// TODO. Returns all precompile events, assuming there is only one kind in `Self`.
    /// The tuple is of the form `(height, (num_memory_local_events, num_global_events))`
    fn precompile_heights(&self) -> impl Iterator<Item = (RiscvAirId, (usize, usize, usize))>;
}

impl Shapeable for ExecutionRecord {
    fn kind(&self) -> ShardKind {
        let contains_global_memory = !self.global_memory_initialize_events.is_empty() ||
            !self.global_memory_finalize_events.is_empty();
        match (self.contains_cpu(), contains_global_memory) {
            (true, true) => ShardKind::PackedCore,
            (true, false) => ShardKind::Core,
            (false, true) => ShardKind::GlobalMemory,
            (false, false) => ShardKind::Precompile,
        }
    }
    fn shard(&self) -> u32 {
        self.public_values.shard
    }

    fn log2_shard_size(&self) -> usize {
        // v6_final: no CPU chip, so "shard size" = max chip log height = 22.
        // SHARD_HEIGHT_THRESHOLD = 1 << 22 is the binding split criterion.
        // This replaces v5's cpu-event-based shard size concept.
        22
    }

    fn debug_stats(&self) -> HashMap<String, usize> {
        self.stats()
    }

    fn core_heights(&self) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::Add, self.add_events.len()),
            (RiscvAirId::Addi, self.addi_events.len()),
            (RiscvAirId::Sub, self.sub_events.len()),
            (RiscvAirId::Mul, self.mul_events.len()),
            (RiscvAirId::Bitwise, self.bitwise_events.len()),
            (RiscvAirId::ShiftLeft, self.shift_left_events.len()),
            (RiscvAirId::ShiftRight, self.shift_right_events.len()),
            (RiscvAirId::DivRem, self.divrem_events.len()),
            (RiscvAirId::Lt, self.lt_events.len()),
            (RiscvAirId::LoadByte, self.load_byte_events.len()),
            (RiscvAirId::LoadHalf, self.load_half_events.len()),
            (RiscvAirId::LoadWord, self.load_word_events.len()),
            (RiscvAirId::StoreByte, self.store_byte_events.len()),
            (RiscvAirId::StoreHalf, self.store_half_events.len()),
            (RiscvAirId::StoreWord, self.store_word_events.len()),
            (RiscvAirId::Auipc, self.auipc_events.len()),
            (RiscvAirId::Branch, self.branch_events.len()),
            (RiscvAirId::Jal, self.jal_events.len()),
            (RiscvAirId::Jalr, self.jalr_events.len()),
            (RiscvAirId::MemoryLocal, self.get_local_mem_events().count()),
            (RiscvAirId::SyscallCore, self.syscall_events.len()),
            (RiscvAirId::SyscallInstrs, self.syscall_events.len()),
            (RiscvAirId::Global, self.global_endpoint_count()),
            (
                RiscvAirId::GlobalTileReducer,
                crate::global::global_tile_reducer_real_rows(self.global_endpoint_count()),
            ),
        ]
    }

    fn memory_heights(&self) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::MemoryGlobalInit, self.global_memory_initialize_events.len()),
            (RiscvAirId::MemoryGlobalFinalize, self.global_memory_finalize_events.len()),
            (RiscvAirId::Global, self.global_endpoint_count()),
            (
                RiscvAirId::GlobalTileReducer,
                crate::global::global_tile_reducer_real_rows(self.global_endpoint_count()),
            ),
        ]
    }

    fn precompile_heights(&self) -> impl Iterator<Item = (RiscvAirId, (usize, usize, usize))> {
        self.precompile_events.events.iter().filter_map(|(code, events)| {
            // Skip empty events.
            (!events.is_empty()).then_some(())?;
            let id = code.as_air_id()?;
            Some((
                id,
                (
                    events.len() * id.rows_per_event(),
                    events.get_local_mem_events().into_iter().count(),
                    self.global_endpoint_count(),
                ),
            ))
        })
    }
}
