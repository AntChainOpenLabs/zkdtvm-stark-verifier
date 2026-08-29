use std::ops::Range;

use dt_core_executor::{
    events::{
        GlobalInteractionEvent, GlobalSourceId, MemoryInitializeFinalizeEvent, MemoryLocalEvent,
    },
    syscalls::SyscallCode,
    ExecutionRecord,
};
use dt_stark::InteractionKind;

/// One statically dispatched batch in the production Global source schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalProducerBatch {
    /// Stable source identity.
    pub source_id: GlobalSourceId,
    /// Stable proof-order ordinal.
    pub ordinal: u16,
    /// Stable diagnostic label.
    pub name: &'static str,
}

impl GlobalProducerBatch {
    /// Count the logical endpoints emitted by this batch.
    #[must_use]
    pub fn endpoint_count(self, record: &ExecutionRecord) -> usize {
        record.global_source_endpoint_count(self.source_id)
    }

    /// Visit this batch's endpoints in exact production order.
    pub fn visit_endpoints(
        self,
        record: &ExecutionRecord,
        mut visit: impl FnMut(GlobalInteractionEvent),
    ) {
        match self.source_id {
            GlobalSourceId::CoreSyscall => {
                for (_, event) in record
                    .syscall_events
                    .iter()
                    .filter(|(_, event)| event.syscall_code.should_send() == 1)
                {
                    visit(GlobalInteractionEvent {
                        message: [
                            event.shard,
                            event.clk,
                            event.syscall_id,
                            event.arg1,
                            event.arg2,
                            0,
                            0,
                        ],
                        is_receive: false,
                        kind: InteractionKind::Syscall as u8,
                    });
                }
            }
            GlobalSourceId::DeferredSyscall => {
                for (event, _) in record
                    .precompile_events
                    .all_precompile_events()
                    .filter(|(event, _)| event.syscall_code.should_send() == 1)
                {
                    visit(GlobalInteractionEvent {
                        message: [
                            event.shard,
                            event.clk,
                            event.syscall_id,
                            event.arg1,
                            event.arg2,
                            0,
                            0,
                        ],
                        is_receive: true,
                        kind: InteractionKind::Syscall as u8,
                    });
                }
            }
            GlobalSourceId::MemoryInitialize => visit_sorted_memory_endpoints(
                &record.global_memory_initialize_events,
                false,
                &mut visit,
            ),
            GlobalSourceId::MemoryFinalize => visit_sorted_memory_endpoints(
                &record.global_memory_finalize_events,
                true,
                &mut visit,
            ),
            GlobalSourceId::MemoryLocal => {
                for event in record.get_local_mem_events() {
                    visit(local_memory_endpoint(event, true));
                    visit(local_memory_endpoint(event, false));
                }
            }
            GlobalSourceId::ShaExtendController => {
                for event in record.precompile_events.sha_extend_events() {
                    visit(GlobalInteractionEvent {
                        message: [
                            event.shard,
                            event.clk,
                            SyscallCode::SHA_EXTEND.syscall_id(),
                            event.w_ptr,
                            0,
                            0,
                            0,
                        ],
                        is_receive: true,
                        kind: InteractionKind::Syscall as u8,
                    });
                }
            }
            GlobalSourceId::ShaCompressController => {
                for event in record.precompile_events.sha_compress_events() {
                    visit(GlobalInteractionEvent {
                        message: [
                            event.shard,
                            event.clk,
                            SyscallCode::SHA_COMPRESS.syscall_id(),
                            event.w_ptr,
                            event.h_ptr,
                            0,
                            0,
                        ],
                        is_receive: true,
                        kind: InteractionKind::Syscall as u8,
                    });
                }
            }
            GlobalSourceId::KeccakController => {
                for event in record.precompile_events.keccak_events() {
                    visit(GlobalInteractionEvent {
                        message: [
                            event.shard,
                            event.clk,
                            SyscallCode::KECCAK_PERMUTE.syscall_id(),
                            event.state_addr,
                            0,
                            0,
                            0,
                        ],
                        is_receive: true,
                        kind: InteractionKind::Syscall as u8,
                    });
                }
            }
        }
    }

    /// Visit a deterministic contiguous subrange of this source.
    ///
    /// The two memory-boundary sources and MemoryLocal may be subdivided. Other sources retain a
    /// single full-range task because their iterators are heterogeneous and comparatively small.
    pub fn visit_endpoint_range(
        self,
        record: &ExecutionRecord,
        range: Range<usize>,
        mut visit: impl FnMut(GlobalInteractionEvent),
    ) {
        match self.source_id {
            GlobalSourceId::MemoryInitialize => visit_sorted_memory_endpoints(
                &record.global_memory_initialize_events[range],
                false,
                &mut visit,
            ),
            GlobalSourceId::MemoryFinalize => visit_sorted_memory_endpoints(
                &record.global_memory_finalize_events[range],
                true,
                &mut visit,
            ),
            GlobalSourceId::MemoryLocal => {
                assert!(range.start <= range.end && range.end <= self.endpoint_count(record));
                let first_event = range.start / 2;
                let event_end = range.end.saturating_add(1) / 2;
                let mut event_index = first_event;
                record.visit_local_mem_event_range(first_event..event_end, |event| {
                    let initial_ordinal = 2 * event_index;
                    if range.contains(&initial_ordinal) {
                        visit(local_memory_endpoint(event, true));
                    }
                    if range.contains(&(initial_ordinal + 1)) {
                        visit(local_memory_endpoint(event, false));
                    }
                    event_index += 1;
                });
            }
            _ => {
                assert_eq!(range, 0..self.endpoint_count(record));
                self.visit_endpoints(record, visit);
            }
        }
    }

    /// Visit a deterministic contiguous subrange while using a once-built MemoryLocal index.
    pub fn visit_endpoint_range_indexed(
        self,
        record: &ExecutionRecord,
        local_memory_events: &[&MemoryLocalEvent],
        range: Range<usize>,
        mut visit: impl FnMut(GlobalInteractionEvent),
    ) {
        if self.source_id != GlobalSourceId::MemoryLocal {
            self.visit_endpoint_range(record, range, visit);
            return;
        }

        let endpoint_count =
            local_memory_events.len().checked_mul(2).expect("MemoryLocal endpoint count overflow");
        assert!(range.start <= range.end && range.end <= endpoint_count);
        for ordinal in range {
            visit(local_memory_endpoint(local_memory_events[ordinal / 2], ordinal % 2 == 0));
        }
    }
}

fn local_memory_endpoint(event: &MemoryLocalEvent, is_initial: bool) -> GlobalInteractionEvent {
    let access = if is_initial { event.initial_mem_access } else { event.final_mem_access };
    GlobalInteractionEvent {
        message: [
            access.shard,
            access.timestamp,
            event.addr,
            access.value & 255,
            (access.value >> 8) & 255,
            (access.value >> 16) & 255,
            (access.value >> 24) & 255,
        ],
        is_receive: is_initial,
        kind: InteractionKind::Memory as u8,
    }
}

/// The one authoritative ordered Global producer schedule.
pub const GLOBAL_PRODUCER_SCHEDULE: [GlobalProducerBatch; 8] = [
    batch(GlobalSourceId::CoreSyscall),
    batch(GlobalSourceId::DeferredSyscall),
    batch(GlobalSourceId::MemoryInitialize),
    batch(GlobalSourceId::MemoryFinalize),
    batch(GlobalSourceId::MemoryLocal),
    batch(GlobalSourceId::ShaExtendController),
    batch(GlobalSourceId::ShaCompressController),
    batch(GlobalSourceId::KeccakController),
];

const fn batch(source_id: GlobalSourceId) -> GlobalProducerBatch {
    GlobalProducerBatch { source_id, ordinal: source_id.ordinal(), name: source_id.name() }
}

/// Return the exact number of logical endpoints from the canonical source schedule.
#[must_use]
pub fn global_endpoint_count(record: &ExecutionRecord) -> usize {
    GLOBAL_PRODUCER_SCHEDULE.into_iter().map(|batch| batch.endpoint_count(record)).sum()
}

fn visit_sorted_memory_endpoints(
    events: &[MemoryInitializeFinalizeEvent],
    is_receive: bool,
    visit: &mut impl FnMut(GlobalInteractionEvent),
) {
    debug_assert!(
        events.windows(2).all(|pair| pair[0].addr < pair[1].addr),
        "Global Memory boundary is not strictly address ordered"
    );

    for event in events {
        let interaction_shard = if is_receive { event.shard } else { 0 };
        let interaction_clk = if is_receive { event.timestamp } else { 0 };
        visit(GlobalInteractionEvent {
            message: [
                interaction_shard,
                interaction_clk,
                event.addr,
                event.value & 255,
                (event.value >> 8) & 255,
                (event.value >> 16) & 255,
                (event.value >> 24) & 255,
            ],
            is_receive,
            kind: InteractionKind::Memory as u8,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{
        events::{
            KeccakPermuteEvent, MemoryLocalEvent, MemoryRecord, MemoryWriteRecord,
            Poseidon2PermuteEvent, PrecompileEvent, ShaCompressEvent, ShaExtendEvent, SyscallEvent,
        },
        RTypeRecord,
    };

    fn syscall(code: SyscallCode, shard: u32, clk: u32, arg1: u32, arg2: u32) -> SyscallEvent {
        SyscallEvent {
            pc: 0,
            next_pc: 4,
            shard,
            clk,
            a_record: MemoryWriteRecord::default(),
            a_record_is_real: false,
            op_a_0: false,
            syscall_code: code,
            syscall_id: code.syscall_id(),
            arg1,
            arg2,
        }
    }

    #[test]
    fn source_schedule_is_stable_and_closed() {
        for (expected, batch) in GlobalSourceId::ALL.into_iter().zip(GLOBAL_PRODUCER_SCHEDULE) {
            assert_eq!(batch.source_id, expected);
            assert_eq!(batch.ordinal, expected as u16);
            assert!(!batch.name.is_empty());
        }
    }

    #[test]
    fn handmade_record_covers_all_eight_sources_in_schedule_order() {
        let mut record = ExecutionRecord::default();
        record
            .syscall_events
            .push((RTypeRecord::default(), syscall(SyscallCode::SHA_COMPRESS, 1, 2, 3, 4)));
        record.add_precompile_event(
            SyscallCode::POSEIDON2_PERMUTE,
            syscall(SyscallCode::POSEIDON2_PERMUTE, 5, 6, 7, 8),
            PrecompileEvent::Poseidon2Permute(Poseidon2PermuteEvent::default()),
        );
        record.add_precompile_event(
            SyscallCode::SHA_EXTEND,
            syscall(SyscallCode::SHA_EXTEND, 9, 10, 11, 0),
            PrecompileEvent::ShaExtend(ShaExtendEvent {
                shard: 9,
                clk: 10,
                w_ptr: 11,
                ..Default::default()
            }),
        );
        record.add_precompile_event(
            SyscallCode::SHA_COMPRESS,
            syscall(SyscallCode::SHA_COMPRESS, 12, 13, 14, 15),
            PrecompileEvent::ShaCompress(ShaCompressEvent {
                shard: 12,
                clk: 13,
                w_ptr: 14,
                h_ptr: 15,
                ..Default::default()
            }),
        );
        record.add_precompile_event(
            SyscallCode::KECCAK_PERMUTE,
            syscall(SyscallCode::KECCAK_PERMUTE, 16, 17, 18, 0),
            PrecompileEvent::KeccakPermute(KeccakPermuteEvent {
                shard: 16,
                clk: 17,
                state_addr: 18,
                ..Default::default()
            }),
        );
        record.global_memory_initialize_events = vec![
            MemoryInitializeFinalizeEvent::initialize(20, 21),
            MemoryInitializeFinalizeEvent::initialize(24, 25),
        ];
        record.global_memory_finalize_events = vec![
            MemoryInitializeFinalizeEvent { addr: 20, value: 22, shard: 2, timestamp: 3 },
            MemoryInitializeFinalizeEvent { addr: 24, value: 26, shard: 4, timestamp: 5 },
        ];
        record.cpu_local_memory_access.push(MemoryLocalEvent {
            addr: 28,
            initial_mem_access: MemoryRecord { shard: 6, timestamp: 7, value: 29 },
            final_mem_access: MemoryRecord { shard: 8, timestamp: 9, value: 30 },
        });

        let expected_counts = [1, 1, 2, 2, 2, 1, 1, 1];
        let mut endpoints = Vec::new();
        for (batch, expected_count) in GLOBAL_PRODUCER_SCHEDULE.into_iter().zip(expected_counts) {
            assert_eq!(batch.endpoint_count(&record), expected_count);
            let before = endpoints.len();
            batch.visit_endpoints(&record, |endpoint| endpoints.push(endpoint));
            assert_eq!(endpoints.len() - before, expected_count);
        }
        assert_eq!(global_endpoint_count(&record), endpoints.len());
        assert_eq!(record.global_endpoint_count(), endpoints.len());
        assert_eq!(endpoints[2].message[2], 20);
        assert_eq!(endpoints[3].message[2], 24);
        assert_eq!(endpoints[4].message[2], 20);
        assert_eq!(endpoints[5].message[2], 24);
        assert!(endpoints[6].is_receive);
        assert!(!endpoints[7].is_receive);
    }

    #[test]
    fn memory_local_endpoint_ranges_are_disjoint_and_order_preserving() {
        let mut record = ExecutionRecord::default();
        record.cpu_local_memory_access = vec![
            MemoryLocalEvent {
                addr: 7,
                initial_mem_access: MemoryRecord { shard: 1, timestamp: 2, value: 3 },
                final_mem_access: MemoryRecord { shard: 4, timestamp: 5, value: 6 },
            },
            MemoryLocalEvent {
                addr: 8,
                initial_mem_access: MemoryRecord { shard: 9, timestamp: 10, value: 11 },
                final_mem_access: MemoryRecord { shard: 12, timestamp: 13, value: 14 },
            },
        ];
        let batch = GLOBAL_PRODUCER_SCHEDULE[GlobalSourceId::MemoryLocal as usize];
        let mut full = Vec::new();
        batch.visit_endpoints(&record, |endpoint| full.push(endpoint));
        let mut partitioned = Vec::new();
        for range in [0..1, 1..3, 3..4] {
            batch.visit_endpoint_range(&record, range, |endpoint| partitioned.push(endpoint));
        }
        assert_eq!(partitioned, full);
    }
}
