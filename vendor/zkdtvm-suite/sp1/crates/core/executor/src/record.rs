use dt_stark::{
    air::{MachineAir, PublicValues},
    shape::Shape,
    DTCoreOpts, MachineRecord, SplitOpts,
};
use enum_map::EnumMap;
use hashbrown::HashMap;
use itertools::{EitherOrBoth, Itertools};
use p3_field::{AbstractField, Field};
use std::{mem::take, ops::Range, str::FromStr, sync::Arc};

use serde::{Deserialize, Serialize};

#[cfg(feature = "koalabear")]
use dt_stark::sumcheck::trace::CompressedMatrix;
#[cfg(feature = "koalabear")]
use p3_koala_bear::KoalaBear;
#[cfg(feature = "koalabear")]
use std::sync::Mutex;

use crate::{
    events::{
        AUIPCEvent, AluEvent, BranchEvent, ByteLookupEvent, ByteRecord, GlobalInteractionEvent,
        GlobalSourceId, JumpEvent, MemInstrEvent, MemoryInitializeFinalizeEvent, MemoryLocalEvent,
        MemoryRecordEnum, PrecompileEvent, PrecompileEvents, SyscallEvent,
    },
    program::Program,
    syscalls::SyscallCode,
    Instruction, RiscvAirId,
};

/// Record-local owner for the one retained Global main trace.
///
/// The slot is populated during dependency generation and consumed during trace generation. The
/// matrix is move-only: cloning after installation is a lifecycle error rather than a second
/// full-height allocation.
#[cfg(feature = "koalabear")]
#[derive(Default)]
pub struct GlobalTraceArtifactSlot {
    inner: Mutex<GlobalTraceArtifact>,
}

#[cfg(feature = "koalabear")]
#[derive(Default)]
struct GlobalTraceArtifact {
    main: Option<CompressedMatrix<KoalaBear>>,
    reducer: Option<CompressedMatrix<KoalaBear>>,
}

#[cfg(feature = "koalabear")]
impl Clone for GlobalTraceArtifactSlot {
    fn clone(&self) -> Self {
        assert!(
            !self.inner.lock().expect("Global trace artifact slot poisoned").is_populated(),
            "Global trace artifact is move-only after dependency generation"
        );
        Self::default()
    }
}

#[cfg(feature = "koalabear")]
impl core::fmt::Debug for GlobalTraceArtifactSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let populated = self.inner.lock().map_err(|_| core::fmt::Error)?.is_populated();
        f.debug_struct("GlobalTraceArtifactSlot").field("populated", &populated).finish()
    }
}

#[cfg(feature = "koalabear")]
impl GlobalTraceArtifactSlot {
    fn install(
        &self,
        main: CompressedMatrix<KoalaBear>,
        reducer: CompressedMatrix<KoalaBear>,
    ) {
        let mut slot = self.inner.lock().expect("Global trace artifact slot poisoned");
        assert!(!slot.is_populated(), "Global trace artifact installed more than once");
        slot.main = Some(main);
        slot.reducer = Some(reducer);
    }

    fn take_main(&self) -> Option<CompressedMatrix<KoalaBear>> {
        self.inner.lock().expect("Global trace artifact slot poisoned").main.take()
    }

    fn take_reducer(&self) -> Option<CompressedMatrix<KoalaBear>> {
        self.inner.lock().expect("Global trace artifact slot poisoned").reducer.take()
    }

    fn clear(&self) {
        *self.inner.lock().expect("Global trace artifact slot poisoned") =
            GlobalTraceArtifact::default();
    }

    fn is_populated(&self) -> bool {
        self.inner.lock().expect("Global trace artifact slot poisoned").is_populated()
    }
}

#[cfg(feature = "koalabear")]
impl GlobalTraceArtifact {
    fn is_populated(&self) -> bool {
        self.main.is_some() || self.reducer.is_some()
    }
}

/// A record of the execution of a program.
///
/// The trace of the execution is represented as a list of "events" that occur every cycle.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ExecutionRecord {
    /// The program.
    pub program: Arc<Program>,
    /// The total number of CPU events (instructions executed) in this shard.
    pub cpu_events: u32,
    /// The program counter of the first instruction in this shard.
    pub start_pc: u32,
    /// The next program counter after the last instruction in this shard.
    pub last_next_pc: u32,
    /// The exit code of the last instruction in this shard.
    pub last_exit_code: u32,
    /// A trace of the ADD events (R-type: 3 register operands).
    pub add_events: Vec<(RTypeRecord, AluEvent)>,
    /// A trace of the ADDI events (may have b as register or immediate, covers LUI-as-ADD too).
    pub addi_events: Vec<(AddiRecord, AluEvent)>,
    /// A trace of the MUL events (R-type).
    pub mul_events: Vec<(RTypeRecord, AluEvent)>,
    /// A trace of the SUB events (R-type).
    pub sub_events: Vec<(RTypeRecord, AluEvent)>,
    /// A trace of the XOR, XORI, OR, ORI, AND, and ANDI events (ALU-type: may have immediate).
    pub bitwise_events: Vec<(ALUTypeRecord, AluEvent)>,
    /// A trace of the SLL and SLLI events (ALU-type).
    pub shift_left_events: Vec<(ALUTypeRecord, AluEvent)>,
    /// A trace of the SRL, SRLI, SRA, and SRAI events (ALU-type).
    pub shift_right_events: Vec<(ALUTypeRecord, AluEvent)>,
    /// A trace of the DIV, DIVU, REM, and REMU events (R-type).
    pub divrem_events: Vec<(RTypeRecord, AluEvent)>,
    /// A trace of the SLT, SLTI, SLTU, and SLTIU events (ALU-type).
    pub lt_events: Vec<(ALUTypeRecord, AluEvent)>,
    /// A trace of the load byte events (I-type).
    pub load_byte_events: Vec<(ITypeRecord, MemInstrEvent)>,
    /// A trace of the load half events (I-type).
    pub load_half_events: Vec<(ITypeRecord, MemInstrEvent)>,
    /// A trace of the load word events (I-type).
    pub load_word_events: Vec<(ITypeRecord, MemInstrEvent)>,
    /// A trace of the store byte events (B-type).
    pub store_byte_events: Vec<(BTypeRecord, MemInstrEvent)>,
    /// A trace of the store half events (B-type).
    pub store_half_events: Vec<(BTypeRecord, MemInstrEvent)>,
    /// A trace of the store word events (B-type).
    pub store_word_events: Vec<(BTypeRecord, MemInstrEvent)>,
    /// A trace of the AUIPC events (J-type).
    pub auipc_events: Vec<(JTypeRecord, AUIPCEvent)>,
    /// A trace of the branch events (B-type).
    pub branch_events: Vec<(BTypeRecord, BranchEvent)>,
    /// A trace of the jal events (J-type).
    pub jal_events: Vec<(JTypeRecord, JumpEvent)>,
    /// A trace of the jalr events (I-type).
    pub jalr_events: Vec<(ITypeRecord, JumpEvent)>,
    /// A trace of the byte lookups that are needed.
    pub byte_lookups: HashMap<ByteLookupEvent, usize>,
    /// A trace of the precompile events.
    pub precompile_events: PrecompileEvents,
    /// A trace of the global memory initialize events.
    pub global_memory_initialize_events: Vec<MemoryInitializeFinalizeEvent>,
    /// A trace of the global memory finalize events.
    pub global_memory_finalize_events: Vec<MemoryInitializeFinalizeEvent>,
    /// A trace of all the shard's local memory events.
    pub cpu_local_memory_access: Vec<MemoryLocalEvent>,
    /// A trace of all the syscall events (R-type: syscall instruction has 3 register operands).
    pub syscall_events: Vec<(RTypeRecord, SyscallEvent)>,
    /// A trace of all the global interaction events.
    pub global_interaction_events: Vec<GlobalInteractionEvent>,
    /// The public values.
    pub public_values: PublicValues<u32, u32>,
    /// The next nonce to use for a new lookup.
    pub next_nonce: u64,
    /// The shape of the proof.
    pub shape: Option<Shape<RiscvAirId>>,
    /// The predicted counts of the proof.
    pub counts: Option<EnumMap<RiscvAirId, u64>>,
    /// One-shot Global trace owner. This is runtime state, never wire data.
    #[cfg(feature = "koalabear")]
    #[serde(skip, default)]
    global_trace_artifact: GlobalTraceArtifactSlot,
}

impl ExecutionRecord {
    /// Create a new [`ExecutionRecord`].
    #[must_use]
    pub fn new(program: Arc<Program>) -> Self {
        Self { program, ..Default::default() }
    }

    /// Returns the execution shard number.
    ///
    /// This is `public_values.execution_shard`, which only increments for shards containing
    /// CPU execution. Use this (not `shard()`) for `CPUState` and memory timestamp comparisons.
    #[inline]
    #[must_use]
    pub fn execution_shard(&self) -> u32 {
        self.public_values.execution_shard
    }

    /// Returns an iterator over all instruction PCs in this record.
    ///
    /// This is useful for counting instruction invocations by `pc` (e.g. in the program chip)
    /// without any cloning or ownership transfer. Each event type has `pc` in its event struct.
    pub fn all_event_pcs(&self) -> impl Iterator<Item = u32> + '_ {
        self.add_events
            .iter()
            .map(|(_, e)| e.pc)
            .chain(self.addi_events.iter().map(|(_, e)| e.pc))
            .chain(self.sub_events.iter().map(|(_, e)| e.pc))
            .chain(self.mul_events.iter().map(|(_, e)| e.pc))
            .chain(self.bitwise_events.iter().map(|(_, e)| e.pc))
            .chain(self.shift_left_events.iter().map(|(_, e)| e.pc))
            .chain(self.shift_right_events.iter().map(|(_, e)| e.pc))
            .chain(self.divrem_events.iter().map(|(_, e)| e.pc))
            .chain(self.lt_events.iter().map(|(_, e)| e.pc))
            .chain(self.load_byte_events.iter().map(|(_, e)| e.pc))
            .chain(self.load_half_events.iter().map(|(_, e)| e.pc))
            .chain(self.load_word_events.iter().map(|(_, e)| e.pc))
            .chain(self.store_byte_events.iter().map(|(_, e)| e.pc))
            .chain(self.store_half_events.iter().map(|(_, e)| e.pc))
            .chain(self.store_word_events.iter().map(|(_, e)| e.pc))
            .chain(self.auipc_events.iter().map(|(_, e)| e.pc))
            .chain(self.branch_events.iter().map(|(_, e)| e.pc))
            .chain(self.jal_events.iter().map(|(_, e)| e.pc))
            .chain(self.jalr_events.iter().map(|(_, e)| e.pc))
            .chain(self.syscall_events.iter().map(|(_, e)| e.pc))
    }

    /// Take out events from the [`ExecutionRecord`] that should be deferred to a separate shard.
    ///
    /// Note: we usually defer events that would increase the recursion cost significantly if
    /// included in every shard.
    #[must_use]
    pub fn defer(&mut self) -> ExecutionRecord {
        let mut execution_record = ExecutionRecord::new(self.program.clone());
        execution_record.precompile_events = std::mem::take(&mut self.precompile_events);
        execution_record.global_memory_initialize_events =
            std::mem::take(&mut self.global_memory_initialize_events);
        execution_record.global_memory_finalize_events =
            std::mem::take(&mut self.global_memory_finalize_events);
        execution_record
    }

    /// Splits the deferred [`ExecutionRecord`] into multiple [`ExecutionRecord`]s, each which
    /// contain a "reasonable" number of deferred events.
    ///
    /// The optional `last_record` will be provided if there are few enough deferred events that
    /// they can all be packed into the already existing last record.
    pub fn split(
        &mut self,
        last: bool,
        last_record: Option<&mut ExecutionRecord>,
        opts: SplitOpts,
    ) -> Vec<ExecutionRecord> {
        let mut shards = Vec::new();

        let precompile_events = take(&mut self.precompile_events);

        for (syscall_code, events) in precompile_events.into_iter() {
            let threshold = syscall_code.precompile_shard_limit();

            let chunks = events.chunks_exact(threshold);
            if last {
                let remainder = chunks.remainder().to_vec();
                if !remainder.is_empty() {
                    let mut execution_record = ExecutionRecord::new(self.program.clone());
                    execution_record.precompile_events.insert(syscall_code, remainder);
                    shards.push(execution_record);
                }
            } else {
                self.precompile_events.insert(syscall_code, chunks.remainder().to_vec());
            }
            let mut event_shards = chunks
                .map(|chunk| {
                    let mut execution_record = ExecutionRecord::new(self.program.clone());
                    execution_record.precompile_events.insert(syscall_code, chunk.to_vec());
                    execution_record
                })
                .collect::<Vec<_>>();
            shards.append(&mut event_shards);
        }

        if last {
            self.global_memory_initialize_events.sort_by_key(|event| event.addr);
            self.global_memory_finalize_events.sort_by_key(|event| event.addr);

            // If there are no precompile shards, and `last_record` is Some, pack the memory events
            // into the last record.
            let pack_memory_events_into_last_record = last_record.is_some() && shards.is_empty();
            let mut blank_record = ExecutionRecord::new(self.program.clone());

            // If `last_record` is None, use a blank record to store the memory events.
            let last_record_ref = if pack_memory_events_into_last_record {
                last_record.unwrap()
            } else {
                &mut blank_record
            };

            let mut init_addr: u32 = 0;
            let mut finalize_addr: u32 = 0;
            for mem_chunks in self
                .global_memory_initialize_events
                .chunks(opts.memory)
                .zip_longest(self.global_memory_finalize_events.chunks(opts.memory))
            {
                let (mem_init_chunk, mem_finalize_chunk) = match mem_chunks {
                    EitherOrBoth::Both(mem_init_chunk, mem_finalize_chunk) => {
                        (mem_init_chunk, mem_finalize_chunk)
                    }
                    EitherOrBoth::Left(mem_init_chunk) => (mem_init_chunk, [].as_slice()),
                    EitherOrBoth::Right(mem_finalize_chunk) => ([].as_slice(), mem_finalize_chunk),
                };
                last_record_ref.global_memory_initialize_events.extend_from_slice(mem_init_chunk);
                last_record_ref.public_values.previous_init_addr = init_addr;
                if let Some(last_event) = mem_init_chunk.last() {
                    init_addr = last_event.addr;
                }
                last_record_ref.public_values.last_init_addr = init_addr;

                last_record_ref.global_memory_finalize_events.extend_from_slice(mem_finalize_chunk);
                last_record_ref.public_values.previous_finalize_addr = finalize_addr;
                if let Some(last_event) = mem_finalize_chunk.last() {
                    finalize_addr = last_event.addr;
                }
                last_record_ref.public_values.last_finalize_addr = finalize_addr;

                if !pack_memory_events_into_last_record {
                    // If not packing memory events into the last record, add 'last_record_ref'
                    // to the returned records. `take` replaces `blank_program` with the default.
                    shards.push(take(last_record_ref));

                    // Reset the last record so its program is the correct one. (The default program
                    // provided by `take` contains no instructions.)
                    last_record_ref.program = self.program.clone();
                }
            }
        }
        shards
    }

    /// Return the number of rows needed for a chip, according to the proof shape specified in the
    /// struct.
    pub fn fixed_log2_rows<F: Field, A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        self.shape.as_ref().map(|shape| {
            shape
                .log2_height(&RiscvAirId::from_str(&air.name()).unwrap())
                .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
        })
    }

    /// Determines whether the execution record contains CPU events.
    #[must_use]
    pub fn contains_cpu(&self) -> bool {
        self.cpu_events > 0
    }

    #[inline]
    /// Add a precompile event to the execution record.
    pub fn add_precompile_event(
        &mut self,
        syscall_code: SyscallCode,
        syscall_event: SyscallEvent,
        event: PrecompileEvent,
    ) {
        self.precompile_events.add_event(syscall_code, syscall_event, event);
    }

    /// Get all the precompile events for a syscall code.
    #[inline]
    #[must_use]
    pub fn get_precompile_events(
        &self,
        syscall_code: SyscallCode,
    ) -> &Vec<(SyscallEvent, PrecompileEvent)> {
        self.precompile_events.get_events(syscall_code).expect("Precompile events not found")
    }

    /// Get all the local memory events.
    #[inline]
    pub fn get_local_mem_events(&self) -> impl Iterator<Item = &MemoryLocalEvent> {
        let precompile_local_mem_events = self.precompile_events.get_local_mem_events();
        precompile_local_mem_events.chain(self.cpu_local_memory_access.iter())
    }

    /// Visit a deterministic range of local-memory lifetimes without materializing endpoints.
    pub fn visit_local_mem_event_range(
        &self,
        range: Range<usize>,
        mut visit: impl FnMut(&MemoryLocalEvent),
    ) {
        let precompile_count = self.precompile_events.get_local_mem_events().count();
        let total = precompile_count + self.cpu_local_memory_access.len();
        assert!(range.start <= range.end && range.end <= total);

        let precompile_end = range.end.min(precompile_count);
        if range.start < precompile_end {
            for event in self
                .precompile_events
                .get_local_mem_events()
                .skip(range.start)
                .take(precompile_end - range.start)
            {
                visit(event);
            }
        }

        if range.end > precompile_count {
            let cpu_start = range.start.saturating_sub(precompile_count);
            let cpu_end = range.end - precompile_count;
            for event in &self.cpu_local_memory_access[cpu_start..cpu_end] {
                visit(event);
            }
        }
    }

    /// Return the exact logical Global endpoint count for one stable producer source.
    #[must_use]
    pub fn global_source_endpoint_count(&self, source: GlobalSourceId) -> usize {
        match source {
            GlobalSourceId::CoreSyscall => self
                .syscall_events
                .iter()
                .filter(|(_, event)| event.syscall_code.should_send() == 1)
                .count(),
            GlobalSourceId::DeferredSyscall => self
                .precompile_events
                .all_precompile_events()
                .filter(|(event, _)| event.syscall_code.should_send() == 1)
                .count(),
            GlobalSourceId::MemoryInitialize => self.global_memory_initialize_events.len(),
            GlobalSourceId::MemoryFinalize => self.global_memory_finalize_events.len(),
            GlobalSourceId::MemoryLocal => self.get_local_mem_events().count() * 2,
            GlobalSourceId::ShaExtendController => {
                self.precompile_events.sha_extend_events().count()
            }
            GlobalSourceId::ShaCompressController => {
                self.precompile_events.sha_compress_events().count()
            }
            GlobalSourceId::KeccakController => self.precompile_events.keccak_events().count(),
        }
    }

    /// Return the exact number of logical Global endpoints after producer closure.
    #[must_use]
    pub fn global_endpoint_count(&self) -> usize {
        GlobalSourceId::ALL
            .into_iter()
            .map(|source| self.global_source_endpoint_count(source))
            .sum()
    }

    /// Install the Global main trace prepared by dependency generation.
    #[cfg(feature = "koalabear")]
    pub fn install_global_trace_artifact(
        &self,
        main: CompressedMatrix<KoalaBear>,
        reducer: CompressedMatrix<KoalaBear>,
    ) {
        self.global_trace_artifact.install(main, reducer);
    }

    /// Consume the retained Global main trace exactly once.
    #[cfg(feature = "koalabear")]
    #[must_use]
    pub fn take_global_trace_artifact(&self) -> Option<CompressedMatrix<KoalaBear>> {
        self.global_trace_artifact.take_main()
    }

    /// Consume the retained `GlobalTileReducerV3` trace exactly once.
    #[cfg(feature = "koalabear")]
    #[must_use]
    pub fn take_global_tile_reducer_trace_artifact(&self) -> Option<CompressedMatrix<KoalaBear>> {
        self.global_trace_artifact.take_reducer()
    }

    /// Whether dependency generation has installed a retained Global trace.
    #[cfg(feature = "koalabear")]
    #[must_use]
    pub fn has_global_trace_artifact(&self) -> bool {
        self.global_trace_artifact.is_populated()
    }
}

/// A memory access record.
#[derive(Debug, Copy, Clone, Default)]
pub struct MemoryAccessRecord {
    /// The memory access of the `a` register.
    pub a: Option<MemoryRecordEnum>,
    /// The memory access of the `b` register.
    pub b: Option<MemoryRecordEnum>,
    /// The memory access of the `c` register.
    pub c: Option<MemoryRecordEnum>,
    /// The memory access of the `memory` register.
    pub memory: Option<MemoryRecordEnum>,
}

impl MachineRecord for ExecutionRecord {
    type Config = DTCoreOpts;

    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cpu_events".to_string(), self.cpu_events as usize);
        stats.insert("add_events".to_string(), self.add_events.len());
        stats.insert("addi_events".to_string(), self.addi_events.len());
        stats.insert("mul_events".to_string(), self.mul_events.len());
        stats.insert("sub_events".to_string(), self.sub_events.len());
        stats.insert("bitwise_events".to_string(), self.bitwise_events.len());
        stats.insert("shift_left_events".to_string(), self.shift_left_events.len());
        stats.insert("shift_right_events".to_string(), self.shift_right_events.len());
        stats.insert("divrem_events".to_string(), self.divrem_events.len());
        stats.insert("lt_events".to_string(), self.lt_events.len());
        stats.insert("load_byte_events".to_string(), self.load_byte_events.len());
        stats.insert("load_half_events".to_string(), self.load_half_events.len());
        stats.insert("load_word_events".to_string(), self.load_word_events.len());
        stats.insert("store_byte_events".to_string(), self.store_byte_events.len());
        stats.insert("store_half_events".to_string(), self.store_half_events.len());
        stats.insert("store_word_events".to_string(), self.store_word_events.len());
        stats.insert("branch_events".to_string(), self.branch_events.len());
        stats.insert("jal_events".to_string(), self.jal_events.len());
        stats.insert("jalr_events".to_string(), self.jalr_events.len());
        stats.insert("auipc_events".to_string(), self.auipc_events.len());

        for (syscall_code, events) in self.precompile_events.iter() {
            stats.insert(format!("syscall {syscall_code:?}"), events.len());
        }

        stats.insert(
            "global_memory_initialize_events".to_string(),
            self.global_memory_initialize_events.len(),
        );
        stats.insert(
            "global_memory_finalize_events".to_string(),
            self.global_memory_finalize_events.len(),
        );
        stats.insert("local_memory_access_events".to_string(), self.cpu_local_memory_access.len());
        if self.cpu_events > 0 {
            stats.insert("byte_lookups".to_string(), self.byte_lookups.len());
        }
        // Filter out the empty events.
        stats.retain(|_, v| *v != 0);
        stats
    }

    fn append(&mut self, other: &mut ExecutionRecord) {
        #[cfg(feature = "koalabear")]
        let incoming_global_artifact = if other.global_trace_artifact.is_populated() {
            Some(take(&mut other.global_trace_artifact))
        } else {
            None
        };
        #[cfg(feature = "koalabear")]
        let changes_global_sources = other.global_endpoint_count() != 0;
        #[cfg(feature = "koalabear")]
        if changes_global_sources {
            self.global_trace_artifact.clear();
        }

        self.cpu_events += other.cpu_events;
        self.add_events.append(&mut other.add_events);
        self.addi_events.append(&mut other.addi_events);
        self.sub_events.append(&mut other.sub_events);
        self.mul_events.append(&mut other.mul_events);
        self.bitwise_events.append(&mut other.bitwise_events);
        self.shift_left_events.append(&mut other.shift_left_events);
        self.shift_right_events.append(&mut other.shift_right_events);
        self.divrem_events.append(&mut other.divrem_events);
        self.lt_events.append(&mut other.lt_events);
        self.load_byte_events.append(&mut other.load_byte_events);
        self.load_half_events.append(&mut other.load_half_events);
        self.load_word_events.append(&mut other.load_word_events);
        self.store_byte_events.append(&mut other.store_byte_events);
        self.store_half_events.append(&mut other.store_half_events);
        self.store_word_events.append(&mut other.store_word_events);
        self.branch_events.append(&mut other.branch_events);
        self.jal_events.append(&mut other.jal_events);
        self.jalr_events.append(&mut other.jalr_events);
        self.auipc_events.append(&mut other.auipc_events);
        self.syscall_events.append(&mut other.syscall_events);

        self.precompile_events.append(&mut other.precompile_events);

        if self.byte_lookups.is_empty() {
            self.byte_lookups = std::mem::take(&mut other.byte_lookups);
        } else {
            self.add_byte_lookup_events_from_maps(vec![&other.byte_lookups]);
        }

        self.global_memory_initialize_events.append(&mut other.global_memory_initialize_events);
        self.global_memory_finalize_events.append(&mut other.global_memory_finalize_events);
        self.cpu_local_memory_access.append(&mut other.cpu_local_memory_access);
        self.global_interaction_events.append(&mut other.global_interaction_events);

        #[cfg(feature = "koalabear")]
        if let Some(artifact) = incoming_global_artifact {
            assert!(
                !self.global_trace_artifact.is_populated(),
                "cannot merge two retained Global trace artifacts"
            );
            self.global_trace_artifact = artifact;
        }
    }

    /// Retrieves the public values.  This method is needed for the `MachineRecord` trait, since
    fn public_values<F: AbstractField>(&self) -> Vec<F> {
        self.public_values.to_vec()
    }
}

impl ByteRecord for ExecutionRecord {
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        *self.byte_lookups.entry(blu_event).or_insert(0) += 1;
    }

    #[inline]
    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        for new_blu_map in new_events {
            for (blu_event, count) in new_blu_map.iter() {
                *self.byte_lookups.entry(*blu_event).or_insert(0) += count;
            }
        }
    }
}

/// Memory record where all three operands are registers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RTypeRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand.
    pub op_a: u8,
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The b operand.
    pub op_b: u32,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The c operand.
    pub op_c: u32,
    /// The register `op_c` record.
    pub c: MemoryRecordEnum,
}

impl RTypeRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in RTypeRecord"),
            op_b: instruction.op_b,
            b: value.b.expect("expected MemoryRecord for op_b in RTypeRecord"),
            op_c: instruction.op_c,
            c: value.c.expect("expected MemoryRecord for op_c in RTypeRecord"),
        }
    }
    #[must_use]
    pub fn op_a_value(&self) -> u32 {
        self.a.current_record().value
    }
    #[must_use]
    pub fn op_b_value(&self) -> u32 {
        self.b.previous_record().value
    }
    #[must_use]
    pub fn op_c_value(&self) -> u32 {
        self.c.previous_record().value
    }
}
/// Memory record for ADDI-like instructions where `op_b` may be either a register or an immediate.
///
/// When `imm_b=false, imm_c=true` (normal ADDI): `b` is `Some` (register read), `op_b` is register
/// index. When `imm_b=true, imm_c=true` (LUI encoded as ADD): `b` is `None`, `op_b` is the
/// immediate value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AddiRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand (destination register index).
    pub op_a: u8,
    /// The register `op_a` record (always a write).
    pub a: MemoryRecordEnum,
    /// The b operand (register index if `imm_b=false`, immediate if `imm_b=true`).
    pub op_b: u32,
    /// The register `op_b` record. `None` when `imm_b=true` (both operands are immediates).
    pub b: Option<MemoryRecordEnum>,
    /// The c operand (always an immediate value).
    pub op_c: u32,
    /// Whether the b operand is an immediate.
    pub imm_b: bool,
}

impl AddiRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in AddiRecord"),
            op_b: instruction.op_b,
            b: value.b,
            op_c: instruction.op_c,
            imm_b: instruction.imm_b,
        }
    }
}

/// Memory record where the first two operands are registers (I-type: a=write, b=read, c=imm).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ITypeRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand.
    pub op_a: u8,
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The b operand.
    pub op_b: u32,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The c operand.
    pub op_c: u32,
}

impl ITypeRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        debug_assert!(value.c.is_none());
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in ITypeRecord"),
            op_b: instruction.op_b,
            b: value.b.expect("expected MemoryRecord for op_b in ITypeRecord"),
            op_c: instruction.op_c,
        }
    }
}
/// Memory record where the first two operands are registers (B-type: a=read, b=read, c=imm).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BTypeRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand.
    pub op_a: u8,
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The b operand.
    pub op_b: u32,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The c operand.
    pub op_c: u32,
}

impl BTypeRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        debug_assert!(value.c.is_none());
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in BTypeRecord"),
            op_b: instruction.op_b,
            b: value.b.expect("expected MemoryRecord for op_b in BTypeRecord"),
            op_c: instruction.op_c,
        }
    }
}

/// Memory record where only one operand is a register (J-type: a=write, b=imm, c=imm).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JTypeRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand.
    pub op_a: u8,
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The b operand.
    pub op_b: u32,
    /// The c operand.
    pub op_c: u32,
}

impl JTypeRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        debug_assert!(value.b.is_none());
        debug_assert!(value.c.is_none());
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in JTypeRecord"),
            op_b: instruction.op_b,
            op_c: instruction.op_c,
        }
    }
}

/// Memory record where the first two operands are known to be registers, third may be immediate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ALUTypeRecord {
    /// The clock cycle.
    pub clk: u32,
    /// The a operand.
    pub op_a: u8,
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The b operand.
    pub op_b: u32,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The c operand.
    pub op_c: u32,
    /// The register `op_c` record.
    pub c: Option<MemoryRecordEnum>,
    /// Whether the instruction has an immediate.
    pub is_imm: bool,
}

impl ALUTypeRecord {
    #[must_use]
    pub fn new(clk: u32, value: &MemoryAccessRecord, instruction: &Instruction) -> Self {
        Self {
            clk,
            op_a: instruction.op_a,
            a: value.a.expect("expected MemoryRecord for op_a in ALUTypeRecord"),
            op_b: instruction.op_b,
            b: value.b.expect("expected MemoryRecord for op_b in ALUTypeRecord"),
            op_c: instruction.op_c,
            c: value.c,
            is_imm: instruction.imm_c,
        }
    }
}
///Memory Records for instructions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum InstructionRecord {
    ///rtype enum
    R(RTypeRecord),
    ///itype enum
    I(ITypeRecord),
    ///jtype enum
    J(JTypeRecord),
    ///alu type enum
    ALU(ALUTypeRecord),
}
impl InstructionRecord {
    ///returns true if the related instruction is an R type
    #[inline]
    #[must_use]
    pub fn is_r_type(&self) -> bool {
        matches!(self, InstructionRecord::R(_))
    }
    ///returns true if the related instruction is an R type
    #[inline]
    #[must_use]
    pub fn is_i_type(&self) -> bool {
        matches!(self, InstructionRecord::I(_))
    }
    ///returns true if the related instruction is an R type
    #[inline]
    #[must_use]
    pub fn is_j_type(&self) -> bool {
        matches!(self, InstructionRecord::J(_))
    }
    ///returns true if the related instruction is an R type
    #[inline]
    #[must_use]
    pub fn is_alu_type(&self) -> bool {
        matches!(self, InstructionRecord::ALU(_))
    }
}

#[cfg(all(test, feature = "koalabear"))]
mod tests {
    use super::*;
    use dt_stark::{
        sumcheck::trace::{CompressedMatrix, PaddingRow},
        MachineRecord,
    };
    use p3_matrix::dense::RowMajorMatrix;

    fn artifact(value: u32) -> CompressedMatrix<KoalaBear> {
        CompressedMatrix::new(
            RowMajorMatrix::new(vec![KoalaBear::from_canonical_u32(value)], 1),
            PaddingRow::None,
            1,
        )
    }

    #[test]
    fn global_trace_artifact_is_consumed_once() {
        let record = ExecutionRecord::default();
        record.install_global_trace_artifact(artifact(7), artifact(8));
        assert!(record.has_global_trace_artifact());
        assert!(record.take_global_trace_artifact().is_some());
        assert!(record.has_global_trace_artifact());
        assert!(record.take_global_tile_reducer_trace_artifact().is_some());
        assert!(!record.has_global_trace_artifact());
        assert!(record.take_global_trace_artifact().is_none());
        assert!(record.take_global_tile_reducer_trace_artifact().is_none());
    }

    #[test]
    fn append_invalidates_stale_artifact_and_moves_incoming_owner() {
        let mut destination = ExecutionRecord::default();
        destination.install_global_trace_artifact(artifact(7), artifact(8));

        let mut incoming = ExecutionRecord::default();
        incoming
            .global_memory_initialize_events
            .push(MemoryInitializeFinalizeEvent::initialize(4, 11));
        incoming.install_global_trace_artifact(artifact(13), artifact(14));

        destination.append(&mut incoming);
        assert!(destination.has_global_trace_artifact());
        assert!(!incoming.has_global_trace_artifact());
        let moved = destination.take_global_trace_artifact().unwrap();
        assert_eq!(moved.main.values, vec![KoalaBear::from_canonical_u32(13)]);
    }
}
