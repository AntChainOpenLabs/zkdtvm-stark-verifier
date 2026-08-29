use super::MemoryChipType;
use crate::{
    operations::AssertLtColsBytes,
    utils::{next_power_of_two, padded_rows_threshold},
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryInitializeFinalizeEvent},
    ByteOpcode, ExecutionRecord, Program,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{AirInteraction, BaseAirBuilder, DTAirBuilder, InteractionScope, MachineAir},
    sumcheck::trace::{CompressedMatrix, PaddingRow},
    InteractionKind, Word,
};
use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Most significant byte threshold for field-element address range check.
/// Address MSB must be < this value, or == this value with lower bytes all zero.
pub(crate) const FIELD_ADDR_MSB_THRESHOLD: u8 =
    if cfg!(feature = "koalabear") { 0x7f } else { 0x78 };

/// A memory chip that can initialize or finalize values in memory.
pub struct MemoryGlobalChip {
    pub kind: MemoryChipType,
}

impl MemoryGlobalChip {
    /// Creates a new memory chip with a certain type.
    pub const fn new(kind: MemoryChipType) -> Self {
        Self { kind }
    }
}

impl<F> BaseAir<F> for MemoryGlobalChip {
    fn width(&self) -> usize {
        NUM_MEMORY_INIT_COLS
    }
}

impl<F: Field> MachineAir<F> for MemoryGlobalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        match self.kind {
            MemoryChipType::Initialize => "MemoryGlobalInit".to_string(),
            MemoryChipType::Finalize => "MemoryGlobalFinalize".to_string(),
        }
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        memory_events.sort_by_key(|event| event.addr);

        // Generate byte lookup events for range checks and comparisons.
        let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();

        for (i, event) in memory_events.iter().enumerate() {
            let addr_bytes = event.addr.to_le_bytes();
            let value_bytes = event.value.to_le_bytes();

            // U8Range for addr bytes.
            blu.add_u8_range_check(addr_bytes[0], addr_bytes[1]);
            blu.add_u8_range_check(addr_bytes[2], addr_bytes[3]);

            // Field range check: LTU(addr_bytes[3], FIELD_ADDR_MSB_THRESHOLD).
            let is_lt = (addr_bytes[3] < FIELD_ADDR_MSB_THRESHOLD) as u16;
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: is_lt,
                a2: 0,
                b: addr_bytes[3],
                c: FIELD_ADDR_MSB_THRESHOLD,
            });

            // U8Range for value bytes.
            blu.add_u8_range_check(value_bytes[0], value_bytes[1]);
            blu.add_u8_range_check(value_bytes[2], value_bytes[3]);

            // prev_addr < addr comparison (when applicable).
            let prev_addr = if i == 0 {
                match self.kind {
                    MemoryChipType::Initialize => input.public_values.previous_init_addr,
                    MemoryChipType::Finalize => input.public_values.previous_finalize_addr,
                }
            } else {
                memory_events[i - 1].addr
            };

            // Skip comparison when both prev_addr and addr are 0 (register x0 in first shard).
            if event.addr != 0 {
                let prev_bytes = prev_addr.to_le_bytes();

                // U8Range for prev_addr bytes.
                blu.add_u8_range_check(prev_bytes[0], prev_bytes[1]);
                blu.add_u8_range_check(prev_bytes[2], prev_bytes[3]);

                // LTU for the first differing byte (MSB first comparison).
                for (a_byte, b_byte) in prev_bytes.iter().rev().zip(addr_bytes.iter().rev()) {
                    if a_byte < b_byte {
                        blu.add_byte_lookup_event(ByteLookupEvent {
                            opcode: ByteOpcode::LTU,
                            a1: 1,
                            a2: 0,
                            b: *a_byte,
                            c: *b_byte,
                        });
                        break;
                    }
                }
            }
        }

        output.add_byte_lookup_events_from_maps(vec![&blu]);
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = match self.kind {
            MemoryChipType::Initialize => &input.global_memory_initialize_events,
            MemoryChipType::Finalize => &input.global_memory_finalize_events,
        };
        let nb_rows = events.len();
        let size_log2 = input.fixed_log2_rows::<F, Self>(self);
        let padded_nb_rows = padded_rows_threshold(next_power_of_two(nb_rows, size_log2));
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> CompressedMatrix<F> {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let previous_addr = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr,
        };

        memory_events.sort_by_key(|event| event.addr);
        let mut rows: Vec<[F; NUM_MEMORY_INIT_COLS]> = memory_events
            .par_iter()
            .map(|event| {
                let MemoryInitializeFinalizeEvent { addr, value, shard, timestamp } =
                    event.to_owned();

                let mut row = [F::zero(); NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_word = Word::from(addr);
                cols.shard = F::from_canonical_u32(shard);
                cols.timestamp = F::from_canonical_u32(timestamp);
                cols.value = Word::from(value);
                cols.is_real = F::one();

                let addr_bytes = addr.to_le_bytes();
                cols.is_addr_lt_threshold =
                    if addr_bytes[3] < FIELD_ADDR_MSB_THRESHOLD { F::one() } else { F::zero() };

                if addr == 0 {
                    cols.is_addr_zero = F::one();
                }

                row
            })
            .collect::<Vec<_>>();

        // Sequential pass: fill prev_addr_word and lt_cols (depend on previous row).
        for i in 0..memory_events.len() {
            let addr = memory_events[i].addr;
            let cols: &mut MemoryInitCols<F> = rows[i].as_mut_slice().borrow_mut();

            let prev_addr = if i == 0 { previous_addr } else { memory_events[i - 1].addr };

            cols.prev_addr_word = Word::from(prev_addr);

            // Populate lt comparison when addr != 0 (when addr == 0, is_addr_zero skips lt).
            if addr != 0 {
                debug_assert!(prev_addr < addr, "prev_addr {prev_addr} < addr {addr}");
                let mut dummy_record: Vec<ByteLookupEvent> = Vec::new();
                cols.lt_cols.populate(
                    &mut dummy_record,
                    &prev_addr.to_le_bytes(),
                    &addr.to_le_bytes(),
                );
            }
        }

        let padded_nb_rows = <MemoryGlobalChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let main = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_MEMORY_INIT_COLS,
        );
        CompressedMatrix::new(
            main,
            PaddingRow::Zero { width: NUM_MEMORY_INIT_COLS },
            padded_nb_rows,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            match self.kind {
                MemoryChipType::Initialize => !shard.global_memory_initialize_events.is_empty(),
                MemoryChipType::Finalize => !shard.global_memory_finalize_events.is_empty(),
            }
        }
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

/// Columns for MemoryGlobal: uses byte-level decomposition and verifier-side
/// boundary handling, reducing column count to 24.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MemoryInitCols<T: Copy> {
    /// The shard number of the memory access.
    pub shard: T,

    /// The timestamp of the memory access.
    pub timestamp: T,

    /// The address of the memory access (field element).
    pub addr: T,

    /// Byte decomposition of `addr` (little-endian, 4 bytes).
    pub addr_word: Word<T>,

    /// Byte decomposition of the previous address (little-endian, 4 bytes).
    pub prev_addr_word: Word<T>,

    /// Byte-level comparison columns for `prev_addr < addr`.
    pub lt_cols: AssertLtColsBytes<T, 4>,

    /// The value of the memory access (4 bytes, little-endian).
    pub value: Word<T>,

    /// Whether the memory access is a real access.
    pub is_real: T,

    /// Whether addr == 0 (register x0). Skips lt comparison and enforces value = 0.
    pub is_addr_zero: T,

    /// Whether addr_word[3] < FIELD_ADDR_MSB_THRESHOLD (for field-element range check).
    pub is_addr_lt_threshold: T,
}

pub const NUM_MEMORY_INIT_COLS: usize = size_of::<MemoryInitCols<u8>>();

impl<AB> Air<AB> for MemoryGlobalChip
where
    AB: DTAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryInitCols<AB::Var> = (*local).borrow();

        // --- Boolean constraints ---
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_addr_zero);
        builder.assert_bool(local.is_addr_lt_threshold);

        // is_addr_zero can only be set on real rows.
        builder.when_not(local.is_real).assert_zero(local.is_addr_zero);

        // --- Addr byte decomposition and range check ---
        let addr_from_word: AB::Expr = local.addr_word.reduce::<AB>();
        builder.when(local.is_real).assert_eq(addr_from_word, local.addr);

        // Range check addr_word bytes via U8Range byte lookups.
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.addr_word[0],
            local.addr_word[1],
            local.is_real,
        );
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.addr_word[2],
            local.addr_word[3],
            local.is_real,
        );

        // Field range check: verify addr < field modulus.
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::LTU as u8),
            local.is_addr_lt_threshold,
            local.addr_word[3],
            AB::F::from_canonical_u8(FIELD_ADDR_MSB_THRESHOLD),
            local.is_real,
        );
        builder
            .when(local.is_real)
            .when_not(local.is_addr_lt_threshold)
            .assert_eq(local.addr_word[3], AB::F::from_canonical_u8(FIELD_ADDR_MSB_THRESHOLD));
        builder
            .when(local.is_real)
            .when_not(local.is_addr_lt_threshold)
            .assert_zero(local.addr_word[2]);
        builder
            .when(local.is_real)
            .when_not(local.is_addr_lt_threshold)
            .assert_zero(local.addr_word[1]);
        builder
            .when(local.is_real)
            .when_not(local.is_addr_lt_threshold)
            .assert_zero(local.addr_word[0]);

        // --- is_addr_zero constraints ---
        // When is_addr_zero = 1: addr must be 0, prev_addr must be 0, value must be 0.
        builder.when(local.is_addr_zero).assert_zero(local.addr);
        let prev_addr_reconstructed: AB::Expr = local.prev_addr_word.reduce::<AB>();
        builder.when(local.is_addr_zero).assert_zero(prev_addr_reconstructed.clone());
        for i in 0..4 {
            builder.when(local.is_addr_zero).assert_zero(local.value[i]);
        }

        // --- Prev addr byte range check (only when lt comparison is active) ---
        let lt_mult: AB::Expr =
            Into::<AB::Expr>::into(local.is_real) - Into::<AB::Expr>::into(local.is_addr_zero);

        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.prev_addr_word[0],
            local.prev_addr_word[1],
            lt_mult.clone(),
        );
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.prev_addr_word[2],
            local.prev_addr_word[3],
            lt_mult.clone(),
        );

        // --- Value range check ---
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.value[0],
            local.value[1],
            local.is_real,
        );
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            local.value[2],
            local.value[3],
            local.is_real,
        );

        // --- Global interaction (memory init/finalize) ---
        if self.kind == MemoryChipType::Initialize {
            builder.send(
                AirInteraction::new(
                    vec![
                        AB::Expr::zero(),
                        AB::Expr::zero(),
                        local.addr.into(),
                        local.value[0].into(),
                        local.value[1].into(),
                        local.value[2].into(),
                        local.value[3].into(),
                        AB::Expr::one(),
                        AB::Expr::zero(),
                        AB::Expr::from_canonical_u8(InteractionKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    InteractionKind::Global,
                ),
                InteractionScope::Local,
            );
        } else {
            builder.send(
                AirInteraction::new(
                    vec![
                        local.shard.into(),
                        local.timestamp.into(),
                        local.addr.into(),
                        local.value[0].into(),
                        local.value[1].into(),
                        local.value[2].into(),
                        local.value[3].into(),
                        AB::Expr::zero(),
                        AB::Expr::one(),
                        AB::Expr::from_canonical_u8(InteractionKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    InteractionKind::Global,
                ),
                InteractionScope::Local,
            );
        }

        // --- Address chain interaction (verifier-side boundary handling) ---
        // All real rows send addr AND receive prev_addr.
        // The first row's unmatched receive and last row's unmatched send are
        // balanced by the verifier via compute_expected_state_imbalance.
        let discriminant: AB::Expr = match self.kind {
            MemoryChipType::Initialize => AB::Expr::zero(),
            MemoryChipType::Finalize => AB::Expr::one(),
        };

        // Send addr for all real rows.
        builder.send(
            AirInteraction::new(
                vec![discriminant.clone(), local.addr.into()],
                local.is_real.into(),
                InteractionKind::MemoryGlobalAddr,
            ),
            InteractionScope::Local,
        );

        // Receive prev_addr for all real rows.
        builder.receive(
            AirInteraction::new(
                vec![discriminant, prev_addr_reconstructed],
                local.is_real.into(),
                InteractionKind::MemoryGlobalAddr,
            ),
            InteractionScope::Local,
        );

        // --- Address ordering: prev_addr < addr (byte-level comparison) ---
        // Active on all real rows except when addr == 0 (is_addr_zero).
        local.lt_cols.eval(builder, &local.prev_addr_word.0, &local.addr_word.0, lt_mult);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::*;
    use crate::{programs::tests::*, riscv::RiscvAir, utils::setup_logger};
    use dt_core_executor::Executor;
    use dt_stark::{
        baby_bear_poseidon2::BabyBearPoseidon2, debug_interactions_with_all_chips, DTCoreOpts,
        InteractionKind, StarkMachine,
    };
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_memory_generate_trace() {
        let program = simple_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.record.clone();

        let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Initialize);

        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress();
        println!("{:?}", trace.values);
        println!(
            "MemoryInitCols width: {} (was ~140, now {})",
            NUM_MEMORY_INIT_COLS, NUM_MEMORY_INIT_COLS
        );

        let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Finalize);
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress();
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{:?}", mem_event);
        }
    }

    #[test]
    fn test_memory_lookup_interactions() {
        setup_logger();
        let program = simple_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let machine: StarkMachine<BabyBearPoseidon2, RiscvAir<BabyBear>> =
            RiscvAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = DTCoreOpts::default();
        machine.generate_dependencies(
            &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
            &opts,
            None,
        );

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
                &machine,
                &pkey,
                &[*shard],
                vec![InteractionKind::Memory],
                InteractionScope::Local,
            );
        }
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
            &machine,
            &pkey,
            &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
            vec![InteractionKind::Memory],
            InteractionScope::Global,
        );
    }

    #[test]
    fn test_byte_lookup_interactions() {
        setup_logger();
        let program = simple_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let machine = RiscvAir::machine(BabyBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = DTCoreOpts::default();
        machine.generate_dependencies(
            &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
            &opts,
            None,
        );

        let shards = runtime.records;
        debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
            &machine,
            &pkey,
            &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
            vec![InteractionKind::Byte],
            InteractionScope::Global,
        );
    }
}
