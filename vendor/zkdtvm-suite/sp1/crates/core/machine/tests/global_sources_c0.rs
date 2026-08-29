use dt_core_executor::{
    events::{
        KeccakPermuteEvent, MemoryInitializeFinalizeEvent, MemoryLocalEvent, MemoryRecord,
        MemoryWriteRecord, Poseidon2PermuteEvent, PrecompileEvent, ShaCompressEvent,
        ShaExtendEvent, SyscallEvent,
    },
    syscalls::SyscallCode,
    ExecutionRecord, RTypeRecord,
};
use dt_core_machine::global::{
    global_relation_accepts_for_test, p7_kats, prepare_global_trace,
    sources::{global_endpoint_count, GLOBAL_PRODUCER_SCHEDULE},
    GlobalCols, GlobalTileReducerCols, LookupDirection, GLOBAL_COL_MAP,
    GLOBAL_INTERACTION_DESCRIPTORS, GLOBAL_LAYOUT_FIELDS, GLOBAL_TILE_REDUCER_COL_MAP,
    NUM_GLOBAL_COLS, NUM_GLOBAL_TILE_REDUCER_COLS,
};
use dt_stark::{
    global_d11::{construct_map, D11ProjectivePointV1, GlobalPackInputV1},
    sumcheck::trace::PaddingRow,
    InteractionKind,
};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use std::borrow::Borrow;

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
fn p7_quotient_reconstructs_dense_and_sparse_products() {
    p7_kats::quotient_reconstructs_dense_and_sparse_products();
}

#[test]
fn p7_map_quotient_matches_high_square_and_x_q2_formula() {
    p7_kats::map_quotient_matches_high_square_and_x_q2_formula();
}

#[test]
fn p7_selected_output_quotients_use_wave2_signed_pairs() {
    p7_kats::selected_output_quotients_use_wave2_signed_pairs();
}

#[test]
fn p8_tile_reducer_fixed_tree_and_product_beta() {
    p7_kats::tile_reducer_fixed_tree_and_product_beta();
}

#[test]
fn p8_tile_reducer_rejects_malicious_schedule() {
    p7_kats::tile_reducer_rejects_malicious_schedule();
}

#[test]
fn canonical_schedule_covers_all_sources_and_range_adjacency() {
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
    record.cpu_local_memory_access = vec![
        MemoryLocalEvent {
            addr: 28,
            initial_mem_access: MemoryRecord { shard: 6, timestamp: 7, value: 29 },
            final_mem_access: MemoryRecord { shard: 8, timestamp: 9, value: 30 },
        },
        MemoryLocalEvent {
            addr: 32,
            initial_mem_access: MemoryRecord { shard: 10, timestamp: 11, value: 33 },
            final_mem_access: MemoryRecord { shard: 12, timestamp: 13, value: 34 },
        },
    ];

    let expected_counts = [1, 1, 2, 2, 4, 1, 1, 1];
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

    let local_batch = GLOBAL_PRODUCER_SCHEDULE[4];
    let mut full_local = Vec::new();
    local_batch.visit_endpoints(&record, |endpoint| full_local.push(endpoint));
    let mut ranged_local = Vec::new();
    for range in [0..1, 1..3, 3..4] {
        local_batch.visit_endpoint_range(&record, range, |endpoint| ranged_local.push(endpoint));
    }
    assert_eq!(ranged_local, full_local);
    assert!(full_local[0].is_receive);
    assert!(!full_local[1].is_receive);
}

#[test]
fn canonical_row_writer_preserves_chain_products_padding_delta_and_claim() {
    let mut record = ExecutionRecord::default();
    record.global_memory_initialize_events = vec![
        MemoryInitializeFinalizeEvent::initialize(0x100, 0x1122_3344),
        MemoryInitializeFinalizeEvent::initialize(0x104, 0x5566_7788),
    ];
    record.global_memory_finalize_events = vec![
        MemoryInitializeFinalizeEvent { addr: 0x100, value: 0x1122_3344, shard: 1, timestamp: 2 },
        MemoryInitializeFinalizeEvent { addr: 0x104, value: 0x99aa_bbcc, shard: 3, timestamp: 4 },
    ];
    record.cpu_local_memory_access = vec![MemoryLocalEvent {
        addr: 0x108,
        initial_mem_access: MemoryRecord { shard: 5, timestamp: 6, value: 7 },
        final_mem_access: MemoryRecord { shard: 8, timestamp: 9, value: 10 },
    }];

    let prepared = prepare_global_trace(&record).unwrap();
    let identity = D11ProjectivePointV1::identity();

    assert_eq!(NUM_GLOBAL_COLS, 228);
    assert_eq!(NUM_GLOBAL_TILE_REDUCER_COLS, 83);
    assert_eq!(GLOBAL_LAYOUT_FIELDS.first().unwrap().offset, 0);
    assert_eq!(
        GLOBAL_LAYOUT_FIELDS.last().unwrap().offset + GLOBAL_LAYOUT_FIELDS.last().unwrap().len,
        NUM_GLOBAL_COLS
    );
    assert_eq!(GLOBAL_COL_MAP.index, 24);
    assert_eq!(GLOBAL_COL_MAP.input.x[0], 25);
    assert_eq!(GLOBAL_COL_MAP.products.u0[0], 58);
    assert_eq!(GLOBAL_COL_MAP.cumulative.z[10], 145);
    assert_eq!(GLOBAL_COL_MAP.quotient.map[0], 146);
    assert_eq!(GLOBAL_COL_MAP.quotient.output_z[9], 227);

    assert_eq!(GLOBAL_INTERACTION_DESCRIPTORS.len(), 10);
    assert_eq!(GLOBAL_INTERACTION_DESCRIPTORS[0].direction, LookupDirection::Receive);
    assert!(GLOBAL_INTERACTION_DESCRIPTORS[1..8]
        .iter()
        .all(|descriptor| descriptor.direction == LookupDirection::Send));
    assert_eq!(GLOBAL_INTERACTION_DESCRIPTORS[8].direction, LookupDirection::Receive);
    assert_eq!(GLOBAL_INTERACTION_DESCRIPTORS[9].direction, LookupDirection::Send);

    assert_eq!(prepared.raw_rows, 6);
    assert_eq!(prepared.trace.main.width(), NUM_GLOBAL_COLS);
    assert_eq!(prepared.trace.main.height(), prepared.raw_rows);
    assert_eq!(prepared.trace.total_height, 8);
    assert_eq!(prepared.reducer_trace.main.width(), NUM_GLOBAL_TILE_REDUCER_COLS);
    assert_eq!(prepared.reducer_trace.main.height(), 22);
    assert_eq!(prepared.reducer_trace.total_height, 32);
    assert_eq!(prepared.log_height, 3);
    assert_eq!(prepared.byte_delta.values().sum::<usize>(), 7 * prepared.raw_rows);

    let rows = prepared.trace.main.values.chunks_exact(NUM_GLOBAL_COLS).collect::<Vec<_>>();
    assert!(rows.iter().all(|row| global_relation_accepts_for_test(row)));
    let first: &GlobalCols<KoalaBear> = rows[0].borrow();
    assert_eq!(first.input.x, *identity.x.coefficients());
    assert_eq!(first.input.y, *identity.y.coefficients());
    assert_eq!(first.input.z, *identity.z.coefficients());
    assert_eq!(first.index, KoalaBear::zero());
    assert_eq!(first.is_receive, KoalaBear::zero());
    assert_eq!(first.is_real, KoalaBear::one());

    for (index, pair) in rows.windows(2).enumerate() {
        let current: &GlobalCols<KoalaBear> = pair[0].borrow();
        let next: &GlobalCols<KoalaBear> = pair[1].borrow();
        assert_eq!(current.index, KoalaBear::from_canonical_usize(index));
        assert_eq!(current.cumulative.x, next.input.x);
        assert_eq!(current.cumulative.y, next.input.y);
        assert_eq!(current.cumulative.z, next.input.z);
        assert!(
            [
                &current.products.u0,
                &current.products.u1,
                &current.products.u3,
                &current.products.u4,
                &current.products.u5,
            ]
            .into_iter()
            .flatten()
            .any(|value| !value.is_zero()),
        );
    }
    let last: &GlobalCols<KoalaBear> = (*rows.last().unwrap()).borrow();
    assert_eq!(last.index, KoalaBear::from_canonical_usize(prepared.raw_rows - 1));
    let raw_terminal = &last.cumulative;

    let PaddingRow::General(padding) = &prepared.trace.padding_row else {
        panic!("Global trace must retain one legal general padding row");
    };
    assert!(global_relation_accepts_for_test(padding));
    let padding: &GlobalCols<KoalaBear> = padding.as_slice().borrow();
    assert_eq!(padding.is_real, KoalaBear::zero());
    assert_eq!(padding.is_receive, KoalaBear::zero());
    assert_eq!(padding.index, KoalaBear::from_canonical_usize(prepared.raw_rows));
    assert_eq!(padding.input.x, raw_terminal.x);
    assert_eq!(padding.input.y, raw_terminal.y);
    assert_eq!(padding.input.z, raw_terminal.z);
    assert_eq!(padding.cumulative.x, padding.input.x);
    assert_eq!(padding.cumulative.y, padding.input.y);
    assert_eq!(padding.cumulative.z, padding.input.z);
    assert!(padding.quotient.map.iter().any(|value| !value.is_zero()));
    assert!([
        padding.quotient.u0.as_slice(),
        padding.quotient.u1.as_slice(),
        padding.quotient.u3.as_slice(),
        padding.quotient.u4.as_slice(),
        padding.quotient.u5.as_slice(),
    ]
    .into_iter()
    .flatten()
    .any(|value| !value.is_zero()));
    assert!(padding.quotient.output_x.iter().all(Field::is_zero));
    assert!(padding.quotient.output_y.iter().all(Field::is_zero));
    assert!(padding.quotient.output_z.iter().all(Field::is_zero));

    let claim = prepared.claim();
    assert_eq!(claim.has_global_opening, KoalaBear::one());
    assert_eq!(claim.count, KoalaBear::from_canonical_usize(prepared.raw_rows));
    assert_eq!(claim.interval.start.x, *identity.x.coefficients());
    assert_eq!(claim.interval.end.x, *prepared.terminal.x.coefficients());
    assert_eq!(prepared.extracted_claim().unwrap(), Some(claim));

    let reducer_rows = prepared
        .reducer_trace
        .main
        .values
        .chunks_exact(NUM_GLOBAL_TILE_REDUCER_COLS)
        .collect::<Vec<_>>();
    assert_eq!(reducer_rows.len(), 22);
    let leaf: &GlobalTileReducerCols<KoalaBear> = reducer_rows[0].borrow();
    assert_eq!(leaf.mode_leaf, KoalaBear::one());
    assert_eq!(leaf.payload.control[0], KoalaBear::from_canonical_usize(prepared.raw_rows));
    assert_eq!(leaf.payload.control[1], KoalaBear::one());
    assert_eq!(leaf.payload.control[4], KoalaBear::one());
    assert_eq!(leaf.payload.control[5], KoalaBear::one());
    assert_eq!(leaf.control_rank, KoalaBear::one());
    assert_eq!(leaf.control_next_rank, KoalaBear::from_canonical_u32(2));
    assert_eq!(leaf.control_next_tag, KoalaBear::from_canonical_u32(8));
    let rebase: &GlobalTileReducerCols<KoalaBear> = reducer_rows[1].borrow();
    assert_eq!(rebase.selector_spare, KoalaBear::one());
    assert_eq!(rebase.payload.values[..11], claim.interval.start.x);
    assert_eq!(rebase.control_rank, KoalaBear::from_canonical_u32(2));
    let rebase_output: &GlobalTileReducerCols<KoalaBear> = reducer_rows[15].borrow();
    assert_eq!(rebase_output.control_next_rank, KoalaBear::from_canonical_u32(4));
    assert_eq!(rebase_output.control_next_tag, KoalaBear::from_canonical_u32(12));
    let root_input: &GlobalTileReducerCols<KoalaBear> = reducer_rows[16].borrow();
    assert_eq!(root_input.control_rank, KoalaBear::from_canonical_u32(4));
    assert_eq!(root_input.control_next_rank, KoalaBear::from_canonical_u32(6));
    assert_eq!(root_input.control_next_tag, KoalaBear::from_canonical_u32(16));
    let root: &GlobalTileReducerCols<KoalaBear> = reducer_rows[21].borrow();
    assert_eq!(root.mode_root_output, KoalaBear::one());
    assert_eq!(root.payload.control[0], KoalaBear::from_canonical_usize(prepared.raw_rows));
    assert_eq!(root.control_rank, KoalaBear::from_canonical_u32(6));
    assert_eq!(root.control_next_rank, KoalaBear::one());
    assert_eq!(root.control_next_tag, KoalaBear::zero());
    assert_eq!(GLOBAL_TILE_REDUCER_COL_MAP.payload.values[11], 25);
    assert_eq!(root.payload.values[11], claim.interval.end.x[0]);
    assert!(matches!(prepared.reducer_trace.padding_row, PaddingRow::Zero { width: 83 }));

    let mut dependencies = ExecutionRecord::default();
    let retained = prepared.consume_byte_delta(&mut dependencies);
    assert_eq!(dependencies.byte_lookups.values().sum::<usize>(), 7 * retained.raw_rows);
    assert_eq!(retained.extracted_claim().unwrap(), Some(claim));
}

#[test]
fn p8_partial_tree_uses_dummy_leaves_and_raw_last_tile_padding() {
    let mut record = ExecutionRecord::default();
    record.global_memory_initialize_events = (0..1025u32)
        .map(|index| MemoryInitializeFinalizeEvent::initialize(index * 4, index + 1))
        .collect();

    let prepared = prepare_global_trace(&record).unwrap();
    assert_eq!(prepared.raw_rows, 1025);
    assert_eq!(prepared.reducer_trace.main.height(), 70);
    assert_eq!(prepared.reducer_trace.total_height, 128);

    let identity = D11ProjectivePointV1::<KoalaBear>::identity();
    let rows = prepared.trace.main.values.chunks_exact(NUM_GLOBAL_COLS).collect::<Vec<_>>();
    let first: &GlobalCols<KoalaBear> = rows[0].borrow();
    let second_tile: &GlobalCols<KoalaBear> = rows[512].borrow();
    let third_tile: &GlobalCols<KoalaBear> = rows[1024].borrow();
    for row in [first, second_tile, third_tile] {
        assert_eq!(row.input.x, *identity.x.coefficients());
        assert_eq!(row.input.y, *identity.y.coefficients());
        assert_eq!(row.input.z, *identity.z.coefficients());
    }

    let reducer_rows = prepared
        .reducer_trace
        .main
        .values
        .chunks_exact(NUM_GLOBAL_TILE_REDUCER_COLS)
        .collect::<Vec<_>>();
    for (ordinal, terminal_row) in [511usize, 1023, 1024].into_iter().enumerate() {
        let leaf: &GlobalTileReducerCols<KoalaBear> = reducer_rows[ordinal].borrow();
        let terminal: &GlobalCols<KoalaBear> = rows[terminal_row].borrow();
        assert_eq!(leaf.mode_leaf, KoalaBear::one());
        assert_eq!(leaf.payload.control[2], KoalaBear::from_canonical_usize(ordinal));
        assert_eq!(leaf.payload.control[4], KoalaBear::one());
        assert_eq!(leaf.control_rank, KoalaBear::from_canonical_usize(2 * ordinal + 1));
        assert_eq!(&leaf.payload.values[0..11], terminal.cumulative.x.as_slice());
        assert_eq!(&leaf.payload.values[11..22], terminal.cumulative.y.as_slice());
        assert_eq!(&leaf.payload.values[22..33], terminal.cumulative.z.as_slice());
    }
    let dummy: &GlobalTileReducerCols<KoalaBear> = reducer_rows[3].borrow();
    assert_eq!(dummy.mode_leaf, KoalaBear::one());
    assert_eq!(dummy.payload.control[4], KoalaBear::zero());
    assert_eq!(dummy.control_rank, KoalaBear::from_canonical_u32(6));
    assert_eq!(dummy.control_next_rank, KoalaBear::from_canonical_u32(8));
    assert_eq!(dummy.control_next_tag, KoalaBear::from_canonical_u32(4));
    assert_eq!(&dummy.payload.values[0..11], identity.x.coefficients());
    assert_eq!(&dummy.payload.values[11..22], identity.y.coefficients());
    assert_eq!(&dummy.payload.values[22..33], identity.z.coefficients());

    let rebase: &GlobalTileReducerCols<KoalaBear> = reducer_rows[49].borrow();
    assert_eq!(rebase.selector_spare, KoalaBear::one());
    assert_eq!(rebase.control_rank, KoalaBear::from_canonical_u32(14));
    let root: &GlobalTileReducerCols<KoalaBear> = reducer_rows[69].borrow();
    assert_eq!(root.mode_root_output, KoalaBear::one());
    assert_eq!(root.control_rank, KoalaBear::from_canonical_u32(18));
    assert_eq!(root.control_next_rank, KoalaBear::one());

    let PaddingRow::General(padding) = &prepared.trace.padding_row else {
        panic!("partial Global trace must retain its honest padding row");
    };
    let padding: &GlobalCols<KoalaBear> = padding.as_slice().borrow();
    let raw_last: &GlobalCols<KoalaBear> = rows[1024].borrow();
    assert_eq!(padding.input.x, raw_last.cumulative.x);
    assert_eq!(padding.input.y, raw_last.cumulative.y);
    assert_eq!(padding.input.z, raw_last.cumulative.z);
    assert_eq!(prepared.claim().interval.end.x.as_slice(), &root.payload.values[11..22]);
}
