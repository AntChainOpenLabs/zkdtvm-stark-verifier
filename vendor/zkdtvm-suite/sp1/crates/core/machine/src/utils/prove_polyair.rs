use std::{
    fs::File,
    io::{Seek, SeekFrom},
    sync::{
        mpsc::{channel, sync_channel, SendError, Sender, SyncSender},
        Arc, Mutex,
    },
    thread::ScopedJoinHandle,
};
use web_time::Instant;

use crate::{
    io::DTStdin, riscv::riscv_polyair::RiscvPolyAir, shape::Shapeable,
    utils::test::MaliciousTracePVGeneratorType,
};
use dt_stark::{
    air::{GlobalClaim, GlobalState, GlobalStateInterval, MachineAir, PolyAirExtendable, PublicValues},
    global_d11::{empty_global_claim, program_global_seed, D11ProjectivePointV1},
    shape::OrderedShape,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        keys::{SCMachineProvingKey, SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
    },
    DTCoreOpts, MachineRecord, Val,
};
use p3_maybe_rayon::prelude::*;
use p3_koala_bear::KoalaBear;
use polyair::prover::SCMachineProver;

use p3_field::PrimeField32;

use super::prove::DTCoreProverError;
use crate::utils::{chunk_vec, concurrency::TurnBasedSync};
use dt_core_executor::{
    events::{format_table_line, sorted_table_lines},
    ExecutionState,
};

use dt_core_executor::{
    subproof::NoOpSubproofVerifier, DTContext, ExecutionRecord, ExecutionReport, Executor, Program,
};

fn global_claim_to_canonical_u32<F: PrimeField32>(claim: GlobalClaim<F>) -> GlobalClaim<u32> {
    let state = |state: GlobalState<F>| GlobalState {
        x: state.x.map(|value| value.as_canonical_u32()),
        y: state.y.map(|value| value.as_canonical_u32()),
        z: state.z.map(|value| value.as_canonical_u32()),
    };
    GlobalClaim {
        has_global_opening: claim.has_global_opening.as_canonical_u32(),
        count: claim.count.as_canonical_u32(),
        interval: GlobalStateInterval {
            start: state(claim.interval.start),
            end: state(claim.interval.end),
        },
    }
}

/// PolyAir sumcheck: all rounds are linear (one variable per round).
pub const POLYAIR_NUM_SKIP_ROUNDS: usize = 1;
/// PolyAir sumcheck: no height threshold split — all chips use linear rounds.
pub const POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD: usize = 0;

// sc_prove_core
#[allow(clippy::too_many_arguments)]
pub fn sc_prove_core<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<SC, RiscvPolyAir<SC::Val>, D>,
    const D: usize,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    _: &SCStarkVerifyingKey<SC>,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>,
) -> Result<(SCMachineProof<SC>, Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32 + PolyAirExtendable<D>,
    SC::MlChallenger: 'static + Clone + Send,
    MlPcsOpeningProof<SC>: Send,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync,
{
    let (proof_tx, proof_rx) = channel();
    let (shape_tx, shape_rx) = channel();
    let (public_values, cycles) = sc_prove_core_stream(
        prover,
        pk,
        program,
        stdin,
        opts,
        context,
        proof_tx,
        shape_tx,
        None,
        malicious_trace_pv_generator,
    )?;

    let _: Vec<_> = shape_rx.iter().collect();
    let shard_proofs: Vec<SCShardProof<SC>> = proof_rx.iter().collect();
    let proof = SCMachineProof { shard_proofs };

    Ok((proof, public_values, cycles))
}
enum CoreProofSender<T> {
    Unbounded(Sender<T>),
    Bounded(SyncSender<T>),
}

impl<T> CoreProofSender<T> {
    fn send(&self, value: T) -> Result<(), SendError<T>> {
        match self {
            Self::Unbounded(sender) => sender.send(value),
            Self::Bounded(sender) => sender.send(value),
        }
    }
}

// sc_prove_core_stream
#[allow(clippy::too_many_arguments)]
pub fn sc_prove_core_stream<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<SC, RiscvPolyAir<SC::Val>, D>,
    const D: usize,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    proof_tx: Sender<SCShardProof<SC>>,
    shape_and_done_tx: Sender<(OrderedShape, bool)>,
    count_ticket_tx: Option<Sender<u32>>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>,
) -> Result<(Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32 + PolyAirExtendable<D>,
    SC::MlChallenger: 'static + Clone + Send,
    MlPcsOpeningProof<SC>: Send,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync,
{
    sc_prove_core_stream_with_sender(
        prover,
        pk,
        program,
        stdin,
        opts,
        context,
        CoreProofSender::Unbounded(proof_tx),
        shape_and_done_tx,
        count_ticket_tx,
        malicious_trace_pv_generator,
    )
}

/// Proves core shards while applying bounded backpressure to the proof consumer.
///
/// This has the same proof ordering and semantics as [`sc_prove_core_stream`]; only the proof
/// transport differs. The shape-and-done stream remains unbounded.
#[allow(clippy::too_many_arguments)]
pub fn sc_prove_core_stream_bounded<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<SC, RiscvPolyAir<SC::Val>, D>,
    const D: usize,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    proof_tx: SyncSender<SCShardProof<SC>>,
    shape_and_done_tx: Sender<(OrderedShape, bool)>,
    count_ticket_tx: Option<Sender<u32>>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>,
) -> Result<(Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32 + PolyAirExtendable<D>,
    SC::MlChallenger: 'static + Clone + Send,
    MlPcsOpeningProof<SC>: Send,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync,
{
    sc_prove_core_stream_with_sender(
        prover,
        pk,
        program,
        stdin,
        opts,
        context,
        CoreProofSender::Bounded(proof_tx),
        shape_and_done_tx,
        count_ticket_tx,
        malicious_trace_pv_generator,
    )
}

#[allow(clippy::too_many_arguments)]
fn sc_prove_core_stream_with_sender<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<SC, RiscvPolyAir<SC::Val>, D>,
    const D: usize,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    proof_tx: CoreProofSender<SCShardProof<SC>>,
    shape_and_done_tx: Sender<(OrderedShape, bool)>,
    count_ticket_tx: Option<Sender<u32>>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>,
) -> Result<(Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32 + PolyAirExtendable<D>,
    SC::MlChallenger: 'static + Clone + Send,
    MlPcsOpeningProof<SC>: Send,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync,
{
    let mut runtime = Box::new(Executor::with_context(program.clone(), opts, context));
    runtime.write_vecs(&stdin.buffer);
    for proof in stdin.proofs.iter() {
        let (proof, vk) = proof.clone();
        runtime.write_proof(proof, vk);
    }
    #[cfg(feature = "debug")]
    let (all_records_tx, all_records_rx) = std::sync::mpsc::channel::<Vec<ExecutionRecord>>();

    let malicious_trace_pv_generator: Option<&MaliciousTracePVGeneratorType<SC::Val, P>> =
        malicious_trace_pv_generator.as_ref();

    let span = tracing::Span::current().clone();
    std::thread::scope(move |s| {
        let _span = span.enter();

        let checkpoint_generator_span = tracing::Span::current().clone();
        let (checkpoints_tx, checkpoints_rx) =
            sync_channel::<(usize, File, bool, u64)>(opts.checkpoints_channel_capacity);
        let checkpoint_generator_handle: ScopedJoinHandle<Result<_, DTCoreProverError>> =
            s.spawn(move || {
                let _span = checkpoint_generator_span.enter();
                tracing::debug_span!("checkpoint generator").in_scope(|| {
                    let mut index = 0;
                    loop {
                        let span = tracing::debug_span!("batch");
                        let _span = span.enter();

                        let (checkpoint, _, done) = runtime
                            .execute_state(false)
                            .map_err(DTCoreProverError::ExecutionError)?;

                        let mut checkpoint_file =
                            tempfile::tempfile().map_err(DTCoreProverError::IoError)?;
                        checkpoint
                            .save(&mut checkpoint_file)
                            .map_err(DTCoreProverError::IoError)?;

                        checkpoints_tx
                            .send((index, checkpoint_file, done, runtime.state.global_clk))
                            .unwrap();

                        if done {
                            break Ok(runtime);
                        }

                        index += 1;
                    }
                })
            });

        let mut challenger = prover.config().mlchallenger();
        pk.observe_into(&mut challenger);

        let p2_record_gen_sync = Arc::new(TurnBasedSync::new());
        let p2_trace_gen_sync = Arc::new(TurnBasedSync::new());
        let (p2_records_and_traces_tx, p2_records_and_traces_rx) =
            sync_channel::<(
                Vec<ExecutionRecord>,
                Vec<Vec<(String, dt_stark::sumcheck::trace::CompressedMatrix<Val<SC>>)>>,
            )>(opts.records_and_traces_channel_capacity);
        let p2_records_and_traces_tx = Arc::new(Mutex::new(p2_records_and_traces_tx));

        let shape_tx = Arc::new(Mutex::new(shape_and_done_tx));
        let report_aggregate = Arc::new(Mutex::new(ExecutionReport::default()));
        let state = Arc::new(Mutex::new(PublicValues::<u32, u32>::default().reset()));
        let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone().into())));
        let global_state = Arc::new(Mutex::new(
            program_global_seed::<KoalaBear>(
                program
                    .prepared_global_program()
                    .expect("Global program boundary preparation failed")
                    .initial_boundary(),
            )
            .expect("Global program boundary must be canonical"),
        ));
        let mut p2_record_and_trace_gen_handles = Vec::new();
        let checkpoints_rx = Arc::new(Mutex::new(checkpoints_rx));
        for _ in 0..opts.trace_gen_workers {
            let record_gen_sync = Arc::clone(&p2_record_gen_sync);
            let trace_gen_sync = Arc::clone(&p2_trace_gen_sync);
            let records_and_traces_tx = Arc::clone(&p2_records_and_traces_tx);
            let checkpoints_rx = Arc::clone(&checkpoints_rx);
            let count_ticket_tx = count_ticket_tx.clone();

            let shape_tx = Arc::clone(&shape_tx);
            let report_aggregate = Arc::clone(&report_aggregate);
            let state = Arc::clone(&state);
            let deferred = Arc::clone(&deferred);
            let global_state = Arc::clone(&global_state);
            let program = program.clone();
            let span = tracing::Span::current().clone();

            #[cfg(feature = "debug")]
            let all_records_tx = all_records_tx.clone();

            let handle: ScopedJoinHandle<Result<(), DTCoreProverError>> = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("phase 2 trace generation").in_scope(|| loop {
                    let received = { checkpoints_rx.lock().unwrap().recv() };
                    if let Ok((index, mut checkpoint, done, num_cycles)) = received {
                        let (mut records, report) = tracing::debug_span!("trace checkpoint")
                            .in_scope(|| trace_checkpoint(program.clone(), &checkpoint, opts));

                        *report_aggregate.lock().unwrap() += report;
                        checkpoint
                            .seek(SeekFrom::Start(0))
                            .expect("failed to seek to start of tempfile");

                        record_gen_sync.wait_for_turn(index);

                        let mut state = state.lock().unwrap();
                        for record in records.iter_mut() {
                            state.shard += 1;
                            state.execution_shard = record.public_values.execution_shard;
                            state.start_pc = record.public_values.start_pc;
                            state.next_pc = record.public_values.next_pc;
                            state.start_clk = record.public_values.start_clk;
                            state.exit_clk = record.public_values.exit_clk;
                            if state.committed_value_digest == [0u32; 8] {
                                state.committed_value_digest =
                                    record.public_values.committed_value_digest;
                            }
                            if state.deferred_proofs_digest == [0u32; 8] {
                                state.deferred_proofs_digest =
                                    record.public_values.deferred_proofs_digest;
                            }
                            record.public_values = *state;
                        }

                        let mut deferred = deferred.lock().unwrap();
                        for record in records.iter_mut() {
                            deferred.append(&mut record.defer());
                        }

                        let combine_terminal_deferred = done &&
                            num_cycles < 1 << 21 &&
                            deferred.global_memory_initialize_events.len() <
                                opts.split_opts.combine_memory_threshold &&
                            deferred.global_memory_finalize_events.len() <
                                opts.split_opts.combine_memory_threshold;

                        let last_record =
                            if combine_terminal_deferred { records.last_mut() } else { None };
                        let mut deferred_records =
                            deferred.split(done, last_record, opts.split_opts);
                        tracing::debug!("deferred {} records", deferred_records.len());

                        if !done {
                            state.execution_shard += 1;
                        }
                        for record in deferred_records.iter_mut() {
                            state.shard += 1;
                            state.previous_init_addr = record.public_values.previous_init_addr;
                            state.last_init_addr = record.public_values.last_init_addr;
                            state.previous_finalize_addr =
                                record.public_values.previous_finalize_addr;
                            state.last_finalize_addr = record.public_values.last_finalize_addr;
                            state.start_pc = state.next_pc;
                            state.start_clk = 0;
                            state.exit_clk = 0;
                            record.public_values = *state;
                        }
                        records.append(&mut deferred_records);

                        let terminal_shard_count = state.shard;
                        drop(deferred);
                        drop(state);

                        tracing::debug_span!("generate dependencies", index).in_scope(|| {
                            for record in records.iter_mut() {
                                if record.global_endpoint_count() != 0 &&
                                    !record.has_global_trace_artifact()
                                {
                                    let start = *global_state.lock().unwrap();
                                    let prepared = crate::global::prepare_global_trace_for_field_from_start::<
                                        KoalaBear,
                                    >(record, start)
                                    .expect("ordered Global trace preparation failed");
                                    let terminal: D11ProjectivePointV1<KoalaBear> = prepared.terminal;
                                    let retained = prepared.consume_byte_delta(record);
                                    record.install_global_trace_artifact(
                                        retained.trace,
                                        retained.reducer_trace,
                                    );
                                    *global_state.lock().unwrap() = terminal;
                                }
                                let mut saw_global = false;
                                let mut saw_global_reducer = false;
                                for chip in prover.machine().chips() {
                                    if chip.included(record) {
                                        saw_global |= matches!(&chip.air, RiscvPolyAir::Global(_));
                                        saw_global_reducer |= matches!(
                                            &chip.air,
                                            RiscvPolyAir::GlobalTileReducer(_)
                                        );
                                    }
                                    let mut output = ExecutionRecord::default();
                                    chip.generate_dependencies(record, &mut output);
                                    record.append(&mut output);
                                }
                                assert!(
                                    (saw_global && saw_global_reducer) ||
                                        record.global_endpoint_count() == 0,
                                    "Global main/link inclusion must be paired"
                                );
                                tracing::debug_span!("register nonces")
                                    .in_scope(|| record.register_nonces(&opts));
                            }
                        });

                        record_gen_sync.advance_turn();

                        // The terminal checkpoint's records and every deferred
                        // shard are now appended and counted: `state.shard` is
                        // the request's exact final shard count. This is the
                        // count-ticket finalization event (the last split/
                        // deferred decision that can change the count) — it
                        // fires on its own control channel, before this
                        // checkpoint's traces or proofs exist.
                        if done {
                            if let Some(count_ticket_tx) = &count_ticket_tx {
                                let _ = count_ticket_tx.send(terminal_shard_count);
                            }
                        }

                        for record in records.iter() {
                            let mut heights = vec![];
                            let chips = prover.shard_chips(record).collect::<Vec<_>>();
                            if let Some(shape) = record.shape.as_ref() {
                                for chip in chips.iter() {
                                    let id = chip.air.id();
                                    let height = shape.log2_height(&id).unwrap();
                                    heights.push((chip.name().clone(), height));
                                }
                                shape_tx
                                    .lock()
                                    .unwrap()
                                    .send((OrderedShape::from_log2_heights(&heights), done))
                                    .unwrap();
                            }
                        }

                        let mut main_traces = Vec::new();
                        if let Some(malicious_trace_pv_generator) = malicious_trace_pv_generator {
                            tracing::info_span!("generate main traces", index).in_scope(|| {
                                main_traces = records
                                    .par_iter_mut()
                                    .map(|record| malicious_trace_pv_generator(prover, record))
                                    .collect::<Vec<_>>();
                            });
                        } else {
                            tracing::info_span!("generate main traces", index).in_scope(|| {
                                main_traces = records
                                    .par_iter()
                                    .map(|record| prover.generate_traces(record))
                                    .collect::<Vec<_>>();
                            });
                        }

                        assert_eq!(records.len(), main_traces.len(), "record/trace batch mismatch");
                        for (record, traces) in records.iter_mut().zip(main_traces.iter()) {
                            let mut reducer_traces = traces
                                .iter()
                                .filter(|(name, _)| name == "GlobalTileReducerV3")
                                .map(|(_, trace)| trace);
                            let claim = if let Some(trace) = reducer_traces.next() {
                                crate::global::claim_from_compressed_tile_reducer_trace(trace)
                                    .expect("canonical GlobalTileReducerV3 claim extraction failed")
                                    .expect("GlobalTileReducerV3 trace produced no claim")
                            } else {
                                empty_global_claim::<Val<SC>>()
                            };
                            assert!(
                                reducer_traces.next().is_none(),
                                "duplicate GlobalTileReducerV3 trace"
                            );
                            record.public_values.global = global_claim_to_canonical_u32(claim);
                        }

                        #[cfg(feature = "debug")]
                        all_records_tx.send(records.clone()).unwrap();

                        trace_gen_sync.wait_for_turn(index);

                        let chunked_records = chunk_vec(records, opts.shard_batch_size);
                        let chunked_main_traces = chunk_vec(main_traces, opts.shard_batch_size);
                        chunked_records.into_iter().zip(chunked_main_traces.into_iter()).for_each(
                            |(records, main_traces)| {
                                records_and_traces_tx
                                    .lock()
                                    .unwrap()
                                    .send((records, main_traces))
                                    .unwrap();
                            },
                        );

                        trace_gen_sync.advance_turn();
                    } else {
                        break Ok(());
                    }
                })
            });
            p2_record_and_trace_gen_handles.push(handle);
        }
        drop(p2_records_and_traces_tx);
        #[cfg(feature = "debug")]
        drop(all_records_tx);

        let p2_prover_span = tracing::Span::current();
        let proof_tx = Arc::new(Mutex::new(proof_tx));
        let proving_start = Instant::now();
        let p2_prover_handle = s.spawn(move || {
            let _span = p2_prover_span.enter();
            tracing::debug_span!("phase 2 prover").in_scope(|| {
                for (records, traces) in p2_records_and_traces_rx.into_iter() {
                    tracing::debug_span!("batch").in_scope(|| {
                        let span = tracing::Span::current().clone();
                        // GPU 单卡:多 shard 并行 prove 会让多个 GPU prover 抢同一张卡
                        // (竞态共享 stream/mem_pool/chip_ctx_cache/全局 twiddle)导致
                        // "chip not in cache" / cudaErrorInvalidValue。CPU 版本可并行,
                        // 但新版 GPU prover 共用单卡,强制串行(commit/open 内部仍有并行)。
                        let proofs = records
                            .into_iter()
                            .zip(traces.into_iter())
                            .map(|(record, main_traces)| {
                                let _span = span.enter();

                                let shard = record.shard();
                                let before = Instant::now();
                                let commit_start = Instant::now();
                                let main_data = tracing::debug_span!("commit", shard).in_scope(|| {
                                    prover.commit_with_pcs_stack_log_height(
                                        &record,
                                        main_traces,
                                        pk.preprocessed_pcs_stack_log_height(),
                                    )
                                });
                                let commit_elapsed = commit_start.elapsed();
                                let open_start = Instant::now();
                                let proof = tracing::debug_span!("opening", shard).in_scope(|| {
                                    prover
                                        .open(
                                            pk,
                                            main_data,
                                            &mut challenger.clone(),
                                            POLYAIR_NUM_SKIP_ROUNDS,
                                            POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD,
                                        )
                                        .unwrap()
                                });
                                let open_elapsed = open_start.elapsed();

                                let elapsed = before.elapsed();

                                let mut total_cells: u64 = 0;
                                let mut chip_details = Vec::new();
                                if let Some(shape) = record.shape.as_ref() {
                                    for (&air_id, &log2_h) in shape.iter() {
                                        if log2_h > 0 {
                                            let h = 1u64 << log2_h;
                                            chip_details.push((air_id, log2_h, h));
                                            total_cells += h;
                                        }
                                    }
                                }

                                tracing::info!(
                                    "SHARD_PROVE: shard={} commit={}ms open={}ms total={}ms cells={} chips={}",
                                    shard,
                                    commit_elapsed.as_millis(),
                                    open_elapsed.as_millis(),
                                    elapsed.as_millis(),
                                    total_cells,
                                    chip_details.len(),
                                );
                                if !chip_details.is_empty() {
                                    let shape_str: Vec<String> = chip_details.iter()
                                        .map(|(air_id, log2_h, _h)| format!("{air_id:?}={log2_h}"))
                                        .collect();
                                    tracing::info!(
                                        "SHARD_SHAPE: shard={} {}",
                                        shard,
                                        shape_str.join(" "),
                                    );
                                }

                                #[allow(unused_variables)]
                                #[cfg(debug_assertions)]
                                {
                                    if let Some(shape) = record.shape.as_ref() {
                                        let _ = shape;
                                    }
                                }

                                rayon::spawn(move || {
                                    drop(record);
                                });

                                proof
                            })
                            .collect::<Vec<_>>();

                        let proof_tx = proof_tx.lock().unwrap();
                        for proof in proofs {
                            proof_tx.send(proof).unwrap();
                        }
                    });
                }
            });
        });

        let runtime = checkpoint_generator_handle.join().unwrap().unwrap();
        let public_values_stream = runtime.state.public_values_stream;

        for handle in p2_record_and_trace_gen_handles {
            handle.join().unwrap()?;
        }

        p2_prover_handle.join().unwrap();
        let proving_end = proving_start.elapsed().as_secs_f64();
        let core_shard_count = state.lock().unwrap().shard;
        tracing::info!("Core proving time is {}", proving_end);
        tracing::info!("CORE_SHARD_COUNT count={}", core_shard_count);
        let report_aggregate = report_aggregate.lock().unwrap();
        tracing::debug!(
            "execution report (totals): total_cycles={}, total_syscall_cycles={}, touched_memory_addresses={}",
            report_aggregate.total_instruction_count(),
            report_aggregate.total_syscall_count(),
            report_aggregate.touched_memory_addresses,
        );
        tracing::debug!("execution report (opcode counts):");
        let (width, lines) = sorted_table_lines(report_aggregate.opcode_counts.as_ref());
        for (label, count) in lines {
            if *count > 0 {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            } else {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            }
        }

        tracing::debug!("execution report (syscall counts):");
        let (width, lines) = sorted_table_lines(report_aggregate.syscall_counts.as_ref());
        for (label, count) in lines {
            if *count > 0 {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            } else {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            }
        }

        let cycles = report_aggregate.total_instruction_count();

        let proving_time = proving_start.elapsed().as_secs_f64();
        tracing::debug!(
            "summary: cycles={}, e2e={}s, khz={:.2}",
            cycles,
            proving_time,
            (cycles as f64 / (proving_time * 1000.0) as f64),
        );

        #[cfg(feature = "debug")]
        {
            let all_records = all_records_rx.iter().flatten().collect::<Vec<_>>();
            let mut challenger = prover.machine().config().mlchallenger();
            let pk_host = prover.pk_to_host(pk);
            prover.machine().debug_constraints(&pk_host, all_records, &mut challenger);
        }

        Ok((public_values_stream, cycles))
    })
}
pub fn trace_checkpoint(
    program: Program,
    file: &File,
    opts: DTCoreOpts,
) -> (Vec<ExecutionRecord>, ExecutionReport) {
    let noop = NoOpSubproofVerifier;

    let mut reader = std::io::BufReader::new(file);
    let state: ExecutionState =
        bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Executor::recover(program, state, opts);

    runtime.subproof_verifier = Some(&noop);

    let (records, done) = runtime.execute_record(true).unwrap();

    let mut records = records.into_iter().map(|r| *r).collect::<Vec<_>>();
    let pv = records.last().unwrap().public_values;

    if !done &&
        (pv.committed_value_digest.iter().any(|v| *v != 0) ||
            pv.deferred_proofs_digest.iter().any(|v| *v != 0))
    {
        runtime.print_report = false;
        let (_, next_pv, _) = runtime.execute_state(true).unwrap();
        for record in records.iter_mut() {
            record.public_values.committed_value_digest = next_pv.committed_value_digest;
            record.public_values.deferred_proofs_digest = next_pv.deferred_proofs_digest;
        }
    }

    (records, runtime.report)
}
