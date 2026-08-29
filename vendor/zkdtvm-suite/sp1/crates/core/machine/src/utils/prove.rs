use std::{
    error::Error,
    fs::File,
    io::{self, Seek, SeekFrom},
    str::FromStr,
    sync::{
        mpsc::{channel, sync_channel, Sender},
        Arc, Mutex,
    },
    thread::ScopedJoinHandle,
};
use web_time::Instant;

use crate::{
    io::DTStdin,
    riscv::RiscvAir,
    shape::{
        chip_log_height_threshold, fix_shape_dynamic, num_skip_rounds, CoreShapeConfig,
        CoreShapeError, Shapeable,
    },
    utils::test::MaliciousTracePVGeneratorType,
};
use dt_stark::{
    air::{MachineAir, PublicValues},
    shape::OrderedShape,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        keys::{SCMachineProvingKey, SCStarkProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
        prover::SCMachineProver,
    },
    Challenge, DTCoreOpts, MachineRecord, StarkGenericConfig, Val,
};
use p3_maybe_rayon::prelude::*;
use thiserror::Error;

use p3_field::PrimeField32;

use crate::utils::{chunk_vec, concurrency::TurnBasedSync};
use dt_core_executor::{
    estimator::RecordEstimator,
    events::{format_table_line, sorted_table_lines},
    ExecutionState, RiscvAirId,
};

use dt_core_executor::{
    subproof::NoOpSubproofVerifier, DTContext, ExecutionError, ExecutionRecord, ExecutionReport,
    Executor, Program,
};

#[allow(clippy::too_many_arguments)]
pub fn sc_prove_core<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<
        SC,
        RiscvAir<SC::Val>,
        RiscvAir<Challenge<SC>>,
        DeviceProvingKey = SCStarkProvingKey<SC>,
    >,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    _: &SCStarkVerifyingKey<SC>,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    shape_config: Option<&CoreShapeConfig<SC::Val>>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>,
) -> Result<(SCMachineProof<SC>, Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32,
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
        shape_config,
        proof_tx,
        shape_tx,
        malicious_trace_pv_generator,
        None,
    )?;

    let _: Vec<_> = shape_rx.iter().collect();
    let shard_proofs: Vec<SCShardProof<SC>> = proof_rx.iter().collect();
    let proof = SCMachineProof { shard_proofs };

    Ok((proof, public_values, cycles))
}

#[allow(clippy::too_many_arguments)]
pub fn sc_prove_core_stream<
    SC: SCStarkGenericConfig,
    P: SCMachineProver<
        SC,
        RiscvAir<SC::Val>,
        RiscvAir<Challenge<SC>>,
        DeviceProvingKey = SCStarkProvingKey<SC>,
    >,
>(
    prover: &P,
    pk: &P::DeviceProvingKey,
    program: Program,
    stdin: &DTStdin,
    opts: DTCoreOpts,
    context: DTContext,
    shape_config: Option<&CoreShapeConfig<SC::Val>>,
    proof_tx: Sender<SCShardProof<SC>>,
    shape_and_done_tx: Sender<(OrderedShape, bool)>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<SC::Val, P>>, /* This is used for failure test cases that generate malicious traces and public values. */
    gas_calculator: Option<Box<dyn FnOnce(&RecordEstimator) -> Result<u64, Box<dyn Error>> + '_>>,
) -> Result<(Vec<u8>, u64), DTCoreProverError>
where
    SC::Val: PrimeField32,
    SC::MlChallenger: 'static + Clone + Send,
    MlPcsOpeningProof<SC>: Send,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync,
{
    // Setup the runtime.
    let mut runtime = Box::new(Executor::with_context(program.clone(), opts, context));
    runtime.maximal_shapes = shape_config.map(|config| {
        config.maximal_core_shapes(opts.shard_size.ilog2() as usize).into_iter().collect()
    });
    runtime.write_vecs(&stdin.buffer);
    for proof in stdin.proofs.iter() {
        let (proof, vk) = proof.clone();
        runtime.write_proof(proof, vk);
    }
    // Set the record estimator to collect data for gas calculation.
    if gas_calculator.is_some() {
        runtime.record_estimator = Some(Box::default());
    }

    #[cfg(feature = "debug")]
    let (all_records_tx, all_records_rx) = std::sync::mpsc::channel::<Vec<ExecutionRecord>>();

    // Need to create an optional reference, because of the `move` below.
    let malicious_trace_pv_generator: Option<&MaliciousTracePVGeneratorType<SC::Val, P>> =
        malicious_trace_pv_generator.as_ref();

    // Record the start of the process.

    let span = tracing::Span::current().clone();
    std::thread::scope(move |s| {
        let _span = span.enter();

        // Spawn the checkpoint generator thread.
        let checkpoint_generator_span = tracing::Span::current().clone();
        let (checkpoints_tx, checkpoints_rx) =
            sync_channel::<(usize, File, bool, u64)>(opts.checkpoints_channel_capacity);
        let checkpoint_generator_handle: ScopedJoinHandle<Result<_, DTCoreProverError>> =
            s.spawn(move || {
                let _span = checkpoint_generator_span.enter();
                tracing::debug_span!("checkpoint generator").in_scope(|| {
                    let mut index = 0;
                    loop {
                        // Enter the span.
                        let span = tracing::debug_span!("batch");
                        let _span = span.enter();

                        // Execute the runtime until we reach a checkpoint.
                        let (checkpoint, _, done) = runtime
                            .execute_state(false)
                            .map_err(DTCoreProverError::ExecutionError)?;

                        // Save the checkpoint to a temp file.
                        let mut checkpoint_file =
                            tempfile::tempfile().map_err(DTCoreProverError::IoError)?;
                        checkpoint
                            .save(&mut checkpoint_file)
                            .map_err(DTCoreProverError::IoError)?;

                        // Send the checkpoint.
                        checkpoints_tx
                            .send((index, checkpoint_file, done, runtime.state.global_clk))
                            .unwrap();

                        // If we've reached the final checkpoint, break out of the loop.
                        if done {
                            break Ok(runtime);
                        }

                        // Update the index.
                        index += 1;
                    }
                })
            });

        // Create the challenger and observe the verifying key.
        let mut challenger = prover.config().mlchallenger();
        pk.observe_into(&mut challenger);

        // Spawn the phase 2 record generator thread.
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
        let mut p2_record_and_trace_gen_handles = Vec::new();
        let checkpoints_rx = Arc::new(Mutex::new(checkpoints_rx));
        for _ in 0..opts.trace_gen_workers {
            let record_gen_sync = Arc::clone(&p2_record_gen_sync);
            let trace_gen_sync = Arc::clone(&p2_trace_gen_sync);
            let records_and_traces_tx = Arc::clone(&p2_records_and_traces_tx);
            let checkpoints_rx = Arc::clone(&checkpoints_rx);

            let shape_tx = Arc::clone(&shape_tx);
            let report_aggregate = Arc::clone(&report_aggregate);
            let state = Arc::clone(&state);
            let deferred = Arc::clone(&deferred);
            let program = program.clone();
            let span = tracing::Span::current().clone();

            #[cfg(feature = "debug")]
            let all_records_tx = all_records_tx.clone();

            let handle = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("phase 2 trace generation").in_scope(|| {
                    loop {
                        let received = { checkpoints_rx.lock().unwrap().recv() };
                        if let Ok((index, mut checkpoint, done, num_cycles)) = received {
                            let (mut records, report) = tracing::debug_span!("trace checkpoint")
                                .in_scope(|| {
                                    trace_checkpoint::<SC>(
                                        program.clone(),
                                        &checkpoint,
                                        opts,
                                        shape_config,
                                    )
                                });

                            // Trace the checkpoint and reconstruct the execution records.
                            *report_aggregate.lock().unwrap() += report;
                            checkpoint
                                .seek(SeekFrom::Start(0))
                                .expect("failed to seek to start of tempfile");

                            // Wait for our turn to update the state.
                            record_gen_sync.wait_for_turn(index);

                            // Update the public values & prover state for the shards which contain
                            // "cpu events".
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

                            // Defer events that are too expensive to include in every shard.
                            let mut deferred = deferred.lock().unwrap();
                            for record in records.iter_mut() {
                                deferred.append(&mut record.defer());
                            }

                            // We combine the memory init/finalize events if they are "small"
                            // and would affect performance.
                            let mut shape_fixed_records = if done
                                && num_cycles < 1 << 21
                                && deferred.global_memory_initialize_events.len()
                                    < opts.split_opts.combine_memory_threshold
                                && deferred.global_memory_finalize_events.len()
                                    < opts.split_opts.combine_memory_threshold
                            {
                                let mut records_clone = records.clone();
                                let last_record = records_clone.last_mut();
                                // See if any deferred shards are ready to be committed to.
                                let mut deferred =
                                    deferred.split(done, last_record, opts.split_opts);
                                tracing::debug!("deferred {} records", deferred.len());

                                // Update the public values & prover state for the shards which do
                                // not contain "cpu events" before
                                // committing to them.
                                if !done {
                                    state.execution_shard += 1;
                                }
                                for record in deferred.iter_mut() {
                                    state.shard += 1;
                                    state.previous_init_addr =
                                        record.public_values.previous_init_addr;
                                    state.last_init_addr = record.public_values.last_init_addr;
                                    state.previous_finalize_addr =
                                        record.public_values.previous_finalize_addr;
                                    state.last_finalize_addr =
                                        record.public_values.last_finalize_addr;
                                    state.start_pc = state.next_pc;
                                    // Deferred shards have no CPU events / State interactions.
                                    state.start_clk = 0;
                                    state.exit_clk = 0;
                                    record.public_values = *state;
                                }
                                records_clone.append(&mut deferred);

                                // Generate the dependencies.
                                tracing::debug_span!("generate dependencies", index).in_scope(
                                    || {
                                        prover.machine().generate_dependencies(
                                            &mut records_clone,
                                            &opts,
                                            None,
                                        );
                                    },
                                );

                                // Let another worker update the state.
                                record_gen_sync.advance_turn();

                                // Fix the shape of the records.
                                let mut fixed_shape = true;
                                if let Some(shape_config) = shape_config {
                                    for record in records_clone.iter_mut() {
                                        if shape_config.fix_shape(record).is_err() {
                                            fixed_shape = false;
                                        }
                                    }
                                }
                                // When shape_config is None, leave record.shape as None so that
                                // traces size themselves dynamically (no fixed log2 rows).
                                fixed_shape.then_some(records_clone)
                            } else {
                                None
                            };

                            if shape_fixed_records.is_none() {
                                // See if any deferred shards are ready to be committed to.
                                let mut deferred = deferred.split(done, None, opts.split_opts);
                                tracing::debug!("deferred {} records", deferred.len());

                                // Update the public values & prover state for the shards which do
                                // not contain "cpu events" before
                                // committing to them.
                                if !done {
                                    state.execution_shard += 1;
                                }
                                for record in deferred.iter_mut() {
                                    state.shard += 1;
                                    state.previous_init_addr =
                                        record.public_values.previous_init_addr;
                                    state.last_init_addr = record.public_values.last_init_addr;
                                    state.previous_finalize_addr =
                                        record.public_values.previous_finalize_addr;
                                    state.last_finalize_addr =
                                        record.public_values.last_finalize_addr;
                                    state.start_pc = state.next_pc;
                                    // Deferred shards have no CPU events / State interactions.
                                    state.start_clk = 0;
                                    state.exit_clk = 0;
                                    record.public_values = *state;
                                }
                                records.append(&mut deferred);

                                // Generate the dependencies.
                                tracing::debug_span!("generate dependencies", index).in_scope(
                                    || {
                                        prover.machine().generate_dependencies(
                                            &mut records,
                                            &opts,
                                            None,
                                        );
                                    },
                                );

                                // Let another worker update the state.
                                record_gen_sync.advance_turn();

                                // Fix the shape of the records.
                                if let Some(shape_config) = shape_config {
                                    for record in records.iter_mut() {
                                        match shape_config.fix_shape(record) {
                                            Ok(()) => {}
                                            Err(
                                                err @ CoreShapeError::ShapeCapacityExceeded {
                                                    ..
                                                },
                                            ) => {
                                                return Err(DTCoreProverError::ShapeError(err));
                                            }
                                            Err(_) => {
                                                tracing::warn!(
                                                    "predefined shape not found, using dynamic shapes"
                                                );
                                                record.shape = None;
                                                fix_shape_dynamic(record);
                                                shape_config
                                                    .ensure_shape_within_active_capacity(
                                                        record.shape.as_ref().unwrap(),
                                                    )
                                                    .map_err(DTCoreProverError::ShapeError)?;
                                            }
                                        }
                                    }
                                }
                                // When shape_config is None, leave record.shape as None so that
                                // traces size themselves dynamically.
                                shape_fixed_records = Some(records);
                            }

                            let mut records = shape_fixed_records.unwrap();

                            // Send the shapes to the channel, if necessary.
                            for record in records.iter() {
                                let mut heights = vec![];
                                let chips = prover.shard_chips(record).collect::<Vec<_>>();
                                if let Some(shape) = record.shape.as_ref() {
                                    for chip in chips.iter() {
                                        let id = RiscvAirId::from_str(&chip.name()).unwrap();
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

                            #[cfg(feature = "debug")]
                            all_records_tx.send(records.clone()).unwrap();

                            let mut main_traces = Vec::new();
                            if let Some(malicious_trace_pv_generator) = malicious_trace_pv_generator
                            {
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

                            trace_gen_sync.wait_for_turn(index);

                            // Send the records to the phase 2 prover.
                            let chunked_records = chunk_vec(records, opts.shard_batch_size);
                            let chunked_main_traces = chunk_vec(main_traces, opts.shard_batch_size);
                            chunked_records
                                .into_iter()
                                .zip(chunked_main_traces.into_iter())
                                .for_each(|(records, main_traces)| {
                                    records_and_traces_tx
                                        .lock()
                                        .unwrap()
                                        .send((records, main_traces))
                                        .unwrap();
                                });

                            trace_gen_sync.advance_turn();
                        } else {
                            break Ok(());
                        }
                    }
                })
            });
            p2_record_and_trace_gen_handles.push(handle);
        }
        drop(p2_records_and_traces_tx);
        #[cfg(feature = "debug")]
        drop(all_records_tx);

        // Spawn the phase 2 prover thread.
        // let p2_prover_span = tracing::Span::current().clone();

        let p2_prover_span = tracing::Span::current();
        let proof_tx = Arc::new(Mutex::new(proof_tx));
        let proving_start = Instant::now();
        let p2_prover_handle = s.spawn(move || {
            let _span = p2_prover_span.enter();
            tracing::debug_span!("phase 2 prover").in_scope(|| {
                for (records, traces) in p2_records_and_traces_rx.into_iter() {
                    tracing::debug_span!("batch").in_scope(|| {
                        let span = tracing::Span::current().clone();
                        let proofs = records
                            .into_par_iter()
                            .zip(traces.into_par_iter())
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
                                            num_skip_rounds(),
                                            chip_log_height_threshold(),
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

                                // SCShardProof does not expose shape(); shape check skipped for SC path.
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

                        // Send the batch of proofs to the channel.
                        let proof_tx = proof_tx.lock().unwrap();
                        for proof in proofs {
                            proof_tx.send(proof).unwrap();
                        }
                    });
                }
            });
        });

        // Wait until the checkpoint generator handle has fully finished.
        let runtime = checkpoint_generator_handle.join().unwrap().unwrap();
        let gas = gas_calculator.map(|calc| calc(runtime.record_estimator.as_ref().unwrap()));
        let public_values_stream = runtime.state.public_values_stream;

        // Wait until the records and traces have been fully generated for phase 2.
        for handle in p2_record_and_trace_gen_handles {
            handle.join().unwrap()?;
        }

        // Wait until the phase 2 prover has finished.
        p2_prover_handle.join().unwrap();
        let proving_end = proving_start.elapsed().as_secs_f64();
        let core_shard_count = state.lock().unwrap().shard;
        tracing::info!("Core proving time is {}", proving_end);
        tracing::info!("CORE_SHARD_COUNT count={}", core_shard_count);
        // Log some of the `ExecutionReport` information.
        let mut report_aggregate = report_aggregate.lock().unwrap();
        tracing::debug!(
            "execution report (totals): total_cycles={}, total_syscall_cycles={}, touched_memory_addresses={}",
            report_aggregate.total_instruction_count(),
            report_aggregate.total_syscall_count(),
            report_aggregate.touched_memory_addresses,
        );
        match gas {
            Some(Ok(gas)) => {
                tracing::debug!("execution report (gas): {}", gas);
                report_aggregate.gas = Some(gas);
            }
            Some(Err(err)) => tracing::error!("Encountered error while calculating gas: {}", err),
            None => (),
        }

        // Print the opcode and syscall count tables like `du`: sorted by count (descending) and
        // with the count in the first column.
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

        // Print the summary.
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

pub fn trace_checkpoint<SC: StarkGenericConfig>(
    program: Program,
    file: &File,
    opts: DTCoreOpts,
    shape_config: Option<&CoreShapeConfig<SC::Val>>,
) -> (Vec<ExecutionRecord>, ExecutionReport)
where
    <SC as StarkGenericConfig>::Val: PrimeField32,
{
    let noop = NoOpSubproofVerifier;

    let mut reader = std::io::BufReader::new(file);
    let state: ExecutionState =
        bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Executor::recover(program, state, opts);
    runtime.maximal_shapes = shape_config.map(|config| {
        config.maximal_core_shapes(opts.shard_size.ilog2() as usize).into_iter().collect()
    });

    // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
    // already verified. So here we use a noop verifier to not print any warnings.
    runtime.subproof_verifier = Some(&noop);

    // Execute from the checkpoint.
    let (records, done) = runtime.execute_record(true).unwrap();

    let mut records = records.into_iter().map(|r| *r).collect::<Vec<_>>();
    let pv = records.last().unwrap().public_values;

    // Handle the case where the COMMIT happens across the last two shards.
    if !done &&
        (pv.committed_value_digest.iter().any(|v| *v != 0) ||
            pv.deferred_proofs_digest.iter().any(|v| *v != 0))
    {
        // We turn off the `print_report` flag to avoid modifying the report.
        runtime.print_report = false;
        let (_, next_pv, _) = runtime.execute_state(true).unwrap();
        for record in records.iter_mut() {
            record.public_values.committed_value_digest = next_pv.committed_value_digest;
            record.public_values.deferred_proofs_digest = next_pv.deferred_proofs_digest;
        }
    }

    (records, runtime.report)
}

#[derive(Error, Debug)]
pub enum DTCoreProverError {
    #[error("failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("io error: {0}")]
    IoError(io::Error),
    #[error("serialization error: {0}")]
    SerializationError(bincode::Error),
    #[error("shape error: {0}")]
    ShapeError(CoreShapeError),
}
