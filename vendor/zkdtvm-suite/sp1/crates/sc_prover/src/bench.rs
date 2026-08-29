use anyhow::{anyhow, bail, Context, Result};
use dt_core_machine::shape::{chip_log_height_threshold, num_skip_rounds};
use dt_recursion_circuit::{sc_machine::SCDTRecursionWitnessValues, witness::Witnessable};
use dt_recursion_core::Runtime as RecursionRuntime;
use dt_stark::{
    air::MachineAir,
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCMachineProvingKey, SCStarkVerifyingKey},
        proof::{SCMachineProof, SCShardProof},
        trace::CompressedMatrix,
    },
    Challenge, DTProverOpts, Val, DIGEST_SIZE,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use polyair::prover::SCMachineProver as PolyAirMachineProver;
use serde::Serialize;
use std::time::Instant;

use dt_recursion_circuit::sc_machine::SCDTCompressWitnessValues;

use crate::{
    components::DTProverComponents, types::DTVerifyingKey, CoreSC, DTProver, InnerConfig, InnerSC,
    INNER_SBOX_DEGREE,
};

/// A lift benchmark that also captures the produced (vk, proof) pair so a join-node
/// benchmark can consume it as a child.
pub struct DTLiftCaptured {
    pub report: DTLiftBenchmarkReport,
    pub vk: SCStarkVerifyingKey<InnerSC>,
    pub proof: SCShardProof<InnerSC>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DTLiftBenchmarkTimings {
    pub input_build_ms: u128,
    pub program_compile_ms: u128,
    pub witness_write_ms: u128,
    pub runtime_ms: u128,
    pub dependencies_ms: u128,
    pub tracegen_ms: u128,
    pub setup_ms: u128,
    pub commit_ms: u128,
    pub open_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct DTTraceCost {
    pub chip: String,
    pub height: usize,
    pub stored_height: usize,
    pub width: usize,
    pub perm_width: usize,
    pub interactions: usize,
    pub constraints: usize,
    pub useful_cells: usize,
    pub padded_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DTLiftBenchmarkReport {
    pub timings: DTLiftBenchmarkTimings,
    pub proof_bytes: usize,
    pub proof_with_vk_bytes: usize,
    pub program_instructions: usize,
    pub trace_costs: Vec<DTTraceCost>,
    pub useful_cells: usize,
    pub padded_cells: usize,
    pub total_cells: usize,
    pub padding_to_useful_ratio: f64,
    pub total_to_useful_ratio: f64,
}

impl<C: DTProverComponents> DTProver<C> {
    pub fn benchmark_lift_core_node(
        &self,
        vk: &DTVerifyingKey,
        shard_proofs: &[dt_stark::sumcheck::proof::SCShardProof<CoreSC>],
        opts: DTProverOpts,
    ) -> Result<DTLiftBenchmarkReport> {
        self.benchmark_lift_core_node_with_flags(vk, shard_proofs, true, true, opts)
    }

    pub fn benchmark_lift_core_node_with_flags(
        &self,
        vk: &DTVerifyingKey,
        shard_proofs: &[dt_stark::sumcheck::proof::SCShardProof<CoreSC>],
        is_first_shard: bool,
        is_complete: bool,
        opts: DTProverOpts,
    ) -> Result<DTLiftBenchmarkReport> {
        self.benchmark_lift_core_node_captured(vk, shard_proofs, is_first_shard, is_complete, opts)
            .map(|captured| captured.report)
    }

    pub fn benchmark_lift_core_node_captured(
        &self,
        vk: &DTVerifyingKey,
        shard_proofs: &[dt_stark::sumcheck::proof::SCShardProof<CoreSC>],
        is_first_shard: bool,
        is_complete: bool,
        opts: DTProverOpts,
    ) -> Result<DTLiftCaptured> {
        let total_start = Instant::now();

        let input_build_start = Instant::now();
        if shard_proofs.is_empty() {
            bail!("S2 lift benchmark requires at least one core shard proof");
        }
        if shard_proofs.len() > 2 {
            bail!(
                "S2 lift benchmark expects at most two shard proofs in one lift node, got {}",
                shard_proofs.len()
            );
        }
        let input = SCDTRecursionWitnessValues {
            vk: vk.vk.clone(),
            shard_proofs: shard_proofs.to_vec(),
            is_complete,
            is_first_shard,
            vk_root: self.recursion_vk_root,
            reconstruct_deferred_digest: [Val::<CoreSC>::zero(); DIGEST_SIZE],
        };
        let input_build_ms = input_build_start.elapsed().as_millis();

        let witness_write_start = Instant::now();
        let mut witness_stream = Vec::new();
        Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
        let witness_write_ms = witness_write_start.elapsed().as_millis();

        let program_compile_start = Instant::now();
        let program = self.recursion_program(&input);
        let program_compile_ms = program_compile_start.elapsed().as_millis();
        let program_instructions = program.inner.iter().count();

        let runtime_start = Instant::now();
        let mut runtime =
            RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _, INNER_SBOX_DEGREE>::new(
                program.clone(),
                self.compress_prover.config().perm.clone(),
            );
        runtime.witness_stream = witness_stream.into();
        runtime.run().map_err(|err| anyhow!("DSL lift runtime failed: {err}"))?;
        let record = runtime.record;
        let runtime_ms = runtime_start.elapsed().as_millis();

        let dependencies_start = Instant::now();
        let mut records = vec![record];
        self.compress_prover.machine().generate_dependencies(
            &mut records,
            &opts.recursion_opts,
            None,
        );
        let record = records.into_iter().next().context("missing recursion execution record")?;
        let dependencies_ms = dependencies_start.elapsed().as_millis();

        let tracegen_start = Instant::now();
        let traces = self.compress_prover.generate_traces(&record);
        let tracegen_ms = tracegen_start.elapsed().as_millis();
        let trace_costs = trace_costs(self, &traces)?;
        let useful_cells = trace_costs.iter().map(|entry| entry.useful_cells).sum::<usize>();
        let padded_cells = trace_costs.iter().map(|entry| entry.padded_cells).sum::<usize>();
        let total_cells = useful_cells + padded_cells;

        let setup_start = Instant::now();
        let (pk, vk) = self.compress_prover.setup(&program);
        let setup_ms = setup_start.elapsed().as_millis();

        let mut challenger = self.compress_prover.config().mlchallenger();
        pk.observe_into(&mut challenger);

        let commit_start = Instant::now();
        let data = self.compress_prover.commit_with_pcs_stack_log_height(
            &record,
            traces,
            pk.preprocessed_pcs_stack_log_height(),
        );
        let commit_ms = commit_start.elapsed().as_millis();

        let open_start = Instant::now();
        let proof = self
            .compress_prover
            .open(&pk, data, &mut challenger, num_skip_rounds(), chip_log_height_threshold())
            .map_err(|err| anyhow!("DSL lift open failed: {err}"))?;
        let open_ms = open_start.elapsed().as_millis();

        let proof_bytes =
            bincode::serialize(&SCMachineProof::<InnerSC> { shard_proofs: vec![proof.clone()] })
                .context("serialize DSL lift proof")?
                .len();
        let proof_with_vk_bytes =
            bincode::serialize(&(&vk, &proof)).context("serialize DSL lift proof with vk")?.len();
        let captured = (vk, proof);

        let useful = useful_cells as f64;
        let report = DTLiftBenchmarkReport {
            timings: DTLiftBenchmarkTimings {
                input_build_ms,
                program_compile_ms,
                witness_write_ms,
                runtime_ms,
                dependencies_ms,
                tracegen_ms,
                setup_ms,
                commit_ms,
                open_ms,
                total_ms: total_start.elapsed().as_millis(),
            },
            proof_bytes,
            proof_with_vk_bytes,
            program_instructions,
            trace_costs,
            useful_cells,
            padded_cells,
            total_cells,
            padding_to_useful_ratio: if useful > 0.0 { padded_cells as f64 / useful } else { 0.0 },
            total_to_useful_ratio: if useful > 0.0 { total_cells as f64 / useful } else { 0.0 },
        };
        Ok(DTLiftCaptured { report, vk: captured.0, proof: captured.1 })
    }

    /// Benchmarks ONE DSL join (compress-of-recursion) node over already-proven recursion
    /// children, with the same line-item split as the lift benchmark. The program compile
    /// is reported separately: production caches join programs (fixed shapes), so the
    /// steady-state ratio should exclude it.
    pub fn benchmark_join_recursion_node(
        &self,
        vks_and_proofs: Vec<(SCStarkVerifyingKey<InnerSC>, SCShardProof<InnerSC>)>,
        opts: DTProverOpts,
    ) -> Result<DTLiftBenchmarkReport> {
        let total_start = Instant::now();

        let input_build_start = Instant::now();
        if vks_and_proofs.is_empty() {
            bail!("join benchmark requires at least one recursion child");
        }
        let input = SCDTCompressWitnessValues { vks_and_proofs, is_complete: false };
        let input_with_merkle = self.make_merkle_proofs(input);
        let input_build_ms = input_build_start.elapsed().as_millis();

        let witness_write_start = Instant::now();
        let mut witness_stream = Vec::new();
        Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);
        let witness_write_ms = witness_write_start.elapsed().as_millis();

        let program_compile_start = Instant::now();
        let program = self.compress_program(&input_with_merkle);
        let program_compile_ms = program_compile_start.elapsed().as_millis();
        let program_instructions = program.inner.iter().count();

        let runtime_start = Instant::now();
        let mut runtime =
            RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _, INNER_SBOX_DEGREE>::new(
                program.clone(),
                self.compress_prover.config().perm.clone(),
            );
        runtime.witness_stream = witness_stream.into();
        runtime.run().map_err(|err| anyhow!("DSL join runtime failed: {err}"))?;
        let record = runtime.record;
        let runtime_ms = runtime_start.elapsed().as_millis();

        let dependencies_start = Instant::now();
        let mut records = vec![record];
        self.compress_prover.machine().generate_dependencies(
            &mut records,
            &opts.recursion_opts,
            None,
        );
        let record = records.into_iter().next().context("missing join execution record")?;
        let dependencies_ms = dependencies_start.elapsed().as_millis();

        let tracegen_start = Instant::now();
        let traces = self.compress_prover.generate_traces(&record);
        let tracegen_ms = tracegen_start.elapsed().as_millis();
        let trace_costs = trace_costs(self, &traces)?;
        let useful_cells = trace_costs.iter().map(|entry| entry.useful_cells).sum::<usize>();
        let padded_cells = trace_costs.iter().map(|entry| entry.padded_cells).sum::<usize>();
        let total_cells = useful_cells + padded_cells;

        let setup_start = Instant::now();
        let (pk, vk) = self.compress_prover.setup(&program);
        let setup_ms = setup_start.elapsed().as_millis();

        let mut challenger = self.compress_prover.config().mlchallenger();
        pk.observe_into(&mut challenger);

        let commit_start = Instant::now();
        let data = self.compress_prover.commit_with_pcs_stack_log_height(
            &record,
            traces,
            pk.preprocessed_pcs_stack_log_height(),
        );
        let commit_ms = commit_start.elapsed().as_millis();

        let open_start = Instant::now();
        let proof = self
            .compress_prover
            .open(&pk, data, &mut challenger, num_skip_rounds(), chip_log_height_threshold())
            .map_err(|err| anyhow!("DSL join open failed: {err}"))?;
        let open_ms = open_start.elapsed().as_millis();

        let proof_bytes =
            bincode::serialize(&SCMachineProof::<InnerSC> { shard_proofs: vec![proof.clone()] })
                .context("serialize DSL join proof")?
                .len();
        let proof_with_vk_bytes =
            bincode::serialize(&(vk, proof)).context("serialize DSL join proof with vk")?.len();

        let useful = useful_cells as f64;
        Ok(DTLiftBenchmarkReport {
            timings: DTLiftBenchmarkTimings {
                input_build_ms,
                program_compile_ms,
                witness_write_ms,
                runtime_ms,
                dependencies_ms,
                tracegen_ms,
                setup_ms,
                commit_ms,
                open_ms,
                total_ms: total_start.elapsed().as_millis(),
            },
            proof_bytes,
            proof_with_vk_bytes,
            program_instructions,
            trace_costs,
            useful_cells,
            padded_cells,
            total_cells,
            padding_to_useful_ratio: if useful > 0.0 { padded_cells as f64 / useful } else { 0.0 },
            total_to_useful_ratio: if useful > 0.0 { total_cells as f64 / useful } else { 0.0 },
        })
    }
}

fn trace_costs<C: DTProverComponents>(
    prover: &DTProver<C>,
    traces: &[(String, CompressedMatrix<Val<InnerSC>>)],
) -> Result<Vec<DTTraceCost>> {
    traces
        .iter()
        .map(|(name, trace)| {
            let chip = prover
                .compress_prover
                .machine()
                .chips
                .iter()
                .find(|chip| chip.name() == *name)
                .ok_or_else(|| anyhow!("generated trace chip {name} not found in DSL machine"))?;
            let width = trace.main.width();
            let useful_cells = trace.stored_height() * width;
            let total_cells = trace.total_height * width;
            Ok(DTTraceCost {
                chip: name.clone(),
                height: trace.total_height,
                stored_height: trace.stored_height(),
                width,
                perm_width: chip.perm_width(),
                interactions: chip.num_lookup(),
                constraints: chip.num_alpha,
                useful_cells,
                padded_cells: total_cells.saturating_sub(useful_cells),
            })
        })
        .collect()
}
