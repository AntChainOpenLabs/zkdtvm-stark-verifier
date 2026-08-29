// mod complete;
mod compress;
mod core;
mod deferred;
mod public_values;
mod root;
#[cfg(all(test, not(feature = "koalabear")))]
mod test;
mod vkey_proof;
mod witness;
mod wrap;

pub use compress::*;
pub use core::*;
pub use deferred::*;
pub use root::*;
pub use vkey_proof::*;
pub use wrap::*;

#[allow(unused_imports)]
pub use witness::*;
#[cfg(all(test, not(feature = "koalabear")))]
pub mod tests {
    use crate::{
        challenger::CanObserveVariable,
        sumcheck::{
            types::SCVerifyingKeyVariable, verifier::SumcheckVerifier, SCBabyBearFriConfig,
            SCBabyBearFriConfigVariable,
        },
        utils::sc_tests::run_test_recursion_with_prover,
        witness::{WitnessBlock, Witnessable},
        CircuitConfig, VerifyingKeyVariable,
    };
    use dt_core_executor::{
        DTContext, ExecutionError, ExecutionReport, Executor, Program, RiscvAirId,
    };
    use dt_core_machine::{
        io::DTStdin,
        riscv::RiscvAir,
        shape::{chip_log_height_threshold, num_skip_rounds},
        utils::{sc_run_test_machine_with_prover, setup_logger},
    };
    use dt_primitives::io::DTPublicValues;
    use dt_recursion_compiler::{
        config::InnerConfig,
        ir::{Builder, Felt},
    };
    use dt_recursion_core::{
        air::RecursionPublicValues,
        machine::RecursionAir,
        shape::{RecursionShape, RecursionShapeConfig},
        ExecutionRecord, RecursionProgram, Runtime as RecursionRuntime,
    };
    use dt_stark::{
        baby_bear_poseidon2::SCBabyBearPoseidon2,
        sumcheck::{
            config::SCStarkGenericConfig,
            keys::{SCStarkProvingKey, SCStarkVerifyingKey},
            proof::{SCMachineProof, SCShardProof},
            prover::{SCMachineProver, SumcheckProver},
        },
        Challenge, CpuProver, DTCoreOpts, DTProverOpts, StarkGenericConfig,
    };
    use log::debug;
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, Field, PrimeField32};
    use std::{
        borrow::Borrow,
        collections::BTreeMap,
        env,
        error::Error,
        num::NonZeroUsize,
        path::Path,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc::{channel, sync_channel},
            Arc, Mutex, OnceLock,
        },
        thread,
    };
    type CoreSC = SCBabyBearPoseidon2;
    type InnerSC = SCBabyBearPoseidon2;
    const COMPRESS_DEGREE: usize = 3;
    const SHRINK_DEGREE: usize = 3;
    const WRAP_DEGREE: usize = if cfg!(feature = "koalabear") { 3 } else { 9 };

    const CORE_CACHE_SIZE: usize = 5;
    pub const REDUCE_BATCH_SIZE: usize = 2;

    type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE>;
    type ShrinkAir<F> = RecursionAir<F, SHRINK_DEGREE>;
    type WrapAir<F> = RecursionAir<F, WRAP_DEGREE>;
    #[derive(Clone)]
    struct DTCoreProofData(pub Vec<SCShardProof<CoreSC>>);

    #[derive(Clone)]
    struct DTReducedProofData(pub SCShardProof<InnerSC>);
    struct DTProofWithMetadata<P: Clone> {
        pub proof: P,
        pub stdin: DTStdin,
        pub public_values: DTPublicValues,
    }
    type DTCoreProof = DTProofWithMetadata<DTCoreProofData>;

    /// An zkDTVM proof that has been recursively reduced into a single proof. This proof can be
    /// verified within zkDTVM programs.
    type DTReducedProof = DTProofWithMetadata<DTReducedProofData>;
    /// The prover for making zkDTVM core proofs.s
    type CoreProver = SumcheckProver<
        CoreSC,
        RiscvAir<<CoreSC as StarkGenericConfig>::Val>,
        RiscvAir<Challenge<CoreSC>>,
        // DeviceProvingKey = SCStarkProvingKey<CoreSC>,
    >;

    /// The prover for making zkDTVM recursive proofs.
    type CompressProver = SumcheckProver<
        InnerSC,
        CompressAir<<InnerSC as StarkGenericConfig>::Val>,
        CompressAir<Challenge<InnerSC>>,
    >;
    pub type DeviceProvingKey = <CoreProver as SCMachineProver<
        SCBabyBearPoseidon2,
        RiscvAir<BabyBear>,
        RiscvAir<Challenge<CoreSC>>,
    >>::DeviceProvingKey;
    /// The prover for shrinking compressed proofs.
    type ShrinkProver = SumcheckProver<
        InnerSC,
        ShrinkAir<<InnerSC as StarkGenericConfig>::Val>,
        ShrinkAir<Challenge<InnerSC>>,
    >;
    #[derive(Clone)]
    pub struct DTVerifyingKey {
        pub vk: SCStarkVerifyingKey<CoreSC>,
    }
    #[derive(Clone)]
    pub struct DTProvingKey {
        pub pk: SCStarkProvingKey<CoreSC>,
        pub elf: Vec<u8>,
        /// Verifying key is also included as we need it for recursion
        pub vk: DTVerifyingKey,
    }
    fn core_prover() -> CoreProver {
        let core_machine = RiscvAir::sc_machine(CoreSC::default());
        CoreProver::new(core_machine)
    }
    fn compress_prover() -> CompressProver {
        let compress_machine = CompressAir::sc_compress_machine(InnerSC::default());
        let compress_prover = CompressProver::new(compress_machine);
        compress_prover
    }
    fn shrink_prover() -> ShrinkProver {
        let shrink_machine = ShrinkAir::sc_shrink_machine(InnerSC::compressed());
        let shrink_prover = ShrinkProver::new(shrink_machine);
        shrink_prover
    }
    fn get_program(elf: &[u8]) -> Program {
        let program = Program::from(elf).expect("Failed to parse ELF");
        program
    }
    fn setup(
        core_prover: &CoreProver,
        elf: &[u8],
    ) -> (DTProvingKey, DeviceProvingKey, Program, DTVerifyingKey) {
        let program = get_program(elf);
        let (pk, vk) = core_prover.setup(&program);
        let vk = DTVerifyingKey { vk };
        let pk =
            DTProvingKey { pk: core_prover.pk_to_host(&pk), elf: elf.to_vec(), vk: vk.clone() };
        let pk_d = core_prover.pk_to_device(&pk.pk);
        (pk, pk_d, program, vk)
    }
    fn execute<'a>(
        elf: &[u8],
        stdin: &DTStdin,
        context: DTContext<'a>,
    ) -> Result<(DTPublicValues, [u8; 32], ExecutionReport), ExecutionError> {
        // context.subproof_verifier = Some(self);

        let (opts, program) = (DTCoreOpts::default(), Program::from(elf).unwrap());

        let mut runtime = Executor::with_context(program, opts, context);

        runtime.maybe_setup_profiler(elf);

        runtime.write_vecs(&stdin.buffer);
        for (proof, vkey) in stdin.proofs.iter() {
            runtime.write_proof(proof.clone(), vkey.clone());
        }
        runtime.run_fast()?;

        let mut committed_value_digest = [0u8; 32];
        runtime.record.public_values.committed_value_digest.iter().enumerate().for_each(
            |(i, word)| {
                let bytes = word.to_le_bytes();
                committed_value_digest[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            },
        );

        Ok((
            DTPublicValues::from(&runtime.state.public_values_stream),
            committed_value_digest,
            runtime.report,
        ))
    }

    fn prove_core<'a>(
        core_prover: &'a CoreProver,
        pk_d: &<CoreProver as SCMachineProver<
            SCBabyBearPoseidon2,
            RiscvAir<BabyBear>,
            RiscvAir<Challenge<SCBabyBearPoseidon2>>,
        >>::DeviceProvingKey,
        program: Program,
        stdin: &DTStdin,
        opts: DTProverOpts,
        context: DTContext<'a>,
    ) -> DTCoreProof {
        // context.subproof_verifier = Some(self);

        // Launch two threads to simultaneously prove the core and compile the first few
        // recursion programs in parallel.
        let span = tracing::Span::current().clone();
        std::thread::scope(|s| {
            let _span = span.enter();
            let (proof_tx, proof_rx) = channel();
            let (shape_tx, shape_rx) = channel();

            let span = tracing::Span::current().clone();
            let handle = s.spawn(move || {
                let _span = span.enter();

                // Copy the proving key to the device.
                let pk = pk_d;

                // We may calculate gas while proving if the opts match the hardcoded variant.
                // This ensures that the gas number is consistent between `execute` and
                // `prove_core`. This behavior is undocumented because it is
                // confusing and not very useful.
                //
                // If `context.calculate_gas` is set, we use the logic from the `gas` module
                // after checkpoint execution to print gas as part of the execution report.

                // Prove the core and stream the proofs and shapes.
                dt_core_machine::utils::sc_prove_core_stream::<CoreSC, CoreProver>(
                    &core_prover,
                    pk,
                    program,
                    stdin,
                    opts.core_opts,
                    context,
                    None,
                    proof_tx,
                    shape_tx,
                    None,
                    None,
                )
            });

            // Receive the first few shapes and comile the recursion programs.
            // for _ in 0..3 {
            //     if let Ok((shape, is_complete)) = shape_rx.recv() {
            //         let recursion_shape =
            //             DTRecursionShape { proof_shapes: vec![shape], is_complete };

            //         // Only need to compile the recursion program if we're not in the one-shard
            //         // case.
            //         let compress_shape = DTCompressProgramShape::Recursion(recursion_shape);

            //         // Insert the program into the cache.
            //         self.program_from_shape(compress_shape, None);
            //     }
            // }

            // Collect the shard proofs and the public values stream.
            let shard_proofs: Vec<SCShardProof<_>> = proof_rx.iter().collect();
            let (public_values_stream, _) = handle.join().unwrap().unwrap();
            let public_values = DTPublicValues::from(&public_values_stream);
            // Self::check_for_high_cycles(cycles);
            DTCoreProof {
                proof: DTCoreProofData(shard_proofs),
                stdin: stdin.clone(),
                public_values,
            }
        })
    }

    #[test]
    fn test_shard_with_perm() {
        let elf = test_artifacts::FIBONACCI_ELF;

        let config = SCBabyBearPoseidon2::new();
        let stdin = DTStdin::default();
        let opts = DTProverOpts::auto();
        let core_prover = core_prover();
        let core_machine = core_prover.machine();
        let context = DTContext::default();
        tracing::info!("setup elf");
        let (_, pk_d, program, vk) = setup(&core_prover, elf);
        let core_proof = prove_core(&core_prover, &pk_d, program, &stdin, opts, context);
        let public_values = core_proof.public_values.clone();
        let shard_proofs = core_proof.proof.0;

        let mut builder = Builder::<InnerConfig>::default();

        let mut witness_stream = Vec::<WitnessBlock<InnerConfig>>::new();
        let vk_plain = vk.vk;
        setup_logger();
        // Add a hash invocation, since the poseidon2 table expects that it's in the first row.
        let mut challenger = config.clone().challenger_variable(&mut builder);
        Witnessable::<InnerConfig>::write(&vk_plain, &mut witness_stream);
        let vk: SCVerifyingKeyVariable<_, _> = vk_plain.read(&mut builder);
        vk.observe_into(&mut builder, &mut challenger);
        debug!("proofs len : {}", shard_proofs.len());
        let proof = shard_proofs[0].clone();
        let mut verifier_challenger = config.mlchallenger();
        let machine_proof = SCMachineProof { shard_proofs: vec![proof.clone()] };
        core_machine
            .verify(
                &vk_plain,
                &machine_proof,
                &mut verifier_challenger,
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .unwrap();

        let proofs = proof.read(&mut builder);
        Witnessable::<InnerConfig>::write(&proof, &mut witness_stream);

        // Verify the first proof.
        let mut challenger = challenger.copy(&mut builder);
        let pv_slice = &proofs.public_values[..core_prover.num_pv_elts()];
        challenger.observe_slice(&mut builder, pv_slice.iter().cloned());
        debug!("start recursion verify");
        SumcheckVerifier::verify_shard(
            &mut builder,
            &vk,
            &core_machine,
            &mut challenger,
            &proofs,
            num_skip_rounds(),
            chip_log_height_threshold(),
        );

        debug!("end recursion verify");

        run_test_recursion_with_prover::<SumcheckProver<_, _, _>>(
            builder.into_root_block(),
            witness_stream,
            num_skip_rounds(),
            chip_log_height_threshold(),
        );
    }
}
