use crate::{
    io::DTStdin,
    riscv::RiscvAir,
    shape::{chip_log_height_threshold, num_skip_rounds, CoreShapeConfig},
    utils::sc_prove_core,
};
use dt_core_executor::{DTContext, ExecutionRecord, Executor, Program};
use dt_primitives::io::DTPublicValues;
use dt_stark::{
    air::MachineAir,
    baby_bear_poseidon2::SCBabyBearPoseidon2,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, MlPcsProverData, SCStarkGenericConfig},
        folder::SumcheckVerifierConstraintFolder,
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::SCMachineProof,
        prover::SCMachineProver,
        trace::CompressedMatrix,
    },
    Challenge, DTCoreOpts, DebugConstraintBuilder, InteractionBuilder, MachineRecord,
    MachineVerificationError, Val,
};
use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_uni_stark::SymbolicAirBuilder;
use serde::{de::DeserializeOwned, Serialize};

/// This type is the function signature used for malicious trace and public values generators for
/// failure test cases.
pub(crate) type MaliciousTracePVGeneratorType<Val, P> =
    Box<dyn Fn(&P, &mut ExecutionRecord) -> Vec<(String, CompressedMatrix<Val>)> + Send + Sync>;

/// The canonical entry point for testing a [`Program`] and [`DTStdin`] with a [`MachineProver`].
pub fn run_test<
    P: SCMachineProver<
        SCBabyBearPoseidon2,
        RiscvAir<BabyBear>,
        RiscvAir<BinomialExtensionField<BabyBear, 4>>,
        DeviceProvingKey = SCStarkProvingKey<SCBabyBearPoseidon2>,
    >,
>(
    mut program: Program,
    inputs: DTStdin,
) -> Result<DTPublicValues, MachineVerificationError<SCBabyBearPoseidon2>> {
    let shape_config = CoreShapeConfig::<BabyBear>::default();
    shape_config.fix_preprocessed_shape(&mut program).unwrap();

    let runtime = tracing::debug_span!("runtime.run(...)").in_scope(|| {
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.maximal_shapes = Some(
            shape_config
                .maximal_core_shapes(DTCoreOpts::default().shard_size.ilog2() as usize)
                .into_iter()
                .collect(),
        );
        runtime.write_vecs(&inputs.buffer);
        runtime.run().unwrap();
        runtime
    });
    let public_values = DTPublicValues::from(&runtime.state.public_values_stream);

    let _ = sc_run_test_core::<P>(runtime, inputs, Some(&shape_config), None)?;
    Ok(public_values)
}

pub fn sc_run_test_core<
    P: SCMachineProver<
        SCBabyBearPoseidon2,
        RiscvAir<BabyBear>,
        RiscvAir<BinomialExtensionField<BabyBear, 4>>,
        DeviceProvingKey = SCStarkProvingKey<SCBabyBearPoseidon2>,
    >,
>(
    runtime: Executor,
    inputs: DTStdin,
    shape_config: Option<&CoreShapeConfig<BabyBear>>,
    malicious_trace_pv_generator: Option<MaliciousTracePVGeneratorType<BabyBear, P>>,
) -> Result<SCMachineProof<SCBabyBearPoseidon2>, MachineVerificationError<SCBabyBearPoseidon2>> {
    let config = SCBabyBearPoseidon2::new();
    let machine = RiscvAir::sc_machine(config);
    let prover = P::new(machine);

    let (pk, vk) = prover.setup(runtime.program.as_ref());
    let (proof, _output, _) = sc_prove_core(
        &prover,
        &pk,
        &vk,
        Program::clone(&runtime.program),
        &inputs,
        DTCoreOpts::default(),
        DTContext::default(),
        shape_config,
        malicious_trace_pv_generator,
    )
    .unwrap();

    let config = SCBabyBearPoseidon2::new();
    let machine = RiscvAir::sc_machine(config);
    // let (pk, vk) = machine.setup(runtime.program.as_ref());
    let mut challenger = machine.config().mlchallenger();
    if let Err(e) =
        machine.verify(&vk, &proof, &mut challenger, num_skip_rounds(), chip_log_height_threshold())
    {
        Err(e)
    } else {
        Ok(proof)
    }
}

pub fn sc_run_test_machine_with_prover<SC, A, AE, P: SCMachineProver<SC, A, AE>>(
    prover: &P,
    records: Vec<A::Record>,
    pk: P::DeviceProvingKey,
    vk: SCStarkVerifyingKey<SC>,
    num_skip_rounds: usize,
    chip_log_height_threshold: usize,
) -> Result<SCMachineProof<SC>, MachineVerificationError<SC>>
where
    SC: SCStarkGenericConfig,
    A: MachineAir<SC::Val>
        + Air<InteractionBuilder<Val<SC>>>
        + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, Challenge<SC>>>
        + for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>
        + Air<SymbolicAirBuilder<SC::Val>>,
    AE: MachineAir<Challenge<SC>>
        + Air<InteractionBuilder<Challenge<SC>>>
        + Air<SymbolicAirBuilder<Challenge<SC>>>,
    A::Record: MachineRecord<Config = DTCoreOpts>,
    SC::Val: p3_field::PrimeField32,
    MlCom<SC>: Send + Sync,
    MlPcsProverData<SC>: Send + Sync + Serialize + DeserializeOwned,
    MlPcsOpeningProof<SC>: Send + Sync,
    SC::MlChallenger: Clone,
{
    let mut challenger = prover.config().mlchallenger();
    let prove_span = tracing::debug_span!("prove").entered();

    #[cfg(feature = "debug")]
    prover.machine().debug_constraints(
        &prover.pk_to_host(&pk),
        records.clone(),
        &mut challenger.clone(),
    );

    let proof = prover
        .prove(
            &pk,
            records,
            &mut challenger,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        )
        .unwrap();
    prove_span.exit();
    let _nb_bytes = bincode::serialize(&proof).unwrap().len();

    let mut challenger = prover.config().mlchallenger();
    prover.machine().verify(
        &vk,
        &proof,
        &mut challenger,
        num_skip_rounds,
        chip_log_height_threshold,
    )?;

    Ok(proof)
}
