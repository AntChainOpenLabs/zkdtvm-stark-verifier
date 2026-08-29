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
use dt_core_executor::{DTContext, ExecutionError, ExecutionReport, Executor, Program, RiscvAirId};
use dt_core_machine::{
    io::DTStdin,
    riscv::RiscvAir,
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
