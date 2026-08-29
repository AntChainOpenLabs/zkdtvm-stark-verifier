//! An end-to-end-prover implementation for the zkDTVM RISC-V zkVM.
//!
//! Separates the proof generation process into multiple stages:
//!
//! 1. Generate shard proofs which split up and prove the valid execution of a RISC-V program.
//! 2. Compress shard proofs into a single shard proof.
//! 3. Wrap the shard proof into a SNARK-friendly field.
//! 4. Wrap the last shard proof, proven over the SNARK-friendly field, into a PLONK proof.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_without_default)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
pub(crate) struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Self
    }

    pub(crate) fn elapsed(self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    pub(crate) fn duration_since(self, _earlier: Self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

pub mod bench;
pub mod build;
pub mod components;
#[cfg(feature = "native-recursion")]
pub mod native_backend;
pub mod shapes;
pub mod stage_ledger;
#[cfg(feature = "native-recursion")]
pub mod tree_plan;
pub mod types;
pub mod utils;
pub mod verify;

// The native ladder machines exist only for koalabear + ext5; a babybear build that pulls
// the backend in would activate both fields on dt-primitives via feature unification.
#[cfg(all(feature = "native-recursion", feature = "babybear"))]
compile_error!(
    "feature `native-recursion` requires koalabear + ext5 and cannot be combined with `babybear`"
);
use crate::{
    shapes::{DTCompressProgramShape, DTProofShape},
    utils::words_to_bytes,
};
use dt_core_executor::{DTContext, ExecutionError, ExecutionReport, Executor, Program};
use dt_core_machine::{
    io::DTStdin,
    reduce::DTReduceProof,
    riscv::riscv_polyair::RiscvPolyAir,
    shape::{chip_log_height_threshold, num_skip_rounds, CoreShapeConfig},
    utils::{concurrency::TurnBasedSync, DTCoreProverError},
};
pub use dt_primitives::io::DTPublicValues;
use dt_primitives::{sc_hash_deferred_proof, SCField};
#[cfg(feature = "koalabear")]
use dt_recursion_circuit::SCWrapConfig as WrapConfig;
#[cfg(feature = "babybear")]
use dt_recursion_circuit::WrapConfig;
use dt_recursion_circuit::{
    hash::FieldHasher,
    machine::{DTCompressWithVkeyShape, DTRecursionShape, PublicValuesOutputDigest},
    merkle_tree::MerkleTree,
    sc_machine::{
        SCDTCompressRootVerifierWithVKey, SCDTCompressWithVKeyVerifier,
        SCDTCompressWithVKeyWitnessValues, SCDTCompressWitnessValues, SCDTDeferredVerifier,
        SCDTDeferredWitnessValues, SCDTMerkleProofWitnessValues, SCDTRecursionWitnessValues,
        SCDTRecursiveVerifier,
    },
    witness::Witnessable,
};
#[cfg(feature = "babybear")]
use dt_recursion_compiler::config::InnerConfig;
#[cfg(feature = "koalabear")]
use dt_recursion_compiler::config::SCInnerConfig as InnerConfig;
// [C1] Shrink-stage Config: a single newtype whose math types mirror
// `SCInnerConfig` (KoalaBear path), enabling `exp_reverse_bits_ext` to be
// specialised (inlined) for the shrink stage so no `Instruction::ExtExpReverseBits`
// is emitted and `ExtExpReverseBitsChip` can be removed from `sc_shrink_machine`.
use dt_recursion_compiler::{
    circuit::AsmCompiler,
    config::ShrinkConfig,
    ir::{Builder, DslIrProgram, Witness},
};
#[cfg(feature = "babybear")]
use dt_recursion_core::stark::SCBabyBearPoseidon2Outer;
#[cfg(feature = "koalabear")]
use dt_recursion_core::stark::SCKoalaBearPoseidon2Outer;
use dt_recursion_core::{
    air::RecursionPublicValues,
    machine::{RecursionAir, RecursionAirEventCount},
    polyair::RecursionPolyAir,
    shape::{RecursionShape, RecursionShapeConfig},
    ExecutionRecord, RecursionProgram, Runtime as RecursionRuntime,
};
use dt_stark::sumcheck::trace::CompressedMatrix;
use lru::LruCache;
use p3_field::{AbstractField, PrimeField, PrimeField32};

pub use dt_recursion_gnark_ffi::proof::{Groth16Bn254Proof, PlonkBn254Proof};

use dt_recursion_gnark_ffi::Groth16Bn254Prover;

use dt_recursion_gnark_ffi::PlonkBn254Prover;
#[cfg(feature = "babybear")]
use dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;
#[cfg(feature = "koalabear")]
use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::{
    SCKoalaBearPoseidon2, SCKoalaBearSha256Root,
};
#[cfg(feature = "debug")]
use dt_stark::sumcheck::proof::SCMachineProof;
use dt_stark::{
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCMachineProvingKey, SCStarkVerifyingKey},
        proof::SCShardProof,
        prover::SCMachineProver as StarkMachineProver,
    },
    Challenge, DTProverOpts, RecursionBackend, StarkGenericConfig, Val, Word, DIGEST_SIZE,
};
use polyair::prover::SCMachineProver as PolyAirMachineProver;

#[cfg(feature = "native-recursion")]
use std::sync::mpsc::TrySendError;
use std::{
    borrow::Borrow,
    collections::{BTreeMap, VecDeque},
    env,
    num::NonZeroUsize,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{channel, sync_channel, Receiver, Sender, SyncSender},
        Arc, Mutex, OnceLock,
    },
    thread,
};
use tracing::instrument;

use components::{DTProverComponents, SCCpuProverComponents};
use dt_recursion_circuit::machine::DTCompressShape;
use dt_stark::shape::OrderedShape;
pub use types::{HashableKey, *};
use utils::{dt_committed_values_digest_bn254, dt_vkey_digest_bn254};

/// The global version for all components of zkDTVM.
///
/// This string should be updated whenever any step in verifying a zkDTVM proof changes, including
/// core, recursion, and plonk-bn254. This string is used to download zkDTVM artifacts and the gnark
/// docker image.
pub const DT_CIRCUIT_VERSION: &str = include_str!("../DT_VERSION");

/// The configuration for the core prover.
#[cfg(feature = "babybear")]
pub type CoreSC = SCBabyBearPoseidon2;
#[cfg(feature = "koalabear")]
pub type CoreSC = SCKoalaBearPoseidon2;

/// The configuration for the inner prover.
#[cfg(feature = "babybear")]
pub type InnerSC = SCBabyBearPoseidon2;
#[cfg(feature = "koalabear")]
pub type InnerSC = SCKoalaBearPoseidon2;

/// The configuration for the final root_shrink proof.
///
/// This proof is verified natively and is not fed into another recursive
/// verifier, so its PCS Merkle hash can use SHA256 instead of Poseidon2.
/// The WHIR JSON files configure PCS parameters, while this final-hash choice
/// is a Rust type-level selection through `RootSC`.
#[cfg(feature = "babybear")]
pub type RootSC = SCBabyBearPoseidon2;
#[cfg(feature = "koalabear")]
pub type RootSC = SCKoalaBearSha256Root;

/// The configuration for the outer prover.
#[cfg(feature = "babybear")]
pub type OuterSC = SCBabyBearPoseidon2Outer;
#[cfg(feature = "koalabear")]
pub type OuterSC = SCKoalaBearPoseidon2Outer;

/// Poseidon2 sbox degree for the inner SC field (7 for BabyBear, 3 for KoalaBear).
const INNER_SBOX_DEGREE: u64 = if cfg!(feature = "koalabear") { 3 } else { 7 };

/// Extension degree used by PolyAir recursion proofs.
#[cfg(feature = "ext5")]
pub const POLYAIR_EXT_DEGREE: usize = 5;
#[cfg(not(feature = "ext5"))]
pub const POLYAIR_EXT_DEGREE: usize = 4;

pub type DeviceProvingKey<C> = <<C as DTProverComponents>::CoreProver as PolyAirMachineProver<
    CoreSC,
    RiscvPolyAir<Val<CoreSC>>,
    POLYAIR_EXT_DEGREE,
>>::DeviceProvingKey;

// KoalaBear uses SBOX_DEGREE=3 with the dedicated `poseidon2_skinny_kb` chip,
// so its recursion degrees stay at 3. BabyBear uses SBOX_DEGREE=7 with the
// `poseidon2_skinny` chip which requires DEGREE >= 9 across all stages.
const COMPRESS_DEGREE: usize = if cfg!(feature = "koalabear") { 3 } else { 9 };
const SHRINK_DEGREE: usize = if cfg!(feature = "koalabear") { 3 } else { 9 };
const WRAP_DEGREE: usize = if cfg!(feature = "koalabear") { 3 } else { 9 };

const CORE_CACHE_SIZE: usize = 5;
pub const REDUCE_BATCH_SIZE: usize = 2;

pub type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE>;
pub type ShrinkAir<F> = RecursionAir<F, SHRINK_DEGREE>;
pub type WrapAir<F> = RecursionAir<F, WRAP_DEGREE>;
pub type CompressPolyAir<F> = RecursionPolyAir<F>;
pub type ShrinkPolyAir<F> = RecursionPolyAir<F>;

#[derive(Debug, Clone, Copy)]
pub enum ShrinkVerifyMachine {
    /// The shrink circuit verifies proofs produced by the ordinary compress stage.
    Compress,
    /// The shrink circuit verifies proofs produced by the shrink stage.
    Shrink,
    /// The shrink circuit verifies proofs produced by the root_shrink stage.
    RootShrink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceProofMachine {
    Compress,
    Shrink,
}

impl ReduceProofMachine {
    fn verify_machine(self) -> ShrinkVerifyMachine {
        match self {
            Self::Compress => ShrinkVerifyMachine::Compress,
            Self::Shrink => ShrinkVerifyMachine::Shrink,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Compress => "compress",
            Self::Shrink => "shrink",
        }
    }
}

fn classify_reduce_proof_machine<SC: SCStarkGenericConfig>(
    proof: &SCShardProof<SC>,
) -> ReduceProofMachine {
    let has_wide_poseidon = proof.chip_ordering.keys().any(|name| name.contains("Poseidon2Wide"));
    let has_skinny_poseidon =
        proof.chip_ordering.keys().any(|name| name.contains("Poseidon2Skinny"));

    match (has_wide_poseidon, has_skinny_poseidon) {
        (true, false) => ReduceProofMachine::Compress,
        (false, true) => ReduceProofMachine::Shrink,
        _ => panic!(
            "compress: cannot classify reduce proof machine from chip_ordering: {:?}",
            proof.chip_ordering.keys().collect::<Vec<_>>()
        ),
    }
}

fn classify_reduce_batch_machine<SC: SCStarkGenericConfig>(
    vks_and_proofs: &[(SCStarkVerifyingKey<SC>, SCShardProof<SC>)],
) -> ReduceProofMachine {
    let mut machines = vks_and_proofs.iter().map(|(_, proof)| classify_reduce_proof_machine(proof));
    let first = machines.next().expect("compress: vks_and_proofs must be non-empty");
    for machine in machines {
        assert_eq!(
            first,
            machine,
            "compress: reduce batch mixes {} and {} proof machines",
            first.as_str(),
            machine.as_str()
        );
    }
    first
}

/// A end-to-end for the zkDTVM RISC-V zkVM.
///
/// This object coordinates the proving along all the steps: core, compression, shrinkage, and
/// wrapping.
pub struct DTProver<C: DTProverComponents = SCCpuProverComponents> {
    /// The core prover.
    pub core_prover: C::CoreProver,
    /// The compress prover (for both lift and join).
    pub compress_prover: C::CompressProver,
    /// The shrink prover.
    pub shrink_prover: C::ShrinkProver,
    /// The shrink-shaped prover used by the final root reduce layer.
    ///
    /// This stage has its own config because its final PCS commitment is not
    /// recursively verified and can use SHA256 for Merkle hashing.
    pub root_shrink_prover: C::RootShrinkProver,
    /// The wrap prover.
    pub wrap_prover: C::WrapProver,
    /// The cache of compiled recursion programs.
    pub lift_programs_lru: Mutex<LruCache<DTRecursionShape, Arc<RecursionProgram<SCField>>>>,
    /// The number of cache misses for recursion programs.
    pub lift_cache_misses: AtomicUsize,
    /// The cache of compiled compression programs.
    pub join_programs_map: BTreeMap<DTCompressWithVkeyShape, Arc<RecursionProgram<SCField>>>,
    /// The number of cache misses for compression programs.
    pub join_cache_misses: AtomicUsize,
    /// The root of the allowed recursion verification keys.
    pub recursion_vk_root: <InnerSC as FieldHasher<SCField>>::Digest,
    /// The allowed VKs and their corresponding indices.
    pub recursion_vk_map: BTreeMap<<InnerSC as FieldHasher<SCField>>::Digest, usize>,
    /// The Merkle tree for the allowed VKs.
    pub recursion_vk_tree: MerkleTree<SCField, InnerSC>,
    /// The recursion shape configuration.
    pub compress_shape_config: Option<RecursionShapeConfig<SCField, CompressAir<SCField>>>,
    /// The program for wrapping.
    pub wrap_program: OnceLock<Arc<RecursionProgram<SCField>>>,
    /// The verifying key for wrapping.
    pub wrap_vk: OnceLock<SCStarkVerifyingKey<OuterSC>>,
    /// Whether to verify verification keys.
    pub vk_verification: bool,
    /// The lazily-built native recursion ladder backend. Build errors are
    /// cached so a failed init fails every subsequent native compress, never retries
    /// into a half-initialized state.
    #[cfg(feature = "native-recursion")]
    pub native_backend: OnceLock<Result<native_backend::NativeRecursionBackend<C::NativeProvider>, String>>,
}

impl<C: DTProverComponents> DTProver<C> {
    /// Initializes a new [DTProver].
    #[instrument(name = "initialize prover", level = "debug", skip_all)]
    pub fn new() -> Self {
        Self::uninitialized()
    }

    /// Creates a new [DTProver] with lazily initialized components.
    pub fn uninitialized() -> Self {
        Self::uninitialized_with_configs(
            CoreSC::default(),
            InnerSC::default(),
            InnerSC::shrink(),
            RootSC::default(),
        )
    }

    /// Creates a new [DTProver] whose lift/join prover uses the supplied recursion config.
    pub fn uninitialized_with_compress_config(compress_config: InnerSC) -> Self {
        Self::uninitialized_with_configs(
            CoreSC::default(),
            compress_config,
            InnerSC::shrink(),
            RootSC::default(),
        )
    }

    /// Creates a new [DTProver] with explicit stage configs.
    pub fn uninitialized_with_configs(
        core_config: CoreSC,
        compress_config: InnerSC,
        shrink_config: InnerSC,
        root_shrink_config: RootSC,
    ) -> Self {
        // Initialize the provers.
        let core_machine = RiscvPolyAir::sc_machine::<CoreSC, POLYAIR_EXT_DEGREE>(core_config);
        let core_prover = C::CoreProver::new(core_machine);

        let compress_machine =
            CompressPolyAir::sc_compress_machine::<InnerSC, POLYAIR_EXT_DEGREE>(compress_config);
        let compress_prover = C::CompressProver::new(compress_machine);

        let shrink_machine =
            ShrinkPolyAir::sc_shrink_machine::<InnerSC, POLYAIR_EXT_DEGREE>(shrink_config);
        let shrink_prover = C::ShrinkProver::new(shrink_machine);
        let root_shrink_machine =
            ShrinkPolyAir::sc_shrink_machine::<RootSC, POLYAIR_EXT_DEGREE>(root_shrink_config);
        let root_shrink_prover = C::RootShrinkProver::new(root_shrink_machine);

        let wrap_machine = WrapAir::sc_wrap_machine(OuterSC::default());
        let wrap_prover = C::WrapProver::new(wrap_machine);

        let core_cache_size = NonZeroUsize::new(
            env::var("PROVER_CORE_CACHE_SIZE")
                .unwrap_or_else(|_| CORE_CACHE_SIZE.to_string())
                .parse()
                .unwrap_or(CORE_CACHE_SIZE),
        )
        .expect("PROVER_CORE_CACHE_SIZE must be a non-zero usize");

        let recursion_shape_config = env::var("FIX_RECURSION_SHAPES")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
            .then_some(RecursionShapeConfig::default());
        let vk_verification =
            env::var("VERIFY_VK").map(|v| v.eq_ignore_ascii_case("true")).unwrap_or(false);
        tracing::debug!("vk verification: {}", vk_verification);

        let allowed_vk_map: BTreeMap<[SCField; DIGEST_SIZE], usize> = if vk_verification {
            bincode::deserialize(include_bytes!(concat!(env!("OUT_DIR"), "/vk_map.bin"))).unwrap()
        } else {
            DTProofShape::dummy_vk_map(
                &CoreShapeConfig::default(),
                recursion_shape_config.as_ref().unwrap_or(&RecursionShapeConfig::default()),
                REDUCE_BATCH_SIZE,
            )
        };
        tracing::debug!("vk map loaded: {} entries", allowed_vk_map.len());

        let (root, merkle_tree) = MerkleTree::commit(allowed_vk_map.keys().copied().collect());

        let mut compress_programs = BTreeMap::new();
        let program_cache_disabled = env::var("DT_DISABLE_PROGRAM_CACHE")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        if !program_cache_disabled {
            if let Some(config) = &recursion_shape_config {
                DTProofShape::generate_compress_shapes(config, REDUCE_BATCH_SIZE).for_each(
                    |shape| {
                        let compress_shape = DTCompressWithVkeyShape {
                            compress_shape: shape.into(),
                            merkle_tree_height: merkle_tree.height,
                        };
                        let input = SCDTCompressWithVKeyWitnessValues::<InnerSC>::dummy_polyair(
                            compress_prover.machine(),
                            &compress_shape,
                        );
                        let program = compress_program_from_input::<C>(
                            recursion_shape_config.as_ref(),
                            &compress_prover,
                            vk_verification,
                            &input,
                        );
                        let program = Arc::new(program);
                        compress_programs.insert(compress_shape, program);
                    },
                );
            }
        }

        Self {
            core_prover,
            compress_prover,
            shrink_prover,
            root_shrink_prover,
            wrap_prover,
            lift_programs_lru: Mutex::new(LruCache::new(core_cache_size)),
            lift_cache_misses: AtomicUsize::new(0),
            join_programs_map: compress_programs,
            join_cache_misses: AtomicUsize::new(0),
            recursion_vk_root: root,
            recursion_vk_tree: merkle_tree,
            recursion_vk_map: allowed_vk_map,
            compress_shape_config: recursion_shape_config,
            vk_verification,
            wrap_program: OnceLock::new(),
            wrap_vk: OnceLock::new(),
            #[cfg(feature = "native-recursion")]
            native_backend: OnceLock::new(),
        }
    }

    /// The native ladder backend, built on first use.
    #[cfg(feature = "native-recursion")]
    pub fn native_backend(
        &self,
    ) -> Result<&native_backend::NativeRecursionBackend<C::NativeProvider>, DTRecursionProverError> {
        self.native_backend
            .get_or_init(|| {
                native_backend::new_native_backend_with_provider::<C::NativeProvider>(self.core_prover.machine().config())
                    .map_err(|err| err.to_string())
            })
            .as_ref()
            .map_err(|err| DTRecursionProverError::RuntimeError(err.clone()))
    }

    /// Creates a proving key and a verifying key for a given RISC-V ELF.
    #[instrument(name = "setup", level = "debug", skip_all)]
    pub fn setup(
        &self,
        elf: &[u8],
    ) -> (DTProvingKey, DeviceProvingKey<C>, Program, DTVerifyingKey) {
        let program = self.get_program(elf).expect("setup: failed to load program from ELF");
        let (pk_d, vk) = self.core_prover.setup(&program);
        let vk = DTVerifyingKey { vk: vk.clone() };
        let pk = DTProvingKey { pk: self.core_prover.pk_to_host(&pk_d), elf: elf.to_vec(), vk: vk.clone() };
        (pk, pk_d, program, vk)
    }

    /// Parse a program without prescribing a core trace shape.
    pub fn get_program(&self, elf: &[u8]) -> eyre::Result<Program> {
        Ok(Program::from(elf)?)
    }

    /// Execute a zkDTVM program with the specified inputs.
    #[instrument(name = "execute", level = "info", skip_all)]
    pub fn execute<'a>(
        &'a self,
        elf: &[u8],
        stdin: &DTStdin,
        context: DTContext<'a>,
    ) -> Result<(DTPublicValues, [u8; 32], ExecutionReport), ExecutionError> {
        // context.subproof_verifier = Some(self);

        let program = Program::from(elf).expect("execute: failed to parse program from ELF");
        let mut runtime = Executor::with_context(program, dt_stark::DTCoreOpts::default(), context);

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

    /// Generate shard proofs which split up and prove the valid execution of a RISC-V program with
    /// the core prover. Uses the provided context.
    #[instrument(name = "prove_core", level = "info", skip_all)]
    pub fn prove_core<'a>(
        &'a self,
        pk_d: &<<C as DTProverComponents>::CoreProver as PolyAirMachineProver<
            CoreSC,
            RiscvPolyAir<Val<CoreSC>>,
            POLYAIR_EXT_DEGREE,
        >>::DeviceProvingKey,
        program: Program,
        stdin: &DTStdin,
        opts: DTProverOpts,
        context: DTContext<'a>,
    ) -> Result<DTCoreProof, DTCoreProverError> {
        let span = tracing::Span::current().clone();
        std::thread::scope(|s| {
            let _span = span.enter();
            let proof_capacity = opts.core_opts.shard_batch_size.max(1);
            let (proof_tx, proof_rx) = sync_channel(proof_capacity);
            let (shape_tx, shape_rx) = channel();

            let span = tracing::Span::current().clone();
            let producer = s.spawn(move || {
                let _span = span.enter();
                self.prove_core_stream_bounded(
                    pk_d, program, stdin, opts, context, proof_tx, shape_tx, None,
                )
            });

            // Shapes are diagnostic-only on this route. Drain them concurrently so a bounded
            // proof sender can never deadlock waiting for a fixed number of shapes.
            let shape_drain = s.spawn(move || for _ in shape_rx {});
            let shard_proofs = proof_rx.into_iter().collect();
            let (public_values_stream, cycles) =
                producer.join().expect("execute: trace-gen thread panicked")?;
            shape_drain.join().expect("core shape drain thread panicked");
            let public_values = DTPublicValues::from(&public_values_stream);
            Ok(DTCoreProof {
                proof: DTCoreProofData(shard_proofs),
                stdin: stdin.clone(),
                public_values,
                cycles,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_core_stream_bounded<'a>(
        &'a self,
        pk_d: &<<C as DTProverComponents>::CoreProver as PolyAirMachineProver<
            CoreSC,
            RiscvPolyAir<Val<CoreSC>>,
            POLYAIR_EXT_DEGREE,
        >>::DeviceProvingKey,
        program: Program,
        stdin: &DTStdin,
        opts: DTProverOpts,
        mut context: DTContext<'a>,
        proof_tx: SyncSender<SCShardProof<CoreSC>>,
        shape_tx: Sender<(OrderedShape, bool)>,
        count_ticket_tx: Option<Sender<u32>>,
    ) -> Result<(Vec<u8>, u64), DTCoreProverError> {
        context.subproof_verifier = Some(self);

        dt_core_machine::utils::prove_polyair::sc_prove_core_stream_bounded::<
            _,
            C::CoreProver,
            POLYAIR_EXT_DEGREE,
        >(
            &self.core_prover,
            pk_d,
            program,
            stdin,
            opts.core_opts,
            context,
            proof_tx,
            shape_tx,
            count_ticket_tx,
            None,
        )
    }

    /// Core proving with native-recursion child recording attached to the shard stream.
    #[cfg(feature = "native-recursion")]
    pub fn prove_core_with_native_handoff<'a>(
        &'a self,
        pk_d: &<<C as DTProverComponents>::CoreProver as PolyAirMachineProver<
            CoreSC,
            RiscvPolyAir<Val<CoreSC>>,
            POLYAIR_EXT_DEGREE,
        >>::DeviceProvingKey,
        vk: &DTVerifyingKey,
        program: Program,
        stdin: &DTStdin,
        opts: DTProverOpts,
        context: DTContext<'a>,
    ) -> Result<native_backend::NativeCoreHandoff, DTRecursionProverError> {
        let backend = self.native_backend()?;
        let pipeline = backend.pipeline_options()?;
        let request =
            native_recursion::compress_dt::NativeRecursionRequest::new().map_err(|err| {
                DTRecursionProverError::RuntimeError(format!("native recursion request: {err}"))
            })?;
        let pipeline_start = Instant::now();
        let span = tracing::Span::current().clone();
        let count_ticket_slot =
            Arc::new(Mutex::new(None::<native_backend::NativeCountTicketTelemetry>));
        let pipeline_output = std::thread::scope(|scope| {
            let request = &request;
            let _span = span.enter();
            let (proof_tx, proof_rx) = sync_channel(pipeline.proof_queue_capacity);
            let (shape_tx, shape_rx) = channel();
            let (count_ticket_tx, count_ticket_rx) = channel::<u32>();
            let core_done_at = Arc::new(Mutex::new(None::<Instant>));

            let span = tracing::Span::current().clone();
            let producer_done_at = Arc::clone(&core_done_at);
            let producer = scope.spawn(move || {
                let _span = span.enter();
                let result = self.prove_core_stream_bounded(
                    pk_d,
                    program,
                    stdin,
                    opts,
                    context,
                    proof_tx,
                    shape_tx,
                    Some(count_ticket_tx),
                );
                *producer_done_at.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                    Some(Instant::now());
                result
            });
            let shape_drain = scope.spawn(move || for _ in shape_rx {});

            let proofs_received = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let plan_gate = Arc::new(native_backend::PlanGate::new());
            // Count ticket (S3/S5): stamp telemetry and resolve the plan gate.
            // The ticket travels on its own control channel; a producer that
            // ends without sending one resolves the gate to an error so the
            // pipeline fails cleanly instead of waiting forever.
            {
                let count_ticket_slot = Arc::clone(&count_ticket_slot);
                let proofs_received = Arc::clone(&proofs_received);
                let plan_gate = Arc::clone(&plan_gate);
                scope.spawn(move || match count_ticket_rx.recv() {
                    Ok(count) => {
                        let ready_ms = pipeline_start.elapsed().as_millis();
                        let received = proofs_received.load(std::sync::atomic::Ordering::SeqCst);
                        tracing::info!(
                            "native count ticket: count={count} ready_ms={ready_ms} \
                             proofs_received={received}"
                        );
                        *count_ticket_slot
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            Some(native_backend::NativeCountTicketTelemetry {
                                count,
                                ready_ms,
                                proofs_received_at_ready: received,
                            });
                        plan_gate.resolve(
                            native_backend::build_tree_plan(
                                count as usize,
                                pipeline.early_lift_workers,
                            )
                            .map_err(|err| err.to_string()),
                        );
                    }
                    Err(_) => plan_gate
                        .resolve(Err("core producer ended without a count ticket".to_string())),
                });
            }
            // Pre-plan intake: absorb the proof stream unboundedly until the
            // count ticket resolves the plan (matching the raw path, which by
            // definition holds every shard), then forward routed jobs into the
            // bounded recorder queue, restoring producer backpressure.
            let (record_job_tx, record_job_rx) =
                sync_channel::<(usize, usize, usize, SCShardProof<CoreSC>)>(
                    pipeline.proof_queue_capacity,
                );
            let record_job_rx = Arc::new(Mutex::new(record_job_rx));
            let intake_received = Arc::clone(&proofs_received);
            let intake_gate = Arc::clone(&plan_gate);
            let intake = scope.spawn(move || {
                let mut pending = std::collections::VecDeque::new();
                let mut plan: Option<Arc<tree_plan::TreePlan>> = None;
                let mut next_idx = 0usize;
                for shard in proof_rx {
                    intake_received.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    pending.push_back((next_idx, shard));
                    next_idx += 1;
                    if plan.is_none() {
                        match intake_gate.try_get() {
                            Some(Ok(resolved)) => plan = Some(resolved),
                            Some(Err(_)) => return next_idx,
                            None => continue,
                        }
                    }
                    let Some(plan) = plan.as_ref() else { continue };
                    let routes = match native_backend::shard_routes(plan) {
                        Ok(routes) => routes,
                        Err(_) => return next_idx,
                    };
                    while let Some((shard_idx, shard)) = pending.pop_front() {
                        let Some(&(node_index, proof_idx)) = routes.get(shard_idx) else {
                            // More shards than the announced count; the final
                            // stream-vs-ticket assertion reports it.
                            return next_idx;
                        };
                        if record_job_tx.send((shard_idx, node_index, proof_idx, shard)).is_err() {
                            return next_idx;
                        }
                    }
                }
                // The stream may finish before the ticket resolves; wait, then
                // drain whatever was buffered.
                let Ok(plan) = intake_gate.wait() else { return next_idx };
                let Ok(routes) = native_backend::shard_routes(&plan) else { return next_idx };
                while let Some((shard_idx, shard)) = pending.pop_front() {
                    let Some(&(node_index, proof_idx)) = routes.get(shard_idx) else {
                        return next_idx;
                    };
                    if record_job_tx.send((shard_idx, node_index, proof_idx, shard)).is_err() {
                        return next_idx;
                    }
                }
                next_idx
            });
            let (result_tx, result_rx) = sync_channel::<(
                usize,
                Result<native_backend::CorePrerecordEntry, _>,
            )>(pipeline.proof_queue_capacity);
            let mut recorder_handles = Vec::with_capacity(pipeline.recorder_workers);
            for _ in 0..pipeline.recorder_workers {
                let record_job_rx = Arc::clone(&record_job_rx);
                let result_tx = result_tx.clone();
                recorder_handles.push(scope.spawn(move || loop {
                    let job = record_job_rx
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .recv();
                    let Ok((shard_idx, _node_index, proof_idx, shard)) = job else {
                        break;
                    };
                    let recorded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        backend.build_core_prerecord(request, &vk.vk, shard, shard_idx, proof_idx)
                    }))
                    .unwrap_or_else(|_| {
                        Err(DTRecursionProverError::RuntimeError(format!(
                            "native core recorder panicked at shard {shard_idx}"
                        )))
                    });
                    if result_tx.send((shard_idx, recorded)).is_err() {
                        break;
                    }
                }));
            }
            drop(record_job_rx);
            drop(result_tx);

            let (lift_job_tx, lift_job_rx) = sync_channel::<(
                usize,
                usize,
                Vec<native_backend::CorePrerecordEntry>,
                u128,
                Option<usize>,
            )>(pipeline.early_lift_queue_capacity);
            let lift_job_rx = Arc::new(Mutex::new(lift_job_rx));
            let (lift_result_tx, lift_result_rx) = sync_channel::<(
                usize,
                Result<native_backend::EarlyLiftResult, DTRecursionProverError>,
            )>(pipeline.early_lift_workers);
            let mut lift_handles = Vec::with_capacity(pipeline.early_lift_workers);
            for _ in 0..pipeline.early_lift_workers {
                let lift_job_rx = Arc::clone(&lift_job_rx);
                let lift_result_tx = lift_result_tx.clone();
                lift_handles.push(scope.spawn(move || loop {
                    let job =
                        lift_job_rx.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).recv();
                    let Ok((node_index, first_shard, entries, ready_ms, l3_parent_slot)) = job
                    else {
                        break;
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        backend.prove_early_lift_bin(
                            request,
                            node_index,
                            first_shard,
                            entries,
                            pipeline_start,
                            ready_ms,
                            l3_parent_slot,
                        )
                    }))
                    .unwrap_or_else(|_| {
                        Err(DTRecursionProverError::RuntimeError(format!(
                            "early lift worker panicked at node {node_index}"
                        )))
                    });
                    if lift_result_tx.send((node_index, result)).is_err() {
                        break;
                    }
                }));
            }
            drop(lift_result_tx);

            fn accept_lift_result(
                node_index: usize,
                result: Result<native_backend::EarlyLiftResult, DTRecursionProverError>,
                pending: &mut BTreeMap<
                    usize,
                    Result<native_backend::EarlyLiftResult, DTRecursionProverError>,
                >,
                ordered: &mut Vec<native_backend::EarlyLiftResult>,
                next_node: &mut usize,
                first_error: &mut Option<DTRecursionProverError>,
            ) {
                if pending.insert(node_index, result).is_some() && first_error.is_none() {
                    *first_error = Some(DTRecursionProverError::RuntimeError(format!(
                        "duplicate early lift result for node {node_index}"
                    )));
                }
                while let Some(result) = pending.remove(next_node) {
                    match result {
                        Ok(lift) => ordered.push(lift),
                        Err(err) if first_error.is_none() => *first_error = Some(err),
                        Err(_) => {}
                    }
                    *next_node += 1;
                }
            }

            let mut pending = BTreeMap::new();
            let mut next_shard_idx = 0usize;
            let mut current_lift_bin = Vec::with_capacity(native_backend::NATIVE_MAX_NODE_ARITY);
            let mut child_wall_ms = Vec::new();
            let mut lift_jobs_sent = 0usize;
            let mut lift_results_pending: BTreeMap<
                usize,
                Result<native_backend::EarlyLiftResult, DTRecursionProverError>,
            > = BTreeMap::new();
            let mut preproved_lifts: Vec<native_backend::EarlyLiftResult> = Vec::new();
            let mut next_lift_node = 0usize;
            let mut first_error = None;
            let l3_prerecord_summary = native_backend::L3PrerecordSummary::default();
            // A record can only exist after the gate resolved to a plan, so
            // this never blocks and never sees the gate's error arm.
            let mut coordinator_plan: Option<Arc<tree_plan::TreePlan>> = None;
            for (shard_idx, recorded) in result_rx {
                if pending.insert(shard_idx, recorded).is_some() && first_error.is_none() {
                    first_error = Some(DTRecursionProverError::RuntimeError(format!(
                        "duplicate native core recorder result for shard {shard_idx}"
                    )));
                }
                while let Some(recorded) = pending.remove(&next_shard_idx) {
                    match recorded {
                        Ok(record) => {
                            child_wall_ms.push(record.wall_ms());
                            if first_error.is_none() {
                                if coordinator_plan.is_none() {
                                    match plan_gate.wait() {
                                        Ok(resolved) => coordinator_plan = Some(resolved),
                                        Err(err) => {
                                            first_error =
                                                Some(DTRecursionProverError::RuntimeError(err));
                                            next_shard_idx += 1;
                                            continue;
                                        }
                                    }
                                }
                                let plan = coordinator_plan.as_ref().expect("resolved above");
                                let spans = native_backend::lift_spans(plan)?;
                                let (span_start, span_end) = spans[lift_jobs_sent];
                                let l3_parent_slot =
                                    native_backend::lift_l3_slot(plan, lift_jobs_sent)?;
                                current_lift_bin.push(record);
                                if current_lift_bin.len() == span_end - span_start {
                                    let ready_ms = pipeline_start.elapsed().as_millis();
                                    let mut job = (
                                        lift_jobs_sent,
                                        span_start,
                                        std::mem::take(&mut current_lift_bin),
                                        ready_ms,
                                        l3_parent_slot,
                                    );
                                    loop {
                                        match lift_job_tx.try_send(job) {
                                            Ok(()) => {
                                                lift_jobs_sent += 1;
                                                break;
                                            }
                                            Err(TrySendError::Full(returned)) => {
                                                job = returned;
                                                match lift_result_rx.recv() {
                                                    Ok((node_index, result)) => accept_lift_result(
                                                        node_index,
                                                        result,
                                                        &mut lift_results_pending,
                                                        &mut preproved_lifts,
                                                        &mut next_lift_node,
                                                        &mut first_error,
                                                    ),
                                                    Err(_) => {
                                                        first_error = Some(
                                                            DTRecursionProverError::RuntimeError(
                                                                "early lift result channel closed while dispatch was backpressured".into(),
                                                            ),
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(TrySendError::Disconnected(_)) => {
                                                first_error =
                                                    Some(DTRecursionProverError::RuntimeError(
                                                        "early lift job channel closed".into(),
                                                    ));
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(err) if first_error.is_none() => first_error = Some(err),
                        Err(_) => {}
                    }
                    next_shard_idx += 1;
                }
                while let Ok((node_index, result)) = lift_result_rx.try_recv() {
                    accept_lift_result(
                        node_index,
                        result,
                        &mut lift_results_pending,
                        &mut preproved_lifts,
                        &mut next_lift_node,
                        &mut first_error,
                    );
                }
            }

            for handle in recorder_handles {
                if handle.join().is_err() && first_error.is_none() {
                    first_error = Some(DTRecursionProverError::RuntimeError(
                        "native core recorder worker panicked".into(),
                    ));
                }
            }
            let assigned_shards = intake.join().unwrap_or(0);
            if (next_shard_idx != assigned_shards || !pending.is_empty()) && first_error.is_none() {
                first_error = Some(DTRecursionProverError::RuntimeError(format!(
                    "native core recorder lost ordering: completed={next_shard_idx}, assigned={assigned_shards}, pending={}",
                    pending.len()
                )));
            }

            // Plan spans close every bin exactly, including the last; a leftover
            // partial bin means the stream ended mid-span (count mismatch).
            if first_error.is_none() && !current_lift_bin.is_empty() {
                first_error = Some(DTRecursionProverError::RuntimeError(format!(
                    "core stream ended mid lift span with {} recorded shards pending",
                    current_lift_bin.len()
                )));
            }
            drop(lift_job_tx);
            for (node_index, result) in lift_result_rx {
                accept_lift_result(
                    node_index,
                    result,
                    &mut lift_results_pending,
                    &mut preproved_lifts,
                    &mut next_lift_node,
                    &mut first_error,
                );
            }
            for handle in lift_handles {
                if handle.join().is_err() && first_error.is_none() {
                    first_error = Some(DTRecursionProverError::RuntimeError(
                        "early lift worker panicked".into(),
                    ));
                }
            }
            if (next_lift_node != lift_jobs_sent || !lift_results_pending.is_empty()) &&
                first_error.is_none()
            {
                first_error = Some(DTRecursionProverError::RuntimeError(format!(
                    "early lift result ordering mismatch: completed={next_lift_node}, dispatched={lift_jobs_sent}, pending={}",
                    lift_results_pending.len()
                )));
            }

            let producer_result = producer.join().map_err(|_| {
                DTRecursionProverError::RuntimeError("core proving thread panicked".into())
            })?;
            shape_drain.join().map_err(|_| {
                DTRecursionProverError::RuntimeError("core shape drain thread panicked".into())
            })?;
            let (public_values_stream, _cycles) = producer_result.map_err(|err| {
                DTRecursionProverError::RuntimeError(format!("core prove before prerecord: {err}"))
            })?;
            if let Some(err) = first_error {
                return Err(err);
            }
            let tail_ms = core_done_at
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .map(|done| done.elapsed().as_millis())
                .unwrap_or(0);
            let public_values = DTPublicValues::from(&public_values_stream);
            Ok((
                coordinator_plan.ok_or_else(|| {
                    DTRecursionProverError::RuntimeError(
                        "core stream finished without installing TreePlan".into(),
                    )
                })?,
                public_values,
                next_shard_idx,
                preproved_lifts,
                child_wall_ms,
                tail_ms,
                l3_prerecord_summary,
            ))
        })?;
        let (
            plan,
            public_values,
            shard_count,
            preproved_lifts,
            child_wall_ms,
            tail_ms,
            l3_prerecord_summary,
        ) = pipeline_output;
        let pipeline_wall_ms = pipeline_start.elapsed().as_millis();
        let count_ticket =
            count_ticket_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
        if count_ticket.is_none() {
            // The producer completed without announcing its count — the S3
            // control channel is wired on every streamed path, so absence is
            // a plumbing regression worth failing loudly on.
            return Err(DTRecursionProverError::RuntimeError(
                "core producer finished without emitting the count ticket".into(),
            ));
        }
        let batch = backend.finish_core_prerecords(
            plan,
            request,
            shard_count,
            preproved_lifts,
            child_wall_ms,
            tail_ms,
            pipeline_wall_ms,
            &vk.vk,
            pipeline,
            l3_prerecord_summary,
            count_ticket,
        )?;
        Ok(native_backend::NativeCoreHandoff::new(public_values, batch))
    }

    /// Compress a request-owned native handoff whose core shards have already been replaced by
    /// authenticated lift proofs. The handoff is never serialized or stored in backend-global
    /// state.
    #[cfg(feature = "native-recursion")]
    pub fn compress_native_handoff(
        &self,
        vk: &DTVerifyingKey,
        handoff: native_backend::NativeCoreHandoff,
        opts: DTProverOpts,
    ) -> Result<DTReduceProof<RootSC>, DTRecursionProverError> {
        match RecursionBackend::resolve(opts.recursion_backend)
            .map_err(DTRecursionProverError::RuntimeError)?
        {
            RecursionBackend::Native => {
                self.native_backend()?.compress_native(vk, handoff.into_batch(), &opts)
            }
            RecursionBackend::Dsl => Err(DTRecursionProverError::RuntimeError(
                "native core handoff cannot be passed to the DSL backend".into(),
            )),
        }
    }

    /// Compress explicit in-memory core shards with the native ladder. This is the raw/saved-core
    /// stage entry used by diagnostics; it moves the shards into the same canonical handoff as the
    /// streamed route and performs no serialization or proof-configuration conversion.
    #[cfg(feature = "native-recursion")]
    pub fn compress_native_core_shards(
        &self,
        vk: &DTVerifyingKey,
        shards: Vec<SCShardProof<CoreSC>>,
        opts: DTProverOpts,
    ) -> Result<DTReduceProof<RootSC>, DTRecursionProverError> {
        if RecursionBackend::resolve(opts.recursion_backend)
            .map_err(DTRecursionProverError::RuntimeError)? !=
            RecursionBackend::Native
        {
            return Err(DTRecursionProverError::RuntimeError(
                "compress_native_core_shards requires the native recursion backend".into(),
            ));
        }
        let backend = self.native_backend()?;
        let batch = backend.normalize_core_shards(&vk.vk, shards)?;
        backend.compress_native(vk, batch, &opts)
    }

    /// Reduce shards proofs to a single shard proof using the recursion prover.
    #[instrument(name = "compress", level = "info", skip_all)]
    pub fn compress(
        &self,
        vk: &DTVerifyingKey,
        proof: DTCoreProof,
        deferred_proofs: Vec<DTReduceProof<InnerSC>>,
        opts: DTProverOpts,
    ) -> Result<DTReduceProof<RootSC>, DTRecursionProverError> {
        // Backend dispatch: resolve opts → env → default-native, then enter exactly the
        // selected native or DSL implementation. Native errors never redirect into DSL.
        match RecursionBackend::resolve(opts.recursion_backend)
            .map_err(DTRecursionProverError::RuntimeError)?
        {
            RecursionBackend::Native => {
                #[cfg(feature = "native-recursion")]
                {
                    if !deferred_proofs.is_empty() {
                        return Err(DTRecursionProverError::RuntimeError(
                            "the native recursion backend does not accept deferred proofs; \
                             select the DSL backend (DT_RECURSION_BACKEND=dsl) for \
                             deferred-proof workloads"
                                .into(),
                        ));
                    }
                    let DTProofWithMetadata { proof: DTCoreProofData(shards), .. } = proof;
                    // Explicit raw/saved-core input is normalized once into the same owned
                    // preproved-lift handoff used by the streamed route. No serialization or
                    // proof-config conversion occurs at this boundary.
                    return self.compress_native_core_shards(vk, shards, opts);
                }
                #[cfg(not(feature = "native-recursion"))]
                {
                    return Err(DTRecursionProverError::RuntimeError(
                        "recursion backend 'native' selected, but dt-prover was built without \
                         the `native-recursion` feature; rebuild with it or select \
                         DT_RECURSION_BACKEND=dsl"
                            .into(),
                    ));
                }
            }
            RecursionBackend::Dsl => {}
        }

        #[allow(clippy::type_complexity)]
        enum TracesOrInput {
            ProgramRecordTraces(
                Box<(
                    Arc<RecursionProgram<SCField>>,
                    ExecutionRecord<SCField>,
                    Vec<(String, CompressedMatrix<SCField>)>,
                    bool,
                )>,
            ),
            CircuitWitness(Box<SCDTCircuitWitness>),
        }

        let compress_start = Instant::now();
        let proof_shard_count = proof.proof.0.len();

        // The batch size for reducing two layers of recursion.
        let batch_size = REDUCE_BATCH_SIZE;
        // The batch size for reducing the first layer of recursion.
        let first_layer_batch_size = 2;

        let shard_proofs = &proof.proof.0;

        // Generate the first layer inputs.
        let first_layer_inputs =
            self.get_first_layer_inputs(vk, shard_proofs, &deferred_proofs, first_layer_batch_size);

        // Calculate the expected height of the tree.
        let mut expected_height = if first_layer_inputs.len() == 1 { 0 } else { 1 };
        let num_first_layer_inputs = first_layer_inputs.len();
        let mut num_layer_inputs = num_first_layer_inputs;
        while num_layer_inputs > batch_size {
            num_layer_inputs = num_layer_inputs.div_ceil(2);
            expected_height += 1;
        }
        // Keep the recursion stage split explicit. Each prover below reads its
        // PCS/FRI/stacking settings from the WHIR JSON stage config:
        //   - pre-penultimate and earlier: compress_prover(default)
        //   - penultimate: shrink_prover(stage = "shrink")
        //   - root: root_shrink_prover(stage = "root_shrink")
        // With the ext5 config, only root_shrink has stacking=true.
        let profile_last_layers = env::var("DT_PROFILE_LAST_LAYERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        tracing::info!(
            "Compress mode: pre-penultimate=compress, penultimate=shrink, root=root_shrink; stage configs are loaded from WHIR JSON"
        );
        if profile_last_layers > 0 {
            tracing::info!(
                "Layer event profiling enabled for the last {} compress layers",
                profile_last_layers
            );
        }

        // Generate the proofs.
        let span = tracing::Span::current().clone();
        let (vk, proof) = thread::scope(|s| {
            let _span = span.enter();

            // Spawn a worker that sends the first layer inputs to a bounded channel.
            let input_sync = Arc::new(TurnBasedSync::new());
            let (input_tx, input_rx) = sync_channel::<(usize, usize, SCDTCircuitWitness, bool)>(
                opts.recursion_opts.checkpoints_channel_capacity,
            );
            let input_tx = Arc::new(Mutex::new(input_tx));
            {
                let input_tx = Arc::clone(&input_tx);
                let input_sync = Arc::clone(&input_sync);
                s.spawn(move || {
                    for (index, input) in first_layer_inputs.into_iter().enumerate() {
                        input_sync.wait_for_turn(index);
                        input_tx.lock().unwrap().send((index, 0, input, false)).unwrap();
                        input_sync.advance_turn();
                    }
                });
            }

            // Spawn workers who generate the records and traces.
            let record_and_trace_sync = Arc::new(TurnBasedSync::new());
            let (record_and_trace_tx, record_and_trace_rx) =
                sync_channel::<(usize, usize, TracesOrInput)>(
                    opts.recursion_opts.records_and_traces_channel_capacity,
                );
            let record_and_trace_tx = Arc::new(Mutex::new(record_and_trace_tx));
            let record_and_trace_rx = Arc::new(Mutex::new(record_and_trace_rx));
            let input_rx = Arc::new(Mutex::new(input_rx));
            for _ in 0..opts.recursion_opts.trace_gen_workers {
                let record_and_trace_sync = Arc::clone(&record_and_trace_sync);
                let record_and_trace_tx = Arc::clone(&record_and_trace_tx);
                let input_rx = Arc::clone(&input_rx);
                let span = tracing::debug_span!("generate records and traces");
                s.spawn(move || {
                    let _span = span.enter();
                    loop {
                        let received = { input_rx.lock().unwrap().recv() };
                        if let Ok((index, height, input, false)) = received {
                            // Get the program and witness stream.
                            let program_start = Instant::now();
                            let (program, witness_stream, use_shrink_reduce_program) = tracing::debug_span!(
                                "get program and witness stream"
                            )
                            .in_scope(|| match input {
                                SCDTCircuitWitness::Core(input) => {
                                    tracing::debug!("enter core proof witness");
                                    let mut witness_stream = Vec::new();
                                    Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
                                    (self.recursion_program(&input), witness_stream, false)
                                }
                                SCDTCircuitWitness::Deferred(input) => {
                                    tracing::debug!("enter deferred proof witness");
                                    let mut witness_stream = Vec::new();
                                    Witnessable::<InnerConfig>::write(&input, &mut witness_stream);
                                    (self.deferred_program(&input), witness_stream, false)
                                }
                                SCDTCircuitWitness::Compress(input) => {
                                    tracing::debug!("enter compress proof witness");

                                    let mut witness_stream = Vec::new();

                                    let input_with_merkle = self.make_merkle_proofs(input);
                                    let input_proof_machine = classify_reduce_batch_machine(
                                        &input_with_merkle.compress_val.vks_and_proofs,
                                    );

                                    Witnessable::<InnerConfig>::write(
                                        &input_with_merkle,
                                        &mut witness_stream,
                                    );

                                    let is_final_layer = height == expected_height;
                                    let is_penultimate_layer =
                                        expected_height > 0 && height + 1 == expected_height;
                                    let proof_count =
                                        input_with_merkle.compress_val.vks_and_proofs.len();
                                    // The top of the compress tree is normalized into shrink-shaped
                                    // proofs: penultimate reduces compress proofs with the `shrink`
                                    // stage, and root reduces shrink proofs with `root_shrink`.
                                    let use_shrink_reduce_program = if is_penultimate_layer {
                                        proof_count >= 1
                                    } else if is_final_layer {
                                        proof_count > 1
                                    } else {
                                        false
                                    };
                                    if use_shrink_reduce_program {
                                        let shrink_layer_desc = if is_penultimate_layer {
                                            "penultimate 2-to-1 reduce"
                                        } else {
                                            "root 2-to-1 reduce"
                                        };
                                        let stage_desc = if is_final_layer {
                                            "root_shrink"
                                        } else {
                                            "shrink"
                                        };
                                        tracing::info!(
                                            "Building {} program with shrink verifier stage={} at index={} height={}",
                                            shrink_layer_desc,
                                            stage_desc,
                                            index,
                                            height
                                        );
                                    }

                                    let program = if use_shrink_reduce_program {
                                        let verify_machine = input_proof_machine.verify_machine();
                                        tracing::info!(
                                            "Shrink reduce input proof machine={} index={} height={}",
                                            input_proof_machine.as_str(),
                                            index,
                                            height,
                                        );
                                        let shrink_shape = if is_final_layer {
                                            self.root_shrink_shape()
                                        } else {
                                            ShrinkAir::<SCField>::shrink_shape()
                                        };
                                        self.shrink_program(
                                            shrink_shape,
                                            &input_with_merkle,
                                            verify_machine,
                                            is_final_layer,
                                            true,
                                        )
                                    } else {
                                        self.compress_program(&input_with_merkle)
                                    };

                                    (program, witness_stream, use_shrink_reduce_program)
                                }
                            });
                            let program_ms = program_start.elapsed().as_millis();

                            // Print instruction statistics.
                            if std::env::var("DEBUG_INSTR").is_ok() {
                                use dt_recursion_core::Instruction;
                                let mut counts = std::collections::HashMap::new();
                                for instr in &program.inner {
                                    let name = match instr {
                                        Instruction::BaseAlu(_) => "BaseAlu",
                                        Instruction::ExtAlu(_) => "ExtAlu",
                                        Instruction::Mem(_) => "Mem",
                                        Instruction::Poseidon2(_) => "Poseidon2",
                                        Instruction::Select(_) => "Select",
                                        Instruction::Hint(_) => "Hint",
                                        Instruction::HintBits(_) => "HintBits",
                                        Instruction::HintExt2Felts(_) => "HintExt2Felts",
                                        Instruction::CommitPublicValues(_) => "CommitPublicValues",
                                        Instruction::Print(_) => "Print",
                                        Instruction::HintAddCurve(_) => "HintAddCurve",
                                        Instruction::PolyEval(_) => "PolyEval",
                                        Instruction::ExtExpReverseBits(_) => "ExtExpReverseBits",
                                        Instruction::SumcheckRound(_) => "SumcheckRound",
                                        Instruction::PrefixSumChecks(_) => "PrefixSumChecks",
                                        _ => "Other",
                                    };
                                    *counts.entry(name).or_insert(0usize) += 1;
                                }
                                let mut sorted: Vec<_> = counts.into_iter().collect();
                                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                                eprintln!("[INSTR] Program instruction counts:");
                                for (name, count) in &sorted {
                                    eprintln!("  {name} = {count}");
                                }
                            }

                            // Execute the runtime.
                            let exec_start = Instant::now();
                            let record = tracing::debug_span!("execute runtime").in_scope(|| {
                                let is_root_shrink_reduce_program =
                                    use_shrink_reduce_program && height == expected_height;
                                let perm = if is_root_shrink_reduce_program {
                                    self.root_shrink_prover.config().perm.clone()
                                } else if use_shrink_reduce_program {
                                    self.shrink_prover.config().perm.clone()
                                } else {
                                    self.compress_prover.config().perm.clone()
                                };
                                let mut runtime = RecursionRuntime::<
                                    Val<InnerSC>,
                                    Challenge<InnerSC>,
                                    _,
                                    INNER_SBOX_DEGREE,
                                >::new(
                                    program.clone(),
                                    perm,
                                );
                                runtime.witness_stream = witness_stream.into();
                                runtime
                                    .run()
                                    .map_err(|e| {
                                        DTRecursionProverError::RuntimeError(e.to_string())
                                    })
                                    .unwrap();
                                runtime.record
                            });
                            let exec_ms = exec_start.elapsed().as_millis();
                            tracing::trace!(
                                "RECURSION_EXEC index={} height={} exec_ms={}",
                                index,
                                height,
                                exec_ms
                            );

                            // Generate the dependencies.
                            let mut records = vec![record];
                            tracing::debug_span!("generate dependencies").in_scope(|| {
                                let is_root_shrink_reduce_program =
                                    use_shrink_reduce_program && height == expected_height;
                                if is_root_shrink_reduce_program {
                                    self.root_shrink_prover.machine().generate_dependencies(
                                        &mut records,
                                        &opts.recursion_opts,
                                        None,
                                    )
                                } else if use_shrink_reduce_program {
                                    self.shrink_prover.machine().generate_dependencies(
                                        &mut records,
                                        &opts.recursion_opts,
                                        None,
                                    )
                                } else {
                                    self.compress_prover.machine().generate_dependencies(
                                        &mut records,
                                        &opts.recursion_opts,
                                        None,
                                    )
                                }
                            });

                            // Generate the traces.
                            let record = records
                                .into_iter()
                                .next()
                                .expect("recursion: expected at least one record from trace gen");
                            let layer_distance_to_root = expected_height.saturating_sub(height);
                            if profile_last_layers > 0 && layer_distance_to_root < profile_last_layers
                            {
                                tracing::info!(
                                    "LAYER_EVENT_STATS index={} height={} distance_to_root={} use_shrink_reduce_program={} \
                                     mem_const={} mem_var={} base_alu={} ext_alu={} p2_wide={} p2_skinny={} \
                                     select={} poly_eval={} eerb={} sumcheck_round={} prefix_sum={}",
                                    index,
                                    height,
                                    layer_distance_to_root,
                                    use_shrink_reduce_program,
                                    record.mem_const_count,
                                    record.mem_var_events.len(),
                                    record.base_alu_events.len(),
                                    record.ext_alu_events.len(),
                                    record.poseidon2_events.len(),
                                    record.poseidon2_skinny_events.len(),
                                    record.select_events.len(),
                                    record.poly_eval_events.len(),
                                    record.ext_exp_reverse_bits_events.len(),
                                    record.sumcheck_round_events.len(),
                                    record.prefix_sum_checks_events.len(),
                                );
                                if use_shrink_reduce_program {
                                    if let Some(shape) = program.shape.as_ref() {
                                        let shape_map = shape.clone_into_hash_map();
                                        let mut utilization = Vec::new();
                                        for (chip, actual_rows) in
                                            ShrinkAir::<SCField>::shrink_heights(&program)
                                        {
                                            if let Some(log_h) = shape_map.get(&chip) {
                                                let capacity = 1usize << *log_h;
                                                let usage_pct =
                                                    (actual_rows as f64 * 100.0) / capacity as f64;
                                                utilization.push((
                                                    chip,
                                                    *log_h,
                                                    actual_rows,
                                                    usage_pct,
                                                ));
                                            }
                                        }
                                        utilization.sort_by(|a, b| b.3.total_cmp(&a.3));
                                        tracing::info!(
                                            "LAYER_SHRINK_UTIL index={} height={} utilization={:?}",
                                            index,
                                            height,
                                            utilization
                                        );
                                    }
                                }
                            }
                            let tracegen_start = Instant::now();
                            let traces = tracing::debug_span!("generate traces").in_scope(|| {
                                let is_root_shrink_reduce_program =
                                    use_shrink_reduce_program && height == expected_height;
                                if is_root_shrink_reduce_program {
                                    self.root_shrink_prover.generate_traces(&record)
                                } else if use_shrink_reduce_program {
                                    self.shrink_prover.generate_traces(&record)
                                } else {
                                    self.compress_prover.generate_traces(&record)
                                }
                            });
                            let tracegen_ms = tracegen_start.elapsed().as_millis();

                            // Stage ledger (opt-in, output-only).
                            if let Some(dir) = stage_ledger::ledger_dir() {
                                stage_ledger::append(
                                    &dir,
                                    "dsl-nodes.jsonl",
                                    &serde_json::json!({
                                        "phase": "tracegen",
                                        "index": index,
                                        "height": height,
                                        "use_shrink_reduce_program": use_shrink_reduce_program,
                                        "program_ms": program_ms,
                                        "exec_ms": exec_ms,
                                        "tracegen_ms": tracegen_ms,
                                        "chips": stage_ledger::chip_rows(&traces),
                                    }),
                                );
                            }

                            // Wait for our turn to update the state.
                            record_and_trace_sync.wait_for_turn(index);

                            // Send the record and traces to the worker.
                            record_and_trace_tx
                                .lock()
                                .unwrap()
                                .send((
                                    index,
                                    height,
                                    TracesOrInput::ProgramRecordTraces(Box::new((
                                        program,
                                        record,
                                        traces,
                                        use_shrink_reduce_program,
                                    ))),
                                ))
                                .unwrap();

                            // Advance the turn.
                            record_and_trace_sync.advance_turn();
                        } else if let Ok((index, height, input, true)) = received {
                            record_and_trace_sync.wait_for_turn(index);

                            // Send the record and traces to the worker.
                            record_and_trace_tx
                                .lock()
                                .unwrap()
                                .send((
                                    index,
                                    height,
                                    TracesOrInput::CircuitWitness(Box::new(input)),
                                ))
                                .unwrap();

                            // Advance the turn.
                            record_and_trace_sync.advance_turn();
                        } else {
                            break;
                        }
                    }
                });
            }

            // Spawn workers who generate the compress proofs.
            let proofs_sync = Arc::new(TurnBasedSync::new());
            enum ProofsChannel {
                Inner(usize, usize, SCStarkVerifyingKey<InnerSC>, SCShardProof<InnerSC>),
                Root(usize, usize, SCStarkVerifyingKey<RootSC>, SCShardProof<RootSC>),
            }
            enum ProvedNode {
                Inner(SCStarkVerifyingKey<InnerSC>, SCShardProof<InnerSC>),
                Root(SCStarkVerifyingKey<RootSC>, SCShardProof<RootSC>),
            }
            let (proofs_tx, proofs_rx) = sync_channel::<ProofsChannel>(num_first_layer_inputs * 2);
            let proofs_tx: Arc<Mutex<SyncSender<ProofsChannel>>> = Arc::new(Mutex::new(proofs_tx));
            let proofs_rx: Arc<Mutex<Receiver<ProofsChannel>>> = Arc::new(Mutex::new(proofs_rx));
            let mut prover_handles = Vec::new();
            // GPU 单卡:多 worker 并行跑 compress prover 会抢同一张卡竞态,强制单 worker 串行消费。
            for _ in 0..1 {
                let prover_sync = Arc::clone(&proofs_sync);
                let record_and_trace_rx = Arc::clone(&record_and_trace_rx);
                let proofs_tx = Arc::clone(&proofs_tx);
                let span = tracing::debug_span!("prove");
                let handle = s.spawn(move || {
                    let _span = span.enter();
                    loop {
                        let received = { record_and_trace_rx.lock().unwrap().recv() };
                        if let Ok((index, height, TracesOrInput::ProgramRecordTraces(boxed_prt))) =
                            received
                        {
                            let (program, record, traces, use_shrink_reduce_program) = *boxed_prt;
                            tracing::debug_span!("batch").in_scope(|| {
                                let proof_start = Instant::now();

                                let is_root_layer = height == expected_height;
                                let is_penultimate_layer =
                                    expected_height > 0 && height + 1 == expected_height;
                                let (proved_node, setup_ms, commit_ms, open_ms) = if use_shrink_reduce_program {
                                    let shrink_layer_desc = if is_penultimate_layer {
                                        "penultimate 2-to-1 reduce"
                                    } else {
                                        "root 2-to-1 reduce"
                                    };
                                    let active_stage = if is_root_layer {
                                        "root_shrink"
                                    } else {
                                        "shrink"
                                    };
                                    tracing::info!(
                                        "Using {}_prover (stage={}) for {} index={} height={}",
                                        active_stage,
                                        active_stage,
                                        shrink_layer_desc,
                                        index, height,
                                    );

                                    if is_root_layer {
                                        let active_prover = &self.root_shrink_prover;

                                        // Get the keys.
                                        let setup_start = Instant::now();
                                        let (pk, vk) = tracing::debug_span!("Setup root_shrink program")
                                            .in_scope(|| active_prover.setup(&program));
                                        let setup_ms = setup_start.elapsed().as_millis();

                                        // Observe the proving key.
                                        let mut challenger = active_prover.config().mlchallenger();
                                        tracing::debug_span!("observe proving key").in_scope(|| {
                                            pk.observe_into(&mut challenger);
                                        });

                                        // Commit to the record and traces.
                                        let commit_start = Instant::now();
                                        let data = tracing::debug_span!("commit").in_scope(|| {
                                            active_prover.commit_with_pcs_stack_log_height(
                                                &record,
                                                traces,
                                                pk.preprocessed_pcs_stack_log_height(),
                                            )
                                        });
                                        let commit_ms = commit_start.elapsed().as_millis();

                                        // Generate the proof.
                                        let open_start = Instant::now();
                                        let proof = tracing::debug_span!("open").in_scope(|| {
                                            active_prover
                                                .open(
                                                    &pk,
                                                    data,
                                                    &mut challenger,
                                                    num_skip_rounds(),
                                                    chip_log_height_threshold(),
                                                )
                                                .unwrap()
                                        });
                                        let open_ms = open_start.elapsed().as_millis();

                                        #[cfg(feature = "debug")]
                                        active_prover
                                            .machine()
                                            .verify(
                                                &vk,
                                                &SCMachineProof {
                                                    shard_proofs: vec![proof.clone()],
                                                },
                                                &mut active_prover.config().challenger(),
                                                num_skip_rounds(),
                                                chip_log_height_threshold(),
                                            )
                                            .unwrap();

                                        (ProvedNode::Root(vk, proof), setup_ms, commit_ms, open_ms)
                                    } else {
                                        let active_prover = &self.shrink_prover;

                                        // Get the keys.
                                        let setup_start = Instant::now();
                                        let (pk, vk) = tracing::debug_span!("Setup shrink program")
                                            .in_scope(|| active_prover.setup(&program));
                                        let setup_ms = setup_start.elapsed().as_millis();

                                        // Observe the proving key.
                                        let mut challenger = active_prover.config().mlchallenger();
                                        tracing::debug_span!("observe proving key").in_scope(|| {
                                            pk.observe_into(&mut challenger);
                                        });

                                        // Debug cumulative sums if requested (only first proof).
                                        if std::env::var("DEBUG_COMPRESS").is_ok() && index == 0 {
                                            use dt_recursion_core::Instruction;
                                            use p3_field::Field;
                                            let mut base_alu_nonzero_mult = 0usize;
                                            let mut base_alu_zero_mult = 0usize;
                                            let mut ext_alu_nonzero_mult = 0usize;
                                            let mut ext_alu_zero_mult = 0usize;
                                            for instr in &program.inner {
                                                match instr {
                                                    Instruction::BaseAlu(i) => {
                                                        if Field::is_zero(&i.mult) {
                                                            base_alu_zero_mult += 1;
                                                        } else {
                                                            base_alu_nonzero_mult += 1;
                                                        }
                                                    }
                                                    Instruction::ExtAlu(i) => {
                                                        if Field::is_zero(&i.mult) {
                                                            ext_alu_zero_mult += 1;
                                                        } else {
                                                            ext_alu_nonzero_mult += 1;
                                                        }
                                                    }
                                                    _ => {}
                                                };
                                            }
                                            eprintln!(
                                                "[MULTS] index={index} BaseAlu: nonzero_mult={base_alu_nonzero_mult}, zero_mult={base_alu_zero_mult}"
                                            );
                                            eprintln!(
                                                "[MULTS] index={index} ExtAlu: nonzero_mult={ext_alu_nonzero_mult}, zero_mult={ext_alu_zero_mult}"
                                            );
                                            eprintln!(
                                                "[COUNTS] index={} base_alu_events={}, ext_alu_events={}",
                                                index,
                                                record.base_alu_events.len(),
                                                record.ext_alu_events.len()
                                            );

                                            eprintln!(
                                                "[DEBUG_COMPRESS] shrink-root path inspected counts for index={index}"
                                            );
                                        }

                                        // Commit to the record and traces.
                                        let commit_start = Instant::now();
                                        let data = tracing::debug_span!("commit").in_scope(|| {
                                            active_prover.commit_with_pcs_stack_log_height(
                                                &record,
                                                traces,
                                                pk.preprocessed_pcs_stack_log_height(),
                                            )
                                        });
                                        let commit_ms = commit_start.elapsed().as_millis();

                                        // Generate the proof.
                                        let open_start = Instant::now();
                                        let proof = tracing::debug_span!("open").in_scope(|| {
                                            active_prover
                                                .open(
                                                    &pk,
                                                    data,
                                                    &mut challenger,
                                                    num_skip_rounds(),
                                                    chip_log_height_threshold(),
                                                )
                                                .unwrap()
                                        });
                                        let open_ms = open_start.elapsed().as_millis();

                                        #[cfg(feature = "debug")]
                                        active_prover
                                            .machine()
                                            .verify(
                                                &vk,
                                                &SCMachineProof {
                                                    shard_proofs: vec![proof.clone()],
                                                },
                                                &mut active_prover.config().challenger(),
                                                num_skip_rounds(),
                                                chip_log_height_threshold(),
                                            )
                                            .unwrap();

                                        (ProvedNode::Inner(vk, proof), setup_ms, commit_ms, open_ms)
                                    }
                                } else {
                                    let active_prover = &self.compress_prover;

                                    // Get the keys.
                                    let setup_start = Instant::now();
                                    let (pk, vk) = tracing::debug_span!("Setup compress program")
                                        .in_scope(|| active_prover.setup(&program));
                                    let setup_ms = setup_start.elapsed().as_millis();

                                    // Observe the proving key.
                                    let mut challenger = active_prover.config().mlchallenger();
                                    tracing::debug_span!("observe proving key").in_scope(|| {
                                        pk.observe_into(&mut challenger);
                                    });

                                    // Debug cumulative sums if requested (only first proof).
                                    if std::env::var("DEBUG_COMPRESS").is_ok() && index == 0 {
                                        use dt_recursion_core::Instruction;
                                        use p3_field::Field;
                                        let mut base_alu_nonzero_mult = 0usize;
                                        let mut base_alu_zero_mult = 0usize;
                                        let mut ext_alu_nonzero_mult = 0usize;
                                        let mut ext_alu_zero_mult = 0usize;
                                        for instr in &program.inner {
                                            match instr {
                                                Instruction::BaseAlu(i) => {
                                                    if Field::is_zero(&i.mult) {
                                                        base_alu_zero_mult += 1;
                                                    } else {
                                                        base_alu_nonzero_mult += 1;
                                                    }
                                                }
                                                Instruction::ExtAlu(i) => {
                                                    if Field::is_zero(&i.mult) {
                                                        ext_alu_zero_mult += 1;
                                                    } else {
                                                        ext_alu_nonzero_mult += 1;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                        eprintln!(
                                            "[MULTS] index={index} BaseAlu: nonzero_mult={base_alu_nonzero_mult}, zero_mult={base_alu_zero_mult}"
                                        );
                                        eprintln!(
                                            "[MULTS] index={index} ExtAlu: nonzero_mult={ext_alu_nonzero_mult}, zero_mult={ext_alu_zero_mult}"
                                        );
                                        eprintln!(
                                            "[COUNTS] index={} base_alu_events={}, ext_alu_events={}",
                                            index,
                                            record.base_alu_events.len(),
                                            record.ext_alu_events.len()
                                        );

                                        eprintln!(
                                            "[DEBUG_COMPRESS] PolyAir debug_constraints is not wired; skipped constraint replay for index={index}"
                                        );
                                    }

                                    // Commit to the record and traces.
                                    let commit_start = Instant::now();
                                    let data = tracing::debug_span!("commit").in_scope(|| {
                                        active_prover.commit_with_pcs_stack_log_height(
                                            &record,
                                            traces,
                                            pk.preprocessed_pcs_stack_log_height(),
                                        )
                                    });
                                    let commit_ms = commit_start.elapsed().as_millis();

                                    // Generate the proof.
                                    let open_start = Instant::now();
                                    let proof = tracing::debug_span!("open").in_scope(|| {
                                        active_prover
                                            .open(
                                                &pk,
                                                data,
                                                &mut challenger,
                                                num_skip_rounds(),
                                                chip_log_height_threshold(),
                                            )
                                            .unwrap()
                                    });
                                    let open_ms = open_start.elapsed().as_millis();

                                    // Verify the proof.
                                    #[cfg(feature = "debug")]
                                    active_prover
                                        .machine()
                                        .verify(
                                            &vk,
                                            &SCMachineProof {
                                                shard_proofs: vec![proof.clone()],
                                            },
                                            &mut active_prover.config().challenger(),
                                            num_skip_rounds(),
                                            chip_log_height_threshold(),
                                        )
                                        .unwrap();

                                    (ProvedNode::Inner(vk, proof), setup_ms, commit_ms, open_ms)
                                };

                                let proof_elapsed = proof_start.elapsed();
                                tracing::trace!(
                                    "COMPRESS_PROOF index={} height={} time_ms={}",
                                    index,
                                    height,
                                    proof_elapsed.as_millis()
                                );

                                // Stage ledger (opt-in, output-only).
                                if let Some(dir) = stage_ledger::ledger_dir() {
                                    let proof_bytes = match &proved_node {
                                        ProvedNode::Inner(_, proof) => {
                                            bincode::serialize(proof).map(|b| b.len()).unwrap_or(0)
                                        }
                                        ProvedNode::Root(_, proof) => {
                                            bincode::serialize(proof).map(|b| b.len()).unwrap_or(0)
                                        }
                                    };
                                    stage_ledger::append(
                                        &dir,
                                        "dsl-nodes.jsonl",
                                        &serde_json::json!({
                                            "phase": "prove",
                                            "index": index,
                                            "height": height,
                                            "use_shrink_reduce_program": use_shrink_reduce_program,
                                            "is_root_layer": is_root_layer,
                                            "is_penultimate_layer": is_penultimate_layer,
                                            "setup_ms": setup_ms,
                                            "commit_ms": commit_ms,
                                            "open_ms": open_ms,
                                            "total_ms": proof_elapsed.as_millis(),
                                            "proof_bytes": proof_bytes,
                                        }),
                                    );
                                }

                                // Wait for our turn to update the state.
                                prover_sync.wait_for_turn(index);

                                // Send the proof.
                                let proof_msg = match proved_node {
                                    ProvedNode::Inner(vk, proof) => {
                                        ProofsChannel::Inner(index, height, vk, proof)
                                    }
                                    ProvedNode::Root(vk, proof) => {
                                        ProofsChannel::Root(index, height, vk, proof)
                                    }
                                };
                                proofs_tx.lock().unwrap().send(proof_msg).unwrap();

                                // Advance the turn.
                                prover_sync.advance_turn();
                            });
                        } else if let Ok((
                            index,
                            height,
                            TracesOrInput::CircuitWitness(witness_box),
                        )) = received
                        {
                            let witness = *witness_box;
                            if let SCDTCircuitWitness::Compress(inner_witness) = witness {
                                let SCDTCompressWitnessValues { vks_and_proofs, is_complete: _ } =
                                    inner_witness;
                                assert!(vks_and_proofs.len() == 1);
                                let (vk, proof) = vks_and_proofs
                                    .last()
                                    .expect("compress: vks_and_proofs must be non-empty");
                                // Wait for our turn to update the state.
                                prover_sync.wait_for_turn(index);

                                // Send the proof.
                                proofs_tx
                                    .lock()
                                    .unwrap()
                                    .send(ProofsChannel::Inner(
                                        index,
                                        height,
                                        vk.clone(),
                                        proof.clone(),
                                    ))
                                    .unwrap();

                                // Advance the turn.
                                prover_sync.advance_turn();
                            }
                        } else {
                            break;
                        }
                    }
                });
                prover_handles.push(handle);
            }

            // Spawn a worker that generates inputs for the next layer.
            let handle = {
                let input_tx = Arc::clone(&input_tx);
                let proofs_rx = Arc::clone(&proofs_rx);
                let span = tracing::debug_span!("generate next layer inputs");
                s.spawn(move || {
                    let _span = span.enter();
                    let mut count = num_first_layer_inputs;
                    let mut batch: VecDeque<(
                        usize,
                        usize,
                        SCStarkVerifyingKey<InnerSC>,
                        SCShardProof<InnerSC>,
                    )> = VecDeque::new();
                    loop {
                        if expected_height == 0 {
                            break;
                        }
                        let received = { proofs_rx.lock().unwrap().recv() };
                        let stream_closed = match received {
                            Ok(ProofsChannel::Inner(index, height, vk, proof)) => {
                                batch.push_back((index, height, vk, proof));
                                false
                            }
                            Ok(ProofsChannel::Root(..)) => {
                                panic!("compress: root proof reached next-layer input builder")
                            }
                            Err(_) => true,
                        };

                        let mut done = false;
                        loop {
                            let Some((_, current_height, _, _)) = batch.front() else {
                                break;
                            };
                            let current_height = *current_height;
                            let same_height_count = batch
                                .iter()
                                .take_while(|(_, height, _, _)| *height == current_height)
                                .count();
                            let has_next_height = same_height_count < batch.len();
                            let height_finalized = has_next_height || stream_closed;

                            let group_size = if same_height_count >= batch_size {
                                batch_size
                            } else if height_finalized {
                                same_height_count
                            } else {
                                break;
                            };
                            if group_size == 0 {
                                break;
                            }

                            let mut inputs = Vec::with_capacity(group_size);
                            for _ in 0..group_size {
                                if let Some(item) = batch.pop_front() {
                                    inputs.push(item);
                                }
                            }
                            if inputs.is_empty() {
                                break;
                            }

                            let next_input_height = inputs[0].1 + 1;
                            let is_complete = next_input_height == expected_height;
                            let is_singleton_remainder_group =
                                group_size == 1 && group_size < batch_size && height_finalized;
                            let force_singleton_reduce = if is_singleton_remainder_group &&
                                expected_height > 0
                            {
                                let is_penultimate_input =
                                    next_input_height.saturating_add(1) == expected_height;
                                let is_before_penultimate_input =
                                    next_input_height.saturating_add(2) == expected_height;
                                // The final compress-tree layers must stay stage-homogeneous:
                                // penultimate reduces compress proofs into shrink proofs, and
                                // root reduces shrink proofs into a root_shrink proof. Singleton
                                // remainder nodes in those positions are reduced instead of
                                // passed through so the next stage never receives mixed proof
                                // machines.
                                is_penultimate_input || is_before_penultimate_input
                            } else {
                                false
                            };

                            let vks_and_proofs = inputs
                                .into_iter()
                                .map(|(_, _, vk, proof)| (vk, proof))
                                .collect::<Vec<_>>();
                            let input = SCDTCircuitWitness::Compress(SCDTCompressWitnessValues {
                                vks_and_proofs,
                                is_complete,
                            });

                            input_sync.wait_for_turn(count);
                            input_tx
                                .lock()
                                .unwrap()
                                .send((
                                    count,
                                    next_input_height,
                                    input,
                                    is_singleton_remainder_group && !force_singleton_reduce,
                                ))
                                .unwrap();
                            input_sync.advance_turn();
                            count += 1;

                            if is_complete {
                                done = true;
                                break;
                            }
                        }

                        if done || stream_closed {
                            break;
                        }
                    }
                })
            };

            // Wait for all the provers to finish.
            drop(input_tx);
            drop(record_and_trace_tx);
            drop(proofs_tx);

            for handle in prover_handles {
                handle.join().expect("compress: a reduce prover thread panicked");
            }
            handle.join().expect("compress: the record/trace gen thread panicked");
            tracing::debug!("joined handles");

            let final_msg = proofs_rx
                .lock()
                .expect("compress: proofs_rx mutex poisoned")
                .recv()
                .expect("compress: no final reduce proof received (sender dropped)");
            match final_msg {
                ProofsChannel::Root(_, _, vk, proof) => (vk, proof),
                ProofsChannel::Inner(_, _, _, _) => {
                    panic!("compress: expected final root_shrink proof, got intermediate proof")
                }
            }
        });

        // Stage ledger summary (opt-in, output-only).
        if let Some(dir) = stage_ledger::ledger_dir() {
            stage_ledger::append(
                &dir,
                "dsl-summary.jsonl",
                &serde_json::json!({
                    "backend": "dsl",
                    "shard_count": proof_shard_count,
                    "total_wall_ms": compress_start.elapsed().as_millis(),
                    "peak_rss_kb": stage_ledger::peak_rss_kb(),
                    "proof_bytes": bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0),
                }),
            );
        }

        Ok(DTReduceProof { vk, proof })
    }

    /// Wrap a reduce proof into a STARK proven over a SNARK-friendly field.
    #[instrument(name = "shrink", level = "info", skip_all)]
    pub fn shrink(
        &self,
        reduced_proof: DTReduceProof<InnerSC>,
        opts: DTProverOpts,
    ) -> Result<DTReduceProof<InnerSC>, DTRecursionProverError> {
        // Make the compress proof.
        let DTReduceProof { vk: compressed_vk, proof: compressed_proof } = reduced_proof;
        let input = SCDTCompressWitnessValues {
            vks_and_proofs: vec![(compressed_vk.clone(), compressed_proof)],
            is_complete: true,
        };

        let input_with_merkle = self.make_merkle_proofs(input);

        let program = self.shrink_program(
            ShrinkAir::<SCField>::shrink_shape(),
            &input_with_merkle,
            ShrinkVerifyMachine::RootShrink,
            true,
            false,
        );

        // Run the compress program.
        let mut runtime =
            RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _, INNER_SBOX_DEGREE>::new(
                program.clone(),
                self.shrink_prover.config().perm.clone(),
            );

        let mut witness_stream = Vec::new();
        Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);

        runtime.witness_stream = witness_stream.into();

        runtime.run().map_err(|e| DTRecursionProverError::RuntimeError(e.to_string()))?;

        runtime.print_stats();
        tracing::debug!("Shrink program executed successfully");

        let (shrink_pk, shrink_vk) =
            tracing::debug_span!("setup shrink").in_scope(|| self.shrink_prover.setup(&program));

        // Prove the compress program.
        let mut compress_challenger = self.shrink_prover.config().challenger();
        let shrink_time = Instant::now();
        let mut compress_proof = self
            .shrink_prover
            .prove(
                &shrink_pk,
                vec![runtime.record],
                &mut compress_challenger,
                opts.recursion_opts,
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .unwrap();
        tracing::info!("Shrink proving time: {:?}", shrink_time.elapsed().as_secs_f64());
        Ok(DTReduceProof {
            vk: shrink_vk,
            proof: compress_proof
                .shard_proofs
                .pop()
                .expect("shrink: expected exactly one shard proof"),
        })
    }

    // Wrap a reduce proof into a STARK proven over a SNARK-friendly field.
    #[instrument(name = "wrap_bn254", level = "info", skip_all)]
    pub fn wrap_bn254(
        &self,
        compressed_proof: DTReduceProof<InnerSC>,
        opts: DTProverOpts,
    ) -> Result<DTReduceProof<OuterSC>, DTRecursionProverError> {
        let DTReduceProof { vk: compressed_vk, proof: compressed_proof } = compressed_proof;
        let input = SCDTCompressWitnessValues {
            vks_and_proofs: vec![(compressed_vk, compressed_proof)],
            is_complete: true,
        };
        let input_with_vk = self.make_merkle_proofs(input);

        let program = self.wrap_program();

        // Run the compress program.
        let mut runtime =
            RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _, INNER_SBOX_DEGREE>::new(
                program.clone(),
                self.shrink_prover.config().perm.clone(),
            );

        let mut witness_stream = Vec::new();
        Witnessable::<InnerConfig>::write(&input_with_vk, &mut witness_stream);

        runtime.witness_stream = witness_stream.into();

        runtime.run().map_err(|e| DTRecursionProverError::RuntimeError(e.to_string()))?;

        runtime.print_stats();
        tracing::debug!("wrap program executed successfully");

        // Setup the wrap program.
        let (wrap_pk, wrap_vk) =
            tracing::debug_span!("setup wrap").in_scope(|| self.wrap_prover.setup(&program));

        if self.wrap_vk.set(wrap_vk.clone()).is_ok() {
            tracing::debug!("wrap verifier key set");
        }

        // Prove the wrap program.
        let mut wrap_challenger = self.wrap_prover.config().challenger();
        let wrap_time = Instant::now();
        let mut wrap_proof = self
            .wrap_prover
            .prove(
                &wrap_pk,
                vec![runtime.record],
                &mut wrap_challenger,
                opts.recursion_opts,
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .unwrap();
        tracing::info!("wrap proving time: {:?}", wrap_time.elapsed().as_secs_f64());
        let mut wrap_challenger = self.wrap_prover.config().challenger();
        self.wrap_prover
            .machine()
            .verify(
                &wrap_vk,
                &wrap_proof,
                &mut wrap_challenger,
                num_skip_rounds(),
                chip_log_height_threshold(),
            )
            .unwrap();
        tracing::debug!("wrapping successful");

        Ok(DTReduceProof {
            vk: wrap_vk,
            proof: wrap_proof.shard_proofs.pop().expect("wrap: expected exactly one shard proof"),
        })
    }

    // Wrap the STARK proven over a SNARK-friendly field into a PLONK proof.
    // The gnark wrap path is quartic-only (BN254-facing); ext5 is shrink-only.
    #[cfg(not(feature = "ext5"))]
    #[instrument(name = "wrap_plonk_bn254", level = "info", skip_all)]
    pub fn wrap_plonk_bn254(
        &self,
        proof: DTReduceProof<OuterSC>,
        build_dir: &Path,
    ) -> PlonkBn254Proof {
        let input = SCDTCompressWitnessValues {
            vks_and_proofs: vec![(proof.vk.clone(), proof.proof.clone())],
            is_complete: true,
        };
        let vkey_hash = dt_vkey_digest_bn254(&proof);
        let committed_values_digest = dt_committed_values_digest_bn254(&proof);

        let mut witness = Witness::default();
        input.write(&mut witness);
        witness.write_committed_values_digest(committed_values_digest);
        witness.write_vkey_hash(vkey_hash);

        let prover = PlonkBn254Prover::new();
        let proof = prover.prove(witness, build_dir.to_path_buf());

        // Verify the proof.
        prover
            .verify(
                &proof,
                &vkey_hash.as_canonical_biguint(),
                &committed_values_digest.as_canonical_biguint(),
                build_dir,
            )
            .unwrap();

        proof
    }

    // Wrap the STARK proven over a SNARK-friendly field into a Groth16 proof.
    #[cfg(not(feature = "ext5"))]
    #[instrument(name = "wrap_groth16_bn254", level = "info", skip_all)]
    pub fn wrap_groth16_bn254(
        &self,
        proof: DTReduceProof<OuterSC>,
        build_dir: &Path,
    ) -> Groth16Bn254Proof {
        let input = SCDTCompressWitnessValues {
            vks_and_proofs: vec![(proof.vk.clone(), proof.proof.clone())],
            is_complete: true,
        };
        let vkey_hash = dt_vkey_digest_bn254(&proof);
        let committed_values_digest = dt_committed_values_digest_bn254(&proof);

        let mut witness = Witness::default();
        input.write(&mut witness);
        witness.write_committed_values_digest(committed_values_digest);
        witness.write_vkey_hash(vkey_hash);

        let prover = Groth16Bn254Prover::new();
        let proof = prover.prove(witness, build_dir.to_path_buf());

        // Verify the proof.
        prover
            .verify(
                &proof,
                &vkey_hash.as_canonical_biguint(),
                &committed_values_digest.as_canonical_biguint(),
                build_dir,
            )
            .unwrap();

        proof
    }

    pub fn recursion_program(
        &self,
        input: &SCDTRecursionWitnessValues<CoreSC>,
    ) -> Arc<RecursionProgram<SCField>> {
        // Check if the program is in the cache.
        let mut cache = self.lift_programs_lru.lock().unwrap_or_else(|e| e.into_inner());
        let shape = input.shape();

        // #[allow(unused_assignments)]
        let program = cache.get(&shape).cloned();
        // #[allow(unused_assignments)]
        // let mut program = program;
        // program = None;
        drop(cache);
        match program {
            Some(program) => program,
            None => {
                let misses = self.lift_cache_misses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("core cache miss, misses: {}", misses);
                // Get the operations.
                let builder_span = tracing::debug_span!("build recursion program").entered();
                let mut builder = Builder::<InnerConfig>::default();

                let input =
                    tracing::debug_span!("read input").in_scope(|| input.read(&mut builder));
                tracing::debug_span!("verify").in_scope(|| {
                    SCDTRecursiveVerifier::verify_polyair::<_, POLYAIR_EXT_DEGREE>(
                        &mut builder,
                        self.core_prover.machine(),
                        input,
                    )
                });
                let block =
                    tracing::debug_span!("build block").in_scope(|| builder.into_root_block());
                builder_span.exit();
                // SAFETY: The circuit is well-formed. It does not use synchronization primitives
                // (or possibly other means) to violate the invariants.
                let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

                // Compile the program.
                let compiler_span = tracing::debug_span!("compile recursion program").entered();
                let mut compiler = AsmCompiler::<InnerConfig>::default();
                let mut program = compiler.compile(dsl_program);
                if let Some(inn_recursion_shape_config) = &self.compress_shape_config {
                    inn_recursion_shape_config.fix_shape(&mut program);
                }
                let program = Arc::new(program);
                compiler_span.exit();

                // Insert the program into the cache.
                let mut cache = self.lift_programs_lru.lock().unwrap_or_else(|e| e.into_inner());
                cache.put(shape, program.clone());
                drop(cache);
                program
            }
        }
    }

    pub fn compress_program(
        &self,
        input: &SCDTCompressWithVKeyWitnessValues<InnerSC>,
    ) -> Arc<RecursionProgram<SCField>> {
        self.join_programs_map.get(&input.shape()).cloned().unwrap_or_else(|| {
            tracing::warn!("join program not found in map, recomputing join program.");
            // Get the operations.
            Arc::new(compress_program_from_input::<C>(
                self.compress_shape_config.as_ref(),
                &self.compress_prover,
                self.vk_verification,
                input,
            ))
        })
    }

    fn root_shrink_shape(&self) -> RecursionShape {
        let mut shape = ShrinkAir::<SCField>::shrink_shape().clone_into_hash_map();
        // Start root_shrink from the historical larger root profile. The
        // dynamic shape pass in `shrink_program` tightens it to the actual
        // root program rows before proving.
        for (chip, log_h) in [
            ("MemoryVar", 19usize),
            ("MemoryConst", 19usize),
            ("ExtAlu", 19usize),
            ("Select", 19usize),
            ("BaseAlu", 17usize),
            ("Poseidon2SkinnyKbDeg3", 19usize),
        ] {
            if shape.contains_key(chip) {
                shape.insert(chip.to_string(), log_h);
            }
        }
        shape.into()
    }

    pub fn shrink_program(
        &self,
        shrink_shape: RecursionShape,
        input: &SCDTCompressWithVKeyWitnessValues<InnerSC>,
        verify_machine: ShrinkVerifyMachine,
        enforce_complete: bool,
        enable_dynamic_shape: bool,
    ) -> Arc<RecursionProgram<SCField>> {
        // Get the operations.
        let builder_span = tracing::debug_span!("build shrink program").entered();
        let mut builder = Builder::<ShrinkConfig>::default();
        let input = input.read(&mut builder);
        // Verify the proof.
        // `verify_machine` describes the stage that produced the input proof.
        // The current proof is produced by whichever prover called this method:
        // shrink for penultimate, root_shrink for root, or shrink when this
        // helper is used by the existing `shrink()` API.
        let verify_kind = verify_machine;
        let verify_machine = match verify_kind {
            ShrinkVerifyMachine::Compress => self.compress_prover.machine(),
            ShrinkVerifyMachine::Shrink => self.shrink_prover.machine(),
            // SHA256 root_shrink proofs are final native-verifier artifacts and
            // are not fed back into recursive verification. Keep the legacy
            // shrink API on the ordinary shrink verifier.
            ShrinkVerifyMachine::RootShrink => self.shrink_prover.machine(),
        };
        if enforce_complete {
            SCDTCompressRootVerifierWithVKey::verify(
                &mut builder,
                verify_machine,
                input,
                self.vk_verification,
                PublicValuesOutputDigest::Reduce,
            );
        } else {
            SCDTCompressWithVKeyVerifier::verify(
                &mut builder,
                verify_machine,
                input,
                self.vk_verification,
                PublicValuesOutputDigest::Reduce,
            );
        }
        let block = builder.into_root_block();
        builder_span.exit();
        // SAFETY: The circuit is well-formed. It does not use synchronization primitives
        // (or possibly other means) to violate the invariants.
        let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

        // Compile the program.
        //
        // The shrink stage registers the **skinny** Poseidon2 chip in
        // `sc_shrink_machine`, so the compiler is configured to lower
        // `CircuitV2Poseidon2PermuteKoalaBear` to `Instruction::Poseidon2Skinny`
        // here. Compress / wrap programs keep the default wide lowering.
        let compiler_span = tracing::debug_span!("compile shrink program").entered();
        let mut compiler = AsmCompiler::<ShrinkConfig>::default().with_poseidon2_skinny();
        let mut program = compiler.compile(dsl_program);

        // [shrink-diag] Verify the compiler-mode switch actually emitted skinny Poseidon2
        // instructions. Expectation after `with_poseidon2_skinny()`:
        //   poseidon2_wide_events   = 0
        //   poseidon2_skinny_events > 0
        let inst_counts =
            program.inner.iter().fold(RecursionAirEventCount::default(), |acc, instr| acc + instr);
        tracing::info!("[shrink-diag] instruction counts after compile: {:?}", inst_counts);

        // Verify that actual shrink heights do not exceed the shrink_shape capacity.
        let actual_heights = ShrinkAir::<SCField>::shrink_heights(&program);
        let mut shape_map = shrink_shape.clone_into_hash_map();
        const DYNAMIC_SHAPE_MARGIN: usize = 0;
        if enable_dynamic_shape {
            for (chip_name, actual_height) in &actual_heights {
                let required_log_height = if *actual_height <= 1 {
                    0
                } else {
                    ((*actual_height - 1).ilog2() as usize) + 1
                };
                shape_map.insert(
                    chip_name.clone(),
                    required_log_height.saturating_add(DYNAMIC_SHAPE_MARGIN),
                );
            }

            let mut dynamic_shape_stats = actual_heights
                .iter()
                .filter_map(|(chip_name, actual_height)| {
                    shape_map.get(chip_name).map(|log_h| {
                        let capacity = 1usize << *log_h;
                        let usage_pct = (*actual_height as f64 * 100.0) / (capacity as f64);
                        (chip_name.clone(), *actual_height, *log_h, usage_pct)
                    })
                })
                .collect::<Vec<_>>();
            dynamic_shape_stats.sort_by(|a, b| b.3.total_cmp(&a.3));
            tracing::info!(
                "Dynamic shrink shape applied (margin={}): {:?}",
                DYNAMIC_SHAPE_MARGIN,
                dynamic_shape_stats,
            );
        }
        for (chip_name, log_height) in &shape_map {
            let shape_capacity = 1usize << log_height;
            if let Some((_, actual_height)) =
                actual_heights.iter().find(|(name, _)| name == chip_name)
            {
                if *actual_height > shape_capacity {
                    panic!(
                        "shrink height overflow: chip '{}' actual height {} exceeds \
                         shrink_shape capacity {} (log2={})",
                        chip_name, actual_height, shape_capacity, log_height
                    );
                }
            }
        }

        *program.shape_mut() = Some(shape_map.into());
        let program = Arc::new(program);
        compiler_span.exit();
        program
    }

    pub fn wrap_program(&self) -> Arc<RecursionProgram<SCField>> {
        self.wrap_program
            .get_or_init(|| {
                // Get the operations.

                let builder_span = tracing::debug_span!("build wrap program").entered();
                let mut builder = Builder::<WrapConfig>::default();

                let shrink_shape: OrderedShape = ShrinkAir::<SCField>::shrink_shape().into();
                let input_shape = DTCompressShape::from(vec![shrink_shape]);
                let shape = DTCompressWithVkeyShape {
                    compress_shape: input_shape,
                    merkle_tree_height: self.recursion_vk_tree.height,
                };

                //TODO: use dummy inputs
                let dummy_input = SCDTCompressWithVKeyWitnessValues::<InnerSC>::dummy_polyair(
                    self.shrink_prover.machine(),
                    &shape,
                );

                let input = dummy_input.read(&mut builder);

                // Attest that the merkle tree root is correct.
                let root = input.merkle_var.root;
                for (val, expected) in root.iter().zip(self.recursion_vk_root.iter()) {
                    builder.assert_felt_eq(*val, *expected);
                }
                // Verify the proof.
                SCDTCompressRootVerifierWithVKey::verify(
                    &mut builder,
                    self.shrink_prover.machine(),
                    input,
                    self.vk_verification,
                    PublicValuesOutputDigest::Root,
                );

                let block = builder.into_root_block();
                builder_span.exit();
                // SAFETY: The circuit is well-formed. It does not use synchronization primitives
                // (or possibly other means) to violate the invariants.
                let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

                // Compile the program.
                let compiler_span = tracing::debug_span!("compile compress program").entered();
                let mut compiler = AsmCompiler::<WrapConfig>::default();
                let program = Arc::new(compiler.compile(dsl_program));

                // *program.shape_mut() = Some(shrink_shape);
                compiler_span.exit();
                program
            })
            .clone()
    }

    pub fn deferred_program(
        &self,
        input: &SCDTDeferredWitnessValues<InnerSC>,
    ) -> Arc<RecursionProgram<SCField>> {
        // Compile the program.
        let input_proof_machine = classify_reduce_batch_machine(&input.vks_and_proofs);

        // Get the operations.
        let operations_span =
            tracing::debug_span!("get operations for the deferred program").entered();
        let mut builder = Builder::<InnerConfig>::default();
        let input_read_span = tracing::debug_span!("Read input values").entered();
        let input = input.read(&mut builder);
        input_read_span.exit();
        let verify_span = tracing::debug_span!("Verify deferred program").entered();

        // Verify the proof.
        let verify_machine = match input_proof_machine {
            ReduceProofMachine::Compress => self.compress_prover.machine(),
            // Final root_shrink proofs are no longer recursively verified, so
            // deferred recursive proofs must remain on the ordinary shrink verifier.
            ReduceProofMachine::Shrink => self.shrink_prover.machine(),
        };
        SCDTDeferredVerifier::verify(&mut builder, verify_machine, input, self.vk_verification);
        verify_span.exit();
        let block = builder.into_root_block();
        operations_span.exit();
        // SAFETY: The circuit is well-formed. It does not use synchronization primitives
        // (or possibly other means) to violate the invariants.
        let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

        let compiler_span = tracing::debug_span!("compile deferred program").entered();
        let mut compiler = AsmCompiler::<InnerConfig>::default();
        let mut program = compiler.compile(dsl_program);
        if let Some(recursion_shape_config) = &self.compress_shape_config {
            recursion_shape_config.fix_shape(&mut program);
        }
        let program = Arc::new(program);
        compiler_span.exit();
        program
    }

    pub fn get_recursion_core_inputs(
        &self,
        vk: &SCStarkVerifyingKey<CoreSC>,
        shard_proofs: &[SCShardProof<CoreSC>],
        batch_size: usize,
        is_complete: bool,
        deferred_digest: [Val<CoreSC>; 8],
    ) -> Vec<SCDTRecursionWitnessValues<CoreSC>> {
        let mut core_inputs = Vec::new();

        // Prepare the inputs for the recursion programs.
        for (batch_idx, batch) in shard_proofs.chunks(batch_size).enumerate() {
            let proofs = batch.to_vec();

            core_inputs.push(SCDTRecursionWitnessValues {
                vk: vk.clone(),
                shard_proofs: proofs.clone(),
                is_complete,
                is_first_shard: batch_idx == 0,
                vk_root: self.recursion_vk_root,
                reconstruct_deferred_digest: deferred_digest,
            });
        }
        core_inputs
    }

    pub fn get_recursion_deferred_inputs_with_initial_digest<'a>(
        &'a self,
        vk: &'a SCStarkVerifyingKey<CoreSC>,
        deferred_proofs: &[DTReduceProof<InnerSC>],
        mut deferred_digest: [Val<CoreSC>; 8],
        batch_size: usize,
    ) -> (Vec<SCDTDeferredWitnessValues<InnerSC>>, [SCField; 8]) {
        // Prepare the inputs for the deferred proofs recursive verification.
        let mut deferred_inputs = Vec::new();

        for batch in deferred_proofs.chunks(batch_size) {
            let vks_and_proofs =
                batch.iter().cloned().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();

            let input = SCDTCompressWitnessValues { vks_and_proofs, is_complete: true };
            let input = self.make_merkle_proofs(input);
            let SCDTCompressWithVKeyWitnessValues { compress_val, merkle_val } = input;

            deferred_inputs.push(SCDTDeferredWitnessValues {
                vks_and_proofs: compress_val.vks_and_proofs,
                vk_merkle_data: merkle_val,
                start_reconstruct_deferred_digest: deferred_digest,
                is_complete: false,
                dt_vk_digest: vk.hash_babybear(),
                end_pc: vk.pc_start,
                end_shard: SCField::one(),
                end_execution_shard: SCField::one(),
                init_addr: SCField::zero(),
                finalize_addr: SCField::zero(),
                committed_value_digest: [Word::<SCField>([SCField::zero(); 4]); 8],
                deferred_proofs_digest: [SCField::zero(); 8],
            });

            deferred_digest = Self::hash_deferred_proofs(deferred_digest, batch);
        }
        (deferred_inputs, deferred_digest)
    }

    pub fn get_recursion_deferred_inputs<'a>(
        &'a self,
        vk: &'a SCStarkVerifyingKey<CoreSC>,
        deferred_proofs: &[DTReduceProof<InnerSC>],
        batch_size: usize,
    ) -> (Vec<SCDTDeferredWitnessValues<InnerSC>>, [SCField; 8]) {
        self.get_recursion_deferred_inputs_with_initial_digest(
            vk,
            deferred_proofs,
            [Val::<CoreSC>::zero(); DIGEST_SIZE],
            batch_size,
        )
    }

    /// Generate the inputs for the first layer of recursive proofs.
    #[allow(clippy::type_complexity)]
    pub fn get_first_layer_inputs<'a>(
        &'a self,
        vk: &'a DTVerifyingKey,
        shard_proofs: &[SCShardProof<InnerSC>],
        deferred_proofs: &[DTReduceProof<InnerSC>],
        batch_size: usize,
    ) -> Vec<SCDTCircuitWitness> {
        let (deferred_inputs, deferred_digest) =
            self.get_recursion_deferred_inputs(&vk.vk, deferred_proofs, batch_size);

        let is_complete = shard_proofs.len() <= batch_size && deferred_proofs.is_empty();
        let core_inputs = self.get_recursion_core_inputs(
            &vk.vk,
            shard_proofs,
            batch_size,
            is_complete,
            deferred_digest,
        );

        let mut inputs = Vec::new();
        inputs.extend(deferred_inputs.into_iter().map(SCDTCircuitWitness::Deferred));
        inputs.extend(core_inputs.into_iter().map(SCDTCircuitWitness::Core));
        inputs
    }

    // Accumulate deferred proofs into a single digest.
    pub fn hash_deferred_proofs(
        prev_digest: [Val<CoreSC>; DIGEST_SIZE],
        deferred_proofs: &[DTReduceProof<InnerSC>],
    ) -> [Val<CoreSC>; 8] {
        let mut digest = prev_digest;
        for proof in deferred_proofs.iter() {
            let pv: &RecursionPublicValues<Val<CoreSC>> =
                proof.proof.public_values.as_slice().borrow();
            let committed_values_digest = words_to_bytes(&pv.committed_value_digest);
            digest = sc_hash_deferred_proof(
                &digest,
                &pv.dt_vk_digest,
                &committed_values_digest.try_into().unwrap(),
            );
        }
        digest
    }

    pub fn make_merkle_proofs(
        &self,
        input: SCDTCompressWitnessValues<CoreSC>,
    ) -> SCDTCompressWithVKeyWitnessValues<CoreSC> {
        let num_vks = self.recursion_vk_map.len();
        let (vk_indices, vk_digest_values): (Vec<_>, Vec<_>) = if self.vk_verification {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_babybear();
                    let index = self
                        .recursion_vk_map
                        .get(&vk_digest)
                        .unwrap_or_else(|| panic!("vk not in allowed set: digest={:?}", vk_digest));
                    (index, vk_digest)
                })
                .unzip()
        } else {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_babybear();
                    let index = (vk_digest[0].as_canonical_u32() as usize) % num_vks;
                    (index, [SCField::from_canonical_usize(index); 8])
                })
                .unzip()
        };

        let proofs = vk_indices
            .iter()
            .map(|index| {
                let (_, proof) = MerkleTree::open(&self.recursion_vk_tree, *index);
                proof
            })
            .collect();

        let merkle_val = SCDTMerkleProofWitnessValues {
            root: self.recursion_vk_root,
            values: vk_digest_values,
            vk_merkle_proofs: proofs,
        };

        SCDTCompressWithVKeyWitnessValues { compress_val: input, merkle_val }
    }
}

pub fn compress_program_from_input<C: DTProverComponents>(
    config: Option<&RecursionShapeConfig<SCField, CompressAir<SCField>>>,
    compress_prover: &C::CompressProver,
    vk_verification: bool,
    input: &SCDTCompressWithVKeyWitnessValues<InnerSC>,
) -> RecursionProgram<SCField> {
    let builder_span = tracing::debug_span!("build compress program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    // read the input.
    let input = input.read(&mut builder);
    // Verify the proof.
    SCDTCompressWithVKeyVerifier::verify(
        &mut builder,
        compress_prover.machine(),
        input,
        vk_verification,
        PublicValuesOutputDigest::Reduce,
    );
    let block = builder.into_root_block();
    builder_span.exit();
    // SAFETY: The circuit is well-formed. It does not use synchronization primitives
    // (or possibly other means) to violate the invariants.
    let dsl_program = unsafe { DslIrProgram::new_unchecked(block) };

    // Compile the program.
    let compiler_span = tracing::debug_span!("compile compress program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let mut program = compiler.compile(dsl_program);
    if let Some(config) = config {
        config.fix_shape(&mut program);
    }
    compiler_span.exit();

    program
}
