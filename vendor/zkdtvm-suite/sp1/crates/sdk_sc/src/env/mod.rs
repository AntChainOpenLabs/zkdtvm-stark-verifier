//! # zkDTVM Environment Prover
//!
//! A prover that can execute programs and generate proofs with a different implementation based on
//! the value of certain environment variables.

mod prove;

use std::env;

use anyhow::Result;
use dt_core_executor::DTContextBuilder;
use dt_core_machine::io::DTStdin;
use dt_prover::{components::SCCpuProverComponents, DTProver, DTProvingKey, DTVerifyingKey};
use prove::EnvProveBuilder;

use super::{DTVerificationError, Prover};
#[cfg(feature = "network")]
use crate::network::builder::NetworkProverBuilder;
use crate::{
    cpu::{execute::CpuExecuteBuilder, CpuProver},
    // cuda::CudaProver,
    utils::check_release_build,
    DTProofMode,
    DTProofWithPublicValues,
};

/// A prover that can execute programs and generate proofs with a different implementation based on
/// the value of certain environment variables.
///
/// The environment variables are described in [`EnvProver::new`].
pub struct EnvProver {
    pub(crate) prover: Box<dyn Prover<SCCpuProverComponents>>,
}

impl EnvProver {
    /// Creates a new [`EnvProver`] with the given configuration.
    ///
    /// The following environment variables are used to configure the prover:
    /// - `DT_PROVER`: The type of prover to use. Must be one of `mock`, `local`, `cuda`, or
    ///   `network`.
    /// - `NETWORK_PRIVATE_KEY`: The private key to use for the network prover.
    /// - `NETWORK_RPC_URL`: The RPC URL to use for the network prover.
    #[must_use]
    pub fn new() -> Self {
        let mode = if let Ok(mode) = env::var("DT_PROVER") {
            mode
        } else {
            tracing::warn!("DT_PROVER environment variable not set, defaulting to 'cpu'");
            "cpu".to_string()
        };

        let prover: Box<dyn Prover<SCCpuProverComponents>> = match mode.as_str() {
            "mock" => Box::new(CpuProver::mock()),
            "cpu" => {
                check_release_build();
                Box::new(CpuProver::new())
            },
            "cuda" => {
                todo!("CUDA prover is not yet implemented");
                // check_release_build();
                // Box::new(CudaProver::new(SCDTProver::new(), MoongateServer::default()))
            }
            "network" => {
                #[cfg(not(feature = "network"))]
                panic!(
                    r#"The network prover requires the 'network' feature to be enabled.
                    Please enable it in your Cargo.toml with:
                    dt-sdk = {{ version = "...", features = ["network"] }}"#
                );

                #[cfg(feature = "network")]
                {
                    Box::new(NetworkProverBuilder::default().build())
                }
            }
            _ => panic!(
                "Invalid DT_PROVER value. Expected one of: mock, cpu, cuda, or network. Got: '{mode}'.\n\
                Please set the DT_PROVER environment variable to one of the supported values."
            ),
        };
        EnvProver { prover }
    }

    /// Creates a new [`CpuExecuteBuilder`] for simulating the execution of a program on the CPU.
    ///
    /// # Details
    /// The builder is used for both the [`crate::cpu::CpuProver`] and [`crate::CudaProver`] client
    /// types.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::from_env();
    /// let (public_values, execution_report) = client.execute(elf, &stdin).run().unwrap();
    /// ```
    #[must_use]
    pub fn execute<'a>(&'a self, elf: &'a [u8], stdin: &DTStdin) -> CpuExecuteBuilder<'a> {
        CpuExecuteBuilder {
            prover: self.prover.inner(),
            elf,
            stdin: stdin.clone(),
            context_builder: DTContextBuilder::default(),
        }
    }

    /// Creates a new [`EnvProve`] for proving a program on the CPU.
    ///
    /// # Details
    /// The builder is used for only the [`crate::cpu::CpuProver`] client type.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::from_env();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).core().run();
    /// ```
    #[must_use]
    pub fn prove<'a>(&'a self, pk: &'a DTProvingKey, stdin: &'a DTStdin) -> EnvProveBuilder<'a> {
        EnvProveBuilder {
            prover: self.prover.as_ref(),
            mode: DTProofMode::Core,
            pk,
            stdin: stdin.clone(),
        }
    }

    /// Verifies that the given proof is valid and matches the given verification key produced by
    /// [`Self::setup`].
    ///
    /// ### Examples
    /// ```no_run
    /// use dt_sdk::{DTStdin, ProverClient};
    ///
    /// let elf = test_artifacts::FIBONACCI_ELF;
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::from_env();
    /// let (pk, vk) = client.setup(elf);
    /// let proof = client.prove(&pk, &stdin).run().unwrap();
    /// client.verify(&proof, &vk).unwrap();
    /// ```
    pub fn verify(
        &self,
        proof: &DTProofWithPublicValues,
        vk: &DTVerifyingKey,
    ) -> Result<(), DTVerificationError> {
        self.prover.verify(proof, vk)
    }

    /// Setup a program to be proven and verified by the zkDTVM RISC-V zkVM by computing the proving
    /// and verifying keys.
    #[must_use]
    pub fn setup(&self, elf: &[u8]) -> (DTProvingKey, DTVerifyingKey) {
        self.prover.setup(elf)
    }
}

impl Default for EnvProver {
    fn default() -> Self {
        Self::new()
    }
}

impl Prover<SCCpuProverComponents> for EnvProver {
    fn inner(&self) -> &DTProver<SCCpuProverComponents> {
        self.prover.inner()
    }

    fn setup(&self, elf: &[u8]) -> (DTProvingKey, DTVerifyingKey) {
        self.prover.setup(elf)
    }

    fn prove(
        &self,
        pk: &DTProvingKey,
        stdin: &DTStdin,
        mode: DTProofMode,
    ) -> Result<DTProofWithPublicValues> {
        self.prover.prove(pk, stdin, mode)
    }

    fn verify(
        &self,
        bundle: &DTProofWithPublicValues,
        vkey: &DTVerifyingKey,
    ) -> Result<(), DTVerificationError> {
        self.prover.verify(bundle, vkey)
    }
}
