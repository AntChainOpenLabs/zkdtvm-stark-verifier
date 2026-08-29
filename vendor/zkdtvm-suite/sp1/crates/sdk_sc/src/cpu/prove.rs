//! # CPU Proving
//!
//! This module provides a builder for proving a program on the CPU.

use anyhow::Result;
use dt_core_executor::{DTContextBuilder, IoWriter};
use dt_core_machine::io::DTStdin;
use dt_prover::DTProvingKey;
use dt_stark::{DTCoreOpts, DTProverOpts, RecursionBackend};

use super::CpuProver;
use crate::{DTProofMode, DTProofWithPublicValues};

/// A builder for proving a program on the CPU.
///
/// This builder provides a typed interface for configuring the zkDTVM RISC-V prover. The builder is
/// used for only the [`crate::cpu::CpuProver`] client type.
pub struct CpuProveBuilder<'a> {
    pub(crate) prover: &'a CpuProver,
    pub(crate) mode: DTProofMode,
    pub(crate) context_builder: DTContextBuilder<'a>,
    pub(crate) pk: &'a DTProvingKey,
    pub(crate) stdin: DTStdin,
    pub(crate) core_opts: DTCoreOpts,
    pub(crate) recursion_opts: DTCoreOpts,
    pub(crate) recursion_backend: Option<RecursionBackend>,
    pub(crate) mock: bool,
}

impl<'a> CpuProveBuilder<'a> {
    /// Set the proof kind to [`DTProofKind::Core`] mode.
    ///
    /// # Details
    /// This is the default mode for the prover. The proofs grow linearly in size with the number
    /// of cycles.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).core().run();
    /// ```
    #[must_use]
    pub fn core(mut self) -> Self {
        self.mode = DTProofMode::Core;
        self
    }

    /// Set the proof kind to [`DTProofKind::Compressed`] mode.
    ///
    /// # Details
    /// This mode produces a proof that is of constant size, regardless of the number of cycles. It
    /// takes longer to prove than [`DTProofKind::Core`] due to the need to recursively aggregate
    /// proofs into a single proof.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).compressed().run();
    /// ```
    #[must_use]
    pub fn compressed(mut self) -> Self {
        self.mode = DTProofMode::Compressed;
        self
    }

    /// Set the proof kind to [`DTProofMode::Shrink`] mode.
    ///
    /// # Details
    /// This mode runs the full pipeline up to and including the shrink stage, returning the
    /// shrink proof directly. It is useful for unit-testing the shrink stage in isolation
    /// without having to wrap into a Plonk/Groth16 proof.
    ///
    /// KoalaBear builds with a SHA256 root_shrink PCS stop at [`DTProofMode::Compressed`];
    /// requesting this mode there returns an error because the compressed proof is already
    /// the final root proof.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).shrink().run();
    /// ```
    #[must_use]
    pub fn shrink(mut self) -> Self {
        self.mode = DTProofMode::Shrink;
        self
    }

    /// Set the proof mode to [`DTProofKind::Plonk`] mode.
    ///
    /// # Details
    /// This mode produces a const size PLONK proof that can be verified on chain for roughly ~300k
    /// gas. This mode is useful for producing a maximally small proof that can be verified on
    /// chain. For more efficient SNARK wrapping, you can use the [`DTProofKind::Groth16`] mode but
    /// this mode is more .
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).plonk().run();
    /// ```
    #[must_use]
    pub fn plonk(mut self) -> Self {
        self.mode = DTProofMode::Plonk;
        self
    }

    /// Set the proof mode to [`DTProofKind::Groth16`] mode.
    ///
    /// # Details
    /// This mode produces a Groth16 proof that can be verified on chain for roughly ~100k gas. This
    /// mode is useful for producing a proof that can be verified on chain with minimal gas.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).groth16().run();
    /// ```
    #[must_use]
    pub fn groth16(mut self) -> Self {
        self.mode = DTProofMode::Groth16;
        self
    }

    /// Set the proof mode to the given [`DTProofKind`].
    ///
    /// # Details
    /// This method is useful for setting the proof mode to a custom mode.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTProofMode, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).mode(DTProofMode::Groth16).run();
    /// ```
    #[must_use]
    pub fn mode(mut self, mode: DTProofMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the shard size for proving.
    ///
    /// # Details
    /// The value should be 2^16, 2^17, ..., 2^22. You must be careful to set this value
    /// correctly, as it will affect the memory usage of the prover and the recursion/verification
    /// complexity. By default, the value is set to some predefined values that are optimized for
    /// performance based on the available amount of RAM on the system.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).shard_size(1 << 16).run();
    /// ```
    #[must_use]
    pub fn shard_size(mut self, value: usize) -> Self {
        assert!(value.is_power_of_two(), "shard size must be a power of 2");
        self.core_opts.shard_size = value;
        self
    }

    /// Set the shard batch size for proving.
    ///
    /// # Details
    /// This is the number of shards that are processed in a single batch in the prover. You should
    /// probably not change this value unless you know what you are doing.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).shard_batch_size(4).run();
    /// ```
    #[must_use]
    pub fn shard_batch_size(mut self, value: usize) -> Self {
        self.core_opts.shard_batch_size = value;
        self
    }

    /// Select the recursion backend for the compress stage.
    ///
    /// # Details
    /// Overrides the `DT_RECURSION_BACKEND` env selector for this run. When neither is set,
    /// the native ladder is used.
    #[must_use]
    pub fn recursion_backend(mut self, backend: RecursionBackend) -> Self {
        self.recursion_backend = Some(backend);
        self
    }

    /// Set the maximum number of cpu cycles to use for execution.
    ///
    /// # Details
    /// If the cycle limit is exceeded, execution will return
    /// [`dt_core_executor::ExecutionError::ExceededCycleLimit`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).cycle_limit(1000000).run();
    /// ```
    #[must_use]
    pub fn cycle_limit(mut self, cycle_limit: u64) -> Self {
        self.context_builder.max_cycles(cycle_limit);
        self
    }

    /// Whether to enable deferred proof verification in the executor.
    ///
    /// # Arguments
    /// * `value` - Whether to enable deferred proof verification in the executor.
    ///
    /// # Details
    /// Default: `true`. If set to `false`, the executor will skip deferred proof verification.
    /// This is useful for reducing the execution time of the program and optimistically assuming
    /// that the deferred proofs are correct. Can also be used for mock proof setups that require
    /// verifying mock compressed proofs.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).deferred_proof_verification(false).run();
    /// ```
    #[must_use]
    pub fn deferred_proof_verification(mut self, value: bool) -> Self {
        self.context_builder.set_deferred_proof_verification(value);
        self
    }

    /// Override the default stdout of the guest program.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let mut stdout = Vec::new();
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// client.execute(elf, &stdin).stdout(&mut stdout).run();
    /// ```
    #[must_use]
    pub fn stdout<W: IoWriter>(mut self, writer: &'a mut W) -> Self {
        self.context_builder.stdout(writer);
        self
    }

    /// Override the default stdout of the guest program.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let mut stderr = Vec::new();
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// client.execute(elf, &stdin).stderr(&mut stderr).run();
    /// ```````
    #[must_use]
    pub fn stderr<W: IoWriter>(mut self, writer: &'a mut W) -> Self {
        self.context_builder.stderr(writer);
        self
    }

    /// Run the prover with the built arguments.
    ///
    /// # Details
    /// This method will run the prover with the built arguments. If the prover fails to run, the
    /// method will return an error.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (pk, vk) = client.setup(elf);
    /// let proof = client.prove(&pk, &stdin).run().unwrap();
    /// ```
    pub fn run(self) -> Result<DTProofWithPublicValues> {
        // Get the arguments.
        let Self {
            prover,
            mode,
            pk,
            stdin,
            mut context_builder,
            core_opts,
            recursion_opts,
            recursion_backend,
            mock,
        } = self;
        let opts = DTProverOpts { core_opts, recursion_opts, recursion_backend };
        let context = context_builder.build();

        // Dump the program and stdin to files for debugging if `DT_DUMP` is set.
        crate::utils::dt_dump(&pk.elf, &stdin);

        // Run the prover.
        if mock {
            prover.mock_prove_impl(pk, &stdin, context, mode)
        } else {
            prover.prove_impl(pk, &stdin, opts, context, mode)
        }
    }
}
