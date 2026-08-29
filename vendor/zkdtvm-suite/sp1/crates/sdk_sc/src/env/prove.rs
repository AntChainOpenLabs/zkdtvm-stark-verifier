use anyhow::Result;
use dt_core_machine::io::DTStdin;
use dt_prover::{components::SCCpuProverComponents, DTProvingKey};

use crate::{DTProofMode, DTProofWithPublicValues, Prover};

/// Builder to prepare and configure proving execution of a program on an input.
/// May be run with [`Self::run`].
pub struct EnvProveBuilder<'a> {
    pub(crate) prover: &'a dyn Prover<SCCpuProverComponents>,
    pub(crate) mode: DTProofMode,
    pub(crate) pk: &'a DTProvingKey,
    pub(crate) stdin: DTStdin,
}

impl EnvProveBuilder<'_> {
    /// Set the proof kind to [`DTProofMode::Core`] mode.
    ///
    /// # Details
    /// This is the default mode for the prover. The proofs grow linearly in size with the number
    /// of cycles.
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
    pub fn core(mut self) -> Self {
        self.mode = DTProofMode::Core;
        self
    }

    /// Set the proof kind to [`DTProofMode::Compressed`] mode.
    ///
    /// # Details
    /// This mode produces a proof that is of constant size, regardless of the number of cycles. It
    /// takes longer to prove than [`DTProofMode::Core`] due to the need to recursively aggregate
    /// proofs into a single proof.
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
    /// let builder = client.prove(&pk, &stdin).compressed().run();
    /// ```
    pub fn compressed(mut self) -> Self {
        self.mode = DTProofMode::Compressed;
        self
    }

    /// Set the proof mode to [`DTProofMode::Plonk`] mode.
    ///
    /// # Details
    /// This mode produces a const size PLONK proof that can be verified on chain for roughly ~300k
    /// gas. This mode is useful for producing a maximally small proof that can be verified on
    /// chain. For more efficient SNARK wrapping, you can use the [`DTProofMode::Groth16`] mode but
    /// this mode is more .
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
    /// let builder = client.prove(&pk, &stdin).plonk().run();
    /// ```
    pub fn plonk(mut self) -> Self {
        self.mode = DTProofMode::Plonk;
        self
    }

    /// Set the proof mode to [`DTProofMode::Groth16`] mode.
    ///
    /// # Details
    /// This mode produces a Groth16 proof that can be verified on chain for roughly ~100k gas. This
    /// mode is useful for producing a proof that can be verified on chain with minimal gas.
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
    /// let builder = client.prove(&pk, &stdin).groth16().run();
    /// ```
    pub fn groth16(mut self) -> Self {
        self.mode = DTProofMode::Groth16;
        self
    }

    /// Set the proof mode to the given [`DTProofMode`].
    ///
    /// # Details
    /// This method is useful for setting the proof mode to a custom mode.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{DTProofMode, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::from_env();
    /// let (pk, vk) = client.setup(elf);
    /// let builder = client.prove(&pk, &stdin).mode(DTProofMode::Groth16).run();
    /// ```
    pub fn mode(mut self, mode: DTProofMode) -> Self {
        self.mode = mode;
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
    /// use dt_sdk::{DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::from_env();
    /// let (pk, vk) = client.setup(elf);
    /// let proof = client.prove(&pk, &stdin).run().unwrap();
    /// ```
    pub fn run(self) -> Result<DTProofWithPublicValues> {
        let Self { prover, mode: kind, pk, stdin } = self;

        // Dump the program and stdin to files for debugging if `DT_DUMP` is set.
        crate::utils::dt_dump(&pk.elf, &stdin);

        prover.prove(pk, &stdin, kind)
    }
}
