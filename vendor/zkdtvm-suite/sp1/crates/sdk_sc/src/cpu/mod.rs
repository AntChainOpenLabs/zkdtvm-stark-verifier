//! # zkDTVM CPU Prover
//!
//! A prover that uses the CPU to execute and prove programs.

pub mod builder;
pub mod execute;
pub mod prove;

#[cfg(not(feature = "ext5"))]
use crate::install::try_install_circuit_artifacts;
use crate::{
    prover::verify_proof, DTProof, DTProofMode, DTProofWithPublicValues, DTProvingKey,
    DTVerificationError, DTVerifyingKey, Prover,
};
use anyhow::Result;
use dt_core_executor::{DTContext, DTContextBuilder};
use dt_core_machine::io::DTStdin;
use dt_prover::{
    components::SCCpuProverComponents,
    verify::{verify_groth16_bn254_public_inputs, verify_plonk_bn254_public_inputs},
    DTCoreProofData, DTProofWithMetadata, DTProver, Groth16Bn254Proof, PlonkBn254Proof,
};
use dt_stark::{DTCoreOpts, DTProverOpts};
use execute::CpuExecuteBuilder;
use prove::CpuProveBuilder;

/// A prover that uses the CPU to execute and prove programs.
pub struct CpuProver {
    pub(crate) prover: DTProver<SCCpuProverComponents>,
    pub(crate) mock: bool,
}

impl CpuProver {
    /// Creates a new [`CpuProver`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The underlying [`DTProver`] (measurement harnesses read backend reports and
    /// call stage entry points directly through this).
    #[must_use]
    pub fn inner(&self) -> &DTProver<SCCpuProverComponents> {
        &self.prover
    }

    /// Creates a new [`CpuProver`] in mock mode.
    #[must_use]
    pub fn mock() -> Self {
        Self { prover: DTProver::new(), mock: true }
    }

    /// Creates a new [`CpuExecuteBuilder`] for simulating the execution of a program on the CPU.
    ///
    /// # Details
    /// The builder is used for both the [`crate::cpu::CpuProver`] and [`crate::CudaProver`] client
    /// types.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{include_elf, DTStdin, Prover, ProverClient};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = DTStdin::new();
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let (public_values, execution_report) = client.execute(elf, &stdin).run().unwrap();
    /// ```
    pub fn execute<'a>(&'a self, elf: &'a [u8], stdin: &DTStdin) -> CpuExecuteBuilder<'a> {
        CpuExecuteBuilder {
            prover: &self.prover,
            elf,
            stdin: stdin.clone(),
            context_builder: DTContextBuilder::default(),
        }
    }

    /// Creates a new [`CpuProveBuilder`] for proving a program on the CPU.
    ///
    /// # Details
    /// The builder is used for only the [`crate::cpu::CpuProver`] client type.
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
    pub fn prove<'a>(&'a self, pk: &'a DTProvingKey, stdin: &DTStdin) -> CpuProveBuilder<'a> {
        CpuProveBuilder {
            prover: self,
            mode: DTProofMode::Core,
            pk,
            stdin: stdin.clone(),
            context_builder: DTContextBuilder::default(),
            core_opts: DTCoreOpts::default(),
            recursion_opts: DTCoreOpts::recursion(),
            recursion_backend: None,
            mock: self.mock,
        }
    }

    // Debug-only short-circuit after the shrink stage leaves the wrap_bn254 / gnark code paths
    // unreachable; silence the resulting unused / unreachable lints at the function scope so the
    // rest of the crate still gets normal lint coverage.
    #[allow(unreachable_code, unused_variables)]
    pub(crate) fn prove_impl<'a>(
        &'a self,
        pk: &DTProvingKey,
        stdin: &DTStdin,
        opts: DTProverOpts,
        context: DTContext<'a>,
        mode: DTProofMode,
    ) -> Result<DTProofWithPublicValues> {
        let program = self.prover.get_program(&pk.elf).unwrap();

        // If we're in mock mode, return a mock proof.
        if self.mock {
            return self.mock_prove_impl(pk, stdin, context, mode);
        }

        // Generate the core proof.
        let core_start = std::time::Instant::now();
        #[cfg(feature = "native-recursion")]
        enum CoreProofHandoff {
            Materialized(DTProofWithMetadata<DTCoreProofData>),
            Native(dt_prover::native_backend::NativeCoreHandoff),
        }
        #[cfg(feature = "native-recursion")]
        let use_native_handoff = mode != DTProofMode::Core &&
            stdin.proofs.is_empty() &&
            dt_stark::RecursionBackend::resolve(opts.recursion_backend)
                .map_err(anyhow::Error::msg)? ==
                dt_stark::RecursionBackend::Native;
        #[cfg(feature = "native-recursion")]
        let core_handoff = if use_native_handoff {
            let proof = self
                .prover
                .prove_core_with_native_handoff(&pk.pk, &pk.vk, program, stdin, opts, context)?;
            CoreProofHandoff::Native(proof)
        } else {
            CoreProofHandoff::Materialized(
                self.prover.prove_core(&pk.pk, program, stdin, opts, context)?,
            )
        };
        #[cfg(not(feature = "native-recursion"))]
        let proof: DTProofWithMetadata<DTCoreProofData> =
            self.prover.prove_core(&pk.pk, program, stdin, opts, context)?;
        let core_elapsed = core_start.elapsed();
        tracing::trace!("CORE_PROVING_TIME time_ms={}", core_elapsed.as_millis());
        if mode == DTProofMode::Core {
            #[cfg(feature = "native-recursion")]
            let proof = match core_handoff {
                CoreProofHandoff::Materialized(proof) => proof,
                CoreProofHandoff::Native(_) => {
                    unreachable!("core mode cannot produce a native handoff")
                }
            };
            return Ok(DTProofWithPublicValues::new(
                DTProof::Core(proof.proof.0),
                proof.public_values,
                self.version().to_string(),
            ));
        }

        // Generate the compressed proof.
        let deferred_proofs: Vec<_> =
            stdin.proofs.iter().map(|(reduce_proof, _)| reduce_proof.clone()).collect();
        #[cfg(feature = "native-recursion")]
        let public_values = match &core_handoff {
            CoreProofHandoff::Materialized(proof) => proof.public_values.clone(),
            CoreProofHandoff::Native(proof) => proof.public_values().clone(),
        };
        #[cfg(not(feature = "native-recursion"))]
        let public_values = proof.public_values.clone();
        let compress_time = std::time::Instant::now();
        #[cfg(feature = "native-recursion")]
        let reduce_proof = match core_handoff {
            CoreProofHandoff::Native(proof) => {
                debug_assert!(deferred_proofs.is_empty());
                self.prover.compress_native_handoff(&pk.vk, proof, opts)?
            }
            CoreProofHandoff::Materialized(proof) => {
                self.prover.compress(&pk.vk, proof, deferred_proofs, opts)?
            }
        };
        #[cfg(not(feature = "native-recursion"))]
        let reduce_proof = self.prover.compress(&pk.vk, proof, deferred_proofs, opts)?;
        let compress_elapsed = compress_time.elapsed();
        tracing::info!("Compress proving time is {}", compress_elapsed.as_secs_f64());
        tracing::trace!("COMPRESS_PROVING_TIME time_ms={}", compress_elapsed.as_millis());

        // Decoupled: when the user requested only a Compressed proof, return the
        // raw reduce proof here without running the shrink stage. This makes
        // .compressed() ~17s faster and lets callers iterate on shrink in
        // isolation by saving the compressed proof and feeding it to
        // [`CpuProver::prove_shrink_only`].
        if mode == DTProofMode::Compressed {
            return Ok(DTProofWithPublicValues::new(
                DTProof::Compressed(Box::new(reduce_proof)),
                public_values,
                self.version().to_string(),
            ));
        }

        #[cfg(feature = "koalabear")]
        {
            anyhow::bail!(
                "this proof mode is not supported after the KoalaBear SHA256 root_shrink stage; \
                 request Compressed, which is the final root proof"
            );
        }

        #[cfg(not(feature = "koalabear"))]
        {
            // Generate the shrink proof.
            let shrink_proof = self.prover.shrink(reduce_proof.clone(), opts)?;

            // Print the serialized byte size of the shrink proof, so we can
            // observe how large the proof emitted by the shrink stage is.
            match bincode::serialize(&shrink_proof) {
                Ok(bytes) => {
                    tracing::info!(
                        "[shrink-proof-size] shrink proof size = {} bytes ({:.2} KiB)",
                        bytes.len(),
                        bytes.len() as f64 / 1024.0,
                    );
                }
                Err(e) => {
                    tracing::warn!("[shrink-proof-size] failed to serialize shrink proof: {}", e);
                }
            }

            // Detailed proof size breakdown (shrink_proof is DTReduceProof with .proof:
            // SCShardProof)
            {
                let shard = &shrink_proof.proof;
                let vk_size = bincode::serialize(&shrink_proof.vk).map(|b| b.len()).unwrap_or(0);
                let commitment_size =
                    bincode::serialize(&shard.commitment).map(|b| b.len()).unwrap_or(0);
                let opened_values_size =
                    bincode::serialize(&shard.opened_values).map(|b| b.len()).unwrap_or(0);
                let opening_proof_size =
                    bincode::serialize(&shard.opening_proof).map(|b| b.len()).unwrap_or(0);
                let sumcheck_proof_size =
                    bincode::serialize(&shard.sumcheck_proof).map(|b| b.len()).unwrap_or(0);
                let dimensions_size =
                    bincode::serialize(&shard.dimensions).map(|b| b.len()).unwrap_or(0);
                let chip_ordering_size =
                    bincode::serialize(&shard.chip_ordering).map(|b| b.len()).unwrap_or(0);
                let public_values_size =
                    bincode::serialize(&shard.public_values).map(|b| b.len()).unwrap_or(0);
                tracing::info!(
                    "[shrink-breakdown] vk={} commitment={} opened_values={} opening_proof={} sumcheck_proof={} dimensions={} chip_ordering={} public_values={}",
                    vk_size, commitment_size, opened_values_size, opening_proof_size, sumcheck_proof_size, dimensions_size, chip_ordering_size, public_values_size,
                );
                for (chip_idx, chip_ov) in shard.opened_values.chips.iter().enumerate() {
                    let chip_name = shard
                        .chip_ordering
                        .iter()
                        .find(|(_, &v)| v == chip_idx)
                        .map(|(k, _)| k.as_str())
                        .unwrap_or("?");
                    let pre_size =
                        bincode::serialize(&chip_ov.preprocessed).map(|b| b.len()).unwrap_or(0);
                    let main_size = bincode::serialize(&chip_ov.main).map(|b| b.len()).unwrap_or(0);
                    let perm_size =
                        bincode::serialize(&chip_ov.permutation).map(|b| b.len()).unwrap_or(0);
                    tracing::info!(
                        "[shrink-chip-ov] chip={} pre={} main={} perm={} log_h={}",
                        chip_name,
                        pre_size,
                        main_size,
                        perm_size,
                        chip_ov.log_height,
                    );
                }
            }

            // For DTProofMode::Shrink, return the shrink proof directly (decoupled from
            // Compressed so callers can tell which stage produced the proof).
            if mode == DTProofMode::Shrink {
                return Ok(DTProofWithPublicValues::new(
                    DTProof::Shrink(Box::new(shrink_proof)),
                    public_values,
                    self.version().to_string(),
                ));
            }

            #[cfg(feature = "ext5")]
            {
                // ext5 currently stops at shrink: BN254 wrap/gnark is quartic-only.
                return Ok(DTProofWithPublicValues::new(
                    DTProof::Shrink(Box::new(shrink_proof)),
                    public_values,
                    self.version().to_string(),
                ));
            }

            #[cfg(not(feature = "ext5"))]
            {
                // Short-circuit: stop the proving pipeline right after the shrink stage
                // so that wrap_bn254 / gnark are skipped. Kept as a safety net in case
                // a non-shrink, non-wrap mode reaches this point during refactoring.
                return Ok(DTProofWithPublicValues::new(
                    DTProof::Shrink(Box::new(shrink_proof)),
                    public_values,
                    self.version().to_string(),
                ));

                // Generate the wrap proof.
                let outer_proof = self.prover.wrap_bn254(shrink_proof, opts)?;

                // Generate the gnark proof.
                match mode {
                    DTProofMode::Groth16 => {
                        let groth16_bn254_artifacts = if dt_prover::build::dt_dev_mode() {
                            dt_prover::build::try_build_groth16_bn254_artifacts_dev(
                                &outer_proof.vk,
                                &outer_proof.proof,
                            )
                        } else {
                            try_install_circuit_artifacts("groth16")
                        };
                        tracing::debug!("end build groth16 artifacts");
                        let proof =
                            self.prover.wrap_groth16_bn254(outer_proof, &groth16_bn254_artifacts);
                        tracing::debug!("end wrap groth16 bn254");
                        Ok(DTProofWithPublicValues::new(
                            DTProof::Groth16(proof),
                            public_values,
                            self.version().to_string(),
                        ))
                    }
                    DTProofMode::Plonk => {
                        let plonk_bn254_artifacts = if dt_prover::build::dt_dev_mode() {
                            dt_prover::build::try_build_plonk_bn254_artifacts_dev(
                                &outer_proof.vk,
                                &outer_proof.proof,
                            )
                        } else {
                            try_install_circuit_artifacts("plonk")
                        };
                        let proof =
                            self.prover.wrap_plonk_bn254(outer_proof, &plonk_bn254_artifacts);
                        Ok(DTProofWithPublicValues::new(
                            DTProof::Plonk(proof),
                            public_values,
                            self.version().to_string(),
                        ))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    pub(crate) fn mock_prove_impl<'a>(
        &'a self,
        pk: &DTProvingKey,
        stdin: &DTStdin,
        context: DTContext<'a>,
        mode: DTProofMode,
    ) -> Result<DTProofWithPublicValues> {
        let (public_values, _, _) = self.prover.execute(&pk.elf, stdin, context)?;
        Ok(DTProofWithPublicValues::create_mock_proof(pk, public_values, mode, self.version()))
    }

    /// Run the shrink stage on top of an already-produced compressed proof.
    ///
    /// # Details
    /// This is available only for configurations where `Compressed` is still an
    /// intermediate recursive proof. KoalaBear SHA256 root_shrink builds return
    /// an error here because `Compressed` is already the final root proof.
    ///
    /// The input bundle must carry a [`DTProof::Compressed`] proof; otherwise
    /// an error is returned.
    ///
    /// # Example
    /// ```rust,no_run
    /// use dt_sdk::{DTProofWithPublicValues, DTStdin, Prover, ProverClient};
    ///
    /// let client = ProverClient::builder().cpu().build();
    /// let bundle = DTProofWithPublicValues::load("compressed.bin").unwrap();
    /// let shrink_bundle = client.prove_shrink_only(&bundle).unwrap();
    /// shrink_bundle.save("shrink.bin").unwrap();
    /// ```
    pub fn prove_shrink_only(
        &self,
        compressed_bundle: &DTProofWithPublicValues,
    ) -> Result<DTProofWithPublicValues> {
        #[cfg(feature = "koalabear")]
        {
            let _ = compressed_bundle;
            anyhow::bail!(
                "prove_shrink_only is disabled for KoalaBear SHA256 root_shrink proofs; \
                 Compressed is already the final root proof"
            );
        }

        #[cfg(not(feature = "koalabear"))]
        {
            let DTProofWithPublicValues { proof, public_values, dt_version, .. } =
                compressed_bundle;

            let reduce_proof = match proof {
                DTProof::Compressed(reduce_proof) => (**reduce_proof).clone(),
                other => {
                    return Err(anyhow::anyhow!(
                        "prove_shrink_only requires a Compressed proof, got {}",
                        other
                    ));
                }
            };

            let opts = DTProverOpts::default();
            let shrink_time = std::time::Instant::now();
            let shrink_proof = self.prover.shrink(reduce_proof, opts)?;
            tracing::info!("Shrink-only proving time is {}", shrink_time.elapsed().as_secs_f64());

            // Same [shrink-proof-size] / [shrink-breakdown] / [shrink-chip-ov]
            // diagnostics that prove_impl emits, so we can compare apples-to-apples
            // against the historical (pre-decoupling) baseline numbers in
            // /tmp/c2_safe_v2_progress.log.
            match bincode::serialize(&shrink_proof) {
                Ok(bytes) => tracing::info!(
                    "[shrink-proof-size] shrink proof size = {} bytes ({:.2} KiB)",
                    bytes.len(),
                    bytes.len() as f64 / 1024.0,
                ),
                Err(e) => {
                    tracing::warn!("[shrink-proof-size] failed to serialize shrink proof: {}", e);
                }
            }
            {
                let shard = &shrink_proof.proof;
                let vk_size = bincode::serialize(&shrink_proof.vk).map(|b| b.len()).unwrap_or(0);
                let commitment_size =
                    bincode::serialize(&shard.commitment).map(|b| b.len()).unwrap_or(0);
                let opened_values_size =
                    bincode::serialize(&shard.opened_values).map(|b| b.len()).unwrap_or(0);
                let opening_proof_size =
                    bincode::serialize(&shard.opening_proof).map(|b| b.len()).unwrap_or(0);
                let sumcheck_proof_size =
                    bincode::serialize(&shard.sumcheck_proof).map(|b| b.len()).unwrap_or(0);
                let dimensions_size =
                    bincode::serialize(&shard.dimensions).map(|b| b.len()).unwrap_or(0);
                let chip_ordering_size =
                    bincode::serialize(&shard.chip_ordering).map(|b| b.len()).unwrap_or(0);
                let public_values_size =
                    bincode::serialize(&shard.public_values).map(|b| b.len()).unwrap_or(0);
                tracing::info!(
                    "[shrink-breakdown] vk={} commitment={} opened_values={} opening_proof={} sumcheck_proof={} dimensions={} chip_ordering={} public_values={}",
                    vk_size,
                    commitment_size,
                    opened_values_size,
                    opening_proof_size,
                    sumcheck_proof_size,
                    dimensions_size,
                    chip_ordering_size,
                    public_values_size,
                );
                for (chip_idx, chip_ov) in shard.opened_values.chips.iter().enumerate() {
                    let chip_name = shard
                        .chip_ordering
                        .iter()
                        .find(|(_, &v)| v == chip_idx)
                        .map(|(k, _)| k.as_str())
                        .unwrap_or("?");
                    let pre_size =
                        bincode::serialize(&chip_ov.preprocessed).map(|b| b.len()).unwrap_or(0);
                    let main_size = bincode::serialize(&chip_ov.main).map(|b| b.len()).unwrap_or(0);
                    let perm_size =
                        bincode::serialize(&chip_ov.permutation).map(|b| b.len()).unwrap_or(0);
                    tracing::info!(
                        "[shrink-chip-ov] chip={} pre={} main={} perm={} log_h={}",
                        chip_name,
                        pre_size,
                        main_size,
                        perm_size,
                        chip_ov.log_height,
                    );
                }
            }

            Ok(DTProofWithPublicValues::new(
                DTProof::Shrink(Box::new(shrink_proof)),
                public_values.clone(),
                dt_version.clone(),
            ))
        }
    }

    fn mock_verify(
        bundle: &DTProofWithPublicValues,
        vkey: &DTVerifyingKey,
    ) -> Result<(), DTVerificationError> {
        match &bundle.proof {
            DTProof::Plonk(PlonkBn254Proof { public_inputs, .. }) => {
                verify_plonk_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                    .map_err(DTVerificationError::Plonk)
            }
            DTProof::Groth16(Groth16Bn254Proof { public_inputs, .. }) => {
                verify_groth16_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                    .map_err(DTVerificationError::Groth16)
            }
            _ => Ok(()),
        }
    }
}

impl Prover<SCCpuProverComponents> for CpuProver {
    fn setup(&self, elf: &[u8]) -> (DTProvingKey, DTVerifyingKey) {
        let (pk, _, _, vk) = self.prover.setup(elf);
        (pk, vk)
    }

    fn inner(&self) -> &DTProver<SCCpuProverComponents> {
        &self.prover
    }

    fn prove(
        &self,
        pk: &DTProvingKey,
        stdin: &DTStdin,
        mode: DTProofMode,
    ) -> Result<DTProofWithPublicValues> {
        self.prove_impl(pk, stdin, DTProverOpts::default(), DTContext::default(), mode)
    }

    fn verify(
        &self,
        bundle: &DTProofWithPublicValues,
        vkey: &DTVerifyingKey,
    ) -> Result<(), DTVerificationError> {
        if self.mock {
            tracing::warn!("using mock verifier");
            return Self::mock_verify(bundle, vkey);
        }
        verify_proof(self.inner(), self.version(), bundle, vkey)
    }
}

impl Default for CpuProver {
    fn default() -> Self {
        let prover = DTProver::new();
        Self { prover, mock: false }
    }
}
