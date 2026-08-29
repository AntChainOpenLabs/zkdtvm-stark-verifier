//! # zkDTVM Prover Trait
//!
//! A trait that each prover variant must implement.

use std::borrow::Borrow;

use anyhow::Result;
use dt_core_executor::{DTContext, ExecutionReport};
use dt_core_machine::io::DTStdin;
use dt_primitives::io::DTPublicValues;
use dt_prover::{
    components::DTProverComponents, CoreSC, DTCoreProofData, DTProver, DTProvingKey,
    DTVerifyingKey, InnerSC, DT_CIRCUIT_VERSION,
};
use dt_stark::{air::PublicValues, MachineVerificationError, Word};
use itertools::Itertools;
use p3_field::PrimeField32;
use thiserror::Error;

use crate::{
    install::try_install_circuit_artifacts, DTProof, DTProofMode, DTProofWithPublicValues,
};

/// A basic set of primitives that each prover variant must implement.
pub trait Prover<C: DTProverComponents>: Send + Sync {
    /// The inner [`DTProver`] struct used by the prover.
    fn inner(&self) -> &DTProver<C>;

    /// The version of the current zkDTVM circuit.
    fn version(&self) -> &str {
        DT_CIRCUIT_VERSION
    }

    /// Generate the proving and verifying keys for the given program.
    fn setup(&self, elf: &[u8]) -> (DTProvingKey, DTVerifyingKey);

    /// Executes the program on the given input.
    fn execute(&self, elf: &[u8], stdin: &DTStdin) -> Result<(DTPublicValues, ExecutionReport)> {
        let (pv, _, report) = self.inner().execute(elf, stdin, DTContext::default())?;
        Ok((pv, report))
    }

    /// Proves the given program on the given input in the given proof mode.
    fn prove(
        &self,
        pk: &DTProvingKey,
        stdin: &DTStdin,
        mode: DTProofMode,
    ) -> Result<DTProofWithPublicValues>;

    /// Verify that a zkDTVM proof is valid given its vkey and metadata.
    /// For Plonk proofs, verifies that the public inputs of the `PlonkBn254` proof match
    /// the hash of the VK and the committed public values of the `DTProofWithPublicValues`.
    fn verify(
        &self,
        bundle: &DTProofWithPublicValues,
        vkey: &DTVerifyingKey,
    ) -> Result<(), DTVerificationError> {
        verify_proof(self.inner(), self.version(), bundle, vkey)
    }
}

/// An error that occurs when calling [`Prover::verify`].
#[derive(Error, Debug)]
pub enum DTVerificationError {
    /// An error that occurs when the public values are invalid.
    #[error("Invalid public values")]
    InvalidPublicValues,
    /// An error that occurs when the zkDTVM version does not match the version of the circuit.
    #[error("Version mismatch")]
    VersionMismatch(String),
    /// An error that occurs when the core machine verification fails.
    #[error("Core machine verification error: {0}")]
    Core(MachineVerificationError<CoreSC>),
    /// An error that occurs when the recursion verification fails.
    #[error("Recursion verification error: {0}")]
    Recursion(MachineVerificationError<InnerSC>),
    /// An error that occurs when the Plonk verification fails.
    #[error("Plonk verification error: {0}")]
    Plonk(anyhow::Error),
    /// An error that occurs when the Groth16 verification fails.
    #[error("Groth16 verification error: {0}")]
    Groth16(anyhow::Error),
    /// An error that occurs when the proof is invalid.
    #[error("Unexpected error: {0:?}")]
    Other(anyhow::Error),
}

/// In zkDTVM, a proof's public values can either be hashed with SHA2 or Blake3. In zkDTVM V4, there
/// is no metadata attached to the proof about which hasher function was used for public values
/// hashing. Instead, when verifying the proof, the public values are hashed with SHA2 and Blake3,
/// and if either matches the `expected_public_values_hash`, the verification is successful.
///
/// The security for this verification in zkDTVM V4 derives from the fact that both SHA2 and Blake3
/// are designed to be collision resistant. It is computationally infeasible to find an input i1 for
/// SHA256 and an input i2 for Blake3 that the same hash value. Doing so would require breaking both
/// algorithms simultaneously.
pub fn verify_proof<C: DTProverComponents>(
    prover: &DTProver<C>,
    version: &str,
    bundle: &DTProofWithPublicValues,
    vkey: &DTVerifyingKey,
) -> Result<(), DTVerificationError> {
    // Check that the zkDTVM version matches the version of the currentcircuit.
    if bundle.dt_version != version {
        return Err(DTVerificationError::VersionMismatch(bundle.dt_version.clone()));
    }

    match &bundle.proof {
        DTProof::Core(proof) => {
            let public_values: &PublicValues<Word<_>, _> =
                proof.last().unwrap().public_values.as_slice().borrow();

            // Get the committed value digest bytes.
            let committed_value_digest_bytes = public_values
                .committed_value_digest
                .iter()
                .flat_map(|w| w.0.iter().map(|x| x.as_canonical_u32() as u8))
                .collect_vec();

            // Make sure the committed value digest matches the public values hash.
            // It is computationally infeasible to find two distinct inputs, one processed with
            // SHA256 and the other with Blake3, that yield the same hash value.
            if committed_value_digest_bytes != bundle.public_values.hash() &&
                committed_value_digest_bytes != bundle.public_values.blake3_hash()
            {
                return Err(DTVerificationError::InvalidPublicValues);
            }

            // Verify the core proof.
            prover.verify(&DTCoreProofData(proof.clone()), vkey).map_err(DTVerificationError::Core)
        }
        DTProof::Compressed(proof) => {
            let public_values: &PublicValues<Word<_>, _> =
                proof.proof.public_values.as_slice().borrow();

            // Get the committed value digest bytes.
            let committed_value_digest_bytes = public_values
                .committed_value_digest
                .iter()
                .flat_map(|w| w.0.iter().map(|x| x.as_canonical_u32() as u8))
                .collect_vec();

            // Make sure the committed value digest matches the public values hash.
            // It is computationally infeasible to find two distinct inputs, one processed with
            // SHA256 and the other with Blake3, that yield the same hash value.
            if committed_value_digest_bytes != bundle.public_values.hash() &&
                committed_value_digest_bytes != bundle.public_values.blake3_hash()
            {
                return Err(DTVerificationError::InvalidPublicValues);
            }

            prover.verify_compressed(proof, vkey).map_err(DTVerificationError::Recursion)
        }
        DTProof::Shrink(proof) => {
            let public_values: &PublicValues<Word<_>, _> =
                proof.proof.public_values.as_slice().borrow();

            // Get the committed value digest bytes.
            let committed_value_digest_bytes = public_values
                .committed_value_digest
                .iter()
                .flat_map(|w| w.0.iter().map(|x| x.as_canonical_u32() as u8))
                .collect_vec();

            // Make sure the committed value digest matches the public values hash.
            if committed_value_digest_bytes != bundle.public_values.hash() &&
                committed_value_digest_bytes != bundle.public_values.blake3_hash()
            {
                return Err(DTVerificationError::InvalidPublicValues);
            }

            prover.verify_shrink(proof, vkey).map_err(DTVerificationError::Recursion)
        }
        DTProof::Plonk(proof) => prover
            .verify_plonk_bn254(
                proof,
                vkey,
                &bundle.public_values,
                &if dt_prover::build::dt_dev_mode() {
                    dt_prover::build::plonk_bn254_artifacts_dev_dir()
                } else {
                    try_install_circuit_artifacts("plonk")
                },
            )
            .map_err(DTVerificationError::Plonk),
        DTProof::Groth16(proof) => prover
            .verify_groth16_bn254(
                proof,
                vkey,
                &bundle.public_values,
                &if dt_prover::build::dt_dev_mode() {
                    dt_prover::build::groth16_bn254_artifacts_dev_dir()
                } else {
                    try_install_circuit_artifacts("groth16")
                },
            )
            .map_err(DTVerificationError::Groth16),
    }
}
