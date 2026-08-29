use std::{fs::File, path::Path};

use anyhow::Result;
use clap::ValueEnum;
use dt_core_machine::{io::DTStdin, reduce::DTReduceProof};
use dt_primitives::{io::DTPublicValues, sc_poseidon2_hash, SCField};
use p3_bn254_fr::Bn254Fr;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
// use dt_recursion_circuit::machine::{
//     DTCompressWitnessValues, DTDeferredWitnessValues, DTRecursionWitnessValues,
// };
use dt_recursion_circuit::sc_machine::{
    SCDTCompressWitnessValues, SCDTDeferredWitnessValues, SCDTRecursionWitnessValues,
};
use dt_recursion_gnark_ffi::proof::{Groth16Bn254Proof, PlonkBn254Proof};

use dt_stark::{
    sumcheck::{
        config::SCStarkGenericConfig,
        keys::{SCStarkProvingKey, SCStarkVerifyingKey},
        proof::SCShardProof,
    },
    DIGEST_SIZE,
};
use thiserror::Error;

use crate::{
    utils::{babybears_to_bn254, words_to_bytes_be},
    CoreSC, InnerSC, RootSC,
};
use pcs::basefold::mlpcs::MlPCS;

/// The information necessary to generate a proof for a given RISC-V program.
#[derive(Clone, Serialize, Deserialize)]
pub struct DTProvingKey {
    pub pk: SCStarkProvingKey<CoreSC>,
    pub elf: Vec<u8>,
    /// Verifying key is also included as we need it for recursion
    pub vk: DTVerifyingKey,
}

/// The information necessary to verify a proof for a given RISC-V program.
#[derive(Clone, Serialize, Deserialize)]
pub struct DTVerifyingKey {
    pub vk: SCStarkVerifyingKey<CoreSC>,
}

/// A trait for keys that can be hashed into a digest.
pub trait HashableKey {
    /// Hash the key into a digest of field elements (BabyBear or KoalaBear depending on feature).
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE];

    /// Hash the key into a digest of u32 elements.
    fn hash_u32(&self) -> [u32; DIGEST_SIZE];

    /// Hash the key into a Bn254Fr element.
    fn hash_bn254(&self) -> Bn254Fr {
        babybears_to_bn254(&self.hash_babybear())
    }

    /// Hash the key into a 32 byte hex string, prefixed with "0x".
    ///
    /// This is ideal for generating a vkey hash for onchain verification.
    fn bytes32(&self) -> String {
        let vkey_digest_bn254 = self.hash_bn254();
        format!("0x{:0>64}", vkey_digest_bn254.as_canonical_biguint().to_str_radix(16))
    }

    /// Hash the key into a 32 byte array.
    ///
    /// This has the same value as `bytes32`, but as a raw byte array.
    fn bytes32_raw(&self) -> [u8; 32] {
        let vkey_digest_bn254 = self.hash_bn254();
        let vkey_bytes = vkey_digest_bn254.as_canonical_biguint().to_bytes_be();
        let mut result = [0u8; 32];
        result[1..].copy_from_slice(&vkey_bytes);
        result
    }

    /// Hash the key into a digest of bytes elements.
    fn hash_bytes(&self) -> [u8; DIGEST_SIZE * 4] {
        words_to_bytes_be(&self.hash_u32())
    }
}

impl HashableKey for DTVerifyingKey {
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE] {
        self.vk.hash_babybear()
    }

    fn hash_u32(&self) -> [u32; DIGEST_SIZE] {
        self.vk.hash_u32()
    }
}

impl<SC: SCStarkGenericConfig<Val = SCField>> HashableKey for SCStarkVerifyingKey<SC>
where
    <SC::Mlpcs as MlPCS>::Commitment: AsRef<[SCField; DIGEST_SIZE]>,
{
    fn hash_babybear(&self) -> [SCField; DIGEST_SIZE] {
        sc_poseidon2_hash(self.canonical_hash_inputs())
    }

    fn hash_u32(&self) -> [u32; 8] {
        self.hash_babybear()
            .into_iter()
            .map(|n| n.as_canonical_u32())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

/// A proof of a RISCV ELF execution with given inputs and outputs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "P: Serialize"))]
#[serde(bound(deserialize = "P: DeserializeOwned"))]
pub struct DTProofWithMetadata<P: Clone> {
    pub proof: P,
    pub stdin: DTStdin,
    pub public_values: DTPublicValues,
    pub cycles: u64,
}

#[cfg(test)]
mod global146_identity_tests {
    #[test]
    fn circuit_version_is_the_version_bound_by_global146_identity() {
        assert_eq!(
            crate::DT_CIRCUIT_VERSION.trim_end(),
            dt_stark::global_d11::GLOBAL146_CIRCUIT_VERSION
        );
    }
}

impl<P: Serialize + DeserializeOwned + Clone> DTProofWithMetadata<P> {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        bincode::serialize_into(File::create(path).expect("failed to open file"), self)
            .map_err(Into::into)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        bincode::deserialize_from(File::open(path).expect("failed to open file"))
            .map_err(Into::into)
    }
}

impl<P: std::fmt::Debug + Clone> std::fmt::Debug for DTProofWithMetadata<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DTProofWithMetadata").field("proof", &self.proof).finish()
    }
}

/// A proof of a zkDTVM program without any wrapping.
pub type DTCoreProof = DTProofWithMetadata<DTCoreProofData>;

/// An zkDTVM proof that has been recursively reduced into a single proof. This proof can be
/// verified within zkDTVM programs.
pub type DTReducedProof = DTProofWithMetadata<DTReducedProofData>;

/// An zkDTVM proof that has been wrapped into a single PLONK proof and can be verified onchain.
pub type DTPlonkBn254Proof = DTProofWithMetadata<DTPlonkBn254ProofData>;

/// An zkDTVM proof that has been wrapped into a single Groth16 proof and can be verified onchain.
pub type DTGroth16Bn254Proof = DTProofWithMetadata<DTGroth16Bn254ProofData>;

/// An zkDTVM proof that has been wrapped into a single proof and can be verified onchain.
pub type DTProof = DTProofWithMetadata<DTBn254ProofData>;

#[derive(Serialize, Deserialize, Clone)]
pub struct DTCoreProofData(pub Vec<SCShardProof<CoreSC>>);

#[derive(Serialize, Deserialize, Clone)]
pub struct DTReducedProofData(pub SCShardProof<RootSC>);

#[derive(Serialize, Deserialize, Clone)]
pub struct DTPlonkBn254ProofData(pub PlonkBn254Proof);

#[derive(Serialize, Deserialize, Clone)]
pub struct DTGroth16Bn254ProofData(pub Groth16Bn254Proof);

#[derive(Serialize, Deserialize, Clone)]
pub enum DTBn254ProofData {
    Plonk(PlonkBn254Proof),
    Groth16(Groth16Bn254Proof),
}

impl DTBn254ProofData {
    pub fn get_proof_system(&self) -> ProofSystem {
        match self {
            DTBn254ProofData::Plonk(_) => ProofSystem::Plonk,
            DTBn254ProofData::Groth16(_) => ProofSystem::Groth16,
        }
    }

    pub fn get_raw_proof(&self) -> &str {
        match self {
            DTBn254ProofData::Plonk(proof) => &proof.raw_proof,
            DTBn254ProofData::Groth16(proof) => &proof.raw_proof,
        }
    }
}

/// The mode of the prover.
#[derive(Debug, Default, Clone, ValueEnum, PartialEq, Eq)]
pub enum ProverMode {
    #[default]
    Cpu,
    Cuda,
    Network,
    Mock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofSystem {
    Plonk,
    Groth16,
}

impl ProofSystem {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProofSystem::Plonk => "Plonk",
            ProofSystem::Groth16 => "Groth16",
        }
    }
}

/// A proof that can be reduced along with other proofs into one proof.
#[derive(Serialize, Deserialize, Clone)]
pub enum DTReduceProofWrapper {
    Core(DTReduceProof<CoreSC>),
    Recursive(DTReduceProof<InnerSC>),
    Root(DTReduceProof<RootSC>),
}

#[derive(Error, Debug)]
pub enum DTRecursionProverError {
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

#[allow(clippy::large_enum_variant)]
pub enum SCDTCircuitWitness {
    Core(SCDTRecursionWitnessValues<CoreSC>),
    Deferred(SCDTDeferredWitnessValues<InnerSC>),
    Compress(SCDTCompressWitnessValues<InnerSC>),
}
