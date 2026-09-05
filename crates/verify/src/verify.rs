use std::cell::OnceCell;

use bincode::Options as _;
use dt_core_executor::deserialize_reduce_proof_bounded;
#[cfg(not(target_arch = "wasm32"))]
use dt_prover::{components::SCCpuProverComponents, DTProver};
use dt_stark::{
    global_d11::{canonical_program_boundary_fields_v1, validate_global146_identity},
    MachineVerificationError,
};
use native_recursion::{
    config::F,
    statement_dt::core_vk_statement_digest,
    verifier_dt::{NativeRootVerifier, NativeRootVerifierArtifactV1},
};
use sha2::{Digest as _, Sha256};

use crate::{CoreSC, DTReduceProof, DTVerifyingKey, RootSC};

pub const MAX_PROOF_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_VK_BYTES: usize = 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 32 * 1024 * 1024;
pub const L4_VERIFIER_ARTIFACT: &[u8] = include_bytes!("../artifacts/l4-q131-full.bin");
// Updated only after exporting and validating the complete trusted release artifact.
pub const L4_VERIFIER_ARTIFACT_SHA256: [u8; 32] = [
    0x22, 0x55, 0xd5, 0xe8, 0x01, 0x72, 0x62, 0x79, 0x65, 0x1f, 0x94, 0x0c, 0xb2, 0x20, 0x67, 0x15,
    0x20, 0x0c, 0x7f, 0x73, 0xd9, 0x8c, 0xb0, 0xbf, 0x35, 0x8e, 0x79, 0x1c, 0x49, 0xae, 0x51, 0xb8,
];

/// Reusable, setup-free verifier for wire-v11, q131, full-opening root proofs.
/// The release supplies L4 authority; the caller supplies its expected application VK.
pub struct CompressedVerifier {
    native_root: NativeRootVerifier,
}

impl CompressedVerifier {
    pub fn new() -> Result<Self, String> {
        Self::from_artifact_bytes(L4_VERIFIER_ARTIFACT)
    }

    /// Loads the exact pinned release artifact. Arbitrary runtime artifacts are rejected.
    pub fn from_artifact_bytes(artifact_bytes: &[u8]) -> Result<Self, String> {
        if artifact_bytes.is_empty() || artifact_bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(
                "DTV_RELEASE_ARTIFACT_INVALID: artifact size is outside the release limit".into(),
            );
        }
        let hash: [u8; 32] = Sha256::digest(artifact_bytes).into();
        if hash != L4_VERIFIER_ARTIFACT_SHA256 {
            return Err(
                "DTV_RELEASE_ARTIFACT_INVALID: artifact does not match the release SHA-256".into(),
            );
        }
        let artifact = bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize::<NativeRootVerifierArtifactV1>(artifact_bytes)
            .map_err(|e| format!("DTV_RELEASE_ARTIFACT_INVALID: {e}"))?;
        let native_root = NativeRootVerifier::from_artifact(artifact)
            .map_err(|e| format!("DTV_RELEASE_ARTIFACT_INVALID: {e}"))?;
        Ok(Self { native_root })
    }

    pub fn verify_compressed_bytes(
        &self,
        proof_bytes: &[u8],
        vk_bytes: &[u8],
    ) -> Result<(), String> {
        validate_input_lengths(proof_bytes.len(), vk_bytes.len())?;
        let proof = deserialize_reduce_proof_bounded::<RootSC>(proof_bytes)
            .map_err(|e| format!("DTV_MALFORMED_PROOF: {e}"))?;
        let vk = deserialize_vk_bytes(vk_bytes)?;
        self.verify_compressed(&proof, &vk)
    }

    pub fn verify_compressed(
        &self,
        proof: &DTReduceProof<RootSC>,
        vk: &DTVerifyingKey,
    ) -> Result<(), String> {
        // The digest helper assumes canonical inputs; validate before calling it.
        validate_global146_identity(&vk.vk.global146_identity)
            .map_err(|e| format!("DTV_INVALID_VK: {e}"))?;
        canonical_program_boundary_fields_v1::<F>(&vk.vk.program_boundary)
            .map_err(|e| format!("DTV_INVALID_VK: {e:?}"))?;
        vk.vk
            .owner_registry
            .validate()
            .map_err(|e| format!("DTV_INVALID_VK: {e:?}"))?;
        let expected = core_vk_statement_digest(
            &vk.vk.commit,
            vk.vk.pc_start,
            &vk.vk.program_boundary,
            &vk.vk.global146_identity,
        );
        self.native_root
            .verify_full_with_statement(proof, &expected, &vk.vk.program_boundary)
            .map_err(|e| format!("DTV_PROOF_INVALID: {e}"))
    }
}

/// Check before copying JS-owned inputs into WASM memory.
pub fn validate_input_lengths(proof_len: usize, vk_len: usize) -> Result<(), String> {
    if proof_len == 0 || vk_len == 0 {
        return Err("DTV_EMPTY_INPUT: proof and full VK are required".into());
    }
    if proof_len > MAX_PROOF_BYTES || vk_len > MAX_VK_BYTES {
        return Err(format!(
            "DTV_INPUT_TOO_LARGE: proof limit={MAX_PROOF_BYTES}, VK limit={MAX_VK_BYTES}"
        ));
    }
    Ok(())
}

pub fn deserialize_vk_bytes(bytes: &[u8]) -> Result<DTVerifyingKey, String> {
    if bytes.is_empty() || bytes.len() > MAX_VK_BYTES {
        return Err("DTV_INVALID_VK: full VK size is outside the release limit".into());
    }
    // deserialize_from keeps bincode's byte budget active for nested collections.
    let mut reader = std::io::Cursor::new(bytes);
    let vk = bincode::DefaultOptions::new()
        .with_little_endian()
        .with_fixint_encoding()
        .with_limit(MAX_VK_BYTES as u64)
        .deserialize_from(&mut reader)
        .map_err(|e| format!("DTV_INVALID_VK: expected full DTVerifyingKey: {e}"))?;
    if reader.position() as usize != bytes.len() {
        return Err("DTV_INVALID_VK: trailing bytes".into());
    }
    Ok(vk)
}

thread_local! {
    static VERIFIER: OnceCell<Result<CompressedVerifier, String>> = const { OnceCell::new() };
}

pub fn verify_compressed_bytes(proof_bytes: &[u8], vk_bytes: &[u8]) -> Result<(), String> {
    validate_input_lengths(proof_bytes.len(), vk_bytes.len())?;
    VERIFIER.with(|cell| match cell.get_or_init(CompressedVerifier::new) {
        Ok(verifier) => verifier.verify_compressed_bytes(proof_bytes, vk_bytes),
        Err(error) => Err(error.clone()),
    })
}

pub fn verify_compressed(
    proof: &DTReduceProof<RootSC>,
    vk: &DTVerifyingKey,
) -> Result<(), MachineVerificationError<CoreSC>> {
    VERIFIER
        .with(|cell| match cell.get_or_init(CompressedVerifier::new) {
            Ok(verifier) => verifier.verify_compressed(proof, vk),
            Err(error) => Err(error.clone()),
        })
        .map_err(MachineVerificationError::NativeRecursion)
}

/// Release tooling only. Verification entry points never construct the producer or run setup.
#[cfg(not(target_arch = "wasm32"))]
pub fn build_l4_verifier_artifact_bytes_for_vk(vk: &DTVerifyingKey) -> Result<Vec<u8>, String> {
    let prover = DTProver::<SCCpuProverComponents>::new();
    let backend = prover.native_backend().map_err(|e| e.to_string())?;
    let artifact = backend
        .root_verifier_artifact(&vk.vk)
        .map_err(|e| e.to_string())?;
    let bytes = bincode::serialize(&artifact).map_err(|e| e.to_string())?;
    artifact
        .validate_serialized_roundtrip(&bytes)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn build_l4_verifier_artifact_bytes() -> Result<Vec<u8>, String> {
    let vk = deserialize_vk_bytes(include_bytes!("../../../vk-full.bin"))?;
    build_l4_verifier_artifact_bytes_for_vk(&vk)
}
