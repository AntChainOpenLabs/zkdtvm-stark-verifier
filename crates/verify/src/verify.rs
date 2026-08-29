use dt_prover::{components::SCCpuProverComponents, DTProver};
use dt_stark::MachineVerificationError;

use crate::{CoreSC, DTReduceProof, DTVerifyingKey, RootSC};

/// Byte-level entry point for WASM and other FFI consumers.
///
/// `proof_bytes` must be a bincode-serialized `DTReduceProof<RootSC>` using the
/// v0.8.0 wire format. `vk_bytes` must contain a full bincode-serialized
/// `DTVerifyingKey`; a digest alone is insufficient for the native-recursion
/// external checks required by this release.
pub fn verify_compressed_bytes(proof_bytes: &[u8], vk_bytes: &[u8]) -> Result<(), String> {
    let proof: DTReduceProof<RootSC> =
        bincode::deserialize(proof_bytes).map_err(|e| format!("proof deserialize: {e}"))?;
    let vk: DTVerifyingKey = bincode::deserialize(vk_bytes)
        .map_err(|e| format!("vk deserialize: expected full DTVerifyingKey: {e}"))?;
    verify_compressed(&proof, &vk).map_err(|e| format!("{e:?}"))
}

pub fn verify_compressed(
    proof: &DTReduceProof<RootSC>,
    vk: &DTVerifyingKey,
) -> Result<(), MachineVerificationError<CoreSC>> {
    DTProver::<SCCpuProverComponents>::new().verify_compressed(proof, vk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
    }

    fn load_fixture_bytes() -> Option<(Vec<u8>, Vec<u8>)> {
        let root = fixture_dir();
        let proof_path = root.join("proof.bin");
        let full_vk_path = root.join("vk-full.bin");

        if !proof_path.exists() || !full_vk_path.exists() {
            eprintln!(
                "Fixture files not found at {}. Skipping test.",
                root.display()
            );
            return None;
        }

        let proof_bytes = std::fs::read(&proof_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", proof_path.display()));
        let vk_bytes = std::fs::read(&full_vk_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", full_vk_path.display()));

        Some((proof_bytes, vk_bytes))
    }

    #[test]
    fn test_verify_compressed_with_fixtures() {
        let Some((proof_bytes, vk_bytes)) = load_fixture_bytes() else {
            return;
        };

        let result = verify_compressed_bytes(&proof_bytes, &vk_bytes);
        assert!(
            result.is_ok(),
            "Compressed proof verification failed: {:?}",
            result.err()
        );
    }
}
