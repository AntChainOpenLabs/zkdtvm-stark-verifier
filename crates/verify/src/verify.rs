use std::borrow::Borrow;

use dt_primitives::SCField;
use dt_recursion::air::RecursionPublicValues;
use dt_recursion::machine::RecursionAir;
use dt_recursion::shape::{CHIP_LOG_HEIGHT_THRESHOLD, NUM_SKIP_ROUNDS};
use dt_stark::sumcheck::config::SCStarkGenericConfig;
use dt_stark::sumcheck::proof::SCMachineProof;
use dt_stark::MachineVerificationError;
use p3_field::AbstractField;

use crate::{CoreSC, DTReduceProof, DTVerifyingKey, HashableKey, InnerSC};

pub const COMPRESS_DEGREE: usize = 3;

pub fn verify_compressed(
    proof: &DTReduceProof<InnerSC>,
    vk: &DTVerifyingKey,
) -> Result<(), MachineVerificationError<CoreSC>> {
    let DTReduceProof { vk: compress_vk, proof } = proof;

    let config = InnerSC::default();

    let compress_machine =
        RecursionAir::<_, COMPRESS_DEGREE>::sc_compress_machine(config.clone());

    let mut challenger = config.mlchallenger();

    let machine_proof = SCMachineProof { shard_proofs: vec![proof.clone()] };

    compress_machine.verify(
        compress_vk,
        &machine_proof,
        &mut challenger,
        NUM_SKIP_ROUNDS,
        CHIP_LOG_HEIGHT_THRESHOLD,
    )?;

    let public_values: &RecursionPublicValues<_> = proof.public_values.as_slice().borrow();

    if public_values.is_complete != SCField::one() {
        return Err(MachineVerificationError::InvalidPublicValues("is_complete is not 1"));
    }

    let vkey_hash = vk.hash_babybear();
    if public_values.dt_vk_digest != vkey_hash {
        return Err(MachineVerificationError::InvalidPublicValues("dt vk hash mismatch"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DIGEST_SIZE;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
    }

    fn load_fixtures() -> Option<(DTReduceProof<InnerSC>, DTVerifyingKey)> {
        let root = fixture_dir();
        let proof_path = root.join("proof.bin");
        let vk_path = root.join("vk.bin");

        if !proof_path.exists() || !vk_path.exists() {
            eprintln!(
                "Fixture files not found at {}. \
                 Generate them with `gen_verifier_fixtures` on a 32GB+ machine. Skipping test.",
                root.display()
            );
            return None;
        }

        let proof_bytes = std::fs::read(&proof_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", proof_path.display()));
        let proof: DTReduceProof<InnerSC> = bincode::deserialize(&proof_bytes)
            .unwrap_or_else(|e| panic!("Failed to deserialize proof: {e}"));

        let vk_bytes = std::fs::read(&vk_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", vk_path.display()));
        let vk: DTVerifyingKey = bincode::deserialize(&vk_bytes)
            .unwrap_or_else(|e| panic!("Failed to deserialize vk: {e}"));

        Some((proof, vk))
    }

    #[test]
    fn test_verify_compressed_with_fixtures() {
        let Some((proof, vk)) = load_fixtures() else {
            return;
        };

        let result = verify_compressed(&proof, &vk);
        assert!(result.is_ok(), "Compressed proof verification failed: {:?}", result.err());
    }

    #[test]
    fn test_proof_deserialization() {
        let root = fixture_dir();
        let proof_path = root.join("proof.bin");
        if !proof_path.exists() {
            eprintln!("proof.bin not found, skipping deserialization test.");
            return;
        }

        let proof_bytes = std::fs::read(&proof_path).unwrap();
        let proof: Result<DTReduceProof<InnerSC>, _> = bincode::deserialize(&proof_bytes);
        assert!(proof.is_ok(), "Failed to deserialize proof.bin: {:?}", proof.err());

        let proof = proof.unwrap();
        assert!(
            !proof.proof.public_values.is_empty(),
            "Proof should have non-empty public values"
        );
    }

    #[test]
    fn test_vk_deserialization_and_hash() {
        let root = fixture_dir();
        let vk_path = root.join("vk.bin");
        if !vk_path.exists() {
            eprintln!("vk.bin not found, skipping VK deserialization test.");
            return;
        }

        let vk_bytes = std::fs::read(&vk_path).unwrap();
        let vk: Result<DTVerifyingKey, _> = bincode::deserialize(&vk_bytes);
        assert!(vk.is_ok(), "Failed to deserialize vk.bin: {:?}", vk.err());

        let vk = vk.unwrap();
        let hash = vk.hash_babybear();
        assert_eq!(hash.len(), DIGEST_SIZE, "VK hash should have {DIGEST_SIZE} elements");

        let hash_u32 = vk.hash_u32();
        assert_eq!(hash_u32.len(), DIGEST_SIZE, "VK hash_u32 should have {DIGEST_SIZE} elements");

        let hash_again = vk.hash_babybear();
        assert_eq!(hash, hash_again, "VK hashing should be deterministic");
    }

    #[test]
    fn test_message_deserialization() {
        let root = fixture_dir();
        let msg_path = root.join("message.bin");
        if !msg_path.exists() {
            eprintln!("message.bin not found, skipping message deserialization test.");
            return;
        }

        let msg_bytes = std::fs::read(&msg_path).unwrap();
        let message: Result<String, _> = bincode::deserialize(&msg_bytes);
        assert!(message.is_ok(), "Failed to deserialize message.bin: {:?}", message.err());

        let message = message.unwrap();
        assert!(!message.is_empty(), "Message should not be empty");
    }

    #[test]
    fn test_verify_with_wrong_vk_fails() {
        let Some((proof, mut vk)) = load_fixtures() else {
            return;
        };

        vk.vk.pc_start = SCField::from_canonical_u32(0xDEAD);

        let result = verify_compressed(&proof, &vk);
        assert!(result.is_err(), "Verification should fail with a tampered VK");
    }
}
