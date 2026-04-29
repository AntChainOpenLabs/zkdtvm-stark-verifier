use std::borrow::Borrow;

use dt_primitives::SCField;
use dt_recursion::air::RecursionPublicValues;
use dt_recursion::machine::RecursionAir;
use dt_recursion::shape::{CHIP_LOG_HEIGHT_THRESHOLD, NUM_SKIP_ROUNDS};
use dt_stark::sumcheck::config::SCStarkGenericConfig;
use dt_stark::sumcheck::proof::SCMachineProof;
use dt_stark::{MachineVerificationError, DIGEST_SIZE};
use p3_field::AbstractField;
use thiserror::Error;

use crate::{CoreSC, DTReduceProof, DTVerifyingKey, HashableKey, InnerSC};

pub const COMPRESS_DEGREE: usize = 3;

/// Errors returned by the byte-oriented verification entry points.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("failed to deserialize input: {0}")]
    DeserializationError(String),

    #[error("machine verification failed: {0}")]
    MachineVerificationFailed(String),

    #[error("invalid public values: {0}")]
    InvalidPublicValues(&'static str),

    #[error("proof is not marked as complete")]
    ProofNotComplete,

    #[error("dt_vk_digest in proof does not match the supplied vk hash")]
    VkDigestMismatch,
}

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

/// Verify a compressed proof against a pre-computed VK digest instead of a full `DTVerifyingKey`.
///
/// Mirrors [`verify_compressed`] but is intended for external integrators that only hold the
/// Poseidon2 digest of the core `DTVerifyingKey` (the value embedded in the proof's public
/// values as `dt_vk_digest`). The compress-stage VK used by the machine verifier itself is
/// taken from inside `proof` (see `DTReduceProof::vk`).
pub fn verify_compressed_raw(
    proof: &DTReduceProof<InnerSC>,
    vk_hash: &[SCField; DIGEST_SIZE],
) -> Result<(), VerifyError> {
    let DTReduceProof { vk: compress_vk, proof } = proof;

    let config = InnerSC::default();
    let compress_machine =
        RecursionAir::<_, COMPRESS_DEGREE>::sc_compress_machine(config.clone());
    let mut challenger = config.mlchallenger();
    let machine_proof = SCMachineProof { shard_proofs: vec![proof.clone()] };

    compress_machine
        .verify(
            compress_vk,
            &machine_proof,
            &mut challenger,
            NUM_SKIP_ROUNDS,
            CHIP_LOG_HEIGHT_THRESHOLD,
        )
        .map_err(|e| VerifyError::MachineVerificationFailed(format!("{:?}", e)))?;

    let public_values: &RecursionPublicValues<_> = proof.public_values.as_slice().borrow();

    if public_values.is_complete != SCField::one() {
        return Err(VerifyError::ProofNotComplete);
    }

    if public_values.dt_vk_digest != *vk_hash {
        return Err(VerifyError::VkDigestMismatch);
    }

    Ok(())
}

/// Byte-oriented entry point: verify a compressed proof from its serialized representation.
///
/// - `proof_bytes`: bincode-serialized [`DTReduceProof<InnerSC>`]
///   (carries both the compress-stage VK and the shard proof).
/// - `vk_hash_bytes`: bincode-serialized `[SCField; DIGEST_SIZE]`
///   (Poseidon2 digest of the core `DTVerifyingKey`).
///
/// Delegates to [`verify_compressed_raw`]. Named with a `_bytes` suffix to avoid clashing
/// with the pre-existing [`verify_compressed`] that takes already-deserialized structures.
pub fn verify_compressed_bytes(
    proof_bytes: &[u8],
    vk_hash_bytes: &[u8],
) -> Result<(), VerifyError> {
    let proof: DTReduceProof<InnerSC> = bincode::deserialize(proof_bytes)
        .map_err(|e| VerifyError::DeserializationError(format!("proof: {}", e)))?;

    let vk_hash: [SCField; DIGEST_SIZE] = bincode::deserialize(vk_hash_bytes)
        .map_err(|e| VerifyError::DeserializationError(format!("vk_hash: {}", e)))?;

    verify_compressed_raw(&proof, &vk_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DIGEST_SIZE;
    use std::path::PathBuf;

    // ----------------------------------------------------------------------
    // Legacy fixtures: project root `proof.bin` + full `DTVerifyingKey`
    // `vk.bin` (bincode-serialized `DTVerifyingKey`).
    // These tests match the main-branch behaviour and exercise the original
    // `verify_compressed(proof, &DTVerifyingKey)` entry point.
    // ----------------------------------------------------------------------

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

    // ----------------------------------------------------------------------
    // Byte-oriented fixtures: `crates/verify/tests/fixtures/example_{1..=4}/`.
    // Each directory contains:
    //   - `proof.bin`: bincode-serialized `DTReduceProof<InnerSC>`
    //   - `vk.bin`:    bincode-serialized `[SCField; DIGEST_SIZE]` (vk digest,
    //                  32 bytes) — the input shape expected by
    //                  `verify_compressed_bytes`.
    // These tests exercise the new `verify_compressed_raw` /
    // `verify_compressed_bytes` entry points designed for external integrators.
    // ----------------------------------------------------------------------

    const EXAMPLE_NAMES: [&str; 4] = ["example_1", "example_2", "example_3", "example_4"];

    fn bytes_fixtures_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn load_example_bytes(name: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        let dir = bytes_fixtures_root().join(name);
        let proof_path = dir.join("proof.bin");
        let vk_path = dir.join("vk.bin");

        if !proof_path.exists() || !vk_path.exists() {
            return None;
        }

        let proof_bytes = std::fs::read(&proof_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", proof_path.display()));
        let vk_hash_bytes = std::fs::read(&vk_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", vk_path.display()));
        Some((proof_bytes, vk_hash_bytes))
    }

    fn load_example_typed(name: &str) -> Option<(DTReduceProof<InnerSC>, [SCField; DIGEST_SIZE])> {
        let (proof_bytes, vk_hash_bytes) = load_example_bytes(name)?;
        let proof: DTReduceProof<InnerSC> = bincode::deserialize(&proof_bytes)
            .unwrap_or_else(|e| panic!("deserialize {} proof: {e}", name));
        let vk_hash: [SCField; DIGEST_SIZE] = bincode::deserialize(&vk_hash_bytes)
            .unwrap_or_else(|e| panic!("deserialize {} vk hash: {e}", name));
        Some((proof, vk_hash))
    }

    #[test]
    fn test_verify_compressed_bytes_bad_proof_bytes() {
        // Garbage bytes must be rejected at the deserialization stage, not panic.
        let proof_bytes = vec![0u8; 16];
        let vk_hash_bytes = bincode::serialize(&[SCField::zero(); DIGEST_SIZE]).unwrap();

        let result = verify_compressed_bytes(&proof_bytes, &vk_hash_bytes);
        assert!(
            matches!(result, Err(VerifyError::DeserializationError(_))),
            "expected DeserializationError, got: {:?}",
            result
        );
    }

    #[test]
    fn test_verify_compressed_bytes_bad_vk_hash_bytes() {
        let Some((proof_bytes, _)) = load_example_bytes(EXAMPLE_NAMES[0]) else {
            eprintln!("example fixtures not found, skipping bad vk hash test.");
            return;
        };

        // Too short to be a valid [SCField; DIGEST_SIZE].
        let vk_hash_bytes = vec![0u8; 4];

        let result = verify_compressed_bytes(&proof_bytes, &vk_hash_bytes);
        assert!(
            matches!(result, Err(VerifyError::DeserializationError(_))),
            "expected DeserializationError, got: {:?}",
            result
        );
    }

    /// Iterate all four example fixtures and exercise both the raw and the byte-oriented
    /// entry points end-to-end, plus a negative control.
    #[test]
    fn test_verify_compressed_bytes_all_example_fixtures() {
        if !bytes_fixtures_root().exists() {
            eprintln!(
                "Byte-fixtures root {} not found, skipping.",
                bytes_fixtures_root().display()
            );
            return;
        }

        let mut verified = 0usize;

        for name in EXAMPLE_NAMES {
            let Some((proof_bytes, vk_hash_bytes)) = load_example_bytes(name) else {
                eprintln!("skipping {}: missing proof.bin or vk.bin", name);
                continue;
            };

            // Sanity: 8 field elements * 4 bytes each = 32 bytes when bincode-encoded.
            assert_eq!(
                vk_hash_bytes.len(),
                DIGEST_SIZE * std::mem::size_of::<u32>(),
                "{}: unexpected vk.bin size {} (expected {})",
                name,
                vk_hash_bytes.len(),
                DIGEST_SIZE * std::mem::size_of::<u32>(),
            );

            // 1. Bytes-in entry point.
            verify_compressed_bytes(&proof_bytes, &vk_hash_bytes).unwrap_or_else(|e| {
                panic!("verify_compressed_bytes failed for {}: {:?}", name, e)
            });

            // 2. Raw entry point (shares the same logic but skips deserialization).
            let (proof, vk_hash) = load_example_typed(name).expect("already checked existence");
            verify_compressed_raw(&proof, &vk_hash).unwrap_or_else(|e| {
                panic!("verify_compressed_raw failed for {}: {:?}", name, e)
            });

            // 3. Negative control: tampering the hash must produce VkDigestMismatch.
            let mut bad_hash = vk_hash;
            bad_hash[0] = bad_hash[0] + SCField::one();
            assert!(
                matches!(
                    verify_compressed_raw(&proof, &bad_hash),
                    Err(VerifyError::VkDigestMismatch)
                ),
                "{}: tampered hash should be rejected",
                name
            );

            verified += 1;
            eprintln!("  ✓ {} verified", name);
        }

        assert!(verified > 0, "no example fixtures were verified");
        eprintln!("verified {} example fixtures", verified);
    }

    /// Confirms the empirical observation that all four example fixtures ship the same
    /// vk digest. In this demo batch every proof's final reduction step lands on the
    /// same compress-shape combination, so `dt_vk_digest` collapses to one value.
    #[test]
    fn test_all_example_vks_are_identical() {
        if !bytes_fixtures_root().exists() {
            eprintln!("Byte-fixtures root not found, skipping.");
            return;
        }

        let mut reference: Option<(String, [SCField; DIGEST_SIZE])> = None;
        for name in EXAMPLE_NAMES {
            let Some((_, vk_hash)) = load_example_typed(name) else {
                eprintln!("skipping {}: missing fixtures", name);
                continue;
            };

            let digest_preview: Vec<u32> = vk_hash
                .iter()
                .take(4)
                .map(p3_field::PrimeField32::as_canonical_u32)
                .collect();
            eprintln!("  {} vk_hash[..4] = {:?}", name, digest_preview);

            match &reference {
                None => reference = Some((name.to_string(), vk_hash)),
                Some((ref_name, ref_hash)) => {
                    assert_eq!(
                        &vk_hash, ref_hash,
                        "vk digest differs: {} vs {}",
                        ref_name, name
                    );
                }
            }
        }
    }

    /// Although all four share the same vk digest, the `proof.bin` payloads themselves
    /// are distinct (different underlying proofs).
    #[test]
    fn test_all_example_proofs_are_distinct() {
        if !bytes_fixtures_root().exists() {
            eprintln!("Byte-fixtures root not found, skipping.");
            return;
        }

        let mut payloads: Vec<(&str, Vec<u8>)> = Vec::new();
        for name in EXAMPLE_NAMES {
            let Some((proof_bytes, _)) = load_example_bytes(name) else { continue };
            payloads.push((name, proof_bytes));
        }

        if payloads.len() < 2 {
            eprintln!("fewer than 2 example proofs available, skipping distinctness check.");
            return;
        }

        for i in 0..payloads.len() {
            for j in (i + 1)..payloads.len() {
                assert_ne!(
                    payloads[i].1, payloads[j].1,
                    "{} and {} should carry distinct proof payloads but are byte-identical",
                    payloads[i].0, payloads[j].0
                );
            }
        }
        eprintln!("confirmed {} example proofs are pairwise distinct", payloads.len());
    }
}
