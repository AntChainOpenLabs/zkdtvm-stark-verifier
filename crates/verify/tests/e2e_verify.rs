//! End-to-end verification test using fixture files.
//!
//! Loads `proof.bin` and `vk.bin` from the project root and runs
//! the full compressed-proof verification pipeline, mirroring the CLI binary.
//!
//! If fixture files are not present the test is silently skipped so that
//! `cargo test` always passes in environments where fixtures have not yet
//! been generated (they require 32GB+ RAM).

use std::path::PathBuf;

use zkdtvm_stark_verifier::{verify_compressed, DTReduceProof, DTVerifyingKey, HashableKey, InnerSC};

fn project_root() -> PathBuf {
    // crates/verify -> ../.. = project root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn has_fixtures() -> bool {
    let root = project_root();
    root.join("proof.bin").exists() && root.join("vk.bin").exists()
}

fn load_proof() -> DTReduceProof<InnerSC> {
    let bytes = std::fs::read(project_root().join("proof.bin")).unwrap();
    bincode::deserialize(&bytes).unwrap()
}

fn load_vk() -> DTVerifyingKey {
    let bytes = std::fs::read(project_root().join("vk.bin")).unwrap();
    bincode::deserialize(&bytes).unwrap()
}

#[test]
fn e2e_verify_compressed_proof() {
    if !has_fixtures() {
        eprintln!(
            "Fixture files (proof.bin, vk.bin) not found at {}. \
             Run `gen_verifier_fixtures` to generate them. Skipping.",
            project_root().display()
        );
        return;
    }

    let proof = load_proof();
    let vk = load_vk();

    let result = verify_compressed(&proof, &vk);
    assert!(result.is_ok(), "E2E verification failed: {:?}", result.err());
}

#[test]
fn e2e_vk_hash_deterministic() {
    if !has_fixtures() {
        return;
    }

    let vk = load_vk();
    let h1 = vk.hash_babybear();
    let h2 = vk.hash_babybear();
    assert_eq!(h1, h2, "VK hash should be deterministic across calls");
}

#[test]
fn e2e_message_readable() {
    let msg_path = project_root().join("message.bin");
    if !msg_path.exists() {
        eprintln!("message.bin not found, skipping.");
        return;
    }

    let bytes = std::fs::read(&msg_path).unwrap();
    let message: String = bincode::deserialize(&bytes).unwrap();
    assert!(!message.is_empty());
    println!("Message: {message}");
}
