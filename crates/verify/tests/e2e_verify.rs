//! End-to-end verification test using fixture files.
//!
//! Loads `proof.bin` and `vk-full.bin` from the project root and runs
//! the full compressed-proof verification pipeline, mirroring the CLI binary.
//!
//! If fixture files are not present the test is silently skipped so that
//! `cargo test` always passes in environments where fixtures have not yet
//! been generated (they require 32GB+ RAM).

use std::path::PathBuf;

use zkdtvm_stark_verifier::{verify_compressed_bytes, DTVerifyingKey, HashableKey, DIGEST_SIZE};

fn project_root() -> PathBuf {
    // crates/verify -> ../.. = project root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn has_fixtures() -> bool {
    let root = project_root();
    root.join("proof.bin").exists() && root.join("vk-full.bin").exists()
}

fn load_proof_bytes() -> Vec<u8> {
    std::fs::read(project_root().join("proof.bin")).unwrap()
}

fn load_vk_bytes() -> Vec<u8> {
    std::fs::read(project_root().join("vk-full.bin")).unwrap()
}

fn load_vk_digest_u32() -> [u32; DIGEST_SIZE] {
    let bytes = load_vk_bytes();
    match bincode::deserialize::<[u32; DIGEST_SIZE]>(&bytes) {
        Ok(digest) => digest,
        Err(_) => {
            let vk: DTVerifyingKey = bincode::deserialize(&bytes).unwrap();
            vk.hash_u32()
        }
    }
}

#[test]
fn e2e_verify_compressed_proof() {
    if !has_fixtures() {
        eprintln!(
            "Fixture files (proof.bin, vk-full.bin) not found at {}. \
             Run `gen_verifier_fixtures` to generate them. Skipping.",
            project_root().display()
        );
        return;
    }

    let result = verify_compressed_bytes(&load_proof_bytes(), &load_vk_bytes());
    assert!(
        result.is_ok(),
        "E2E verification failed: {:?}",
        result.err()
    );
}

#[test]
fn e2e_vk_hash_deterministic() {
    if !has_fixtures() {
        return;
    }

    let h1 = load_vk_digest_u32();
    let h2 = load_vk_digest_u32();
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
