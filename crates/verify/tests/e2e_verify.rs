use std::path::PathBuf;

use dt_core_executor::deserialize_reduce_proof_bounded;
use p3_field::AbstractField;
use zkdtvm_stark_verifier::{
    deserialize_vk_bytes, verify_compressed_bytes, CompressedVerifier, RootSC, SCField,
    L4_VERIFIER_ARTIFACT, MAX_PROOF_BYTES, MAX_VK_BYTES,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> (Vec<u8>, Vec<u8>) {
    (
        std::fs::read(root().join("proof.bin")).expect("required q131 proof fixture"),
        std::fs::read(root().join("vk-full.bin")).expect("required full VK fixture"),
    )
}

#[test]
fn q131_full_proof_verifies_repeatedly_without_setup() {
    let (proof, vk) = fixtures();
    assert_eq!(proof.len(), 264_238);
    let reduce = deserialize_reduce_proof_bounded::<RootSC>(&proof).unwrap();
    let opening = reduce
        .proof
        .opening_proof
        .query_openings
        .pruned
        .as_ref()
        .unwrap();
    assert_eq!(opening.round_pruned.len(), 3);
    assert_eq!(opening.round_opened_values.len(), 3);
    assert_eq!(opening.query_to_unique_slot.len(), 3);
    let verifier = CompressedVerifier::new().unwrap();
    for _ in 0..3 {
        verifier.verify_compressed_bytes(&proof, &vk).unwrap();
    }
    verify_compressed_bytes(&proof, &vk).unwrap();
}

#[test]
fn rejects_elided_and_tampered_full_proofs_then_reuses_runtime() {
    let (proof, vk) = fixtures();
    let verifier = CompressedVerifier::new().unwrap();
    let original = deserialize_reduce_proof_bounded::<RootSC>(&proof).unwrap();
    let mut elided = original.clone();
    let pruned = elided
        .proof
        .opening_proof
        .query_openings
        .pruned
        .as_mut()
        .unwrap();
    pruned.round_pruned.remove(0);
    pruned.round_opened_values.remove(0);
    pruned.query_to_unique_slot.remove(0);
    let elided_bytes = bincode::serialize(&elided).unwrap();
    let error = verifier
        .verify_compressed_bytes(&elided_bytes, &vk)
        .unwrap_err();
    assert!(
        error.contains("does not carry every input-opening batch"),
        "{error}"
    );

    let mut changed_public = original.clone();
    changed_public.proof.public_values[0] += SCField::one();
    assert!(verifier
        .verify_compressed_bytes(&bincode::serialize(&changed_public).unwrap(), &vk)
        .is_err());

    let mut changed_root_vk = original.clone();
    changed_root_vk.vk.pc_start += SCField::one();
    assert!(verifier
        .verify_compressed_bytes(&bincode::serialize(&changed_root_vk).unwrap(), &vk)
        .is_err());

    let mut changed_sumcheck = original.clone();
    changed_sumcheck.proof.sumcheck_proof.unipolys[0].evals[0] +=
        native_recursion::config::EF::one();
    assert!(verifier
        .verify_compressed_bytes(&bincode::serialize(&changed_sumcheck).unwrap(), &vk)
        .is_err());

    let mut changed_merkle = original.clone();
    let pruned = changed_merkle
        .proof
        .opening_proof
        .query_openings
        .pruned
        .as_mut()
        .unwrap();
    pruned.query_to_unique_slot[0][0] = u32::MAX as _;
    assert!(verifier
        .verify_compressed_bytes(&bincode::serialize(&changed_merkle).unwrap(), &vk)
        .is_err());

    let mut changed_vk = deserialize_vk_bytes(&vk).unwrap();
    changed_vk.vk.pc_start += SCField::one();
    assert!(verifier
        .verify_compressed_bytes(&proof, &bincode::serialize(&changed_vk).unwrap())
        .is_err());
    let mut invalid_vk_identity = changed_vk.clone();
    invalid_vk_identity.vk.global146_identity[0] ^= 1;
    assert!(verifier
        .verify_compressed_bytes(&proof, &bincode::serialize(&invalid_vk_identity).unwrap())
        .is_err());

    // Explicit release tooling may export negatives for the Node/browser WASM gate.
    if let Some(dir) = std::env::var_os("DT_EXPORT_NEGATIVE_FIXTURES") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in [
            ("current-elided.bin", elided_bytes),
            (
                "changed-public.bin",
                bincode::serialize(&changed_public).unwrap(),
            ),
            (
                "changed-root-vk.bin",
                bincode::serialize(&changed_root_vk).unwrap(),
            ),
            (
                "changed-sumcheck.bin",
                bincode::serialize(&changed_sumcheck).unwrap(),
            ),
            (
                "changed-merkle.bin",
                bincode::serialize(&changed_merkle).unwrap(),
            ),
            ("wrong-vk.bin", bincode::serialize(&changed_vk).unwrap()),
        ] {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
    verifier.verify_compressed_bytes(&proof, &vk).unwrap();
}

#[test]
fn rejects_malformed_wire_and_resource_amplification() {
    let (proof, vk) = fixtures();
    let verifier = CompressedVerifier::new().unwrap();
    for len in [0, 1, 4, 35, 36, 256, proof.len() - 1] {
        assert!(verifier
            .verify_compressed_bytes(&proof[..len], &vk)
            .is_err());
    }
    let mut trailing = proof.clone();
    trailing.push(0);
    assert!(verifier.verify_compressed_bytes(&trailing, &vk).is_err());
    let mut wrong_wire = proof.clone();
    wrong_wire[..4].copy_from_slice(&12u32.to_le_bytes());
    assert!(verifier.verify_compressed_bytes(&wrong_wire, &vk).is_err());
    let mut wrong_identity = proof.clone();
    wrong_identity[4] ^= 1;
    assert!(verifier
        .verify_compressed_bytes(&wrong_identity, &vk)
        .is_err());
    let mut trailing_vk = vk.clone();
    trailing_vk.push(0);
    assert!(verifier
        .verify_compressed_bytes(&proof, &trailing_vk)
        .is_err());
    assert!(verifier.verify_compressed_bytes(&proof, &vk[..32]).is_err());
    assert!(verifier
        .verify_compressed_bytes(&vec![0; MAX_PROOF_BYTES + 1], &vk)
        .is_err());
    assert!(verifier
        .verify_compressed_bytes(&proof, &vec![0; MAX_VK_BYTES + 1])
        .is_err());
    let old = include_bytes!("fixtures/legacy-elided.bin");
    assert!(verifier.verify_compressed_bytes(old, &vk).is_err());
    verifier.verify_compressed_bytes(&proof, &vk).unwrap();
}

#[test]
fn rejects_unpinned_artifacts_before_initialization() {
    assert!(CompressedVerifier::from_artifact_bytes(&[]).is_err());
    let mut corrupt = L4_VERIFIER_ARTIFACT.to_vec();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    assert!(CompressedVerifier::from_artifact_bytes(&corrupt).is_err());
    corrupt.push(0);
    assert!(CompressedVerifier::from_artifact_bytes(&corrupt).is_err());
}
