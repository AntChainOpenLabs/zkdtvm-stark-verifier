use anyhow::{Context, Result};
use dt_prover::{components::SCCpuProverComponents, DTProver};
use p3_field::{AbstractField, PrimeField32};
use zkdtvm_stark_verifier::{DTReduceProof, DTVerifyingKey, HashableKey, RootSC, SCField};

fn main() -> Result<()> {
    let proof_bytes = std::fs::read("proof.bin").context("read proof.bin")?;
    let vk_full_bytes = std::fs::read("vk-full.bin").context("read vk-full.bin")?;
    let vk_digest_bytes = std::fs::read("vk.bin").context("read vk.bin")?;

    let proof: DTReduceProof<RootSC> =
        bincode::deserialize(&proof_bytes).context("deserialize proof")?;
    let vk: DTVerifyingKey = bincode::deserialize(&vk_full_bytes).context("deserialize full vk")?;
    let expected_digest: [u32; zkdtvm_stark_verifier::DIGEST_SIZE] =
        bincode::deserialize(&vk_digest_bytes).context("deserialize digest vk")?;
    let full_digest = vk.hash_babybear();
    let expected_digest = expected_digest.map(SCField::from_canonical_u32);

    println!(
        "proof bytes: {}, full vk bytes: {}, digest bytes: {}",
        proof_bytes.len(),
        vk_full_bytes.len(),
        vk_digest_bytes.len()
    );
    println!(
        "proof root vk chips: {}, proof chip_ordering: {}, opened chips: {}, dimensions: {}",
        proof.vk.chip_information.len(),
        proof.proof.chip_ordering.len(),
        proof.proof.opened_values.chips.len(),
        proof.proof.dimensions.len()
    );
    println!(
        "program vk chips: {}, program vk ordering: {}",
        vk.vk.chip_information.len(),
        vk.vk.chip_ordering.len()
    );
    println!(
        "full vk digest matches vk.bin: {}",
        full_digest == expected_digest
    );
    println!(
        "full vk digest u32: {:?}",
        full_digest.map(|value| value.as_canonical_u32())
    );
    println!(
        "vk.bin digest u32: {:?}",
        expected_digest.map(|value| value.as_canonical_u32())
    );

    let prover = DTProver::<SCCpuProverComponents>::new();
    let native_backend = prover.native_backend().context("native backend")?;
    println!(
        "is native proof: {}",
        native_backend
            .is_native_proof(&proof)
            .context("native proof check")?
    );

    Ok(())
}
