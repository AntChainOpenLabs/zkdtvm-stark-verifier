use std::{env, path::PathBuf};

use anyhow::{bail, Context, Result};
use dt_core_machine::reduce::DTReduceProof;
use dt_prover::RootSC;
use dt_sdk::{
    include_elf, utils, DTProof, DTStdin, HashableKey, Prover, ProverClient, RecursionBackend,
};
use p3_field::PrimeField32;
use sha2::{Digest, Sha256};

const SUITE_COMMIT: &str = "82a57cadf6921e4fb45181d98f1a5af0148ab491";
const ELF: &[u8] = include_elf!("fibonacci-program");

fn main() -> Result<()> {
    utils::setup_logger();

    let out_dir = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;

    let n = env::var("FIBONACCI_N")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(500);

    let backend = match env::var("DT_RECURSION_BACKEND")
        .unwrap_or_else(|_| "native".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "native" => RecursionBackend::Native,
        "dsl" => RecursionBackend::Dsl,
        other => bail!("DT_RECURSION_BACKEND must be 'native' or 'dsl', got {other:?}"),
    };

    let mut stdin = DTStdin::new();
    stdin.write(&n);

    let client = ProverClient::builder().cpu().build();
    let (pk, vk) = client.setup(ELF);
    let mut proof_bundle = client
        .prove(&pk, &stdin)
        .compressed()
        .recursion_backend(backend)
        .run()
        .context("generate compressed proof")?;

    let input = proof_bundle.public_values.read::<u32>();
    let a = proof_bundle.public_values.read::<u32>();
    let b = proof_bundle.public_values.read::<u32>();
    println!("public values: n={input}, a={a}, b={b}");

    client.verify(&proof_bundle, &vk).context("sdk verify compressed proof")?;

    let reduce_proof = match &proof_bundle.proof {
        DTProof::Compressed(proof) => proof.as_ref(),
        other => bail!("expected compressed RootSC proof, got {other}"),
    };

    let proof_bytes = bincode::serialize(reduce_proof).context("serialize reduce proof")?;
    let roundtrip: DTReduceProof<RootSC> =
        bincode::deserialize(&proof_bytes).context("deserialize roundtrip reduce proof")?;
    client
        .inner()
        .verify_compressed(&roundtrip, &vk)
        .context("verify roundtrip reduce proof")?;

    std::fs::write(out_dir.join("proof.bin"), proof_bytes).context("write proof.bin")?;

    let vk_digest = vk.hash_babybear().map(|field| field.as_canonical_u32());
    std::fs::write(
        out_dir.join("vk.bin"),
        bincode::serialize(&vk_digest).context("serialize vk digest")?,
    )
    .context("write vk.bin")?;

    std::fs::write(
        out_dir.join("vk-full.bin"),
        bincode::serialize(&vk).context("serialize full verifying key")?,
    )
    .context("write vk-full.bin")?;

    proof_bundle
        .save(out_dir.join("proof-with-public-values.bin"))
        .context("write SDK proof bundle")?;

    let elf_sha256 = format!("{:x}", Sha256::digest(ELF));
    let metadata = serde_json::json!({
        "suite_commit": SUITE_COMMIT,
        "program": "fibonacci-program",
        "elf_sha256": elf_sha256,
        "fibonacci_n": n,
        "field": "koalabear",
        "extension_degree": 5,
        "proof_config": "RootSC (SHA256)",
    });
    std::fs::write(
        out_dir.join("fixture-metadata.json"),
        serde_json::to_vec_pretty(&metadata).context("serialize fixture metadata")?,
    )
    .context("write fixture-metadata.json")?;

    println!("wrote verifier fixtures to {}", out_dir.display());
    Ok(())
}
