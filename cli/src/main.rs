use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use zkdtvm_stark_verifier::{verify_compressed, DTReduceProof, DTVerifyingKey, InnerSC};

#[derive(Parser)]
#[command(name = "zkdtvm-stark-verifier", about = "Verify zkdtvm STARK compressed proofs")]
struct Cli {
    /// Path to the serialized proof file (bincode)
    #[arg(long)]
    proof: PathBuf,

    /// Path to the serialized verifying key file (bincode)
    #[arg(long)]
    vk: PathBuf,

    /// Optional path to the serialized message file (bincode)
    #[arg(long)]
    message: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    tracing::info!("Loading proof from {}...", cli.proof.display());
    let proof_bytes = fs::read(&cli.proof)?;
    let proof: DTReduceProof<InnerSC> = bincode::deserialize(&proof_bytes)?;
    tracing::info!("Proof loaded ({} bytes)", proof_bytes.len());

    tracing::info!("Loading verifying key from {}...", cli.vk.display());
    let vk_bytes = fs::read(&cli.vk)?;
    let vk: DTVerifyingKey = bincode::deserialize(&vk_bytes)?;
    tracing::info!("Verifying key loaded ({} bytes)", vk_bytes.len());

    tracing::info!("Verifying compressed proof...");
    verify_compressed(&proof, &vk).map_err(|e| anyhow::anyhow!("Verification failed: {:?}", e))?;

    tracing::info!("Proof verified successfully!");

    if let Some(msg_path) = cli.message {
        let msg_bytes = fs::read(&msg_path)?;
        let message: String = bincode::deserialize(&msg_bytes)?;
        println!("{message}");
    }

    Ok(())
}
