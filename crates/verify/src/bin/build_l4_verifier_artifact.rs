use std::{env, fs, path::PathBuf};

use zkdtvm_stark_verifier::build_l4_verifier_artifact_bytes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_l4_verifier_artifact <output-path>")?;
    let bytes = build_l4_verifier_artifact_bytes()?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, &bytes)?;
    println!("wrote {} bytes to {}", bytes.len(), output.display());
    Ok(())
}
