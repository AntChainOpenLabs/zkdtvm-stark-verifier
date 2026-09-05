use std::{env, fs, path::PathBuf};

use dt_prover::{components::SCCpuProverComponents, DTProver};
use zkdtvm_stark_verifier::{
    build_l4_verifier_artifact_bytes, build_l4_verifier_artifact_bytes_for_vk,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let output = args.first().map(PathBuf::from).ok_or(
        "usage: build_l4_verifier_artifact <output-path> [<application-elf> <vk-output-path>]",
    )?;
    let bytes = match args.len() {
        1 => build_l4_verifier_artifact_bytes()?,
        3 => {
            let elf = fs::read(&args[1])?;
            let prover = DTProver::<SCCpuProverComponents>::new();
            let (_, _, _, vk) = prover.setup(&elf);
            let bytes = build_l4_verifier_artifact_bytes_for_vk(&vk)?;
            fs::write(&args[2], bincode::serialize(&vk)?)?;
            bytes
        }
        _ => return Err("expected output, or output + ELF + VK output".into()),
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, &bytes)?;
    println!("wrote {} bytes to {}", bytes.len(), output.display());
    Ok(())
}
