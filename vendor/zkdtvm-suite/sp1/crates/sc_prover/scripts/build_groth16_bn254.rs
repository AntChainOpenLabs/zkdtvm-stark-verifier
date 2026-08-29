#[cfg(not(feature = "ext5"))]
use std::path::PathBuf;

#[cfg(not(feature = "ext5"))]
use clap::Parser;
#[cfg(not(feature = "ext5"))]
use dt_core_machine::utils::setup_logger;
#[cfg(not(feature = "ext5"))]
use dt_prover::build::build_groth16_bn254_artifacts_with_dummy;

#[cfg(not(feature = "ext5"))]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    build_dir: PathBuf,
}

#[cfg(not(feature = "ext5"))]
pub fn main() {
    setup_logger();
    let args = Args::parse();
    build_groth16_bn254_artifacts_with_dummy(args.build_dir);
}

#[cfg(feature = "ext5")]
pub fn main() {
    panic!("build_groth16_bn254 is unavailable under ext5; ext5 stops at the shrink proof stage");
}
