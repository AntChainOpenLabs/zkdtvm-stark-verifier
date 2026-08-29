use std::path::PathBuf;

use clap::Parser;
use dt_core_machine::utils::setup_logger;
use dt_recursion_gnark_ffi::Groth16Bn254Prover;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    build_dir: PathBuf,
}

pub fn main() {
    setup_logger();
    let args = Args::parse();
    Groth16Bn254Prover::build_contracts(args.build_dir);
}
