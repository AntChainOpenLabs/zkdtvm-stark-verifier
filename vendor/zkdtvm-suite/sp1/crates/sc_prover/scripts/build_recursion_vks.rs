use std::path::PathBuf;

use clap::Parser;
use dt_core_machine::utils::setup_logger;
use dt_prover::{
    components::SCCpuProverComponents, shapes::build_vk_map_to_file, REDUCE_BATCH_SIZE,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long)]
    build_dir: PathBuf,
    #[clap(short, long, default_value_t = false)]
    dummy: bool,
    #[clap(short, long)]
    reduce_batch_size: Option<usize>,
    #[clap(short, long, default_value_t = 1)]
    num_compiler_workers: usize,
    #[clap(short = 'w', long, default_value_t = 1)]
    num_setup_workers: usize,
    #[clap(short, long)]
    start: Option<usize>,
    #[clap(short, long)]
    end: Option<usize>,
}

fn main() {
    setup_logger();
    let args = Args::parse();

    let reduce_batch_size = args.reduce_batch_size.unwrap_or(REDUCE_BATCH_SIZE);
    let build_dir = args.build_dir;

    build_vk_map_to_file::<SCCpuProverComponents>(
        build_dir,
        reduce_batch_size,
        args.dummy,
        args.num_compiler_workers,
        args.num_setup_workers,
        args.start,
        args.end,
    )
    .unwrap();
}
