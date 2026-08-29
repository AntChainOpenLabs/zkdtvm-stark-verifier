use std::hint::black_box;
use std::time::Instant;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use p3_fri::FriConfig;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand_core::SeedableRng;
use rand_xoshiro::Xoroshiro128Plus;
use rayon::prelude::*;
use whir::whir::mlpcs::{MlCommitOptions, MlPCS};
use whir::whir::WhirPcs;

const D: u64 = 7;
const EXT_DEGREE: usize = 4;

type F = BabyBear;
type EF = BinomialExtensionField<F, EXT_DEGREE>;
type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, D>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    FieldMerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn main() {
    let log_height = env_usize("BENCH_LOG_HEIGHT", 20);
    let width = env_usize("BENCH_WIDTH", 128);
    let log_blowup = env_usize("BENCH_LOG_BLOWUP", 1);
    let iters = env_usize("BENCH_ITERS", 1);
    let use_stacking = env_bool("BENCH_STACKING", true);
    let cache_stacked_matrix = env_bool("BENCH_CACHE_STACKED_MATRIX", true);

    let height = 1usize << log_height;
    let total_values = height * width;

    println!(
        "whir commit bench: height=2^{log_height} ({height}), width={width}, values={total_values}, log_blowup={log_blowup}, stacking={use_stacking}, cache_stacked_matrix={cache_stacked_matrix}, iters={iters}"
    );

    let mut rng = Xoroshiro128Plus::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = FriConfig {
        log_blowup,
        num_queries: 1,
        grinding_bits_query: 0,
        grinding_bits_batching: 0,
        grinding_bits_folding: 0,
        log_final_poly_len: 0,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(val_mmcs, fri_config);

    let gen_start = Instant::now();
    let values = (0..total_values)
        .into_par_iter()
        .map(|i| F::from_canonical_u32((i as u32).wrapping_mul(17).wrapping_add(5)))
        .collect::<Vec<_>>();
    let matrix = RowMajorMatrix::new(values, width);
    let compressed = CompressedMatrix::from_full_matrix_no_padding(matrix);
    println!("matrix generation: {:.3?}", gen_start.elapsed());

    for iter in 0..iters {
        let start = Instant::now();
        if use_stacking {
            let options =
                MlCommitOptions::auto_stacking().with_stacked_matrix_cache(cache_stacked_matrix);
            let (commitment, prover_data) = pcs.commit_with_options(vec![&compressed], options);
            black_box(commitment);
            black_box(prover_data);
        } else {
            let (commitment, prover_data) = pcs.commit(vec![&compressed]);
            black_box(commitment);
            black_box(prover_data);
        }
        println!("commit iteration {}: {:.3?}", iter + 1, start.elapsed());
    }
}
