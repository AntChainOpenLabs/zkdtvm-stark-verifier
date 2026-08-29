use crate::utils::mlpoly::{MultilinearExtension, MultilinearPolynomial};
use crate::whir::mlpcs::{MlCommitOptions, MlPCS};
use crate::whir::whir_types::{WhirConfig, WhirPcs};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, Field};
use p3_field::{AbstractExtensionField, AbstractField};
use p3_fri::FriConfig;
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

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

#[test]
fn test_whir_stacked_open_verify_general() {
    const TALL_LOG_HEIGHT: usize = 20;
    const SHORT_LOG_HEIGHT: usize = 19;
    const MATRIX_WIDTH: usize = 10;
    const LOG_FINAL_POLY_LEN: usize = 6;
    const CROSS_ROUND_LOG_FOLDINGS: [usize; 3] = [5, 5, 4];
    const ROUND_QUERY_COUNTS: [usize; 3] = [90, 60, 45];

    let mut rng_bb = Xoroshiro128Plus::seed_from_u64(11);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng_bb,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 90,
        grinding_bits_query: 0,
        grinding_bits_batching: 0,
        grinding_bits_folding: 0,
        log_final_poly_len: LOG_FINAL_POLY_LEN,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    };

    type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
    let config = WhirConfig::new(fri_config)
        .with_cross_round_log_foldings(CROSS_ROUND_LOG_FOLDINGS.to_vec())
        .with_round_query_counts(ROUND_QUERY_COUNTS.to_vec());
    let pcs = Pcs::from_config(val_mmcs, config);
    assert_eq!(
        pcs.cross_round_log_foldings(),
        CROSS_ROUND_LOG_FOLDINGS.to_vec()
    );

    let tall_height = 1usize << TALL_LOG_HEIGHT;
    let short_height = 1usize << SHORT_LOG_HEIGHT;
    let tall_matrix = deterministic_matrix(tall_height, MATRIX_WIDTH, 0x11);
    let short_matrix = deterministic_matrix(short_height, MATRIX_WIDTH, 0x29);
    let opening_point: Vec<EF> = (0..TALL_LOG_HEIGHT).map(|_| rand::random()).collect();

    let tall_values = opened_values_for_base_matrix(&tall_matrix, TALL_LOG_HEIGHT, &opening_point);
    let short_values =
        opened_values_for_base_matrix(&short_matrix, SHORT_LOG_HEIGHT, &opening_point);
    let opened_values = vec![vec![tall_values, short_values]];

    let dims = vec![vec![
        Dimensions {
            height: tall_height,
            width: MATRIX_WIDTH,
        },
        Dimensions {
            height: short_height,
            width: MATRIX_WIDTH,
        },
    ]];
    let compressed_matrices = vec![vec![
        CompressedMatrix::from_full_matrix_no_padding(tall_matrix),
        CompressedMatrix::from_full_matrix_no_padding(short_matrix),
    ]];

    let commit_options = MlCommitOptions::stacking_log_height(TALL_LOG_HEIGHT);
    let (commitment, prover_data) =
        pcs.commit_with_options(compressed_matrices[0].iter().collect(), commit_options);
    assert!(prover_data
        .stacked
        .as_ref()
        .and_then(|stacked| stacked.cached_evaluations.as_ref())
        .is_some());

    let mut challenger = Challenger::new(perm.clone());
    let proof = pcs
        .open(
            compressed_matrices,
            vec![prover_data],
            &opening_point,
            &opened_values,
            &mut challenger,
        )
        .unwrap();
    assert_eq!(proof.stack_log_height, Some(TALL_LOG_HEIGHT));
    assert_eq!(proof.final_poly.len(), 1 << LOG_FINAL_POLY_LEN);
    let round_iopp = proof.round_iopp.as_ref().expect("per-round proof");
    assert_eq!(round_iopp.rounds.len(), ROUND_QUERY_COUNTS.len());
    for (round, &num_queries) in round_iopp.rounds.iter().zip(ROUND_QUERY_COUNTS.iter()) {
        assert_eq!(round.query_proofs.len(), num_queries);
    }
    assert!(round_iopp.pruned.is_none());
    assert!(proof.query_openings.pruned.is_none());
    let proof_size = bincode::serialize(&proof).unwrap().len();
    eprintln!(
        "test_whir_stacked_open_verify_general proof size: {proof_size} bytes ({:.2} KiB)",
        proof_size as f64 / 1024.0
    );

    let mut verifier_challenger = Challenger::new(perm.clone());
    pcs.verify(
        vec![commitment],
        &dims,
        &opening_point,
        &opened_values,
        &proof,
        &mut verifier_challenger,
    )
    .unwrap();
}

#[test]
fn test_whir_stacked_per_round_queries() {
    let mut rng_bb = Xoroshiro128Plus::seed_from_u64(17);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng_bb,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);

    let log_heights = [3, 2];
    let matrices = [
        RowMajorMatrix::<F>::new((0..8).map(|_| rand::random()).collect(), 1),
        RowMajorMatrix::<F>::new((0..16).map(|_| rand::random()).collect(), EXT_DEGREE),
    ];
    let compressed_matrices = vec![matrices
        .iter()
        .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
        .collect::<Vec<_>>()];
    let opening_point: Vec<EF> = (0..3).map(|_| rand::random()).collect();

    let tall_value = {
        let poly = MultilinearPolynomial::from_evals(get_col(&matrices[0], 0));
        poly.evaluate_mix(&opening_point[..log_heights[0]])
    };
    let short_ext_matrix = {
        let mut ef_values = Vec::with_capacity(4);
        for row in 0..4 {
            let idx = row * EXT_DEGREE;
            ef_values.push(EF::from_base_slice(
                &matrices[1].values[idx..idx + EXT_DEGREE],
            ));
        }
        RowMajorMatrix::<EF>::new(ef_values, 1)
    };
    let short_value = {
        let poly = MultilinearPolynomial::from_evals(get_ef_col(&short_ext_matrix, 0));
        poly.evaluate_mix(&opening_point[..log_heights[1]])
    };
    let opened_values = vec![vec![vec![tall_value], vec![short_value]]];
    let dims = vec![vec![
        Dimensions {
            height: 8,
            width: 1,
        },
        Dimensions {
            height: 4,
            width: EXT_DEGREE,
        },
    ]];

    type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
    for cache_stacked_matrix in [true, false] {
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 3,
            grinding_bits_query: 0,
            grinding_bits_batching: 0,
            grinding_bits_folding: 0,
            log_final_poly_len: 2,
            cross_round_log_foldings: Vec::new(),
            num_committed_groups: None,
            mmcs: challenge_mmcs,
        };
        let config = WhirConfig::new(fri_config).with_round_query_counts(vec![3, 2]);
        let pcs = Pcs::from_config(val_mmcs.clone(), config);

        let (commitment, prover_data) = pcs.commit_with_options(
            compressed_matrices[0].iter().collect(),
            MlCommitOptions::stacking_log_height(4).with_stacked_matrix_cache(cache_stacked_matrix),
        );
        assert_eq!(
            prover_data
                .stacked
                .as_ref()
                .and_then(|stacked| stacked.cached_evaluations.as_ref())
                .is_some(),
            cache_stacked_matrix
        );
        assert!(prover_data.stacked.as_ref().is_some());

        let mut challenger = Challenger::new(perm.clone());
        let proof = pcs
            .open(
                compressed_matrices.clone(),
                vec![prover_data],
                &opening_point,
                &opened_values,
                &mut challenger,
            )
            .unwrap();
        let round_iopp = proof.round_iopp.as_ref().expect("per-round proof");
        assert!(proof.iopp_queries.is_empty());
        assert_eq!(proof.ood_values.len(), 1);
        assert!(round_iopp.pruned.is_none());
        assert_eq!(round_iopp.rounds.len(), 2);
        assert_eq!(round_iopp.rounds[0].query_proofs.len(), 3);
        assert_eq!(round_iopp.rounds[1].query_proofs.len(), 2);

        let mut verifier_challenger = Challenger::new(perm.clone());
        pcs.verify(
            vec![commitment],
            &dims,
            &opening_point,
            &opened_values,
            &proof,
            &mut verifier_challenger,
        )
        .unwrap();

        let mut bad_proof = proof.clone();
        bad_proof.ood_values[0] += EF::one();
        let mut bad_challenger = Challenger::new(perm.clone());
        assert!(pcs
            .verify(
                vec![commitment],
                &dims,
                &opening_point,
                &opened_values,
                &bad_proof,
                &mut bad_challenger,
            )
            .is_err());
    }
}

#[test]
fn test_whir_stacked_cross_round_rate_modes() {
    run_whir_stacked_cross_round(true);
    run_whir_stacked_cross_round(false);
}

fn run_whir_stacked_cross_round(reduced_rate: bool) {
    let mut rng_bb = Xoroshiro128Plus::seed_from_u64(29);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng_bb,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 3,
        grinding_bits_query: 0,
        grinding_bits_batching: 0,
        grinding_bits_folding: 0,
        log_final_poly_len: 1,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    };

    type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
    let config = WhirConfig::new(fri_config)
        .with_reduced_rate(reduced_rate)
        .with_cross_round_log_foldings(vec![2, 2])
        .with_round_query_counts(vec![3, 2]);
    let pcs = Pcs::from_config(val_mmcs, config);

    let log_heights = [4, 3];
    let matrices = [
        RowMajorMatrix::<F>::new((0..16).map(|_| rand::random()).collect(), 1),
        RowMajorMatrix::<F>::new((0..32).map(|_| rand::random()).collect(), EXT_DEGREE),
    ];
    let compressed_matrices = vec![matrices
        .iter()
        .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
        .collect::<Vec<_>>()];
    let opening_point: Vec<EF> = (0..4).map(|_| rand::random()).collect();

    let tall_value = {
        let poly = MultilinearPolynomial::from_evals(get_col(&matrices[0], 0));
        poly.evaluate_mix(&opening_point[..log_heights[0]])
    };
    let short_ext_matrix = {
        let mut ef_values = Vec::with_capacity(8);
        for row in 0..8 {
            let idx = row * EXT_DEGREE;
            ef_values.push(EF::from_base_slice(
                &matrices[1].values[idx..idx + EXT_DEGREE],
            ));
        }
        RowMajorMatrix::<EF>::new(ef_values, 1)
    };
    let short_value = {
        let poly = MultilinearPolynomial::from_evals(get_ef_col(&short_ext_matrix, 0));
        poly.evaluate_mix(&opening_point[..log_heights[1]])
    };
    let opened_values = vec![vec![vec![tall_value], vec![short_value]]];

    let (commitment, prover_data) = pcs.commit_with_options(
        compressed_matrices[0].iter().collect(),
        MlCommitOptions::stacking_log_height(5),
    );

    let mut challenger = Challenger::new(perm.clone());
    let proof = pcs
        .open(
            compressed_matrices.clone(),
            vec![prover_data],
            &opening_point,
            &opened_values,
            &mut challenger,
        )
        .unwrap();
    let round_iopp = proof.round_iopp.as_ref().expect("per-round proof");
    assert_eq!(round_iopp.rounds.len(), 2);
    assert_eq!(
        round_iopp.rounds[0].query_proofs[0]
            .current_opening
            .opened_values
            .len(),
        4
    );
    assert_eq!(
        round_iopp.rounds[1].query_proofs[0]
            .current_opening
            .opening_proof
            .len(),
        if reduced_rate { 3 } else { 2 }
    );

    let dims = vec![vec![
        Dimensions {
            height: 16,
            width: 1,
        },
        Dimensions {
            height: 8,
            width: EXT_DEGREE,
        },
    ]];
    let mut verifier_challenger = Challenger::new(perm.clone());
    pcs.verify(
        vec![commitment],
        &dims,
        &opening_point,
        &opened_values,
        &proof,
        &mut verifier_challenger,
    )
    .unwrap();
}

#[test]
fn test_whir_stacked_per_round_queries_with_path_pruning() {
    let mut rng_bb = Xoroshiro128Plus::seed_from_u64(23);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng_bb,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 3,
        grinding_bits_query: 0,
        grinding_bits_batching: 0,
        grinding_bits_folding: 0,
        log_final_poly_len: 2,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    };

    type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
    let config = WhirConfig::new(fri_config)
        .with_round_query_counts(vec![3, 2])
        .with_path_pruning(true);
    let pcs = Pcs::from_config(val_mmcs, config);

    let log_heights = [3, 2];
    let matrices = [
        RowMajorMatrix::<F>::new((0..8).map(|_| rand::random()).collect(), 1),
        RowMajorMatrix::<F>::new((0..16).map(|_| rand::random()).collect(), EXT_DEGREE),
    ];
    let compressed_matrices = vec![matrices
        .iter()
        .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
        .collect::<Vec<_>>()];
    let opening_point: Vec<EF> = (0..3).map(|_| rand::random()).collect();

    let tall_value = {
        let poly = MultilinearPolynomial::from_evals(get_col(&matrices[0], 0));
        poly.evaluate_mix(&opening_point[..log_heights[0]])
    };
    let short_ext_matrix = {
        let mut ef_values = Vec::with_capacity(4);
        for row in 0..4 {
            let idx = row * EXT_DEGREE;
            ef_values.push(EF::from_base_slice(
                &matrices[1].values[idx..idx + EXT_DEGREE],
            ));
        }
        RowMajorMatrix::<EF>::new(ef_values, 1)
    };
    let short_value = {
        let poly = MultilinearPolynomial::from_evals(get_ef_col(&short_ext_matrix, 0));
        poly.evaluate_mix(&opening_point[..log_heights[1]])
    };
    let opened_values = vec![vec![vec![tall_value], vec![short_value]]];

    let (commitment, prover_data) = pcs.commit_with_options(
        compressed_matrices[0].iter().collect(),
        MlCommitOptions::stacking_log_height(4),
    );

    let mut challenger = Challenger::new(perm.clone());
    let proof = pcs
        .open(
            compressed_matrices.clone(),
            vec![prover_data],
            &opening_point,
            &opened_values,
            &mut challenger,
        )
        .unwrap();
    let round_iopp = proof.round_iopp.as_ref().expect("per-round proof");
    let pruned_iopp = round_iopp.pruned.as_ref().expect("pruned round proof");
    assert!(proof.iopp_queries.is_empty());
    assert!(proof.iopp_pruned.is_none());
    assert!(round_iopp.rounds.is_empty());
    assert_eq!(pruned_iopp.rounds.len(), 2);
    assert!(proof.query_openings.per_query.is_empty());
    assert!(proof.query_openings.pruned.is_some());

    let dims = vec![vec![
        Dimensions {
            height: 8,
            width: 1,
        },
        Dimensions {
            height: 4,
            width: EXT_DEGREE,
        },
    ]];
    let mut verifier_challenger = Challenger::new(perm.clone());
    pcs.verify(
        vec![commitment],
        &dims,
        &opening_point,
        &opened_values,
        &proof,
        &mut verifier_challenger,
    )
    .unwrap();

    let mut bad_iopp = proof.clone();
    bad_iopp
        .round_iopp
        .as_mut()
        .unwrap()
        .pruned
        .as_mut()
        .unwrap()
        .rounds[0]
        .opened_rows[0][0][0] += EF::one();
    let mut bad_iopp_challenger = Challenger::new(perm.clone());
    assert!(pcs
        .verify(
            vec![commitment],
            &dims,
            &opening_point,
            &opened_values,
            &bad_iopp,
            &mut bad_iopp_challenger,
        )
        .is_err());

    let mut bad_input = proof.clone();
    bad_input
        .query_openings
        .pruned
        .as_mut()
        .unwrap()
        .round_opened_values[0][0][0][0] += F::one();
    let mut bad_input_challenger = Challenger::new(perm.clone());
    assert!(pcs
        .verify(
            vec![commitment],
            &dims,
            &opening_point,
            &opened_values,
            &bad_input,
            &mut bad_input_challenger,
        )
        .is_err());
}

fn get_col(mat: &RowMajorMatrix<F>, col: usize) -> Vec<F> {
    (0..mat.height()).map(|row| mat.get(row, col)).collect()
}

fn get_ef_col(mat: &RowMajorMatrix<EF>, col: usize) -> Vec<EF> {
    (0..mat.height()).map(|row| mat.get(row, col)).collect()
}

fn deterministic_matrix(height: usize, width: usize, salt: usize) -> RowMajorMatrix<F> {
    RowMajorMatrix::new(
        (0..height * width)
            .map(|idx| F::from_canonical_usize(idx.wrapping_mul(17).wrapping_add(salt)))
            .collect(),
        width,
    )
}

fn opened_values_for_base_matrix(
    matrix: &RowMajorMatrix<F>,
    log_height: usize,
    opening_point: &[EF],
) -> Vec<EF> {
    (0..matrix.width())
        .map(|col| {
            let poly = MultilinearPolynomial::from_evals(get_col(matrix, col));
            poly.evaluate_mix(&opening_point[..log_height])
        })
        .collect()
}

/// Verify: T = Σ_{x ∈ {0,1}^L} Σ_c F_c(x) · Q_c(x) where T = Σ_i λ^i · claim_i
#[test]
fn test_stacking_reduction_identity() {
    use crate::whir::whir_helpers::{
        build_q_matrix_for_batch, reduction_target_for_batch, StackedBatchLayout,
    };

    let mut rng = Xoroshiro128Plus::seed_from_u64(42);

    // Three matrices: A (2^4, width 2), B (2^3, width 1), C (2^2, width 1)
    // Stack into L = 4
    let l: usize = 4;
    let stack_height = 1usize << l;
    let a_height = 1usize << 4;
    let b_height = 1usize << 3;
    let c_height = 1usize << 2;

    let a_data: Vec<F> = (0..a_height * 2)
        .map(|_| F::from_canonical_u32(rng.next_u32() % 97))
        .collect();
    let b_data: Vec<F> = (0..b_height * 1)
        .map(|_| F::from_canonical_u32(rng.next_u32() % 97))
        .collect();
    let c_data: Vec<F> = (0..c_height * 1)
        .map(|_| F::from_canonical_u32(rng.next_u32() % 97))
        .collect();

    let a_rmm = RowMajorMatrix::new(a_data.clone(), 2);
    let b_rmm = RowMajorMatrix::new(b_data.clone(), 1);
    let c_rmm = RowMajorMatrix::new(c_data.clone(), 1);

    let a_mat = CompressedMatrix::from_full_matrix_no_padding(a_rmm.clone());
    let b_mat = CompressedMatrix::from_full_matrix_no_padding(b_rmm.clone());
    let c_mat = CompressedMatrix::from_full_matrix_no_padding(c_rmm.clone());

    let dimensions = vec![
        Dimensions {
            width: 2,
            height: a_height,
        },
        Dimensions {
            width: 1,
            height: b_height,
        },
        Dimensions {
            width: 1,
            height: c_height,
        },
    ];

    let layout = StackedBatchLayout::from_dimensions(&dimensions, l, 1).unwrap();

    // Random opening point (length L)
    let full_opening_point: Vec<EF> = (0..l)
        .map(|_| EF::from_base_fn(|_| F::from_canonical_u32(rng.next_u32() % 97)))
        .collect();

    // Compute opened_values: f_i(z_prefix_i) for each matrix column
    let opened_a: Vec<EF> = opened_values_for_base_matrix(&a_rmm, 4, &full_opening_point);
    let opened_b: Vec<EF> = opened_values_for_base_matrix(&b_rmm, 3, &full_opening_point);
    let opened_c: Vec<EF> = opened_values_for_base_matrix(&c_rmm, 2, &full_opening_point);
    let opened_values = vec![opened_a, opened_b, opened_c];

    // Sample lambda
    let lambda = EF::from_base_fn(|_| F::from_canonical_u32(rng.next_u32() % 97));

    // Compute T
    let (t, _consumed, _next) = reduction_target_for_batch::<EF, F>(
        &layout,
        &dimensions,
        &opened_values,
        lambda,
        EF::one(),
        false,
    );

    // Build F_c eval tables (stacked matrix)
    let matrices: Vec<&CompressedMatrix<F>> = vec![&a_mat, &b_mat, &c_mat];
    let stacked = crate::whir::whir_helpers::build_stacked_evaluations(&matrices, &layout);

    // Build Q_c eval matrix
    let (q_matrix, _consumed, _next) =
        build_q_matrix_for_batch::<EF, F>(&layout, &full_opening_point, lambda, EF::one(), false);

    // Check identity: T = Σ_x Σ_c F_c(x) · Q_c(x)
    let mut sum = EF::zero();
    for row in 0..stack_height {
        let f_row = stacked.row_slice(row);
        let q_row = q_matrix.row_slice(row);
        for col in 0..layout.width {
            sum += EF::from_base(f_row[col]) * q_row[col];
        }
    }

    assert_eq!(sum, t, "stacking reduction identity failed: T != Σ F·Q");
}

/// Soundness regression: forging opened_values within a shared stacked column
/// must be caught by the reduction sumcheck.
///
/// Setup: A (2^4, w=1) and B (2^3, w=1) stacked into L=4.
/// B gets selector_bits=1. If two opened_values (from A at full height and
/// B at half height) share a stacked column, an attacker who knows the old
/// selector_eq(z) can forge the B claim while preserving the old combined
/// claim. The new λ-reduction must detect this.
#[test]
fn test_stacking_reduction_soundness() {
    let mut rng_bb = Xoroshiro128Plus::seed_from_u64(77);
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng_bb,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 3,
        grinding_bits_query: 0,
        grinding_bits_batching: 0,
        grinding_bits_folding: 0,
        log_final_poly_len: 2,
        cross_round_log_foldings: Vec::new(),
        num_committed_groups: None,
        mmcs: challenge_mmcs,
    };

    type Pcs = WhirPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
    let config = WhirConfig::new(fri_config).with_round_query_counts(vec![3, 2]);
    let pcs = Pcs::from_config(val_mmcs, config);

    // A: height 2^4, width 1; B: height 2^3, width 1
    let a_matrix = RowMajorMatrix::<F>::new((0..16).map(|_| rand::random()).collect(), 1);
    let b_matrix = RowMajorMatrix::<F>::new((0..8).map(|_| rand::random()).collect(), 1);
    let compressed_matrices = vec![vec![
        CompressedMatrix::from_full_matrix_no_padding(a_matrix.clone()),
        CompressedMatrix::from_full_matrix_no_padding(b_matrix.clone()),
    ]];
    let opening_point: Vec<EF> = (0..4).map(|_| rand::random()).collect();

    let a_value = {
        let poly = MultilinearPolynomial::from_evals(get_col(&a_matrix, 0));
        poly.evaluate_mix(&opening_point[..4])
    };
    let b_value = {
        let poly = MultilinearPolynomial::from_evals(get_col(&b_matrix, 0));
        poly.evaluate_mix(&opening_point[..3])
    };
    let opened_values = vec![vec![vec![a_value], vec![b_value]]];

    let dims = vec![vec![
        Dimensions {
            width: 1,
            height: 16,
        },
        Dimensions {
            width: 1,
            height: 8,
        },
    ]];

    let (commitment, prover_data) = pcs.commit_with_options(
        compressed_matrices[0].iter().collect(),
        MlCommitOptions::stacking_log_height(4),
    );

    // Generate valid proof
    let mut challenger = Challenger::new(perm.clone());
    let proof = pcs
        .open(
            compressed_matrices.clone(),
            vec![prover_data],
            &opening_point,
            &opened_values,
            &mut challenger,
        )
        .unwrap();

    // Verify valid proof passes
    let mut verifier_challenger = Challenger::new(perm.clone());
    pcs.verify(
        vec![commitment.clone()],
        &dims,
        &opening_point,
        &opened_values,
        &proof,
        &mut verifier_challenger,
    )
    .unwrap();

    // Forge opened_values: modify B's claim while keeping it "consistent"
    // in any naive sense. The λ-reduction should still catch it.
    let forged_b_value = b_value + EF::one();
    let forged_opened_values = vec![vec![vec![a_value], vec![forged_b_value]]];

    let mut forged_challenger = Challenger::new(perm.clone());
    let result = pcs.verify(
        vec![commitment],
        &dims,
        &opening_point,
        &forged_opened_values,
        &proof,
        &mut forged_challenger,
    );
    assert!(
        result.is_err(),
        "forged opened_values should be rejected by stacking reduction"
    );
}
