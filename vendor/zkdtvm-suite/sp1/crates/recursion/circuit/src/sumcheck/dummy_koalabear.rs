use dt_stark::sumcheck::types::UniPolyEvals;
use p3_field::AbstractField;

use crate::fri::PolynomialBatchShape;
use p3_commit::Mmcs;
use p3_field::extension::BinomialExtensionField;
#[cfg(feature = "ext5")]
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_fri::{BatchOpening, CommitPhaseProofStep, QueryProof};
use p3_koala_bear::KoalaBear;
use pcs::{
    basefold::{
        basefold_pcs::{BasefoldInputProof, BasefoldProof},
        sumcheck::SumcheckInstanceProof,
        StackingReductionProof,
    },
    utils::unipoly::UniPoly,
};

use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::{ChallengeMmcs, ValMmcs};

type F = KoalaBear;
#[cfg(not(feature = "ext5"))]
type EF = BinomialExtensionField<F, 4>;
#[cfg(feature = "ext5")]
type EF = QuinticTrinomialExtensionField<F>;

/// Basefold proof type matching the active SC (KoalaBear) challenge extension
/// `EF` (binomial quartic by default, quintic under `ext5`).
type InnerBasefoldProof = BasefoldProof<EF, ChallengeMmcs, F, BasefoldInputProof<F, ValMmcs>>;

pub fn dummy_unipoly(degree: usize) -> UniPolyEvals<EF> {
    UniPolyEvals { evals: vec![EF::zero(); degree + 1] }
}

pub fn sc_dummy_pcs_proof(
    batch_shapes: &[PolynomialBatchShape],
    log_blowup: usize,
    fri_queries: usize,
    log_final_poly_len: usize,
) -> InnerBasefoldProof {
    let (log_heights, widths): (Vec<Vec<usize>>, Vec<Vec<usize>>) = batch_shapes
        .iter()
        .map(|single_batch| {
            single_batch.shapes.iter().map(|shape| (shape.log_degree, shape.width)).unzip()
        })
        .unzip();
    generate_dummy_proof(&log_heights, &widths, fri_queries, log_blowup, log_final_poly_len)
}

fn generate_dummy_proof(
    log_heights: &Vec<Vec<usize>>,
    widths: &Vec<Vec<usize>>,
    fri_num_queries: usize,
    fri_blowup_bits: usize,
    log_final_poly_len: usize,
) -> InnerBasefoldProof {
    let num_batches = log_heights.len();
    let num_vars = log_heights.iter().flat_map(|batch| batch.iter()).copied().max().unwrap_or(0);
    let k = log_final_poly_len.min(num_vars);
    let num_iopp_rounds = if k == 0 { num_vars + 1 } else { num_vars - k };

    let uni_polys: Vec<UniPoly<EF>> =
        (0..num_vars).map(|_| UniPoly::from_coeff(vec![EF::zero(); 3])).collect();

    let zero_commitment = <ValMmcs as Mmcs<F>>::Commitment::from([F::zero(); 8]);
    let iopp_oracles = vec![zero_commitment; num_iopp_rounds];

    let iopp_queries: Vec<QueryProof<EF, ChallengeMmcs>> = (0..fri_num_queries)
        .map(|_| {
            let steps: Vec<CommitPhaseProofStep<EF, ChallengeMmcs>> = (0..num_iopp_rounds)
                .map(|step_idx| {
                    let round = num_vars.saturating_sub(step_idx);
                    let merkle_path_len = round + fri_blowup_bits - 1;
                    CommitPhaseProofStep {
                        sibling_value: EF::zero(),
                        opened_values: Vec::new(),
                        opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                    }
                })
                .collect();
            QueryProof { commit_phase_openings: steps }
        })
        .collect();

    let query_openings: Vec<Vec<BatchOpening<F, ValMmcs>>> = (0..fri_num_queries)
        .map(|_| {
            (0..num_batches)
                .map(|batch_idx| {
                    let opened_values: Vec<Vec<F>> =
                        widths[batch_idx].iter().map(|&width| vec![F::zero(); width]).collect();
                    let max_log_height_in_batch =
                        log_heights[batch_idx].iter().copied().max().unwrap_or(0);
                    let merkle_path_len = max_log_height_in_batch + fri_blowup_bits;
                    BatchOpening {
                        opened_values,
                        opening_proof: vec![[F::zero(); 8]; merkle_path_len],
                    }
                })
                .collect()
        })
        .collect();

    let grinding_batching_witness: Vec<F> = vec![F::zero(); 2];
    let grinding_query_witness: Vec<F> = vec![F::zero(); 2];
    let final_poly_len = if k == 0 { 0 } else { 1usize << k };
    let final_poly = vec![EF::zero(); final_poly_len];

    BasefoldProof {
        stack_log_height: Some(num_vars),
        sumcheck_transcript: SumcheckInstanceProof::new(uni_polys),
        iopp_oracles,
        ood_values: Vec::new(),
        iopp_queries,
        round_iopp: None,
        // [D6-N6] wrap per-query openings into BasefoldInputProof; pruned=None for dummy.
        query_openings: BasefoldInputProof::from_per_query(query_openings),
        grinding_batching_witness,
        grinding_query_witness,
        final_poly,
        iopp_pruned: None,
        stacking_reduction: Some(StackingReductionProof {
            sumcheck: SumcheckInstanceProof::new(
                (0..num_vars).map(|_| UniPoly::from_evals(&[EF::zero(); 3])).collect(),
            ),
        }),
    }
}
