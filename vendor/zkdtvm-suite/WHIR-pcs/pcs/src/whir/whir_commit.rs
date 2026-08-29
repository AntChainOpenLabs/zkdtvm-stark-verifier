use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::whir::profile;
use crate::whir::whir_helpers::{
    build_stacked_evaluations, with_thread_local_evals_dft, StackedBatchLayout,
};
use crate::whir::whir_types::{WhirPcs, WhirPcsProverData};

impl<F, InputMmcs, FriMmcs, EF, Challenger> WhirPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField + 'static,
    InputMmcs: Mmcs<F> + Send + Sync,
    InputMmcs::ProverData<RowMajorMatrix<F>>: Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    /// Commit multiple compressed matrices with different dimensions.
    ///
    /// Each CompressedMatrix stores only non-padding rows; padding rows are
    /// decompressed before DFT encoding.
    ///
    /// Returns: (merkle_root, merkle_tree)
    #[tracing::instrument(skip_all, level = "debug", name = "WHIR::commit")]
    pub fn commit_impl(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
    ) -> (
        InputMmcs::Commitment,
        InputMmcs::ProverData<RowMajorMatrix<F>>,
    ) {
        assert!(
            self.config.fri.log_blowup > 0,
            "log_blowup must be greater than 0"
        );

        let repeat_times = 1 << self.config.fri.log_blowup;
        let codewords: Vec<RowMajorMatrix<F>> = profile::time("commit.dft_ms", || {
            let _span = tracing::debug_span!("decompress_and_dft").entered();
            evaluations
                .into_par_iter()
                .map(|compressed| {
                    with_thread_local_evals_dft(|dft| {
                        dft.dft_batch_by_evals_skip(
                            compressed.decompress_and_repeat(repeat_times),
                            self.config.fri.log_blowup,
                        )
                        .to_row_major_matrix()
                    })
                })
                .collect()
        });

        let commitment = profile::time("commit.leaf_hash_and_tree_ms", || {
            let _span = tracing::debug_span!("merkle_tree_commit").entered();
            let mut profiler = profile::MmcsCommitProfiler::new();
            self.mmcs.commit_with_observer(codewords, &mut profiler)
        });

        commitment
    }
    pub fn commit_stacked_impl(
        &self,
        evaluations: Vec<&CompressedMatrix<F>>,
        stack_log_height: usize,
        cache_stacked_matrix: bool,
    ) -> (InputMmcs::Commitment, WhirPcsProverData<F, InputMmcs>) {
        assert!(
            self.config.fri.log_blowup > 0,
            "log_blowup must be greater than 0"
        );

        let layout = profile::time("commit.stacked_layout_ms", || {
            StackedBatchLayout::from_matrices(&evaluations, stack_log_height, EF::D).unwrap_or_else(
                |()| {
                    let dimensions = evaluations
                        .iter()
                        .map(|matrix| Dimensions {
                            width: matrix.width(),
                            height: matrix.height(),
                        })
                        .collect::<Vec<_>>();
                    panic!(
                        "invalid stacking layout: stack_log_height={}, dimensions={:?}",
                        stack_log_height, dimensions
                    );
                },
            )
        });
        let stacked_evaluations = profile::time("commit.stacked_build_ms", || {
            build_stacked_evaluations(&evaluations, &layout)
        });

        let repeat_times = 1 << self.config.fri.log_blowup;
        let one_copy = stacked_evaluations.values.len();
        let (mut repeated_values, cached_evaluations) = if cache_stacked_matrix {
            let mut values = Vec::with_capacity(one_copy * repeat_times);
            values.extend_from_slice(&stacked_evaluations.values);
            (values, Some(stacked_evaluations))
        } else {
            let mut values = stacked_evaluations.values;
            values.reserve(one_copy * (repeat_times - 1));
            (values, None)
        };
        profile::time("commit.repeat_ms", || {
            for _ in 1..repeat_times {
                repeated_values.extend_from_within(0..one_copy);
            }
        });

        let codeword = profile::time("commit.dft_ms", || {
            with_thread_local_evals_dft(|dft| {
                dft.dft_batch_by_evals_skip(
                    RowMajorMatrix::new(repeated_values, layout.width),
                    self.config.fri.log_blowup,
                )
                .to_row_major_matrix()
            })
        });

        let (commitment, mmcs_prover_data) = profile::time("commit.leaf_hash_and_tree_ms", || {
            let mut profiler = profile::MmcsCommitProfiler::new();
            self.mmcs
                .commit_with_observer(vec![codeword], &mut profiler)
        });
        (
            commitment,
            WhirPcsProverData::stacked(mmcs_prover_data, layout, cached_evaluations),
        )
    }
}
