use core::fmt::Debug;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::Dimensions;
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlCommitOptions {
    pub stacking: Option<StackingConfig>,
}

impl MlCommitOptions {
    pub const fn no_stacking() -> Self {
        Self { stacking: None }
    }

    pub const fn auto_stacking() -> Self {
        Self {
            stacking: Some(StackingConfig {
                log_height: None,
                cache_stacked_matrix: true,
            }),
        }
    }

    pub const fn stacking_log_height(log_height: usize) -> Self {
        Self {
            stacking: Some(StackingConfig {
                log_height: Some(log_height),
                cache_stacked_matrix: true,
            }),
        }
    }

    pub fn with_stacked_matrix_cache(mut self, enabled: bool) -> Self {
        if let Some(stacking) = self.stacking.as_mut() {
            stacking.cache_stacked_matrix = enabled;
        }
        self
    }

    pub fn with_stacked_matrix_cache_from_env(self) -> Self {
        let default = self
            .stacking
            .as_ref()
            .is_none_or(|stacking| stacking.cache_stacked_matrix);
        self.with_stacked_matrix_cache(stacked_matrix_cache_from_env(default))
    }
}

/// Optional commit-local stacking configuration.
///
/// `None` means stack to the tallest matrix in the commit batch.
/// `Some(log_height)` means stack to exactly that height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackingConfig {
    pub log_height: Option<usize>,
    pub cache_stacked_matrix: bool,
}

fn stacked_matrix_cache_from_env(default: bool) -> bool {
    std::env::var("WHIR_CACHE_STACKED_MATRIX")
        .or_else(|_| std::env::var("PCS_CACHE_STACKED_MATRIX"))
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => false,
            "1" | "true" | "on" | "yes" => true,
            _ => default,
        })
        .unwrap_or(default)
}

/// Multilinear Polynomial Commitment Scheme (PCS) trait.
///
/// Provides `commit`, `open`, and `verify` for batches of multilinear polynomials
/// represented as compressed matrices (one column per polynomial).
/// `CompressedMatrix` stores only non-padding rows, with padding rows efficiently
/// represented by a `PaddingRow` descriptor.
pub trait MlPCS {
    type Field: TwoAdicField;
    type ExtensionField: TwoAdicField + ExtensionField<Self::Field>;
    type ProverData;
    type Commitment: Clone + Serialize + DeserializeOwned + Send + Sync;
    type Challenger;
    type BatchProof: Clone + Serialize + DeserializeOwned;
    type VerificationTrace;

    type Error: Debug;

    /// Commit to a batch of matrices. Each column of each matrix is treated as a
    /// multilinear polynomial over the hypercube.
    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<Self::Field>>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Commit with explicit per-call options.
    #[allow(clippy::type_complexity)]
    fn commit_with_options(
        &self,
        evaluations: Vec<&CompressedMatrix<Self::Field>>,
        options: MlCommitOptions,
    ) -> (Self::Commitment, Self::ProverData) {
        assert!(
            options.stacking.is_none(),
            "this PCS implementation does not support commit-time stacking options"
        );
        self.commit(evaluations)
    }

    /// Open multiple batches of polynomials at a single opening point, without rotation support.
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<Self::Field>>>,
        prover_data: Vec<Self::ProverData>,
        opening_point: &[Self::ExtensionField],
        opened_values: &[Vec<Vec<Self::ExtensionField>>],
        challenger: &mut Self::Challenger,
    ) -> Result<Self::BatchProof, Self::Error>;

    /// Verify multiple batches of polynomial openings at a single opening point, without rotation support.
    fn verify(
        &self,
        commitments: Vec<Self::Commitment>,
        matrices_size: &[Vec<Dimensions>],
        opening_point: &[Self::ExtensionField],
        opened_values: &[Vec<Vec<Self::ExtensionField>>],
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::VerificationTrace, Self::Error>;
}
