use core::fmt::Debug;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::Dimensions;
use serde::de::DeserializeOwned;
use serde::Serialize;

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

    type Error: Debug;

    /// Commit to a batch of matrices. Each column of each matrix is treated as a
    /// multilinear polynomial over the hypercube.
    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<Self::Field>>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Open multiple batches of polynomials at a single opening point, without rotation support.
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<Self::Field>>>,
        prover_data: Vec<Self::ProverData>,
        opening_point: &[Self::ExtensionField],
        opened_values: &Vec<Vec<Vec<Self::ExtensionField>>>,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::BatchProof, Self::Error>;

    /// Verify multiple batches of polynomial openings at a single opening point, without rotation support.
    fn verify(
        &self,
        commitments: Vec<Self::Commitment>,
        matrices_size: &Vec<Vec<Dimensions>>,
        opening_point: &[Self::ExtensionField],
        opened_values: &Vec<Vec<Vec<Self::ExtensionField>>>,
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error>;
}