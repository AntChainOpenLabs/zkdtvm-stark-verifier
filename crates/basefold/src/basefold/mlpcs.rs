use core::fmt::Debug;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::Dimensions;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Multilinear Polynomial Commitment Scheme (PCS) trait.
///
/// In zkdtvm-stark-verifier only `verify` is implemented. The `commit` and `open` signatures
/// are retained for type-system compatibility but must not be called at runtime.
pub trait MlPCS {
    type Field: TwoAdicField;
    type ExtensionField: TwoAdicField + ExtensionField<Self::Field>;
    type ProverData;
    type Commitment: Clone + Serialize + DeserializeOwned + Send + Sync;
    type Challenger;
    type BatchProof: Clone + Serialize + DeserializeOwned;
    type Error: Debug;

    /// Stub — not available in the verifier-only build.
    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<&CompressedMatrix<Self::Field>>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Stub — not available in the verifier-only build.
    fn open(
        &self,
        polynomials_batch: Vec<Vec<CompressedMatrix<Self::Field>>>,
        prover_data: Vec<Self::ProverData>,
        opening_point: &[Self::ExtensionField],
        opened_values: &Vec<Vec<Vec<Self::ExtensionField>>>,
        challenger: &mut Self::Challenger,
    ) -> Result<Self::BatchProof, Self::Error>;

    /// Verify multiple batches of polynomial openings at a single opening point.
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
