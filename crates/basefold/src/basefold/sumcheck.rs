use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::utils::unipoly::UniPoly;

/// Univariate polynomials generated in the sumcheck protocol.
///
/// Verifier-only: we deserialize this from proofs but never construct it.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SumcheckInstanceProof<F> {
    pub uni_polys: Vec<UniPoly<F>>,
}

impl<F: Field> SumcheckInstanceProof<F> {
    pub fn new(uni_polys: Vec<UniPoly<F>>) -> Self {
        Self { uni_polys }
    }
}
