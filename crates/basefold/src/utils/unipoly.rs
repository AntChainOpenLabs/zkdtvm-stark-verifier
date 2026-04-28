#![allow(dead_code)]
use crate::utils::math::gaussian_elimination;
use p3_field::Field;
use serde::{Deserialize, Serialize};

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct UniPoly<F> {
    pub coeffs: Vec<F>,
}

impl<F: Field> UniPoly<F> {
    #[allow(dead_code)]
    pub fn from_coeff(coeffs: Vec<F>) -> Self {
        UniPoly { coeffs }
    }

    pub fn from_evals(evals: &[F]) -> Self {
        UniPoly {
            coeffs: Self::vandermonde_interpolation(evals),
        }
    }

    /// Uses the Vandermonde interpolation method to generate polynomial coefficients from given evaluation points.
    ///
    /// # Parameters
    /// - `evals`: A slice containing the values of the polynomial evaluated at different points.
    ///
    /// # Returns
    /// Returns a `Vec<F>` representing the interpolated polynomial coefficients.
    ///
    /// # Algorithm
    /// 1. Generates a Vandermonde matrix, where each row represents the powers of a point.
    /// 2. Appends the evaluation value to the end of each row of the Vandermonde matrix.
    /// 3. Uses Gaussian elimination to solve the linear system and obtain the polynomial coefficients.
    fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
        let n = evals.len();
        let xs: Vec<F> = (0..n).map(|x| F::from_canonical_u64(x as u64)).collect();

        let mut vandermonde: Vec<Vec<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let x = xs[i];
            row.push(F::one());
            row.push(x);
            for j in 2..n {
                row.push(row[j - 1] * x);
            }
            row.push(evals[i]);
            vandermonde.push(row);
        }

        gaussian_elimination(&mut vandermonde)
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn as_vec(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    pub fn eval_at_zero(&self) -> F {
        self.coeffs[0]
    }

    pub fn eval_at_one(&self) -> F {
        (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
    }

    pub fn evaluate(&self, r: &F) -> F {
        let mut eval = self.coeffs[0];
        let mut power = *r;
        for i in 1..self.coeffs.len() {
            eval += power * self.coeffs[i];
            power *= *r;
        }
        eval
    }

}
