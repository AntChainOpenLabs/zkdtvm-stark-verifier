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
        match evals.len() {
            0 => UniPoly { coeffs: Vec::new() },
            1 => UniPoly {
                coeffs: vec![evals[0]],
            },
            2 => UniPoly {
                coeffs: vec![evals[0], evals[1] - evals[0]],
            },
            3 => Self::from_evals_quadratic_012(evals[0], evals[1], evals[2]),
            _ => UniPoly {
                coeffs: Self::vandermonde_interpolation(evals),
            },
        }
    }

    /// Interpolate a degree-2 polynomial from evaluations at x = 0, 1, 2.
    pub fn from_evals_quadratic_012(eval_0: F, eval_1: F, eval_2: F) -> Self {
        let coeff_0 = eval_0;
        let coeff_2 = (eval_2 - eval_1 - eval_1 + eval_0).halve();
        let coeff_1 = eval_1 - eval_0 - coeff_2;
        UniPoly {
            coeffs: vec![coeff_0, coeff_1, coeff_2],
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
        self.coeffs
            .iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * *r + *coeff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;

    type F = BabyBear;
    fn test_from_evals_quad_helper<F: Field>() {
        // polynomial is 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_canonical_u64(6u64);
        let e2 = F::from_canonical_u64(15u64);
        let evals = vec![e0, e1, e2];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 3);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_canonical_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_canonical_u64(2u64));

        let e3 = F::from_canonical_u64(28u64);
        assert_eq!(poly.evaluate(&F::from_canonical_u64(3u64)), e3);
    }

    fn test_from_evals_cubic_helper<F: Field>() {
        // polynomial is x^3 + 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_canonical_u64(7u64);
        let e2 = F::from_canonical_u64(23u64);
        let e3 = F::from_canonical_u64(55u64);
        let evals = vec![e0, e1, e2, e3];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 4);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_canonical_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_canonical_u64(2u64));
        assert_eq!(poly.coeffs[3], F::one());

        let e4 = F::from_canonical_u64(109u64);
        assert_eq!(poly.evaluate(&F::from_canonical_u64(4u64)), e4);
    }

    #[test]
    fn test_from_evals_quad() {
        test_from_evals_quad_helper::<F>();
        test_from_evals_cubic_helper::<F>();
    }
}
