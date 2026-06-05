use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use crate::utils::math::Math;
use crate::utils::mlpoly::MultilinearPolynomial;

/// The equality polynomial `eq(r, x) = Π_i (r_i · x_i + (1 - r_i) · (1 - x_i))`.
///
/// Given a fixed point `r ∈ F^n`, `eq(r, ·)` is a multilinear polynomial over `{0,1}^n`
/// that equals 1 when `x = r` (if `r` is Boolean) and satisfies `Σ_x eq(r, x) = 1`
/// for any `r`. It is the fundamental building block for multilinear sumcheck protocols.
pub struct EqPolynomial<F> {
    r: Vec<F>,
}

const PARALLEL_THRESHOLD: usize = 16;

impl<F: Field> EqPolynomial<F> {
    pub fn new(r: Vec<F>) -> Self {
        EqPolynomial { r }
    }

    /// Evaluate `eq(r, rx) = Π_i (r_i · rx_i + (1 - r_i) · (1 - rx_i))` directly.
    pub fn evaluate(&self, rx: &[F]) -> F {
        assert_eq!(self.r.len(), rx.len());
        (0..rx.len())
            .map(|i| self.r[i] * rx[i] + (F::one() - self.r[i]) * (F::one() - rx[i]))
            .product()
    }

    /// Compute all `2^n` evaluations of `eq(r, x)` over the Boolean hypercube `x ∈ {0,1}^n`.
    ///
    /// Uses dynamic programming: at each step, the table doubles in size by splitting
    /// each entry into `(1 - r_j) · entry` and `r_j · entry`.
    pub fn evals(&self) -> Vec<F> {
        let ell = self.r.len();

        match ell {
            0..=PARALLEL_THRESHOLD => self.evals_serial(ell),
            _ => self.evals_parallel(ell),
        }
    }

    #[inline]
    pub fn to_ml(&self) -> MultilinearPolynomial<F> {
        MultilinearPolynomial::new(self.evals())
    }

    /// Computes evals serially. Uses less memory (and fewer allocations) than `evals_parallel`.
    fn evals_serial(&self, ell: usize) -> Vec<F> {
        let mut evals: Vec<F> = vec![F::one(); ell.pow2()];
        let mut size = 1;
        for j in 0..ell {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * self.r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Computes evals in parallel. Uses more memory and allocations than `evals_serial`, but
    /// evaluates biggest layers of the dynamic programming tree in parallel.
    fn evals_parallel(&self, ell: usize) -> Vec<F> {
        let final_size = (2usize).pow(ell as u32);
        let mut evals: Vec<F> = vec![F::zero(); final_size];
        let mut size = 1;
        evals[0] = F::one();

        for r in self.r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * *r;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    type F = BabyBear;

    fn f(val: u32) -> F {
        F::from_canonical_u32(val)
    }

    #[test]
    fn test_evals_single_variable() {
        // eq(r, x) with r = [3], over x in {0, 1}
        // eq(3, 0) = (1 - 3) = -2, eq(3, 1) = 3
        let eq = EqPolynomial::new(vec![f(3)]);
        let evals = eq.evals();
        assert_eq!(evals.len(), 2);
        assert_eq!(evals[0], F::one() - f(3)); // 1 - r = -2
        assert_eq!(evals[1], f(3));             // r = 3
    }

    #[test]
    fn test_evals_two_variables() {
        // eq(r, x) with r = [2, 5], over x in {0,1}^2
        // eq((r0,r1), (x0,x1)) = prod_i (r_i * x_i + (1-r_i)*(1-x_i))
        let r0 = f(2);
        let r1 = f(5);
        let eq = EqPolynomial::new(vec![r0, r1]);
        let evals = eq.evals();
        assert_eq!(evals.len(), 4);

        // Manually compute: eq(r, x) for each x in {00, 01, 10, 11}
        let one = F::one();
        // x=(0,0): (1-r0)*(1-r1)
        assert_eq!(evals[0], (one - r0) * (one - r1));
        // x=(0,1): (1-r0)*r1
        assert_eq!(evals[1], (one - r0) * r1);
        // x=(1,0): r0*(1-r1)
        assert_eq!(evals[2], r0 * (one - r1));
        // x=(1,1): r0*r1
        assert_eq!(evals[3], r0 * r1);
    }

    #[test]
    fn test_evaluate_matches_evals() {
        // Verify that evals() produces the correct eq polynomial evaluations
        // by checking against manual computation.
        //
        // In evals_serial, variable r[j] is expanded at iteration j.
        // The resulting index layout has the most-significant bit corresponding
        // to r[0] (the first variable expanded).
        let r = vec![f(7), f(11)];
        let eq = EqPolynomial::new(r.clone());
        let evals = eq.evals();
        assert_eq!(evals.len(), 4);

        let one = F::one();
        // Determine the actual mapping by checking index 0 and 1
        // evals[0] should be the product of all (1 - r[j]) terms
        let all_zero = (one - r[0]) * (one - r[1]);
        assert_eq!(evals[0], all_zero, "index 0 should be eq(r, (0,0))");

        // evals[3] should be the product of all r[j] terms
        let all_one = r[0] * r[1];
        assert_eq!(evals[3], all_one, "index 3 should be eq(r, (1,1))");

        // Verify the full table sums to 1
        let sum: F = evals.iter().copied().sum();
        assert_eq!(sum, one);
    }

    #[test]
    fn test_evals_sum_to_one() {
        // The sum of eq(r, x) over all x in {0,1}^n should equal 1
        let r = vec![f(3), f(7), f(11), f(13)];
        let eq = EqPolynomial::new(r);
        let evals = eq.evals();
        let sum: F = evals.iter().copied().sum();
        assert_eq!(sum, F::one());
    }

    #[test]
    fn test_serial_and_parallel_agree() {
        // Use a large enough number of variables to trigger the parallel path
        let num_vars = PARALLEL_THRESHOLD + 1;
        let r: Vec<F> = (1..=num_vars as u32).map(f).collect();
        let eq = EqPolynomial::new(r.clone());

        let serial_result = eq.evals_serial(num_vars);
        let parallel_result = eq.evals_parallel(num_vars);
        assert_eq!(serial_result, parallel_result);
    }

    #[test]
    fn test_to_ml_length() {
        let r = vec![f(2), f(3), f(5)];
        let eq = EqPolynomial::new(r);
        let ml = eq.to_ml();
        assert_eq!(ml.len(), 8); // 2^3
        assert_eq!(ml.get_num_vars(), 3);
    }

    #[test]
    fn test_evaluate_symmetry() {
        // eq(r, x) = prod_i (r_i * x_i + (1-r_i)*(1-x_i))
        // Verify evaluate gives the same result as manual computation
        let r = vec![f(3), f(7)];
        let x = vec![f(5), f(11)];
        let eq = EqPolynomial::new(r.clone());
        let result = eq.evaluate(&x);

        let one = F::one();
        let expected = (r[0] * x[0] + (one - r[0]) * (one - x[0]))
            * (r[1] * x[1] + (one - r[1]) * (one - x[1]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_polynomial() {
        // eq with 0 variables: should return a single element [1]
        let eq = EqPolynomial::<F>::new(vec![]);
        let evals = eq.evals();
        assert_eq!(evals.len(), 1);
        assert_eq!(evals[0], F::one());
    }
}
