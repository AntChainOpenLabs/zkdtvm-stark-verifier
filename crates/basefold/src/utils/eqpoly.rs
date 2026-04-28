use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use crate::utils::math::Math;

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
