//! Polynomial types used in the sumcheck protocol.
//!
//! This module defines four core polynomial types:
//! - [`EqPoly`]: Equality polynomial for multilinear extensions.
//! - [`BitExpandPoly`]: Bit-expansion polynomial for skip-round optimization.
//! - [`UnivariatePolynomial`]: Dense univariate polynomial with standard arithmetic.
//! - [`UniPolyEvals`]: Dense univariate polynomial in evaluation form at consecutive integers.

use core::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// Equality polynomial for multilinear extensions.
///
/// Maintains a tensor-product representation of `eq(z; x)` and supports
/// incremental variable fixing during sumcheck rounds.
#[derive(Debug)]
pub struct EqPoly<F, EF> {
    /// Verifier-sampled challenges `[z_0, z_1, ..., z_{n-1}]`.
    pub eq_challenges: Vec<EF>,
    /// Number of linear (degree-1) sumcheck rounds.
    pub num_linear_vars: usize,
    /// Degree of the nonlinear variables (`2^k - 1`).
    pub degree: usize,
    /// Number of variables fixed so far.
    pub num_vars_fixed: usize,
    /// Accumulated evaluation of fixed variables: `eq(z_fixed; r_fixed)`.
    pub eval: EF,
    /// Tensor-product coefficients, built layer by layer.
    ///
    /// `coeffs[k]` has length `(degree+1)^{k+1}` (or `2^{k+1}` for linear layers).
    /// After each round, the outermost layer is truncated.
    pub coeffs: Vec<Vec<EF>>,
    /// Precomputed Lagrange barycentric weights for nonlinear evaluation.
    pub weights: Vec<F>,
}

impl<F: Field, EF: ExtensionField<F>> EqPoly<F, EF> {
    /// Create a new `EqPoly` from the given challenges, number of linear rounds, and degree.
    pub fn new(eq_challenges: Vec<EF>, num_rounds_linear: usize, degree: usize) -> Self {
        debug_assert!(eq_challenges.len() >= num_rounds_linear);
        let coeffs = if eq_challenges.len() > 1 {
            Self::compute_eq_poly_coeffs(
                &eq_challenges[..eq_challenges.len() - 1],
                num_rounds_linear,
                degree,
            )
        } else {
            vec![]
        };
        let weights = Self::compute_weights(degree);
        Self {
            eq_challenges,
            num_linear_vars: num_rounds_linear,
            degree,
            num_vars_fixed: 0,
            eval: EF::one(),
            coeffs,
            weights,
        }
    }

    /// Fix the next variable to `challenge`, updating `eval` and truncating `coeffs`.
    pub fn update(&mut self, challenge: EF) {
        debug_assert!(self.num_vars_fixed < self.eq_challenges.len());
        let eq_challenge = self.eq_challenges[self.eq_challenges.len() - self.num_vars_fixed - 1];

        if self.num_vars_fixed < self.num_linear_vars {
            self.eval *= Self::eval_eq(challenge, eq_challenge, 1);
        } else {
            self.eval *= Self::eval_eq(challenge, eq_challenge, self.degree);
        }

        if self.coeffs.len() < 2 {
            self.coeffs.clear();
        } else {
            self.coeffs.truncate(self.coeffs.len() - 1);
        }

        self.num_vars_fixed += 1;
    }

    /// Compute eq poly coefficients for the small-endian (low-bit-first) folding
    /// order used by this sumcheck implementation.
    ///
    /// In the original (big-endian) version, the Kronecker product is `cur ⊗ prev`, placing the
    /// newest variable in the high-order (outer) position. In the small-endian version here, we
    /// reverse the order to `prev ⊗ cur`, so that the newest variable occupies the low-order
    /// (inner) position. This ensures that `coeffs[i]` is indexed consistently with the trace
    /// row index after folding from the high end.
    ///
    /// Concretely, for `eq_challenges = [z_0, z_1, ..., z_{n-2}]`:
    /// - `coeffs[0]` = `single(z_0)` (length `degree+1` or `2`)
    /// - `coeffs[1]` = `coeffs[0] ⊗ single(z_1)` — `z_0` in high bits, `z_1` in low bits
    /// - `coeffs[k]` = `coeffs[k-1] ⊗ single(z_k)` — newest variable always in lowest bits
    ///
    /// The last layer `coeffs[n-2]` covers all variables `z_0, ..., z_{n-2}`.
    /// After each sumcheck round, `update()` truncates the last layer, removing the
    /// highest-bit variable from the tensor product.
    fn compute_eq_poly_coeffs(
        eq_challenges: &[EF],
        num_rounds_linear: usize,
        degree: usize,
    ) -> Vec<Vec<EF>> {
        let mut ret = Vec::with_capacity(eq_challenges.len());
        let num_rounds_nonlinear = eq_challenges.len() + 1 - num_rounds_linear;
        let weights = Self::compute_weights(degree);
        eq_challenges.iter().enumerate().for_each(|(i, &r)| {
            let cur = if i < num_rounds_nonlinear {
                Self::compute_eq_poly_coeffs_single(r, degree, &weights)
            } else {
                vec![EF::one() - r, r]
            };
            if i == 0 {
                ret.push(cur);
            } else {
                let prev = ret.last().unwrap();
                // prev ⊗ cur: prev in high bits (outer), cur in low bits (inner)
                ret.push(
                    prev.par_iter()
                        .flat_map(|&x| cur.iter().map(|&y| x * y).collect::<Vec<EF>>())
                        .collect(),
                );
            }
        });
        ret
    }

    /// Compute Lagrange barycentric weights: `w_i = 1 / ∏_{j≠i} (i - j)`.
    fn compute_weights(degree: usize) -> Vec<F> {
        (0..=degree)
            .map(|i| {
                (0..=degree)
                    .filter(|j| *j != i)
                    .map(|j| F::from_canonical_usize(i) - F::from_canonical_usize(j))
                    .product::<F>()
                    .inverse()
            })
            .collect()
    }

    /// Compute single-variable eq coefficients: `eq(r; 0), eq(r; 1), ..., eq(r; degree)`.
    pub fn compute_eq_poly_coeffs_single(r: EF, degree: usize, weights: &[F]) -> Vec<EF> {
        if degree == 1 {
            return vec![EF::one() - r, r];
        }

        let mut coeffs = vec![EF::zero(); degree + 1];

        if let Some(i) = (0..=degree).find(|&i| r == EF::from_canonical_usize(i)) {
            coeffs[i] = EF::one();
            return coeffs;
        }

        let numerator: EF = (0..=degree).map(|i| r - F::from_canonical_usize(i)).product();
        coeffs.iter_mut().enumerate().for_each(|(i, c)| {
            // \prod_{j\ne i} (challenge-j)/(i-j)
            let i_base = F::from_canonical_usize(i);
            *c = numerator * (r - i_base).inverse() * weights[i];
        });
        coeffs
    }

    /// Evaluate `eq(a, b)` for the given degree.
    ///
    /// For degree 1: `eq(a, b) = 2ab - a - b + 1`.
    /// For higher degrees: Lagrange interpolation over `{0, 1, ..., degree}`.
    pub fn eval_eq(a: EF, b: EF, degree: usize) -> EF {
        if degree == 1 {
            return (a * b).double() - a - b + EF::one();
        }
        let len = degree + 1;
        let roots: Vec<EF> = (0..len).map(|i| EF::from_canonical_usize(i)).collect();

        match (roots.contains(&a), roots.contains(&b)) {
            (false, false) => Self::compute_eq_none(a, b, degree),
            (true, false) => Self::compute_eq_one(a, b, degree),
            (false, true) => Self::compute_eq_one(b, a, degree),
            _ => EF::from_bool(a == b),
        }
    }

    /// Evaluate `eq(a, b)` when exactly one argument is a root in `{0, ..., degree}`.
    fn compute_eq_one(a: EF, b: EF, degree: usize) -> EF {
        let roots: Vec<EF> = (0..=degree).map(|i| EF::from_canonical_usize(i)).collect();
        let numerator: EF = roots.iter().filter(|r| **r != a).map(|r| b - *r).product();
        let denominator: EF = roots.into_iter().filter(|r| *r != a).map(|r| a - r).product();
        numerator * denominator.inverse()
    }

    /// Evaluate `eq(a, b)` when neither argument is a root in `{0, ..., degree}`.
    fn compute_eq_none(a: EF, b: EF, degree: usize) -> EF {
        let len = degree + 1;
        let roots: Vec<F> = (0..len).map(|i| F::from_canonical_usize(i)).collect();
        let numerator: EF = roots.iter().map(|r| (a - *r) * (b - *r)).product();
        roots
            .par_iter()
            .enumerate()
            .map(|(i, r)| {
                let mut denominator: EF =
                    (0..len).filter(|j| *j != i).map(|j| *r - roots[j]).product::<F>().into();
                denominator = denominator.square();
                denominator *= (a - *r) * (b - *r);
                numerator * denominator.inverse()
            })
            .sum()
    }
}

/// Bit-expansion polynomial for skip-round optimization.
///
/// Given `n = 2^k` interpolation points, this type precomputes Lagrange weights
/// and bit-indexed subsets so that the `k` bit-expansion polynomials
/// `M_t(x) = Σ_{j: bit t of j is 1} L_j(x)` can be evaluated efficiently.
pub struct BitExpandPoly<F> {
    /// Interpolation points `[p_0, p_1, ..., p_{n-1}]`.
    pub points: Vec<F>,
    /// Lagrange barycentric weights: `w_i = 1 / ∏_{j≠i} (p_i - p_j)`.
    pub lagrange_weights: Vec<F>,
    /// For each bit position `t`, the set of indices `j` whose `t`-th bit is 1.
    pub indices_for_bit: Vec<Vec<usize>>,
}

impl<F: std::fmt::Debug> std::fmt::Debug for BitExpandPoly<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitExpandPoly")
            .field("points", &self.points)
            .field("lagrange_weights", &self.lagrange_weights)
            .field("indices_for_bit", &self.indices_for_bit)
            .finish()
    }
}

impl<F: Field> BitExpandPoly<F> {
    /// Create a new `BitExpandPoly` from the given interpolation points.
    ///
    /// # Panics
    ///
    /// Panics if `points.len()` is not a power of two, or if points are not distinct.
    #[must_use]
    pub fn new(points: Vec<F>) -> Self {
        let n = points.len();
        assert!(n.is_power_of_two(), "length of points should be power of 2");
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(points[i] != points[j], "all points should be distinct");
            }
        }

        let lagrange_weights = Self::compute_lagrange_weights(&points);
        let indices_for_bit = Self::compute_indices_for_bits(n);
        Self { points, lagrange_weights, indices_for_bit }
    }

    /// Compute Lagrange barycentric weights: `w_i = 1 / ∏_{j≠i} (p_i - p_j)`.
    fn compute_lagrange_weights(points: &[F]) -> Vec<F> {
        let n = points.len();
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let mut denominator = F::one();
            for j in 0..n {
                if i != j {
                    denominator *= points[i] - points[j];
                }
            }
            weights.push(denominator.inverse());
        }
        weights
    }

    /// For each bit position `t` in `0..log2(n)`, collect indices `j` where bit `t` is set.
    fn compute_indices_for_bits(n: usize) -> Vec<Vec<usize>> {
        debug_assert!(n.is_power_of_two());
        let num_bits = n.trailing_zeros() as usize;
        (0..num_bits)
            .map(|bit_index| (0..n).filter(|j| (j >> bit_index) & 1 == 1).collect())
            .collect()
    }

    /// Return the polynomial degree (`n - 1`).
    #[cfg(test)]
    #[must_use]
    pub fn degree(&self) -> usize {
        self.points.len() - 1
    }

    /// Evaluate the bit-expansion polynomial `M_{bit_index}(x)`.
    ///
    /// `M_t(x) = Σ_{j: bit t of j is 1} L_j(x)`, where `L_j` are Lagrange basis polynomials.
    ///
    /// # Panics
    ///
    /// Panics if `bit_index >= log2(n)`.
    #[cfg(test)]
    #[must_use]
    pub fn eval_at<T: ExtensionField<F>>(&self, bit_index: usize, x: T) -> T {
        assert!(bit_index < self.indices_for_bit.len(), "bit index out of range");
        let indices = &self.indices_for_bit[bit_index];
        if indices.is_empty() {
            return T::zero();
        }

        let x_minus_points: Vec<T> = self.points.iter().map(|point| x - *point).collect();

        // Factor out `(x - p_k)` for all `k ∉ indices` (common to every L_j in the sum).
        let mut common_factors = T::one();
        for (i, factor) in x_minus_points.iter().enumerate() {
            if !indices.contains(&i) {
                common_factors *= *factor;
            }
        }

        let mut result = T::zero();
        for &j in indices {
            // Non-common part: ∏_{i ∈ indices, i ≠ j} (x - p_i)
            let mut non_common_part = T::one();
            for &i in indices {
                if i != j {
                    non_common_part *= x_minus_points[i];
                }
            }
            result += common_factors * non_common_part * self.lagrange_weights[j];
        }
        result
    }

    /// Evaluate all `k` bit-expansion polynomials at `x` (simple per-bit method).
    #[cfg(test)]
    pub fn evals_all_simple<T: ExtensionField<F>>(&self, x: T) -> Vec<T> {
        let num_bits = self.points.len().trailing_zeros() as usize;
        (0..num_bits).map(|i| self.eval_at(i, x)).collect()
    }

    /// Evaluate all `k` bit-expansion polynomials at `x` (optimized batch method).
    pub fn evals_all<T: ExtensionField<F>>(&self, x: T) -> Vec<T> {
        let x_minus_points: Vec<_> = self.points.iter().map(|point| x - *point).collect();
        let numerators = super::utils::products_all_but_one(&x_minus_points);
        let lagrange_evals: Vec<_> =
            numerators.into_iter().zip(self.lagrange_weights.iter()).map(|(n, w)| n * *w).collect();
        let num_bits = self.points.len().trailing_zeros() as usize;
        (0..num_bits)
            .map(|i| {
                let chunk_size = 1 << i;
                let mut result = T::zero();
                for chunk in lagrange_evals.chunks_exact(chunk_size).skip(1).step_by(2) {
                    result += chunk.iter().copied().sum::<T>();
                }
                result
            })
            .collect()
    }
}

/// Dense univariate polynomial in coefficient form: `c_0 + c_1·x + c_2·x² + ...`
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnivariatePolynomial<K> {
    /// Coefficients in ascending degree order: `[c_0, c_1, ..., c_d]`.
    pub coefficients: Vec<K>,
}

impl<K: Field> UnivariatePolynomial<K> {
    /// Create a polynomial from the given coefficients.
    #[must_use]
    pub fn new(coefficients: Vec<K>) -> Self {
        Self { coefficients }
    }

    /// Return `x · P(x)` (shift all coefficients up by one degree).
    #[must_use]
    pub fn mul_by_x(&self) -> Self {
        let mut result = Vec::with_capacity(self.coefficients.len() + 1);
        result.push(K::zero());
        result.extend(&self.coefficients[..]);
        Self::new(result)
    }

    /// Create the zero polynomial with `degree + 1` zero coefficients.
    #[must_use]
    pub fn zero(degree: usize) -> Self {
        Self { coefficients: vec![K::zero(); degree + 1] }
    }

    /// Create the constant-one polynomial with `degree + 1` coefficients.
    #[must_use]
    pub fn one(degree: usize) -> Self {
        let mut coefficients = vec![K::zero(); degree + 1];
        coefficients[0] = K::one();
        Self { coefficients }
    }

    /// Evaluate the polynomial at `point` using Horner's method.
    pub fn eval_at_point(&self, point: K) -> K {
        self.coefficients.iter().rev().fold(K::zero(), |acc, x| acc * point + *x)
    }

    /// Return `P(0) + P(1)`, used in sumcheck round verification.
    #[must_use]
    pub fn eval_one_plus_eval_zero(&self) -> K {
        if self.coefficients.is_empty() {
            K::zero()
        } else {
            self.coefficients[0] + self.coefficients.iter().copied().sum::<K>()
        }
    }
}

impl<K> IntoIterator for UnivariatePolynomial<K> {
    type Item = K;
    type IntoIter = std::vec::IntoIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        self.coefficients.into_iter()
    }
}

impl<K: Field> Mul<K> for UnivariatePolynomial<K> {
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        Self { coefficients: self.coefficients.into_iter().map(|x| x * rhs).collect() }
    }
}

impl<K: Field> MulAssign<K> for UnivariatePolynomial<K> {
    fn mul_assign(&mut self, rhs: K) {
        self.coefficients.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<K: Field> MulAssign for UnivariatePolynomial<K> {
    fn mul_assign(&mut self, rhs: Self) {
        if rhs.coefficients.len() < 2 {
            *self *= *rhs.coefficients.first().unwrap_or(&K::zero());
            return;
        }
        let mut t = self.clone();
        for (i, c) in rhs.coefficients.into_iter().enumerate() {
            if i > 0 {
                t = t.mul_by_x();
                *self += t.clone() * c;
                continue;
            }
            *self *= c;
        }
    }
}

impl<K: Field> Add for UnivariatePolynomial<K> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new_coeffs = vec![K::zero(); self.coefficients.len().max(rhs.coefficients.len())];
        for (i, x) in new_coeffs.iter_mut().enumerate() {
            *x = *self.coefficients.get(i).unwrap_or(&K::zero()) +
                *rhs.coefficients.get(i).unwrap_or(&K::zero());
        }
        UnivariatePolynomial::new(new_coeffs)
    }
}

impl<F: Field> AddAssign for UnivariatePolynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        let mut new_coeffs = vec![F::zero(); self.coefficients.len().max(rhs.coefficients.len())];
        for (i, x) in new_coeffs.iter_mut().enumerate() {
            *x = *self.coefficients.get(i).unwrap_or(&F::zero()) +
                *rhs.coefficients.get(i).unwrap_or(&F::zero());
        }
        self.coefficients = new_coeffs;
    }
}

impl<K: Field> AddAssign<K> for UnivariatePolynomial<K> {
    fn add_assign(&mut self, rhs: K) {
        if self.coefficients.is_empty() {
            self.coefficients.push(rhs);
            return;
        }
        self.coefficients[0] += rhs;
    }
}

impl<F: Field> Sum for UnivariatePolynomial<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(0), |acc, x| acc + x)
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    pub fn sum_refs<'a>(iter: impl Iterator<Item = &'a Self>) -> Self {
        let mut result = Self::zero(0);
        for p in iter {
            if result.coefficients.len() < p.coefficients.len() {
                result.coefficients.resize(p.coefficients.len(), F::zero());
            }
            for (i, c) in p.coefficients.iter().enumerate() {
                result.coefficients[i] += *c;
            }
        }
        result
    }
}

/// Dense univariate polynomial in evaluation form at consecutive integers.
///
/// Stores `[f(0), f(1), ..., f(d)]` where `d` is the polynomial degree.
/// Any degree-`d` polynomial is uniquely determined by `d+1` evaluations,
/// and this representation avoids costly coefficient interpolation on the prover
/// while enabling O(d) evaluation at arbitrary points via the barycentric formula
/// for equally-spaced nodes.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct UniPolyEvals<K> {
    /// Evaluations at consecutive integers: `[f(0), f(1), ..., f(d)]`.
    pub evals: Vec<K>,
}

impl<K: Field> UniPolyEvals<K> {
    #[must_use]
    pub fn new(evals: Vec<K>) -> Self {
        Self { evals }
    }

    #[must_use]
    pub fn zero(degree: usize) -> Self {
        Self { evals: vec![K::zero(); degree + 1] }
    }

    #[must_use]
    pub fn degree(&self) -> usize {
        debug_assert!(!self.evals.is_empty());
        self.evals.len() - 1
    }

    /// Evaluate the polynomial at an arbitrary field element `r` using the O(d)
    /// barycentric Lagrange formula for nodes {0, 1, ..., d}.
    ///
    /// If `r` is one of the nodes, returns the stored value directly.
    pub fn eval_at_point(&self, r: K) -> K {
        let d = self.evals.len() - 1;

        for i in 0..=d {
            if r == K::from_canonical_usize(i) {
                return self.evals[i];
            }
        }

        let mut r_minus: Vec<K> = Vec::with_capacity(d + 1);
        for i in 0..=d {
            r_minus.push(r - K::from_canonical_usize(i));
        }

        let mut prefix = vec![K::one(); d + 2];
        for i in 0..=d {
            prefix[i + 1] = prefix[i] * r_minus[i];
        }

        let mut suffix = vec![K::one(); d + 2];
        for i in (0..=d).rev() {
            suffix[i] = suffix[i + 1] * r_minus[i];
        }

        let weights = Self::barycentric_weights_consecutive(d);

        let mut result = K::zero();
        for i in 0..=d {
            result += self.evals[i] * weights[i] * prefix[i] * suffix[i + 1];
        }
        result
    }

    #[must_use]
    pub fn eval_one_plus_eval_zero(&self) -> K {
        if self.evals.len() < 2 {
            return if self.evals.is_empty() { K::zero() } else { self.evals[0] };
        }
        self.evals[0] + self.evals[1]
    }

    #[must_use]
    pub fn sum_over_range(&self, n: usize) -> K {
        self.evals[..n].iter().copied().sum()
    }

    /// Pointwise addition of eval-form polynomials.
    ///
    /// Shorter polynomials are extended to the maximum length via barycentric
    /// interpolation so that the sum is exact at all evaluation nodes.
    pub fn sum_refs<'a>(iter: impl Iterator<Item = &'a Self>) -> Self
    where
        K: 'a,
    {
        let polys: Vec<&Self> = iter.collect();
        if polys.is_empty() {
            return Self { evals: vec![K::zero()] };
        }
        let max_len = polys.iter().map(|p| p.evals.len()).max().unwrap_or(1);
        let mut result = vec![K::zero(); max_len];
        for p in &polys {
            for i in 0..max_len {
                if i < p.evals.len() {
                    result[i] += p.evals[i];
                } else {
                    result[i] += p.eval_at_point(K::from_canonical_usize(i));
                }
            }
        }
        Self { evals: result }
    }

    /// Extend the evaluation vector to `target_len` points by computing
    /// the polynomial at new consecutive-integer nodes using the barycentric formula.
    pub fn extend_to(&mut self, target_len: usize) {
        while self.evals.len() < target_len {
            let i = self.evals.len();
            let val = self.eval_at_point(K::from_canonical_usize(i));
            self.evals.push(val);
        }
    }

    /// Element-wise multiplication of two eval-form polynomials.
    ///
    /// Both must be evaluated at the same points. The result correctly represents
    /// the product polynomial **only** when `self.evals.len()` is at least
    /// `deg(self) + deg(other) + 1`.
    #[must_use]
    pub fn mul_pointwise(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.evals.len(),
            other.evals.len(),
            "pointwise mul requires same number of evaluation points"
        );
        Self { evals: self.evals.iter().zip(other.evals.iter()).map(|(&a, &b)| a * b).collect() }
    }

    /// Precompute barycentric weights for nodes {0, 1, ..., d}.
    ///
    /// `w_i = 1 / (i! * (-1)^{d-i} * (d-i)!)`
    fn barycentric_weights_consecutive(d: usize) -> Vec<K> {
        let mut weights = Vec::with_capacity(d + 1);
        let mut factorial_d = K::one();
        for j in 1..=d {
            factorial_d *= K::from_canonical_usize(j);
        }
        let sign_d = if d.is_multiple_of(2) { K::one() } else { K::neg_one() };
        weights.push((sign_d * factorial_d).inverse());

        for i in 1..=d {
            let prev = *weights.last().unwrap();
            let ratio = K::neg_one() *
                K::from_canonical_usize(d - i + 1) *
                K::from_canonical_usize(i).inverse();
            weights.push(prev * ratio);
        }
        weights
    }
}

impl<K: Field> Add for UniPolyEvals<K> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<K: Field> AddAssign for UniPolyEvals<K> {
    fn add_assign(&mut self, rhs: Self) {
        let max_len = self.evals.len().max(rhs.evals.len());
        self.extend_to(max_len);
        for i in 0..max_len {
            let rhs_val = if i < rhs.evals.len() {
                rhs.evals[i]
            } else {
                rhs.eval_at_point(K::from_canonical_usize(i))
            };
            self.evals[i] += rhs_val;
        }
    }
}

impl<K: Field> Mul<K> for UniPolyEvals<K> {
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        Self { evals: self.evals.into_iter().map(|x| x * rhs).collect() }
    }
}

impl<K: Field> MulAssign<K> for UniPolyEvals<K> {
    fn mul_assign(&mut self, rhs: K) {
        self.evals.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<K: Field> Sum for UniPolyEvals<K> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(0), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests_bit_expand_poly {
    #![allow(clippy::print_stdout)]
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};

    #[test]
    fn test_bit_expand_poly_n_4_indices_for_bit() {
        let points = vec![
            BabyBear::from_canonical_u32(0),
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(3),
        ];
        let poly = BitExpandPoly::new(points);

        println!("=== BitExpandPoly (n=4) ===");
        println!("points: {:?}", poly.points);
        println!("lagrange_weights: {:?}", poly.lagrange_weights);
        println!();
        println!("indices_for_bit:");
        for (i, indices) in poly.indices_for_bit.iter().enumerate() {
            println!("  bit_index {i}: {indices:?}");
        }
    }

    #[test]
    fn test_bit_expand_poly_n_8_indices_for_bit() {
        let points: Vec<BabyBear> = (0..8).map(BabyBear::from_canonical_u32).collect();
        let poly = BitExpandPoly::new(points);

        println!("=== BitExpandPoly (n=8) ===");
        println!("points: {:?}", poly.points);
        println!();
        println!("indices_for_bit:");
        for (i, indices) in poly.indices_for_bit.iter().enumerate() {
            println!("  bit_index {i}: {indices:?}");
        }
    }

    #[test]
    fn test_bit_expand_poly_evals_all_with_binary_field() {
        type EF = BinomialExtensionField<BabyBear, 4>;

        let points: Vec<BabyBear> = (0..4).map(BabyBear::from_canonical_u32).collect();
        let poly = BitExpandPoly::new(points);

        println!("=== BitExpandPoly Eval Tests (n=4) ===");

        let test_x =
            EF::from_base_fn(|i| BabyBear::from_canonical_u32((5 + i * 13).try_into().unwrap()));
        println!("Test x = {test_x:?}");

        let results_simple = poly.evals_all_simple(test_x);
        let results_optimized = poly.evals_all(test_x);

        println!();
        println!("evals_all_simple results:");
        for (i, result) in results_simple.iter().enumerate() {
            println!("  bit_index {i}: {result:?}");
        }

        println!();
        println!("evals_all results:");
        for (i, result) in results_optimized.iter().enumerate() {
            println!("  bit_index {i}: {result:?}");
        }

        assert_eq!(results_simple, results_optimized);
        println!();
        println!("✓ evals_all_simple == evals_all");
    }

    #[test]
    fn test_bit_expand_poly_evals_all_n_8() {
        type EF = BinomialExtensionField<BabyBear, 4>;

        let points: Vec<BabyBear> = (0..8).map(BabyBear::from_canonical_u32).collect();
        let poly = BitExpandPoly::new(points);

        println!("=== BitExpandPoly Eval Tests (n=8) ===");
        println!("degree: {}", poly.degree());

        let test_x =
            EF::from_base_fn(|i| BabyBear::from_canonical_u32((3 + i * 17).try_into().unwrap()));
        println!("Test x = {test_x:?}");

        let results_simple = poly.evals_all_simple(test_x);
        let results_optimized = poly.evals_all(test_x);

        println!();
        println!("evals_all_simple results:");
        for (i, result) in results_simple.iter().enumerate() {
            println!("  bit_index {i}: {result:?}");
        }

        println!();
        println!("evals_all results:");
        for (i, result) in results_optimized.iter().enumerate() {
            println!("  bit_index {i}: {result:?}");
        }

        assert_eq!(results_simple, results_optimized);
        println!();
        println!("✓ evals_all_simple == evals_all");
    }

    #[test]
    fn test_bit_expand_poly_n_2() {
        type EF = BinomialExtensionField<BabyBear, 4>;

        let points: Vec<BabyBear> =
            (0..2).map(|i| BabyBear::from_canonical_u32(i * 2 + 1)).collect();
        let poly = BitExpandPoly::new(points);

        println!("=== BitExpandPoly (n=2) ===");
        println!("points: {:?}", poly.points);
        println!("degree: {}", poly.degree());

        println!();
        println!("indices_for_bit:");
        for (i, indices) in poly.indices_for_bit.iter().enumerate() {
            println!("  bit_index {i}: {indices:?}");
        }

        let test_x = EF::from_base_fn(|i| BabyBear::from_canonical_u32(i.try_into().unwrap()));
        let results_simple = poly.evals_all_simple(test_x);
        let results_optimized = poly.evals_all(test_x);

        println!();
        println!("evals_all_simple: {results_simple:?}");
        println!("evals_all: {results_optimized:?}");

        assert_eq!(results_simple, results_optimized);
        println!();
        println!("✓ evals_all_simple == evals_all");
    }
}
