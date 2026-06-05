//! Utility functions for the sumcheck protocol.
//!
//! This module provides helper routines used throughout the sumcheck implementation:
//! - **Vector operations**: `linear_combination_slices`.
//! - **Polynomial utilities**: `interpolate_univariate_polynomial`, `barycentric_weights`,
//!   `multiply_poly_by_eq`.
//! - **Selector helpers**: packed first/last row selectors in the [`selectors`] submodule.

use p3_field::{
    extension::{BinomialExtensionField, BinomiallyExtendable},
    AbstractExtensionField, AbstractField, ExtensionField, Field, PackedField, PackedValue,
    PrimeField32,
};
use p3_maybe_rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::types::{EqPoly, UniPolyEvals, UnivariatePolynomial};

/// Compute the number of active chips in each sumcheck round.
///
/// Chips with smaller log-heights join later rounds. This function returns a vector
/// where `result[i]` is the cumulative chip count at round `i`.
pub(crate) fn compute_num_chips_each_round(
    log_heights: &[usize],
    num_skip_rounds: usize,
    log_height_threshold: usize,
) -> Vec<usize> {
    debug_assert!(!log_heights.is_empty());
    debug_assert_eq!(
        log_height_threshold % num_skip_rounds,
        0,
        "log_threshold must be divisible by num_skip_rounds"
    );

    let max_height = log_heights[0];

    let mut count = vec![0; max_height + 1];
    for &height in log_heights {
        count[height] += 1;
    }

    let num_rounds_degree_one = max_height.saturating_sub(log_height_threshold);
    let num_rounds_high_degree = std::cmp::min(max_height, log_height_threshold) / num_skip_rounds;
    let num_rounds = num_rounds_degree_one + num_rounds_high_degree;
    let mut result = Vec::with_capacity(num_rounds);
    let mut cur_count = count[max_height];
    let mut cur_val = max_height;

    for _ in 0..num_rounds {
        result.push(cur_count);
        let skip_idx = if cur_val > log_height_threshold { 1 } else { num_skip_rounds };
        count[cur_val - skip_idx] += cur_count;
        cur_val -= skip_idx;
        cur_count = count[cur_val];
    }

    result
}

/// Compute per-chip powers of the constraint-combining random challenge `alpha`.
///
/// For each chip with `num_constraints[i]` constraints, collects the next batch of
/// consecutive powers of `alpha` (in reverse order within each batch).
pub(crate) fn compute_powers_of_alpha<F: Field>(
    alpha: F,
    num_constraints: Vec<usize>,
) -> Vec<Vec<F>> {
    let mut powers = alpha.powers();
    let mut powers_of_alpha = Vec::with_capacity(num_constraints.len());
    for len in num_constraints {
        let mut powers_of_alpha_i = Vec::with_capacity(len);
        for _ in 0..len {
            powers_of_alpha_i.push(powers.next().unwrap());
        }
        powers_of_alpha_i.reverse();
        powers_of_alpha.push(powers_of_alpha_i);
    }
    powers_of_alpha
}

/// Lift base-field public values into extension-field elements.
pub fn extend_pv<F: Field, EF: ExtensionField<F>>(public_values: &[F]) -> Vec<EF> {
    public_values.par_iter().map(|v| EF::from_base(*v)).collect()
}

/// Packed first/last row selector constructors for sumcheck constraint evaluation.
pub mod selectors {
    use super::{
        AbstractExtensionField, AbstractField, BinomialExtensionField, BinomiallyExtendable, Field,
        PackedField, PackedValue,
    };

    /// Build a packed first-row selector for base-field constraints.
    pub fn get_first_sel_packed<F: Field, P: PackedField<Scalar = F>>(
        offset: usize,
        first_value: F,
        _total_height: usize,
        _stored_height: usize,
    ) -> P {
        P::from_fn(|i| if i == 0 && offset == 0 { first_value } else { F::zero() })
    }

    /// Build a packed last-row selector for base-field constraints.
    pub fn get_last_sel_packed<F: Field, P: PackedField<Scalar = F>>(
        offset: usize,
        last_value: F,
        total_height: usize,
        stored_height: usize,
    ) -> P {
        if stored_height < total_height {
            return P::zero();
        }
        // stored_height == total_height
        if total_height <= P::WIDTH {
            debug_assert_eq!(offset, 0);
            P::from_fn(|i| if i + 1 == total_height { last_value } else { F::zero() })
        } else if offset + P::WIDTH == total_height {
            P::from_fn(|i| if i + 1 == P::WIDTH { last_value } else { F::zero() })
        } else {
            P::zero()
        }
    }

    /// Build a packed first-row selector for extension-field constraints.
    pub fn get_first_sel_ext_packed<F: BinomiallyExtendable<D>, const D: usize>(
        offset: usize,
        first_value: BinomialExtensionField<F, D>,
        _total_height: usize,
        _stored_height: usize,
    ) -> BinomialExtensionField<F::Packing, D> {
        if offset == 0 {
            return BinomialExtensionField::from_base_fn(|coeff_idx| {
                F::Packing::from_fn(|idx_in_packing| {
                    if idx_in_packing == 0 {
                        first_value.as_base_slice()[coeff_idx]
                    } else {
                        F::zero()
                    }
                })
            });
        }
        BinomialExtensionField::zero()
    }

    /// Build a packed last-row selector for extension-field constraints.
    pub fn get_last_sel_ext_packed<F: BinomiallyExtendable<D>, const D: usize>(
        offset: usize,
        last_value: BinomialExtensionField<F, D>,
        total_height: usize,
        stored_height: usize,
    ) -> BinomialExtensionField<F::Packing, D> {
        if stored_height < total_height {
            return BinomialExtensionField::zero();
        }
        // stored_height == total_height
        if total_height <= F::Packing::WIDTH {
            debug_assert_eq!(offset, 0);
            BinomialExtensionField::from_base_fn(|coeff_idx| {
                F::Packing::from_fn(|idx_in_packing| {
                    if idx_in_packing + 1 == total_height {
                        last_value.as_base_slice()[coeff_idx]
                    } else {
                        F::zero()
                    }
                })
            })
        } else if offset + F::Packing::WIDTH == total_height {
            BinomialExtensionField::from_base_fn(|coeff_idx| {
                F::Packing::from_fn(|idx_in_packing| {
                    if idx_in_packing + 1 == F::Packing::WIDTH {
                        last_value.as_base_slice()[coeff_idx]
                    } else {
                        F::zero()
                    }
                })
            })
        } else {
            BinomialExtensionField::zero()
        }
    }
}

/// Deinterleave a slice into even-indexed and odd-indexed elements.
///
/// Returns `(evens, odds)` where `evens` contains elements at indices 0, 2, 4, ...
/// and `odds` contains elements at indices 1, 3, 5, ...
pub(crate) fn deinterleave<F: Copy>(data: &[F]) -> (Vec<F>, Vec<F>) {
    let half = data.len() / 2;
    let mut evens = Vec::with_capacity(half);
    let mut odds = Vec::with_capacity(half);
    for (i, val) in data.iter().enumerate() {
        if i % 2 == 0 {
            evens.push(*val);
        } else {
            odds.push(*val);
        }
    }
    (evens, odds)
}

/// Compute the product of all elements except the one at each index.
///
/// Given `a = [a_0, a_1, ..., a_{m-1}]`, returns a vector where
/// `result[i] = ∏_{j ≠ i} a[j]`.
pub(crate) fn products_all_but_one<T: Field>(a: &[T]) -> Vec<T> {
    let m = a.len();
    let mut result = vec![T::one(); m];

    // Prefix products: result[i] = a[0] * ... * a[i-1]
    let mut prefix = T::one();
    for i in 0..m {
        result[i] = prefix;
        prefix *= a[i];
    }

    // Suffix products: suffix = a[i+1] * ... * a[m-1]
    let mut suffix = T::one();
    for i in (0..m).rev() {
        result[i] *= suffix;
        suffix *= a[i];
    }

    result
}

/// Construct a univariate polynomial by Lagrange interpolation through `(xs[i], ys[i])`.
pub fn interpolate_univariate_polynomial<K: Field>(xs: &[K], ys: &[K]) -> UnivariatePolynomial<K> {
    let mut result = UnivariatePolynomial::new(vec![K::zero()]);
    for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
        let (denominator, numerator) = xs.iter().enumerate().filter(|(j, _)| *j != i).fold(
            (K::one(), UnivariatePolynomial::new(vec![*y])),
            |(denominator, numerator), (_, xj)| {
                (denominator * (*x - *xj), numerator.mul_by_x() + numerator * (-*xj))
            },
        );
        result += numerator * denominator.inverse();
    }
    result
}

/// Compute barycentric interpolation coefficients `a_i(r)` such that
/// `P(r) = Σ a_i(r) · y_i`, given interpolation nodes `xs` and target point `r`.
pub fn barycentric_weights<F: Field>(xs: &[F], r: &F) -> Vec<F> {
    let n = xs.len();
    assert!(n > 0);

    // Compute barycentric weights w_i = 1 / Π_{j≠i}(x_i - x_j)
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let mut prod = F::one();
        for j in 0..n {
            if i != j {
                prod *= xs[i] - xs[j];
            }
        }
        w.push(prod.inverse());
    }

    // If r matches one of the nodes, return a unit vector
    if let Some(idx) = xs.iter().position(|&xi| xi == *r) {
        let mut coeffs = vec![F::zero(); n];
        coeffs[idx] = F::one();
        return coeffs;
    }

    // Compute denominator: denom = Σ w_i / (r - x_i)
    let mut denom = F::zero();
    let mut wi_div = Vec::with_capacity(n);
    for i in 0..n {
        let inv = (*r - xs[i]).inverse(); // 1 / (r - x_i)
        let term = w[i] * inv;
        wi_div.push(term); // store numerator term before dividing
        denom += term;
    }

    // Compute final coefficients a_i(r)
    wi_div.into_iter().map(|num| num / denom).collect()
}

/// Compute a weighted sum of vectors: `result = Σ weights[i] · vectors[i]`.
pub fn linear_combination_slices<F: Field>(weights: &[F], vectors: &[&[F]]) -> Vec<F> {
    let k = weights.len();
    assert_eq!(vectors.len(), k, "Number of weights must match number of vectors");
    if k == 0 {
        return Vec::new();
    }

    let m = vectors[0].len();
    assert!(vectors.iter().all(|v| v.len() == m), "All slices must have the same length");

    // Parallel computation for each position in the result vector
    let result: Vec<F> = (0..m)
        .into_par_iter() // Iterate over positions in parallel
        .map(|i| {
            // Compute the weighted sum at this position
            let mut acc = F::zero();
            for (a, vec) in weights.iter().zip(vectors.iter()) {
                acc += *a * vec[i];
            }
            acc
        })
        .collect();

    result
}

/// Multiply an eval-form polynomial by the next unfixed eq-polynomial factor (pointwise).
///
/// Extends `poly.evals` to accommodate the increased degree before multiplying.
pub(crate) fn multiply_evals_by_eq<F: Field + PrimeField32, EF: ExtensionField<F>>(
    poly: &mut UniPolyEvals<EF>,
    eq_poly: &EqPoly<F, EF>,
) {
    let challenge = eq_poly.eq_challenges[eq_poly.eq_challenges.len() - eq_poly.num_vars_fixed - 1];

    if eq_poly.num_vars_fixed < eq_poly.num_linear_vars {
        poly.extend_to(poly.evals.len() + 1);

        for (i, v) in poly.evals.iter_mut().enumerate() {
            let x = EF::from_canonical_usize(i);
            let eq_val = (EF::one() - challenge) * (EF::one() - x) + challenge * x;
            *v *= eq_poly.eval * eq_val;
        }
    } else {
        let eq_degree = eq_poly.degree;
        poly.extend_to(poly.evals.len() + eq_degree);

        let eq_evals = EqPoly::<F, EF>::compute_eq_poly_coeffs_single(
            challenge,
            eq_poly.degree,
            &eq_poly.weights,
        );
        for (i, v) in poly.evals.iter_mut().enumerate() {
            let eq_val = if i < eq_evals.len() {
                eq_evals[i]
            } else {
                let x = EF::from_canonical_usize(i);
                EqPoly::<F, EF>::eval_eq(x, challenge, eq_poly.degree)
            };
            *v *= eq_poly.eval * eq_val;
        }
    }
}
