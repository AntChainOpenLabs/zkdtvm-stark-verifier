use crate::utils::math::{compute_dotproduct, is_power_of_two};
use itertools::Itertools;
use p3_maybe_rayon::prelude::*;
use std::slice::IterMut;

use crate::utils::eqpoly::EqPolynomial;
use p3_field::ExtensionField;
use p3_field::{AbstractExtensionField, AbstractField, Field, PackedValue};
use p3_matrix::compressed::{CompressedMatrix, PaddingRow};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use rand::distributions::{Distribution, Standard};
use rand::random;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, MulAssign, Neg, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MultilinearPolynomial<F> {
    pub evals: Vec<F>,
    num_vars: usize,
}

impl<F: Field> Default for MultilinearPolynomial<F> {
    fn default() -> Self {
        MultilinearPolynomial::zero()
    }
}

impl<F: Field> MultilinearPolynomial<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let num_vars = if evals.is_empty() {
            0
        } else {
            let num_vars = evals.len().ilog2() as usize;
            assert!(
                evals.len() == 1 || is_power_of_two(evals.len()),
                "Dense multi-linear polynomials must be made from a power of 2 (not {})",
                evals.len()
            );
            num_vars
        };

        MultilinearPolynomial { num_vars, evals }
    }

    pub const fn zero() -> Self {
        MultilinearPolynomial {
            num_vars: 0,
            evals: Vec::new(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num_vars == 0
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        1 << self.num_vars
    }

    pub fn is_empty(&self) -> bool {
        self.evals.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.evals.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, F> {
        self.evals.iter_mut()
    }

    pub fn evals(&self) -> Vec<F> {
        self.evals.clone()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.evals.as_ref()
    }

    /// Bind the first variable (MSB of the index) to `r`, halving the polynomial size.
    ///
    /// For each pair `(evals[i], evals[i + n])`, computes the linear interpolation
    /// `evals[i] + r · (evals[i + n] - evals[i])` and stores it in `evals[i]`.
    /// After this call, `num_vars` decreases by 1 and the evaluation table is truncated.
    ///
    /// This is "big-endian" folding: it splits the evaluations into two halves
    /// (low indices = x₀=0, high indices = x₀=1) and interpolates.
    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.evals.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.evals.truncate(n);
    }

    /// Bind the last variable (LSB of the index) to `r`, halving the polynomial size.
    ///
    /// Folds even-indexed and odd-indexed evaluations:
    /// `result[i] = evals[2i] + r · (evals[2i+1] - evals[2i])`.
    /// After this call, `num_vars` decreases by 1.
    ///
    /// This is "little-endian" folding: it interleaves even/odd positions
    /// (even = xₙ₋₁=0, odd = xₙ₋₁=1) and interpolates.
    pub fn bound_poly_var_bottom(&mut self, r: &F) {
        let n = self.len() / 2;

        for i in 0..n {
            let even = self.evals[2 * i];
            let odd = self.evals[2 * i + 1];
            self.evals[i] = even + *r * (odd - even);
        }

        self.num_vars -= 1;
        self.evals.truncate(n);
    }

    pub fn par_clone_matrix(mat: &RowMajorMatrix<F>) -> RowMajorMatrix<F>
    where
        F: Send + Sync + Clone,
    {
        let values = mat.values.par_iter().cloned().collect();
        RowMajorMatrix::new(values, mat.width())
    }

    /// Evaluate this base-field multilinear polynomial at an extension-field point `r`.
    ///
    /// Uses the EQ polynomial identity: `f(r) = Σ_x f(x) · eq(r, x)` where the sum
    /// ranges over all Boolean hypercube points `x ∈ {0,1}^n`.
    pub fn evaluate_mix<EF: ExtensionField<F>>(&self, r: &[EF]) -> EF {
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.evals.len());
        let z: Vec<EF> = self.evals.iter().cloned().map(Into::into).collect_vec();
        compute_dotproduct(&z, &chis)
    }

    /// Combines column polynomials from multiple matrices into a single polynomial
    /// using random coefficients (e.g. powers of alpha).
    ///
    /// Each column of each matrix is treated as a multilinear polynomial.
    /// The result accumulates: `result[row] += Σ_mat Σ_col coefficients[mat][col] * matrix[mat][row, col]`
    pub fn random_linear_combine_columns<EF: ExtensionField<F>>(
        matrices: Vec<&RowMajorMatrix<F>>,
        coefficients: &[Vec<EF>],
        result: &mut [EF],
    ) {
        for (mat_idx, mat) in matrices.iter().enumerate() {
            Self::random_linear_combine_single_matrix_columns(mat, &coefficients[mat_idx], result);
        }
    }

    /// Combines column polynomials from a single matrix into a polynomial
    /// using random coefficients.
    ///
    /// Accumulates: `result[row] += Σ_col coeff[col] * matrix[row, col]`
    pub fn random_linear_combine_single_matrix_columns<EF: ExtensionField<F>>(
        polys: &RowMajorMatrix<F>,
        coeff: &[EF],
        result: &mut [EF],
    ) {
        let height = polys.height();
        if polys.width == 0 || polys.height() == 0 {
            return;
        }

        assert!(height != 0 && (height & (height - 1)) == 0);
        assert!(F::Packing::WIDTH != 0 && (F::Packing::WIDTH & (F::Packing::WIDTH - 1)) == 0);

        let width = polys.width();
        let is_from_ef = coeff.len() != width;
        if is_from_ef {
            debug_assert_eq!(width, coeff.len() * EF::D);
        } else {
            debug_assert_eq!(width, coeff.len());
        }

        if height <= F::Packing::WIDTH {
            result.par_iter_mut().enumerate().for_each(|(idx, item)| {
                *item += if is_from_ef {
                    polys
                        .row_slice(idx)
                        .chunks(EF::D)
                        .zip_eq(coeff.iter())
                        .map(|(poly_val, coeff_val)| EF::from_base_slice(poly_val) * *coeff_val)
                        .fold(EF::zero(), |acc: EF, val: EF| acc + val)
                } else {
                    (0..polys.width())
                        .map(|poly_idx| coeff[poly_idx] * polys.values[idx * width + poly_idx])
                        .fold(EF::zero(), |acc: EF, val: EF| acc + val)
                };
            });
            return;
        }

        result
            .par_chunks_mut(F::Packing::WIDTH)
            .enumerate()
            .for_each(|(i_start, chunk)| {
                let i_start = i_start * F::Packing::WIDTH;
                let packed_vals = polys
                    .vertically_packed_row::<F::Packing>(i_start)
                    .collect_vec();
                let mut packed_result = EF::ExtensionPacking::zero();
                if is_from_ef {
                    packed_vals.chunks(EF::D).zip_eq(coeff.iter()).for_each(
                        |(poly_vals, coeff_val)| {
                            let scale = EF::ExtensionPacking::from_base_fn(|i| {
                                F::Packing::from(coeff_val.as_base_slice()[i])
                            });
                            let val = EF::ExtensionPacking::from_base_slice(poly_vals);
                            packed_result += scale * val;
                        },
                    );
                } else {
                    for j in 0..width {
                        let scale = EF::ExtensionPacking::from_base_fn(|i| {
                            F::Packing::from(coeff[j].as_base_slice()[i])
                        });
                        packed_result += scale * packed_vals[j];
                    }
                }

                for i in 0..F::Packing::WIDTH {
                    if i_start + i < height {
                        chunk[i] +=
                            EF::from_base_fn(|j| packed_result.as_base_slice()[j].as_slice()[i]);
                    }
                }
            });
    }

    /// Combines column polynomials from multiple CompressedMatrix into a single polynomial
    /// using random coefficients.
    ///
    /// Like `random_linear_combine_columns` but handles CompressedMatrix efficiently:
    /// stored (non-padding) rows are processed directly, and padding rows contribute
    /// a precomputed constant per row.
    pub fn random_linear_combine_columns_compressed<EF: ExtensionField<F>>(
        matrices: Vec<&CompressedMatrix<F>>,
        coefficients: &[Vec<EF>],
        result: &mut [EF],
    ) {
        for (mat_idx, mat) in matrices.iter().enumerate() {
            Self::random_linear_combine_single_compressed_matrix_columns(
                mat,
                &coefficients[mat_idx],
                result,
            );
        }
    }

    /// Combines column polynomials from a single CompressedMatrix into a polynomial
    /// using random coefficients.
    ///
    /// For stored rows (row < stored_height): accumulates normally from main data.
    /// For padding rows (row >= stored_height): accumulates the constant padding contribution.
    pub fn random_linear_combine_single_compressed_matrix_columns<EF: ExtensionField<F>>(
        compressed: &CompressedMatrix<F>,
        coeff: &[EF],
        result: &mut [EF],
    ) {
        let total_height = compressed.height();
        let stored_height = compressed.stored_height();
        let main = &compressed.main;
        let width = main.width();

        if width == 0 || total_height == 0 {
            return;
        }

        let is_from_ef = coeff.len() != width;
        if is_from_ef {
            debug_assert_eq!(width, coeff.len() * EF::D);
        } else {
            debug_assert_eq!(width, coeff.len());
        }

        // Process stored rows directly from main data
        if stored_height > 0 {
            let main_values = &main.values;
            result[..stored_height]
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, item)| {
                    *item += if is_from_ef {
                        main_values[idx * width..(idx + 1) * width]
                            .chunks(EF::D)
                            .zip(coeff.iter())
                            .map(|(poly_val, coeff_val)| EF::from_base_slice(poly_val) * *coeff_val)
                            .fold(EF::zero(), |acc: EF, val: EF| acc + val)
                    } else {
                        (0..width)
                            .map(|poly_idx| coeff[poly_idx] * main_values[idx * width + poly_idx])
                            .fold(EF::zero(), |acc: EF, val: EF| acc + val)
                    };
                });
        }

        // Process padding rows: compute the constant padding contribution once
        if stored_height < total_height {
            let padding_contribution: EF = match &compressed.padding_row {
                PaddingRow::None => EF::zero(),
                PaddingRow::Zero { .. } => EF::zero(),
                PaddingRow::Constant { value, .. } => {
                    if is_from_ef {
                        let ef_width = width / EF::D;
                        let base_val = EF::from_base(*value);
                        (0..ef_width)
                            .zip(coeff.iter())
                            .map(|(_, c)| base_val * *c)
                            .fold(EF::zero(), |acc, val| acc + val)
                    } else {
                        coeff
                            .iter()
                            .map(|c| *c * *value)
                            .fold(EF::zero(), |acc, val| acc + val)
                    }
                }
                PaddingRow::General(values) => {
                    if is_from_ef {
                        values
                            .chunks(EF::D)
                            .zip(coeff.iter())
                            .map(|(chunk, c)| EF::from_base_slice(chunk) * *c)
                            .fold(EF::zero(), |acc, val| acc + val)
                    } else {
                        values
                            .iter()
                            .zip(coeff.iter())
                            .map(|(&v, c)| *c * v)
                            .fold(EF::zero(), |acc, val| acc + val)
                    }
                }
            };

            result[stored_height..total_height]
                .par_iter_mut()
                .for_each(|val| {
                    *val += padding_contribution;
                });
        }
    }
}

impl<F: Field> Clone for MultilinearPolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.evals[0..self.evals.len()].to_vec())
    }
}

impl<F: Field> AsRef<MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn as_ref(&self) -> &MultilinearPolynomial<F> {
        self
    }
}

impl<F: Field> AddAssign<&MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, rhs: &MultilinearPolynomial<F>) {
        if rhs.is_zero() {
            return;
        }
        assert_eq!(self.num_vars, rhs.num_vars);
        assert_eq!(self.evals.len(), rhs.evals.len());
        let summed_evaluations: Vec<F> = self
            .evals
            .iter()
            .zip(&rhs.evals)
            .map(|(a, b)| *a + *b)
            .collect();

        *self = Self {
            num_vars: self.num_vars,
            evals: summed_evaluations,
        }
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a, F: Field> AddAssign<(&'a F, &'a MultilinearPolynomial<F>)> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, (f, other): (&'a F, &'a MultilinearPolynomial<F>)) {
        let other = Self {
            num_vars: other.num_vars,
            evals: other.evals.iter().map(|x| *f * *x).collect(),
        };
        *self = &*self + &other;
    }
}

impl<F: Field> MulAssign<&F> for MultilinearPolynomial<F> {
    fn mul_assign(&mut self, rhs: &F) {
        let summed_evaluations: Vec<F> = self.evals.iter().map(|a| *a * *rhs).collect();

        *self = Self {
            num_vars: self.num_vars,
            evals: summed_evaluations,
        }
    }
}

impl<F: Field> Add<&MultilinearPolynomial<F>> for &MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn add(self, rhs: &MultilinearPolynomial<F>) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return rhs.clone();
        }
        assert_eq!(self.num_vars, rhs.num_vars);
        let result: Vec<F> = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Self::Output::new(result)
    }
}

impl<F: Field> Neg for MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn neg(self) -> MultilinearPolynomial<F> {
        Self {
            num_vars: self.num_vars,
            evals: self.evals.iter().map(|x| -*x).collect(),
        }
    }
}

impl<F: Field> Sub<&MultilinearPolynomial<F>> for &MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn sub(self, rhs: &MultilinearPolynomial<F>) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return rhs.clone().neg();
        }
        assert_eq!(self.num_vars, rhs.num_vars);
        let result = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            .map(|(a, b)| *a - *b)
            .collect();

        Self::Output::new(result)
    }
}

impl<F: Field> SubAssign for MultilinearPolynomial<F> {
    fn sub_assign(&mut self, other: Self) {
        *self = &*self - &other;
    }
}

impl<'a, F: Field> SubAssign<&'a MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sub_assign(&mut self, other: &'a MultilinearPolynomial<F>) {
        *self = &*self - other;
    }
}

impl<'a, F: Field> Sum<&'a MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sum<I: Iterator<Item = &'a MultilinearPolynomial<F>>>(
        mut iter: I,
    ) -> MultilinearPolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => lhs + rhs,
            (Some(lhs), None) => return lhs.clone(),
            _ => unreachable!(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += poly;
            acc
        })
    }
}

impl<F: Field> Sum<MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sum<I: Iterator<Item = MultilinearPolynomial<F>>>(iter: I) -> MultilinearPolynomial<F> {
        iter.reduce(|mut acc, poly| {
            acc += &poly;
            acc
        })
        .unwrap()
    }
}

impl<F: Field> Deref for MultilinearPolynomial<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.evals[..]
    }
}

impl<F: Field> DerefMut for MultilinearPolynomial<F> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.evals[..]
    }
}

pub trait MultilinearExtension<F: Field>:
    Clone + Debug + for<'a> AddAssign<(&'a F, &'a Self)>
{
    type Point: Clone + Debug;

    fn from_evals(evals: Vec<F>) -> Self;

    fn into_evals(self) -> Vec<F>;

    fn evals(&self) -> &[F];

    fn evaluate(&self, point: &[F]) -> Option<F>;

    fn rand<R: Rng>(num_vars: usize, rng: &mut R) -> Self;
}

impl<F: Field> MultilinearExtension<F> for MultilinearPolynomial<F>
where
    Standard: Distribution<F>,
{
    type Point = Vec<F>;

    fn from_evals(evals: Vec<F>) -> Self {
        Self::new(evals)
    }

    fn into_evals(self) -> Vec<F> {
        self.evals
    }

    fn evals(&self) -> &[F] {
        self.evals.as_slice()
    }

    fn rand<R: Rng>(num_vars: usize, _rng: &mut R) -> Self {
        Self::from_evals((0..(1 << num_vars)).map(|_| random()).collect())
    }

    fn evaluate(&self, point: &[F]) -> Option<F> {
        if point.len() == self.num_vars {
            Some(fix_last_variables(self, point)[0])
        } else {
            None
        }
    }
}

macro_rules! impl_index {
    (@ $name:ty, $field:tt, [$($range:ty => $output:ty),*$(,)?]) => {
        $(
            impl<F: Field> std::ops::Index<$range> for $name {
                type Output = $output;

                fn index(&self, index: $range) -> &$output {
                    self.$field.index(index)
                }
            }

            impl<F: Field> std::ops::IndexMut<$range> for $name {
                fn index_mut(&mut self, index: $range) -> &mut $output {
                    self.$field.index_mut(index)
                }
            }
        )*
    };
    (@ $name:ty, $field:tt) => {
        impl_index!(
            @ $name, $field,
            [
                usize => F,
                std::ops::Range<usize> => [F],
                std::ops::RangeFrom<usize> => [F],
                std::ops::RangeFull => [F],
                std::ops::RangeInclusive<usize> => [F],
                std::ops::RangeTo<usize> => [F],
                std::ops::RangeToInclusive<usize> => [F],
            ]
        );
    };
    ($name:ident, $field:tt) => {
        impl_index!(@ $name<F>, $field);
    };
}

impl_index!(MultilinearPolynomial, evals);

/// Bind the last (highest-indexed) variable of a multilinear polynomial to `point`.
///
/// Given evaluations over `{0,1}^nv`, produces evaluations over `{0,1}^{nv-1}` by
/// interpolating: `result[b] = data[b] + point · (data[b + 2^{nv-1}] - data[b])`.
fn fix_last_variable<F: Field>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let half_len = 1 << (nv - 1);
    let mut res = vec![F::zero(); half_len];

    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i] + (data[i + half_len] - data[i]) * *point;
    });

    res
}

/// Sequentially bind the last variables of a multilinear polynomial to the given partial point.
///
/// Applies `fix_last_variable` for each coordinate in `partial_point` (in reverse order),
/// reducing the polynomial from `n` variables to `n - partial_point.len()` variables.
/// This is equivalent to evaluating the polynomial at a partial assignment of its
/// highest-indexed variables.
pub fn fix_last_variables<F: Field>(
    poly: &MultilinearPolynomial<F>,
    partial_point: &[F],
) -> MultilinearPolynomial<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evals.to_vec();
    let dim = partial_point.len();

    for (i, point) in partial_point.iter().rev().enumerate().take(dim) {
        poly = fix_last_variable(&poly, nv - i, point);
    }

    MultilinearPolynomial::new(poly[..(1 << (nv - dim))].to_vec())
}
