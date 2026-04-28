use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use rayon::current_num_threads;

// ---------------------------------------------------------------------------
// Math trait for usize
// ---------------------------------------------------------------------------

/// Extension trait that adds common math helpers to `usize`.
pub trait Math {
    /// Returns `2^self`.
    fn pow2(self) -> usize;

    /// Returns the base-2 logarithm.
    /// For powers of two this is exact; for other values it returns the ceiling.
    fn log_2(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn pow2(self) -> usize {
        2usize.pow(self as u32)
    }

    #[inline]
    fn log_2(self) -> usize {
        assert_ne!(self, 0);
        if self.is_power_of_two() {
            (1usize.leading_zeros() - self.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - self.leading_zeros()) as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination
// ---------------------------------------------------------------------------

/// Solves a system of linear equations via Gaussian elimination.
///
/// `matrix` is an augmented matrix of size **n × (n + 1)** where the last column
/// holds the right-hand-side values. The function returns the solution vector of
/// length **n**.
///
/// # Panics
///
/// Panics if the number of rows does not equal the number of columns minus one.
pub fn gaussian_elimination<F: Field>(matrix: &mut [Vec<F>]) -> Vec<F> {
    let size = matrix.len();
    assert_eq!(size, matrix[0].len() - 1);

    // Forward elimination: reduce to row-echelon form.
    for i in 0..size - 1 {
        for j in i..size - 1 {
            echelon(matrix, i, j);
        }
    }

    // Back substitution: reduce to diagonal form.
    for i in (1..size).rev() {
        eliminate(matrix, i);
    }

    // Read off the solution x_i = matrix[i][n] / matrix[i][i].
    let mut result: Vec<F> = vec![F::zero(); size];
    #[allow(clippy::needless_range_loop)]
    for i in 0..size {
        result[i] = matrix[i][size] / matrix[i][i];
    }
    result
}

/// Forward-elimination step: zeroes out column `i` in row `j + 1`.
fn echelon<F: Field>(matrix: &mut [Vec<F>], i: usize, j: usize) {
    let size = matrix.len();
    if matrix[i][i] != F::zero() {
        let factor = matrix[j + 1][i] / matrix[i][i];
        (i..size + 1).for_each(|k| {
            let tmp = matrix[i][k];
            matrix[j + 1][k] -= factor * tmp;
        });
    }
}

/// Back-substitution step: eliminates column `i` from all rows above it.
fn eliminate<F: Field>(matrix: &mut [Vec<F>], i: usize) {
    let size = matrix.len();
    if matrix[i][i] != F::zero() {
        for j in (1..=i).rev() {
            let factor = matrix[j - 1][i] / matrix[i][i];
            for k in (0..size + 1).rev() {
                let tmp = matrix[i][k];
                matrix[j - 1][k] -= factor * tmp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dot products
// ---------------------------------------------------------------------------

/// Computes the dot product of two slices in parallel.
pub fn compute_dotproduct<F: Field>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// Computes the dot product of an extension-field slice with a base-field slice.
/// Supports both same-length and packed (base-field width = EF::D × extension-field width) layouts.
pub fn compute_dotproduct_mix<F: Field, EF: ExtensionField<F>>(a: &[EF], b: &[F]) -> EF {
    let num_cpus = current_num_threads();
    let total_len = a.len();

    let chunk_size = std::cmp::max(1024, total_len / (num_cpus * 4));
    if total_len == b.len() {
        a.par_chunks(chunk_size)
            .zip_eq(b.par_chunks(chunk_size))
            .map(|(a_chunk, b_chunk)| {
                a_chunk
                    .iter()
                    .zip(b_chunk.iter())
                    .map(|(a_i, b_i)| *a_i * *b_i)
                    .sum::<EF>()
            })
            .sum::<EF>()
    } else {
        a.par_chunks(chunk_size)
            .zip_eq(b.par_chunks(chunk_size * EF::D))
            .map(|(a_chunk, b_chunk)| {
                b_chunk
                    .chunks(EF::D)
                    .zip(a_chunk.iter())
                    .map(|(a_i, b_i)| *b_i * EF::from_base_slice(a_i))
                    .sum::<EF>()
            })
            .sum::<EF>()
    }
}

// ---------------------------------------------------------------------------
// Power-of-two check
// ---------------------------------------------------------------------------

/// Checks if `num` is a power of 2 (and greater than 1).
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && num != 1 && (num & (num - 1)) == 0
}
