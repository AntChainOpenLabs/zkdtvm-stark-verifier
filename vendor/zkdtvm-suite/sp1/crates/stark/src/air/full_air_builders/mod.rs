//! Builder implementations for different phases of sumcheck-based proving.
//!
//! This module provides builder types corresponding to the different phases:
//!
//! - [`precompute`]: `PrecomputeRowBuilder` for computing linear combinations
//! - [`permutation`]: `PermutationRowBuilder` for generating permutation traces
//! - [`evaluator`]: `ConstraintFolder` for constraint evaluation
//! - [`symbloic`]: `SymbolicAirBuilder` for symbolic degree analysis

pub mod evaluator;
pub mod permutation;
pub mod precompute;
pub mod symbloic;

use p3_field::Field;
use p3_matrix::dense::DenseMatrix;

use super::full_air::PairCol;

/// Collect reserved polynomial values from main and preprocessed traces.
///
/// Given slices of main and preprocessed trace rows, and a list of column
/// references, returns a vector containing the values at those columns.
pub fn collect_reserved_poly<F: Field>(
    main: &[F],
    prep: &[F],
    reserved_poly: &[PairCol],
) -> Vec<F> {
    reserved_poly
        .iter()
        .map(|i| match i {
            PairCol::Prep(i) => prep[*i],
            PairCol::Main(i) => main[*i],
        })
        .collect()
}

/// Get a row slice from the preprocessed trace, or an empty slice if none.
///
/// This helper function safely handles the case where there is no preprocessed
/// trace, returning an empty slice instead of panicking.
pub fn get_preprocessed_row<'a, F: Field>(
    preprocessed: Option<&'a DenseMatrix<F>>,
    index: usize,
) -> &'a [F] {
    preprocessed
        .map(|p| {
            let start = index * p.width;
            &p.values[start..(start + p.width)]
        })
        .unwrap_or(&[])
}
