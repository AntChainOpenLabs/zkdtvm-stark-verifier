//! Permutation phase builder for generating permutation traces.
//!
//! This module provides [`PermutationRowBuilder`] which implements [`FullAirBuilder`]
//! for the permutation trace generation phase, where lookup multiplicities are
//! recorded and running products are computed.

use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use super::{
    super::{FullAir, FullAirBuilder},
    collect_reserved_poly, get_preprocessed_row,
};

/// Compute the width of the permutation trace.
///
/// The permutation trace has one column per batch of lookups, plus one
/// column for the cumulative sum.
pub const fn local_permutation_trace_width(num_lookup: usize, batch_size: usize) -> usize {
    if num_lookup == 0 {
        return 0;
    }
    num_lookup.div_ceil(batch_size) + 1
}

/// Builder for the permutation trace generation phase.
///
/// This builder is used during permutation trace generation to record
/// lookup multiplicities and compute running products for the permutation argument.
///
/// # Type Parameters
///
/// * `F` - The base field
/// * `EF` - The extension field
pub struct PermutationRowBuilder<'a, F: Field, EF: Field + ExtensionField<F>> {
    /// Reserved polynomial values for the current row.
    pub reserved_poly_row: &'a [F],
    /// Precomputed linear combination values for the current row.
    pub precomputed_lc_row: &'a [EF],
    /// Powers of the beta challenge.
    pub beta_powers: &'a [EF],
    /// The alpha challenge.
    pub alpha: EF,
    /// Output row for storing running products.
    pub row: &'a mut [EF],
    /// Number of lookups per batch.
    pub batch_size: usize,
    /// Total number of lookups.
    pub num_lookup: usize,
    /// Cached multiplicities from send/recv calls.
    pub cached_multiplicitys: Vec<F>,
}

impl<'a, F: Field, EF: Field + ExtensionField<F>> PermutationRowBuilder<'a, F, EF> {
    /// Compute the running products from cached multiplicities.
    ///
    /// This method is called after all lookup operations have been recorded
    /// via `send` and `recv`. It computes the running product values for
    /// each batch of lookups.
    pub fn finalize(&mut self) {
        assert!(self.cached_multiplicitys.len() == self.num_lookup);
        for (row, (chunk_multiplicity, chunk_value)) in self.row.iter_mut().zip(
            self.cached_multiplicitys
                .chunks(self.batch_size)
                .zip(self.precomputed_lc_row.chunks(self.batch_size)),
        ) {
            *row = chunk_multiplicity
                .iter()
                .copied()
                .zip(chunk_value.iter().copied())
                .map(|(multiplicity, value)| value.inverse() * multiplicity)
                .sum()
        }
    }
}

impl<'a, F: Field, EF: Field + ExtensionField<F>> FullAirBuilder
    for PermutationRowBuilder<'a, F, EF>
{
    type F = F;
    type EF = EF;
    type VarBase = F;
    type VarMaybeExt = F;
    type VarExt = EF;
    type MatMaybeExt = RowMajorMatrixView<'a, Self::VarMaybeExt>;
    type MatExt = RowMajorMatrixView<'a, Self::VarExt>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        unreachable!("preprocessed should not be used in permutation trace generation phase")
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        unreachable!("main should not be used in permutation trace generation phase")
    }

    fn public(&self) -> &[Self::VarBase] {
        unreachable!("public should not be used in permutation trace generation phase")
    }

    fn alpha(&self) -> Self::VarExt {
        self.alpha
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        self.beta_powers
    }

    fn beta_septix(&self) -> Self::VarExt {
        unreachable!("beta_septix should not be used in permutation trace generation phase")
    }

    fn retain_precomputed(&mut self, _: Self::VarExt) {
        unreachable!(
            "retain_precomputed should not be called in permutation trace generation phase"
        )
    }

    fn precomputed(&self) -> Self::MatExt {
        RowMajorMatrixView::new(self.precomputed_lc_row, self.precomputed_lc_row.len())
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        RowMajorMatrixView::new(self.reserved_poly_row, self.reserved_poly_row.len())
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        let multiplicity = if is_send { multiplicity } else { multiplicity.neg() };
        self.cached_multiplicitys.push(multiplicity);
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_first_row should not be used in permutation trace generation phase")
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_last_row should not be used in permutation trace generation phase")
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        unreachable!("is_transition should not be used in permutation trace generation phase")
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        ef
    }
    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, _: I) {
        unreachable!("assert_zero should not be called in permutation trace generation phase")
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, _: I) {
        unreachable!("assert_zero_ext should not be called in permutation trace generation phase")
    }
}

/// Generate the permutation trace for an AIR.
///
/// This function processes each row of the main trace, calling `air.lookup()`
/// to record multiplicities, then computes running products and cumulative sums.
///
/// # Arguments
///
/// * `air` - The AIR implementation
/// * `preprocessed` - Optional preprocessed trace
/// * `main` - Main trace
/// * `precomputed_lc` - Precomputed linear combinations
/// * `alpha` - Alpha challenge
/// * `beta_powers` - Powers of beta challenge
/// * `batch_size` - Number of lookups per batch
/// * `num_lookup` - Total number of lookups
///
/// # Returns
///
/// A tuple containing:
/// * The permutation trace matrix
/// * The local cumulative sum
pub fn generate_permutation_trace_<
    AIR: for<'a> FullAir<PermutationRowBuilder<'a, F, EF>>,
    F: Field,
    EF: ExtensionField<F>,
>(
    air: &AIR,
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    precomputed_lc: &RowMajorMatrix<EF>,
    alpha: EF,
    beta_powers: &[EF],
    batch_size: usize,
    num_lookup: usize,
) -> (RowMajorMatrix<EF>, EF) {
    let reserved_poly = air.reserved_poly();
    let height = precomputed_lc.height() - 1;
    assert!(air.required_max_beta_power() < beta_powers.len());
    assert!(main.height() == height);
    // // BATCH_SIZE must evenly divide NUM_LOOKUP for correct permutation trace generation
    // assert!(
    //     num_lookup % batch_size == 0,
    //     "NUM_LOOKUP ({}) must be divisible by BATCH_SIZE ({})",
    //     num_lookup,
    //     batch_size
    // );
    let permutation_trace_width = local_permutation_trace_width(num_lookup, batch_size);
    let mut permutation_trace = RowMajorMatrix::new(
        vec![EF::zero(); permutation_trace_width * (height + 1)],
        permutation_trace_width,
    );

    // Generate running products for each row
    permutation_trace.par_rows_mut().take(height).enumerate().for_each(|(index, row)| {
        use std::ops::Deref;
        let precomputed_lc_binding = precomputed_lc.row_slice(index);
        let precomputed_lc_row: &[_] = precomputed_lc_binding.deref();
        let main_binding = main.row_slice(index);
        let main: &[_] = main_binding.deref();
        let prep: &[_] = get_preprocessed_row(preprocessed, index);
        let reserved_poly_row = collect_reserved_poly(main, prep, &reserved_poly);
        let mut builder = PermutationRowBuilder {
            reserved_poly_row: &reserved_poly_row[..],
            precomputed_lc_row,
            beta_powers,
            alpha,
            row,
            batch_size,
            num_lookup,
            cached_multiplicitys: vec![],
        };
        air.lookup(&mut builder);
        builder.finalize();
    });

    // Copy first row to the extra row for cyclic consistency
    use std::ops::Deref;
    let first_row: Vec<_> = {
        let first_row_binding = permutation_trace.row_slice(0);
        first_row_binding.deref().to_vec()
    };
    for i in 0..permutation_trace_width {
        permutation_trace.row_mut(height)[i] = first_row[i];
    }

    // Compute cumulative sums for each row
    let local_cumulative_sums: Vec<EF> = permutation_trace
        .par_row_slices()
        .take(height)
        .map(|row| row[0..permutation_trace_width - 1].iter().copied().sum::<EF>())
        .collect();

    // Compute prefix sum (serial scan as we don't have rayon_scan)
    let local_cumulative_sums: Vec<EF> = local_cumulative_sums
        .iter()
        .scan(EF::zero(), |acc, x| {
            *acc = *acc + *x;
            Some(*acc)
        })
        .collect();

    let local_cumulative_sum = *local_cumulative_sums.last().unwrap();
    let first_prefix_sum = local_cumulative_sums[0];

    // Fill in the cumulative sum column
    permutation_trace
        .par_rows_mut()
        .take(height)
        .zip_eq(local_cumulative_sums.into_par_iter())
        .for_each(|(row, prefix_sum)| {
            row[permutation_trace_width - 1] = prefix_sum;
        });
    permutation_trace.row_mut(height)[permutation_trace_width - 1] = first_prefix_sum;
    (permutation_trace, local_cumulative_sum)
}
