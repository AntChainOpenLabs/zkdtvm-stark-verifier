//! Permutation phase builder for generating permutation traces.
//!
//! This module provides [`PermutationRowBuilder`] which implements [`FullAirBuilder`]
//! for the permutation trace generation phase, where lookup multiplicities are
//! recorded and running products are computed.

use dt_stark::air::{FullAir, FullAirBuilder, PairCol};
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    compressed::{padding_row_sum, padding_row_to_base_vec, CompressedMatrix, PaddingRow},
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use crate::evaluator::uinit_vec;

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
            let mut values = Vec::with_capacity(chunk_value.len());
            let mut multiplicities = Vec::with_capacity(chunk_multiplicity.len());
            for (multiplicity, value) in
                chunk_multiplicity.iter().copied().zip(chunk_value.iter().copied())
            {
                if multiplicity.is_zero() {
                    continue;
                }
                values.push(value);
                multiplicities.push(multiplicity);
            }
            let inverses = p3_field::batch_multiplicative_inverse(&values);
            *row = inverses
                .into_iter()
                .zip(multiplicities.into_iter())
                .map(|(inverse, multiplicity)| inverse * multiplicity)
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
        unreachable!("perm_alpha should not be used in permutation multiplicity")
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        unreachable!("beta_powers should not be used in permutation multiplicity")
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
        unreachable!("precomputed should not be used in permutation trace generation phase")
    }

    /// we only need to compute multiplicity and we can only use reserved_poly.
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

/// T-K fused gather: one row-parallel pass over (prep, main) producing the
/// precompute-LC, reserved, and permutation rows together — the three
/// separate full passes each re-read every main trace from DRAM; fusing keeps
/// the row hot in cache and writes all three outputs at once. Output bytes
/// are identical to the three-pass form by construction (same builders, same
/// per-row order).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fused_precompute_reserved_permutation<AIR, F, EF>(
    air: &AIR,
    preprocessed: Option<&CompressedMatrix<F, F>>,
    main: &CompressedMatrix<F, F>,
    public: &[F],
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    num_precompute: usize,
    reserved_pairs: &[PairCol],
    batch_size: usize,
    num_lookup: usize,
) -> (CompressedMatrix<EF, EF>, CompressedMatrix<F, F>, CompressedMatrix<EF, EF>, EF)
where
    AIR: for<'a> FullAir<crate::precompute::PrecomputeRowBuilder<'a, F, F, EF>>
        + for<'a> FullAir<PermutationRowBuilder<'a, F, EF>>,
    F: Field,
    EF: ExtensionField<F>,
{
    let height = main.stored_height();
    let total_height = main.total_height;
    if let Some(prep) = preprocessed {
        assert_eq!(prep.stored_height(), height);
    }
    let perm_width = num_lookup.div_ceil(batch_size);
    let reserved_width = reserved_pairs.len();

    let mut precompute = RowMajorMatrix::new(uinit_vec(num_precompute * height), num_precompute);
    let mut reserved = RowMajorMatrix::new(uinit_vec::<F>(reserved_width * height), reserved_width);
    let mut perm = RowMajorMatrix::new(uinit_vec(perm_width * height), perm_width);

    let prep = preprocessed.unwrap_or(main);
    precompute
        .values
        .par_chunks_mut(num_precompute.max(1))
        .zip_eq(reserved.values.par_chunks_mut(reserved_width.max(1)))
        .zip_eq(perm.values.par_chunks_mut(perm_width.max(1)))
        .enumerate()
        .for_each(|(index, ((precompute_row, reserved_row), perm_row))| {
            let binding = main.main.row_slice(index);
            let main_row: &[F] = binding.as_ref();
            let binding = prep.main.row_slice(index);
            let prep_row: &[F] = binding.as_ref();

            let mut builder = crate::precompute::PrecomputeRowBuilder::<F, F, EF> {
                preprocessed: prep_row,
                main: main_row,
                beta_powers,
                public,
                alpha,
                beta_septix,
                row: precompute_row,
                col_index: 0,
            };
            air.precompute_lc(&mut builder);

            for (pair, slot) in reserved_pairs.iter().zip(reserved_row.iter_mut()) {
                *slot = match pair {
                    PairCol::Prep(i) => prep_row[*i],
                    PairCol::Main(i) => main_row[*i],
                };
            }

            let mut builder = PermutationRowBuilder {
                reserved_poly_row: &reserved_row[..],
                precomputed_lc_row: &precompute_row[..],
                beta_powers,
                alpha,
                row: perm_row,
                batch_size,
                num_lookup,
                cached_multiplicitys: vec![],
            };
            air.lookup(&mut builder);
            builder.finalize();
        });

    let padding_count = total_height - height;
    let (precompute_padding, reserved_padding, perm_padding) = if padding_count != 0 {
        let main_padding = padding_row_to_base_vec(&main.padding_row);
        let prep_padding = padding_row_to_base_vec(&prep.padding_row);

        let mut precompute_pad = uinit_vec(num_precompute);
        let mut builder = crate::precompute::PrecomputeRowBuilder::<F, F, EF> {
            preprocessed: &prep_padding,
            main: &main_padding,
            beta_powers,
            public,
            alpha,
            beta_septix,
            row: &mut precompute_pad,
            col_index: 0,
        };
        air.precompute_lc(&mut builder);

        let reserved_pad: Vec<F> = reserved_pairs
            .iter()
            .map(|pair| match pair {
                PairCol::Prep(i) => prep_padding[*i],
                PairCol::Main(i) => main_padding[*i],
            })
            .collect();

        let mut perm_pad = uinit_vec(perm_width);
        let mut builder = PermutationRowBuilder {
            reserved_poly_row: &reserved_pad[..],
            precomputed_lc_row: &precompute_pad[..],
            beta_powers,
            alpha,
            row: &mut perm_pad,
            batch_size,
            num_lookup,
            cached_multiplicitys: vec![],
        };
        air.lookup(&mut builder);
        builder.finalize();

        (
            PaddingRow::General(precompute_pad),
            PaddingRow::General(reserved_pad),
            PaddingRow::General(perm_pad),
        )
    } else {
        (PaddingRow::None, PaddingRow::None, PaddingRow::None)
    };

    let perm_main_sum: EF = perm.values.iter().copied().sum();
    let pad_row_sum: EF = padding_row_sum(&perm_padding);
    let local_cumulative_sum =
        perm_main_sum + pad_row_sum * EF::from_canonical_usize(padding_count);

    (
        CompressedMatrix::new(precompute, precompute_padding, total_height),
        CompressedMatrix::new(reserved, reserved_padding, total_height),
        CompressedMatrix::new(perm, perm_padding, total_height),
        local_cumulative_sum,
    )
}

pub fn generate_permutation_trace_<
    AIR: for<'a> FullAir<PermutationRowBuilder<'a, F, EF>>,
    F: Field,
    EF: ExtensionField<F>,
>(
    air: &AIR,
    reserved_poly: &CompressedMatrix<F, F>,
    precomputed_lc: &CompressedMatrix<EF, EF>,
    alpha: EF,
    beta_powers: &[EF],
    batch_size: usize,
    num_lookup: usize,
) -> (CompressedMatrix<EF, EF>, EF) {
    let height = precomputed_lc.stored_height();
    let permutation_trace_width = num_lookup.div_ceil(batch_size);
    let mut permutation_trace =
        RowMajorMatrix::new(uinit_vec(permutation_trace_width * height), permutation_trace_width);
    let padding_count = reserved_poly.total_height - reserved_poly.stored_height();

    // Generate running products for each row
    permutation_trace.par_rows_mut().enumerate().for_each(|(index, row)| {
        let binding = precomputed_lc.main.row_slice(index);
        let precomputed_lc_row: &[_] = binding.as_ref();
        let binding = reserved_poly.main.row_slice(index);
        let reserved_poly_row: &[_] = binding.as_ref();

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

    let padding_row = if padding_count != 0 {
        let reserved_poly_padding = padding_row_to_base_vec(&reserved_poly.padding_row);
        let precomputed_lc_padding = padding_row_to_base_vec(&precomputed_lc.padding_row);
        let mut padding_row = uinit_vec(permutation_trace_width);

        let mut builder = PermutationRowBuilder {
            reserved_poly_row: &reserved_poly_padding[..],
            precomputed_lc_row: &precomputed_lc_padding[..],
            beta_powers,
            alpha,
            row: &mut padding_row,
            batch_size,
            num_lookup,
            cached_multiplicitys: vec![],
        };
        air.lookup(&mut builder);
        builder.finalize();
        PaddingRow::General(padding_row)
    } else {
        PaddingRow::None
    };

    let perm_main_sum: EF = permutation_trace.values.iter().copied().sum();
    let pad_row_sum: EF = padding_row_sum(&padding_row);
    let local_cumulative_sum =
        perm_main_sum + pad_row_sum * EF::from_canonical_usize(padding_count);

    (
        CompressedMatrix::new(permutation_trace, padding_row, reserved_poly.total_height),
        local_cumulative_sum,
    )
}
