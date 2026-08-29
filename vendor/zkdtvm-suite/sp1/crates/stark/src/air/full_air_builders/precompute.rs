//! Precompute phase builder for computing linear combinations.
//!
//! This module provides [`PrecomputeRowBuilder`] which implements [`FullAirBuilder`]
//! for the precompute phase, where linear combinations are computed and stored
//! for later use in the proving protocol.

use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use super::super::{FullAir, FullAirBuilder};

/// Builder for the precompute phase.
///
/// This builder is used during the precompute phase to compute and retain
/// linear combination values. It provides access to trace columns and
/// challenges, and stores computed values via `retain_precomputed()`.
///
/// # Type Parameters
///
/// * `F` - The base field
/// * `MaybyExt` - A field that may be base or extension (for trace values)
/// * `EF` - The extension field for computed values
pub struct PrecomputeRowBuilder<
    'a,
    F: Field,
    MaybyExt: Field + ExtensionField<F>,
    EF: Field + ExtensionField<F> + ExtensionField<MaybyExt>,
> {
    /// Preprocessed trace columns for the current row.
    pub preprocessed: &'a [MaybyExt],
    /// Main trace columns for the current row.
    pub main: &'a [MaybyExt],
    /// Powers of the beta challenge.
    pub beta_powers: &'a [EF],
    /// Public values.
    pub public: &'a [F],
    /// The alpha challenge.
    pub alpha: EF,
    /// Beta raised to the 7th power in the septic extension.
    pub beta_septix: EF,
    /// Output row for storing precomputed values.
    pub row: &'a mut [EF],
    /// Current column index in the output row.
    pub col_index: usize,
}

impl<
        'a,
        F: Field,
        MaybyExt: Field + ExtensionField<F>,
        EF: Field + ExtensionField<F> + ExtensionField<MaybyExt>,
    > FullAirBuilder for PrecomputeRowBuilder<'a, F, MaybyExt, EF>
{
    type F = F;
    type EF = EF;
    type VarBase = F;
    type VarMaybeExt = MaybyExt;
    type VarExt = EF;
    type MatMaybeExt = RowMajorMatrixView<'a, Self::VarMaybeExt>;
    type MatExt = RowMajorMatrixView<'a, Self::VarExt>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        self.preprocessed
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        self.main
    }

    fn public(&self) -> &[Self::VarBase] {
        self.public
    }

    fn alpha(&self) -> Self::VarExt {
        self.alpha
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        self.beta_powers
    }

    fn beta_septix(&self) -> Self::VarExt {
        self.beta_septix
    }

    fn retain_precomputed(&mut self, x: Self::VarExt) {
        self.row[self.col_index] = x;
        self.col_index += 1;
    }

    fn precomputed(&self) -> Self::MatExt {
        unreachable!("precomputed is not ready in precompute linear combination phase")
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        unreachable!("reserved_poly should not be used in precompute linear combination phase")
    }

    fn local_lookup(&mut self, _: Self::VarMaybeExt, _: bool) {
        unreachable!("local_lookup is not ready in precompute linear combination phase")
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_first_row should not be used in precompute linear combination phase")
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        unreachable!("is_last_row should not be used in precompute linear combination phase")
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        unreachable!("is_transition should not be used in precompute linear combination phase")
    }

    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        a * b
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        ef
    }
    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, _: I) {
        unreachable!("assert_zero should not be used in precompute linear combination phase")
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, _: I) {
        unreachable!("assert_zero_ext should not be used in precompute linear combination phase")
    }
}

/// Compute linear combinations for all rows in parallel.
///
/// This function processes each row of the main trace in parallel, calling
/// `air.precompute_lc()` for each row and storing the results in the output matrix.
///
/// # Arguments
///
/// * `air` - The AIR implementation
/// * `preprocessed` - Optional preprocessed trace
/// * `main` - Main trace
/// * `public` - Public values
/// * `alpha` - Alpha challenge
/// * `beta_powers` - Powers of beta challenge
/// * `beta_septix` - Beta raised to 7th power
/// * `num_precompute` - Number of precomputed values per row
///
/// # Returns
///
/// A matrix with `num_precompute` columns and `height + 1` rows, where `height`
/// is the number of rows in the main trace. The extra row is a copy of the first
/// row for cyclic consistency.
pub fn precompute_linear_combination<
    AIR: for<'a> FullAir<PrecomputeRowBuilder<'a, F, F, EF>>,
    F: Field,
    EF: Field + ExtensionField<F>,
>(
    air: &AIR,
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    public: &[F],
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    num_precompute: usize,
) -> RowMajorMatrix<EF> {
    let height = main.height();
    assert!(air.required_max_beta_power() < beta_powers.len());
    let mut precomputed_lc =
        RowMajorMatrix::new(vec![EF::zero(); num_precompute * (height + 1)], num_precompute);

    if let Some(prep) = preprocessed {
        assert!(prep.height() == height);
        precomputed_lc
            .par_rows_mut()
            .take(height)
            .zip_eq(prep.par_row_slices())
            .zip_eq(main.par_row_slices())
            .for_each(|((row, preprocessed), main)| {
                let mut builder = PrecomputeRowBuilder::<F, F, EF> {
                    preprocessed,
                    main,
                    beta_powers,
                    public,
                    alpha,
                    beta_septix,
                    row,
                    col_index: 0,
                };
                air.precompute_lc(&mut builder);
            });
    } else {
        precomputed_lc.par_rows_mut().take(height).zip_eq(main.par_row_slices()).for_each(
            |(row, main)| {
                let mut builder = PrecomputeRowBuilder {
                    preprocessed: &[],
                    main,
                    beta_powers,
                    public,
                    alpha,
                    beta_septix,
                    row,
                    col_index: 0,
                };
                air.precompute_lc(&mut builder);
            },
        );
    }
    // Copy first row to the extra row for cyclic consistency
    use std::ops::Deref;
    let first_row: Vec<_> = {
        let first_row_binding = precomputed_lc.row_slice(0);
        first_row_binding.deref().to_vec()
    };
    for i in 0..num_precompute {
        precomputed_lc.row_mut(height)[i] = first_row[i];
    }
    precomputed_lc
}
