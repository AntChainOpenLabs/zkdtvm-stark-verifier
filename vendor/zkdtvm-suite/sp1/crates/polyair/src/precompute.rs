//! Precompute phase builder for computing linear combinations.
//!
//! This module provides [`PrecomputeRowBuilder`] which implements [`FullAirBuilder`]
//! for the precompute phase, where linear combinations are computed and stored
//! for later use in the proving protocol.

use dt_stark::air::{FullAir, FullAirBuilder, PairCol};
use p3_field::{AbstractExtensionField, ExtensionField, Field};
use p3_matrix::{
    compressed::{padding_row_to_base_vec, CompressedMatrix, PaddingRow},
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use crate::evaluator::uinit_vec;

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

    fn pack_ext_limbs(limbs: &[Self::VarMaybeExt]) -> Self::VarExt {
        let degree = <EF as AbstractExtensionField<F>>::D;
        assert!(
            !limbs.is_empty() && limbs.len() <= degree,
            "extension limb count must be in 1..={degree}, got {}",
            limbs.len()
        );
        if <EF as AbstractExtensionField<MaybyExt>>::D > 1 {
            <EF as AbstractExtensionField<MaybyExt>>::from_base_fn(|idx| {
                limbs.get(idx).copied().unwrap_or_else(MaybyExt::zero)
            })
        } else {
            let theta = <EF as AbstractExtensionField<F>>::monomial(1);
            let mut limbs = limbs.iter().rev();
            let mut packed = EF::zero() + *limbs.next().expect("checked non-empty");
            for limb in limbs {
                packed = theta * packed + *limb;
            }
            packed
        }
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
    preprocessed: Option<&CompressedMatrix<F, F>>,
    main: &CompressedMatrix<F, F>,
    public: &[F],
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    num_precompute: usize,
) -> CompressedMatrix<EF, EF> {
    let height = main.stored_height();
    let total_height = main.total_height;
    preprocessed.map(|prep| assert_eq!(prep.stored_height(), height));
    let mut precomputed_lc =
        RowMajorMatrix::new(uinit_vec(num_precompute * height), num_precompute);
    let prep = preprocessed.unwrap_or(main);
    precomputed_lc
        .par_rows_mut()
        .zip_eq(prep.main.par_row_slices())
        .zip_eq(main.main.par_row_slices())
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
    let padding_row = if total_height == height {
        PaddingRow::None
    } else {
        let main_padding = padding_row_to_base_vec(&main.padding_row);
        let prep_padding = padding_row_to_base_vec(&prep.padding_row);
        let mut padding_row = uinit_vec(num_precompute);
        let mut builder = PrecomputeRowBuilder::<F, F, EF> {
            preprocessed: &prep_padding,
            main: &main_padding,
            beta_powers,
            public,
            alpha,
            beta_septix,
            row: &mut padding_row,
            col_index: 0,
        };
        air.precompute_lc(&mut builder);

        PaddingRow::General(padding_row)
    };

    CompressedMatrix::new(precomputed_lc, padding_row, main.total_height)
}

pub fn collect_reserved_poly<F: Field>(
    preprocessed: Option<&CompressedMatrix<F, F>>,
    main: &CompressedMatrix<F, F>,
    reserved_poly: &[PairCol],
) -> CompressedMatrix<F, F> {
    let height = main.stored_height();
    let total_height = main.total_height;
    preprocessed.as_ref().map(|i| assert!(i.stored_height() == height));
    let mut res =
        RowMajorMatrix::new(uinit_vec::<F>(reserved_poly.len() * height), reserved_poly.len());
    res.par_rows_mut().enumerate().for_each(|(i, row)| {
        let binding = main.main.row_slice(i);
        let main_row = binding.as_ref();
        // when preprocessed == None, assert preprocessed is not used, so we can assign it to any
        // value.
        let binding = preprocessed.unwrap_or(main).main.row_slice(i);
        let prep_row = binding.as_ref();
        for (i, v) in reserved_poly.iter().zip(row.iter_mut()) {
            *v = match i {
                PairCol::Prep(i) => prep_row[*i],
                PairCol::Main(i) => main_row[*i],
            }
        }
    });
    let padding_row = if total_height == height {
        PaddingRow::None
    } else {
        let main_padding = padding_row_to_base_vec(&main.padding_row);
        let prep_padding =
            preprocessed.map(|prep| padding_row_to_base_vec(&prep.padding_row)).unwrap_or_default();
        let padding_row = reserved_poly
            .iter()
            .map(|i| match i {
                PairCol::Prep(i) => prep_padding[*i],
                PairCol::Main(i) => main_padding[*i],
            })
            .collect();
        PaddingRow::General(padding_row)
    };

    CompressedMatrix::new(res, padding_row, main.total_height)
}
