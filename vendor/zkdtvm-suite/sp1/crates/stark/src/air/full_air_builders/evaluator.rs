//! Constraint checking phase builder for evaluating constraints.
//!
//! This module provides [`ConstraintFolder`] which implements [`FullAirBuilder`]
//! for the constraint evaluation phase, where constraints are verified against
//! the trace and precomputed values.

use core::mem::take;

use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use super::{
    super::{FullAir, FullAirBuilder, PairCol},
    collect_reserved_poly, get_preprocessed_row,
};

/// Builder for the constraint evaluation phase.
///
/// This builder is used during constraint evaluation to record constraint
/// violations and verify lookup arguments. It accumulates constraint values
/// into an accumulator that should be zero for valid traces.
///
/// # Type Parameters
///
/// * `F` - The base field
/// * `MaybyExt` - A field that may be base or extension
/// * `EF` - The extension field
pub struct ConstraintFolder<
    'a,
    F: Field,
    MaybyExt: Field + ExtensionField<F>,
    EF: Field + ExtensionField<F> + ExtensionField<MaybyExt>,
> {
    /// Public values.
    pub public: &'a [F],
    /// The alpha challenge.
    pub alpha: EF,
    /// Powers of the beta challenge.
    pub beta_powers: &'a [EF],
    /// Beta raised to the 7th power in the septic extension.
    pub beta_septix: EF,

    /// Precomputed linear combinations (current and next row).
    pub precomputed: VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>,
    /// Reserved polynomial values (current and next row).
    pub reserved_poly:
        VerticalPair<RowMajorMatrixView<'a, MaybyExt>, RowMajorMatrixView<'a, MaybyExt>>,
    /// Indicator for the first row.
    pub is_first_row: MaybyExt,
    /// Indicator for the last row.
    pub is_last_row: MaybyExt,
    /// Indicator for transition rows.
    pub is_transition: MaybyExt,

    /// The local cumulative sum.
    pub local_sum: EF,
    /// Permutation trace (current and next row).
    pub permutation: VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>,
    /// Cached multiplicities from lookup operations.
    pub multiplicitys: Vec<MaybyExt>,
    /// Number of lookups per batch.
    pub batch_size: usize,

    /// Accumulator for constraint values.
    pub accumulator: &'a mut EF,
    /// Coefficients for combining constraints.
    pub constraint_reducer: &'a Vec<EF>,
    /// Current constraint index.
    pub constraint_index: usize,
}

impl<
        'a,
        F: Field,
        MaybyExt: Field + ExtensionField<F>,
        EF: Field + ExtensionField<F> + ExtensionField<MaybyExt>,
    > ConstraintFolder<'a, F, MaybyExt, EF>
{
    /// Verify lookup constraints.
    ///
    /// This method is called after `air.lookup()` and `air.eval()` to verify
    /// that the permutation trace running products match the precomputed
    /// lookup denominators.
    pub fn constrain_lookup(&mut self) {
        use std::ops::Deref;
        let perm_local: Vec<EF> = self.permutation.row_slice(0).deref().to_vec();
        let perm_next: Vec<EF> = self.permutation.row_slice(1).deref().to_vec();
        let multiplicitys = take(&mut self.multiplicitys);
        let values: Vec<EF> = self.precomputed.row_slice(0).deref()[..multiplicitys.len()].to_vec();

        let perm_width = perm_local.len();

        // Verify each batch of lookups
        for (lookup_index, (value, multiplicity)) in
            values.chunks(self.batch_size).zip(multiplicitys.chunks(self.batch_size)).enumerate()
        {
            let denumerator: EF = value.iter().copied().product();
            let mut numerator = EF::zero();
            for (i, m) in multiplicity.iter().copied().enumerate() {
                let mut all_but_current = EF::one();
                for other_rlc in
                    value.iter().enumerate().filter(|(j, _)| i != *j).map(|(_, rlc)| rlc)
                {
                    all_but_current = all_but_current.clone() * other_rlc.clone();
                }
                numerator += all_but_current * m;
            }
            self.assert_eq_ext(numerator, denumerator * perm_local[lookup_index]);
        }

        let sum_local: EF = perm_local[..perm_width - 1].iter().copied().sum();
        let sum_next: EF = perm_next[..perm_width - 1].iter().copied().sum();

        let phi_local: EF = perm_local[perm_width - 1];
        let phi_next: EF = perm_next[perm_width - 1];
        let local_sum = self.local_sum;

        // Cumulative sum constraints
        self.when_first_row().assert_eq_ext(phi_local, sum_local);
        self.when_transition().assert_eq_ext(phi_next - phi_local, sum_next);
        self.when_last_row().assert_eq_ext(phi_next, local_sum);
    }
}

impl<
        'a,
        F: Field,
        MaybyExt: Field + ExtensionField<F>,
        EF: Field + ExtensionField<F> + ExtensionField<MaybyExt>,
    > FullAirBuilder for ConstraintFolder<'a, F, MaybyExt, EF>
{
    type F = F;
    type EF = EF;
    type VarBase = F;
    type VarMaybeExt = MaybyExt;
    type VarExt = EF;
    type MatMaybeExt =
        VerticalPair<RowMajorMatrixView<'a, MaybyExt>, RowMajorMatrixView<'a, MaybyExt>>;
    type MatExt = VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        unreachable!("preprocessed should not be used in constraint evaluation phase")
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        unreachable!("main should not be used in constraint evaluation phase")
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

    fn retain_precomputed(&mut self, _: Self::VarExt) {
        unreachable!("retain_precomputed should not be called in constraint evaluation phase")
    }

    fn precomputed(&self) -> Self::MatExt {
        self.precomputed
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        self.reserved_poly
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        let multiplicity = if is_send { multiplicity } else { multiplicity.neg() };
        self.multiplicitys.push(multiplicity);
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        self.is_last_row
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        self.is_transition
    }

    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        a * b
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        ef
    }
    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        let x = x.into();
        *self.accumulator += self.constraint_reducer[self.constraint_index] * x;
        self.constraint_index += 1;
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        let x = x.into();
        *self.accumulator += self.constraint_reducer[self.constraint_index] * x;
        self.constraint_index += 1;
    }
}

/// Evaluate constraints for the first round of sumcheck.
///
/// This function processes each row of the main trace, evaluating constraints
/// and verifying lookups. It uses the full main and preprocessed traces
/// (not reserved polynomials).
///
/// # Returns
///
/// A vector of constraint accumulator values, one per row. All values should
/// be zero for a valid trace.
pub fn first_round_evaluation<
    AIR: for<'a> FullAir<ConstraintFolder<'a, F, F, EF>>,
    F: Field,
    EF: Field + ExtensionField<F>,
>(
    air: &AIR,
    public: &[F],
    preprocessed: Option<&RowMajorMatrix<F>>,
    main: &RowMajorMatrix<F>,
    precomputed_lc: &RowMajorMatrix<EF>,
    permutation: &RowMajorMatrix<EF>,
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    selector_first: F,
    selector_last: F,
    local_sum: EF,
    batch_size: usize,
    constraint_reducer: &Vec<EF>,
) -> Vec<EF> {
    let height = main.height() - 1;
    assert!(precomputed_lc.height() == height + 1);
    assert!(permutation.height() == height + 1);
    let selector_transition = F::one() - selector_last;
    let mut res = vec![EF::zero(); height];
    let reserved_poly = air.reserved_poly();

    res.par_iter_mut().enumerate().for_each(|(local_idx, accumulator)| {
        let next_idx = local_idx + 1;
        let is_first_row = if local_idx == 0 { selector_first } else { F::zero() };
        let (is_last_row, is_transition) = if local_idx == height - 1 {
            (selector_last, selector_transition)
        } else {
            (F::zero(), F::one())
        };
        let precomputed_lc_local = precomputed_lc.row_slice(local_idx);
        let precomputed_lc_next = precomputed_lc.row_slice(next_idx);
        let permutation_local = permutation.row_slice(local_idx);
        let permutation_next = permutation.row_slice(next_idx);
        let precomputed_lc_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&precomputed_lc_local),
            RowMajorMatrixView::new_row(&precomputed_lc_next),
        );
        let permutation_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&permutation_local),
            RowMajorMatrixView::new_row(&permutation_next),
        );
        let main_local_binding = main.row_slice(local_idx);
        use std::ops::Deref;
        let main_local: &[F] = main_local_binding.deref();
        let prep_local: &[F] = get_preprocessed_row(preprocessed, local_idx);
        let main_next_binding = main.row_slice(next_idx % height);
        let main_next: &[F] = main_next_binding.deref();
        let prep_next: &[F] = get_preprocessed_row(preprocessed, next_idx % height);
        let reserved_poly_local = collect_reserved_poly(main_local, prep_local, &reserved_poly);
        let reserved_poly_next = collect_reserved_poly(main_next, prep_next, &reserved_poly);
        let reserved_poly_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&reserved_poly_local),
            RowMajorMatrixView::new_row(&reserved_poly_next),
        );
        let mut folder = ConstraintFolder {
            public,
            alpha,
            beta_powers,
            beta_septix,
            precomputed: precomputed_lc_row,
            reserved_poly: reserved_poly_row,
            is_first_row,
            is_last_row,
            is_transition,
            local_sum,
            permutation: permutation_row,
            multiplicitys: vec![],
            batch_size,
            accumulator,
            constraint_reducer,
            constraint_index: 0,
        };
        air.eval(&mut folder);
        air.lookup(&mut folder);
        folder.constrain_lookup();
    });
    res
}

/// Evaluate constraints for non-first rounds of sumcheck.
///
/// This function processes each row using reserved polynomials instead of
/// the full main trace. This is more efficient for later sumcheck rounds
/// where only a subset of columns are needed.
///
/// # Returns
///
/// A vector of constraint accumulator values, one per row. All values should
/// be zero for a valid trace.
pub fn nonfirst_round_evaluation<
    AIR: for<'a> FullAir<ConstraintFolder<'a, F, EF, EF>>,
    F: Field,
    EF: Field + ExtensionField<F>,
>(
    air: &AIR,
    public: &[F],
    reserved_poly: &RowMajorMatrix<EF>,
    precomputed_lc: &RowMajorMatrix<EF>,
    permutation: &RowMajorMatrix<EF>,
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    selector_first: EF,
    selector_last: EF,
    local_sum: EF,
    batch_size: usize,
    constraint_reducer: &Vec<EF>,
) -> Vec<EF> {
    let height = reserved_poly.height() - 1;
    assert!(precomputed_lc.height() == height + 1);
    assert!(permutation.height() == height + 1);
    let selector_transition = EF::one() - selector_last;

    let mut res = vec![EF::zero(); height];
    res.par_iter_mut().enumerate().for_each(|(local_idx, accumulator)| {
        let next_idx = local_idx + 1;
        let is_first_row = if local_idx == 0 { selector_first } else { EF::zero() };
        let (is_last_row, is_transition) = if local_idx == height - 1 {
            (selector_last, selector_transition)
        } else {
            (EF::zero(), EF::one())
        };

        let reserved_poly_local = reserved_poly.row_slice(local_idx);
        let reserved_poly_next = reserved_poly.row_slice(next_idx);
        let precomputed_lc_local = precomputed_lc.row_slice(local_idx);
        let precomputed_lc_next = precomputed_lc.row_slice(next_idx);
        let permutation_local = permutation.row_slice(local_idx);
        let permutation_next = permutation.row_slice(next_idx);
        let reserved_poly_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&reserved_poly_local),
            RowMajorMatrixView::new_row(&reserved_poly_next),
        );
        let precomputed_lc_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&precomputed_lc_local),
            RowMajorMatrixView::new_row(&precomputed_lc_next),
        );
        let permutation_row = VerticalPair::new(
            RowMajorMatrixView::new_row(&permutation_local),
            RowMajorMatrixView::new_row(&permutation_next),
        );
        let mut folder = ConstraintFolder::<F, EF, EF> {
            public,
            alpha,
            beta_powers,
            beta_septix,
            precomputed: precomputed_lc_row,
            reserved_poly: reserved_poly_row,
            is_first_row,
            is_last_row,
            is_transition,
            local_sum,
            permutation: permutation_row,
            multiplicitys: vec![],
            batch_size,
            accumulator,
            constraint_reducer,
            constraint_index: 0,
        };
        air.eval(&mut folder);
        air.lookup(&mut folder);
        folder.constrain_lookup();
    });
    res
}

/// Bind variables for main and preprocessed traces during sumcheck.
///
/// This function folds the reserved polynomials after each sumcheck round,
/// reducing the number of rows by half.
pub fn bound_var_main_prep<F: Field, EF: Field + ExtensionField<F>>(
    main: &RowMajorMatrix<F>,
    preprocessed: Option<&RowMajorMatrix<F>>,
    reserved_poly: &[PairCol],
    r: EF,
) -> RowMajorMatrix<EF> {
    let height = main.height();
    assert!(height.is_power_of_two());
    let half_height = height / 2;
    let mut res = RowMajorMatrix::new(
        vec![EF::zero(); reserved_poly.len() * (half_height + 1)],
        reserved_poly.len(),
    );

    res.par_rows_mut().enumerate().for_each(|(i, row)| {
        use std::ops::Deref;
        let main_0_binding = main.row_slice(i);
        let main_0: &[F] = main_0_binding.deref();
        let main_1_binding = main.row_slice((i + half_height) % height);
        let main_1: &[F] = main_1_binding.deref();
        let prep_0: &[F] = get_preprocessed_row(preprocessed, i);
        let prep_1: &[F] = get_preprocessed_row(preprocessed, (i + half_height) % height);
        let reserved_0 = collect_reserved_poly(main_0, prep_0, reserved_poly);
        let reserved_1 = collect_reserved_poly(main_1, prep_1, reserved_poly);
        row.into_iter().zip_eq(reserved_0).zip_eq(reserved_1).for_each(|((col, col_0), col_1)| {
            *col = r * (col_1 - col_0) + col_0;
        });
    });
    res
}

/// Bind variables for a matrix during sumcheck.
///
/// This function folds a matrix after each sumcheck round,
/// reducing the number of rows by half.
pub fn bound_var_mat<EF: Field>(mat: &RowMajorMatrix<EF>, r: EF) -> RowMajorMatrix<EF> {
    let height = mat.height() - 1;
    assert!(height.is_power_of_two());
    let half_height = height / 2;
    let mut res =
        RowMajorMatrix::new(vec![EF::zero(); mat.width() * (half_height + 1)], mat.width());

    res.par_rows_mut().enumerate().for_each(|(i, row)| {
        use std::ops::Deref;
        let temp_0_binding = mat.row_slice(i);
        let temp_0: &[EF] = temp_0_binding.deref();
        let temp_1_binding = mat.row_slice(i + half_height);
        let temp_1: &[EF] = temp_1_binding.deref();
        row.into_iter().zip_eq(temp_0).zip_eq(temp_1).for_each(|((col, col_0), col_1)| {
            *col = r * (*col_1 - *col_0) + *col_0;
        });
    });
    res
}
