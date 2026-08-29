use std::{mem::take, slice};

use dt_stark::air::{FullAir, FullAirBuilder};
use p3_field::{AbstractExtensionField, ExtensionField, Field};
use p3_matrix::{compressed::CompressedMatrix, dense::RowMajorMatrixView, Matrix};
use p3_maybe_rayon::prelude::*;
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

    /// Precomputed linear combinations (no next row).
    pub precomputed: RowMajorMatrixView<'a, EF>,
    /// Reserved polynomial values (current and next row).
    pub reserved_poly: RowMajorMatrixView<'a, MaybyExt>,
    /// Indicator for the first row.
    pub is_first_row: MaybyExt,
    /// Indicator for the last row.
    pub is_last_row: MaybyExt,
    /// Indicator for transition rows.

    /// The local cumulative sum.
    pub local_sum: EF,
    /// Permutation trace (no next row).
    pub permutation: RowMajorMatrixView<'a, EF>,
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
        let perm_local = unsafe {
            slice::from_raw_parts(self.permutation.values.as_ptr(), self.permutation.values.len())
        };
        let multiplicitys = take(&mut self.multiplicitys);
        let values =
            unsafe { slice::from_raw_parts(self.precomputed.values.as_ptr(), multiplicitys.len()) };

        // Verify each batch of lookups
        for (lookup_index, (value, multiplicity)) in
            values.chunks(self.batch_size).zip(multiplicitys.chunks(self.batch_size)).enumerate()
        {
            if self.batch_size == 2 {
                self.assert_zero_ext(batch_two_lookup_residual(
                    value,
                    multiplicity,
                    perm_local[lookup_index],
                ));
                continue;
            }

            let mut denominator = EF::one();
            let mut numerator = EF::zero();
            let mut zero_count = 0usize;
            let mut zero_index = 0usize;

            for (i, rlc) in value.iter().copied().enumerate() {
                if rlc == EF::zero() {
                    zero_count += 1;
                    zero_index = i;
                } else {
                    denominator *= rlc;
                    numerator += rlc.inverse() * multiplicity[i];
                }
            }

            if zero_count == 0 {
                numerator *= denominator;
            } else if zero_count == 1 {
                numerator = denominator * multiplicity[zero_index];
                denominator = EF::zero();
            } else {
                numerator = EF::zero();
                denominator = EF::zero();
            }

            #[cfg(debug_assertions)]
            {
                let expected_denominator: EF = value.iter().copied().product();
                let mut expected_numerator = EF::zero();
                for (i, m) in multiplicity.iter().copied().enumerate() {
                    let mut all_but_current = EF::one();
                    for (j, rlc) in value.iter().copied().enumerate() {
                        if i != j {
                            all_but_current *= rlc;
                        }
                    }
                    expected_numerator += all_but_current * m;
                }
                debug_assert!(numerator == expected_numerator);
                debug_assert!(denominator == expected_denominator);
            }

            self.assert_eq_ext(numerator, denominator * perm_local[lookup_index]);
        }
    }
}

#[inline]
fn batch_two_lookup_residual<MaybeExt, EF>(
    values: &[EF],
    multiplicities: &[MaybeExt],
    permutation: EF,
) -> EF
where
    MaybeExt: Field,
    EF: Field + ExtensionField<MaybeExt>,
{
    debug_assert_eq!(values.len(), multiplicities.len());
    match (values, multiplicities) {
        ([d0, d1], [m0, m1]) => *d1 * *m0 + *d0 * (EF::from_base(*m1) - *d1 * permutation),
        ([d0], [m0]) => EF::from_base(*m0) - *d0 * permutation,
        _ => unreachable!("batch-size-two chunks contain one or two lookups"),
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
    type MatMaybeExt = RowMajorMatrixView<'a, MaybyExt>;
    type MatExt = RowMajorMatrixView<'a, EF>;

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
        unreachable!("is_transition is deprecated")
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

pub fn uinit_vec<F: Sized>(length: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(length);
    unsafe {
        res.set_len(length);
    }
    res
}

pub fn first_round_evaluation<
    AIR: for<'a> FullAir<ConstraintFolder<'a, F, F, EF>>,
    F: Field,
    EF: Field + ExtensionField<F>,
>(
    air: &AIR,
    public: &[F],
    reserved_poly: &CompressedMatrix<F, F>,
    precomputed_lc: &CompressedMatrix<EF, EF>,
    permutation: &CompressedMatrix<EF, EF>,
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    local_sum: EF,
    batch_size: usize,
    constraint_reducer: &Vec<EF>,
    is_first_row: F,
    is_last_row: F,
) -> Vec<EF> {
    let height = reserved_poly.stored_height();
    assert_eq!(precomputed_lc.stored_height(), height);
    assert_eq!(permutation.stored_height(), height);
    let mut res = vec![EF::zero(); height];

    res.par_iter_mut().enumerate().for_each(|(local_idx, accumulator)| {
        let binding = precomputed_lc.main.row_slice(local_idx);
        let precomputed_lc_local = RowMajorMatrixView::new_row(&binding);
        let binding = permutation.main.row_slice(local_idx);
        let permutation_local = RowMajorMatrixView::new_row(&binding);
        let binding = reserved_poly.main.row_slice(local_idx);
        let reserved_poly_local = RowMajorMatrixView::new_row(&binding);
        let is_first_row = if local_idx == 0 { is_first_row } else { F::zero() };
        let is_last_row =
            if local_idx == (reserved_poly.total_height - 1) { is_last_row } else { F::zero() };

        let mut folder = ConstraintFolder {
            public,
            alpha,
            beta_powers,
            beta_septix,
            precomputed: precomputed_lc_local,
            reserved_poly: reserved_poly_local,
            is_first_row,
            is_last_row,
            local_sum,
            permutation: permutation_local,
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

pub fn nofirst_round_evaluation<
    AIR: for<'a> FullAir<ConstraintFolder<'a, F, EF, EF>>,
    F: Field,
    EF: Field + ExtensionField<F>,
>(
    air: &AIR,
    public: &[F],
    reserved_poly: &CompressedMatrix<F, EF>,
    precomputed_lc: &CompressedMatrix<EF, EF>,
    permutation: &CompressedMatrix<EF, EF>,
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    local_sum: EF,
    batch_size: usize,
    constraint_reducer: &Vec<EF>,
    is_first_row: EF,
    is_last_row: EF,
) -> Vec<EF> {
    let height = reserved_poly.stored_height();
    assert_eq!(precomputed_lc.stored_height(), height);
    assert_eq!(permutation.stored_height(), height);
    let mut res = vec![EF::zero(); height];

    res.par_iter_mut().enumerate().for_each(|(local_idx, accumulator)| {
        let binding = precomputed_lc.main.row_slice(local_idx);
        let precomputed_lc_local = RowMajorMatrixView::new_row(&binding);
        let binding = permutation.main.row_slice(local_idx);
        let permutation_local = RowMajorMatrixView::new_row(&binding);
        let binding = reserved_poly.main.row_slice(local_idx);
        let reserved_poly_local = RowMajorMatrixView::new_row(&binding);

        let is_first_row = if local_idx == 0 { is_first_row } else { EF::zero() };
        let is_last_row =
            if local_idx == (reserved_poly.total_height - 1) { is_last_row } else { EF::zero() };

        let mut folder = ConstraintFolder::<F, EF, EF> {
            public,
            alpha,
            beta_powers,
            beta_septix,
            precomputed: precomputed_lc_local,
            reserved_poly: reserved_poly_local,
            is_first_row,
            is_last_row,
            local_sum,
            permutation: permutation_local,
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

#[cfg(test)]
mod tests {
    use p3_field::{
        extension::QuinticTrinomialExtensionField, AbstractExtensionField, AbstractField, Field,
    };
    use p3_koala_bear::KoalaBear;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::batch_two_lookup_residual;

    type F = KoalaBear;
    type EF = QuinticTrinomialExtensionField<F>;

    fn random_ext(rng: &mut StdRng) -> EF {
        EF::from_base_fn(|_| F::from_canonical_u32(rng.gen_range(0..0x7f000001)))
    }

    fn verifier_oracle<M: Field>(values: &[EF], multiplicities: &[M]) -> (EF, EF)
    where
        EF: p3_field::ExtensionField<M>,
    {
        let denominator = values.iter().copied().product();
        let numerator = multiplicities
            .iter()
            .copied()
            .enumerate()
            .map(|(index, multiplicity)| {
                values
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(other, value)| (other != index).then_some(value))
                    .product::<EF>() *
                    multiplicity
            })
            .sum();
        (numerator, denominator)
    }

    fn legacy_inverse_terms<M: Field>(values: &[EF], multiplicities: &[M]) -> (EF, EF)
    where
        EF: p3_field::ExtensionField<M>,
    {
        let mut denominator = EF::one();
        let mut numerator = EF::zero();
        let mut zero_count = 0usize;
        let mut zero_index = 0usize;
        for (index, value) in values.iter().copied().enumerate() {
            if value == EF::zero() {
                zero_count += 1;
                zero_index = index;
            } else {
                denominator *= value;
                numerator += value.inverse() * multiplicities[index];
            }
        }
        if zero_count == 0 {
            numerator *= denominator;
        } else if zero_count == 1 {
            numerator = denominator * multiplicities[zero_index];
            denominator = EF::zero();
        } else {
            numerator = EF::zero();
            denominator = EF::zero();
        }
        (numerator, denominator)
    }

    fn assert_matches_both_oracles<M: Field>(values: &[EF], multiplicities: &[M], permutation: EF)
    where
        EF: p3_field::ExtensionField<M>,
    {
        let actual = batch_two_lookup_residual(values, multiplicities, permutation);
        let (legacy_numerator, legacy_denominator) = legacy_inverse_terms(values, multiplicities);
        let (oracle_numerator, oracle_denominator) = verifier_oracle(values, multiplicities);
        assert_eq!(actual, legacy_numerator - legacy_denominator * permutation);
        assert_eq!(actual, oracle_numerator - oracle_denominator * permutation);
    }

    #[test]
    fn batch_two_lookup_matches_oracles_for_random_extension_multiplicities() {
        let mut rng = StdRng::seed_from_u64(0x6c6f_6f6b_7570_0002);
        for _ in 0..512 {
            let values = [random_ext(&mut rng), random_ext(&mut rng)];
            let multiplicities = [random_ext(&mut rng), random_ext(&mut rng)];
            assert_matches_both_oracles(&values, &multiplicities, random_ext(&mut rng));
        }
    }

    #[test]
    fn batch_two_lookup_matches_oracles_for_base_multiplicities_and_zero_denominators() {
        let mut rng = StdRng::seed_from_u64(0x6261_7365_0000_0002);
        let nonzero = EF::from_base(F::from_canonical_u32(17));
        let multiplicities = [F::from_canonical_u32(19), F::from_canonical_u32(23)];
        for values in [[EF::zero(), nonzero], [nonzero, EF::zero()], [EF::zero(), EF::zero()]] {
            assert_matches_both_oracles(&values, &multiplicities, random_ext(&mut rng));
        }
        for _ in 0..512 {
            let values = [random_ext(&mut rng), random_ext(&mut rng)];
            let multiplicities = [
                F::from_canonical_u32(rng.gen_range(0..0x7f000001)),
                F::from_canonical_u32(rng.gen_range(0..0x7f000001)),
            ];
            assert_matches_both_oracles(&values, &multiplicities, random_ext(&mut rng));
        }
    }

    #[test]
    fn batch_two_lookup_matches_oracles_for_single_tail() {
        let value = [EF::from_base(F::from_canonical_u32(29))];
        let multiplicity = [F::from_canonical_u32(31)];
        assert_matches_both_oracles(
            &value,
            &multiplicity,
            EF::from_base(F::from_canonical_u32(37)),
        );
    }
}
