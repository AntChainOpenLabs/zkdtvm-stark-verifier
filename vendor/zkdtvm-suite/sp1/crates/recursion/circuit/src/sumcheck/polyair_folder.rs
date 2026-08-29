use std::mem::take;

use dt_recursion_compiler::ir::{Config, Felt, SymbolicExt};
use dt_stark::air::FullAirBuilder;
use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;

/// Circuit-side constraint folder for PolyAir's FullAir five-phase protocol.
///
/// Mirrors the native `SumcheckVerifierConstraintFolder` in `polyair/src/verifier.rs:484-636`
/// but uses circuit symbolic types (`SymbolicExt`) instead of concrete field elements.
pub struct RecursivePolyAirConstraintFolder<'a, C: Config> {
    pub public: &'a [Felt<C::F>],
    pub alpha: SymbolicExt<C::F, C::EF>,
    pub beta_powers: &'a [SymbolicExt<C::F, C::EF>],
    pub beta_septix: SymbolicExt<C::F, C::EF>,
    pub precomputed: RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>,
    pub reserved_poly: RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>,
    pub is_first_row: SymbolicExt<C::F, C::EF>,
    pub is_last_row: SymbolicExt<C::F, C::EF>,
    pub local_sum: SymbolicExt<C::F, C::EF>,
    pub permutation: RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>,
    pub multiplicities: Vec<SymbolicExt<C::F, C::EF>>,
    pub batch_size: usize,
    pub accumulator: SymbolicExt<C::F, C::EF>,
    /// Single constraint reducer (alpha challenge), used in Horner accumulation:
    /// `acc = acc * reducer + x`
    pub constraint_reducer: SymbolicExt<C::F, C::EF>,
}

impl<'a, C: Config> RecursivePolyAirConstraintFolder<'a, C>
where
    C::F: Field,
    C::EF: ExtensionField<C::F>,
{
    /// Verify lookup constraints (mirrors `polyair/src/verifier.rs:527-552`).
    ///
    /// For each batch: denominator = product(precomputed values),
    /// numerator = sum(all_but_current * multiplicity),
    /// then assert numerator == denominator * perm_local[lookup_index].
    pub fn constrain_lookup(&mut self) {
        let perm_local = self.permutation.values;
        let multiplicities = take(&mut self.multiplicities);
        let values = &self.precomputed.values[..multiplicities.len()];

        for (lookup_index, (value, multiplicity)) in
            values.chunks(self.batch_size).zip(multiplicities.chunks(self.batch_size)).enumerate()
        {
            if self.batch_size == 2 {
                let residual = match (value, multiplicity) {
                    ([d0, d1], [m0, m1]) => {
                        d1.clone() * m0.clone() +
                            d0.clone() *
                                (m1.clone() - d1.clone() * perm_local[lookup_index].clone())
                    }
                    ([d0], [m0]) => m0.clone() - d0.clone() * perm_local[lookup_index].clone(),
                    _ => unreachable!("batch-size-two chunks contain one or two lookups"),
                };
                self.assert_zero_ext(residual);
                continue;
            }
            let denominator: SymbolicExt<C::F, C::EF> = value.iter().cloned().product();
            let mut numerator = SymbolicExt::<C::F, C::EF>::zero();
            for (i, m) in multiplicity.iter().cloned().enumerate() {
                let mut all_but_current = SymbolicExt::<C::F, C::EF>::one();
                for (j, rlc) in value.iter().enumerate() {
                    if i != j {
                        all_but_current = all_but_current * rlc.clone();
                    }
                }
                numerator = numerator + all_but_current * m;
            }
            self.assert_eq_ext(numerator, denominator * perm_local[lookup_index]);
        }
    }
}

impl<'a, C: Config> FullAirBuilder for RecursivePolyAirConstraintFolder<'a, C>
where
    C::F: Field,
    C::EF: ExtensionField<C::F>,
{
    type F = C::F;
    type EF = C::EF;
    type VarBase = Felt<C::F>;
    type VarMaybeExt = SymbolicExt<C::F, C::EF>;
    type VarExt = SymbolicExt<C::F, C::EF>;
    type MatMaybeExt = RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>;
    type MatExt = RowMajorMatrixView<'a, SymbolicExt<C::F, C::EF>>;

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
        let multiplicity = if is_send { multiplicity } else { -multiplicity };
        self.multiplicities.push(multiplicity);
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

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        SymbolicExt::from_f(ef)
    }

    // Horner accumulation: acc = acc * reducer + x
    // Mirrors polyair/src/verifier.rs:627-629
    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        let x = x.into();
        self.accumulator = self.accumulator * self.constraint_reducer + x;
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        let x = x.into();
        self.accumulator = self.accumulator * self.constraint_reducer + x;
    }
}
