//! FullAirBuilder trait and FilteredAirBuilder for sumcheck-based constraint evaluation.
//!
//! This module provides the builder traits used during different phases of
//! sumcheck-based proving:
//! - Precompute phase: Build linear combinations
//! - Permutation phase: Generate permutation traces
//! - Evaluation phase: Check constraints

use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use p3_matrix::Matrix;

use crate::septic_curve_params::SepticCurveParams;

/// A builder trait for sumcheck-based AIR constraint evaluation.
///
/// This trait provides the interface used during different phases of the
/// sumcheck proving protocol. It defines associated types for different
/// variable representations and methods for accessing trace data and
/// recording constraints.
///
/// # Associated Types
///
/// - `F`: The base field
/// - `EF`: The extension field
/// - `VarBase`: Variables that live in the base field
/// - `VarMaybeExt`: Variables that may be in base or extension field
/// - `VarExt`: Variables in the extension field
/// - `MatMaybeExt`: Matrix of maybe-extension variables
/// - `MatExt`: Matrix of extension variables
pub trait FullAirBuilder: Sized {
    /// The base field.
    type F: Field;

    /// The extension field.
    type EF: Field + ExtensionField<Self::F>;

    /// Variable type that lives in the base field.
    type VarBase: Clone + Send + Sync + Into<Self::VarMaybeExt>;

    /// Variable type that may be in base or extension field.
    /// Must support field arithmetic operations.
    type VarMaybeExt: AbstractField
        + From<Self::F>
        + core::ops::Mul<Self::F, Output = Self::VarMaybeExt>
        + Send
        + Sync;

    /// Variable type in the extension field.
    /// Must support arithmetic operations with `VarMaybeExt`.
    type VarExt: core::ops::Add<Self::VarMaybeExt, Output = Self::VarExt>
        + core::ops::Add<Output = Self::VarExt>
        + core::ops::Sub<Output = Self::VarExt>
        + core::ops::Mul<Self::VarMaybeExt, Output = Self::VarExt>
        + core::ops::Mul<Output = Self::VarExt>
        + Clone
        + Send
        + Sync;

    /// Matrix type for maybe-extension variables.
    type MatMaybeExt: Matrix<Self::VarMaybeExt>;

    /// Matrix type for extension variables.
    type MatExt: Matrix<Self::VarExt>;

    /// Returns the preprocessed trace columns for the current row.
    fn preprocessed(&self) -> &[Self::VarMaybeExt];

    /// Returns the main trace columns for the current row.
    fn main(&self) -> &[Self::VarMaybeExt];

    /// Returns the public values.
    fn public(&self) -> &[Self::VarBase];

    /// Returns the alpha challenge.
    fn alpha(&self) -> Self::VarExt;

    /// Returns the powers of beta challenge.
    fn beta_powers(&self) -> &[Self::VarExt];

    /// Returns beta raised to the 7th power in the septic extension.
    fn beta_septix(&self) -> Self::VarExt;

    /// Retain a precomputed linear combination value for later use.
    fn retain_precomputed(&mut self, x: Self::VarExt);

    /// Returns the matrix of precomputed linear combinations.
    fn precomputed(&self) -> Self::MatExt;

    /// Returns the matrix of reserved polynomials.
    fn reserved_poly(&self) -> Self::MatMaybeExt;

    /// Record a local lookup operation.
    ///
    /// # Arguments
    ///
    /// * `multiplicity` - The multiplicity of the lookup
    /// * `is_send` - Whether this is a send operation (true) or receive (false)
    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool);

    /// Returns the indicator for the first row.
    fn is_first_row(&self) -> Self::VarMaybeExt;

    /// Returns the indicator for the last row.
    fn is_last_row(&self) -> Self::VarMaybeExt;

    /// Returns the indicator for transition rows (not first, not last).
    fn is_transition(&self) -> Self::VarMaybeExt;

    /// Assert that a base field expression equals zero.
    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I);

    /// Assert that an extension field expression equals zero.
    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I);

    // =========================================================================
    // Helper methods with default implementations
    // =========================================================================

    /// Returns the zero value for `VarMaybeExt`.
    fn zero_maybe() -> Self::VarMaybeExt {
        Self::VarMaybeExt::from(Self::F::zero())
    }

    /// Returns the one value for `VarMaybeExt`.
    fn one_maybe() -> Self::VarMaybeExt {
        Self::VarMaybeExt::from(Self::F::one())
    }

    /// Convert an extension field constant into a `VarExt`.
    fn from_ef(ef: Self::EF) -> Self::VarExt;

    /// Pack a non-empty low-limb prefix in the protocol extension basis.
    ///
    /// Missing high limbs are zero. Concrete first-round/precompute builders
    /// override this default to construct the extension directly; the default
    /// Horner form is correct for extension-valued verifier/circuit variables.
    fn pack_ext_limbs(limbs: &[Self::VarMaybeExt]) -> Self::VarExt {
        let degree = <Self::EF as AbstractExtensionField<Self::F>>::D;
        assert!(
            !limbs.is_empty() && limbs.len() <= degree,
            "extension limb count must be in 1..={degree}, got {}",
            limbs.len()
        );
        let theta = Self::from_ef(<Self::EF as AbstractExtensionField<Self::F>>::monomial(1));
        let mut limbs = limbs.iter().rev();
        let mut packed =
            Self::from_ef(Self::EF::zero()) + limbs.next().expect("checked non-empty").clone();
        for limb in limbs {
            packed = theta.clone() * packed + limb.clone();
        }
        packed
    }

    /// Send a lookup with the given multiplicity.
    fn send(&mut self, multiplicity: Self::VarMaybeExt) {
        self.local_lookup(multiplicity, true)
    }

    /// Receive a lookup with the given multiplicity.
    fn recv(&mut self, multiplicity: Self::VarMaybeExt) {
        self.local_lookup(multiplicity, false)
    }

    /// Returns a sub-builder whose constraints are enforced only when `condition` is nonzero.
    fn when<I: Into<Self::VarMaybeExt>>(&mut self, condition: I) -> FilteredAirBuilder<'_, Self> {
        FilteredAirBuilder { inner: self, condition: condition.into() }
    }

    /// Returns a sub-builder whose constraints are enforced only when `x != y`.
    fn when_ne<I1, I2>(&mut self, x: I1, y: I2) -> FilteredAirBuilder<'_, Self>
    where
        I1: Into<Self::VarMaybeExt>,
        I2: Into<Self::VarMaybeExt>,
        Self::VarMaybeExt: core::ops::Sub<Output = Self::VarMaybeExt>,
    {
        let diff = x.into() - y.into();
        self.when(diff)
    }

    /// Returns a sub-builder whose constraints are enforced only on the first row.
    fn when_first_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_first_row())
    }

    /// Returns a sub-builder whose constraints are enforced only on the last row.
    fn when_last_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_last_row())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last.
    fn when_transition(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_transition())
    }

    /// Assert that a base field expression equals one.
    fn assert_one<I>(&mut self, x: I)
    where
        I: Into<Self::VarMaybeExt>,
        Self::F: Into<Self::VarMaybeExt>,
        Self::VarMaybeExt: core::ops::Sub<Output = Self::VarMaybeExt>,
    {
        let diff = x.into() - Self::F::one().into();
        self.assert_zero(diff);
    }

    /// Assert that an extension field expression equals one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::VarExt>,
        Self::EF: Into<Self::VarExt>,
        Self::VarExt: core::ops::Sub<Output = Self::VarExt>,
    {
        let diff = x.into() - Self::EF::one().into();
        self.assert_zero_ext(diff);
    }

    /// Assert that two base field expressions are equal.
    fn assert_eq<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::VarMaybeExt>,
        I2: Into<Self::VarMaybeExt>,
        Self::VarMaybeExt: core::ops::Sub<Output = Self::VarMaybeExt>,
    {
        let diff = x.into() - y.into();
        self.assert_zero(diff);
    }

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::VarExt>,
        I2: Into<Self::VarExt>,
        Self::VarExt: core::ops::Sub<Output = Self::VarExt>,
    {
        let diff = x.into() - y.into();
        self.assert_zero_ext(diff);
    }

    /// Compute the RLC denominator for a lookup interaction.
    ///
    /// The denominator is defined as:
    ///   `alpha + interaction_kind + beta^1 * values[0] + beta^2 * values[1] + ...`
    ///
    /// This is the standard LogUp denominator used in all PolyAir chip implementations.
    /// Chip authors should call this instead of reimplementing the formula.
    fn lookup_denominator<I>(&self, interaction_kind: Self::VarMaybeExt, values: I) -> Self::VarExt
    where
        I: IntoIterator<Item = Self::VarMaybeExt>,
    {
        let mut denominator = self.alpha() + interaction_kind;
        let beta_powers = self.beta_powers();
        for (i, value) in values.into_iter().enumerate() {
            denominator = denominator + beta_powers[i + 1].clone() * value;
        }
        denominator
    }

    /// Compute a lookup denominator whose logical values are already packed
    /// extension-field blocks. Beta advances once per block, not per base limb.
    fn lookup_denominator_ext_blocks<I>(
        &self,
        interaction_kind: Self::VarMaybeExt,
        blocks: I,
    ) -> Self::VarExt
    where
        I: IntoIterator<Item = Self::VarExt>,
    {
        let mut denominator = self.alpha() + interaction_kind;
        let beta_powers = self.beta_powers();
        for (i, block) in blocks.into_iter().enumerate() {
            denominator = denominator + beta_powers[i + 1].clone() * block;
        }
        denominator
    }

    /// Multiply `VarMaybeExt` by a base field constant.
    ///
    /// **Default**: `a * VarMaybeExt::from(b)` → `EF × EF` (16 base muls).
    /// **Optimized override**: `a * b` → `EF × F` (4 base muls, 4× faster).
    ///
    /// The default cannot use `a * b` directly because `VarMaybeExt` only has
    /// `Mul<Output = VarMaybeExt>`, not `Mul<F>`. The `Mul<F>` bound is only
    /// available in concrete impls where `VarMaybeExt: ExtensionField<F>`.
    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        a * Self::VarMaybeExt::from(b)
    }

    /// Multiply two septic extension expressions using the irreducible polynomial from `P`.
    ///
    /// The reduction step uses `mul_base` for the constant coefficients `neg_const`
    /// and `neg_linear`, which are base field values. In Round 1+ of sumcheck this
    /// avoids 12 unnecessary `EF × EF` multiplications (16 base-field muls each),
    /// replacing them with `EF × F` (4 base-field muls each) — a **4× speedup**
    /// on the reduction phase.
    fn septic_mul<P: SepticCurveParams>(
        a: &[Self::VarMaybeExt; 7],
        b: &[Self::VarMaybeExt; 7],
    ) -> [Self::VarMaybeExt; 7] {
        let mut res: [Self::VarMaybeExt; 13] = core::array::from_fn(|_| Self::zero_maybe());
        for i in 0..7 {
            for j in 0..7 {
                res[i + j] = res[i + j].clone() + a[i].clone() * b[j].clone();
            }
        }

        let neg_const = Self::F::from_canonical_u32(
            (-P::POLY_COEFFS[0]).try_into().expect("negative septic constant coefficient"),
        );
        let neg_linear = Self::F::from_canonical_u32(
            (-P::POLY_COEFFS[1]).try_into().expect("negative septic linear coefficient"),
        );
        let mut ret: [Self::VarMaybeExt; 7] = core::array::from_fn(|i| res[i].clone());
        for i in 7..13 {
            ret[i - 7] = ret[i - 7].clone() + Self::mul_base(res[i].clone(), neg_const);
            ret[i - 6] = ret[i - 6].clone() + Self::mul_base(res[i].clone(), neg_linear);
        }
        ret
    }
}

/// A filtered builder that multiplies all constraints by a condition.
///
/// This is returned by `FullAirBuilder::when()` and similar methods to
/// enable conditional constraint enforcement.
#[derive(Debug)]
pub struct FilteredAirBuilder<'a, AB: FullAirBuilder> {
    /// Reference to the underlying builder where constraints are ultimately recorded.
    pub inner: &'a mut AB,

    /// Condition expression that controls when constraints are enforced.
    pub condition: AB::VarMaybeExt,
}

impl<'a, AB: FullAirBuilder> FullAirBuilder for FilteredAirBuilder<'a, AB> {
    type F = AB::F;
    type EF = AB::EF;
    type VarBase = AB::VarBase;
    type VarMaybeExt = AB::VarMaybeExt;
    type VarExt = AB::VarExt;
    type MatMaybeExt = AB::MatMaybeExt;
    type MatExt = AB::MatExt;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        self.inner.preprocessed()
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        self.inner.main()
    }

    fn public(&self) -> &[Self::VarBase] {
        self.inner.public()
    }

    fn alpha(&self) -> Self::VarExt {
        self.inner.alpha()
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        self.inner.beta_powers()
    }

    fn beta_septix(&self) -> Self::VarExt {
        self.inner.beta_septix()
    }

    fn retain_precomputed(&mut self, x: Self::VarExt) {
        self.inner.retain_precomputed(x)
    }

    fn precomputed(&self) -> Self::MatExt {
        self.inner.precomputed()
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        self.inner.reserved_poly()
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        self.inner.local_lookup(multiplicity, is_send)
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        self.inner.is_last_row()
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        self.inner.is_transition()
    }

    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        AB::mul_base(a, b)
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        AB::from_ef(ef)
    }

    fn pack_ext_limbs(limbs: &[Self::VarMaybeExt]) -> Self::VarExt {
        AB::pack_ext_limbs(limbs)
    }

    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        let x = x.into();
        self.inner.assert_zero(self.condition.clone() * x);
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        let x = x.into();
        self.inner.assert_zero_ext(x * self.condition.clone());
    }
}
