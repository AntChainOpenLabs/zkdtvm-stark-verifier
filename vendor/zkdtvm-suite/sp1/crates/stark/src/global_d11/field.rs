use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Field, PrimeField32};

use super::{
    constants::{BASE_PRIME, D11_DEGREE, D11_SPARSE_WIDTH, FROBENIUS_COLUMNS},
    kernels::{mul_dense_sparse_7, mul_schoolbook, square_sparse_7, square_symmetric},
};

/// An element of `Fp[z] / (z^11 - z^3 - 2)` in low-to-high coefficient order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct D11<F>([F; D11_DEGREE]);

/// A fixed-width multiplier whose coefficients `z^7..z^10` are zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct D11Sparse7<F>([F; D11_SPARSE_WIDTH]);

/// A value together with the base-field norm established by one QR test.
/// Construction is private so production square root extraction cannot skip
/// the residue admission step or recompute its norm.
#[derive(Clone, Copy, Debug)]
pub struct D11VerifiedResidue<F: Field> {
    value: D11<F>,
    norm: F,
}

impl<F: Field> D11<F> {
    #[must_use]
    pub const fn new(coefficients_low_to_high: [F; D11_DEGREE]) -> Self {
        Self(coefficients_low_to_high)
    }

    #[must_use]
    pub const fn coefficients(&self) -> &[F; D11_DEGREE] {
        &self.0
    }

    #[must_use]
    pub fn into_coefficients(self) -> [F; D11_DEGREE] {
        self.0
    }

    #[must_use]
    pub fn zero() -> Self {
        Self([F::zero(); D11_DEGREE])
    }

    #[must_use]
    pub fn one() -> Self {
        Self::from_base(F::one())
    }

    #[must_use]
    pub fn z() -> Self {
        let mut out = [F::zero(); D11_DEGREE];
        out[1] = F::one();
        Self(out)
    }

    #[must_use]
    pub fn from_base(value: F) -> Self {
        let mut out = [F::zero(); D11_DEGREE];
        out[0] = value;
        Self(out)
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(Field::is_zero)
    }

    #[must_use]
    pub fn double(&self) -> Self {
        *self + *self
    }

    /// The 66-product symmetric square kernel.
    #[must_use]
    pub fn square(&self) -> Self {
        square_symmetric(self)
    }

    /// Multiplies by `z`, applying `z^11 = z^3 + 2` directly.
    #[must_use]
    pub fn mul_by_z(&self) -> Self {
        let mut out = [F::zero(); D11_DEGREE];
        out[0] = self.0[10].double();
        for i in 0..10 {
            out[i + 1] = self.0[i];
        }
        out[3] += self.0[10];
        Self(out)
    }

    /// Multiplies by the curve coefficient `z + 36` without a dense multiply.
    #[must_use]
    pub fn mul_by_z_plus_36(&self) -> Self {
        self.mul_by_z() + *self * F::from_canonical_u32(36)
    }

    #[must_use]
    pub fn mul_sparse_7(&self, rhs: &D11Sparse7<F>) -> Self {
        mul_dense_sparse_7(self, rhs)
    }

    #[must_use]
    pub fn pow_u64(&self, mut exponent: u64) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result *= base;
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }
}

impl<F: PrimeField32> D11<F> {
    fn assert_target_field() {
        assert_eq!(
            F::ORDER_U32,
            BASE_PRIME,
            "Projective228QDeltaV2 is defined only over KoalaBear"
        );
    }

    #[must_use]
    pub fn from_canonical_u32(coefficients_low_to_high: [u32; D11_DEGREE]) -> Self {
        Self(coefficients_low_to_high.map(F::from_canonical_u32))
    }

    #[must_use]
    pub fn to_canonical_u32(self) -> [u32; D11_DEGREE] {
        self.0.map(|value| value.as_canonical_u32())
    }

    /// Applies the frozen KoalaBear `p`-power linear map.
    #[must_use]
    pub fn frobenius(&self) -> Self {
        Self::assert_target_field();
        let mut out = [F::zero(); D11_DEGREE];
        for (column, input) in FROBENIUS_COLUMNS.iter().zip(self.0) {
            for (output, coefficient) in out.iter_mut().zip(column) {
                *output += input * F::from_canonical_u32(*coefficient);
            }
        }
        Self(out)
    }

    #[must_use]
    pub fn frobenius_pow(&self, power: usize) -> Self {
        let mut out = *self;
        for _ in 0..power % D11_DEGREE {
            out = out.frobenius();
        }
        out
    }

    /// Returns the norm as a D11 value.  For the target field only coefficient
    /// zero is nonzero; retaining the typed value makes that invariant testable.
    #[must_use]
    pub fn norm_element(&self) -> Self {
        Self::assert_target_field();
        let mut product = Self::one();
        let mut conjugate = *self;
        for _ in 0..D11_DEGREE {
            product *= conjugate;
            conjugate = conjugate.frobenius();
        }
        product
    }

    #[must_use]
    pub fn norm(&self) -> F {
        let norm = self.norm_element();
        debug_assert!(norm.0[1..].iter().all(Field::is_zero));
        norm.0[0]
    }

    /// Inversion through the product of the ten nontrivial conjugates.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Self::assert_target_field();
        let mut conjugate = self.frobenius();
        let mut cofactor = conjugate;
        for _ in 2..D11_DEGREE {
            conjugate = conjugate.frobenius();
            cofactor *= conjugate;
        }
        let norm = (*self * cofactor).0[0];
        norm.try_inverse().map(|inverse_norm| cofactor * inverse_norm)
    }

    #[must_use]
    pub fn inverse(&self) -> Self {
        self.try_inverse().expect("tried to invert zero in D11")
    }

    /// Quadratic residuosity in this odd-degree extension is the Legendre
    /// character of the base-field norm.
    #[must_use]
    pub fn is_quadratic_residue(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        self.norm().exp_u64(u64::from((BASE_PRIME - 1) / 2)) == F::one()
    }

    /// Perform the norm and Legendre work once and carry the verified norm to
    /// square-root extraction.
    #[must_use]
    pub fn verified_quadratic_residue(&self) -> Option<D11VerifiedResidue<F>> {
        if self.is_zero() {
            return Some(D11VerifiedResidue { value: *self, norm: F::zero() });
        }
        let norm = self.norm();
        if norm.exp_u64(u64::from((BASE_PRIME - 1) / 2)) == F::one() {
            Some(D11VerifiedResidue { value: *self, norm })
        } else {
            None
        }
    }

    /// Deterministic reference square root using the norm identity from the
    /// reproduction package.  Tonelli--Shanks runs only in the base field.
    #[must_use]
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        if !self.is_quadratic_residue() {
            return None;
        }

        self.sqrt_from_verified_residue()
    }

    /// Square root after an explicit QR test.  The final squaring check keeps a
    /// false precondition from manufacturing a value.
    #[must_use]
    pub fn sqrt_from_verified_residue(&self) -> Option<Self> {
        let root = self.verified_quadratic_residue()?.sqrt()?;
        (root.square() == *self).then_some(root)
    }
}

impl<F: PrimeField32> D11VerifiedResidue<F> {
    /// Extract the deterministic root using the QR-carried norm. Algebraic
    /// self-checks remain in the reference/KAT wrapper above, not the writer.
    #[must_use]
    pub fn sqrt(self) -> Option<D11<F>> {
        if self.value.is_zero() {
            return Some(D11::zero());
        }

        // E=(S+1)/2 = 1 + ((p+1)/2)(p+p^3+p^5+p^7+p^9).
        let mut conjugate = self.value;
        let mut odd_product = D11::one();
        for power in 1..=9 {
            conjugate = conjugate.frobenius();
            if power & 1 == 1 {
                odd_product *= conjugate;
            }
        }
        let numerator = self.value * odd_product.pow_u64(u64::from((BASE_PRIME + 1) / 2));
        let denominator = base_field_sqrt_from_verified_residue(self.norm)?;
        Some(numerator * denominator.try_inverse()?)
    }
}

fn base_field_sqrt<F: PrimeField32>(value: F) -> Option<F> {
    if value.is_zero() {
        return Some(F::zero());
    }
    if value.exp_u64(u64::from((BASE_PRIME - 1) / 2)) != F::one() {
        return None;
    }

    base_field_sqrt_from_verified_residue(value)
}

fn base_field_sqrt_from_verified_residue<F: PrimeField32>(value: F) -> Option<F> {
    if value.is_zero() {
        return Some(F::zero());
    }

    // p - 1 = 127 * 2^24.
    // `AbstractField::generator()` for KoalaBear is a convenient protocol
    // generator but is a quadratic residue.  Three is the frozen least small
    // quadratic nonresidue for this prime.
    let mut c = F::from_canonical_u32(3).exp_u64(127);
    let mut root = value.exp_u64(64);
    let mut residue = value.exp_u64(127);
    let mut power_of_two = 24usize;

    while residue != F::one() {
        let mut i = 1usize;
        let mut squared = residue.square();
        while i < power_of_two && squared != F::one() {
            squared = squared.square();
            i += 1;
        }
        if i == power_of_two {
            return None;
        }
        let correction = c.exp_u64(1u64 << (power_of_two - i - 1));
        root *= correction;
        let correction_squared = correction.square();
        residue *= correction_squared;
        c = correction_squared;
        power_of_two = i;
    }
    Some(root)
}

impl<F: Field> D11Sparse7<F> {
    #[must_use]
    pub const fn new(coefficients_low_to_high: [F; D11_SPARSE_WIDTH]) -> Self {
        Self(coefficients_low_to_high)
    }

    #[must_use]
    pub const fn coefficients(&self) -> &[F; D11_SPARSE_WIDTH] {
        &self.0
    }

    #[must_use]
    pub fn into_coefficients(self) -> [F; D11_SPARSE_WIDTH] {
        self.0
    }

    #[must_use]
    pub fn to_d11(self) -> D11<F> {
        let mut out = [F::zero(); D11_DEGREE];
        out[..D11_SPARSE_WIDTH].copy_from_slice(&self.0);
        D11::new(out)
    }

    /// Symmetric 28-product square followed by the protocol reduction.
    #[must_use]
    pub fn square(&self) -> D11<F> {
        square_sparse_7(self)
    }
}

impl<F: PrimeField32> D11Sparse7<F> {
    #[must_use]
    pub fn from_canonical_u32(coefficients_low_to_high: [u32; D11_SPARSE_WIDTH]) -> Self {
        assert_eq!(F::ORDER_U32, BASE_PRIME);
        Self(coefficients_low_to_high.map(F::from_canonical_u32))
    }
}

impl<F: Field> Default for D11<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> From<[F; D11_DEGREE]> for D11<F> {
    fn from(value: [F; D11_DEGREE]) -> Self {
        Self::new(value)
    }
}

impl<F: Field> Add for D11<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<F: Field> AddAssign for D11<F> {
    fn add_assign(&mut self, rhs: Self) {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left += right;
        }
    }
}

impl<F: Field> Sub for D11<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<F: Field> SubAssign for D11<F> {
    fn sub_assign(&mut self, rhs: Self) {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left -= right;
        }
    }
}

impl<F: Field> Neg for D11<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.map(Neg::neg))
    }
}

impl<F: Field> Mul for D11<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        mul_schoolbook(&self, &rhs)
    }
}

impl<F: Field> MulAssign for D11<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> Mul<F> for D11<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0.map(|coefficient| coefficient * rhs))
    }
}
