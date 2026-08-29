use std::ops::{Add, Div, Mul, Neg, Sub};

use dt_derive::AlignedBorrow;
#[cfg(not(feature = "ext5"))]
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
#[cfg(feature = "ext5")]
use p3_field::extension::{QuinticTrinomialExtendable, QuinticTrinomialExtensionField};
use p3_field::{AbstractExtensionField, AbstractField, Field};

#[cfg(feature = "ext5")]
const D: usize = 5;
#[cfg(not(feature = "ext5"))]
const D: usize = 4;

/// A challenge extension element represented over a generic type `T`.
///
/// `D = 4` (default) uses the binomial extension `x^4 - W`; `D = 5` (feature
/// `ext5`) uses the KoalaBear quintic trinomial extension `x^5 + x^2 - 1`.
#[derive(AlignedBorrow, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ChallengeExtension<T>(pub [T; D]);

impl<T> ChallengeExtension<T> {
    /// Creates a new challenge extension element from a base element.
    pub fn from_base(b: T) -> Self
    where
        T: AbstractField,
    {
        let mut arr: [T; D] = core::array::from_fn(|_| T::zero());
        arr[0] = b;
        Self(arr)
    }

    /// Returns a reference to the underlying slice.
    pub const fn as_base_slice(&self) -> &[T] {
        &self.0
    }

    /// Creates a new challenge extension element from a challenge extension element.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from<S: Into<T> + Clone>(from: ChallengeExtension<S>) -> Self {
        ChallengeExtension(core::array::from_fn(|i| from.0[i].clone().into()))
    }
}

impl<T: Add<Output = T> + Clone> Add for ChallengeExtension<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(core::array::from_fn(|i| self.0[i].clone() + rhs.0[i].clone()))
    }
}

impl<T: Sub<Output = T> + Clone> Sub for ChallengeExtension<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(core::array::from_fn(|i| self.0[i].clone() - rhs.0[i].clone()))
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AbstractField> Mul
    for ChallengeExtension<T>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "ext5")]
        {
            // Quintic trinomial reduction modulo x^5 + x^2 - 1, i.e.
            //   x^5 = 1 - x^2, applied as x^k = x^{k-5} - x^{k-3} for k >= 5.
            let mut scratch: [T; 2 * D - 1] = core::array::from_fn(|_| T::zero());

            for i in 0..D {
                for j in 0..D {
                    scratch[i + j] = scratch[i + j].clone() + self.0[i].clone() * rhs.0[j].clone();
                }
            }

            for k in (D..(2 * D - 1)).rev() {
                let c = scratch[k].clone();
                scratch[k - D] = scratch[k - D].clone() + c.clone();
                scratch[k - D + 2] = scratch[k - D + 2].clone() - c;
            }

            Self(core::array::from_fn(|i| scratch[i].clone()))
        }

        #[cfg(not(feature = "ext5"))]
        {
            let mut result: [T; D] = core::array::from_fn(|_| T::zero());
            // The irreducible polynomial is `x^4 - W` where W depends on the base field:
            //   BabyBear:  W = 11  (x^4 - 11)
            //   KoalaBear: W = 3   (x^4 - 3)
            #[cfg(feature = "babybear")]
            let w = T::from_canonical_u32(11);
            #[cfg(feature = "koalabear")]
            let w = T::from_canonical_u32(3);

            for i in 0..D {
                for j in 0..D {
                    if i + j >= D {
                        result[i + j - D] = result[i + j - D].clone() +
                            w.clone() * self.0[i].clone() * rhs.0[j].clone();
                    } else {
                        result[i + j] =
                            result[i + j].clone() + self.0[i].clone() * rhs.0[j].clone();
                    }
                }
            }

            Self(result)
        }
    }
}

/// Extension field * base field: [x0*b, x1*b, ...].
/// This avoids the overhead of converting the base element to an extension element
/// and performing a full extension multiplication.
impl<T: Mul<Output = T> + Clone> Mul<T> for ChallengeExtension<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(core::array::from_fn(|i| self.0[i].clone() * rhs.clone()))
    }
}

#[cfg(not(feature = "ext5"))]
impl<F: BinomiallyExtendable<D>> Div for ChallengeExtension<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let p3_ef_lhs = BinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_rhs = BinomialExtensionField::from_base_slice(&rhs.0);
        let p3_ef_result = p3_ef_lhs / p3_ef_rhs;
        Self(p3_ef_result.as_base_slice().try_into().unwrap())
    }
}

#[cfg(feature = "ext5")]
impl<F: QuinticTrinomialExtendable> Div for ChallengeExtension<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let p3_ef_lhs = QuinticTrinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_rhs = QuinticTrinomialExtensionField::from_base_slice(&rhs.0);
        let p3_ef_result = p3_ef_lhs / p3_ef_rhs;
        Self(p3_ef_result.as_base_slice().try_into().unwrap())
    }
}

#[cfg(not(feature = "ext5"))]
impl<F: BinomiallyExtendable<D>> ChallengeExtension<F> {
    /// Returns the multiplicative inverse of the element.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let p3_ef = BinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_inverse = p3_ef.inverse();
        Self(p3_ef_inverse.as_base_slice().try_into().unwrap())
    }

    /// Returns the multiplicative inverse of the element, if it exists.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let p3_ef = BinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_inverse = p3_ef.try_inverse()?;
        Some(Self(p3_ef_inverse.as_base_slice().try_into().unwrap()))
    }
}

#[cfg(feature = "ext5")]
impl<F: QuinticTrinomialExtendable> ChallengeExtension<F> {
    /// Returns the multiplicative inverse of the element.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let p3_ef = QuinticTrinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_inverse = p3_ef.inverse();
        Self(p3_ef_inverse.as_base_slice().try_into().unwrap())
    }

    /// Returns the multiplicative inverse of the element, if it exists.
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let p3_ef = QuinticTrinomialExtensionField::from_base_slice(&self.0);
        let p3_ef_inverse = p3_ef.try_inverse()?;
        Some(Self(p3_ef_inverse.as_base_slice().try_into().unwrap()))
    }
}

impl<T: AbstractField + Copy> Neg for ChallengeExtension<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(core::array::from_fn(|i| -self.0[i]))
    }
}

#[cfg(not(feature = "ext5"))]
impl<AF> From<BinomialExtensionField<AF, D>> for ChallengeExtension<AF>
where
    AF: AbstractField + Copy,
    AF::F: BinomiallyExtendable<D>,
{
    fn from(value: BinomialExtensionField<AF, D>) -> Self {
        let arr: [AF; D] = value.as_base_slice().try_into().unwrap();
        Self(arr)
    }
}

#[cfg(not(feature = "ext5"))]
impl<AF> From<ChallengeExtension<AF>> for BinomialExtensionField<AF, D>
where
    AF: AbstractField + Copy,
    AF::F: BinomiallyExtendable<D>,
{
    fn from(value: ChallengeExtension<AF>) -> Self {
        BinomialExtensionField::from_base_slice(&value.0)
    }
}

#[cfg(feature = "ext5")]
impl<AF> From<QuinticTrinomialExtensionField<AF>> for ChallengeExtension<AF>
where
    AF: AbstractField + Copy,
    AF::F: QuinticTrinomialExtendable,
{
    fn from(value: QuinticTrinomialExtensionField<AF>) -> Self {
        let arr: [AF; D] = value.as_base_slice().try_into().unwrap();
        Self(arr)
    }
}

#[cfg(feature = "ext5")]
impl<AF> From<ChallengeExtension<AF>> for QuinticTrinomialExtensionField<AF>
where
    AF: AbstractField + Copy,
    AF::F: QuinticTrinomialExtendable,
{
    fn from(value: ChallengeExtension<AF>) -> Self {
        QuinticTrinomialExtensionField::from_base_slice(&value.0)
    }
}

impl<T> IntoIterator for ChallengeExtension<T> {
    type Item = T;
    type IntoIter = core::array::IntoIter<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Backwards-compatible alias. Historically this type was named
/// `BinomialExtension`; it now models the active challenge extension (binomial
/// quartic by default, quintic trinomial under feature `ext5`).
pub type BinomialExtension<T> = ChallengeExtension<T>;
