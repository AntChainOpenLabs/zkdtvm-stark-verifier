//! Elliptic Curve digests with a starting point to avoid weierstrass addition exceptions.
use crate::{
    septic_curve::SepticCurve, septic_curve_params::SepticCurveParams,
    septic_extension::SepticExtension,
};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// The x-coordinate for a curve point used as a starting cumulative sum for global permutation
/// trace generation, derived from `sqrt(2)`.
pub const CURVE_CUMULATIVE_SUM_START_X: [u32; 7] =
    [0x1434213, 0x5623730, 0x9504880, 0x1688724, 0x2096980, 0x7856967, 0x1875376];

/// The y-coordinate for a curve point used as a starting cumulative sum for global permutation
/// trace generation, derived from `sqrt(2)`.
pub const CURVE_CUMULATIVE_SUM_START_Y: [u32; 7] =
    [885797405, 1130275556, 567836311, 52700240, 239639200, 442612155, 1839439733];

/// The x-coordinate for a curve point used as a starting random point for digest accumulation,
/// derived from `sqrt(3)`.
pub const DIGEST_SUM_START_X: [u32; 7] =
    [0x1742050, 0x8075688, 0x7729352, 0x7446341, 0x5058723, 0x6694280, 0x5253810];

/// The y-coordinate for a curve point used as a starting random point for digest accumulation,
/// derived from `sqrt(3)`.
pub const DIGEST_SUM_START_Y: [u32; 7] =
    [462194069, 1842131493, 281651264, 1684885851, 483907222, 1097389352, 1648978901];

/// A global cumulative sum digest, a point on the elliptic curve that `SepticCurve<F>` represents.
/// As these digests start with the `CURVE_CUMULATIVE_SUM_START` point, they require special summing
/// logic.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SepticDigest<F>(pub SepticCurve<F>);

/// Converts the elements in digest to extension field elements.
pub fn digest_to_extend<F: Field, EF: ExtensionField<F>>(
    digest: SepticDigest<F>,
) -> SepticDigest<EF> {
    SepticDigest(SepticCurve {
        x: SepticExtension(digest.0.x.0.map(|x| EF::from_base(x))),
        y: SepticExtension(digest.0.y.0.map(|x| EF::from_base(x))),
    })
}

impl<F: AbstractField> SepticDigest<F> {
    /// Constructs a `SepticDigest` from a slice of at least 14 field elements.
    ///
    /// The first 7 elements are the x-coordinate, the next 7 are the y-coordinate.
    /// This is a pure data construction that does not depend on any curve parameters.
    #[must_use]
    pub fn from_slice(values: &[F]) -> Self {
        SepticDigest(SepticCurve {
            x: SepticExtension::<F>::from_base_fn(|i| values[i].clone()),
            y: SepticExtension::<F>::from_base_fn(|i| values[i + 7].clone()),
        })
    }

    #[must_use]
    /// The zero digest, the starting point of the accumulation of curve points derived from the
    /// scheme. Uses `KoalaBear` constants when the `koalabear` feature is enabled, otherwise
    /// `BabyBear` constants.
    pub fn zero() -> Self {
        use crate::septic_curve_params::KoalaBearCurveParams;
        Self::zero_generic::<KoalaBearCurveParams>()
    }

    #[must_use]
    /// The zero digest using generic curve parameters.
    pub fn zero_generic<P: SepticCurveParams>() -> Self {
        SepticDigest(SepticCurve {
            x: SepticExtension::<F>::from_base_fn(|i| {
                F::from_canonical_u32(P::CURVE_CUMULATIVE_SUM_START_X[i])
            }),
            y: SepticExtension::<F>::from_base_fn(|i| {
                F::from_canonical_u32(P::CURVE_CUMULATIVE_SUM_START_Y[i])
            }),
        })
    }

    #[must_use]
    /// The digest used for starting the accumulation of digests. Uses `KoalaBear` constants when
    /// the `koalabear` feature is enabled, otherwise `BabyBear` constants.
    pub fn starting_digest() -> Self {
        use crate::septic_curve_params::KoalaBearCurveParams;
        Self::starting_digest_generic::<KoalaBearCurveParams>()
    }

    #[must_use]
    /// The digest used for starting the accumulation of digests using generic curve parameters.
    pub fn starting_digest_generic<P: SepticCurveParams>() -> Self {
        SepticDigest(SepticCurve {
            x: SepticExtension::<F>::from_base_fn(|i| {
                F::from_canonical_u32(P::DIGEST_SUM_START_X[i])
            }),
            y: SepticExtension::<F>::from_base_fn(|i| {
                F::from_canonical_u32(P::DIGEST_SUM_START_Y[i])
            }),
        })
    }
}

impl<F: Field> SepticDigest<F> {
    /// Returns the zero digest appropriate for the current field.
    #[must_use]
    pub fn zero_for_field() -> Self {
        use crate::septic_curve_params::{BabyBearCurveParams, KoalaBearCurveParams};
        // Detect field: BabyBear has p-1 = 2013265920.
        let is_babybear = F::from_canonical_u32(2013265920u32) == F::neg_one();
        if is_babybear {
            Self::zero_generic::<BabyBearCurveParams>()
        } else {
            Self::zero_generic::<KoalaBearCurveParams>()
        }
    }

    /// Returns the starting digest appropriate for the current field.
    #[must_use]
    pub fn starting_digest_for_field() -> Self {
        use crate::septic_curve_params::{BabyBearCurveParams, KoalaBearCurveParams};
        // Detect field: BabyBear has p-1 = 2013265920.
        let is_babybear = F::from_canonical_u32(2013265920u32) == F::neg_one();
        if is_babybear {
            Self::starting_digest_generic::<BabyBearCurveParams>()
        } else {
            Self::starting_digest_generic::<KoalaBearCurveParams>()
        }
    }

    /// Checks that the digest is zero, the starting point of the accumulation.
    pub fn is_zero(&self) -> bool {
        *self == Self::zero_for_field()
    }
}

impl<F: Field> Sum for SepticDigest<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = SepticDigest::<F>::zero_for_field().0;
        let start = SepticDigest::<F>::starting_digest_for_field().0;

        // Computation order is start + (digest1 - offset) + (digest2 - offset) + ... + (digestN -
        // offset) + offset - start.
        let mut ret = iter.fold(start, |acc, x| {
            let sum_offset = acc.add_incomplete(x.0);
            sum_offset.sub_incomplete(zero)
        });

        ret.add_assign(zero);
        ret.sub_assign(start);
        SepticDigest(ret)
    }
}
