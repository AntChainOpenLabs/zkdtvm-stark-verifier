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
        #[cfg(feature = "babybear")]
        {
            SepticDigest(SepticCurve {
                x: SepticExtension::<F>::from_base_fn(|i| {
                    F::from_canonical_u32(CURVE_CUMULATIVE_SUM_START_X[i])
                }),
                y: SepticExtension::<F>::from_base_fn(|i| {
                    F::from_canonical_u32(CURVE_CUMULATIVE_SUM_START_Y[i])
                }),
            })
        }
        #[cfg(feature = "koalabear")]
        {
            use crate::septic_curve_params::KoalaBearCurveParams;
            Self::zero_generic::<KoalaBearCurveParams>()
        }
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
        #[cfg(feature = "babybear")]
        {
            SepticDigest(SepticCurve {
                x: SepticExtension::<F>::from_base_fn(|i| {
                    F::from_canonical_u32(DIGEST_SUM_START_X[i])
                }),
                y: SepticExtension::<F>::from_base_fn(|i| {
                    F::from_canonical_u32(DIGEST_SUM_START_Y[i])
                }),
            })
        }
        #[cfg(feature = "koalabear")]
        {
            use crate::septic_curve_params::KoalaBearCurveParams;
            Self::starting_digest_generic::<KoalaBearCurveParams>()
        }
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

#[cfg(test)]
mod test {
    use crate::{
        septic_curve::{CURVE_WITNESS_DUMMY_POINT_X, CURVE_WITNESS_DUMMY_POINT_Y},
        septic_curve_params::{BabyBearCurveParams, KoalaBearCurveParams},
    };

    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_koala_bear::KoalaBear;

    #[test]
    fn test_const_points() {
        let x: SepticExtension<BabyBear> = SepticExtension::from_base_fn(|i| {
            BabyBear::from_canonical_u32(CURVE_CUMULATIVE_SUM_START_X[i])
        });
        let y: SepticExtension<BabyBear> = SepticExtension::from_base_fn(|i| {
            BabyBear::from_canonical_u32(CURVE_CUMULATIVE_SUM_START_Y[i])
        });
        let point = SepticCurve { x, y };
        assert!(point.check_on_point());
        let x: SepticExtension<BabyBear> =
            SepticExtension::from_base_fn(|i| BabyBear::from_canonical_u32(DIGEST_SUM_START_X[i]));
        let y: SepticExtension<BabyBear> =
            SepticExtension::from_base_fn(|i| BabyBear::from_canonical_u32(DIGEST_SUM_START_Y[i]));
        let point = SepticCurve { x, y };
        assert!(point.check_on_point());
        let x: SepticExtension<BabyBear> = SepticExtension::from_base_fn(|i| {
            BabyBear::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_X[i])
        });
        let y: SepticExtension<BabyBear> = SepticExtension::from_base_fn(|i| {
            BabyBear::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_Y[i])
        });
        let point = SepticCurve { x, y };
        assert!(point.check_on_point());
    }

    #[test]
    fn test_koalabear_const_points_on_curve() {
        // Verify KoalaBear CURVE_CUMULATIVE_SUM_START is on y^2 = x^3 + 45x + 41z^5
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::CURVE_CUMULATIVE_SUM_START_X[i])
        });
        let y: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::CURVE_CUMULATIVE_SUM_START_Y[i])
        });
        let point = SepticCurve { x, y };
        let y_sq = point.y * point.y;
        let curve_rhs = SepticCurve::curve_formula_for_field(point.x);
        assert_eq!(y_sq, curve_rhs, "KoalaBear CURVE_CUMULATIVE_SUM_START not on curve");

        // Verify KoalaBear DIGEST_SUM_START is on the curve
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::DIGEST_SUM_START_X[i])
        });
        let y: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::DIGEST_SUM_START_Y[i])
        });
        let point = SepticCurve { x, y };
        let y_sq = point.y * point.y;
        let curve_rhs = SepticCurve::curve_formula_for_field(point.x);
        assert_eq!(y_sq, curve_rhs, "KoalaBear DIGEST_SUM_START not on curve");

        // Verify KoalaBear CURVE_WITNESS_DUMMY_POINT is on the curve
        let x: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::CURVE_WITNESS_DUMMY_POINT_X[i])
        });
        let y: SepticExtension<KoalaBear> = SepticExtension::from_base_fn(|i| {
            KoalaBear::from_canonical_u32(KoalaBearCurveParams::CURVE_WITNESS_DUMMY_POINT_Y[i])
        });
        let point = SepticCurve { x, y };
        let y_sq = point.y * point.y;
        let curve_rhs = SepticCurve::curve_formula_for_field(point.x);
        assert_eq!(y_sq, curve_rhs, "KoalaBear CURVE_WITNESS_DUMMY_POINT not on curve");
    }

    #[test]
    fn test_recompute_koalabear_const_points() {
        use p3_field::PrimeField32;

        // Helper: given a base X coordinate array, try small offsets on x[0] until
        // we find an X that lies on the KoalaBear curve (i.e., x^3+45x+41z^5 is a QR).
        fn find_valid_point(
            base_x: [u32; 7],
            name: &str,
        ) -> (SepticExtension<KoalaBear>, SepticExtension<KoalaBear>) {
            for offset in 0u32..256 {
                let mut x_vals = base_x;
                x_vals[0] = x_vals[0].wrapping_add(offset);
                let x: SepticExtension<KoalaBear> =
                    SepticExtension::from_base_fn(|i| KoalaBear::from_canonical_u32(x_vals[i]));
                let y_sq = SepticCurve::<KoalaBear>::curve_formula_for_field(x);
                if let Some(y) = y_sq.sqrt() {
                    println!("{name} (offset={offset}):");
                    println!("  X: {x_vals:?}");
                    println!(
                        "  Y: {:?}",
                        y.0.iter()
                            .map(p3_field::PrimeField32::as_canonical_u32)
                            .collect::<Vec<_>>()
                    );
                    return (x, y);
                }
            }
            panic!("{name}: no valid point found within 256 offsets");
        }

        // 1. CURVE_CUMULATIVE_SUM_START
        let (x1, y1) = find_valid_point(
            KoalaBearCurveParams::CURVE_CUMULATIVE_SUM_START_X,
            "CURVE_CUMULATIVE_SUM_START",
        );
        assert_eq!(y1 * y1, SepticCurve::<KoalaBear>::curve_formula_for_field(x1));

        // 2. DIGEST_SUM_START
        let (x2, y2) =
            find_valid_point(KoalaBearCurveParams::DIGEST_SUM_START_X, "DIGEST_SUM_START");
        assert_eq!(y2 * y2, SepticCurve::<KoalaBear>::curve_formula_for_field(x2));

        // 3. CURVE_WITNESS_DUMMY_POINT
        let (x3, y3) = find_valid_point(
            KoalaBearCurveParams::CURVE_WITNESS_DUMMY_POINT_X,
            "CURVE_WITNESS_DUMMY_POINT",
        );
        assert_eq!(y3 * y3, SepticCurve::<KoalaBear>::curve_formula_for_field(x3));
    }

    #[test]
    fn test_koalabear_zero_digest_is_zero() {
        // Verify that zero_for_field() returns the correct KoalaBear zero digest
        let zero = SepticDigest::<KoalaBear>::zero_for_field();
        let zero_generic = SepticDigest::<KoalaBear>::zero_generic::<KoalaBearCurveParams>();
        assert_eq!(zero, zero_generic, "zero_for_field should match zero_generic for KoalaBear");

        // Verify it does NOT match BabyBear's zero
        let babybear_zero = SepticDigest::<KoalaBear>::zero_generic::<BabyBearCurveParams>();
        assert_ne!(zero, babybear_zero, "KoalaBear zero should differ from BabyBear zero");
    }

    #[test]
    fn test_koalabear_digest_sum_identity() {
        // Summing zero digests should give zero
        let zero = SepticDigest::<KoalaBear>::zero_for_field();
        let sum: SepticDigest<KoalaBear> = vec![zero, zero, zero].into_iter().sum();
        assert!(sum.is_zero(), "Sum of zero digests should be zero");
    }

    #[test]
    fn test_koalabear_digest_sum_with_real_points() {
        // Create real curve points via lift_x and accumulate them as digests
        let mut digests = Vec::new();
        for i in 0..5u32 {
            let x: SepticExtension<KoalaBear> = SepticExtension::from_base_slice(&[
                KoalaBear::from_canonical_u32(i + 100),
                KoalaBear::from_canonical_u32(2 * i + 200),
                KoalaBear::from_canonical_u32(3 * i + 300),
                KoalaBear::from_canonical_u32(4 * i + 400),
                KoalaBear::from_canonical_u32(5 * i + 500),
                KoalaBear::from_canonical_u32(6 * i + 600),
                KoalaBear::from_canonical_u32(7 * i + 700),
            ]);
            let (curve_point, _, _, _) = SepticCurve::<KoalaBear>::lift_x(x);
            digests.push(SepticDigest(curve_point));
        }

        // Sum should produce a valid curve point (not panic)
        let sum: SepticDigest<KoalaBear> = digests.into_iter().sum();
        // Verify the result is on the KoalaBear curve
        let y_sq = sum.0.y * sum.0.y;
        let curve_rhs = SepticCurve::curve_formula_for_field(sum.0.x);
        assert_eq!(y_sq, curve_rhs, "Digest sum result not on KoalaBear curve");
    }
}
