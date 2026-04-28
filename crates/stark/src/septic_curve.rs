//! Elliptic Curve `y^2 = x^3 + ax + b*z^k` over the `F_{p^7}` extension field.
//!
//! For `BabyBear`: `y^2 = x^3 + 2x + 26z^5` over `F_{p^7} = F_p[z]/(z^7 - 2z - 5)`.
//! For `KoalaBear`: `y^2 = x^3 + 45x + 41z^3` over `F_{p^7} = F_p[z]/(z^7 - 3z - 5)`.
use crate::{septic_curve_params::SepticCurveParams, septic_extension::SepticExtension};
use dt_primitives::{koalabear_poseidon2_init, poseidon2_init};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_symmetric::Permutation;
use serde::{Deserialize, Serialize};
use std::ops::Add;

/// A septic elliptic curve point on y^2 = x^3 + 2x + 26z^5 over field `F_{p^7} = F_p[z]/(z^7 - 2z -
/// 5)`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SepticCurve<F> {
    /// The x-coordinate of an elliptic curve point.
    pub x: SepticExtension<F>,
    /// The y-coordinate of an elliptic curve point.
    pub y: SepticExtension<F>,
}

impl<F: Field> SepticCurve<F> {
    /// Converts field elements to extension field elements.
    pub fn convert_to_ext<EF: ExtensionField<F>>(&self, scale: EF) -> SepticCurve<EF> {
        SepticCurve { x: self.x.convert_to_ext(scale), y: self.y.convert_to_ext(scale) }
    }
}

/// The x-coordinate for a curve point used as a witness for padding interactions, derived from `e`.
pub const CURVE_WITNESS_DUMMY_POINT_X: [u32; 7] =
    [0x2738281, 0x8284590, 0x4523536, 0x0287471, 0x3526624, 0x9775724, 0x7093699];

/// The y-coordinate for a curve point used as a witness for padding interactions, derived from `e`.
pub const CURVE_WITNESS_DUMMY_POINT_Y: [u32; 7] =
    [48041908, 550064556, 415267377, 1726976249, 1253299140, 209439863, 1302309485];

impl<F: Field> SepticCurve<F> {
    /// Returns the dummy point. Uses `KoalaBear` constants when the `koalabear` feature is enabled,
    /// otherwise `BabyBear` constants.
    #[must_use]
    pub fn dummy() -> Self {
        use crate::septic_curve_params::KoalaBearCurveParams;
        Self::dummy_generic::<KoalaBearCurveParams>()
    }

    /// Returns the dummy point using generic curve parameters.
    #[must_use]
    pub fn dummy_generic<P: SepticCurveParams>() -> Self {
        Self {
            x: SepticExtension::from_base_fn(|i| {
                F::from_canonical_u32(P::CURVE_WITNESS_DUMMY_POINT_X[i])
            }),
            y: SepticExtension::from_base_fn(|i| {
                F::from_canonical_u32(P::CURVE_WITNESS_DUMMY_POINT_Y[i])
            }),
        }
    }

    /// Returns the dummy point appropriate for the current field (`BabyBear` or `KoalaBear`).
    #[must_use]
    pub fn dummy_for_field() -> Self {
        use crate::septic_curve_params::{BabyBearCurveParams, KoalaBearCurveParams};
        // Detect field: BabyBear has p-1 = 2013265920.
        let is_babybear = F::from_canonical_u32(2013265920u32) == F::neg_one();
        if is_babybear {
            Self::dummy_generic::<BabyBearCurveParams>()
        } else {
            Self::dummy_generic::<KoalaBearCurveParams>()
        }
    }

    /// Check if a `SepticCurve` struct is on the elliptic curve (`BabyBear` default).
    pub fn check_on_point(&self) -> bool {
        self.y.square() == Self::curve_formula(self.x)
    }

    /// Check if a `SepticCurve` struct is on the elliptic curve using generic parameters.
    pub fn check_on_point_generic<P: SepticCurveParams>(&self) -> bool {
        self.y.square_generic::<P>() == Self::curve_formula_generic::<P>(self.x)
    }

    /// Negates a `SepticCurve` point.
    #[must_use]
    pub fn neg(&self) -> Self {
        SepticCurve { x: self.x, y: -self.y }
    }

    #[must_use]
    /// Adds two elliptic curve points, assuming that the addition doesn't lead to the exception
    /// cases of weierstrass addition.
    pub fn add_incomplete(&self, other: SepticCurve<F>) -> Self {
        let slope = (other.y - self.y) / (other.x - self.x);
        let result_x = slope.square() - self.x - other.x;
        let result_y = slope * (self.x - result_x) - self.y;
        Self { x: result_x, y: result_y }
    }

    /// Add assigns an elliptic curve point, assuming that the addition doesn't lead to the
    /// exception cases of weierstrass addition.
    pub fn add_assign(&mut self, other: SepticCurve<F>) {
        let result = self.add_incomplete(other);
        self.x = result.x;
        self.y = result.y;
    }

    #[must_use]
    /// Double the elliptic curve point (field-aware: selects a=2 for `BabyBear`, a=45 for `KoalaBear`).
    pub fn double(&self) -> Self {
        // Detect field: BabyBear has p-1 = 2013265920.
        let is_babybear = F::from_canonical_u32(2013265920u32) == F::neg_one();
        let curve_a = if is_babybear {
            F::two() // BabyBear: a = 2
        } else {
            F::from_canonical_u16(45) // KoalaBear: a = 45
        };
        let slope = (self.x * self.x * F::from_canonical_u8(3u8) + curve_a) / (self.y * F::two());
        let result_x = slope.square() - self.x * F::two();
        let result_y = slope * (self.x - result_x) - self.y;
        Self { x: result_x, y: result_y }
    }

    #[must_use]
    /// Double the elliptic curve point using generic curve parameters.
    pub fn double_generic<P: SepticCurveParams>(&self) -> Self {
        let slope = (self.x * self.x * F::from_canonical_u8(3u8)
            + F::from_canonical_u16(P::CURVE_A))
            / (self.y * F::two());
        let result_x = slope.square() - self.x * F::two();
        let result_y = slope * (self.x - result_x) - self.y;
        Self { x: result_x, y: result_y }
    }

    /// Subtracts two elliptic curve points, assuming that the subtraction doesn't lead to the
    /// exception cases of weierstrass addition.
    #[must_use]
    pub fn sub_incomplete(&self, other: SepticCurve<F>) -> Self {
        self.add_incomplete(other.neg())
    }

    /// Subtract assigns an elliptic curve point, assuming that the subtraction doesn't lead to the
    /// exception cases of weierstrass addition.
    pub fn sub_assign(&mut self, other: SepticCurve<F>) {
        let result = self.add_incomplete(other.neg());
        self.x = result.x;
        self.y = result.y;
    }
}

impl<F: AbstractField> SepticCurve<F> {
    /// Evaluates the curve formula x^3 + 2x + 26z^5 (`BabyBear` default).
    pub fn curve_formula(x: SepticExtension<F>) -> SepticExtension<F> {
        x.cube()
            + x * F::two()
            + SepticExtension::from_base_slice(&[
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::from_canonical_u32(26),
                F::zero(),
            ])
    }

    /// Evaluates the curve formula `x^3 + ax + b*z^CURVE_B_Z_INDEX` using generic curve parameters.
    ///
    /// The coefficients `a`, `b`, and the z-index are taken from the `SepticCurveParams`
    /// implementation, and the multiplication uses the corresponding irreducible polynomial
    /// coefficients.
    pub fn curve_formula_generic<P: SepticCurveParams>(
        x: SepticExtension<F>,
    ) -> SepticExtension<F> {
        let mut b_term =
            [F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero(), F::zero()];
        b_term[P::CURVE_B_Z_INDEX] = F::from_canonical_u32(P::CURVE_B_CONST);
        x.cube_generic::<P>()
            + x * F::from_canonical_u16(P::CURVE_A)
            + SepticExtension::from_base_slice(&b_term)
    }

    /// Evaluates the curve formula appropriate for the current field (`BabyBear` or `KoalaBear`).
    ///
    /// Uses `F::F::order()` at runtime to select the correct curve parameters.
    pub fn curve_formula_for_field(x: SepticExtension<F>) -> SepticExtension<F> {
        use crate::septic_curve_params::{BabyBearCurveParams, KoalaBearCurveParams};
        use num_bigint::BigUint;
        // Detect field via F::F::order() divisibility by BabyBear's prime.
        // When F::F is SepticExtension<BabyBear>, order() = p^7 which is still divisible by p.
        // For KoalaBear, order() is NOT divisible by BabyBear's prime.
        let babybear_prime = BigUint::from(2013265921u32);
        let is_babybear = F::F::order() % &babybear_prime == BigUint::from(0u32);
        if is_babybear {
            Self::curve_formula_generic::<BabyBearCurveParams>(x)
        } else {
            Self::curve_formula_generic::<KoalaBearCurveParams>(x)
        }
    }
}

impl<F: Field> SepticCurve<F> {
    /// Lift an x coordinate into an elliptic curve.
    ///
    /// Automatically selects the correct Poseidon2 permutation and curve formula based on the
    /// field's order at runtime (`BabyBear` vs `KoalaBear`).
    ///
    /// As an x-coordinate may not be a valid one, we allow an additional value in `[0, 256)` to the
    /// hash input. Also, we always return the curve point with y-coordinate within `[1,
    /// (p-1)/2]`, where p is the characteristic. The returned values are the curve point, the
    /// offset used, and the hash input and output.
    pub fn lift_x(m: SepticExtension<F>) -> (Self, u8, [F; 16], [F; 16]) {
        // Detect field: BabyBear has p-1 = 2013265920.
        let is_koalabear = F::from_canonical_u32(2013265920u32) != F::neg_one();

        for offset in 0..=255 {
            let m_trial = [
                m.0[0],
                m.0[1],
                m.0[2],
                m.0[3],
                m.0[4],
                m.0[5],
                m.0[6],
                F::from_canonical_u8(offset),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
            ];

            let m_hash = if is_koalabear {
                let perm = koalabear_poseidon2_init();
                let input = m_trial.map(|x| KoalaBear::from_canonical_u32(x.as_u32()));
                perm.permute(input).map(|x| F::from_canonical_u32(x.as_canonical_u32()))
            } else {
                let perm = poseidon2_init();
                let input = m_trial.map(|x| BabyBear::from_canonical_u32(x.as_u32()));
                perm.permute(input).map(|x| F::from_canonical_u32(x.as_canonical_u32()))
            };
            let x_trial = SepticExtension(m_hash[..7].try_into().unwrap());

            let y_sq = if is_koalabear {
                Self::curve_formula_generic::<crate::septic_curve_params::KoalaBearCurveParams>(
                    x_trial,
                )
            } else {
                Self::curve_formula(x_trial)
            };
            if let Some(y) = y_sq.sqrt() {
                if y.is_exception() {
                    continue;
                }
                if y.is_send() {
                    return (Self { x: x_trial, y: -y }, offset, m_trial, m_hash);
                }
                return (Self { x: x_trial, y }, offset, m_trial, m_hash);
            }
        }
        panic!("curve point couldn't be found after 256 attempts");
    }

    /// Lift an x coordinate into an elliptic curve for `KoalaBear`.
    ///
    /// Unlike the `BabyBear` `lift_x` which takes 7 field elements (`SepticExtension<F>`),
    /// the `KoalaBear` version takes 8 field elements (`[F; 8]`). The 8th element is combined
    /// with the offset in `m_trial[7] = m[7] + (1 << 16) * offset`.
    pub fn lift_x_kb(m: [F; 8]) -> (Self, u8, [F; 16], [F; 16]) {
        let perm = koalabear_poseidon2_init();
        for offset in 0..=255 {
            let m_trial = [
                m[0],
                m[1],
                m[2],
                m[3],
                m[4],
                m[5],
                m[6],
                m[7] + F::from_canonical_u32(1 << 16) * F::from_canonical_u8(offset),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
            ];

            let m_hash = {
                let input = m_trial.map(|x| KoalaBear::from_canonical_u32(x.as_u32()));
                perm.permute(input).map(|x| F::from_canonical_u32(x.as_canonical_u32()))
            };
            let x_trial = SepticExtension(m_hash[..7].try_into().unwrap());

            let y_sq = Self::curve_formula_generic::<
                crate::septic_curve_params::KoalaBearCurveParams,
            >(x_trial);
            if let Some(y) = y_sq.sqrt() {
                if y.is_exception() {
                    continue;
                }
                if y.is_send() {
                    return (Self { x: x_trial, y: -y }, offset, m_trial, m_hash);
                }
                return (Self { x: x_trial, y }, offset, m_trial, m_hash);
            }
        }
        panic!("curve point couldn't be found after 256 attempts");
    }

    /// Lift an x coordinate into an elliptic curve using generic curve parameters and permutation.
    ///
    /// This is the fully generic version of `lift_x` that works with any field configuration
    /// by accepting a permutation and using `SepticCurveParams` for curve constants.
    pub fn lift_x_generic<P, Perm>(
        m: SepticExtension<F>,
        perm: &Perm,
    ) -> (Self, u8, [F; 16], [F; 16])
    where
        P: SepticCurveParams,
        Perm: Permutation<[F; 16]>,
    {
        for offset in 0..=255 {
            let m_trial = [
                m.0[0],
                m.0[1],
                m.0[2],
                m.0[3],
                m.0[4],
                m.0[5],
                m.0[6],
                F::from_canonical_u8(offset),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
                F::zero(),
            ];

            let m_hash = perm.permute(m_trial);
            let x_trial = SepticExtension(m_hash[..7].try_into().unwrap());

            let y_sq = Self::curve_formula_generic::<P>(x_trial);
            if let Some(y) = y_sq.sqrt() {
                if y.is_exception() {
                    continue;
                }
                if y.is_send() {
                    return (Self { x: x_trial, y: -y }, offset, m_trial, m_hash);
                }
                return (Self { x: x_trial, y }, offset, m_trial, m_hash);
            }
        }
        panic!("curve point couldn't be found after 256 attempts");
    }
}

impl<F: AbstractField> SepticCurve<F> {
    /// Given three points p1, p2, p3, the function is zero if and only if p3.x == (p1 + p2).x
    /// assuming that no weierstrass edge cases occur.
    pub fn sum_checker_x(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        (p1.x.clone() + p2.x.clone() + p3.x) * (p2.x.clone() - p1.x.clone()).square()
            - (p2.y - p1.y).square()
    }

    /// Given three points p1, p2, p3, the function is zero if and only if p3.y == (p1 + p2).y
    /// assuming that no weierstrass edge cases occur.
    pub fn sum_checker_y(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        (p1.y.clone() + p3.y.clone()) * (p2.x.clone() - p1.x.clone())
            - (p2.y - p1.y.clone()) * (p1.x - p3.x)
    }

    /// Generic version of `sum_checker_x` using explicit curve parameters for multiplication.
    ///
    /// This avoids the runtime `F::F::order()` check in `SepticExtension::Mul` which incorrectly
    /// returns `p^7` instead of `p`, causing `KoalaBear` to use `BabyBear`'s irreducible polynomial.
    pub fn sum_checker_x_generic<P: SepticCurveParams>(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        let dx = p2.x.clone() - p1.x.clone();
        let dy = p2.y - p1.y;
        (p1.x + p2.x + p3.x).mul_generic::<P>(&dx.square_generic::<P>()) - dy.square_generic::<P>()
    }

    /// Generic version of `sum_checker_y` using explicit curve parameters for multiplication.
    pub fn sum_checker_y_generic<P: SepticCurveParams>(
        p1: SepticCurve<F>,
        p2: SepticCurve<F>,
        p3: SepticCurve<F>,
    ) -> SepticExtension<F> {
        let dx = p2.x.clone() - p1.x.clone();
        let dy = p2.y - p1.y.clone();
        (p1.y + p3.y).mul_generic::<P>(&dx) - dy.mul_generic::<P>(&(p1.x - p3.x))
    }
}

impl<T> SepticCurve<T> {
    /// Convert a `SepticCurve<S>` into `SepticCurve<T>`, with a map that implements `FnMut(S) ->
    /// T`.
    pub fn convert<S: Copy, G: FnMut(S) -> T>(point: SepticCurve<S>, mut f: G) -> Self {
        SepticCurve {
            x: SepticExtension(point.x.0.map(&mut f)),
            y: SepticExtension(point.y.0.map(&mut f)),
        }
    }
}

/// A septic elliptic curve point on y^2 = x^3 + 2x + 26z^5 over field `F_{p^7} = F_p[z]/(z^7 - 2z -
/// 5)`, including the point at infinity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SepticCurveComplete<T> {
    /// The point at infinity.
    Infinity,
    /// The affine point which can be represented with a `SepticCurve<T>` structure.
    Affine(SepticCurve<T>),
}

impl<F: Field> Add for SepticCurveComplete<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_infinity() {
            return rhs;
        }
        if rhs.is_infinity() {
            return self;
        }
        let point1 = self.point();
        let point2 = rhs.point();
        if point1.x != point2.x {
            return Self::Affine(point1.add_incomplete(point2));
        }
        if point1.y == point2.y {
            return Self::Affine(point1.double());
        }
        Self::Infinity
    }
}

impl<F: Field> SepticCurveComplete<F> {
    /// Returns whether or not the point is a point at infinity.
    pub fn is_infinity(&self) -> bool {
        match self {
            Self::Infinity => true,
            Self::Affine(_) => false,
        }
    }

    /// Asserts that the point is not a point at infinity, and returns the `SepticCurve` value.
    pub fn point(&self) -> SepticCurve<F> {
        match self {
            Self::Infinity => panic!("point() called for point at infinity"),
            Self::Affine(point) => *point,
        }
    }
}
