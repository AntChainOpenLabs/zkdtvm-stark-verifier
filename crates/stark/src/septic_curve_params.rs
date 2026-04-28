//! Generic septic curve parameters trait for supporting multiple field configurations.
//!
//! This module provides a trait-based abstraction for septic elliptic curve parameters,
//! allowing the same curve operations to work transparently with different field configurations
//! (e.g., `BabyBear` with irreducible polynomial `z^7 - 2z - 5` and `KoalaBear` with `z^7 - 3z - 5`).

/// Trait defining the curve parameters for a specific field configuration.
///
/// Each field configuration (`BabyBear`, `KoalaBear`, etc.) implements this trait
/// to specify the unique curve constants and coefficients for that field.
pub trait SepticCurveParams: Send + Sync {
    /// The irreducible polynomial coefficients.
    ///
    /// For `z^7 - 2z - 5`, this would be `[-5, -2, 0, 0, 0, 0, 0]`
    /// For `z^7 - 3z - 5`, this would be `[-5, -3, 0, 0, 0, 0, 0]`
    const POLY_COEFFS: [i32; 7];

    /// The curve equation y^2 = x^3 + ax + b*`z^CURVE_B_Z_INDEX`
    const CURVE_A: u16;
    const CURVE_B_CONST: u32;
    /// The index of z in the constant term of the curve equation.
    /// `BabyBear`: 5 (for 26z^5), `KoalaBear`: 3 (for 41z^3).
    const CURVE_B_Z_INDEX: usize;

    /// Dummy point coordinates used as a witness for padding interactions.
    /// Derived from mathematical constants (e, sqrt(2), sqrt(3), etc.)
    const CURVE_WITNESS_DUMMY_POINT_X: [u32; 7];
    const CURVE_WITNESS_DUMMY_POINT_Y: [u32; 7];

    /// Starting point for cumulative sum in global permutation trace generation.
    /// Derived from sqrt(2).
    const CURVE_CUMULATIVE_SUM_START_X: [u32; 7];
    const CURVE_CUMULATIVE_SUM_START_Y: [u32; 7];

    /// Starting point for digest accumulation.
    /// Derived from sqrt(3).
    const DIGEST_SUM_START_X: [u32; 7];
    const DIGEST_SUM_START_Y: [u32; 7];

    /// Descriptive name for this field configuration.
    const NAME: &'static str;
}

/// `BabyBear` field configuration parameters.
///
/// Uses irreducible polynomial `z^7 - 2z - 5` over `BabyBear` prime field.
pub struct BabyBearCurveParams;

impl SepticCurveParams for BabyBearCurveParams {
    const POLY_COEFFS: [i32; 7] = [-5, -2, 0, 0, 0, 0, 0];
    const CURVE_A: u16 = 2;
    const CURVE_B_CONST: u32 = 26;
    const CURVE_B_Z_INDEX: usize = 5;

    const CURVE_WITNESS_DUMMY_POINT_X: [u32; 7] =
        [0x2738281, 0x8284590, 0x4523536, 0x0287471, 0x3526624, 0x9775724, 0x7093699];
    const CURVE_WITNESS_DUMMY_POINT_Y: [u32; 7] =
        [48041908, 550064556, 415267377, 1726976249, 1253299140, 209439863, 1302309485];

    const CURVE_CUMULATIVE_SUM_START_X: [u32; 7] =
        [0x1434213, 0x5623730, 0x9504880, 0x1688724, 0x2096980, 0x7856967, 0x1875376];
    const CURVE_CUMULATIVE_SUM_START_Y: [u32; 7] =
        [885797405, 1130275556, 567836311, 52700240, 239639200, 442612155, 1839439733];

    const DIGEST_SUM_START_X: [u32; 7] =
        [0x1742050, 0x8075688, 0x7729352, 0x7446341, 0x5058723, 0x6694280, 0x5253810];
    const DIGEST_SUM_START_Y: [u32; 7] =
        [462194069, 1842131493, 281651264, 1684885851, 483907222, 1097389352, 1648978901];

    const NAME: &'static str = "BabyBear";
}

/// `KoalaBear` field configuration parameters.
///
/// Uses irreducible polynomial `z^7 - 3z - 5` over `KoalaBear` prime field.
pub struct KoalaBearCurveParams;

impl SepticCurveParams for KoalaBearCurveParams {
    const POLY_COEFFS: [i32; 7] = [-5, -3, 0, 0, 0, 0, 0];
    const CURVE_A: u16 = 45;
    const CURVE_B_CONST: u32 = 41;
    const CURVE_B_Z_INDEX: usize = 3;

    const CURVE_WITNESS_DUMMY_POINT_X: [u32; 7] =
        [0x2718281 + (1 << 24), 0x8284590, 0x4523536, 0x0287471, 0x3526624, 0x9775724, 0x7093699];
    const CURVE_WITNESS_DUMMY_POINT_Y: [u32; 7] =
        [1250555984, 1592495468, 656721246, 420301347, 2125819749, 819876460, 17687681];

    const CURVE_CUMULATIVE_SUM_START_X: [u32; 7] =
        [0x1414213, 0x5623730, 0x9504880, 0x1688724, 0x2096980, 0x7856967, 0x1875376];
    const CURVE_CUMULATIVE_SUM_START_Y: [u32; 7] =
        [2020310104, 1513506566, 1843922297, 2003644209, 805967281, 1882435203, 1623804682];

    const DIGEST_SUM_START_X: [u32; 7] =
        [0x1732050, 0x8075688, 0x7729352, 0x7446341, 0x5058723, 0x6694280, 0x5253810];
    const DIGEST_SUM_START_Y: [u32; 7] =
        [1095433104, 7540207, 1124564165, 2035506693, 11121645, 102781365, 398772161];

    const NAME: &'static str = "KoalaBear";
}
