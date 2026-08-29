//! Frozen D11 parameter and kernel constants.

/// The KoalaBear prime `2^31 - 2^24 + 1`.
pub const BASE_PRIME: u32 = 2_130_706_433;
/// Number of base-field coefficients in a D11 value.
pub const D11_DEGREE: usize = 11;
/// Number of coefficients in an unreduced product.
pub const D11_WIDE_DEGREE: usize = 21;
/// Number of nonzero-capable coefficients in the PackV1 sparse multiplier.
pub const D11_SPARSE_WIDTH: usize = 7;

pub const D11_SCHEME_ID: &str =
    "Projective228QIntervalV6+GlobalTileReducerV3-83+GlobalFirstRoundV2";
pub const D11_FIELD_ID: &str = "KoalaBearD11-z11-z3-2-v1";
pub const D11_PACK_ID: &str = "GlobalPackV1-tag1-tweak9";
pub const D11_MIXED_ADD_FORMULA_ID: &str = "madd-2015-rcb-simple-projective-v1";
pub const D11_FULL_ADD_FORMULA_ID: &str = "add-2015-rcb-a-minus-3-algorithm4-v1";
/// Canonical digest used to bind SDK public-value bytes in every D11 wire.
pub const D11_PUBLIC_VALUES_DIGEST_ID: &str = "sha256-v1";
pub const GLOBAL_TAG_V1: u8 = 1;
pub const TWEAK_COUNT_V1: u16 = 512;
pub const HALF_BASE_MINUS_ONE: u32 = 1_065_353_216;

/// Frozen serialized interval-V6 protocol identity. It is never used as a runtime selector.
pub const D11_PROJECTIVE_228_QDELTA_WIRE_ID: u8 = 5;

pub const CURVE_A_SIGNED: i32 = -3;
pub const CURVE_B_COEFFICIENTS: [u32; D11_DEGREE] = [36, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
pub const CURVE_ORDER_DECIMAL: &str =
    "4109223735958498767327749237445279581763589078874653919311204688072208279167564865251763222976454194979";
pub const CURVE_GENERATOR_Y: [u32; D11_DEGREE] = [
    1509524249, 1053795221, 48957795, 2063853676, 357530200, 115363041, 1569878163, 242779857,
    763639801, 1508020847, 744291015,
];

pub const PADDING_DUMMY_TWEAK: u16 = 1;
pub const PADDING_DUMMY_X: [u32; D11_DEGREE] = [0, 0, 0, 0, 0, 65536, 256, 0, 0, 0, 0];
pub const PADDING_DUMMY_CANONICAL_Y: [u32; D11_DEGREE] = [
    1276224553, 734672244, 1514828954, 2127367348, 1918436764, 315638403, 1766730550, 1786423097,
    1104622764, 490438735, 587574789,
];

pub const PARAMETER_MANIFEST_SHA256_HEX: &str =
    "2e92b828c3c0f3c4d3db1f8b4301750e4774c8519c6bc4bb047c7a2a39682267";
pub const PARAMETER_MANIFEST_SHA256: [u8; 32] = [
    0x2e, 0x92, 0xb8, 0x28, 0xc3, 0xc0, 0xf3, 0xc4, 0xd3, 0xdb, 0x1f, 0x8b, 0x43, 0x01, 0x75, 0x0e,
    0x47, 0x74, 0xc8, 0x51, 0x9c, 0x6b, 0xc4, 0xbb, 0x04, 0x7c, 0x7a, 0x2a, 0x39, 0x68, 0x22, 0x67,
];

/// Canonical JSON bytes hashed by the parameter reproduction package after
/// deleting its self-referential digest field.  Keeping these bytes in code
/// makes the replay independent of the docs tree at test/runtime.
pub const PARAMETER_MANIFEST_CANONICAL_JSON: &str = r#"{"base_field":{"p":2130706433,"p_is_prime":true,"p_minus_1_factorization":{"127":1,"2":24}},"curve":{"anomalous_r_equals_q":false,"cm_discriminant_factorization":{"109":1,"16272128514631717573081":1,"3929969":1,"4480538896833318876730795422366342439904845751958979552595946506020763":1,"67":1,"7":1},"cm_discriminant_squarefree_fundamental":true,"cofactor":1,"discriminant_coefficients_low_to_high":[2130148289,2130675329,2130706001,0,0,0,0,0,0,0,0],"equation":"y^2=x^3-3x+(z+36)","exact_order_proved_from_point_plus_hasse":true,"frobenius_cm_discriminant":-14647485702923551369199548362102332374730015346769368098442005047755216919083195399078578913554482154947,"generator_on_curve":true,"generator_y_coefficients":[1509524249,1053795221,48957795,2063853676,357530200,115363041,1569878163,242779857,763639801,1508020847,744291015],"hasse_deduction":{"ceil_sqrt_q":2027122032823504872904856514762411624809682887301937,"conservative_hasse_upper":4109223735958498767327749237445279581763589078874659311243291104664722543064309878975110125531511240292,"margin_2r_minus_upper":4109223735958498767327749237445279581763589078874648527379118271479694015270819851528416320421397149666,"two_r":8218447471916997534655498474890559163527178157749307838622409376144416558335129730503526445952908389958,"upper_lt_2r":true},"j_coefficients_low_to_high":[452452690,2037994302,1353824058,1643336798,823536449,1499740459,478147093,1533941688,872305332,814357775,989611126],"j_in_prime_field":false,"minimal_field_of_definition_degree_from_j":11,"nonsingular":true,"order_r":4109223735958498767327749237445279581763589078874653919311204688072208279167564865251763222976454194979,"ordinary":true,"r_is_prime_sympy_crosscheck":true,"r_times_generator_is_infinity":true,"trace":1337688020769582768454183715488900097283189282441439,"trace_mod_p":1083337979},"embedding_degree":{"claimed_k":2054611867979249383663874618722639790881794539437326959655602344036104139583782432625881611488227097489,"proper_divisor_checks":{"105760627133166013":true,"24049":true,"311":true,"370733185604032409914640474101721240248823195701":true,"383":true,"45564971":true,"47600533":true,"8434225587583":true},"q_to_k_mod_r":1,"verified":true},"extension":{"degree":11,"modulus":"z^11-z^3-2","q":4109223735958498767327749237445279581763589078874655256999225457654976733351280354151860506165736636417,"rabin_details":{"X_p11_mod_f_equals_X":true,"gcd_f_Xp_minus_X":[1]},"rabin_irreducible":true,"sympy_irreducible":true},"tool":{"platform":"Linux-6.12.13-x86_64-with-glibc2.41","python":"3.13.5","random_seed":null,"sympy":"1.14.0"},"twist":{"all_factors_prime_sympy_crosscheck":true,"factor_primality":{"20889046108248601":true,"2551680139474847853141941288685174810617433833212933372631050932011377434389385749":true,"77093":true},"factor_product_ok":true,"factorization":{"20889046108248601":1,"2551680139474847853141941288685174810617433833212933372631050932011377434389385749":1,"77093":1},"order":4109223735958498767327749237445279581763589078874656594687246227237745187534995843051957789355019077857,"order_formula_ok":true}}"#;

/// Column-major matrix for `z^j -> (z^j)^p`, with each inner array ordered
/// low-to-high.  It was generated independently by the reproduction field
/// oracle and is locked by external KATs.
pub const FROBENIUS_COLUMNS: [[u32; D11_DEGREE]; D11_DEGREE] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        740528895, 1543165906, 273099228, 855460459, 692227400, 1139590874, 1624567107, 140427628,
        313464290, 605969248, 1843588616,
    ],
    [
        843537357, 1185389608, 827390567, 150511649, 1427808512, 1909725054, 1664078702, 179425687,
        704504263, 1104503767, 2116480242,
    ],
    [
        334931213, 1827520639, 1050136537, 1771026751, 1465679891, 653951004, 749018818, 570517883,
        1403837711, 710900031, 364177968,
    ],
    [
        1842731591, 1894119806, 1865904935, 467151796, 1630933267, 711360379, 1939170979,
        1539831839, 928642016, 2125335178, 626792483,
    ],
    [
        1879859221, 1171747726, 2041515308, 305693142, 1158587983, 1322931250, 1939362591,
        721207783, 1410268133, 429993999, 1422953164,
    ],
    [
        655431041, 755853588, 314313271, 242473437, 652725771, 1776048174, 1835806725, 1945174618,
        2028503664, 885429429, 1688407482,
    ],
    [
        1084631026, 558142371, 906049101, 669624508, 815014905, 1279478099, 912344955, 371636054,
        106662164, 1004234549, 460761563,
    ],
    [
        1796975320, 127666930, 1273970632, 1313446273, 846148171, 1664364261, 1062412715,
        1301899604, 1790571802, 524939200, 1394855813,
    ],
    [
        1669010106, 1973576581, 1515322586, 281377, 22469209, 587215224, 1185386074, 934600757,
        1433847362, 2116188943, 975369751,
    ],
    [
        1704869879, 1513587201, 351131565, 199139715, 1826490289, 626709047, 170604677, 601527860,
        1118201870, 263419283, 1705293765,
    ],
];

/// Static delayed-reduction certificate.  It proves that canonical `u32`
/// limbs require `u128`: both dense and sparse reduced maxima exceed `u64`.
pub const OVERFLOW_CERTIFICATE_JSON: &str = r#"{"base_prime":2130706433,"dense_products":121,"dense_raw_max":49939008893027876864,"dense_reduced_max":136197296980985118720,"reduction_weights":[5,5,3,5,5,4,4,4,4,4,2],"sparse7_products":77,"sparse7_raw_max":31779369295563194368,"sparse7_reduced_max":72638558389858729984,"square_products":66,"storage_bits":128}"#;
pub const OVERFLOW_CERTIFICATE_SHA256_HEX: &str =
    "b3438f63a800114252cb75a8713a30157d149b8215e177b190a455e47578837f";
pub const OVERFLOW_CERTIFICATE_SHA256: [u8; 32] = [
    0xb3, 0x43, 0x8f, 0x63, 0xa8, 0x00, 0x11, 0x42, 0x52, 0xcb, 0x75, 0xa8, 0x71, 0x3a, 0x30, 0x15,
    0x7d, 0x14, 0x9b, 0x82, 0x15, 0xe1, 0x77, 0xb1, 0x90, 0xa4, 0x55, 0xe4, 0x75, 0x78, 0x83, 0x7f,
];
#[cfg(test)]
pub(crate) const DENSE_RAW_MAX: u128 = 49_939_008_893_027_876_864;
#[cfg(test)]
pub(crate) const DENSE_REDUCED_MAX: u128 = 136_197_296_980_985_118_720;
#[cfg(test)]
pub(crate) const SPARSE7_RAW_MAX: u128 = 31_779_369_295_563_194_368;
#[cfg(test)]
pub(crate) const SPARSE7_REDUCED_MAX: u128 = 72_638_558_389_858_729_984;
pub const REDUCTION_WEIGHTS: [u8; D11_DEGREE] = [5, 5, 3, 5, 5, 4, 4, 4, 4, 4, 2];
