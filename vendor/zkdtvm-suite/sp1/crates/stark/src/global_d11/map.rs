use p3_field::{Field, PrimeField32};

use super::{
    constants::{
        BASE_PRIME, CURVE_B_COEFFICIENTS, D11_DEGREE, GLOBAL_TAG_V1, HALF_BASE_MINUS_ONE,
        PADDING_DUMMY_CANONICAL_Y, PADDING_DUMMY_TWEAK, PADDING_DUMMY_X, TWEAK_COUNT_V1,
    },
    curve::{curve_b, D11AffinePointV1},
    field::{D11Sparse7, D11},
    kernels::reduce_wide,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalPackInputV1 {
    pub message: [u32; 7],
    pub kind: u8,
}

/// Proof-visible equality domain for each PackV1 message word.
///
/// `m1..m4` are not integer limbs: the producer and consumer AIRs compare
/// them as base-field elements. Their host `u32` carrier is therefore reduced
/// modulo the base prime. The remaining words are injectively embedded after
/// the listed integer bounds have been checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPackWordSemanticsV1 {
    Unsigned24,
    BaseField,
    Unsigned8,
}

pub const GLOBAL_PACK_WORD_SEMANTICS_V1: [GlobalPackWordSemanticsV1; 7] = [
    GlobalPackWordSemanticsV1::Unsigned24,
    GlobalPackWordSemanticsV1::BaseField,
    GlobalPackWordSemanticsV1::BaseField,
    GlobalPackWordSemanticsV1::BaseField,
    GlobalPackWordSemanticsV1::BaseField,
    GlobalPackWordSemanticsV1::Unsigned8,
    GlobalPackWordSemanticsV1::Unsigned8,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalPackErrorV1 {
    Message0Exceeds24Bits(u32),
    Message5ExceedsByte(u32),
    Message6ExceedsByte(u32),
    TweakOutOfRange(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalMapWitnessV1 {
    pub tweak: u16,
    pub canonical_y: [u32; D11_DEGREE],
    pub candidate_rounds: u16,
    pub zero_top_residue_skips: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalSignedMapRowV1<F: Field> {
    pub packed_x: D11<F>,
    pub signed_y: D11<F>,
    pub is_receive: bool,
    pub witness: GlobalMapWitnessV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalMapErrorV1 {
    Pack(GlobalPackErrorV1),
    AllTweaksFailed,
}

impl From<GlobalPackErrorV1> for GlobalMapErrorV1 {
    fn from(value: GlobalPackErrorV1) -> Self {
        Self::Pack(value)
    }
}

#[must_use]
pub fn pack_unsigned<F: PrimeField32>(
    input: GlobalPackInputV1,
    tweak: u16,
) -> Result<D11Sparse7<F>, GlobalPackErrorV1> {
    let [m0, m1, m2, m3, m4, m5, m6] = input.message;
    if m0 >= 1 << 24 {
        return Err(GlobalPackErrorV1::Message0Exceeds24Bits(m0));
    }
    if m5 >= 1 << 8 {
        return Err(GlobalPackErrorV1::Message5ExceedsByte(m5));
    }
    if m6 >= 1 << 8 {
        return Err(GlobalPackErrorV1::Message6ExceedsByte(m6));
    }
    if tweak >= TWEAK_COUNT_V1 {
        return Err(GlobalPackErrorV1::TweakOutOfRange(tweak));
    }

    Ok(D11Sparse7::new([
        F::from_canonical_u32(m0),
        F::from_wrapped_u32(m1),
        F::from_wrapped_u32(m2),
        F::from_wrapped_u32(m3),
        F::from_wrapped_u32(m4),
        F::from_canonical_u32(m5 + (m6 << 8) + (u32::from(tweak) << 16)),
        F::from_canonical_u32(u32::from(input.kind) + (u32::from(GLOBAL_TAG_V1) << 8)),
    ]))
}

/// Converts either square root to the unique half-interval representative.
/// A zero top coefficient is deliberately not representable by a real row.
#[must_use]
pub fn canonicalize_y<F: PrimeField32>(root: D11<F>) -> Option<D11<F>> {
    let top = root.coefficients()[10].as_canonical_u32();
    if top == 0 {
        None
    } else if top <= HALF_BASE_MINUS_ONE {
        Some(root)
    } else {
        Some(-root)
    }
}

#[must_use]
pub fn apply_direction<F: PrimeField32>(canonical_y: D11<F>, is_receive: bool) -> D11<F> {
    debug_assert!({
        let top = canonical_y.coefficients()[10].as_canonical_u32();
        (1..=HALF_BASE_MINUS_ONE).contains(&top)
    });
    if is_receive {
        canonical_y
    } else {
        -canonical_y
    }
}

pub fn construct_map_reference<F: PrimeField32>(
    input: GlobalPackInputV1,
    is_receive: bool,
) -> Result<GlobalSignedMapRowV1<F>, GlobalMapErrorV1> {
    let mut zero_top_residue_skips = 0u16;
    for tweak in 0..TWEAK_COUNT_V1 {
        let packed = pack_unsigned::<F>(input, tweak)?;
        let x = packed.to_d11();
        let rhs = x.square() * x - x * F::from_canonical_u32(3) + curve_b::<F>();
        if !rhs.is_quadratic_residue() {
            continue;
        }
        let root = rhs
            .sqrt_from_verified_residue()
            .expect("QR-tested D11 candidate must have a verified square root");
        let Some(canonical_y) = canonicalize_y(root) else {
            zero_top_residue_skips += 1;
            continue;
        };
        let signed_y = apply_direction(canonical_y, is_receive);
        return Ok(GlobalSignedMapRowV1 {
            packed_x: x,
            signed_y,
            is_receive,
            witness: GlobalMapWitnessV1 {
                tweak,
                canonical_y: canonical_y.to_canonical_u32(),
                candidate_rounds: tweak + 1,
                zero_top_residue_skips,
            },
        });
    }
    Err(GlobalMapErrorV1::AllTweaksFailed)
}

/// Production first-success constructor. Pack bounds are validated once,
/// four sparse candidates seed the exact cubic finite-difference recurrence,
/// and every later tweak advances with extension-field additions only.
pub fn construct_map<F: PrimeField32>(
    input: GlobalPackInputV1,
    is_receive: bool,
) -> Result<GlobalSignedMapRowV1<F>, GlobalMapErrorV1> {
    let base = pack_unsigned::<F>(input, 0)?;
    let mut packed_coefficients = *base.coefficients();
    let tweak_step = F::from_canonical_u32(1 << 16);

    let rhs0 = sparse_candidate_rhs(&base);
    let mut seed = packed_coefficients;
    seed[5] += tweak_step;
    let rhs1 = sparse_candidate_rhs(&D11Sparse7::new(seed));
    seed[5] += tweak_step;
    let rhs2 = sparse_candidate_rhs(&D11Sparse7::new(seed));
    seed[5] += tweak_step;
    let rhs3 = sparse_candidate_rhs(&D11Sparse7::new(seed));

    let mut rhs = rhs0;
    let mut first_difference = rhs1 - rhs0;
    let mut second_difference = rhs2 - rhs1.double() + rhs0;
    let third_difference =
        rhs3 - rhs2 * F::from_canonical_u32(3) + rhs1 * F::from_canonical_u32(3) - rhs0;
    let mut zero_top_residue_skips = 0u16;

    for tweak in 0..TWEAK_COUNT_V1 {
        if let Some(residue) = rhs.verified_quadratic_residue() {
            let root =
                residue.sqrt().expect("QR-admitted D11 candidate has a base-field square root");
            if let Some(canonical_y) = canonicalize_y(root) {
                let packed_x = D11Sparse7::new(packed_coefficients).to_d11();
                return Ok(GlobalSignedMapRowV1 {
                    packed_x,
                    signed_y: apply_direction(canonical_y, is_receive),
                    is_receive,
                    witness: GlobalMapWitnessV1 {
                        tweak,
                        canonical_y: canonical_y.to_canonical_u32(),
                        candidate_rounds: tweak + 1,
                        zero_top_residue_skips,
                    },
                });
            }
            zero_top_residue_skips += 1;
        }

        rhs += first_difference;
        first_difference += second_difference;
        second_difference += third_difference;
        packed_coefficients[5] += tweak_step;
    }
    Err(GlobalMapErrorV1::AllTweaksFailed)
}

fn sparse_candidate_rhs<F: PrimeField32>(x: &D11Sparse7<F>) -> D11<F> {
    x.square().mul_sparse_7(x) - x.to_d11() * F::from_canonical_u32(3) + curve_b::<F>()
}

/// The fixed legal padding witness.  Padding applies neither receive nor send
/// direction, so its signed value is the canonical value and `w=y10`.
#[must_use]
pub fn fixed_padding_dummy<F: PrimeField32>() -> GlobalSignedMapRowV1<F> {
    let packed_x = D11::from_canonical_u32(PADDING_DUMMY_X);
    let canonical_y = D11::from_canonical_u32(PADDING_DUMMY_CANONICAL_Y);
    debug_assert!(D11AffinePointV1 { x: packed_x, y: canonical_y }.is_on_curve());
    GlobalSignedMapRowV1 {
        packed_x,
        signed_y: canonical_y,
        is_receive: false,
        witness: GlobalMapWitnessV1 {
            tweak: PADDING_DUMMY_TWEAK,
            canonical_y: PADDING_DUMMY_CANONICAL_Y,
            candidate_rounds: PADDING_DUMMY_TWEAK + 1,
            zero_top_residue_skips: 0,
        },
    }
}

/// Raw degree-20 map residual followed by the protocol closed reduction.  The
/// packed x input is sparse, hence x^3 has degree at most 18.
#[must_use]
pub fn direct_map_residual<F: Field>(x: &D11Sparse7<F>, y: &D11<F>) -> D11<F> {
    let mut y_squared = [F::zero(); 21];
    for i in 0..11 {
        for j in 0..11 {
            y_squared[i + j] += y.coefficients()[i] * y.coefficients()[j];
        }
    }
    let mut x_squared = [F::zero(); 13];
    for i in 0..7 {
        for j in 0..7 {
            x_squared[i + j] += x.coefficients()[i] * x.coefficients()[j];
        }
    }
    let mut x_cubed = [F::zero(); 19];
    for i in 0..13 {
        for j in 0..7 {
            x_cubed[i + j] += x_squared[i] * x.coefficients()[j];
        }
    }
    for (coefficient, cube) in y_squared.iter_mut().zip(x_cubed) {
        *coefficient -= cube;
    }
    for i in 0..7 {
        y_squared[i] += x.coefficients()[i] * F::from_canonical_u32(3);
    }
    for (coefficient, curve_coefficient) in y_squared.iter_mut().zip(CURVE_B_COEFFICIENTS) {
        *coefficient -= F::from_canonical_u32(curve_coefficient);
    }
    reduce_wide(y_squared)
}

#[cfg(test)]
pub(crate) fn first_success_tweak_for_test(
    mut candidate_succeeds: impl FnMut(u16) -> bool,
) -> Result<u16, GlobalMapErrorV1> {
    for tweak in 0..TWEAK_COUNT_V1 {
        if candidate_succeeds(tweak) {
            return Ok(tweak);
        }
    }
    Err(GlobalMapErrorV1::AllTweaksFailed)
}

const _: () = assert!(BASE_PRIME == 2 * HALF_BASE_MINUS_ONE + 1);

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_koala_bear::KoalaBear;

    use super::*;

    #[test]
    fn pack_base_field_words_use_proof_visible_mod_p_equality() {
        let canonical = GlobalPackInputV1 { message: [0, 1, 2, 3, 4, 0, 0], kind: 0 };
        let mut equivalent_transport = canonical;
        equivalent_transport.message[1] += BASE_PRIME;

        let canonical = pack_unsigned::<KoalaBear>(canonical, 0).unwrap();
        let equivalent = pack_unsigned::<KoalaBear>(equivalent_transport, 0).unwrap();
        assert_eq!(canonical, equivalent);
        assert_eq!(canonical.coefficients()[1], KoalaBear::from_canonical_u32(1));
    }

    #[test]
    fn pack_bounded_words_remain_integer_injective() {
        let low = pack_unsigned::<KoalaBear>(
            GlobalPackInputV1 { message: [1, 0, 0, 0, 0, 2, 3], kind: 4 },
            5,
        )
        .unwrap();
        let high = pack_unsigned::<KoalaBear>(
            GlobalPackInputV1 { message: [2, 0, 0, 0, 0, 2, 3], kind: 4 },
            5,
        )
        .unwrap();
        assert_ne!(low, high);
    }
}
