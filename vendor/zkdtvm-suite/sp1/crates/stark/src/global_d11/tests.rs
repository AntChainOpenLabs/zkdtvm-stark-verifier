use p3_field::{AbstractField, PrimeField32};
use p3_koala_bear::KoalaBear;

use super::{
    apply_direction,
    constants::{
        BASE_PRIME, CURVE_GENERATOR_Y, D11_DEGREE, DENSE_RAW_MAX, DENSE_REDUCED_MAX,
        HALF_BASE_MINUS_ONE, OVERFLOW_CERTIFICATE_SHA256, PADDING_DUMMY_CANONICAL_Y,
        PARAMETER_MANIFEST_SHA256, REDUCTION_WEIGHTS, SPARSE7_RAW_MAX, SPARSE7_REDUCED_MAX,
    },
    construct_map, construct_map_reference, direct_map_residual, fixed_padding_dummy,
    kernels::reduce_wide,
    map::first_success_tweak_for_test,
    overflow_certificate_digest, pack_unsigned, parameter_manifest_digest, D11AffinePointV1,
    D11ProjectivePointV1, D11Sparse7, GlobalMapErrorV1, GlobalPackErrorV1, GlobalPackInputV1,
    ProjectivePointError, D11, DENSE_SPARSE_7_COST, SCHOOLBOOK_COST, SQUARE_COST,
};

type K = D11<KoalaBear>;

fn d11(values: [u32; D11_DEGREE]) -> K {
    K::from_canonical_u32(values)
}

#[test]
fn parameter_and_overflow_manifests_replay() {
    assert_eq!(parameter_manifest_digest(), PARAMETER_MANIFEST_SHA256);
    assert_eq!(overflow_certificate_digest(), OVERFLOW_CERTIFICATE_SHA256);
}

#[test]
fn modulus_and_named_kernels_are_frozen() {
    let z = K::z();
    let mut z_to_11 = K::one();
    for _ in 0..11 {
        z_to_11 *= z;
    }
    assert_eq!(z_to_11, z.square() * z + K::from_base(KoalaBear::two()));

    let a = d11([3, 6, 17, 91, 37, 35, 33, 1_234_567, 7_654_321, 42, 99]);
    let sparse = D11Sparse7::from_canonical_u32([5, 7, 11, 13, 17, 19, 23]);
    assert_eq!(
        a.mul_sparse_7(&sparse).to_canonical_u32(),
        [
            240992813, 307165299, 347658442, 472600806, 153589456, 173835510, 176054336, 6178787,
            46919418, 67164514, 100249768
        ]
    );
    assert_eq!(
        a.mul_by_z_plus_36(),
        a * (K::z() + K::from_base(KoalaBear::from_canonical_u32(36)))
    );
}

#[test]
fn external_arithmetic_kats_match_limb_for_limb() {
    let a = d11([3, 6, 17, 91, 37, 35, 33, 1_234_567, 7_654_321, 42, 99]);
    let b = d11([13, 29, 41, 67, 83, 97, 11, 314_159, 271_828, 123, 456]);
    assert_eq!(
        (a * b).to_canonical_u32(),
        [
            1303426307, 1552372501, 1551974052, 961644597, 1126724158, 1153992993, 1118194425,
            834726477, 1392440231, 1826627688, 1816343928
        ]
    );
    let expected_square = [
        838208820, 1305722724, 1234610174, 1765298631, 1009216959, 2015094770, 849891530,
        1086000939, 759659225, 1021262810, 2000500195,
    ];
    assert_eq!(a.square().to_canonical_u32(), expected_square);
    assert_eq!(a.square(), a * a);
    assert_eq!(
        a.frobenius().to_canonical_u32(),
        [
            1082533312, 1967489175, 258145726, 1484574460, 541187408, 824507629, 277362778,
            2086197311, 383539150, 695525731, 1371031896
        ]
    );
    assert_eq!(
        a.inverse().to_canonical_u32(),
        [
            2049846099, 995213654, 1026391463, 779017490, 439990730, 687076723, 809691436,
            1928542624, 1936263482, 244229099, 1498432309
        ]
    );
    assert_eq!(a * a.inverse(), K::one());
    assert_eq!(a.norm().as_canonical_u32(), 1_428_609_842);

    let square = d11(expected_square);
    let root = square.sqrt().expect("external square KAT must have a root");
    assert_eq!(root.square(), square);
    assert!(root == a || root == -a);
}

#[test]
fn frobenius_norm_qr_and_sqrt_identities_hold() {
    for seed in 0..16u32 {
        let value =
            d11(core::array::from_fn(|limb| (31 + 109 * seed + 23 * limb as u32) % BASE_PRIME));
        assert_eq!(value.frobenius_pow(11), value);
        assert_eq!(value.frobenius().square(), value.square().frobenius());
        let norm = value.norm_element().to_canonical_u32();
        assert!(norm[1..].iter().all(|coefficient| *coefficient == 0));

        let square = value.square();
        assert!(square.is_quadratic_residue());
        assert_eq!(square.sqrt().expect("a square must have a root").square(), square);
    }

    // A prime-field nonresidue stays a nonresidue in an odd-degree extension.
    let nonresidue = K::from_base(KoalaBear::from_canonical_u32(3));
    assert!(!nonresidue.is_quadratic_residue());
    assert_eq!(nonresidue.sqrt(), None);
    assert_eq!(K::zero().sqrt(), Some(K::zero()));
}

#[test]
fn closed_reduction_order_handles_created_high_terms() {
    let mut wide = [KoalaBear::zero(); 21];
    wide[20] = KoalaBear::one();
    let reduced = reduce_wide(wide);
    let mut via_field = K::one();
    let z = K::z();
    for _ in 0..20 {
        via_field *= z;
    }
    assert_eq!(reduced, via_field);
}

#[test]
fn overflow_certificate_replays_with_u128_and_rejects_u64() {
    let limb_max = u128::from(BASE_PRIME - 1);
    let product_max = limb_max * limb_max;
    let dense_counts: [u128; 21] =
        core::array::from_fn(|degree| degree.saturating_add(1).min(21 - degree).min(11) as u128);
    let sparse_counts: [u128; 21] = core::array::from_fn(|degree| {
        (0..11).filter(|left| degree >= *left && degree - *left < 7).count() as u128
    });

    fn reduce_bounds(mut bounds: [u128; 21]) -> [u128; 11] {
        for degree in (11..21).rev() {
            let value = bounds[degree];
            bounds[degree] = 0;
            bounds[degree - 8] += value;
            bounds[degree - 11] += 2 * value;
        }
        core::array::from_fn(|i| bounds[i])
    }

    let dense_raw = dense_counts.map(|count| count * product_max);
    let sparse_raw = sparse_counts.map(|count| count * product_max);
    let dense_reduced = reduce_bounds(dense_raw);
    let sparse_reduced = reduce_bounds(sparse_raw);
    assert_eq!(*dense_raw.iter().max().unwrap(), DENSE_RAW_MAX);
    assert_eq!(*sparse_raw.iter().max().unwrap(), SPARSE7_RAW_MAX);
    assert_eq!(*dense_reduced.iter().max().unwrap(), DENSE_REDUCED_MAX);
    assert_eq!(*sparse_reduced.iter().max().unwrap(), SPARSE7_REDUCED_MAX);
    assert_eq!(
        reduce_bounds([product_max; 21]).map(|bound| (bound / product_max) as u8),
        REDUCTION_WEIGHTS
    );
    assert!(DENSE_REDUCED_MAX > u128::from(u64::MAX));
    assert!(SPARSE7_REDUCED_MAX > u128::from(u64::MAX));
    assert!(DENSE_REDUCED_MAX < u128::MAX);

    assert_eq!(SCHOOLBOOK_COST.base_products, 121);
    assert_eq!(SQUARE_COST.base_products, 66);
    assert_eq!(DENSE_SPARSE_7_COST.base_products, 77);
    assert_eq!(SCHOOLBOOK_COST.dynamic_allocations, 0);
    assert_eq!(SQUARE_COST.dynamic_allocations, 0);
    assert_eq!(DENSE_SPARSE_7_COST.dynamic_allocations, 0);
}

fn generator() -> D11AffinePointV1<KoalaBear> {
    D11AffinePointV1 { x: K::zero(), y: d11(CURVE_GENERATOR_Y) }
}

fn projective(values: [[u32; 11]; 3]) -> D11ProjectivePointV1<KoalaBear> {
    D11ProjectivePointV1 { x: d11(values[0]), y: d11(values[1]), z: d11(values[2]) }
}

#[test]
fn pack_boundaries_and_direction_are_exact() {
    let input =
        GlobalPackInputV1 { message: [(1 << 24) - 1, u32::MAX, 2, 3, 4, 255, 255], kind: 255 };
    assert_eq!(
        pack_unsigned::<KoalaBear>(input, 511).unwrap().to_d11().to_canonical_u32(),
        [16_777_215, 33_554_429, 2, 3, 4, 33_554_431, 511, 0, 0, 0, 0]
    );
    assert_eq!(
        pack_unsigned::<KoalaBear>(
            GlobalPackInputV1 { message: [1 << 24, 0, 0, 0, 0, 0, 0], kind: 0 },
            0
        ),
        Err(GlobalPackErrorV1::Message0Exceeds24Bits(1 << 24))
    );
    assert!(matches!(
        pack_unsigned::<KoalaBear>(
            GlobalPackInputV1 { message: [0, 0, 0, 0, 0, 256, 0], kind: 0 },
            0
        ),
        Err(GlobalPackErrorV1::Message5ExceedsByte(256))
    ));
    assert!(matches!(
        pack_unsigned::<KoalaBear>(
            GlobalPackInputV1 { message: [0, 0, 0, 0, 0, 0, 256], kind: 0 },
            0
        ),
        Err(GlobalPackErrorV1::Message6ExceedsByte(256))
    ));
    assert!(matches!(
        pack_unsigned::<KoalaBear>(input, 512),
        Err(GlobalPackErrorV1::TweakOutOfRange(512))
    ));

    let canonical = d11(PADDING_DUMMY_CANONICAL_Y);
    let receive = apply_direction(canonical, true);
    let send = apply_direction(canonical, false);
    assert_eq!(receive, canonical);
    assert_eq!(send, -canonical);
    assert_eq!(receive.coefficients()[10].as_canonical_u32(), 587_574_789);
    assert_eq!(send.coefficients()[10].as_canonical_u32(), 1_543_131_644);
    let receive_w = receive.coefficients()[10].as_canonical_u32() - 1;
    let send_w = HALF_BASE_MINUS_ONE - canonical.coefficients()[10].as_canonical_u32();
    assert_eq!(receive_w + 1, receive.coefficients()[10].as_canonical_u32());
    assert_eq!(send_w + HALF_BASE_MINUS_ONE + 1, send.coefficients()[10].as_canonical_u32());
}

#[test]
fn constructor_and_legal_dummy_match_external_witness() {
    let input = GlobalPackInputV1 { message: [0; 7], kind: 0 };
    let receive = construct_map_reference::<KoalaBear>(input, true).unwrap();
    assert_eq!(construct_map::<KoalaBear>(input, true).unwrap(), receive);
    assert_eq!(receive.witness.tweak, 1);
    assert_eq!(receive.witness.candidate_rounds, 2);
    assert_eq!(receive.witness.canonical_y, PADDING_DUMMY_CANONICAL_Y);
    assert_eq!(receive.signed_y.to_canonical_u32(), PADDING_DUMMY_CANONICAL_Y);

    let send = construct_map_reference::<KoalaBear>(input, false).unwrap();
    assert_eq!(send.packed_x, receive.packed_x);
    assert_eq!(send.signed_y, -receive.signed_y);

    let dummy = fixed_padding_dummy::<KoalaBear>();
    assert_eq!(dummy.packed_x, receive.packed_x);
    assert_eq!(dummy.signed_y, receive.signed_y);
    let packed_sparse = pack_unsigned::<KoalaBear>(input, dummy.witness.tweak).unwrap();
    assert_eq!(direct_map_residual(&packed_sparse, &dummy.signed_y), K::zero());
    assert!(D11AffinePointV1 { x: dummy.packed_x, y: dummy.signed_y }.is_on_curve());
}

#[test]
fn production_constructor_matches_reference_across_pack_domains() {
    let cases = [
        GlobalPackInputV1 { message: [0; 7], kind: 0 },
        GlobalPackInputV1 { message: [1, 2, 3, 4, 5, 6, 7], kind: 1 },
        GlobalPackInputV1 {
            message: [(1 << 24) - 1, u32::MAX, BASE_PRIME + 17, 19, 23, 255, 255],
            kind: u8::MAX,
        },
    ];
    for input in cases {
        for is_receive in [false, true] {
            assert_eq!(
                construct_map::<KoalaBear>(input, is_receive),
                construct_map_reference::<KoalaBear>(input, is_receive)
            );
        }
    }
}

#[test]
fn first_success_order_and_failure_are_explicit() {
    assert_eq!(first_success_tweak_for_test(|tweak| tweak == 0), Ok(0));
    assert_eq!(first_success_tweak_for_test(|tweak| tweak == 511), Ok(511));
    assert_eq!(first_success_tweak_for_test(|_| false), Err(GlobalMapErrorV1::AllTweaksFailed));
}

#[test]
fn full_projective_rcb_external_goldens_and_identity_rules() {
    let affine_g = generator();
    assert!(affine_g.is_on_curve());
    let g = affine_g.to_projective();
    let identity = D11ProjectivePointV1::identity();
    let negative_g = g.negated();

    let scaled_g = projective([[0; 11], [36, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], CURVE_GENERATOR_Y]);
    assert_eq!(identity.add_complete(&g), scaled_g);
    assert_eq!(g.add_complete(&identity), scaled_g);
    assert!(scaled_g.equivalent(&g));
    assert_eq!(identity.add_complete(&identity), identity);

    let doubled = projective([
        [
            1602959286, 1922662514, 881240310, 927356807, 43424301, 2076534738, 558623305,
            108624560, 961277820, 1575898050, 612999672,
        ],
        [2130696092, 2130705857, 2130706425, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            1333995455, 224665556, 1223142398, 2007341495, 159966760, 1993494480, 1338051476,
            1512779666, 277032512, 1493597146, 565097198,
        ],
    ]);
    assert_eq!(g.add_complete(&g), doubled);
    assert!(doubled.is_on_curve());

    let cancelled = g.add_complete(&negative_g);
    assert!(cancelled.is_identity());
    assert_eq!(
        cancelled.y.to_canonical_u32(),
        [2130696092, 2130705857, 2130706425, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_ne!(cancelled, identity);
    assert!(cancelled.equivalent(&identity));

    let scale = d11([7, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let rescaled = g.rescaled(scale);
    assert!(rescaled.is_on_curve());
    assert!(rescaled.equivalent(&g));
    assert!(rescaled.add_complete(&negative_g).is_identity());

    let zero_triple = projective([[0; 11], [0; 11], [0; 11]]);
    assert_eq!(zero_triple.validate(), Err(ProjectivePointError::AllZeroTriple));
    let invalid_infinity =
        projective([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0; 11]]);
    assert_eq!(invalid_infinity.validate(), Err(ProjectivePointError::InvalidInfinityEncoding));
}

#[test]
fn mixed_rcb_matches_full_formula_and_materializes_five_products() {
    let affine_g = generator();
    let g = affine_g.to_projective();
    let identity = D11ProjectivePointV1::identity();
    let (from_identity, products) = identity.add_mixed_complete(&affine_g);
    assert_eq!(from_identity, identity.add_complete(&g));
    assert_eq!(products.len(), 5);

    let (doubled_mixed, _) = g.add_mixed_complete(&affine_g);
    assert_eq!(doubled_mixed, g.add_complete(&g));
    let (cancelled, _) = g.negated().add_mixed_complete(&affine_g);
    assert!(cancelled.is_identity());
    assert!(cancelled.equivalent(&g.negated().add_complete(&g)));
}
