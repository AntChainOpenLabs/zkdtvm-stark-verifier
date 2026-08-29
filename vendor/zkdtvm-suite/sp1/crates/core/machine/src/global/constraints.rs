use p3_field::{AbstractField, Field};

use super::columns::{D11PointCols, D11ProductCols, GlobalCols};

pub(crate) const D11: usize = 11;
const WIDE: usize = 21;

pub(crate) type D11Expr<T> = [T; D11];

fn zero<T: AbstractField>() -> D11Expr<T> {
    core::array::from_fn(|_| T::zero())
}

#[derive(Clone)]
pub(crate) struct D11ProductWithQuotient<T> {
    pub reduced: D11Expr<T>,
    pub quotient: [T; 10],
}

fn divide_wide<T: AbstractField + Clone>(mut wide: [T; WIDE]) -> D11ProductWithQuotient<T> {
    let mut quotient = core::array::from_fn(|_| T::zero());
    for degree in (D11..WIDE).rev() {
        let coefficient = wide[degree].clone();
        quotient[degree - D11] = coefficient.clone();
        wide[degree - D11] = wide[degree - D11].clone() + coefficient.clone() * T::two();
        wide[degree - 8] = wide[degree - 8].clone() + coefficient;
    }
    D11ProductWithQuotient {
        reduced: core::array::from_fn(|i| wide[i].clone()),
        quotient,
    }
}

fn reduce_wide<T: AbstractField + Clone>(wide: [T; WIDE]) -> D11Expr<T> {
    divide_wide(wide).reduced
}

pub(crate) fn mul<T: AbstractField + Clone>(lhs: &D11Expr<T>, rhs: &D11Expr<T>) -> D11Expr<T> {
    mul_with_quotient(lhs, rhs).reduced
}

pub(crate) fn mul_with_quotient<T: AbstractField + Clone>(
    lhs: &D11Expr<T>,
    rhs: &D11Expr<T>,
) -> D11ProductWithQuotient<T> {
    let mut wide: [T; WIDE] = core::array::from_fn(|_| T::zero());
    for i in 0..D11 {
        for j in 0..D11 {
            wide[i + j] = wide[i + j].clone() + lhs[i].clone() * rhs[j].clone();
        }
    }
    divide_wide(wide)
}

pub(crate) fn mul_sparse_7_with_quotient<T: AbstractField + Clone>(
    dense: &D11Expr<T>,
    sparse: &D11Expr<T>,
) -> D11ProductWithQuotient<T> {
    let mut wide: [T; WIDE] = core::array::from_fn(|_| T::zero());
    for i in 0..D11 {
        for j in 0..7 {
            wide[i + j] = wide[i + j].clone() + dense[i].clone() * sparse[j].clone();
        }
    }
    divide_wide(wide)
}

fn mul_sparse_7<T: AbstractField + Clone>(dense: &D11Expr<T>, sparse: &D11Expr<T>) -> D11Expr<T> {
    mul_sparse_7_with_quotient(dense, sparse).reduced
}

fn add<T: AbstractField + Clone>(lhs: &D11Expr<T>, rhs: &D11Expr<T>) -> D11Expr<T> {
    core::array::from_fn(|i| lhs[i].clone() + rhs[i].clone())
}

fn sub<T: AbstractField + Clone>(lhs: &D11Expr<T>, rhs: &D11Expr<T>) -> D11Expr<T> {
    core::array::from_fn(|i| lhs[i].clone() - rhs[i].clone())
}

fn scale<T: AbstractField + Clone>(value: &D11Expr<T>, constant: u32) -> D11Expr<T> {
    let constant = T::from_canonical_u32(constant);
    core::array::from_fn(|i| value[i].clone() * constant.clone())
}

fn mul_by_b<T: AbstractField + Clone>(value: &D11Expr<T>) -> D11Expr<T> {
    let mut by_z: D11Expr<T> = core::array::from_fn(|_| T::zero());
    by_z[0] = value[10].clone() * T::two();
    for i in 0..10 {
        by_z[i + 1] = value[i].clone();
    }
    by_z[3] = by_z[3].clone() + value[10].clone();
    add(&by_z, &scale(value, 36))
}

fn point<T: Clone>(cols: &D11PointCols<T>) -> [D11Expr<T>; 3] {
    [cols.x.clone(), cols.y.clone(), cols.z.clone()]
}

/// Reconstructs all proof-visible linear header values.
pub(crate) struct HeaderExpressions<T> {
    pub message: [T; 7],
    pub kind: T,
    pub tweak: T,
    pub signed_y: D11Expr<T>,
    pub is_send: T,
}

pub(crate) fn header<T: AbstractField + Clone>(cols: &GlobalCols<T>) -> HeaderExpressions<T> {
    let two16 = T::from_canonical_u32(1 << 16);
    let two8 = T::from_canonical_u32(1 << 8);
    let half_plus_one = T::from_canonical_u32(1_065_353_217);
    let m0 = cols.m0_lo16.clone() + cols.m0_hi8.clone() * two16.clone();
    let [m1, m2, m3, m4, m5, m6] = cols.message_rest.clone();
    let kind = cols.x6.clone() - two8.clone();
    let two16_inverse = T::from_f(T::F::from_canonical_u32(1 << 16).inverse());
    let tweak = (cols.x5.clone() - m5.clone() - m6.clone() * two8) * two16_inverse;
    let is_send = cols.is_real.clone() - cols.is_receive.clone();
    let w = cols.w_lo16.clone() + cols.w_hi.clone() * two16;
    let y10 = w + cols.is_receive.clone() + is_send.clone() * half_plus_one;
    let signed_y =
        core::array::from_fn(|i| if i < 10 { cols.y_lower[i].clone() } else { y10.clone() });
    HeaderExpressions { message: [m0, m1, m2, m3, m4, m5, m6], kind, tweak, signed_y, is_send }
}

pub(crate) fn packed_x<T: AbstractField + Clone>(
    cols: &GlobalCols<T>,
    header: &HeaderExpressions<T>,
) -> D11Expr<T> {
    let mut x = zero();
    x[0] = header.message[0].clone();
    x[1] = header.message[1].clone();
    x[2] = header.message[2].clone();
    x[3] = header.message[3].clone();
    x[4] = header.message[4].clone();
    x[5] = cols.x5.clone();
    x[6] = cols.x6.clone();
    x
}

pub(crate) fn map_quotient<T: AbstractField + Clone>(
    x: &D11Expr<T>,
    y: &D11Expr<T>,
) -> [T; 10] {
    let x2 = mul_sparse_7_with_quotient(x, x);
    let x3 = mul_sparse_7_with_quotient(&x2.reduced, x);
    let y2 = mul_with_quotient(y, y);
    let mut x_q2: [T; 10] = core::array::from_fn(|_| T::zero());
    for i in 0..7 {
        for j in 0..2 {
            x_q2[i + j] = x_q2[i + j].clone() +
                x[i].clone() * x2.quotient[j].clone();
        }
    }
    core::array::from_fn(|i| {
        y2.quotient[i].clone() - x3.quotient[i].clone() - x_q2[i].clone()
    })
}

fn direct_map_residual<T: AbstractField + Clone>(x: &D11Expr<T>, y: &D11Expr<T>) -> D11Expr<T> {
    let mut y2: [T; WIDE] = core::array::from_fn(|_| T::zero());
    for i in 0..D11 {
        y2[2 * i] = y2[2 * i].clone() + y[i].clone() * y[i].clone();
        for j in i + 1..D11 {
            y2[i + j] = y2[i + j].clone() + (y[i].clone() * y[j].clone()).double();
        }
    }

    let mut x2: [T; 13] = core::array::from_fn(|_| T::zero());
    for i in 0..7 {
        x2[2 * i] = x2[2 * i].clone() + x[i].clone() * x[i].clone();
        for j in i + 1..7 {
            x2[i + j] = x2[i + j].clone() + (x[i].clone() * x[j].clone()).double();
        }
    }
    let mut x3: [T; 19] = core::array::from_fn(|_| T::zero());
    for i in 0..13 {
        for j in 0..7 {
            x3[i + j] = x3[i + j].clone() + x2[i].clone() * x[j].clone();
        }
    }
    for i in 0..19 {
        y2[i] = y2[i].clone() - x3[i].clone();
    }
    for i in 0..7 {
        y2[i] = y2[i].clone() + x[i].clone() * T::from_canonical_u32(3);
    }
    y2[0] = y2[0].clone() - T::from_canonical_u32(36);
    y2[1] = y2[1].clone() - T::one();
    reduce_wide(y2)
}

pub(crate) struct MixedOutputWithQuotients<T> {
    pub output: [D11Expr<T>; 3],
    pub product_residuals: [D11Expr<T>; 5],
    pub product_quotients: [[T; 10]; 5],
    pub output_quotients: [[T; 10]; 3],
}

pub(crate) fn mixed_output_with_quotients<T: AbstractField + Clone>(
    input: &[D11Expr<T>; 3],
    affine: &[D11Expr<T>; 2],
    products: &D11ProductCols<T>,
) -> MixedOutputWithQuotients<T> {
    let [input_x, input_y, input_z] = input;
    let [x, y] = affine;
    let u0 = &products.u0;
    let u1 = &products.u1;
    let u3 = &products.u3;
    let u4 = &products.u4;
    let u5 = &products.u5;

    let raw_u0 = mul_sparse_7_with_quotient(input_x, x);
    let raw_u1 = mul_with_quotient(input_y, y);
    let raw_u3 = mul_with_quotient(&add(input_x, input_y), &add(x, y));
    let raw_u4 = mul_sparse_7_with_quotient(input_z, x);
    let raw_u5 = mul_with_quotient(y, input_z);
    let product_residuals = [
        sub(u0, &raw_u0.reduced),
        sub(u1, &raw_u1.reduced),
        sub(u3, &raw_u3.reduced),
        sub(u4, &raw_u4.reduced),
        sub(u5, &raw_u5.reduced),
    ];
    let product_quotients = [
        raw_u0.quotient,
        raw_u1.quotient,
        raw_u3.quotient,
        raw_u4.quotient,
        raw_u5.quotient,
    ];

    let sxy = sub(&sub(u3, u0), u1);
    let sxz = add(input_x, u4);
    let syz = add(input_y, u5);
    let delta = scale(&sub(&sxz, &mul_by_b(input_z)), 3);
    let l0 = add(u1, &delta);
    let l3 = sub(u1, &delta);
    let l2 = scale(&sub(u0, input_z), 3);
    let l1 = scale(&sub(&sub(&mul_by_b(&sxz), u0), &scale(input_z, 3)), 3);

    let q0 = mul_with_quotient(&sxy, &l0);
    let q1 = mul_with_quotient(&syz, &l1);
    let q2 = mul_with_quotient(&l2, &l1);
    let q3 = mul_with_quotient(&l3, &l0);
    let q4 = mul_with_quotient(&syz, &l3);
    let q5 = mul_with_quotient(&sxy, &l2);
    let output = [
        sub(&q0.reduced, &q1.reduced),
        add(&q2.reduced, &q3.reduced),
        add(&q4.reduced, &q5.reduced),
    ];
    let output_quotients = [
        core::array::from_fn(|i| q0.quotient[i].clone() - q1.quotient[i].clone()),
        core::array::from_fn(|i| q2.quotient[i].clone() + q3.quotient[i].clone()),
        core::array::from_fn(|i| q4.quotient[i].clone() + q5.quotient[i].clone()),
    ];
    MixedOutputWithQuotients { output, product_residuals, product_quotients, output_quotients }
}

/// Returns the exact coefficient KAT vector for the committed quotient protocol.
/// Production evaluation uses the nine beta-evaluated identities instead.
pub(crate) fn for_each_constraint_residual<T: AbstractField + Clone>(
    cols: &GlobalCols<T>,
    mut emit: impl FnMut(T),
) {
    let one = T::one();
    let h = header(cols);
    emit(cols.is_real.clone() * (one.clone() - cols.is_real.clone()));
    emit(cols.is_receive.clone() * (one.clone() - cols.is_receive.clone()));
    emit(cols.is_receive.clone() * (one.clone() - cols.is_real.clone()));

    let x = packed_x(cols, &h);
    for residual in direct_map_residual(&x, &h.signed_y) {
        emit(residual);
    }
    let expected_map_q = map_quotient(&x, &h.signed_y);
    for (actual, expected) in cols.quotient.map.iter().zip(expected_map_q) {
        emit(actual.clone() - expected);
    }

    let input = point(&cols.input);
    let mixed = mixed_output_with_quotients(&input, &[x, h.signed_y], &cols.products);
    for product in mixed.product_residuals {
        for residual in product {
            emit(residual);
        }
    }
    for (group, expected) in mixed.product_quotients.into_iter().enumerate() {
        for (limb, expected) in expected.into_iter().enumerate() {
            let actual = match group {
                0 if limb < 6 => cols.quotient.u0[limb].clone(),
                1 => cols.quotient.u1[limb].clone(),
                2 => cols.quotient.u3[limb].clone(),
                3 if limb < 6 => cols.quotient.u4[limb].clone(),
                4 => cols.quotient.u5[limb].clone(),
                _ => T::zero(),
            };
            emit(actual - expected);
        }
    }

    let cumulative = point(&cols.cumulative);
    for coordinate in 0..3 {
        for limb in 0..D11 {
            emit(
                cols.is_real.clone() * mixed.output[coordinate][limb].clone() +
                    (one.clone() - cols.is_real.clone()) * input[coordinate][limb].clone() -
                    cumulative[coordinate][limb].clone(),
            );
        }
        let actual_q = match coordinate {
            0 => &cols.quotient.output_x,
            1 => &cols.quotient.output_y,
            _ => &cols.quotient.output_z,
        };
        for (actual, expected) in actual_q.iter().zip(&mixed.output_quotients[coordinate]) {
            emit(actual.clone() - cols.is_real.clone() * expected.clone());
        }
    }
}

/// Diagnostic/test collection wrapper. Production AIR evaluation uses the
/// callback form above and allocates no per-row residual vector.
pub(crate) fn constraint_residuals<T: AbstractField + Clone>(cols: &GlobalCols<T>) -> Vec<T> {
    let mut residuals = Vec::with_capacity(192);
    for_each_constraint_residual(cols, |residual| residuals.push(residual));
    residuals
}

const _: () = assert!(3 + (11 + 10) + 5 * (11 + 10) + 3 * (11 + 10) == 192);

#[cfg(feature = "test-utils")]
pub(crate) mod p7_kats {
    use p3_field::{AbstractField, Field};
    use p3_koala_bear::KoalaBear;

    use super::{
        add, direct_map_residual, map_quotient, mul_with_quotient, mixed_output_with_quotients,
        mul_sparse_7_with_quotient, D11Expr, D11ProductCols, D11ProductWithQuotient, D11, WIDE,
    };

    fn polynomial(seed: u32) -> D11Expr<KoalaBear> {
        core::array::from_fn(|i| KoalaBear::from_canonical_u32(seed + 3 * i as u32))
    }

    fn raw_product(lhs: &D11Expr<KoalaBear>, rhs: &D11Expr<KoalaBear>) -> [KoalaBear; WIDE] {
        let mut raw = [KoalaBear::zero(); WIDE];
        for i in 0..D11 {
            for j in 0..D11 {
                raw[i + j] += lhs[i] * rhs[j];
            }
        }
        raw
    }

    fn raw_add(
        left: [KoalaBear; WIDE],
        right: [KoalaBear; WIDE],
    ) -> [KoalaBear; WIDE] {
        core::array::from_fn(|i| left[i] + right[i])
    }

    fn raw_sub(
        left: [KoalaBear; WIDE],
        right: [KoalaBear; WIDE],
    ) -> [KoalaBear; WIDE] {
        core::array::from_fn(|i| left[i] - right[i])
    }

    fn reconstruct(product: &D11ProductWithQuotient<KoalaBear>) -> [KoalaBear; WIDE] {
        let mut raw = [KoalaBear::zero(); WIDE];
        raw[..D11].copy_from_slice(&product.reduced);
        for (degree, quotient) in product.quotient.iter().copied().enumerate() {
            raw[degree] -= quotient.double();
            raw[degree + 3] -= quotient;
            raw[degree + D11] += quotient;
        }
        raw
    }

    pub(crate) fn quotient_reconstructs_dense_and_sparse_products() {
        let lhs = polynomial(2);
        let mut sparse = polynomial(7);
        sparse[7..].fill(KoalaBear::zero());
        let dense = polynomial(13);

        let sparse_product = mul_sparse_7_with_quotient(&lhs, &sparse);
        assert_eq!(reconstruct(&sparse_product), raw_product(&lhs, &sparse));
        assert!(sparse_product.quotient[6..].iter().all(Field::is_zero));

        let dense_product = mul_with_quotient(&lhs, &dense);
        assert_eq!(reconstruct(&dense_product), raw_product(&lhs, &dense));
    }

    pub(crate) fn map_quotient_matches_high_square_and_x_q2_formula() {
        let mut x = polynomial(5);
        x[7..].fill(KoalaBear::zero());
        let y = polynomial(17);
        let x2 = mul_sparse_7_with_quotient(&x, &x);
        let x3 = mul_sparse_7_with_quotient(&x2.reduced, &x);

        let mut high = [KoalaBear::zero(); 10];
        for i in 0..D11 {
            for j in i..D11 {
                if i + j >= D11 {
                    let product = y[i] * y[j];
                    high[i + j - D11] += if i == j { product } else { product.double() };
                }
            }
        }
        let qy = [
            high[0] + high[8],
            high[1] + high[9],
            high[2],
            high[3],
            high[4],
            high[5],
            high[6],
            high[7],
            high[8],
            high[9],
        ];
        let mut x_q2 = [KoalaBear::zero(); 10];
        for i in 0..7 {
            for j in 0..2 {
                x_q2[i + j] += x[i] * x2.quotient[j];
            }
        }
        let expected = core::array::from_fn(|i| qy[i] - x3.quotient[i] - x_q2[i]);
        let quotient = map_quotient(&x, &y);
        assert_eq!(quotient, expected);

        let x2_raw = raw_product(&x, &x);
        let mut x3_raw = [KoalaBear::zero(); WIDE];
        for i in 0..13 {
            for j in 0..7 {
                x3_raw[i + j] += x2_raw[i] * x[j];
            }
        }
        let mut map_raw = raw_sub(raw_product(&y, &y), x3_raw);
        for i in 0..7 {
            map_raw[i] += x[i] * KoalaBear::from_canonical_u32(3);
        }
        map_raw[0] -= KoalaBear::from_canonical_u32(36);
        map_raw[1] -= KoalaBear::one();
        assert_eq!(
            reconstruct(&D11ProductWithQuotient {
                reduced: direct_map_residual(&x, &y),
                quotient,
            }),
            map_raw
        );
    }

    pub(crate) fn selected_output_quotients_use_wave2_signed_pairs() {
        let input = [polynomial(3), polynomial(11), polynomial(19)];
        let mut x = polynomial(23);
        x[7..].fill(KoalaBear::zero());
        let y = polynomial(29);
        let raw_u0 = mul_sparse_7_with_quotient(&input[0], &x);
        let raw_u1 = mul_with_quotient(&input[1], &y);
        let raw_u3 = mul_with_quotient(&add(&input[0], &input[1]), &add(&x, &y));
        let raw_u4 = mul_sparse_7_with_quotient(&input[2], &x);
        let raw_u5 = mul_with_quotient(&y, &input[2]);
        let products = D11ProductCols {
            u0: raw_u0.reduced,
            u1: raw_u1.reduced,
            u3: raw_u3.reduced,
            u4: raw_u4.reduced,
            u5: raw_u5.reduced,
        };
        let mixed = mixed_output_with_quotients(&input, &[x, y], &products);

        let sxy = super::sub(&super::sub(&products.u3, &products.u0), &products.u1);
        let sxz = add(&input[0], &products.u4);
        let syz = add(&input[1], &products.u5);
        let delta = super::scale(&super::sub(&sxz, &super::mul_by_b(&input[2])), 3);
        let l0 = add(&products.u1, &delta);
        let l3 = super::sub(&products.u1, &delta);
        let l2 = super::scale(&super::sub(&products.u0, &input[2]), 3);
        let l1 = super::scale(
            &super::sub(
                &super::sub(&super::mul_by_b(&sxz), &products.u0),
                &super::scale(&input[2], 3),
            ),
            3,
        );
        let wave2 = [
            mul_with_quotient(&sxy, &l0),
            mul_with_quotient(&syz, &l1),
            mul_with_quotient(&l2, &l1),
            mul_with_quotient(&l3, &l0),
            mul_with_quotient(&syz, &l3),
            mul_with_quotient(&sxy, &l2),
        ];
        assert_eq!(
            mixed.output_quotients[0],
            core::array::from_fn(|i| wave2[0].quotient[i] - wave2[1].quotient[i])
        );
        assert_eq!(
            mixed.output_quotients[1],
            core::array::from_fn(|i| wave2[2].quotient[i] + wave2[3].quotient[i])
        );
        assert_eq!(
            mixed.output_quotients[2],
            core::array::from_fn(|i| wave2[4].quotient[i] + wave2[5].quotient[i])
        );
        for (coordinate, raw) in [
            raw_sub(raw_product(&sxy, &l0), raw_product(&syz, &l1)),
            raw_add(raw_product(&l2, &l1), raw_product(&l3, &l0)),
            raw_add(raw_product(&syz, &l3), raw_product(&sxy, &l2)),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                reconstruct(&D11ProductWithQuotient {
                    reduced: mixed.output[coordinate],
                    quotient: mixed.output_quotients[coordinate],
                }),
                raw
            );
        }
    }
}
