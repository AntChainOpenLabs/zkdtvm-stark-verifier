use p3_field::Field;

use super::{
    constants::{D11_DEGREE, D11_SPARSE_WIDTH, D11_WIDE_DEGREE},
    field::{D11Sparse7, D11},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelCost {
    pub base_products: usize,
    pub raw_limbs: usize,
    pub reduction_steps: usize,
    pub dynamic_allocations: usize,
}

pub const SCHOOLBOOK_COST: KernelCost =
    KernelCost { base_products: 121, raw_limbs: 21, reduction_steps: 10, dynamic_allocations: 0 };
pub const SQUARE_COST: KernelCost =
    KernelCost { base_products: 66, raw_limbs: 21, reduction_steps: 10, dynamic_allocations: 0 };
pub const DENSE_SPARSE_7_COST: KernelCost =
    KernelCost { base_products: 77, raw_limbs: 21, reduction_steps: 10, dynamic_allocations: 0 };

#[must_use]
pub(crate) fn mul_schoolbook<F: Field>(left: &D11<F>, right: &D11<F>) -> D11<F> {
    let mut wide = [F::zero(); D11_WIDE_DEGREE];
    for i in 0..D11_DEGREE {
        for j in 0..D11_DEGREE {
            wide[i + j] += left.coefficients()[i] * right.coefficients()[j];
        }
    }
    reduce_wide(wide)
}

#[must_use]
pub(crate) fn square_symmetric<F: Field>(value: &D11<F>) -> D11<F> {
    let mut wide = [F::zero(); D11_WIDE_DEGREE];
    for i in 0..D11_DEGREE {
        wide[2 * i] += value.coefficients()[i].square();
        for j in i + 1..D11_DEGREE {
            wide[i + j] += (value.coefficients()[i] * value.coefficients()[j]).double();
        }
    }
    reduce_wide(wide)
}

#[must_use]
pub(crate) fn square_sparse_7<F: Field>(value: &D11Sparse7<F>) -> D11<F> {
    let mut wide = [F::zero(); D11_WIDE_DEGREE];
    for i in 0..D11_SPARSE_WIDTH {
        wide[2 * i] += value.coefficients()[i].square();
        for j in i + 1..D11_SPARSE_WIDTH {
            wide[i + j] += (value.coefficients()[i] * value.coefficients()[j]).double();
        }
    }
    reduce_wide(wide)
}

#[must_use]
pub(crate) fn mul_dense_sparse_7<F: Field>(left: &D11<F>, right: &D11Sparse7<F>) -> D11<F> {
    let mut wide = [F::zero(); D11_WIDE_DEGREE];
    for i in 0..D11_DEGREE {
        for j in 0..D11_SPARSE_WIDTH {
            wide[i + j] += left.coefficients()[i] * right.coefficients()[j];
        }
    }
    reduce_wide(wide)
}

/// Closed reduction for `z^11 - z^3 - 2`; descending order is essential
/// because reducing degrees 19 and 20 creates new degree-11/12 terms.
#[must_use]
pub(crate) fn reduce_wide<F: Field>(mut wide: [F; D11_WIDE_DEGREE]) -> D11<F> {
    for degree in (D11_DEGREE..D11_WIDE_DEGREE).rev() {
        let value = wide[degree];
        wide[degree] = F::zero();
        wide[degree - 8] += value;
        wide[degree - 11] += value.double();
    }
    D11::new(core::array::from_fn(|i| wide[i]))
}
