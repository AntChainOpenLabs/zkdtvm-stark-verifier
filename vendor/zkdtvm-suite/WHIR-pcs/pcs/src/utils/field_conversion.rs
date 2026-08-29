use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedValue};
use p3_maybe_rayon::prelude::*;
use std::mem::ManuallyDrop;
use std::slice;

/// Convert a vector of `BaseArray` elements to a vector of `Base` elements without any
/// reallocations. From Plonky3.
///
/// # Safety
///
/// Assumes that `BaseArray` has the same alignment and memory layout as `[Base; N]`.
#[inline]
pub unsafe fn flatten_to_base<Base, BaseArray>(vec: Vec<BaseArray>) -> Vec<Base> {
    assert_eq!(align_of::<Base>(), align_of::<BaseArray>());
    assert!(size_of::<BaseArray>().is_multiple_of(size_of::<Base>()));

    let d = size_of::<BaseArray>() / size_of::<Base>();
    let mut values = ManuallyDrop::new(vec);
    let new_len = values.len() * d;
    let new_cap = values.capacity() * d;
    let ptr = values.as_mut_ptr() as *mut Base;

    unsafe { Vec::from_raw_parts(ptr, new_len, new_cap) }
}

/// Convert a vector of `Base` elements to a vector of `BaseArray` elements.
/// Inverse of `flatten_to_base`. From Plonky3.
///
/// # Safety
///
/// Assumes that `BaseArray` has the same alignment and memory layout as `[Base; N]`.
#[inline]
pub unsafe fn reconstitute_from_base<Base, BaseArray: Clone>(mut vec: Vec<Base>) -> Vec<BaseArray> {
    assert!(size_of::<BaseArray>().is_multiple_of(size_of::<Base>()));

    let d = size_of::<BaseArray>() / size_of::<Base>();
    assert!(vec.len().is_multiple_of(d));

    let new_len = vec.len() / d;
    let cap = vec.capacity();

    if cap.is_multiple_of(d) {
        let mut values = ManuallyDrop::new(vec);
        let new_cap = cap / d;
        let ptr = values.as_mut_ptr() as *mut BaseArray;
        unsafe { Vec::from_raw_parts(ptr, new_len, new_cap) }
    } else {
        let buf_ptr = vec.as_mut_ptr().cast::<BaseArray>();
        let slice = unsafe { slice::from_raw_parts(buf_ptr, new_len) };
        slice.to_vec()
    }
}

/// Unpacks an `ExtensionPacking` value into a vector of extension field elements.
pub fn into_ef_unpacked<F: Field, EF: ExtensionField<F>>(x: EF::ExtensionPacking) -> Vec<EF> {
    let mut result = vec![EF::zero(); F::Packing::WIDTH];
    result.iter_mut().enumerate().for_each(|(i, value)| {
        *value = EF::from_base_fn(|j| x.as_base_slice()[j].as_slice()[i]);
    });
    result
}

/// Linearly interpolates between `x` and `y` with scalar `s`: `x[i] ← x[i] + s · (y[i] - x[i])`.
///
/// When `s = 0`, `x` is unchanged; when `s = 1`, `x` becomes `y`.
/// Uses SIMD-style packed arithmetic for performance.
pub fn ef_vector_add_with_scale<F: Field, EF: ExtensionField<F>>(x: &mut [EF], y: &[EF], s: EF) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let packed_s = EF::ExtensionPacking::from_base_fn(|i| F::Packing::from(s.as_base_slice()[i]));

    x.par_chunks_mut(F::Packing::WIDTH)
        .zip(y.par_chunks(F::Packing::WIDTH))
        .enumerate()
        .for_each(|(index, (chunk_x, chunk_y))| {
            let mut packed_x = EF::ExtensionPacking::from_base_fn(|i| {
                F::Packing::from_fn(|j| {
                    if j + index * F::Packing::WIDTH < len {
                        chunk_x[j].as_base_slice()[i]
                    } else {
                        F::zero()
                    }
                })
            });

            let packed_y = EF::ExtensionPacking::from_base_fn(|i| {
                F::Packing::from_fn(|j| {
                    if j + index * F::Packing::WIDTH < len {
                        chunk_y[j].as_base_slice()[i]
                    } else {
                        F::zero()
                    }
                })
            });
            packed_x += packed_s * (packed_y - packed_x);
            for i in 0..F::Packing::WIDTH {
                if i + index * F::Packing::WIDTH < len {
                    chunk_x[i] = EF::from_base_fn(|j| packed_x.as_base_slice()[j].as_slice()[i]);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, PackedValue};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn flatten_to_base_test() {
        let num_ef = 100;
        let vec_ef: Vec<EF> = (0..num_ef).map(|_| rand::random()).collect();
        let vec_ef2 = vec_ef.clone();

        let vec_f: Vec<F> = unsafe { flatten_to_base(vec_ef) };
        let vec_f_tmp = vec_f.clone();
        let vec_ef3: Vec<EF> = unsafe { reconstitute_from_base(vec_f_tmp) };
        assert_eq!(vec_ef2, vec_ef3);

        let expect_f = vec_ef2
            .iter()
            .flat_map(|ef| ef.as_base_slice())
            .cloned()
            .collect::<Vec<F>>();
        assert_eq!(vec_f, expect_f);

        let vec_ef: Vec<EF> = (0..num_ef).map(|_| rand::random()).collect();

        let vec_ef_packed: Vec<<EF as ExtensionField<F>>::ExtensionPacking> = vec_ef
            .chunks(<F as Field>::Packing::WIDTH)
            .map(|chunk| {
                <EF as ExtensionField<F>>::ExtensionPacking::from_base_fn(|i| {
                    <F as Field>::Packing::from_fn(|j| chunk[j].as_base_slice()[i])
                })
            })
            .collect();

        let vec_ef_packed2 = vec_ef_packed.clone();

        let vec_f_packed: Vec<<F as Field>::Packing> = unsafe { flatten_to_base(vec_ef_packed) };
        let vec_f_packed_tmp = vec_f_packed.clone();
        let vec_ef3_packed: Vec<<EF as ExtensionField<F>>::ExtensionPacking> =
            unsafe { reconstitute_from_base(vec_f_packed_tmp) };
        assert_eq!(vec_ef_packed2, vec_ef3_packed);
        let expect_f_packed = vec_ef_packed2
            .iter()
            .flat_map(|ef| ef.as_base_slice())
            .cloned()
            .collect::<Vec<<F as Field>::Packing>>();

        assert_eq!(vec_f_packed, expect_f_packed);
    }
}
