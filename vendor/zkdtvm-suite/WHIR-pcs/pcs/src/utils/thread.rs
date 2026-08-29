use p3_field::Field;

/// Allocates a zero-initialized vector of field elements using `alloc_zeroed`.
///
/// This is faster than `vec![F::zero(); size]` for large allocations because it
/// bypasses per-element initialization and relies on the OS providing zeroed pages.
///
/// # Safety
///
/// This function asserts at runtime that `F::zero()` has an all-zero byte representation.
/// If this invariant does not hold, the function will panic.
pub fn unsafe_allocate_zero_vec<F: Field + Sized>(size: usize) -> Vec<F> {
    unsafe {
        let value = &F::zero();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        Vec::from_raw_parts(ptr, size, size)
    }
}
