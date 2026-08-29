use core::mem::size_of;

use dt_derive::AlignedBorrow;

pub(crate) const WIDTH: usize = 24;
pub(crate) const NUM_EXTERNAL_ROUNDS: usize = 8;
pub(crate) const HFROUNDS: usize = NUM_EXTERNAL_ROUNDS / 2;
pub(crate) const NUM_INTERNAL_ROUNDS: usize = 23;

/// Columns for a KoalaBear Poseidon2 permutation (degree-3 S-box, width 24).
///
/// With a degree-3 S-box, no intermediate S-box columns are needed. The state
/// at each external round boundary and the s[0] value after each internal round
/// are sufficient to keep constraints at degree 3.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct Poseidon2Cols<T> {
    pub export: T,
    pub inputs: [T; WIDTH],
    pub external_rounds_state: [[T; WIDTH]; NUM_EXTERNAL_ROUNDS],
    pub internal_rounds_state: [T; WIDTH],
    pub internal_rounds_s0: [T; NUM_INTERNAL_ROUNDS - 1],
    pub output_state: [T; WIDTH],
}

impl<T> Poseidon2Cols<T> {
    pub fn output_state(&self) -> &[T; WIDTH] {
        &self.output_state
    }
}

pub const fn num_cols() -> usize {
    size_of::<Poseidon2Cols<u8>>()
}
