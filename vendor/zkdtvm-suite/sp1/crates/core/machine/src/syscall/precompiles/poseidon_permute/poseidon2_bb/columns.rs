use core::mem::size_of;

use dt_derive::AlignedBorrow;

pub(crate) const WIDTH: usize = 24;
pub(crate) const HFROUNDS: usize = 4;
pub(crate) const PROUNDS: usize = 21;
pub(crate) const SBOX_REGISTERS: usize = 1;

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct Poseidon2Cols<T> {
    pub export: T,
    pub inputs: [T; WIDTH],
    pub beginning_full_rounds: [FullRound<T>; HFROUNDS],
    pub partial_rounds: [PartialRound<T>; PROUNDS],
    pub ending_full_rounds: [FullRound<T>; HFROUNDS],
}

impl<T: Clone> Poseidon2Cols<T> {
    pub fn output_state(&self) -> &[T; WIDTH] {
        &self.ending_full_rounds[HFROUNDS - 1].post
    }
}

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FullRound<T> {
    pub sbox: [SBox<T>; WIDTH],
    pub post: [T; WIDTH],
}

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct PartialRound<T> {
    pub sbox: SBox<T>,
    pub post_sbox: T,
}

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct SBox<T>(pub [T; SBOX_REGISTERS]);

pub const fn num_cols() -> usize {
    size_of::<Poseidon2Cols<u8>>()
}
