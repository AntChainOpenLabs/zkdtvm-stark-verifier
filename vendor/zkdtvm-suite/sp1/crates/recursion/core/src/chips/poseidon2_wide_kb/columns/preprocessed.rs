use dt_core_machine::operations::poseidon2::WIDTH;
use dt_derive::AlignedBorrow;

use crate::{chips::mem::MemoryAccessColsChips, Address};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2PreprocessedColsWideKb<T: Copy> {
    pub input: [Address<T>; WIDTH],
    pub output: [MemoryAccessColsChips<T>; WIDTH],
    pub is_real_neg: T,
}
