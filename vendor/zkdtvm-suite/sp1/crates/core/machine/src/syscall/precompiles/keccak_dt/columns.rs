use core::mem::size_of;

use dt_derive::AlignedBorrow;

use crate::syscall::precompiles::keccak_dt::keccak_cols::KeccakCols;

pub const NUM_KECCAK_PERMUTE_COLS: usize = size_of::<KeccakPermuteCols<u8>>();

#[derive(AlignedBorrow)]
#[repr(C)]
pub(crate) struct KeccakPermuteCols<T> {
    pub shard: T,
    pub clk: T,

    pub keccak: KeccakCols<T>,

    pub is_real: T,
}
