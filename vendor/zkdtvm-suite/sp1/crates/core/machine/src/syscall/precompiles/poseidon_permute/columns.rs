use dt_derive::AlignedBorrow;

use super::poseidon2_inner::Poseidon2Cols;
use core::mem::size_of;

use super::STATE_NUM_WORDS;
use crate::memory::MemoryReadWriteCols;

#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub(crate) struct Poseidon2MemCols<T> {
    pub poseidon2_cols: Poseidon2Cols<T>,
    pub shard: T,
    pub clk: T,
    pub state_addr: T,
    pub state_mem_read: [MemoryReadWriteCols<T>; STATE_NUM_WORDS],
    pub state_mem_write: [MemoryReadWriteCols<T>; STATE_NUM_WORDS],
    pub is_real: T,
}

pub const NUM_POSEIDON2_MEM_COLS: usize = size_of::<Poseidon2MemCols<u8>>();
pub const NUM_POSEIDON2_COLS: usize = size_of::<Poseidon2Cols<u8>>();
