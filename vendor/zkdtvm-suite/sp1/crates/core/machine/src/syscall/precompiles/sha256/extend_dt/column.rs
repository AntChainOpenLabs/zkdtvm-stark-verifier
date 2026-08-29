use dt_derive::AlignedBorrow;
use typenum::{U3, U4};

use crate::{
    memory::{MemoryReadCols, MemoryWriteCols},
    operations_dt::{
        AddNOperationWithoutResult, CompactWordToWordWitness, FixedRotateRightOperation,
        FixedShiftRightOperation, XorNOperation,
    },
};

pub const NUM_SHA_EXTEND_COLS: usize = size_of::<ShaExtendCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendCols<T> {
    /// Inputs.
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,

    /// Control flags.
    pub i: T,

    /// Inputs to `s0`.
    pub w_i_minus_15: MemoryReadCols<T>,
    pub w_i_minus_15_rr_7: FixedRotateRightOperation<T>,
    pub w_i_minus_15_rr_18: FixedRotateRightOperation<T>,
    pub w_i_minus_15_rs_3: FixedShiftRightOperation<T>,

    pub w_i_minus_15_rr_7_witness: CompactWordToWordWitness<T>,
    pub w_i_minus_15_rr_18_witness: CompactWordToWordWitness<T>,
    pub w_i_minus_15_rs_3_witness: CompactWordToWordWitness<T>,

    /// `s0 := (w[i-15] rightrotate  7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)`.
    pub s0: XorNOperation<T, U3>,

    /// Inputs to `s1`.
    pub w_i_minus_2: MemoryReadCols<T>,
    pub w_i_minus_2_rr_17: FixedRotateRightOperation<T>,
    pub w_i_minus_2_rr_19: FixedRotateRightOperation<T>,
    pub w_i_minus_2_rs_10: FixedShiftRightOperation<T>,

    pub w_i_minus_2_rr_17_witness: CompactWordToWordWitness<T>,
    pub w_i_minus_2_rr_19_witness: CompactWordToWordWitness<T>,
    pub w_i_minus_2_rs_10_witness: CompactWordToWordWitness<T>,

    /// `s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)`.
    pub s1: XorNOperation<T, U3>,

    /// Inputs to `s2`.
    pub w_i_minus_16: MemoryReadCols<T>,
    pub w_i_minus_7: MemoryReadCols<T>,

    /// `w[i] := w[i-16] + s0 + w[i-7] + s1`.
    pub s2: AddNOperationWithoutResult<T, U4>,

    /// Result.
    pub w_i: MemoryWriteCols<T>,

    /// Selector.
    pub is_real: T,
}
