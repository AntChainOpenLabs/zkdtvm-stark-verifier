use std::mem::size_of;

use dt_derive::AlignedBorrow;
use typenum::{U2, U3, U5};

use crate::{
    memory::MemoryReadCols,
    operations_dt::{
        AddNOperation, AndNOperation, CompactWord, CompactWordToWordWitness,
        FixedRotateRightOperation, NotOperation, XorNOperation,
    },
};

pub const NUM_SHA_COMPRESS_COLS: usize = size_of::<ShaCompressCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaCompressCols<T> {
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,

    pub i: T,
    pub i_low_one_hot: [T; 8],
    pub i_high_one_hot: [T; 8],

    pub w_access: MemoryReadCols<T>,

    pub a: CompactWord<T>,
    pub b: CompactWord<T>,
    pub c: CompactWord<T>,
    pub d: CompactWord<T>,
    pub e: CompactWord<T>,
    pub f: CompactWord<T>,
    pub g: CompactWord<T>,
    pub h: CompactWord<T>,

    pub a_witness: CompactWordToWordWitness<T>,
    pub b_witness: CompactWordToWordWitness<T>,
    pub c_witness: CompactWordToWordWitness<T>,
    pub e_witness: CompactWordToWordWitness<T>,
    pub f_witness: CompactWordToWordWitness<T>,
    pub g_witness: CompactWordToWordWitness<T>,

    pub k: CompactWord<T>,

    pub e_rr_6: FixedRotateRightOperation<T>,
    pub e_rr_11: FixedRotateRightOperation<T>,
    pub e_rr_25: FixedRotateRightOperation<T>,

    pub e_rr_6_witness: CompactWordToWordWitness<T>,
    pub e_rr_11_witness: CompactWordToWordWitness<T>,
    pub e_rr_25_witness: CompactWordToWordWitness<T>,

    /// `S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)`.
    pub s1: XorNOperation<T, U3>,

    pub e_and_f: AndNOperation<T, U2>,
    pub e_not: NotOperation<T>,
    pub e_not_and_g: AndNOperation<T, U2>,

    /// `ch := (e and f) xor ((not e) and g)`.
    pub ch: XorNOperation<T, U2>,

    /// `temp1 := h + S1 + ch + k[i] + w[i]`.
    pub temp1: AddNOperation<T, U5>,

    pub a_rr_2: FixedRotateRightOperation<T>,
    pub a_rr_13: FixedRotateRightOperation<T>,
    pub a_rr_22: FixedRotateRightOperation<T>,

    pub a_rr_2_witness: CompactWordToWordWitness<T>,
    pub a_rr_13_witness: CompactWordToWordWitness<T>,
    pub a_rr_22_witness: CompactWordToWordWitness<T>,

    /// `S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22)`.
    pub s0: XorNOperation<T, U3>,

    pub a_and_b: AndNOperation<T, U2>,
    pub a_and_c: AndNOperation<T, U2>,
    pub b_and_c: AndNOperation<T, U2>,

    /// `maj := (a and b) xor (a and c) xor (b and c)`.
    pub maj: XorNOperation<T, U3>,

    /// `temp2 := S0 + maj`.
    pub temp2: AddNOperation<T, U2>,

    /// The next value of `e` is `d + temp1`.
    pub d_add_temp1: AddNOperation<T, U2>,
    /// The next value of `a` is `temp1 + temp2`.
    pub temp1_add_temp2: AddNOperation<T, U2>,

    pub is_real: T,
}
