pub mod air;
pub mod byte_polyair;
pub mod columns;
pub mod polyair;
pub mod trace;
pub mod utils;

use dt_core_executor::{events::ByteLookupEvent, ByteOpcode};

use core::borrow::BorrowMut;
use std::marker::PhantomData;

use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use self::{
    columns::{BytePreprocessedCols, NUM_BYTE_PREPROCESSED_COLS},
    utils::shr_carry,
};
use crate::{bytes::trace::NUM_ROWS, utils::zeroed_f_vec};

/// The number of different byte operations. + bit range
pub const NUM_BYTE_OPS: usize = 11;

/// A chip for computing byte operations.
///
/// The chip contains a preprocessed table of all possible byte operations. Other chips can then
/// use lookups into this table to compute their own operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteChip<F>(PhantomData<F>);

impl<F: Field> ByteChip<F> {
    /// Creates the preprocessed byte trace.
    ///
    /// This function returns a `trace` which is a matrix containing all possible byte operations.
    pub fn trace() -> RowMajorMatrix<F> {
        // The trace containing all values, with all multiplicities set to zero.
        let mut initial_trace = RowMajorMatrix::new(
            zeroed_f_vec(NUM_ROWS * NUM_BYTE_PREPROCESSED_COLS),
            NUM_BYTE_PREPROCESSED_COLS,
        );

        // Record all the necessary operations for each byte lookup.
        let opcodes = ByteOpcode::all();

        // Iterate over all options for pairs of bytes `a` and `b`.
        for (row_index, (b, c)) in (0..=u8::MAX).cartesian_product(0..=u8::MAX).enumerate() {
            let b = b as u8;
            let c = c as u8;
            let col: &mut BytePreprocessedCols<F> = initial_trace.row_mut(row_index).borrow_mut();

            // Set the values of `b` and `c`.
            col.b = F::from_canonical_u8(b);
            col.c = F::from_canonical_u8(c);
            // Iterate over all operations for results and updating the table map.
            for opcode in opcodes.iter() {
                match opcode {
                    ByteOpcode::AND => {
                        let and = b & c;
                        col.and = F::from_canonical_u8(and);
                        ByteLookupEvent::new(*opcode, and as u16, 0, b, c)
                    }
                    ByteOpcode::OR => {
                        let or = b | c;
                        col.or = F::from_canonical_u8(or);
                        ByteLookupEvent::new(*opcode, or as u16, 0, b, c)
                    }
                    ByteOpcode::XOR => {
                        let xor = b ^ c;
                        col.xor = F::from_canonical_u8(xor);
                        ByteLookupEvent::new(*opcode, xor as u16, 0, b, c)
                    }
                    ByteOpcode::SLL => {
                        let sll = b << (c & 7);
                        col.sll = F::from_canonical_u8(sll);
                        ByteLookupEvent::new(*opcode, sll as u16, 0, b, c)
                    }
                    ByteOpcode::U8Range => ByteLookupEvent::new(*opcode, 0, 0, b, c),
                    ByteOpcode::ShrCarry => {
                        let (res, carry) = shr_carry(b, c);
                        col.shr = F::from_canonical_u8(res);
                        col.shr_carry = F::from_canonical_u8(carry);
                        ByteLookupEvent::new(*opcode, res as u16, carry, b, c)
                    }
                    ByteOpcode::LTU => {
                        let ltu = b < c;
                        col.ltu = F::from_bool(ltu);
                        ByteLookupEvent::new(*opcode, ltu as u16, 0, b, c)
                    }
                    ByteOpcode::MSB => {
                        let msb = (b & 0b1000_0000) != 0;
                        col.msb = F::from_bool(msb);
                        ByteLookupEvent::new(*opcode, msb as u16, 0, b, 0)
                    }
                    ByteOpcode::BitRange => {
                        //prepare range table
                        let (bit_width, value) = get_bit_value_from_u32(row_index as u32);
                        col.bit_range =
                            [F::from_canonical_u32(value), F::from_canonical_u32(bit_width)];
                        ByteLookupEvent::new(*opcode, value as u16, bit_width as u8, 0, 0)
                    }
                    ByteOpcode::U16Range => {
                        let v = ((b as u32) << 8) + c as u32;
                        col.value_u16 = F::from_canonical_u32(v);
                        ByteLookupEvent::new(*opcode, v as u16, 0, 0, 0)
                    }
                    ByteOpcode::BitVec => {
                        for bit in 0..8u8 {
                            col.bit_vec[bit as usize] = F::from_bool((b >> bit) & 1 == 1);
                        }
                        for bit in 0..8u8 {
                            col.bit_vec[8 + bit as usize] = F::from_bool((c >> bit) & 1 == 1);
                        }
                        ByteLookupEvent::new(*opcode, 0, 0, b, c)
                    }
                };
            }
        }

        initial_trace
    }
}
/*
0bit 0
1bit 0,1
2bit 0,1,2,3
3bit 0,1,2,3,4,5,6,7
...
*/
fn get_bit_value_from_u32(index: u32) -> (u32, u32) {
    if index == 0 {
        (0, 0)
    } else {
        let bit_width = index.ilog2();
        let start_index = 1 << bit_width;
        let value = index - start_index;
        (bit_width, value)
    }
}
fn get_index_from_range_value(value: u16, bit: u8) -> usize {
    let start_index = 1u32 << bit;
    (start_index + value as u32) as usize
}
#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use p3_baby_bear::BabyBear;
    use std::time::Instant;

    use super::*;

    #[test]
    pub fn test_trace_and_map() {
        let start = Instant::now();
        ByteChip::<BabyBear>::trace();
        println!("trace and map: {:?}", start.elapsed());
    }
}
