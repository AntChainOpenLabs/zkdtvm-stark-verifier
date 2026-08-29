// use std::ops::{Add, Sub};

// use generic_array::{ArrayLength, GenericArray};
// use p3_field::Field;
// use dt_core_executor::{events::ByteRecord, ByteOpcode};
// use dt_derive::AlignedBorrow;
// use dt_stark::{air::DTAirBuilder, Word};
// use typenum::{IsGreaterOrEqual, True, U1, U2};

// use crate::operations_dt::{CompactWord, CompactWordToWordWitness};

// #[derive(AlignedBorrow, Default, Debug, Clone)]
// #[repr(C)]
// pub struct XorNOperation<
//     T,
//     NW: ArrayLength, // number of Word
//     NCW: ArrayLength // number of CompactWord
//         + Add< NW, Output: ArrayLength
//                         + Sub<U1, Output: ArrayLength>
//                         + IsGreaterOrEqual<U2, Output = True>,
//         >,
// > { /// The result of the xor operation. pub value: GenericArray<Word<T>, <<NCW as
// > Add<NW>>::Output as Sub<U1>>::Output>,

//     /// Higher bytes.
//     pub higher_bytes: GenericArray<CompactWordToWordWitness<T>, NCW>,
// }

// impl<
//         T: Copy,
//         NW: ArrayLength,
//         NCW: ArrayLength<ArrayType<CompactWordToWordWitness<T>>: Copy>
//             + Add< NW, Output: ArrayLength
//                             + Sub<U1, Output: ArrayLength<ArrayType<Word<T>>: Copy>>
//                             + IsGreaterOrEqual<U2, Output = True>,
//             >,
//     > Copy for XorNOperation<T, NW, NCW>
// {
// }

// impl<
//         F: Field,
//         NW: ArrayLength,
//         NCW: ArrayLength
//             + Add< NW, Output: ArrayLength
//                             + Sub<U1, Output: ArrayLength>
//                             + IsGreaterOrEqual<U2, Output = True>,
//             >,
//     > XorNOperation<F, NW, NCW>
// {
//     pub fn populate(
//         &mut self,
//         record: &mut impl ByteRecord,
//         word: &[u32],
//         compress_word: &[u32],
//     ) -> u32 {
//         debug_assert_eq!(word.len(), NW::USIZE);
//         debug_assert_eq!(compress_word.len(), NCW::USIZE);

//         let mut result = if NW::USIZE == 0 {
//             self.higher_bytes[0] = compress_word[0].into();
//             compress_word[0]
//         } else {
//             word[0]
//         };

//         for (i, &c) in word.iter().chain(compress_word.iter()).enumerate().skip(1) {
//             if i >= NW::USIZE {
//                 self.higher_bytes[i - NW::USIZE] = c.into();
//             }

//             let b_bytes = result.to_le_bytes();
//             let c_bytes = c.to_le_bytes();
//             for j in 0..4 {
//                 record.lookup_xor(b_bytes[j], c_bytes[j]);
//             }

//             result = result ^ c;
//             self.value[i - 1] = result.into();
//         }

//         result
//     }

//     /// Evaluate the xor operation over two u32s of two u16 limbs.
//     /// Assumes that the two words are valid u32s of two u16 limbs.
//     /// If `is_real` is true, the return value is constrained to be correct.
//     pub fn eval<AB: DTAirBuilder<F = F>>(
//         cols: XorNOperation<AB::Var, NW, NCW>,
//         builder: &mut AB,
//         word: &[Word<impl Into<AB::Expr> + Clone>],
//         compress_word: &[CompactWord<impl Into<AB::Expr> + Clone>],
//         is_real: AB::Var,
//     ) {
//         debug_assert_eq!(word.len(), NW::USIZE);
//         debug_assert_eq!(compress_word.len(), NCW::USIZE);

//         let mut result = if NW::USIZE == 0 {
//             CompactWord::<F>::into_word(compress_word[0].clone(), cols.higher_bytes[0])
//         } else {
//             word[0].clone().map(|byte| byte.into())
//         };

//         for (i, c) in word
//             .iter()
//             .map(|word| word.clone().map(|byte| byte.into()))
//             .chain(compress_word.iter().zip(cols.higher_bytes.iter()).map(
//                 |(compact_word, higher_bytes)| {
//                     CompactWord::<F>::into_word(compact_word.clone(), *higher_bytes)
//                 },
//             ))
//             .enumerate()
//         {
//             let r = cols.value[i - 1].map(|a| a.into());

//             for j in 0..4 {
//                 builder.send_byte(
//                     F::from_canonical_u32(ByteOpcode::XOR as u32),
//                     r[j].clone(),
//                     result[j].clone(),
//                     c[j].clone(),
//                     is_real,
//                 );
//             }

//             result = r;
//         }
//     }
// }

use std::ops::Sub;

use dt_core_executor::{events::ByteRecord, ByteOpcode};
use dt_derive::AlignedBorrow;
use dt_stark::{air::DTAirBuilder, Word};
use generic_array::{ArrayLength, GenericArray};
use p3_field::Field;
use typenum::{IsGreaterOrEqual, True, U1, U2};

#[derive(AlignedBorrow, Default, Debug, Clone)]
#[repr(C)]
pub struct XorNOperation<
    T,
    N: ArrayLength + Sub<U1, Output: ArrayLength> + IsGreaterOrEqual<U2, Output = True>,
> {
    pub value: GenericArray<Word<T>, <N as Sub<U1>>::Output>,
}

impl<
        T: Copy,
        N: ArrayLength
            + Sub<U1, Output: ArrayLength<ArrayType<Word<T>>: Copy>>
            + IsGreaterOrEqual<U2, Output = True>,
    > Copy for XorNOperation<T, N>
{
}

impl<
        F: Field,
        N: ArrayLength + Sub<U1, Output: ArrayLength> + IsGreaterOrEqual<U2, Output = True>,
    > XorNOperation<F, N>
{
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        input: impl IntoIterator<Item = u32>,
    ) -> u32 {
        input.into_iter().enumerate().fold(0u32, |acc, (i, input)| {
            if i == 0 {
                input
            } else {
                let b_bytes = acc.to_le_bytes();
                let c_bytes = input.to_le_bytes();
                for i in 0..4 {
                    record.lookup_xor(b_bytes[i], c_bytes[i]);
                }

                let result = acc ^ input;
                self.value[i - 1] = result.into();
                result
            }
        })
    }

    pub fn eval<AB: DTAirBuilder<F = F>>(
        cols: &XorNOperation<AB::Var, N>,
        builder: &mut AB,
        input: impl IntoIterator<Item = Word<impl Into<AB::Expr>>>,
        is_real: impl Into<AB::Expr>,
    ) -> Word<AB::Expr> {
        let is_real = is_real.into();

        input.into_iter().enumerate().fold(Word::default(), |acc, (i, input)| {
            let input = input.map(|input| input.into());
            if i == 0 {
                input
            } else {
                for j in 0..4 {
                    builder.send_byte(
                        F::from_canonical_u32(ByteOpcode::XOR as u32),
                        cols.value[i - 1][j],
                        acc[j].clone(),
                        input[j].clone(),
                        is_real.clone(),
                    );
                }
                cols.value[i - 1].map(|v| v.into())
            }
        })
    }
}

// ============================================================================
// PolyAir three-phase helpers for XorNOperation
// ============================================================================

use crate::bytes::polyair::{xor_lookup, xor_precompute_lc};

pub const fn xor_n_num_interactions(n: usize) -> usize {
    (n - 1) * 4
}

pub fn xor_n_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    result_words: &[[AB::VarMaybeExt; 4]],
    acc_words: &[[AB::VarMaybeExt; 4]],
    input_words: &[[AB::VarMaybeExt; 4]],
) {
    debug_assert_eq!(result_words.len(), acc_words.len());
    debug_assert_eq!(result_words.len(), input_words.len());

    for (result, (acc, input)) in result_words.iter().zip(acc_words.iter().zip(input_words.iter()))
    {
        for j in 0..4 {
            xor_precompute_lc(builder, result[j].clone(), acc[j].clone(), input[j].clone());
        }
    }
}

pub fn xor_n_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    n: usize,
) {
    let count = xor_n_num_interactions(n);
    for _ in 0..count {
        xor_lookup(builder, is_real.clone());
    }
}
