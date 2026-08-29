use std::ops::{Index, IndexMut};

use dt_derive::AlignedBorrow;
use dt_stark::{air::FullAirBuilder, Word};
use p3_field::{AbstractField, Field};
use serde::{Deserialize, Serialize};

#[derive(
    AlignedBorrow, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
#[repr(C)]
pub struct CompactWord<T>(pub [T; 2]);

impl<T> Index<usize> for CompactWord<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for CompactWord<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: AbstractField> From<u32> for CompactWord<F> {
    fn from(value: u32) -> Self {
        CompactWord([
            F::from_canonical_u32(value & 0xFFFFu32),
            F::from_canonical_u32((value >> 16) & 0xFFFFu32),
        ])
    }
}

impl<F: AbstractField, T: Into<F>> From<Word<T>> for CompactWord<F> {
    fn from(value: Word<T>) -> Self {
        let [value0, value1, value2, value3] = value.0;
        let multiplier = F::from_canonical_u32(1u32 << 8);
        CompactWord([
            value0.into() + value1.into() * multiplier.clone(),
            value2.into() + value3.into() * multiplier,
        ])
    }
}

#[derive(
    AlignedBorrow, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
#[repr(C)]
pub struct CompactWordToWordWitness<T>(pub [T; 2]);

impl<T> Index<usize> for CompactWordToWordWitness<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for CompactWordToWordWitness<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: AbstractField> From<u32> for CompactWordToWordWitness<F> {
    fn from(value: u32) -> Self {
        Self([
            F::from_canonical_u32((value >> 8) & 0xFFu32),
            F::from_canonical_u32((value >> 24) & 0xFFu32),
        ])
    }
}

impl<F: AbstractField> CompactWord<F> {
    pub fn into_word<Expr: AbstractField + From<F>>(
        cols: CompactWord<impl Into<Expr>>,
        higher_bytes: CompactWordToWordWitness<impl Into<Expr>>,
    ) -> Word<Expr> {
        let multiplier: Expr = F::from_canonical_u32(1u32 << 8).into();

        let [cols0, cols1] = cols.0.map(|a| a.into());
        let [higher_bytes0, higher_bytes1] = higher_bytes.0.map(|a| a.into());
        let [lower_bytes0, lower_bytes1] = [
            cols0 - higher_bytes0.clone() * multiplier.clone(),
            cols1 - higher_bytes1.clone() * multiplier,
        ];

        Word([lower_bytes0, higher_bytes0, lower_bytes1, higher_bytes1])
    }
}

impl<F: Field> CompactWord<F> {
    pub fn as_u32(&self) -> u32 {
        self[0].as_u32() | (self[1].as_u32() << 16)
    }
}

pub fn compact_word_to_arr<AB: FullAirBuilder>(
    compact: &CompactWord<AB::VarMaybeExt>,
    witness: &CompactWordToWordWitness<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; 4] {
    let byte_shift = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8));
    let low0 = compact.0[0].clone() - witness.0[0].clone() * byte_shift.clone();
    let low1 = compact.0[1].clone() - witness.0[1].clone() * byte_shift;
    [low0, witness.0[0].clone(), low1, witness.0[1].clone()]
}

pub fn word_to_compact<AB: FullAirBuilder>(
    word: &Word<AB::VarMaybeExt>,
) -> CompactWord<AB::VarMaybeExt> {
    let byte_shift = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8));
    CompactWord([
        word[0].clone() + word[1].clone() * byte_shift.clone(),
        word[2].clone() + word[3].clone() * byte_shift,
    ])
}
