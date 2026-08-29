use std::marker::PhantomData;

use dt_derive::AlignedBorrow;
use dt_stark::Word;
use p3_field::{AbstractField, Field};

use crate::operations_dt::CompactWord;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
pub struct NotOperation<T> {
    _p: PhantomData<T>,
}

impl<F: Field> NotOperation<F> {
    pub fn populate(input: u32) -> u32 {
        !input
    }

    pub fn eval<Expr: AbstractField>(input: CompactWord<impl Into<Expr>>) -> CompactWord<Expr> {
        let input = input.0.map(|input| input.into());
        let mask = Expr::from_canonical_u32(u16::MAX as u32);

        CompactWord([mask.clone() - input[0].clone(), mask - input[1].clone()])
    }

    pub fn eval_word<Expr: AbstractField>(input: Word<impl Into<Expr>>) -> Word<Expr> {
        let input = input.map(|input| input.into());
        let mask = Expr::from_canonical_u32(u8::MAX as u32);

        input.map(|input| mask.clone() - input)
    }
}
