//! An operation to check if the input word is 0.
//!
//! This is bijective (i.e., returns 1 if and only if the input is 0). It is also worth noting that
//! this operation doesn't do a range check.
use dt_derive::AlignedBorrow;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};
use p3_air::AirBuilder;
use p3_field::Field;

use super::IsZeroOperation;

/// A set of columns needed to compute whether the given word is 0.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct IsZeroWordOperation<T> {
    /// `IsZeroOperation` to check if each byte in the input word is zero.
    pub is_zero_byte: [IsZeroOperation<T>; WORD_SIZE],

    /// A boolean flag indicating whether the lower word (the bottom 16 bits of the input) is 0.
    /// This equals `is_zero_byte[0] * is_zero_byte[1]`.
    pub is_lower_half_zero: T,

    /// A boolean flag indicating whether the upper word (the top 16 bits of the input) is 0. This
    /// equals `is_zero_byte[2] * is_zero_byte[3]`.
    pub is_upper_half_zero: T,

    /// A boolean flag indicating whether the word is zero. This equals `is_zero_byte[0] * ... *
    /// is_zero_byte[WORD_SIZE - 1]`.
    pub result: T,
}

impl<F: Field> IsZeroWordOperation<F> {
    pub fn populate(&mut self, a_u32: u32) -> u32 {
        self.populate_from_field_element(Word::from(a_u32))
    }

    pub fn populate_from_field_element(&mut self, a: Word<F>) -> u32 {
        let mut is_zero = true;
        for i in 0..WORD_SIZE {
            is_zero &= self.is_zero_byte[i].populate_from_field_element(a[i]) == 1;
        }
        self.is_lower_half_zero = self.is_zero_byte[0].result * self.is_zero_byte[1].result;
        self.is_upper_half_zero = self.is_zero_byte[2].result * self.is_zero_byte[3].result;
        self.result = F::from_bool(is_zero);
        is_zero as u32
    }

    pub fn eval<AB: DTAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Expr>,
        cols: IsZeroWordOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Calculate whether each byte is 0.
        for i in 0..WORD_SIZE {
            IsZeroOperation::<AB::F>::eval(
                builder,
                a[i].clone(),
                cols.is_zero_byte[i],
                is_real.clone(),
            );
        }

        // From here, we only assert when is_real is true.
        builder.assert_bool(is_real.clone());
        let mut builder_is_real = builder.when(is_real.clone());

        // Calculate is_upper_half_zero and is_lower_half_zero and finally the result.
        builder_is_real.assert_bool(cols.is_lower_half_zero);
        builder_is_real.assert_bool(cols.is_upper_half_zero);
        builder_is_real.assert_bool(cols.result);
        builder_is_real.assert_eq(
            cols.is_lower_half_zero,
            cols.is_zero_byte[0].result * cols.is_zero_byte[1].result,
        );
        builder_is_real.assert_eq(
            cols.is_upper_half_zero,
            cols.is_zero_byte[2].result * cols.is_zero_byte[3].result,
        );
        builder_is_real.assert_eq(cols.result, cols.is_lower_half_zero * cols.is_upper_half_zero);
    }
}

pub fn is_zero_word_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a: [AB::VarMaybeExt; 4],
    byte_inverse: [AB::VarMaybeExt; 4],
    byte_result: [AB::VarMaybeExt; 4],
    is_lower_half_zero: AB::VarMaybeExt,
    is_upper_half_zero: AB::VarMaybeExt,
    result: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    for i in 0..4 {
        super::is_zero::is_zero_op_gate_constraints(
            builder,
            a[i].clone(),
            byte_inverse[i].clone(),
            byte_result[i].clone(),
            is_real.clone(),
        );
    }
    builder
        .when(is_real.clone())
        .assert_zero(is_lower_half_zero.clone() - byte_result[0].clone() * byte_result[1].clone());
    builder
        .when(is_real.clone())
        .assert_zero(is_upper_half_zero.clone() - byte_result[2].clone() * byte_result[3].clone());
    builder.when(is_real).assert_zero(result - is_lower_half_zero * is_upper_half_zero);
}
