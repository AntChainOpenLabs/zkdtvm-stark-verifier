use dt_core_executor::events::ByteRecord;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{air::DTAirBuilder, Word};

use dt_derive::AlignedBorrow;
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use crate::air::WordAirBuilder;

/// A set of columns needed to compute the add of two words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct SubOperation<T> {
    /// The result of `a - b`.
    pub value: Word<T>,
}

impl<F: Field> SubOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32) -> u32 {
        let expected = a_u32.wrapping_sub(b_u32);
        self.value = Word::from(expected);

        // Range check

        record.add_u8_range_checks(&expected.to_le_bytes());

        expected
    }

    pub fn eval<AB: DTAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: SubOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base = AB::F::from_canonical_u32(256);
        builder.assert_bool(is_real.clone());
        let one = AB::Expr::one();
        let mut builder_is_real = builder.when(is_real.clone());
        let mut carry = AB::Expr::one();

        for i in 0..WORD_SIZE {
            carry = (a[i] + base - one.clone() - b[i] - cols.value[i] + carry) * base.inverse();
            builder_is_real.assert_bool(carry.clone());
        }

        // Range check each byte.
        // builder.slice_range_check_u8(&a.0, is_real.clone());
        //     builder.slice_range_check_u8(&b.0, is_real.clone());
        builder.slice_range_check_u8(&cols.value.0, is_real);
    }
}
