use dt_core_executor::events::ByteRecord;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};

use dt_derive::AlignedBorrow;
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use crate::air::WordAirBuilder;

/// A set of columns needed to compute the add of two words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddOperation<T> {
    /// The result of `a + b`.
    pub value: Word<T>,
    // /// Trace.
    // pub carry: [T; 3],
}

impl<F: Field> AddOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32) -> u32 {
        let expected = a_u32.wrapping_add(b_u32);
        self.value = Word::from(expected);

        // Range check

        record.add_u8_range_checks(&expected.to_le_bytes());

        expected
    }

    pub fn eval<AB: DTAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base = AB::F::from_canonical_u32(256);
        builder.assert_bool(is_real.clone());
        let mut builder_is_real = builder.when(is_real.clone());
        let mut carry = AB::Expr::zero();
        // The set of constraints are
        //  - carry is initialized to zero
        //  - 2^8 * carry_next + value[i] = a[i] + b[i] + carry
        //  - carry is boolean
        //  - 0 <= value[i] < 2^8
        for i in 0..WORD_SIZE {
            carry = (Into::<AB::Expr>::into(a[i]) + Into::<AB::Expr>::into(b[i]) - cols.value[i] +
                carry) *
                base.inverse();
            builder_is_real.assert_bool(carry.clone());
        }

        // Range check each byte.
        builder.slice_range_check_u8(&cols.value.0, is_real);
    }
}

pub(crate) const ADD_OP_NUM_INTERACTIONS: usize = 2;

pub fn add_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a: [AB::VarMaybeExt; 4],
    b: [AB::VarMaybeExt; 4],
    value: [AB::VarMaybeExt; 4],
    is_real: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    let base_inv: AB::VarMaybeExt = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256).inverse());
    let mut carry = AB::zero_maybe();
    for i in 0..4 {
        carry = (a[i].clone() + b[i].clone() - value[i].clone() + carry) * base_inv.clone();
        builder.when(is_real.clone()).assert_zero(carry.clone() * (one.clone() - carry.clone()));
    }
}

pub fn add_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    value: &dt_stark::Word<AB::VarMaybeExt>,
) {
    use crate::bytes::polyair::u8_range_pair_precompute_lc;
    u8_range_pair_precompute_lc(builder, value[0].clone(), value[1].clone());
    u8_range_pair_precompute_lc(builder, value[2].clone(), value[3].clone());
}

pub fn add_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    builder.send(is_real.clone());
    builder.send(is_real);
}
