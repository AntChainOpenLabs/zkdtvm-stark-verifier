use std::marker::PhantomData;

use dt_core_executor::events::ByteRecord;
use dt_derive::AlignedBorrow;
use dt_stark::air::DTAirBuilder;
use generic_array::ArrayLength;
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use typenum::{IsGreaterOrEqual, True, U2};

use crate::{air::WordAirBuilder, operations_dt::CompactWord};

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddNOperationWithoutResult<T, N: ArrayLength + IsGreaterOrEqual<U2, Output = True>> {
    _p: PhantomData<(T, N)>,
}

impl<F: Field, N: ArrayLength + IsGreaterOrEqual<U2, Output = True>>
    AddNOperationWithoutResult<F, N>
{
    pub fn populate(record: &mut impl ByteRecord, input: impl IntoIterator<Item = u32>) -> u32 {
        let (result, sum) = input.into_iter().fold((0u32, [0u32; 2]), |acc, input| {
            (acc.0 + input, [acc.1[0] + (input & 0xFFFFu32), acc.1[1] + (input >> 16)])
        });

        let mut carry = 0u32;
        let carry: [_; 2] = std::array::from_fn(|i| {
            carry = (sum[i] + carry) >> 16;
            carry as u8
        });

        if N::USIZE > 2 {
            record.add_u8_range_checks(&carry);
        }

        result
    }

    /// Evaluate the addn operation.
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder<F = F>>(
        builder: &mut AB,
        input: impl IntoIterator<Item = CompactWord<impl Into<AB::Expr>>>,
        result: CompactWord<impl Into<AB::Expr>>,
        is_real: AB::Var,
    ) {
        let sum = input.into_iter().fold(CompactWord::<AB::Expr>::default(), |acc, input| {
            let [acc0, acc1] = acc.0;
            let [input0, input1] = input.0.map(|input| input.into());
            CompactWord([acc0 + input0, acc1 + input1])
        });

        let divisor = F::from_canonical_u32(1u32 << 16).inverse();
        let result = result.0.map(|result| result.into());
        let is_real = is_real.into();

        let mut carry = AB::Expr::zero();
        let carry: [_; 2] = std::array::from_fn(|i| {
            carry = (sum[i].clone() + carry.clone() - result[i].clone()) * divisor;
            carry.clone()
        });

        if N::USIZE == 2 {
            carry.into_iter().for_each(|c| builder.when(is_real.clone()).assert_bool(c));
        } else if N::USIZE > 2 {
            builder.slice_range_check_u8(&carry, is_real);
        }
    }
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddNOperation<T, N: ArrayLength + IsGreaterOrEqual<U2, Output = True>> {
    pub value: CompactWord<T>,
    _p: PhantomData<N>,
}

impl<F: Field, N: ArrayLength + IsGreaterOrEqual<U2, Output = True>> AddNOperation<F, N> {
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        input: impl IntoIterator<Item = u32>,
    ) -> u32 {
        let result = AddNOperationWithoutResult::<F, N>::populate(record, input);
        self.value = result.into();
        for i in 0..2 {
            record.add_u16_range_check(((result >> (16 * i)) & 0xFFFFu32) as u16);
        }
        result
    }

    pub fn eval<AB: DTAirBuilder<F = F>>(
        cols: AddNOperation<AB::Var, N>,
        builder: &mut AB,
        input: impl IntoIterator<Item = CompactWord<impl Into<AB::Expr>>>,
        is_real: AB::Var,
    ) {
        AddNOperationWithoutResult::<F, N>::eval(builder, input, cols.value, is_real);
        builder.slice_range_check_u16(&cols.value.0, is_real);
    }
}

// ============================================================================
// PolyAir three-phase helpers for AddNOperation{WithoutResult}
// ============================================================================

use crate::bytes::polyair::{
    slice_u16_range_lookup, slice_u16_range_precompute_lc, slice_u8_range_lookup,
    slice_u8_range_precompute_lc,
};
use dt_stark::air::FullAirBuilder as _;

fn add_n_carries<AB: dt_stark::air::FullAirBuilder>(
    input: &[CompactWord<AB::VarMaybeExt>],
    result: CompactWord<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; 2] {
    let sum = input.iter().fold([AB::zero_maybe(), AB::zero_maybe()], |[acc0, acc1], word| {
        [acc0 + word[0].clone(), acc1 + word[1].clone()]
    });
    let divisor = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 16).inverse());

    let carry_0 = (sum[0].clone() - result[0].clone()) * divisor.clone();
    let carry_1 = (sum[1].clone() + carry_0.clone() - result[1].clone()) * divisor;
    [carry_0, carry_1]
}

pub fn add_n_without_result_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    input: &[CompactWord<AB::VarMaybeExt>],
    result: CompactWord<AB::VarMaybeExt>,
) {
    if input.len() > 2 {
        let carry = add_n_carries::<AB>(input, result);
        slice_u8_range_precompute_lc(builder, &carry);
    }
}

pub fn add_n_without_result_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    input: &[CompactWord<AB::VarMaybeExt>],
    result: CompactWord<AB::VarMaybeExt>,
    is_real: AB::VarMaybeExt,
) {
    if input.len() == 2 {
        let carry = add_n_carries::<AB>(input, result);
        for limb in carry {
            builder.when(is_real.clone()).assert_zero(limb.clone() * (AB::one_maybe() - limb));
        }
    }
}

pub fn add_n_without_result_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    n: usize,
) {
    if n > 2 {
        slice_u8_range_lookup(builder, is_real, 1);
    }
}

pub fn add_n_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    input: &[CompactWord<AB::VarMaybeExt>],
    result: CompactWord<AB::VarMaybeExt>,
) {
    add_n_without_result_precompute_lc(builder, input, result.clone());
    slice_u16_range_precompute_lc(builder, &result.0);
}

pub fn add_n_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    n: usize,
) {
    add_n_without_result_lookup(builder, is_real.clone(), n);
    slice_u16_range_lookup(builder, is_real, 2);
}
