use dt_core_executor::{events::ByteRecord, ByteOpcode};
use dt_derive::AlignedBorrow;
use dt_stark::air::DTAirBuilder;
use p3_field::{AbstractField, Field};

use crate::operations_dt::CompactWord;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct FixedShiftRightOperation<T> {
    /// The output value.
    // pub value: CompactWord<T>,

    /// The higher bits.
    pub higher_bits: [T; 2],
}

impl<F: Field> FixedShiftRightOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, input: u32, rotation: usize) -> u32 {
        let result = input >> rotation;
        // self.value = result.into();

        let num_bytes_to_rotate = rotation / 16;
        let num_bits_to_rotate = rotation % 16;

        let input = [input & 0xFFFFu32, input >> 16];
        let input: [_; 2] = std::array::from_fn(|i| {
            if num_bytes_to_rotate + i < 2 {
                input[num_bytes_to_rotate + i]
            } else {
                0
            }
        });

        let lower_mask = (1u32 << num_bits_to_rotate) - 1;
        for i in 0..2 {
            let lower_bits = input[i] & lower_mask;
            let higher_bits = input[i] >> num_bits_to_rotate;

            self.higher_bits[i] = F::from_canonical_u32(higher_bits);

            if num_bits_to_rotate >= 2 {
                record.add_bit_range_check(lower_bits as u16, num_bits_to_rotate as u8);
            }
            if num_bits_to_rotate <= 14 {
                record.add_bit_range_check(higher_bits as u16, (16 - num_bits_to_rotate) as u8);
            }
        }

        result
    }

    /// Evaluates the u32 fixed rotate right.
    /// If `is_real` is true, the result `value` will be the correct result with two u16 limbs.
    /// This function assumes that the `input` is a u32 with valid two u16 limbs.
    pub fn eval<AB: DTAirBuilder<F = F>>(
        cols: &FixedShiftRightOperation<AB::Var>,
        builder: &mut AB,
        input: CompactWord<impl Into<AB::Expr>>,
        rotation: usize,
        is_real: impl Into<AB::Expr>,
    ) -> CompactWord<AB::Expr> {
        let num_bytes_to_rotate = rotation / 16;
        let num_bits_to_rotate = rotation % 16;

        let input = input.0.map(|input| input.into());
        let is_real = is_real.into();

        let multiplier = AB::F::from_canonical_u32(1u32 << num_bits_to_rotate);
        let lower_bits: [AB::Expr; 2] = std::array::from_fn(|i| {
            let lower_bits = if num_bytes_to_rotate + i < 2 {
                input[num_bytes_to_rotate + i].clone()
            } else {
                AB::Expr::zero()
            } - cols.higher_bits[i] * multiplier;

            if num_bits_to_rotate == 0 {
                builder.assert_zero(lower_bits.clone());
            } else if num_bits_to_rotate == 1 {
                builder.assert_bool(lower_bits.clone());
            } else {
                builder.send_byte(
                    AB::F::from_canonical_u32(ByteOpcode::BitRange as u32),
                    lower_bits.clone(),
                    AB::F::from_canonical_u32(num_bits_to_rotate as u32),
                    AB::F::zero(),
                    is_real.clone(),
                );
            }

            // TODO: deal with this
            if num_bits_to_rotate == 16 {
                builder.assert_zero(cols.higher_bits[i]);
            } else if num_bits_to_rotate == 15 {
                builder.assert_bool(cols.higher_bits[i]);
            } else if num_bits_to_rotate == 0 {
                builder.send_byte(
                    AB::F::from_canonical_u32(ByteOpcode::U16Range as u32),
                    cols.higher_bits[i],
                    AB::F::zero(),
                    AB::F::zero(),
                    is_real.clone(),
                );
            } else {
                builder.send_byte(
                    AB::F::from_canonical_u32(ByteOpcode::BitRange as u32),
                    cols.higher_bits[i],
                    AB::F::from_canonical_u32((16 - num_bits_to_rotate) as u32),
                    AB::F::zero(),
                    is_real.clone(),
                );
            }

            lower_bits
        });

        let multiplier = AB::F::from_canonical_u32(1u32 << (16 - num_bits_to_rotate));
        // builder.when(is_real).assert_eq(cols.value[1], cols.higher_bits[1].into());
        // builder.when(is_real).assert_eq(
        //     cols.value[0],
        //     cols.higher_bits[0].into() + lower_bits[1].clone() * multiplier,
        // );

        CompactWord([
            cols.higher_bits[0].into() + lower_bits[1].clone() * multiplier,
            cols.higher_bits[1].into(),
        ])
    }
}

// ============================================================================
// PolyAir three-phase helpers for FixedShiftRightOperation
// ============================================================================

use crate::bytes::polyair::{
    bit_range_lookup, bit_range_precompute_lc, slice_u16_range_lookup, u16_range_precompute_lc,
};

#[cfg(feature = "koalabear")]
pub const fn fixed_shift_right_num_interactions(rotation: usize) -> usize {
    let num_bits = rotation % 16;
    let per_limb = if num_bits == 0 || num_bits == 1 || num_bits == 15 { 1 } else { 2 };
    per_limb * 2
}

pub fn fixed_shift_right_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    cols: &FixedShiftRightOperation<AB::VarMaybeExt>,
    input: CompactWord<AB::VarMaybeExt>,
    rotation: usize,
) {
    let num_bytes_to_rotate = rotation / 16;
    let num_bits_to_rotate = rotation % 16;
    let multiplier = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << num_bits_to_rotate));

    for i in 0..2 {
        let input_limb = if num_bytes_to_rotate + i < 2 {
            input[num_bytes_to_rotate + i].clone()
        } else {
            AB::zero_maybe()
        };
        let lower_bits = input_limb - cols.higher_bits[i].clone() * multiplier.clone();

        if num_bits_to_rotate >= 2 {
            bit_range_precompute_lc(
                builder,
                lower_bits,
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(num_bits_to_rotate as u32)),
            );
        }

        if num_bits_to_rotate == 0 {
            u16_range_precompute_lc(builder, cols.higher_bits[i].clone());
        } else if num_bits_to_rotate < 15 {
            bit_range_precompute_lc(
                builder,
                cols.higher_bits[i].clone(),
                AB::VarMaybeExt::from(AB::F::from_canonical_u32((16 - num_bits_to_rotate) as u32)),
            );
        }
    }
}

pub fn fixed_shift_right_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    cols: &FixedShiftRightOperation<AB::VarMaybeExt>,
    input: CompactWord<AB::VarMaybeExt>,
    rotation: usize,
) {
    let num_bytes_to_rotate = rotation / 16;
    let num_bits_to_rotate = rotation % 16;
    let multiplier = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << num_bits_to_rotate));

    for i in 0..2 {
        let input_limb = if num_bytes_to_rotate + i < 2 {
            input[num_bytes_to_rotate + i].clone()
        } else {
            AB::zero_maybe()
        };
        let lower_bits = input_limb - cols.higher_bits[i].clone() * multiplier.clone();

        if num_bits_to_rotate == 0 {
            builder.assert_zero(lower_bits);
        } else if num_bits_to_rotate == 1 {
            builder.assert_zero(lower_bits.clone() * (AB::one_maybe() - lower_bits));
        }

        if num_bits_to_rotate == 15 {
            let higher = cols.higher_bits[i].clone();
            builder.assert_zero(higher.clone() * (AB::one_maybe() - higher));
        }
    }
}

pub fn fixed_shift_right_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    rotation: usize,
) {
    let num_bits_to_rotate = rotation % 16;
    for _ in 0..2 {
        if num_bits_to_rotate >= 2 {
            bit_range_lookup(builder, is_real.clone());
        }

        if num_bits_to_rotate == 0 {
            slice_u16_range_lookup(builder, is_real.clone(), 1);
        } else if num_bits_to_rotate < 15 {
            bit_range_lookup(builder, is_real.clone());
        }
    }
}
