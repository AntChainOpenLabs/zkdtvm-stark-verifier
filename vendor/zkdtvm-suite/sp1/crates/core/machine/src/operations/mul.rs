use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use dt_derive::AlignedBorrow;
use dt_primitives::consts::{BYTE_SIZE, LONG_WORD_SIZE, WORD_SIZE};
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};

use crate::air::WordAirBuilder;

/// The mask for a byte.
const BYTE_MASK: u8 = 0xff;
/// Get the most significant bit of a 32-bit integer.
pub const fn get_msb(a: [u8; 4]) -> u8 {
    ((a[3] >> (BYTE_SIZE - 1)) & 1) as u8
}
/// A set of columns needed to compute the add of two words.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulOperation<T> {
    /// Trace
    pub carry: [T; LONG_WORD_SIZE],
    /// bytes of product of 'b * c'
    pub product: [T; LONG_WORD_SIZE],
    /// sign of 'b'
    pub b_msb: T,
    /// sign of 'c'
    pub c_msb: T,
    /// sign extension of 'b'
    pub b_sign_extend: T,
    /// sign extension of 'c'
    pub c_sign_extend: T,
}

impl<F: Field> MulOperation<F> {
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        b_u32: u32,
        c_u32: u32,
        is_mulh: bool,
        is_mulhsu: bool,
    ) {
        let b_bytes = b_u32.to_le_bytes(); // [u8; 4]
        let c_bytes = c_u32.to_le_bytes();

        let mut b_extended = b_bytes.to_vec();
        let mut c_extended = c_bytes.to_vec();

        // msb
        let b_msb = get_msb(b_bytes);
        let c_msb = get_msb(c_bytes);
        self.b_msb = F::from_canonical_u8(b_msb);
        self.c_msb = F::from_canonical_u8(c_msb);

        //extension: mulh/mulhsu && b negative
        if (is_mulh || is_mulhsu) && b_msb == 1 {
            self.b_sign_extend = F::one();
            b_extended.resize(BYTE_SIZE, BYTE_MASK);
        } else {
            self.b_sign_extend = F::zero();
            b_extended.resize(BYTE_SIZE, 0);
        }

        //mulh && c negative
        if is_mulh && c_msb == 1 {
            self.c_sign_extend = F::one();
            c_extended.resize(8, BYTE_MASK);
        } else {
            self.c_sign_extend = F::zero();
            c_extended.resize(8, 0);
        }

        // msb constraint
        {
            let words = [b_bytes, c_bytes];
            let mut blu_events: Vec<ByteLookupEvent> = vec![];
            for word in words.iter() {
                let most_significant_byte = word[WORD_SIZE - 1];
                blu_events.push(ByteLookupEvent {
                    opcode: ByteOpcode::MSB,
                    a1: get_msb(*word) as u16,
                    a2: 0,
                    b: most_significant_byte,
                    c: 0,
                });
            }
            record.add_byte_lookup_events(blu_events);
        }

        // mul operation
        let mut product_sum = [0u32; 8];
        for i in 0..4 {
            for j in 0..4 {
                product_sum[i + j] += (b_extended[i] as u32) * (c_extended[j] as u32);
            }
        }

        for i in 0..8 {
            for j in 0..8 {
                if (i >= 4 || j >= 4) && i + j < 8 {
                    product_sum[i + j] += (b_extended[i] as u32) * (c_extended[j] as u32);
                }
            }
        }

        let mut carry = [0u32; 8];
        let mut final_product = [0u32; 8];
        for i in 0..8 {
            let current_sum = product_sum[i] + if i > 0 { carry[i - 1] } else { 0 };
            final_product[i] = current_sum % 256;
            carry[i] = current_sum / 256;

            self.product[i] = F::from_canonical_u32(final_product[i]);
            self.carry[i] = F::from_canonical_u32(carry[i]);
        }

        // 范围检查
        record.add_u8_range_checks(&final_product.map(|x| x as u8));
        record.add_u16_range_checks(&carry.map(|x| x as u16));
    }
    pub fn eval_self<AB: DTAirBuilder>(
        builder: &mut AB,
        b_word: Word<AB::Expr>,
        c_word: Word<AB::Expr>,
        cols: MulOperation<AB::Var>,
        is_real: AB::Expr,
        is_b_signed: AB::Expr,
        is_c_signed: AB::Expr,
    ) {
        let zero: AB::Expr = AB::F::zero().into();
        let base = AB::F::from_canonical_u32(256);
        let byte_mask = AB::F::from_canonical_u8(0xff);
        let msb_opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);

        // msb
        {
            builder.send_byte(
                msb_opcode,
                cols.b_msb,
                b_word[3].clone(),
                zero.clone(),
                is_real.clone(),
            );

            builder.send_byte(
                msb_opcode,
                cols.c_msb,
                c_word[3].clone(),
                zero.clone(),
                is_real.clone(),
            );
        }

        {
            builder.assert_eq(cols.b_sign_extend, is_b_signed.clone() * cols.b_msb);
            builder.assert_eq(cols.c_sign_extend, is_c_signed.clone() * cols.c_msb);
        }

        // b,c extend
        let mut b_ext = vec![AB::Expr::zero(); 8];
        let mut c_ext = vec![AB::Expr::zero(); 8];
        for i in 0..8 {
            if i < 4 {
                b_ext[i] = b_word[i].clone();
                c_ext[i] = c_word[i].clone();
            } else {
                //constant does not increase constraint degree
                b_ext[i] = cols.b_sign_extend * byte_mask;
                c_ext[i] = cols.c_sign_extend * byte_mask;
            }
        }
        // m[k] = sum_{i+j=k} b_ext[i] * c_ext[j]
        let mut m = vec![AB::Expr::zero(); 8];
        for i in 0..8 {
            for j in 0..8 {
                if i + j < 8 {
                    m[i + j] = m[i + j].clone() + b_ext[i].clone() * c_ext[j].clone();
                }
            }
        }

        for i in 0..8 {
            if i == 0 {
                builder
                    .when(is_real.clone())
                    .assert_eq(cols.product[i], m[i].clone() - cols.carry[i] * base);
            } else {
                builder.when(is_real.clone()).assert_eq(
                    cols.product[i],
                    m[i].clone() + cols.carry[i - 1] - cols.carry[i] * base,
                );
            }
        }

        //other constraints
        {
            builder.assert_bool(cols.b_msb);
            builder.assert_bool(cols.c_msb);
            builder.assert_bool(cols.b_sign_extend);
            builder.assert_bool(cols.c_sign_extend);

            builder.assert_bool(is_b_signed);
            builder.assert_bool(is_c_signed);

            builder.slice_range_check_u8(&cols.product, is_real.clone());
            builder.slice_range_check_u16(&cols.carry, is_real);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder>(
        builder: &mut AB,
        a_word: Word<AB::Expr>,
        b_word: Word<AB::Expr>,
        c_word: Word<AB::Expr>,
        cols: MulOperation<AB::Var>,
        is_real: AB::Expr,
        is_mul: AB::Expr,
        is_mulh: AB::Expr,
        is_mulhu: AB::Expr,
        is_mulhsu: AB::Expr,
    ) {
        let zero: AB::Expr = AB::F::zero().into();
        let base = AB::F::from_canonical_u32(256);
        let byte_mask = AB::F::from_canonical_u8(0xff);
        let msb_opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);

        // msb
        {
            builder.send_byte(
                msb_opcode,
                cols.b_msb,
                b_word[3].clone(),
                zero.clone(),
                is_real.clone(),
            );

            builder.send_byte(
                msb_opcode,
                cols.c_msb,
                c_word[3].clone(),
                zero.clone(),
                is_real.clone(),
            );
        }

        // MULH:   signed * signed
        // MULHU:  unsigned * unsigned
        // MULHSU: signed * unsigned
        {
            let is_b_signed = is_mulh.clone() + is_mulhsu.clone();
            let is_c_signed = is_mulh.clone();

            builder.assert_eq(cols.b_sign_extend, is_b_signed * cols.b_msb);
            builder.assert_eq(cols.c_sign_extend, is_c_signed * cols.c_msb);
        }

        // b,c extend
        let mut b_ext = vec![AB::Expr::zero(); 8];
        let mut c_ext = vec![AB::Expr::zero(); 8];
        for i in 0..8 {
            if i < 4 {
                b_ext[i] = b_word[i].clone();
                c_ext[i] = c_word[i].clone();
            } else {
                //constant does not increase constraint degree
                b_ext[i] = cols.b_sign_extend * byte_mask;
                c_ext[i] = cols.c_sign_extend * byte_mask;
            }
        }
        // m[k] = sum_{i+j=k} b_ext[i] * c_ext[j]
        let mut m = vec![AB::Expr::zero(); 8];
        for i in 0..8 {
            for j in 0..8 {
                if i + j < 8 {
                    m[i + j] = m[i + j].clone() + b_ext[i].clone() * c_ext[j].clone();
                }
            }
        }

        for i in 0..8 {
            if i == 0 {
                builder
                    .when(is_real.clone())
                    .assert_eq(cols.product[i], m[i].clone() - cols.carry[i] * base);
            } else {
                builder.when(is_real.clone()).assert_eq(
                    cols.product[i],
                    m[i].clone() + cols.carry[i - 1] - cols.carry[i] * base,
                );
            }
        }

        {
            // Mul / MULH* result must match `a_word` only when we write a real destination
            // (same as `perform_calc` in MulChip: skip tying op_a when rd is x0).
            //mul
            builder.when(is_mul.clone()).when(is_real.clone()).assert_word_eq(
                a_word.clone(),
                Word([
                    cols.product[0].into(),
                    cols.product[1].into(),
                    cols.product[2].into(),
                    cols.product[3].into(),
                ]),
            );

            // MULH / MULHU / MULHSU
            let is_upper = is_mulh.clone() + is_mulhu.clone() + is_mulhsu.clone();
            builder.when(is_upper).when(is_real.clone()).assert_word_eq(
                a_word,
                Word([
                    cols.product[4].into(),
                    cols.product[5].into(),
                    cols.product[6].into(),
                    cols.product[7].into(),
                ]),
            );
        }
        //other constraints
        {
            builder.assert_bool(cols.b_msb);
            builder.assert_bool(cols.c_msb);
            builder.assert_bool(cols.b_sign_extend);
            builder.assert_bool(cols.c_sign_extend);

            let sum_flags = is_mul + is_mulh + is_mulhu + is_mulhsu;
            builder.assert_bool(sum_flags.clone());

            builder.slice_range_check_u8(&cols.product, is_real.clone());
            builder.slice_range_check_u16(&cols.carry, is_real);
        }
    }
}

pub(crate) const MUL_OP_NUM_INTERACTIONS: usize = 14;

#[allow(clippy::too_many_arguments)]
pub fn mul_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    b_word: [AB::VarMaybeExt; 4],
    c_word: [AB::VarMaybeExt; 4],
    product: [AB::VarMaybeExt; 8],
    carry: [AB::VarMaybeExt; 8],
    b_msb: AB::VarMaybeExt,
    c_msb: AB::VarMaybeExt,
    b_sign_extend: AB::VarMaybeExt,
    c_sign_extend: AB::VarMaybeExt,
    is_b_signed: AB::VarMaybeExt,
    is_c_signed: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
    let byte_mask = AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xff));
    builder.assert_eq(b_sign_extend.clone(), is_b_signed * b_msb);
    builder.assert_eq(c_sign_extend.clone(), is_c_signed * c_msb);
    let mut b_ext: [AB::VarMaybeExt; 8] = core::array::from_fn(|_| AB::zero_maybe());
    let mut c_ext: [AB::VarMaybeExt; 8] = core::array::from_fn(|_| AB::zero_maybe());
    for i in 0..8 {
        if i < 4 {
            b_ext[i] = b_word[i].clone();
            c_ext[i] = c_word[i].clone();
        } else {
            b_ext[i] = b_sign_extend.clone() * byte_mask.clone();
            c_ext[i] = c_sign_extend.clone() * byte_mask.clone();
        }
    }
    let mut m: [AB::VarMaybeExt; 8] = core::array::from_fn(|_| AB::zero_maybe());
    for i in 0..8 {
        for j in 0..8 {
            if i + j < 8 {
                m[i + j] = m[i + j].clone() + b_ext[i].clone() * c_ext[j].clone();
            }
        }
    }
    for i in 0..8 {
        let expected = if i == 0 {
            m[i].clone() - carry[i].clone() * base.clone()
        } else {
            m[i].clone() + carry[i - 1].clone() - carry[i].clone() * base.clone()
        };
        builder.when(is_real.clone()).assert_zero(product[i].clone() - expected);
    }
}

pub fn mul_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    b_msb: AB::VarMaybeExt,
    c_msb: AB::VarMaybeExt,
    b_word_msb_byte: AB::VarMaybeExt,
    c_word_msb_byte: AB::VarMaybeExt,
    product: &[AB::VarMaybeExt; 8],
    carry: &[AB::VarMaybeExt; 8],
) {
    use crate::bytes::polyair::{
        msb_precompute_lc, slice_u16_range_precompute_lc, slice_u8_range_precompute_lc,
    };
    msb_precompute_lc(builder, b_msb, b_word_msb_byte);
    msb_precompute_lc(builder, c_msb, c_word_msb_byte);
    slice_u8_range_precompute_lc(builder, product);
    slice_u16_range_precompute_lc(builder, carry);
}

pub fn mul_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    for _ in 0..MUL_OP_NUM_INTERACTIONS {
        builder.send(is_real.clone());
    }
}
