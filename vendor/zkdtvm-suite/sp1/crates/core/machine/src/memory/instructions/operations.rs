use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use dt_derive::AlignedBorrow;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{air::FullAirBuilder, Word};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use crate::{
    air::DTCoreAirBuilder,
    operations::{AddOperation, BabyBearWordRangeChecker, IsZeroOperation},
};
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddressOperation<T> {
    pub addr_word: AddOperation<T>,
    pub addr_range_checker: BabyBearWordRangeChecker<T>,
    ///addr % 4
    pub offset_bit: [T; WORD_SIZE],
    pub addr_ls_two_bits: T,
    /// aligned address
    pub aligned_address: T,
    /// This is used to check if the most significant three bytes of the memory address are all
    /// zero.
    pub most_sig_bytes_zero: IsZeroOperation<T>,
}

impl<F: Field> AddressOperation<F> {
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        addr_base: u32,
        addr_offset: u32,
    ) -> u32 {
        let addr_effect = self.addr_word.populate(record, addr_base, addr_offset);
        self.addr_range_checker.populate(self.addr_word.value, record);
        let ls_two_bits = addr_effect & 0b11;
        self.addr_ls_two_bits = F::from_canonical_u32(ls_two_bits);
        self.offset_bit = [F::zero(); WORD_SIZE];
        for i in 0..WORD_SIZE {
            self.offset_bit[i] = F::from_bool(ls_two_bits as usize == i);
        }
        let addr_aligned = addr_effect & (!0b11);
        self.aligned_address = F::from_canonical_u32(addr_aligned);
        let addr_bytes = addr_effect.to_le_bytes();
        let most_sig_bytes_sum = addr_bytes[3] as u32 + addr_bytes[1] as u32 + addr_bytes[2] as u32;
        self.most_sig_bytes_zero.populate(most_sig_bytes_sum);
        //LTU bytecode
        if most_sig_bytes_sum == 0 {
            record.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: 31,
                c: addr_bytes[0],
            });
        }
        //slice range check for addr_word,already check in addoperation
        // record.add_u8_range_checks(&addr_bytes);
        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: ls_two_bits as u16,
            a2: 0,
            b: addr_bytes[0],
            c: 0b11,
        });
        addr_effect
    }
    pub fn eval<AB: DTCoreAirBuilder>(
        builder: &mut AB,
        addr_base: Word<AB::Var>,
        addr_offset: Word<AB::Var>,
        cols: AddressOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        //bool checks
        {
            for i in 0..WORD_SIZE {
                builder.assert_bool(cols.offset_bit[i]);
            }
            builder.assert_eq(
                is_real.clone(),
                cols.offset_bit.iter().fold(AB::Expr::zero(), |acc, bit| acc + *bit),
            );
            builder.assert_bool(is_real.clone());
        }

        //address add
        AddOperation::<AB::F>::eval(
            builder,
            addr_base,
            addr_offset,
            cols.addr_word,
            is_real.clone(),
        );
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            cols.addr_word.value,
            cols.addr_range_checker,
            is_real.clone(),
        );
        //ls 2 bits constraints: ls_two_bits -> offset_bit
        {
            builder.assert_eq(
                is_real.clone(),
                cols.offset_bit.iter().fold(AB::Expr::zero(), |acc, x| acc + *x), // cols.offset_bit[0] + cols.offset_bit[1] + cols.offset_bit[2] + cols.offset_bit[3],
            );
            builder.when(cols.offset_bit[0]).assert_zero(cols.addr_ls_two_bits);
            builder.when(cols.offset_bit[1]).assert_one(cols.addr_ls_two_bits);
            builder
                .when(cols.offset_bit[2])
                .assert_eq(cols.addr_ls_two_bits, AB::Expr::from_canonical_u8(2));
            builder
                .when(cols.offset_bit[3])
                .assert_eq(cols.addr_ls_two_bits, AB::Expr::from_canonical_u8(3));
        }
        //addr_word -> ls_two_bits
        {
            builder.send_byte(
                AB::F::from_canonical_u8(ByteOpcode::AND as u8),
                cols.addr_ls_two_bits,
                cols.addr_word.value[0],
                AB::F::from_canonical_u8(0b11),
                is_real.clone(),
            );
        }
        //addr_word + ls_two_bits -> addr_aligned
        builder.assert_eq(
            cols.aligned_address + cols.addr_ls_two_bits,
            cols.addr_word.value.reduce::<AB>(),
        );
        // if most_sig_bytes_zero is true, addr_word is bigger than 31
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::LTU as u8),
            AB::Expr::one(),
            AB::F::from_canonical_u8(31),
            cols.addr_word.value[0],
            cols.most_sig_bytes_zero.result,
        );
        // most_sig_bytes_zero correctness
        IsZeroOperation::<AB::F>::eval(
            builder,
            cols.addr_word.value[1].into() +
                cols.addr_word.value[2].into() +
                cols.addr_word.value[3].into(),
            cols.most_sig_bytes_zero,
            is_real.clone(),
        );
    }
}

// ============================================================================
// PolyAir three-phase helpers for AddressOperation
// ============================================================================

pub(crate) const ADDRESS_OP_NUM_INTERACTIONS: usize = 5;

#[allow(clippy::too_many_arguments)]
pub fn address_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    addr_base: [AB::VarMaybeExt; 4],
    addr_offset: [AB::VarMaybeExt; 4],
    addr_word_value: [AB::VarMaybeExt; 4],
    most_sig_byte_lt_threshold: AB::VarMaybeExt,
    offset_bit: [AB::VarMaybeExt; 4],
    addr_ls_two_bits: AB::VarMaybeExt,
    aligned_address: AB::VarMaybeExt,
    most_sig_bytes_zero_inverse: AB::VarMaybeExt,
    most_sig_bytes_zero_result: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    use crate::operations::{
        add::add_op_gate_constraints, baby_bear_word::baby_bear_range_check_gate_constraints,
        is_zero::is_zero_op_gate_constraints,
    };

    add_op_gate_constraints(
        builder,
        addr_base,
        addr_offset,
        addr_word_value.clone(),
        is_real.clone(),
    );

    baby_bear_range_check_gate_constraints(
        builder,
        addr_word_value.clone(),
        most_sig_byte_lt_threshold,
        is_real.clone(),
    );

    builder.assert_zero(
        is_real.clone() -
            offset_bit[0].clone() -
            offset_bit[1].clone() -
            offset_bit[2].clone() -
            offset_bit[3].clone(),
    );

    builder.when(offset_bit[0].clone()).assert_zero(addr_ls_two_bits.clone());
    builder.when(offset_bit[1].clone()).assert_one(addr_ls_two_bits.clone());
    builder
        .when(offset_bit[2].clone())
        .assert_eq(addr_ls_two_bits.clone(), AB::VarMaybeExt::from(AB::F::from_canonical_u8(2)));
    builder
        .when(offset_bit[3].clone())
        .assert_eq(addr_ls_two_bits.clone(), AB::VarMaybeExt::from(AB::F::from_canonical_u8(3)));

    let base_w = |i: u32| AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << (8 * i)));
    let addr_reduced = addr_word_value[0].clone() * base_w(0) +
        addr_word_value[1].clone() * base_w(1) +
        addr_word_value[2].clone() * base_w(2) +
        addr_word_value[3].clone() * base_w(3);
    builder.assert_zero(aligned_address + addr_ls_two_bits - addr_reduced);

    let ms_sum =
        addr_word_value[1].clone() + addr_word_value[2].clone() + addr_word_value[3].clone();
    is_zero_op_gate_constraints(
        builder,
        ms_sum,
        most_sig_bytes_zero_inverse,
        most_sig_bytes_zero_result,
        is_real,
    );
}

pub fn address_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    addr_word_value: &dt_stark::Word<AB::VarMaybeExt>,
    most_sig_byte_lt_threshold: AB::VarMaybeExt,
    addr_ls_two_bits: AB::VarMaybeExt,
) {
    use crate::operations::{
        add::add_op_precompute_lc, baby_bear_word::baby_bear_range_check_precompute_lc,
    };
    use dt_core_executor::ByteOpcode;
    use dt_stark::InteractionKind;

    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let and_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::AND as u8));
    let ltu_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::LTU as u8));
    let one = AB::VarMaybeExt::from(AB::F::one());
    let mask = AB::VarMaybeExt::from(AB::F::from_canonical_u8(0b11));

    add_op_precompute_lc(builder, addr_word_value);

    baby_bear_range_check_precompute_lc(
        builder,
        addr_word_value[3].clone(),
        most_sig_byte_lt_threshold,
    );

    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind.clone(),
        vec![and_opcode, addr_ls_two_bits, zero.clone(), addr_word_value[0].clone(), mask],
    ));

    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![
            ltu_opcode,
            one,
            zero.clone(),
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(31)),
            addr_word_value[0].clone(),
        ],
    ));
}

pub fn address_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    most_sig_bytes_zero_result: AB::VarMaybeExt,
) {
    use crate::operations::{add::add_op_lookup, baby_bear_word::baby_bear_range_check_lookup};

    add_op_lookup(builder, is_real.clone());
    baby_bear_range_check_lookup(builder, is_real.clone());
    builder.send(is_real);
    builder.send(most_sig_bytes_zero_result);
}
