use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{BaseAirBuilder, DTAirBuilder, FullAirBuilder},
    Word,
};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

const FIELD_MS_BYTE_THRESHOLD: u8 = if cfg!(feature = "koalabear") { 127 } else { 120 };

/// A set of columns needed to range check a BabyBear word.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BabyBearWordRangeChecker<T> {
    /// Most sig byte is less than FIELD_MS_BYTE_THRESHOLD.
    pub most_sig_byte_lt_120: T,
}

impl<F: Field> BabyBearWordRangeChecker<F> {
    pub fn populate(&mut self, value: Word<F>, record: &mut impl ByteRecord) {
        let ms_byte_u8 = value[3].as_u32() as u8;
        self.most_sig_byte_lt_120 = F::from_bool(ms_byte_u8 < FIELD_MS_BYTE_THRESHOLD);

        // Add the byte lookup for the range check bit.
        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: if ms_byte_u8 < FIELD_MS_BYTE_THRESHOLD { 1 } else { 0 },
            a2: 0,
            b: ms_byte_u8,
            c: FIELD_MS_BYTE_THRESHOLD,
        });
    }
}

impl<F: Field> BabyBearWordRangeChecker<F> {
    pub fn range_check<AB: DTAirBuilder>(
        builder: &mut AB,
        value: Word<AB::Var>,
        cols: BabyBearWordRangeChecker<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Range check that value is less than baby bear modulus.  To do this, it is sufficient
        // to just do comparisons for the most significant byte. BabyBear's modulus is (in big
        // endian binary) 01111000_00000000_00000000_00000001.  So we need to check the
        // following conditions:
        // 1) if most_sig_byte > 01111000 (or 120 in decimal), then fail.
        // 2) if most_sig_byte == 01111000, then value's lower sig bytes must all be 0.
        // 3) if most_sig_byte < 01111000, then pass.

        let ms_byte = value[3];

        // The range check bit is on if and only if the most significant byte of the word is < 120.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            cols.most_sig_byte_lt_120,
            ms_byte,
            AB::Expr::from_canonical_u8(FIELD_MS_BYTE_THRESHOLD),
            is_real.clone(),
        );

        let mut is_real_builder = builder.when(is_real.clone());

        // If the range check bit is off, the most significant byte is >= threshold, so to be a
        // valid field word we need the most significant byte to be = threshold.
        is_real_builder
            .when_not(cols.most_sig_byte_lt_120)
            .assert_eq(ms_byte, AB::Expr::from_canonical_u8(FIELD_MS_BYTE_THRESHOLD));

        // Moreover, if the most significant byte =120, then the 3 other bytes must all be zero.s
        let mut assert_zero_builder = is_real_builder.when_not(cols.most_sig_byte_lt_120);
        assert_zero_builder.assert_zero(value[0]);
        assert_zero_builder.assert_zero(value[1]);
        assert_zero_builder.assert_zero(value[2]);
    }
}

pub fn baby_bear_range_check_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    value: [AB::VarMaybeExt; 4],
    most_sig_byte_lt_threshold: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    let threshold = AB::VarMaybeExt::from(AB::F::from_canonical_u8(FIELD_MS_BYTE_THRESHOLD));
    let guard = is_real.clone() * (AB::one_maybe() - most_sig_byte_lt_threshold);
    builder.when(guard.clone()).assert_zero(value[3].clone() - threshold);
    for i in 0..3 {
        builder.when(guard.clone()).assert_zero(value[i].clone());
    }
}

pub fn baby_bear_range_check_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    ms_byte: AB::VarMaybeExt,
    most_sig_byte_lt_threshold: AB::VarMaybeExt,
) {
    use dt_core_executor::ByteOpcode;
    use dt_stark::InteractionKind;

    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let ltu_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::LTU as u8));
    let threshold = AB::VarMaybeExt::from(AB::F::from_canonical_u8(FIELD_MS_BYTE_THRESHOLD));
    let zero = AB::zero_maybe();

    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![ltu_opcode, most_sig_byte_lt_threshold, zero, ms_byte, threshold],
    ));
}

pub fn baby_bear_range_check_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    builder.send(is_real);
}
