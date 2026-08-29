use itertools::izip;

use dt_stark::air::FullAirBuilder;
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use crate::{air::DTCoreAirBuilder, operations::mul::get_msb};
use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use dt_derive::AlignedBorrow;
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{air::DTAirBuilder, Word};
/// Operation columns for verifying that an element is within the range `[0, modulus)`.
#[derive(Debug, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct AssertLtColsBytes<T, const N: usize> {
    /// Boolean flags to indicate the first byte in which the element is smaller than the modulus.
    pub byte_flags: [T; N],

    pub a_comparison_byte: T,
    pub b_comparison_byte: T,
}

impl<T: Default, const N: usize> Default for AssertLtColsBytes<T, N> {
    fn default() -> Self {
        Self {
            byte_flags: core::array::from_fn(|_| T::default()),
            a_comparison_byte: T::default(),
            b_comparison_byte: T::default(),
        }
    }
}

impl<F: Field, const N: usize> AssertLtColsBytes<F, N> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a: &[u8], b: &[u8]) {
        let mut byte_flags = vec![0u8; N];

        for (a_byte, b_byte, flag) in
            izip!(a.iter().rev(), b.iter().rev(), byte_flags.iter_mut().rev())
        {
            assert!(a_byte <= b_byte);
            if a_byte < b_byte {
                *flag = 1;
                self.a_comparison_byte = F::from_canonical_u8(*a_byte);
                self.b_comparison_byte = F::from_canonical_u8(*b_byte);
                record.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::LTU,
                    a1: 1,
                    a2: 0,
                    b: *a_byte,
                    c: *b_byte,
                });
                break;
            }
        }

        for (byte, flag) in izip!(byte_flags.iter(), self.byte_flags.iter_mut()) {
            *flag = F::from_canonical_u8(*byte);
        }
    }
}

impl<V: Copy, const N: usize> AssertLtColsBytes<V, N> {
    pub fn eval<AB: DTAirBuilder<Var = V>, Ea: Into<AB::Expr> + Clone, Eb: Into<AB::Expr> + Clone>(
        &self,
        builder: &mut AB,
        a: &[Ea],
        b: &[Eb],
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
    {
        // The byte flags give a specification of which byte is `first_eq`, i,e, the first most
        // significant byte for which the element `a` is smaller than `b`. To verify the
        // less-than claim we need to check that:
        // * For all bytes until `first_eq` the element `a` byte is equal to the `b` byte.
        // * For the `first_eq` byte the `a`` byte is smaller than the `b`byte.
        // * all byte flags are boolean.
        // * only one byte flag is set to one, and the rest are set to zero.

        // Check the flags are of valid form.

        // Verrify that only one flag is set to one.
        let mut sum_flags: AB::Expr = AB::Expr::zero();
        for &flag in self.byte_flags.iter() {
            // Assert that the flag is boolean.
            builder.assert_bool(flag);
            // Add the flag to the sum.
            sum_flags = sum_flags.clone() + flag.into();
        }
        // Assert that the sum is equal to one.
        builder.when(is_real.clone()).assert_one(sum_flags);

        // Check the less-than condition.

        // A flag to indicate whether an equality check is necessary (this is for all bytes from
        // most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::zero();

        // The bytes of the modulus.

        let a: [AB::Expr; N] = core::array::from_fn(|i| a[i].clone().into());
        let b: [AB::Expr; N] = core::array::from_fn(|i| b[i].clone().into());

        let mut first_lt_byte = AB::Expr::zero();
        let mut b_comparison_byte = AB::Expr::zero();
        for (a_byte, b_byte, &flag) in
            izip!(a.iter().rev(), b.iter().rev(), self.byte_flags.iter().rev())
        {
            // Once the byte flag was set to one, we turn off the quality check flag.
            // We can do this by calculating the sum of the flags since only `1` is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            first_lt_byte = first_lt_byte.clone() + a_byte.clone() * flag;
            b_comparison_byte = b_comparison_byte.clone() + b_byte.clone() * flag;

            builder
                .when_not(is_inequality_visited.clone())
                .when(is_real.clone())
                .assert_eq(a_byte.clone(), b_byte.clone());
        }

        builder.when(is_real.clone()).assert_eq(self.a_comparison_byte, first_lt_byte);
        builder.when(is_real.clone()).assert_eq(self.b_comparison_byte, b_comparison_byte);

        // Send the comparison interaction.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::F::one(),
            self.a_comparison_byte,
            self.b_comparison_byte,
            is_real,
        )
    }
}

/// Signed less-than operation columns.
#[derive(Debug, Default, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct LtOperationSigned<T> {
    /// result of SLTU operation
    pub result: LtOperationUnsigned<T>,
    /// most significant bit of b
    pub b_msb: T,
    /// most significant bit of c
    pub c_msb: T,
}

impl<F: Field> LtOperationSigned<F> {
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        a_u32: u32,
        b_u32: u32,
        c_u32: u32,
        is_signed: bool,
    ) {
        let b_comp = b_u32.to_le_bytes();
        let c_comp = c_u32.to_le_bytes();
        if is_signed {
            let mut blu_events: Vec<ByteLookupEvent> = Vec::new();
            let b_msb = get_msb(b_comp);
            blu_events.push(ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: b_msb as u16,
                a2: 0,
                b: b_comp[3],
                c: 0,
            });
            let c_msb = get_msb(c_comp);
            blu_events.push(ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: c_msb as u16,
                a2: 0,
                b: c_comp[3],
                c: 0,
            });

            record.add_byte_lookup_events(blu_events);

            self.b_msb = F::from_canonical_u8(b_msb);
            self.c_msb = F::from_canonical_u8(c_msb);
            self.result.populate(record, a_u32, b_u32 ^ (1 << 31), c_u32 ^ (1 << 31));
        } else {
            self.b_msb = F::zero();
            self.c_msb = F::zero();
            self.result.populate(record, a_u32, b_u32, c_u32);
        }
    }
    pub fn eval<AB>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: LtOperationSigned<AB::Var>,
        is_signed: AB::Expr,
        is_real: AB::Expr,
    ) where
        AB: DTCoreAirBuilder,
    {
        builder.assert_bool(is_signed.clone());
        builder.assert_bool(is_real.clone());
        // If `is_real` is false, assert that `is_signed` is zero.
        builder.when_not(is_real.clone()).assert_zero(is_signed.clone());

        // Constrain the MSB of `b` and `c` if `is_signed` is true.
        // This will be used to determine the sign of `b` and `c`.
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::MSB as u8),
            cols.b_msb,
            b.0[WORD_SIZE - 1].clone(),
            AB::F::zero(),
            is_signed.clone(),
        );
        builder.send_byte(
            AB::F::from_canonical_u8(ByteOpcode::MSB as u8),
            cols.c_msb,
            c.0[WORD_SIZE - 1].clone(),
            AB::F::zero(),
            is_signed.clone(),
        );

        // Constrain `b` and `c` to be considered positive if `is_signed` is false.
        builder.when_not(is_signed.clone()).assert_zero(cols.b_msb);
        builder.when_not(is_signed.clone()).assert_zero(cols.c_msb);

        let mut b_compare = b;
        let mut c_compare = c;

        let base = AB::Expr::from_canonical_u32(1 << 8);

        b_compare[WORD_SIZE - 1] = b_compare[WORD_SIZE - 1].clone() +
            is_signed.clone() * AB::Expr::from_canonical_u32(1 << 7) -
            base.clone() * cols.b_msb;
        c_compare[WORD_SIZE - 1] = c_compare[WORD_SIZE - 1].clone() +
            is_signed.clone() * AB::Expr::from_canonical_u32(1 << 7) -
            base.clone() * cols.c_msb;

        // Now apply the unsigned LT operation.
        LtOperationUnsigned::<AB::F>::eval(
            builder,
            b_compare,
            c_compare,
            cols.result,
            is_real.clone(),
        );
    }
    pub fn comparison_result<AB: DTCoreAirBuilder>(&self) -> impl Into<AB::Expr>
    where
        F: Into<AB::Expr>,
    {
        self.result.result
    }
}

/// Signed less-than operation columns
#[derive(Debug, Default, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct LtOperationUnsigned<T> {
    ///flags to indicate the first byte in which the element is smaller than the modulus
    pub byte_flags: [T; WORD_SIZE],
    /// comparison bytes of 'b' and 'c'
    pub comparison_bytes: [T; 2],
    /// An inverse of differing byte if b != c
    pub not_eq_inv: T,
    ///Lt comp result, if 1 b < c, if 0 b >= c
    pub result: T,
}
impl<F: Field> LtOperationUnsigned<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32, c_u32: u32) {
        self.comparison_bytes[0] = F::zero();
        self.comparison_bytes[1] = F::zero();
        self.not_eq_inv = F::zero();
        self.byte_flags = [F::zero(), F::zero(), F::zero(), F::zero()];

        let a_bytes = a_u32.to_le_bytes();
        let b_bytes = b_u32.to_le_bytes();
        let c_bytes = c_u32.to_le_bytes();

        let a_effect = a_bytes[0];

        let mut comparison_limbs = [0u8; 2];
        for (b_limb, c_limb, flag) in
            izip!(b_bytes.iter().rev(), c_bytes.iter().rev(), self.byte_flags.iter_mut().rev())
        {
            if b_limb != c_limb {
                *flag = F::one();
                comparison_limbs[0] = *b_limb;
                comparison_limbs[1] = *c_limb;
                let b_limb = F::from_canonical_u8(*b_limb);
                let c_limb = F::from_canonical_u8(*c_limb);
                self.not_eq_inv = (b_limb - c_limb).inverse();
                self.comparison_bytes = [b_limb, c_limb];
                break;
            }
        }
        self.result = F::from_canonical_u8(a_effect);
        assert_eq!(b_u32 < c_u32, a_effect != 0);
        let diff = comparison_limbs[0].wrapping_sub(comparison_limbs[1]);
        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: diff,
            c: 0,
        });
    }

    /// Evaluate that LT operation.
    /// Assumes that `b`, `c` are either valid `Word`s of u16 limbs.
    /// Constrains that `is_real` is boolean.
    /// If `is_real` is true, constrains that the result is the LT of `b` and `c`.
    pub fn eval<AB>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: LtOperationUnsigned<AB::Var>,
        is_real: AB::Expr,
    ) where
        AB: DTCoreAirBuilder,
    {
        builder.assert_bool(is_real.clone());

        // Verify that the limb equality flags are set correctly, i.e. all are boolean and only
        // at most a single flag is set to one.
        let sum_flags =
            cols.byte_flags[0] + cols.byte_flags[1] + cols.byte_flags[2] + cols.byte_flags[3];
        builder.assert_bool(cols.byte_flags[0]);
        builder.assert_bool(cols.byte_flags[1]);
        builder.assert_bool(cols.byte_flags[2]);
        builder.assert_bool(cols.byte_flags[3]);
        builder.assert_bool(sum_flags.clone());

        let is_comp_eq = AB::Expr::one() - sum_flags;

        // A flag to indicate whether an equality check is necessary.
        // This is for all limbs from most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::zero();

        // Iterate over the limbs in reverse order and select the differing limbs using the limb
        // flag columns values.
        let mut b_comparison_limb = AB::Expr::zero();
        let mut c_comparison_limb = AB::Expr::zero();
        for (b_limb, c_limb, &flag) in
            izip!(b.0.iter().rev(), c.0.iter().rev(), cols.byte_flags.iter().rev())
        {
            // Once the byte flag was set to one, we turn off the equality check flag.
            // We can do this by calculating the sum of the flags since only one is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            // If inequality is not visited, assert that the limbs are equal.
            builder
                .when(is_real.clone() - is_inequality_visited.clone())
                .assert_eq(b_limb.clone(), c_limb.clone());

            b_comparison_limb = b_comparison_limb.clone() + b_limb.clone() * flag.into();
            c_comparison_limb = c_comparison_limb.clone() + c_limb.clone() * flag.into();
        }

        let (b_comp_limb, c_comp_limb) = (cols.comparison_bytes[0], cols.comparison_bytes[1]);
        builder.assert_eq(b_comparison_limb, b_comp_limb);
        builder.assert_eq(c_comparison_limb, c_comp_limb);

        // Using the values above, we can constrain the `is_comp_eq` flag. We already asserted
        // in the loop that when `is_comp_eq == 1` then all limbs are equal. It is left to
        // verify that when `is_comp_eq == 0` the comparison limbs are indeed not equal.
        // This is done using the inverse hint `not_eq_inv`, when `is_real` is true.
        builder
            .when_not(is_comp_eq)
            .assert_eq(cols.not_eq_inv * (b_comp_limb - c_comp_limb), is_real.clone());

        // Compare the two comparison limbs.
        // result is boolean
        builder.assert_bool(cols.result);
        let base = AB::Expr::from_canonical_u32(256);
        let diff = b_comp_limb - c_comp_limb + cols.result * base;
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::F::zero(),
            diff,
            AB::F::zero(),
            is_real.clone(),
        );
    }
}

pub fn assert_lt_bytes_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a: [AB::VarMaybeExt; 4],
    b: [AB::VarMaybeExt; 4],
    byte_flags: [AB::VarMaybeExt; 4],
    a_comparison_byte: AB::VarMaybeExt,
    b_comparison_byte: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    let sum_flags = byte_flags[0].clone() +
        byte_flags[1].clone() +
        byte_flags[2].clone() +
        byte_flags[3].clone();
    builder.when(is_real.clone()).assert_zero(sum_flags - one.clone());
    let mut is_inequality_visited = AB::zero_maybe();
    let mut first_lt_byte = AB::zero_maybe();
    let mut b_comp_byte = AB::zero_maybe();
    for i in (0..4).rev() {
        is_inequality_visited = is_inequality_visited.clone() + byte_flags[i].clone();
        first_lt_byte = first_lt_byte.clone() + a[i].clone() * byte_flags[i].clone();
        b_comp_byte = b_comp_byte.clone() + b[i].clone() * byte_flags[i].clone();
        builder
            .when_ne(is_inequality_visited.clone(), one.clone())
            .when(is_real.clone())
            .assert_zero(a[i].clone() - b[i].clone());
    }
    builder.when(is_real.clone()).assert_zero(a_comparison_byte - first_lt_byte);
    builder.when(is_real).assert_zero(b_comparison_byte - b_comp_byte);
}

pub(crate) const LT_SIGNED_NUM_INTERACTIONS: usize = 3;

#[allow(clippy::too_many_arguments)]
pub fn lt_signed_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    b: [AB::VarMaybeExt; 4],
    c: [AB::VarMaybeExt; 4],
    b_msb: AB::VarMaybeExt,
    c_msb: AB::VarMaybeExt,
    byte_flags: [AB::VarMaybeExt; 4],
    comparison_bytes: [AB::VarMaybeExt; 2],
    not_eq_inv: AB::VarMaybeExt,
    result: AB::VarMaybeExt,
    is_signed: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    let zero = AB::zero_maybe();
    builder.when(one.clone() - is_signed.clone()).assert_zero(b_msb.clone());
    builder.when(one.clone() - is_signed.clone()).assert_zero(c_msb.clone());
    let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
    let sign_bit = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 7));
    let mut b_cmp = [b[0].clone(), b[1].clone(), b[2].clone(), b[3].clone()];
    let mut c_cmp = [c[0].clone(), c[1].clone(), c[2].clone(), c[3].clone()];
    b_cmp[3] = b[3].clone() + is_signed.clone() * sign_bit.clone() - base.clone() * b_msb;
    c_cmp[3] = c[3].clone() + is_signed * sign_bit - base * c_msb;
    let sum_flags = byte_flags[0].clone() +
        byte_flags[1].clone() +
        byte_flags[2].clone() +
        byte_flags[3].clone();
    let mut is_inequality_visited = zero.clone();
    let mut b_comparison_limb = zero.clone();
    let mut c_comparison_limb = zero;
    for i in (0..4).rev() {
        let flag = byte_flags[i].clone();
        is_inequality_visited = is_inequality_visited.clone() + flag.clone();
        builder
            .when(is_real.clone() - is_inequality_visited.clone())
            .assert_eq(b_cmp[i].clone(), c_cmp[i].clone());
        b_comparison_limb = b_comparison_limb + b_cmp[i].clone() * flag.clone();
        c_comparison_limb = c_comparison_limb + c_cmp[i].clone() * flag;
    }
    let b_comp = comparison_bytes[0].clone();
    let c_comp = comparison_bytes[1].clone();
    builder.assert_eq(b_comparison_limb, b_comp.clone());
    builder.assert_eq(c_comparison_limb, c_comp.clone());
    let is_comp_eq = one.clone() - sum_flags;
    builder.when(one - is_comp_eq).assert_eq(not_eq_inv * (b_comp - c_comp), is_real);
    let _ = result;
}

pub fn lt_signed_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    b_msb: AB::VarMaybeExt,
    c_msb: AB::VarMaybeExt,
    b_word_msb_byte: AB::VarMaybeExt,
    c_word_msb_byte: AB::VarMaybeExt,
    comparison_bytes: [AB::VarMaybeExt; 2],
    result: AB::VarMaybeExt,
) {
    use crate::bytes::polyair::{msb_precompute_lc, u8_range_pair_precompute_lc};
    msb_precompute_lc(builder, b_msb, b_word_msb_byte);
    msb_precompute_lc(builder, c_msb, c_word_msb_byte);
    let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
    let diff = comparison_bytes[0].clone() - comparison_bytes[1].clone() + result * base;
    u8_range_pair_precompute_lc(builder, diff, AB::zero_maybe());
}

pub fn lt_signed_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    msb_multiplicity: AB::VarMaybeExt,
    diff_multiplicity: AB::VarMaybeExt,
) {
    use crate::bytes::polyair::{msb_lookup, slice_u8_range_lookup};
    msb_lookup(builder, msb_multiplicity.clone());
    msb_lookup(builder, msb_multiplicity);
    slice_u8_range_lookup(builder, diff_multiplicity, 1);
}
