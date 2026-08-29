use dt_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use dt_stark::air::{BaseAirBuilder, DTAirBuilder, FullAirBuilder, Polynomial};
use itertools::izip;
use std::fmt::Debug;

use num::BigUint;

use dt_curves::params::{FieldParameters, Limbs};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

use dt_derive::AlignedBorrow;

/// Operation columns for verifying that `lhs < rhs`.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FieldLtCols<T, P: FieldParameters> {
    /// Boolean flags to indicate the first byte in which the element is smaller than the modulus.
    pub(crate) byte_flags: Limbs<T, P::Limbs>,

    pub(crate) lhs_comparison_byte: T,

    pub(crate) rhs_comparison_byte: T,
}

impl<F: Field, P: FieldParameters> FieldLtCols<F, P> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, lhs: &BigUint, rhs: &BigUint) {
        assert!(lhs < rhs);

        let value_limbs = P::to_limbs(lhs);
        let modulus = P::to_limbs(rhs);

        let mut byte_flags = vec![0u8; P::NB_LIMBS];

        for (byte, modulus_byte, flag) in
            izip!(value_limbs.iter().rev(), modulus.iter().rev(), byte_flags.iter_mut().rev())
        {
            assert!(byte <= modulus_byte);
            if byte < modulus_byte {
                *flag = 1;
                self.lhs_comparison_byte = F::from_canonical_u8(*byte);
                self.rhs_comparison_byte = F::from_canonical_u8(*modulus_byte);
                record.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::LTU,
                    a1: 1,
                    a2: 0,
                    b: *byte,
                    c: *modulus_byte,
                });
                break;
            }
        }

        for (byte, flag) in izip!(byte_flags.iter(), self.byte_flags.0.iter_mut()) {
            *flag = F::from_canonical_u8(*byte);
        }
    }
}

impl<V: Copy, P: FieldParameters> FieldLtCols<V, P> {
    pub fn eval<
        AB: DTAirBuilder<Var = V>,
        E1: Into<Polynomial<AB::Expr>> + Clone,
        E2: Into<Polynomial<AB::Expr>> + Clone,
    >(
        &self,
        builder: &mut AB,
        lhs: &E1,
        rhs: &E2,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        // The byte flags give a specification of which byte is `first_eq`, i,e, the first most
        // significant byte for which the lhs is smaller than the modulus. To verify the
        // less-than claim we need to check that:
        // * For all bytes until `first_eq` the lhs byte is equal to the modulus byte.
        // * For the `first_eq` byte the lhs byte is smaller than the modulus byte.
        // * all byte flags are boolean.
        // * only one byte flag is set to one, and the rest are set to zero.

        // Check the flags are of valid form.

        // Verify that only one flag is set to one.
        let mut sum_flags: AB::Expr = AB::Expr::zero();
        for &flag in self.byte_flags.0.iter() {
            // Assert that the flag is boolean.
            builder.when(is_real.clone()).assert_bool(flag);
            // Add the flag to the sum.
            sum_flags = sum_flags.clone() + flag.into();
        }
        // Assert that the sum is equal to one.
        builder.when(is_real.clone()).assert_one(sum_flags);

        // Check the less-than condition.

        // A flag to indicate whether an equality check is necessary (this is for all bytes from
        // most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::zero();

        let rhs: Polynomial<_> = rhs.clone().into();
        let lhs: Polynomial<_> = lhs.clone().into();

        let mut lhs_comparison_byte = AB::Expr::zero();
        let mut rhs_comparison_byte = AB::Expr::zero();
        for (lhs_byte, rhs_byte, &flag) in izip!(
            lhs.coefficients().iter().rev(),
            rhs.coefficients().iter().rev(),
            self.byte_flags.0.iter().rev()
        ) {
            // Once the byte flag was set to one, we turn off the quality check flag.
            // We can do this by calculating the sum of the flags since only `1` is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            lhs_comparison_byte = lhs_comparison_byte.clone() + lhs_byte.clone() * flag;
            rhs_comparison_byte = rhs_comparison_byte.clone() + flag.into() * rhs_byte.clone();

            builder
                .when(is_real.clone())
                .when_not(is_inequality_visited.clone())
                .assert_eq(lhs_byte.clone(), rhs_byte.clone());
        }

        builder.when(is_real.clone()).assert_eq(self.lhs_comparison_byte, lhs_comparison_byte);
        builder.when(is_real.clone()).assert_eq(self.rhs_comparison_byte, rhs_comparison_byte);

        // Send the comparison interaction.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::F::one(),
            self.lhs_comparison_byte,
            self.rhs_comparison_byte,
            is_real,
        )
    }
}

// ============================================================================
// PolyAir three-phase helpers for FieldLtCols
// ============================================================================

/// Number of lookup interactions produced by a single `FieldLtCols<P>` instance:
///   1 (LTU) + ceil(P::NB_LIMBS / 16) (BitVec for byte_flags booleans).
pub const fn field_lt_num_interactions<P: FieldParameters>() -> usize {
    1 + (P::NB_LIMBS + 15) / 16
}

/// Precompute lookup denominators for a `FieldLtCols` instance.
///
/// Emits in order:
///   1. LTU send_byte for the comparison (1 interaction)
///   2. BitVec lookups for `byte_flags` booleans (ceil(NB_LIMBS/16) interactions)
pub fn field_lt_precompute_lc<AB: FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    lhs_comparison_byte: AB::VarMaybeExt,
    rhs_comparison_byte: AB::VarMaybeExt,
    byte_flags: &[AB::VarMaybeExt],
) {
    // #1: LTU comparison
    crate::bytes::polyair::ltu_precompute_lc(builder, lhs_comparison_byte, rhs_comparison_byte);

    // #2..N: BitVec for byte_flags (chunks of 16)
    for chunk in byte_flags.chunks(16) {
        crate::bytes::polyair::bitvec_precompute_lc(builder, chunk.to_vec());
    }
}

/// Gate constraints for `FieldLtCols`.
///
/// Reproduces:
/// - Sum of flags == 1
/// - Equality prefix: for bytes more significant than the first inequality, lhs == rhs
/// - Comparison byte linkage
///
/// NOTE: `byte_flags` boolean enforcement is handled via BitVec lookups in
/// `field_lt_precompute_lc` / `field_lt_lookup`, NOT via gate constraints here.
pub fn field_lt_gate_constraints<AB: FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    lhs_limbs: &[AB::VarMaybeExt],
    rhs_limbs: &[AB::VarMaybeExt],
    lt_cols: &FieldLtCols<AB::VarMaybeExt, P>,
    is_real: AB::VarMaybeExt,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let zero = AB::zero_maybe();
    let nb_limbs = P::NB_LIMBS;

    // Sum of flags must be 1.
    let mut sum_flags = zero.clone();
    for i in 0..nb_limbs {
        sum_flags = sum_flags + lt_cols.byte_flags.0[i].clone();
    }
    builder.when(is_real.clone()).assert_zero(sum_flags - one.clone());

    // Equality prefix and comparison byte accumulation.
    let mut is_inequality_visited = zero.clone();
    let mut lhs_comparison_byte = zero.clone();
    let mut rhs_comparison_byte = zero;
    for i in (0..nb_limbs).rev() {
        let flag = lt_cols.byte_flags.0[i].clone();
        is_inequality_visited = is_inequality_visited + flag.clone();

        lhs_comparison_byte = lhs_comparison_byte + lhs_limbs[i].clone() * flag.clone();
        rhs_comparison_byte = rhs_comparison_byte + rhs_limbs[i].clone() * flag;

        // Before the inequality flag: lhs[i] must equal rhs[i].
        builder.assert_zero(
            is_real.clone() *
                (one.clone() - is_inequality_visited.clone()) *
                (lhs_limbs[i].clone() - rhs_limbs[i].clone()),
        );
    }

    // Comparison byte linkage.
    builder
        .when(is_real.clone())
        .assert_zero(lt_cols.lhs_comparison_byte.clone() - lhs_comparison_byte);
    builder.when(is_real).assert_zero(lt_cols.rhs_comparison_byte.clone() - rhs_comparison_byte);
}

/// Declare multiplicities for a `FieldLtCols` instance's lookups.
///
/// Emits in the same order as `field_lt_precompute_lc`:
///   1. LTU send (1 interaction)
///   2. BitVec sends for byte_flags (ceil(NB_LIMBS/16) interactions)
pub fn field_lt_lookup<AB: FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    // #1: LTU
    crate::bytes::polyair::ltu_lookup(builder, is_real.clone());

    // #2..N: BitVec (conditional on is_real)
    let num_bitvec = (P::NB_LIMBS + 15) / 16;
    for _ in 0..num_bitvec {
        crate::bytes::polyair::bitvec_lookup(builder, is_real.clone());
    }
}
