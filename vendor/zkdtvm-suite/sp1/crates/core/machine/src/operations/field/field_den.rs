use std::fmt::Debug;

use dt_core_executor::events::ByteRecord;
use dt_curves::params::{FieldParameters, Limbs};
use dt_derive::AlignedBorrow;
use dt_stark::air::{DTAirBuilder, Polynomial};
use num::BigUint;
use p3_air::AirBuilder;
use p3_field::Field;

use super::{util::compute_root_quotient_and_shift, util_air::eval_field_operation};
use crate::air::WordAirBuilder;

/// A set of columns to compute `FieldDen(a, b)` where `a`, `b` are field elements.
///
/// `a / (1 + b)` if `sign`
/// `a / (1 - b) ` if `!sign`
///
/// *Safety*: the operation assumes that the denominators are never zero. It is the responsibility
/// of the caller to ensure that condition.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FieldDenCols<T, P: FieldParameters> {
    /// The result of `a den b`, where a, b are field elements
    pub result: Limbs<T, P::Limbs>,
    pub(crate) carry: Limbs<T, P::Limbs>,
    pub(crate) witness: Limbs<T, P::Witness>,
}

impl<F: Field, P: FieldParameters> FieldDenCols<F, P> {
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        b: &BigUint,
        sign: bool,
    ) -> BigUint {
        let p = P::modulus();
        let minus_b_int = &p - b;
        let b_signed = if sign { b.clone() } else { minus_b_int };
        let denominator = (b_signed + 1u32) % &(p.clone());
        let den_inv = denominator.modpow(&(&p - 2u32), &p);
        let result = (a * &den_inv) % &p;
        debug_assert_eq!(&den_inv * &denominator % &p, BigUint::from(1u32));
        debug_assert!(result < p);

        let equation_lhs = if sign { b * &result + &result } else { b * &result + a };
        let equation_rhs = if sign { a.clone() } else { result.clone() };
        let carry = (&equation_lhs - &equation_rhs) / &p;
        debug_assert!(carry < p);
        debug_assert_eq!(&carry * &p, &equation_lhs - &equation_rhs);

        let p_a: Polynomial<F> = P::to_limbs_field::<F, _>(a).into();
        let p_b: Polynomial<F> = P::to_limbs_field::<F, _>(b).into();
        let p_p: Polynomial<F> = P::to_limbs_field::<F, _>(&p).into();
        let p_result: Polynomial<F> = P::to_limbs_field::<F, _>(&result).into();
        let p_carry: Polynomial<F> = P::to_limbs_field::<F, _>(&carry).into();

        // Compute the vanishing polynomial.
        let vanishing_poly = if sign {
            &p_b * &p_result + &p_result - &p_a - &p_carry * &p_p
        } else {
            &p_b * &p_result + &p_a - &p_result - &p_carry * &p_p
        };
        debug_assert_eq!(vanishing_poly.degree(), P::NB_WITNESS_LIMBS);

        let p_witness = compute_root_quotient_and_shift(
            &vanishing_poly,
            P::WITNESS_OFFSET,
            P::NB_BITS_PER_LIMB as u32,
            P::NB_WITNESS_LIMBS,
        );

        self.result = p_result.into();
        self.carry = p_carry.into();
        self.witness = Limbs(p_witness.try_into().unwrap());

        // Range checks
        record.add_u8_range_checks_field(&self.result.0);
        record.add_u8_range_checks_field(&self.carry.0);
        record.add_u16_range_checks_field(&self.witness.0);

        result
    }
}

impl<V: Copy, P: FieldParameters> FieldDenCols<V, P>
where
    Limbs<V, P::Limbs>: Copy,
{
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &Limbs<AB::Var, P::Limbs>,
        b: &Limbs<AB::Var, P::Limbs>,
        sign: bool,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
    {
        let p_a: Polynomial<<AB as AirBuilder>::Expr> = (*a).into();
        let p_b: Polynomial<<AB as AirBuilder>::Expr> = (*b).into();
        let p_result: Polynomial<<AB as AirBuilder>::Expr> = self.result.into();
        let p_carry: Polynomial<<AB as AirBuilder>::Expr> = self.carry.into();

        // Compute the vanishing polynomial:
        //      lhs(x) = sign * (b(x) * result(x) + result(x)) + (1 - sign) * (b(x) * result(x) +
        // a(x))      rhs(x) = sign * a(x) + (1 - sign) * result(x)
        //      lhs(x) - rhs(x) - carry(x) * p(x)
        let p_equation_lhs =
            if sign { &p_b * &p_result + &p_result } else { &p_b * &p_result + &p_a };
        let p_equation_rhs = if sign { p_a } else { p_result };

        let p_lhs_minus_rhs = &p_equation_lhs - &p_equation_rhs;
        let p_limbs: Polynomial<<AB as AirBuilder>::Expr> =
            Polynomial::from_iter(P::modulus_field_iter::<AB::F>().map(AB::Expr::from));

        let p_vanishing: Polynomial<<AB as AirBuilder>::Expr> =
            p_lhs_minus_rhs - &p_carry * &p_limbs;

        let p_witness = self.witness.0.iter().into();

        eval_field_operation::<AB, P>(builder, &p_vanishing, &p_witness);

        // Range checks for the result, carry, and witness columns.
        builder.slice_range_check_u8(&self.result.0, is_real.clone());
        builder.slice_range_check_u8(&self.carry.0, is_real.clone());
        builder.slice_range_check_u16(&self.witness.0, is_real);
    }
}

// ============================================================================
// PolyAir three-phase helpers for FieldDenCols
// ============================================================================
//
// FieldDenCols has the same column structure as FieldOpCols (result, carry, witness
// all as Limbs), so the interaction pattern is identical:
//   P::NB_LIMBS / 2 (U8Range result) + P::NB_LIMBS / 2 (U8Range carry) + P::NB_WITNESS_LIMBS
// (U16Range witness)
//
// We delegate to the FieldOpCols helpers for precompute_lc, gate_constraints, and lookup.

/// Compute the number of interactions generated by a single `FieldDenCols` instance.
///
/// Same as `field_op_num_interactions`: `P::NB_LIMBS + P::NB_WITNESS_LIMBS`.
pub const fn field_den_num_interactions<P: FieldParameters>() -> usize {
    super::field_op::field_op_num_interactions::<P>()
}

/// Precompute lookup denominators for a `FieldDenCols` instance.
///
/// Delegates to `field_op_precompute_lc` since the column layout is identical.
pub fn field_den_precompute_lc<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    result_limbs: &[AB::VarMaybeExt],
    carry_limbs: &[AB::VarMaybeExt],
    witness_limbs: &[AB::VarMaybeExt],
) {
    super::field_op::field_op_precompute_lc::<AB, P>(
        builder,
        result_limbs,
        carry_limbs,
        witness_limbs,
    );
}

/// Declare multiplicities for a `FieldDenCols` instance's range check lookups.
///
/// Delegates to `field_op_lookup` since the interaction pattern is identical.
pub fn field_den_lookup<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    super::field_op::field_op_lookup::<AB, P>(builder, is_real);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use dt_core_executor::{ExecutionRecord, Program};
    use dt_curves::params::FieldParameters;
    use dt_stark::{
        air::{DTAirBuilder, MachineAir},
        baby_bear_poseidon2::BabyBearPoseidon2,
        sumcheck::trace::CompressedMatrix,
        StarkGenericConfig,
    };
    use num::BigUint;
    use p3_air::BaseAir;
    use p3_field::Field;

    use crate::utils::uni_stark::{uni_stark_prove, uni_stark_verify};

    use super::{FieldDenCols, Limbs};

    use core::{
        borrow::{Borrow, BorrowMut},
        mem::size_of,
    };
    use dt_curves::edwards::ed25519::Ed25519BaseField;
    use dt_derive::AlignedBorrow;
    use num::bigint::RandBigInt;
    use p3_air::Air;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::thread_rng;

    #[derive(Debug, Clone, AlignedBorrow)]
    pub struct TestCols<T, P: FieldParameters> {
        pub a: Limbs<T, P::Limbs>,
        pub b: Limbs<T, P::Limbs>,
        pub a_den_b: FieldDenCols<T, P>,
    }

    pub const NUM_TEST_COLS: usize = size_of::<TestCols<u8, Ed25519BaseField>>();

    struct FieldDenChip<P: FieldParameters> {
        pub sign: bool,
        pub _phantom: std::marker::PhantomData<P>,
    }

    impl<P: FieldParameters> FieldDenChip<P> {
        pub const fn new(sign: bool) -> Self {
            Self { sign, _phantom: std::marker::PhantomData }
        }
    }

    impl<F: Field, P: FieldParameters> MachineAir<F> for FieldDenChip<P> {
        type Record = ExecutionRecord;

        type Program = Program;

        fn name(&self) -> String {
            "FieldDen".to_string()
        }

        fn generate_trace(
            &self,
            _: &ExecutionRecord,
            output: &mut ExecutionRecord,
        ) -> CompressedMatrix<F> {
            let mut rng = thread_rng();
            let num_rows = 1 << 8;
            let mut operands: Vec<(BigUint, BigUint)> = (0..num_rows - 4)
                .map(|_| {
                    let a = rng.gen_biguint(256) % &P::modulus();
                    let b = rng.gen_biguint(256) % &P::modulus();
                    (a, b)
                })
                .collect();
            // Hardcoded edge cases.
            operands.extend(vec![
                (BigUint::from(0u32), BigUint::from(0u32)),
                (BigUint::from(1u32), BigUint::from(2u32)),
                (BigUint::from(4u32), BigUint::from(5u32)),
                (BigUint::from(10u32), BigUint::from(19u32)),
            ]);
            // It is important that the number of rows is an exact power of 2,
            // otherwise the padding will not work correctly.
            assert_eq!(operands.len(), num_rows);

            let rows = operands
                .iter()
                .map(|(a, b)| {
                    let mut row = [F::zero(); NUM_TEST_COLS];
                    let cols: &mut TestCols<F, P> = row.as_mut_slice().borrow_mut();
                    cols.a = P::to_limbs_field::<F, _>(a);
                    cols.b = P::to_limbs_field::<F, _>(b);
                    cols.a_den_b.populate(output, a, b, self.sign);
                    row
                })
                .collect::<Vec<_>>();
            // Convert the trace to a row major matrix.

            // Note we do not pad the trace here because we cannot just pad with all 0s.

            CompressedMatrix::from_full_matrix_no_padding(RowMajorMatrix::new(
                rows.into_iter().flatten().collect::<Vec<_>>(),
                NUM_TEST_COLS,
            ))
        }

        fn included(&self, _: &Self::Record) -> bool {
            true
        }
    }

    impl<F: Field, P: FieldParameters> BaseAir<F> for FieldDenChip<P> {
        fn width(&self) -> usize {
            NUM_TEST_COLS
        }
    }

    impl<AB, P: FieldParameters> Air<AB> for FieldDenChip<P>
    where
        AB: DTAirBuilder,
        Limbs<AB::Var, P::Limbs>: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0);
            let local: &TestCols<AB::Var, P> = (*local).borrow();
            local.a_den_b.eval(builder, &local.a, &local.b, self.sign, AB::F::zero());
        }
    }

    #[test]
    fn generate_trace() {
        let shard = ExecutionRecord::default();
        let chip: FieldDenChip<Ed25519BaseField> = FieldDenChip::new(true);
        let trace: CompressedMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.main.values)
    }

    #[test]
    fn prove_field() {
        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let shard = ExecutionRecord::default();

        let chip: FieldDenChip<Ed25519BaseField> = FieldDenChip::new(true);
        let trace: CompressedMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        let proof =
            uni_stark_prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace.main);

        let mut challenger = config.challenger();
        uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
