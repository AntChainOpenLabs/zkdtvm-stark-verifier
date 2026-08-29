use std::fmt::Debug;

use crate::air::WordAirBuilder;
use num::BigUint;

use p3_air::AirBuilder;
use p3_field::Field;

use dt_core_executor::events::{ByteRecord, FieldOperation};
use dt_derive::AlignedBorrow;
use dt_stark::air::{DTAirBuilder, Polynomial};

use super::{util::compute_root_quotient_and_shift, util_air::eval_field_operation};
use dt_curves::params::{FieldParameters, Limbs};

/// A set of columns to compute an emulated modular arithmetic operation.
///
/// *Safety* The input operands (a, b) (not included in the operation columns) are assumed to be
/// elements within the range `[0, 2^{P::nb_bits()})`. the result is also assumed to be within the
/// same range. Let `M = P:modulus()`. The constraints of the function [`FieldOpCols::eval`] assert
/// that:
/// * When `op` is `FieldOperation::Add`, then `result = a + b mod M`.
/// * When `op` is `FieldOperation::Mul`, then `result = a * b mod M`.
/// * When `op` is `FieldOperation::Sub`, then `result = a - b mod M`.
/// * When `op` is `FieldOperation::Div`, then `result * b = a mod M`.
///
/// **Warning**: The constraints do not check for division by zero. The caller is responsible for
/// ensuring that the division operation is valid.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FieldMulDigitCols<T, P: FieldParameters> {
    /// The result of `a op b`, where a is field element, b is digit
    pub result: Limbs<T, P::Limbs>,
    pub carry: T,
    pub(crate) witness: Limbs<T, P::AddWitness>, /* Note: The witness length of muldigit is
                                                  * identical to that of add_op,
                                                  * so the AddWitness type can be directly
                                                  * reused. */
}

impl<F: Field, P: FieldParameters> FieldMulDigitCols<F, P> {
    pub fn populate_carry_and_witness(
        &mut self,
        a: &BigUint,
        b: u8,
        _op: FieldOperation,
        modulus: &BigUint,
    ) -> BigUint {
        // Here, op can only be mul.
        let p_a: Polynomial<F> = P::to_limbs_field::<F, _>(a).into();
        let d_b: F = F::from_canonical_u8(b);
        // let (result, carry) = match op {
        //     FieldOperation::Mul => {
        //         ((a * b) % modulus, ((a * b - (a * b) % modulus) / modulus).to_bytes_le()[0])
        //     }
        //     FieldOperation::Sqr
        //     | FieldOperation::Add
        //     | FieldOperation::Sub
        //     | FieldOperation::Div => unreachable!(),
        // };
        let result = a * b % modulus;
        let carry = ((a * b - (a * b) % modulus) / modulus).to_bytes_le()[0];
        // println!(
        //     "a: {:?}, b: {:?} modulus: {:?} result: {:?}, carry: {:?}",
        //     a, b, modulus, result, carry
        // );
        debug_assert!(&result < modulus);
        debug_assert!((carry as u16) < 256u16);
        // match op {
        //     FieldOperation::Mul => debug_assert_eq!(modulus * carry, a * b - &result),
        //     FieldOperation::Sqr
        //     | FieldOperation::Add
        //     | FieldOperation::Sub
        //     | FieldOperation::Div => unreachable!(),
        // }
        debug_assert_eq!(modulus * carry, a * b - &result);

        // Here we have special logic for p_modulus because to_limbs_field only works for numbers in
        // the field, but modulus can == the field modulus so it can have 1 extra limb (ex.
        // uint256).
        let p_modulus_limbs =
            modulus.to_bytes_le().iter().map(|x| F::from_canonical_u8(*x)).collect::<Vec<F>>();
        let p_modulus: Polynomial<F> = p_modulus_limbs.iter().into();
        let p_result: Polynomial<F> = P::to_limbs_field::<F, _>(&result).into();
        let d_carry: F = F::from_canonical_u8(carry);

        // Compute the vanishing polynomial.
        // let p_op = match op {
        //     FieldOperation::Mul => &p_a * d_b,
        //     FieldOperation::Sqr
        //     | FieldOperation::Add
        //     | FieldOperation::Sub
        //     | FieldOperation::Div => unreachable!(),
        // };
        let p_op = &p_a * d_b;
        let p_vanishing: Polynomial<F> = &p_op - &p_result - &p_modulus * d_carry;

        let p_witness = compute_root_quotient_and_shift(
            &p_vanishing,
            P::WITNESS_OFFSET,
            P::NB_BITS_PER_LIMB as u32,
            P::NB_ADD_WITNESS_LIMBS,
        );

        self.result = p_result.into();
        self.carry = d_carry;
        self.witness = Limbs(p_witness.try_into().unwrap());

        result
    }

    /// Populate these columns with a specified modulus. This is useful in the `mulmod` precompile
    /// as an example.
    #[allow(clippy::too_many_arguments)]
    pub fn populate_with_modulus(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        b: u8,
        modulus: &BigUint,
        op: FieldOperation,
    ) -> BigUint {
        assert_ne!(op, FieldOperation::Add, "add is not allowed");
        assert_ne!(op, FieldOperation::Sub, "sub is not allowed");
        assert_ne!(op, FieldOperation::Sqr, "sqr is not allowed");
        assert_ne!(op, FieldOperation::Div, "div is not allowed");

        let result = self.populate_carry_and_witness(a, b, op, modulus);

        // Range checks
        record.add_u8_range_checks_field(&self.result.0);
        record.add_u8_range_checks_field(&[self.carry]);
        record.add_u16_range_checks_field(&self.witness.0);

        result
    }

    /// Populate these columns without a specified modulus (will use the modulus of the field
    /// parameters).
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        b: u8,
        op: FieldOperation,
    ) -> BigUint {
        self.populate_with_modulus(record, a, b, &P::modulus(), op)
    }
}

impl<V: Copy, P: FieldParameters> FieldMulDigitCols<V, P> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval_with_modulus<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<AB::Expr> + Clone),
        modulus: &(impl Into<Polynomial<AB::Expr>> + Clone),
        op: FieldOperation,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_a: Polynomial<AB::Expr> = (a).clone().into();
        let p_b: AB::Expr = (b).clone().into();
        let p_result: Polynomial<AB::Expr> = self.result.into();
        let p_op: Polynomial<<AB as AirBuilder>::Expr> = match op {
            FieldOperation::Mul => p_a * p_b,
            FieldOperation::Sqr |
            FieldOperation::Add |
            FieldOperation::Sub |
            FieldOperation::Div => unreachable!(),
        };
        self.eval_with_polynomials(builder, p_op, modulus.clone(), p_result, is_real);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval_with_polynomials<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        op: impl Into<Polynomial<AB::Expr>>,
        modulus: impl Into<Polynomial<AB::Expr>>,
        result: impl Into<Polynomial<AB::Expr>>,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_op: Polynomial<AB::Expr> = op.into();
        let p_result: Polynomial<AB::Expr> = result.into();
        let p_modulus: Polynomial<AB::Expr> = modulus.into();
        let p_carry: AB::Expr = self.carry.into();
        let p_op_minus_result: Polynomial<AB::Expr> = p_op - &p_result;
        let p_vanishing = p_op_minus_result - p_modulus * p_carry;
        let p_witness = self.witness.0.iter().into();
        eval_field_operation::<AB, P>(builder, &p_vanishing, &p_witness);

        // Range checks for the result, carry, and witness columns.
        builder.slice_range_check_u8(&self.result.0, is_real.clone());
        builder.slice_range_check_u8(&[self.carry], is_real.clone());
        builder.slice_range_check_u16(p_witness.coefficients(), is_real);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<AB::Expr> + Clone),
        op: FieldOperation,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_limbs = Polynomial::from_iter(P::modulus_field_iter::<AB::F>().map(AB::Expr::from));
        self.eval_with_modulus::<AB>(builder, a, b, &p_limbs, op, is_real);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use dt_core_executor::{ExecutionRecord, Program};
    use dt_curves::params::FieldParameters;
    use dt_stark::{
        air::{DTAirBuilder, MachineAir},
        sumcheck::trace::CompressedMatrix,
        StarkGenericConfig, PROOF_MAX_NUM_PVS,
    };
    use num::BigUint;
    use p3_air::BaseAir;
    use p3_field::Field;
    use p3_uni_stark::{get_symbolic_constraints, SymbolicAirBuilder};

    use super::{FieldMulDigitCols, FieldOperation, Limbs};

    use crate::utils::{
        pad_to_power_of_two,
        uni_stark::{uni_stark_prove, uni_stark_verify},
    };
    use core::borrow::{Borrow, BorrowMut};
    use dt_core_executor::events::ByteRecord;
    use dt_curves::{
        edwards::ed25519::Ed25519BaseField, weierstrass::secp256k1::Secp256k1BaseField,
    };
    use dt_derive::AlignedBorrow;
    use dt_stark::baby_bear_poseidon2::BabyBearPoseidon2;
    use num::bigint::RandBigInt;
    use p3_air::Air;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};
    use std::mem::size_of;

    #[derive(AlignedBorrow, Debug, Clone)]
    pub struct TestCols<T, P: FieldParameters> {
        pub a: Limbs<T, P::Limbs>,
        pub b: T,
        pub a_op_b: FieldMulDigitCols<T, P>,
    }

    pub const NUM_TEST_COLS: usize = size_of::<TestCols<u8, Secp256k1BaseField>>();

    struct FieldOpChip<P: FieldParameters> {
        pub operation: FieldOperation,
        pub _phantom: std::marker::PhantomData<P>,
    }

    impl<P: FieldParameters> FieldOpChip<P> {
        pub const fn new(operation: FieldOperation) -> Self {
            Self { operation, _phantom: std::marker::PhantomData }
        }
    }

    impl<F: Field, P: FieldParameters> MachineAir<F> for FieldOpChip<P> {
        type Record = ExecutionRecord;

        type Program = Program;

        fn name(&self) -> String {
            format!("FieldOp{:?}", self.operation)
        }

        fn generate_trace(
            &self,
            _: &ExecutionRecord,
            output: &mut ExecutionRecord,
        ) -> CompressedMatrix<F> {
            let mut rng = thread_rng();
            let num_rows = 1 << 8;
            let mut operands: Vec<(BigUint, u8)> = (0..num_rows - 4)
                .map(|_| {
                    let a = rng.gen_biguint(256) % &P::modulus();
                    let b = rng.gen_range(0..=255);
                    (a, b)
                })
                .collect();

            // Hardcoded edge cases.
            operands.extend(vec![
                (BigUint::from(0u32), 1u8),
                (BigUint::from(1u32), 2u8),
                (BigUint::from(4u32), 5u8),
                (BigUint::from(10u32), 19u8),
            ]);

            let rows = operands
                .iter()
                .map(|(a, b)| {
                    let mut blu_events = Vec::new();
                    let mut row = [F::zero(); NUM_TEST_COLS];
                    let cols: &mut TestCols<F, P> = row.as_mut_slice().borrow_mut();
                    cols.a = P::to_limbs_field::<F, _>(a);
                    cols.b = F::from_canonical_u8(*b);
                    cols.a_op_b.populate(&mut blu_events, a, *b, self.operation);
                    output.add_byte_lookup_events(blu_events);
                    row
                })
                .collect::<Vec<_>>();
            // Convert the trace to a row major matrix.
            let mut trace =
                RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_TEST_COLS);

            // Pad the trace to a power of two.
            pad_to_power_of_two::<NUM_TEST_COLS, F>(&mut trace.values);

            CompressedMatrix::from_full_matrix_no_padding(trace)
        }

        fn included(&self, _: &Self::Record) -> bool {
            true
        }
    }

    impl<F: Field, P: FieldParameters> BaseAir<F> for FieldOpChip<P> {
        fn width(&self) -> usize {
            NUM_TEST_COLS
        }
    }

    impl<AB, P: FieldParameters> Air<AB> for FieldOpChip<P>
    where
        AB: DTAirBuilder,
        Limbs<AB::Var, P::Limbs>: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0);
            let local: &TestCols<AB::Var, P> = (*local).borrow();
            local.a_op_b.eval(builder, &local.a, &local.b, self.operation, AB::F::one());
        }
    }

    #[test]
    fn generate_trace() {
        for op in [FieldOperation::Mul].iter() {
            println!("op: {:?}", op);
            let chip: FieldOpChip<Ed25519BaseField> = FieldOpChip::new(*op);
            let shard = ExecutionRecord::default();
            let _: CompressedMatrix<BabyBear> =
                chip.generate_trace(&shard, &mut ExecutionRecord::default());
        }
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();

        for op in [FieldOperation::Mul].iter() {
            println!("op: {:?}", op);

            let mut challenger = config.challenger();

            let chip: FieldOpChip<Ed25519BaseField> = FieldOpChip::new(*op);
            let shard = ExecutionRecord::default();
            let trace: CompressedMatrix<BabyBear> =
                chip.generate_trace(&shard, &mut ExecutionRecord::default());
            let proof = uni_stark_prove::<BabyBearPoseidon2, _>(
                &config,
                &chip,
                &mut challenger,
                trace.main,
            );

            let mut challenger = config.challenger();
            uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
        }
    }
}

// ============================================================================
// PolyAir helpers for FieldMulDigitCols
// ============================================================================

/// Number of interactions emitted by a `FieldMulDigitCols` range-check.
///
/// - U8Range pairs for `result` limbs: `P::NB_LIMBS / 2`
/// - U8Range pair for `carry` (carry, 0): `1`
/// - U16Range for `witness` limbs: `P::NB_ADD_WITNESS_LIMBS`
pub const fn field_mul_digit_num_interactions<P: FieldParameters>() -> usize {
    P::NB_LIMBS / 2 + 1 + P::NB_ADD_WITNESS_LIMBS
}

/// Precompute lookup denominators for a `FieldMulDigitCols` instance.
///
/// Emits denominators in the same order as `eval_with_polynomials`:
/// 1. U8Range pairs for `result` limbs (`P::NB_LIMBS / 2` interactions)
/// 2. U8Range pair for `carry` with zero-padding (`1` interaction)
/// 3. U16Range for `witness` limbs (`P::NB_ADD_WITNESS_LIMBS` interactions)
///
/// Chips that need `witness(beta)` must retain it explicitly after all lookup
/// denominators for the row have been emitted.
pub fn field_mul_digit_precompute_lc<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    result_limbs: &[AB::VarMaybeExt],
    carry: AB::VarMaybeExt,
    witness_limbs: &[AB::VarMaybeExt],
) {
    use crate::bytes::polyair::{
        slice_u16_range_precompute_lc, slice_u8_range_precompute_lc, u8_range_pair_precompute_lc,
    };

    // U8Range for result limbs (pairs)
    slice_u8_range_precompute_lc(builder, result_limbs);
    // U8Range for carry paired with zero (single-byte case)
    u8_range_pair_precompute_lc(builder, carry, AB::zero_maybe());
    // U16Range for witness limbs
    slice_u16_range_precompute_lc(builder, witness_limbs);
}

/// Gate constraints for a `FieldMulDigitCols` instance with a **fixed Mul** operation.
///
/// Mirrors `eval_with_modulus` for `FieldOperation::Mul` where `b` is a single digit:
///   `p_op = p_a * digit_b`, `p_result = cols.result`,
///   `vanishing = p_op - p_result - carry * modulus`.
///
/// The vanishing identity is checked directly at `beta`.
///
/// NOTE: `carry` boolean enforcement is NOT included — handled at chip layer.
pub fn field_mul_digit_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    digit_b: AB::VarMaybeExt,
    result_limbs: &[AB::VarMaybeExt],
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) {
    let a_beta = super::field_op::field_op_beta_from_coeffs(builder, a_limbs);
    let result_beta = super::field_op::field_op_beta_from_coeffs(builder, result_limbs);
    let vanishing_beta = a_beta * digit_b - result_beta - consts.modulus_beta.clone() * carry;
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

/// Same as `field_mul_digit_gate_constraints`, but `a_beta` and `result_beta` are
/// pre-computed (typical when neither's limbs need to live in `reserved_poly`).
/// `carry` is still passed as a single limb (range-checked via main trace).
pub fn field_mul_digit_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    digit_b: AB::VarMaybeExt,
    result_beta: AB::VarExt,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) {
    let vanishing_beta = a_beta * digit_b - result_beta - consts.modulus_beta.clone() * carry;
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

/// Declare multiplicities for a `FieldMulDigitCols` instance's range check lookups.
///
/// Emits `send` calls in the same order as `field_mul_digit_precompute_lc`:
/// 1. U8Range pairs for result (`P::NB_LIMBS / 2` sends)
/// 2. U8Range pair for carry (`1` send)
/// 3. U16Range for witness (`P::NB_ADD_WITNESS_LIMBS` sends)
pub fn field_mul_digit_lookup<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    use crate::bytes::polyair::{slice_u16_range_lookup, slice_u8_range_lookup};

    // U8Range for result limbs
    slice_u8_range_lookup(builder, is_real.clone(), P::NB_LIMBS / 2);
    // U8Range for carry
    slice_u8_range_lookup(builder, is_real.clone(), 1);
    // U16Range for witness limbs
    slice_u16_range_lookup(builder, is_real, P::NB_ADD_WITNESS_LIMBS);
}
