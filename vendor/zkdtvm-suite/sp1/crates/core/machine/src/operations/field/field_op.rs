use std::fmt::Debug;

use crate::air::WordAirBuilder;
use num::{BigUint, Zero};

use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

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
pub struct FieldOpCols<T, P: FieldParameters> {
    /// The result of `a op b`, where a, b are field elements
    pub result: Limbs<T, P::Limbs>,
    pub carry: Limbs<T, P::Limbs>,
    pub(crate) witness: Limbs<T, P::Witness>,
}

impl<F: Field, P: FieldParameters> FieldOpCols<F, P> {
    #[allow(clippy::too_many_arguments)]
    /// Populate result and carry columns from the equation (a*b + c) % modulus
    pub fn populate_mul_and_carry(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        b: &BigUint,
        c: &BigUint,
        modulus: &BigUint,
    ) -> (BigUint, BigUint) {
        let p_a: Polynomial<F> = P::to_limbs_field::<F, _>(a).into();
        let p_b: Polynomial<F> = P::to_limbs_field::<F, _>(b).into();
        let p_c: Polynomial<F> = P::to_limbs_field::<F, _>(c).into();

        let mul_add = a * b + c;
        let result = &mul_add % modulus;
        let carry = (mul_add - &result) / modulus;
        debug_assert!(&result < modulus);
        debug_assert!(&carry < modulus);
        debug_assert_eq!(&carry * modulus, a * b + c - &result);

        let p_modulus_limbs =
            modulus.to_bytes_le().iter().map(|x| F::from_canonical_u8(*x)).collect::<Vec<F>>();
        let p_modulus: Polynomial<F> = p_modulus_limbs.iter().into();
        let p_result: Polynomial<F> = P::to_limbs_field::<F, _>(&result).into();
        let p_carry: Polynomial<F> = P::to_limbs_field::<F, _>(&carry).into();

        let p_op = &p_a * &p_b + &p_c;
        let p_vanishing = &p_op - &p_result - &p_carry * &p_modulus;

        let p_witness = compute_root_quotient_and_shift(
            &p_vanishing,
            P::WITNESS_OFFSET,
            P::NB_BITS_PER_LIMB as u32,
            P::NB_WITNESS_LIMBS,
        );

        self.result = p_result.into();
        self.carry = p_carry.into();
        self.witness = Limbs(p_witness.try_into().unwrap());

        record.add_u8_range_checks_field(&self.result.0);
        record.add_u8_range_checks_field(&self.carry.0);
        record.add_u16_range_checks_field(&self.witness.0);

        (result, carry)
    }

    pub fn populate_carry_and_witness(
        &mut self,
        a: &BigUint,
        b: &BigUint,
        op: FieldOperation,
        modulus: &BigUint,
    ) -> BigUint {
        let p_a: Polynomial<F> = P::to_limbs_field::<F, _>(a).into();
        let p_b: Polynomial<F> = P::to_limbs_field::<F, _>(b).into();
        let (result, carry) = match op {
            FieldOperation::Add => ((a + b) % modulus, (a + b - (a + b) % modulus) / modulus),
            FieldOperation::Mul | FieldOperation::Sqr => {
                ((a * b) % modulus, (a * b - (a * b) % modulus) / modulus)
            }
            FieldOperation::Sub | FieldOperation::Div => unreachable!(),
        };
        debug_assert!(&result < modulus);
        debug_assert!(&carry < modulus);
        match op {
            FieldOperation::Add => debug_assert_eq!(&carry * modulus, a + b - &result),
            FieldOperation::Mul | FieldOperation::Sqr => {
                debug_assert_eq!(&carry * modulus, a * b - &result)
            }
            FieldOperation::Sub | FieldOperation::Div => unreachable!(),
        }

        // Here we have special logic for p_modulus because to_limbs_field only works for numbers in
        // the field, but modulus can == the field modulus so it can have 1 extra limb (ex.
        // uint256).
        let p_modulus_limbs =
            modulus.to_bytes_le().iter().map(|x| F::from_canonical_u8(*x)).collect::<Vec<F>>();
        let p_modulus: Polynomial<F> = p_modulus_limbs.iter().into();
        let p_result: Polynomial<F> = P::to_limbs_field::<F, _>(&result).into();
        let p_carry: Polynomial<F> = P::to_limbs_field::<F, _>(&carry).into();

        // Compute the vanishing polynomial.
        let p_op = match op {
            FieldOperation::Add => &p_a + &p_b,
            FieldOperation::Mul => &p_a * &p_b,
            FieldOperation::Sqr => p_a.square(),
            FieldOperation::Sub | FieldOperation::Div => unreachable!(),
        };
        let p_vanishing: Polynomial<F> = &p_op - &p_result - &p_carry * &p_modulus;

        let mut p_witness = compute_root_quotient_and_shift(
            &p_vanishing,
            P::WITNESS_OFFSET,
            P::NB_BITS_PER_LIMB as u32,
            P::NB_WITNESS_LIMBS,
        );

        self.result = p_result.into();
        self.carry = p_carry.into();
        p_witness.resize(P::NB_WITNESS_LIMBS, F::zero());
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
        b: &BigUint,
        modulus: &BigUint,
        op: FieldOperation,
    ) -> BigUint {
        if op == FieldOperation::Sqr {
            assert_eq!(*a, *b, "square operation requires a == b");
        }
        if op == FieldOperation::Div {
            assert_ne!(*b, BigUint::zero(), "division by zero is not allowed");
            assert_ne!(*b, *modulus, "division by zero is not allowed");
        }

        let result = match op {
            // If doing the subtraction operation, a - b = result, equivalent to a = result + b.
            FieldOperation::Sub => {
                let result = (modulus.clone() + a - b) % modulus;
                // We populate the carry, witness_low, witness_high as if we were doing an addition
                // with result + b. But we populate `result` with the actual result
                // of the subtraction because those columns are expected to contain
                // the result by the user. Note that this reversal means we have to
                // flip result, a correspondingly in the `eval` function.
                self.populate_carry_and_witness(&result, b, FieldOperation::Add, modulus);
                self.result = P::to_limbs_field::<F, _>(&result);
                result
            }
            // a / b = result is equivalent to a = result * b.
            FieldOperation::Div => {
                // As modulus is prime, we can use Fermat's little theorem to compute the
                // inverse.
                cfg_if::cfg_if! {
                    if #[cfg(feature = "bigint-rug")] {
                        use dt_curves::utils::{biguint_to_rug, rug_to_biguint};
                        let rug_a = biguint_to_rug(a);
                        let rug_b = biguint_to_rug(b);
                        let rug_modulus = biguint_to_rug(modulus);
                        let rug_result = (rug_a
                            * rug_b.pow_mod(&(rug_modulus.clone() - 2u32), &rug_modulus.clone()).unwrap())
                            % rug_modulus.clone();
                        let result = rug_to_biguint(&rug_result);
                    } else {
                        let result =
                            (a * b.modpow(&(modulus.clone() - 2u32), &modulus.clone())) % modulus.clone();
                    }
                }
                // We populate the carry, witness_low, witness_high as if we were doing a
                // multiplication with result * b. But we populate `result` with the
                // actual result of the multiplication because those columns are
                // expected to contain the result by the user. Note that this
                // reversal means we have to flip result, a correspondingly in the `eval`
                // function.
                self.populate_carry_and_witness(&result, b, FieldOperation::Mul, modulus);
                self.result = P::to_limbs_field::<F, _>(&result);
                result
            }
            _ => self.populate_carry_and_witness(a, b, op, modulus),
        };

        // Range checks
        record.add_u8_range_checks_field(&self.result.0);
        record.add_u8_range_checks_field(&self.carry.0);
        record.add_u16_range_checks_field(&self.witness.0);

        result
    }

    /// Populate these columns without a specified modulus (will use the modulus of the field
    /// parameters).
    pub fn populate(
        &mut self,
        record: &mut impl ByteRecord,
        a: &BigUint,
        b: &BigUint,
        op: FieldOperation,
    ) -> BigUint {
        self.populate_with_modulus(record, a, b, &P::modulus(), op)
    }
}

impl<V: Copy, P: FieldParameters> FieldOpCols<V, P> {
    /// Allows an evaluation over opetations specified by boolean flags.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_variable<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<Polynomial<AB::Expr>> + Clone),
        modulus: &(impl Into<Polynomial<AB::Expr>> + Clone),
        is_add: impl Into<AB::Expr> + Clone,
        is_sub: impl Into<AB::Expr> + Clone,
        is_mul: impl Into<AB::Expr> + Clone,
        is_div: impl Into<AB::Expr> + Clone,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_a_param: Polynomial<AB::Expr> = (a).clone().into();
        let p_b: Polynomial<AB::Expr> = (b).clone().into();
        let p_res_param: Polynomial<AB::Expr> = self.result.into();

        let is_add: AB::Expr = is_add.into();
        let is_sub: AB::Expr = is_sub.into();
        let is_mul: AB::Expr = is_mul.into();
        let is_div: AB::Expr = is_div.into();

        let p_result = p_res_param.clone() * (is_add.clone() + is_mul.clone()) +
            p_a_param.clone() * (is_sub.clone() + is_div.clone());

        let p_add = p_a_param.clone() + p_b.clone();
        let p_sub = p_res_param.clone() + p_b.clone();
        let p_mul = p_a_param.clone() * p_b.clone();
        let p_div = p_res_param * p_b.clone();
        let p_op = p_add * is_add + p_sub * is_sub + p_mul * is_mul + p_div * is_div;

        self.eval_with_polynomials(builder, p_op, modulus.clone(), p_result, is_real);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval_mul_and_carry<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<Polynomial<AB::Expr>> + Clone),
        c: &(impl Into<Polynomial<AB::Expr>> + Clone),
        modulus: &(impl Into<Polynomial<AB::Expr>> + Clone),
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_a: Polynomial<AB::Expr> = (a).clone().into();
        let p_b: Polynomial<AB::Expr> = (b).clone().into();
        let p_c: Polynomial<AB::Expr> = (c).clone().into();

        let p_result: Polynomial<_> = self.result.into();
        let p_op = p_a * p_b + p_c;

        self.eval_with_polynomials(builder, p_op, modulus.clone(), p_result, is_real);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval_with_modulus<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<Polynomial<AB::Expr>> + Clone),
        modulus: &(impl Into<Polynomial<AB::Expr>> + Clone),
        op: FieldOperation,
        is_real: impl Into<AB::Expr> + Clone,
    ) where
        V: Into<AB::Expr>,
        Limbs<V, P::Limbs>: Copy,
    {
        let p_a_param: Polynomial<AB::Expr> = (a).clone().into();
        let p_b: Polynomial<AB::Expr> = (b).clone().into();

        let (p_a, p_result): (Polynomial<_>, Polynomial<_>) = match op {
            FieldOperation::Add | FieldOperation::Mul | FieldOperation::Sqr => {
                (p_a_param, self.result.into())
            }
            FieldOperation::Sub | FieldOperation::Div => (self.result.into(), p_a_param),
        };
        let p_op: Polynomial<<AB as AirBuilder>::Expr> = match op {
            FieldOperation::Add | FieldOperation::Sub => p_a + p_b,
            FieldOperation::Mul | FieldOperation::Div => p_a * p_b,
            FieldOperation::Sqr => p_a.square(),
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
        let p_carry: Polynomial<<AB as AirBuilder>::Expr> = self.carry.into();
        let p_op_minus_result: Polynomial<AB::Expr> = p_op - &p_result;
        let p_vanishing = p_op_minus_result - &(&p_carry * &p_modulus);
        let p_witness = self.witness.0.iter().into();
        eval_field_operation::<AB, P>(builder, &p_vanishing, &p_witness);

        // Range checks for the result, carry, and witness columns.
        builder.slice_range_check_u8(&self.result.0, is_real.clone());
        builder.slice_range_check_u8(&self.carry.0, is_real.clone());
        builder.slice_range_check_u16(p_witness.coefficients(), is_real);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder<Var = V>>(
        &self,
        builder: &mut AB,
        a: &(impl Into<Polynomial<AB::Expr>> + Clone),
        b: &(impl Into<Polynomial<AB::Expr>> + Clone),
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

// ============================================================================
// PolyAir three-phase helpers for FieldOpCols
// ============================================================================

use crate::bytes::polyair::{
    slice_u16_range_lookup, slice_u16_range_precompute_lc, slice_u8_range_lookup,
    slice_u8_range_precompute_lc,
};

pub const fn field_op_num_interactions<P: FieldParameters>() -> usize {
    P::NB_LIMBS + P::NB_WITNESS_LIMBS
}

pub fn field_op_precompute_lc<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    result_limbs: &[AB::VarMaybeExt],
    carry_limbs: &[AB::VarMaybeExt],
    witness_limbs: &[AB::VarMaybeExt],
) {
    slice_u8_range_precompute_lc(builder, result_limbs);
    slice_u8_range_precompute_lc(builder, carry_limbs);
    slice_u16_range_precompute_lc(builder, witness_limbs);
}

pub fn field_op_witness_beta_from_coeffs<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &AB,
    witness_coeffs: &[AB::VarMaybeExt],
) -> AB::VarExt {
    let offset = AB::VarMaybeExt::from(AB::F::from_canonical_u32(P::WITNESS_OFFSET as u32));
    let shifted_witness =
        witness_coeffs.iter().cloned().map(|coeff| coeff - offset.clone()).collect::<Vec<_>>();
    let p_witness = Polynomial::new(shifted_witness);

    let beta_powers = builder.beta_powers();
    let zero_ext = AB::from_ef(AB::EF::zero());
    p_witness.eval_with_powers(beta_powers, zero_ext)
}

pub fn field_op_beta_from_coeffs<AB: dt_stark::air::FullAirBuilder>(
    builder: &AB,
    coeffs: &[AB::VarMaybeExt],
) -> AB::VarExt {
    let beta_powers = builder.beta_powers();
    let zero_ext = AB::from_ef(AB::EF::zero());
    Polynomial::from_coefficients(coeffs).eval_with_powers(beta_powers, zero_ext)
}

pub fn field_op_precompute_witness_beta<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    witness_coeffs: &[AB::VarMaybeExt],
) {
    let witness_beta = field_op_witness_beta_from_coeffs::<AB, P>(builder, witness_coeffs);
    builder.retain_precomputed(witness_beta);
}

pub fn field_op_modulus_beta<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &AB,
) -> AB::VarExt {
    let modulus_coeffs = P::MODULUS
        .iter()
        .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
        .collect::<Vec<_>>();
    field_op_beta_from_coeffs(builder, &modulus_coeffs)
}

pub fn field_op_beta_minus_limb_shift<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &AB,
) -> AB::VarExt {
    let beta = builder.beta_powers()[1].clone();
    let limb_shift = AB::from_ef(AB::EF::from_canonical_u32(1u32 << P::NB_BITS_PER_LIMB));
    beta - limb_shift
}

pub struct FieldOpBetaConsts<AB: dt_stark::air::FullAirBuilder> {
    pub beta_minus_limb_shift: AB::VarExt,
    pub modulus_beta: AB::VarExt,
}

impl<AB: dt_stark::air::FullAirBuilder> FieldOpBetaConsts<AB> {
    pub fn new<P: FieldParameters>(builder: &AB) -> Self {
        Self {
            beta_minus_limb_shift: field_op_beta_minus_limb_shift::<AB, P>(builder),
            modulus_beta: field_op_modulus_beta::<AB, P>(builder),
        }
    }
}

pub fn field_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    vanishing_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    beta_minus_limb_shift: AB::VarExt,
) {
    let result = vanishing_beta - witness_beta * beta_minus_limb_shift;
    builder.assert_zero_ext(result);
}

pub fn field_op_lookup<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    slice_u8_range_lookup(builder, is_real.clone(), P::NB_LIMBS / 2);
    slice_u8_range_lookup(builder, is_real.clone(), P::NB_LIMBS / 2);
    slice_u16_range_lookup(builder, is_real, P::NB_WITNESS_LIMBS);
}

pub fn field_op_mul_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let a_beta = field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta =
        field_op_beta_from_coeffs(builder, &cols.result.0.iter().cloned().collect::<Vec<_>>());
    let carry_beta =
        field_op_beta_from_coeffs(builder, &cols.carry.0.iter().cloned().collect::<Vec<_>>());
    let vanishing_beta = a_beta * b_beta - result_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

pub fn field_op_mul_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) {
    let vanishing_beta = a_beta * b_beta - result_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

pub fn field_op_mul_gate_constraints_from_betas<
    AB: dt_stark::air::FullAirBuilder,
    P: FieldParameters,
>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    cols: &FieldOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let result_beta =
        field_op_beta_from_coeffs(builder, &cols.result.0.iter().cloned().collect::<Vec<_>>());
    let carry_beta =
        field_op_beta_from_coeffs(builder, &cols.carry.0.iter().cloned().collect::<Vec<_>>());
    let vanishing_beta = a_beta * b_beta - result_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn field_op_variable_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    is_add: AB::VarMaybeExt,
    is_sub: AB::VarMaybeExt,
    is_mul: AB::VarMaybeExt,
    is_div: AB::VarMaybeExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let result_param_beta = result_beta.clone() * (is_add.clone() + is_mul.clone()) +
        a_beta.clone() * (is_sub.clone() + is_div.clone());
    let op_beta = (a_beta.clone() + b_beta.clone()) * is_add +
        (result_beta.clone() + b_beta.clone()) * is_sub +
        (a_beta.clone() * b_beta.clone()) * is_mul +
        (result_beta * b_beta) * is_div;
    let vanishing_beta = op_beta - result_param_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use dt_core_executor::{ExecutionRecord, Program};
    use dt_curves::params::FieldParameters;
    use dt_stark::{
        air::{DTAirBuilder, MachineAir},
        sumcheck::trace::CompressedMatrix,
        StarkGenericConfig,
    };
    use num::BigUint;
    use p3_air::BaseAir;
    use p3_field::Field;

    use super::{FieldOpCols, FieldOperation, Limbs};

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
    use rand::thread_rng;
    use std::mem::size_of;

    #[derive(AlignedBorrow, Debug, Clone)]
    pub struct TestCols<T, P: FieldParameters> {
        pub a: Limbs<T, P::Limbs>,
        pub b: Limbs<T, P::Limbs>,
        pub a_op_b: FieldOpCols<T, P>,
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
            let mut operands: Vec<(BigUint, BigUint)> = (0..num_rows - 4)
                .map(|_| {
                    let a = rng.gen_biguint(256) % &P::modulus();
                    let b = rng.gen_biguint(256) % &P::modulus();
                    (a, b)
                })
                .collect();

            // Hardcoded edge cases.
            operands.extend(vec![
                (BigUint::from(0u32), BigUint::from(1u32)),
                (BigUint::from(1u32), BigUint::from(2u32)),
                (BigUint::from(4u32), BigUint::from(5u32)),
                (BigUint::from(10u32), BigUint::from(19u32)),
            ]);

            let rows = operands
                .iter()
                .map(|(a, b)| {
                    let mut blu_events = Vec::new();
                    let mut row = [F::zero(); NUM_TEST_COLS];
                    let cols: &mut TestCols<F, P> = row.as_mut_slice().borrow_mut();
                    cols.a = P::to_limbs_field::<F, _>(a);
                    cols.b = P::to_limbs_field::<F, _>(b);
                    cols.a_op_b.populate(&mut blu_events, a, b, self.operation);
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
        for op in [FieldOperation::Add, FieldOperation::Mul, FieldOperation::Sub].iter() {
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

        for op in
            [FieldOperation::Add, FieldOperation::Sub, FieldOperation::Mul, FieldOperation::Div]
                .iter()
        {
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

pub fn field_op_add_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let a_beta = field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta =
        field_op_beta_from_coeffs(builder, &cols.result.0.iter().cloned().collect::<Vec<_>>());
    let carry_beta =
        field_op_beta_from_coeffs(builder, &cols.carry.0.iter().cloned().collect::<Vec<_>>());
    let vanishing_beta = a_beta + b_beta - result_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

pub fn field_op_sub_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let a_beta = field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta =
        field_op_beta_from_coeffs(builder, &cols.result.0.iter().cloned().collect::<Vec<_>>());
    let carry_beta =
        field_op_beta_from_coeffs(builder, &cols.carry.0.iter().cloned().collect::<Vec<_>>());
    let vanishing_beta = result_beta + b_beta - a_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

pub fn field_op_div_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let a_beta = field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta =
        field_op_beta_from_coeffs(builder, &cols.result.0.iter().cloned().collect::<Vec<_>>());
    let carry_beta =
        field_op_beta_from_coeffs(builder, &cols.carry.0.iter().cloned().collect::<Vec<_>>());
    let vanishing_beta = result_beta * b_beta - a_beta - carry_beta * consts.modulus_beta.clone();

    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

/// Fixed-Add `FieldOpCols` gate where **all four** β-evaluations are precomputed
/// by the caller. Result/carry limbs do not need to be in `reserved_poly`.
pub fn field_op_add_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) {
    let vanishing_beta = a_beta + b_beta - result_beta - carry_beta * consts.modulus_beta.clone();
    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

/// Fixed-Sub `FieldOpCols` gate, all β-evaluations precomputed.
/// Identity: `result + b = a (mod M)`.
pub fn field_op_sub_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) {
    let vanishing_beta = result_beta + b_beta - a_beta - carry_beta * consts.modulus_beta.clone();
    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}

/// Fixed-Div `FieldOpCols` gate, all β-evaluations precomputed.
/// Identity: `result * b = a (mod M)`.
pub fn field_op_div_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry_beta: AB::VarExt,
    witness_beta: AB::VarExt,
    consts: &FieldOpBetaConsts<AB>,
) {
    let vanishing_beta = result_beta * b_beta - a_beta - carry_beta * consts.modulus_beta.clone();
    field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
}
