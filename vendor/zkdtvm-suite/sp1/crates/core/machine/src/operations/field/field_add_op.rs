use std::fmt::Debug;

use crate::air::WordAirBuilder;
use num::BigUint;

use p3_air::AirBuilder;
use p3_field::Field;

use dt_core_executor::events::{ByteRecord, FieldOperation};
use dt_derive::AlignedBorrow;
use dt_stark::air::{DTAirBuilder, Polynomial};

use super::{util::compute_root_quotient_and_shift, util_air::eval_field_operation};
use dt_curves::params::{FieldParameters, Limbs, NumLimbs};

/// A set of columns to compute an emulated modular arithmetic operation.
///
/// *Safety* The input operands (a, b) (not included in the operation columns) are assumed to be
/// elements within the range `[0, 2^{P::nb_bits()})`. the result is also assumed to be within the
/// same range. Let `M = P:modulus()`. The constraints of the function [`FieldOpCols::eval`] assert
/// that:
/// * When `op` is `FieldOperation::Add`, then `result = a + b mod M`.
/// * When `op` is `FieldOperation::Sub`, then `result = a - b mod M`.
///
/// **Warning**: The constraints do not check for division by zero. The caller is responsible for
/// ensuring that the division operation is valid.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct FieldAddOpCols<T, P: FieldParameters> {
    /// The result of `a op b`, where a, b are field elements
    pub result: Limbs<T, P::Limbs>,
    pub carry: T,
    pub(crate) witness: Limbs<T, P::AddWitness>,
}

impl<F: Field, P: FieldParameters> FieldAddOpCols<F, P> {
    pub fn populate_carry_and_witness(
        &mut self,
        a: &BigUint,
        b: &BigUint,
        _op: FieldOperation,
        modulus: &BigUint,
    ) -> BigUint {
        // Here, op can only be Add.
        let p_a: Polynomial<F> = P::to_limbs_field::<F, _>(a).into();
        let p_b: Polynomial<F> = P::to_limbs_field::<F, _>(b).into();
        // let (result, carry) = match op {
        //     FieldOperation::Add => {
        //         let sum = a + b;
        //         (sum.clone() % modulus, sum.cmp(modulus).is_ge() as u8)
        //     }
        //     FieldOperation::Sub
        //     | FieldOperation::Mul
        //     | FieldOperation::Sqr
        //     | FieldOperation::Div => unreachable!(),
        // };
        let sum = a + b;
        let result = sum.clone() % modulus;
        let carry = sum.cmp(modulus).is_ge() as u8;
        // println!("a: {:?}, b: {:?} result: {:?}, carry: {}", a, b, result, carry);
        debug_assert!(&result < modulus);
        debug_assert!(carry == 0 || carry == 1);
        // match op {
        //     FieldOperation::Add => {
        //         debug_assert_eq!(modulus * carry, a + b - &result)
        //     }
        //     FieldOperation::Sub
        //     | FieldOperation::Mul
        //     | FieldOperation::Sqr
        //     | FieldOperation::Div => unreachable!(),
        // }
        debug_assert_eq!(modulus * carry, a + b - &result);
        // Here we have special logic for p_modulus because to_limbs_field only works for numbers in
        // the field, but modulus can == the field modulus so it can have 1 extra limb (ex.
        // uint256).
        let p_modulus_limbs =
            modulus.to_bytes_le().iter().map(|x| F::from_canonical_u8(*x)).collect::<Vec<F>>();
        let p_modulus: Polynomial<F> = p_modulus_limbs.iter().into();
        let p_result: Polynomial<F> = P::to_limbs_field::<F, _>(&result).into();
        // Compute the vanishing polynomial.
        // let p_op = match op {
        //     FieldOperation::Add => &p_a + &p_b,
        //     FieldOperation::Sub
        //     | FieldOperation::Mul
        //     | FieldOperation::Sqr
        //     | FieldOperation::Div => unreachable!(),
        // };
        let p_op = &p_a + &p_b;
        let p_vanishing: Polynomial<F> =
            if carry == 0 { &p_op - &p_result } else { &p_op - &p_result - &p_modulus };
        let p_witness = compute_root_quotient_and_shift(
            &p_vanishing,
            P::WITNESS_OFFSET,
            P::NB_BITS_PER_LIMB as u32,
            P::NB_ADD_WITNESS_LIMBS,
        );
        self.result = p_result.into();
        self.carry = F::from_canonical_u8(carry);
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
            FieldOperation::Mul | FieldOperation::Sqr | FieldOperation::Div => unreachable!(),
            _ => self.populate_carry_and_witness(a, b, op, modulus),
        };

        // Range checks
        record.add_u8_range_checks_field(&self.result.0);
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

impl<V: Copy, P: FieldParameters> FieldAddOpCols<V, P> {
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
            FieldOperation::Add => (p_a_param, self.result.into()),
            FieldOperation::Sub => (self.result.into(), p_a_param),
            FieldOperation::Mul | FieldOperation::Sqr | FieldOperation::Div => unreachable!(),
        };
        let p_op: Polynomial<<AB as AirBuilder>::Expr> = match op {
            FieldOperation::Add | FieldOperation::Sub => p_a + p_b,
            FieldOperation::Mul | FieldOperation::Sqr | FieldOperation::Div => unreachable!(),
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
        let has_carry: AB::Expr = self.carry.into();
        let p_op_minus_result: Polynomial<AB::Expr> = p_op - &p_result;
        let p_vanishing = p_op_minus_result - p_modulus * has_carry;
        let p_witness = self.witness.0.iter().into();
        eval_field_operation::<AB, P>(builder, &p_vanishing, &p_witness);

        // Range checks for the result, carry, and witness columns.
        builder.slice_range_check_u8(&self.result.0, is_real.clone());
        builder.assert_bool(self.carry);
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
// These helpers decompose `FieldAddOpCols::eval_with_polynomials` into the
// PolyAir four-phase model.
//
// Interaction count per FieldAddOpCols instance:
//   P::NB_LIMBS / 2 (U8Range for result) + P::NB_ADD_WITNESS_LIMBS (U16Range for witness)
//
// Note: `carry` is a single boolean — its `assert_bool` is NOT handled here.
// The chip layer collects it into BitVec (>3 booleans) or direct gate constraint (≤3).

use crate::bytes::polyair::{
    slice_u16_range_lookup, slice_u16_range_precompute_lc, slice_u8_range_lookup,
    slice_u8_range_precompute_lc,
};

/// Compute the number of interactions generated by a single `FieldAddOpCols` instance.
///
/// This equals `P::NB_LIMBS / 2 + P::NB_ADD_WITNESS_LIMBS`:
/// - `P::NB_LIMBS / 2` U8Range pairs for result limbs
/// - `P::NB_ADD_WITNESS_LIMBS` U16Range checks for witness limbs
pub const fn field_add_op_num_interactions<P: FieldParameters>() -> usize {
    P::NB_LIMBS / 2 + P::NB_ADD_WITNESS_LIMBS
}

/// Precompute lookup denominators for a `FieldAddOpCols` instance.
///
/// Emits range check denominators in the same order as `eval_with_polynomials`:
/// 1. U8Range pairs for `result` limbs (P::NB_LIMBS / 2 interactions)
/// 2. U16Range for `witness` limbs (P::NB_ADD_WITNESS_LIMBS interactions)
///
/// Chips that need `witness(beta)` must retain it explicitly after all lookup
/// denominators for the row have been emitted.
pub fn field_add_op_precompute_lc<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    result_limbs: &[AB::VarMaybeExt],
    witness_limbs: &[AB::VarMaybeExt],
) {
    // U8Range for result limbs (pairs)
    slice_u8_range_precompute_lc(builder, result_limbs);
    // U16Range for witness limbs
    slice_u16_range_precompute_lc(builder, witness_limbs);
}

/// Declare multiplicities for a `FieldAddOpCols` instance's range check lookups.
///
/// Emits `send` calls in the same order as `field_add_op_precompute_lc`:
/// 1. U8Range pairs for result (P::NB_LIMBS / 2 sends)
/// 2. U16Range for witness (P::NB_ADD_WITNESS_LIMBS sends)
pub fn field_add_op_lookup<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    // U8Range for result limbs
    slice_u8_range_lookup(builder, is_real.clone(), P::NB_LIMBS / 2);
    // U16Range for witness limbs
    slice_u16_range_lookup(builder, is_real, P::NB_ADD_WITNESS_LIMBS);
}

/// Gate constraints for `FieldAddOpCols::eval_variable` (add/sub variable selection).
///
/// Mirrors the original AIR's `eval_variable` → `eval_with_polynomials` → `eval_field_operation`
/// chain using `Polynomial` arithmetic and the shared `field_op_gate_constraints` helper.
///
/// Parameters:
/// - `a_limbs`: limbs of the first operand (prev_value from memory)
/// - `b_limbs`: limbs of the second operand (value from memory)
/// - `result`: result limbs
/// - `carry`: carry bit
/// - `is_add`: selector — 1 for addition, 0 for subtraction
///
/// NOTE: `carry` boolean enforcement is included here via `assert_zero(carry * (1 - carry))`.
pub fn field_add_op_variable_gate_constraints<
    AB: dt_stark::air::FullAirBuilder,
    P: FieldParameters,
>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    result: &Limbs<AB::VarMaybeExt, <P as NumLimbs>::Limbs>,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    is_add: AB::VarMaybeExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let is_sub = one.clone() - is_add.clone();
    let a_beta = super::field_op::field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = super::field_op::field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &result.0.iter().cloned().collect::<Vec<_>>(),
    );

    let op_beta = a_beta.clone() * is_add.clone() + result_beta.clone() * is_sub.clone() + b_beta;
    let result_param_beta = result_beta * is_add + a_beta * is_sub;
    let vanishing_beta = op_beta - result_param_beta - consts.modulus_beta.clone() * carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );

    // carry boolean
    builder.assert_zero(carry.clone() * (one - carry));
}

/// Gate constraints for `FieldAddOpCols::eval_variable` (add/sub variable selection),
/// but `a_beta / b_beta` are pre-computed β-evaluations passed in by the caller.
///
/// This avoids reading operand limbs from reserved_poly during eval — the caller
/// computes `a_beta = Σ a_i β^i` and `b_beta = Σ b_i β^i` in `precompute_lc` and
/// passes them via the precomputed slice.  `result_beta` and `modulus_beta` are still
/// computed internally because result limbs are needed by other gates (e.g. LT).
pub fn field_add_op_variable_gate_constraints_from_betas<
    AB: dt_stark::air::FullAirBuilder,
    P: FieldParameters,
>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result: &Limbs<AB::VarMaybeExt, <P as NumLimbs>::Limbs>,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    is_add: AB::VarMaybeExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let is_sub = one.clone() - is_add.clone();
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &result.0.iter().cloned().collect::<Vec<_>>(),
    );
    let op_beta = a_beta.clone() * is_add.clone() + result_beta.clone() * is_sub.clone() + b_beta;
    let result_param_beta = result_beta * is_add + a_beta * is_sub;
    let vanishing_beta = op_beta - result_param_beta - consts.modulus_beta.clone() * carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
    builder.assert_zero(carry.clone() * (one - carry));
}

/// Same as `field_add_op_variable_gate_constraints_from_betas`, but **`result_beta`**
/// is also precomputed by the caller (typically when result limbs are not in
/// `reserved_poly`). `carry` is still passed as a single-limb value because it is
/// used in the boolean assertion `carry * (1 - carry) = 0`.
pub fn field_add_op_variable_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    is_add: AB::VarMaybeExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let is_sub = one.clone() - is_add.clone();
    let op_beta = a_beta.clone() * is_add.clone() + result_beta.clone() * is_sub.clone() + b_beta;
    let result_param_beta = result_beta * is_add + a_beta * is_sub;
    let vanishing_beta = op_beta - result_param_beta - consts.modulus_beta.clone() * carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
    builder.assert_zero(carry.clone() * (one - carry));
}

/// Gate constraints for a `FieldAddOpCols` instance with a **fixed Add** operation,
/// but `a_beta / b_beta` are pre-computed β-evaluations passed in by the caller.
///
/// This avoids reading operand limbs from reserved_poly during eval — the caller
/// computes `a_beta = Σ a_i β^i` and `b_beta = Σ b_i β^i` in `precompute_lc` and
/// passes them via the precomputed slice.  `result_beta` and `modulus_beta` are still
/// computed internally because result limbs are needed by other gates (e.g. LT).
pub fn field_add_op_add_gate_constraints_from_betas<
    AB: dt_stark::air::FullAirBuilder,
    P: FieldParameters,
>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    cols: &FieldAddOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &cols.result.0.iter().cloned().collect::<Vec<_>>(),
    );

    let vanishing_beta =
        a_beta + b_beta - result_beta - consts.modulus_beta.clone() * cols.carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );

    // carry boolean
    builder.assert_zero(cols.carry.clone() * (one - cols.carry.clone()));
}

/// Same as `field_add_op_add_gate_constraints_from_betas`, but `result_beta` is also
/// precomputed (typical when the result limbs are not in `reserved_poly`).
/// `carry` is still passed as a limb for the boolean assertion.
pub fn field_add_op_add_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let vanishing_beta =
        a_beta + b_beta - result_beta - consts.modulus_beta.clone() * carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
    builder.assert_zero(carry.clone() * (one - carry));
}

/// Gate constraints for a `FieldAddOpCols` instance with a **fixed Sub** operation,
/// but `a_beta / b_beta` are pre-computed β-evaluations passed in by the caller.
///
/// For sub, a - b = result ⟺ result + b = a (mod M).
/// `vanishing = (result + b) - a - carry * modulus`.
pub fn field_add_op_sub_gate_constraints_from_betas<
    AB: dt_stark::air::FullAirBuilder,
    P: FieldParameters,
>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    cols: &FieldAddOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &cols.result.0.iter().cloned().collect::<Vec<_>>(),
    );

    let vanishing_beta =
        result_beta + b_beta - a_beta - consts.modulus_beta.clone() * cols.carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );

    // carry boolean
    builder.assert_zero(cols.carry.clone() * (one - cols.carry.clone()));
}

/// Same as `field_add_op_sub_gate_constraints_from_betas`, but `result_beta` is also
/// precomputed. `carry` is still passed as a limb for the boolean assertion.
pub fn field_add_op_sub_gate_constraints_all_betas<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    a_beta: AB::VarExt,
    b_beta: AB::VarExt,
    result_beta: AB::VarExt,
    carry: AB::VarMaybeExt,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let vanishing_beta =
        result_beta + b_beta - a_beta - consts.modulus_beta.clone() * carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );
    builder.assert_zero(carry.clone() * (one - carry));
}

/// Gate constraints for a `FieldAddOpCols` instance with a **fixed Add** operation.
///
/// Mirrors `eval_with_modulus` for `FieldOperation::Add`:
///   `p_op = p_a + p_b`, `p_result = cols.result`,
///   `vanishing = p_op - p_result - carry * modulus`.
///
/// The modulus is taken from `P::MODULUS` at compile time.
///
/// NOTE: `carry` boolean enforcement IS included via `assert_zero(carry * (1 - carry))`.
pub fn field_add_op_add_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldAddOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let a_beta = super::field_op::field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = super::field_op::field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &cols.result.0.iter().cloned().collect::<Vec<_>>(),
    );

    let vanishing_beta =
        a_beta + b_beta - result_beta - consts.modulus_beta.clone() * cols.carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );

    // carry boolean
    builder.assert_zero(cols.carry.clone() * (one - cols.carry.clone()));
}

/// Gate constraints for a `FieldAddOpCols` instance with a **fixed Sub** operation.
///
/// Mirrors `eval_with_modulus` for `FieldOperation::Sub`:
///   For sub, a - b = result ⟺ result + b = a (mod M).
///   So `p_op = result + b`, `p_result_param = a`,
///   `vanishing = (result + b) - a - carry * modulus`.
///
/// The modulus is taken from `P::MODULUS` at compile time.
///
/// NOTE: `carry` boolean enforcement IS included via `assert_zero(carry * (1 - carry))`.
pub fn field_add_op_sub_gate_constraints<AB: dt_stark::air::FullAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    a_limbs: &[AB::VarMaybeExt],
    b_limbs: &[AB::VarMaybeExt],
    cols: &FieldAddOpCols<AB::VarMaybeExt, P>,
    witness_beta: AB::VarExt,
    consts: &super::field_op::FieldOpBetaConsts<AB>,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let a_beta = super::field_op::field_op_beta_from_coeffs(builder, a_limbs);
    let b_beta = super::field_op::field_op_beta_from_coeffs(builder, b_limbs);
    let result_beta = super::field_op::field_op_beta_from_coeffs(
        builder,
        &cols.result.0.iter().cloned().collect::<Vec<_>>(),
    );

    let vanishing_beta =
        result_beta + b_beta - a_beta - consts.modulus_beta.clone() * cols.carry.clone();
    super::field_op::field_op_gate_constraints::<AB>(
        builder,
        vanishing_beta,
        witness_beta,
        consts.beta_minus_limb_shift.clone(),
    );

    // carry boolean
    builder.assert_zero(cols.carry.clone() * (one - cols.carry.clone()));
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

    use super::{FieldAddOpCols, FieldOperation, Limbs};

    use crate::utils::{
        pad_to_power_of_two,
        uni_stark::{uni_stark_prove, uni_stark_verify},
    };
    use core::borrow::{Borrow, BorrowMut};
    use dt_core_executor::events::ByteRecord;
    use dt_curves::{
        edwards::ed25519::Ed25519BaseField,
        weierstrass::{
            bls12_381::Bls12381BaseField, bn254::Bn254BaseField, secp256k1::Secp256k1BaseField,
            secp256r1::Secp256r1BaseField,
        },
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
        pub a_op_b: FieldAddOpCols<T, P>,
    }

    pub const NUM_TEST_COLS: usize = size_of::<TestCols<u8, Secp256k1BaseField>>();
    // pub const NUM_TEST_COLS: usize = size_of::<TestCols<u8, Bls12381BaseField>>();

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
        for op in [FieldOperation::Add, FieldOperation::Sub].iter() {
            println!("op: {:?}", op);
            let chip: FieldOpChip<Secp256r1BaseField> = FieldOpChip::new(*op);
            let shard = ExecutionRecord::default();
            let _: CompressedMatrix<BabyBear> =
                chip.generate_trace(&shard, &mut ExecutionRecord::default());
        }
    }

    #[test]
    fn prove_babybear() {
        let config = BabyBearPoseidon2::new();

        for op in [FieldOperation::Add, FieldOperation::Sub].iter() {
            println!("op: {:?}", op);

            let mut challenger = config.challenger();

            let chip: FieldOpChip<Secp256r1BaseField> = FieldOpChip::new(*op);
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
