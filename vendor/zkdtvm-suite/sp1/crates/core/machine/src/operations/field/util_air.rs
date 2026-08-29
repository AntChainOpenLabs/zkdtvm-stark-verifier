use dt_curves::params::FieldParameters;
use dt_stark::air::{DTAirBuilder, Polynomial};
use p3_field::AbstractField;

pub fn eval_field_operation<AB: DTAirBuilder, P: FieldParameters>(
    builder: &mut AB,
    p_vanishing: &Polynomial<AB::Expr>,
    p_witness: &Polynomial<AB::Expr>,
) {
    // Reconstruct and shift back the witness polynomial
    let limb: AB::Expr = AB::F::from_canonical_u32(2u32.pow(P::NB_BITS_PER_LIMB as u32)).into();

    // Shift down the witness polynomial. Shifting is needed to range check that each
    // coefficient w_i of the witness polynomial satisfies |w_i| < 2^WITNESS_OFFSET.
    let offset: AB::Expr = AB::F::from_canonical_u32(P::WITNESS_OFFSET as u32).into();
    let len = p_witness.coefficients().len();
    let p_witness = p_witness - &Polynomial::new(vec![offset; len]);

    // Multiply by (x-2^NB_BITS_PER_LIMB) and make the constraint
    let root_monomial = Polynomial::new(vec![-limb, AB::F::one().into()]);

    let constraints = p_vanishing - &(p_witness * root_monomial);
    for constr in constraints.as_coefficients() {
        builder.assert_zero(constr);
    }
}
