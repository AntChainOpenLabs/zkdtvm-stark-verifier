use std::borrow::Borrow;

use p3_air::{Air, BaseAir, PairBuilder};
use p3_matrix::Matrix;

use dt_core_machine::operations::poseidon2_kb::{
    air::{eval_external_round, eval_internal_rounds},
    permutation::NUM_POSEIDON2_DEGREE3_COLS,
    NUM_EXTERNAL_ROUNDS, WIDTH,
};

use super::Poseidon2WideKbChip;
use crate::{
    builder::DTRecursionAirBuilder,
    chips::poseidon2_wide_kb::columns::preprocessed::Poseidon2PreprocessedColsWideKb,
};

impl<F, const DEGREE: usize> BaseAir<F> for Poseidon2WideKbChip<DEGREE> {
    fn width(&self) -> usize {
        if DEGREE == 3 {
            NUM_POSEIDON2_DEGREE3_COLS
        } else {
            panic!("KoalaBear mode only supports degree 3, got degree {DEGREE}");
        }
    }
}

impl<AB, const DEGREE: usize> Air<AB> for Poseidon2WideKbChip<DEGREE>
where
    AB: DTRecursionAirBuilder + PairBuilder,
    AB::Var: 'static,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let prepr = builder.preprocessed();
        let local_row = Self::convert::<AB::Var>(main.row_slice(0));
        let prep_local = prepr.row_slice(0);
        let prep_local: &Poseidon2PreprocessedColsWideKb<_> = (*prep_local).borrow();

        let lhs = (0..DEGREE)
            .map(|_| local_row.external_rounds_state()[0][0].into())
            .product::<AB::Expr>();
        let rhs = (0..DEGREE)
            .map(|_| local_row.external_rounds_state()[0][0].into())
            .product::<AB::Expr>();
        builder.assert_eq(lhs, rhs);

        (0..WIDTH).for_each(|i| {
            builder.send_single(
                prep_local.input[i],
                local_row.external_rounds_state()[0][i],
                prep_local.is_real_neg,
            )
        });

        (0..WIDTH).for_each(|i| {
            builder.send_single(
                prep_local.output[i].addr,
                local_row.perm_output()[i],
                prep_local.output[i].mult,
            )
        });

        for r in 0..NUM_EXTERNAL_ROUNDS {
            eval_external_round(builder, local_row.as_ref(), r);
        }

        eval_internal_rounds(builder, local_row.as_ref());
    }
}
