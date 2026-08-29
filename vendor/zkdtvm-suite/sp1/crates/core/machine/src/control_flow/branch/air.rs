use std::borrow::Borrow;

use dt_core_executor::{Opcode, DEFAULT_PC_INC};
use dt_stark::air::BaseAirBuilder;
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    adapter::{BTypeRegisterOp, CPUState},
    air::{DTCoreAirBuilder, WordAirBuilder},
    operations::{AddOperation, BabyBearWordRangeChecker, LtOperationSigned},
};

use super::{BranchChip, BranchColumns};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the helper bool columns (a_eq_b, a_gt_b,
///    a_lt_b).
/// 3. It verifier the correct values of the helper bool columns based on op_a and op_b.
impl<AB> Air<AB> for BranchChip
where
    AB: DTCoreAirBuilder,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BranchColumns<AB::Var> = (*local).borrow();
        let execution_shard: AB::Expr = builder.current_shard().into();
        let shard: AB::Expr = local.cpu_state.shard.into();
        let clk: AB::Expr = local.cpu_state.clk::<AB>();
        let a_word = local.mem_ops.op_a_value();
        let b_word = local.mem_ops.op_b_value();
        let c_word = local.mem_ops.op_c_value();
        // SAFETY: All selectors `is_beq`, `is_bne`, `is_blt`, `is_bge`, `is_bltu`, `is_bgeu` are
        // checked to be boolean. Each "real" row has exactly one selector turned on, as
        // `is_real`, the sum of the six selectors, is boolean. Therefore, the `opcode`
        // matches the corresponding opcode.
        builder.assert_bool(local.is_beq);
        builder.assert_bool(local.is_bne);
        builder.assert_bool(local.is_blt);
        builder.assert_bool(local.is_bge);
        builder.assert_bool(local.is_bltu);
        builder.assert_bool(local.is_bgeu);
        builder.assert_bool(local.is_branching);
        let is_real = local.is_beq +
            local.is_bne +
            local.is_blt +
            local.is_bge +
            local.is_bltu +
            local.is_bgeu;
        builder.assert_bool(is_real.clone());
        //cpu state
        CPUState::<AB::F>::eval(
            builder,
            local.cpu_state,
            local.add_op.value.reduce::<AB>(),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            is_real.clone(),
            execution_shard,
        );
        let opcode = local.is_beq * Opcode::BEQ.as_field::<AB::F>() +
            local.is_bne * Opcode::BNE.as_field::<AB::F>() +
            local.is_blt * Opcode::BLT.as_field::<AB::F>() +
            local.is_bge * Opcode::BGE.as_field::<AB::F>() +
            local.is_bltu * Opcode::BLTU.as_field::<AB::F>() +
            local.is_bgeu * Opcode::BGEU.as_field::<AB::F>();
        BTypeRegisterOp::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.cpu_state.pc.into(),
            opcode,
            local.mem_ops,
            is_real.clone(),
        );
        builder.assert_eq(local.cpu_state.pc, local.pc.reduce::<AB>());

        // Evaluate program counter constraints.
        {
            BabyBearWordRangeChecker::<AB::F>::range_check(
                builder,
                local.pc,
                local.pc_range_checker,
                is_real.clone(),
            );
            BabyBearWordRangeChecker::<AB::F>::range_check(
                builder,
                local.add_op.value,
                local.next_pc_range_checker,
                is_real.clone(),
            );

            AddOperation::<AB::F>::eval(
                builder,
                local.pc,
                *c_word,
                local.add_op,
                local.is_branching.into(),
            );
            // not_branching = is_real - is_branching (eliminated column, use expression).
            let not_branching: AB::Expr = is_real.clone() - local.is_branching;
            builder.when(not_branching).assert_eq(
                local.add_op.value.reduce::<AB>(),
                local.cpu_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            );
        }

        // Evaluate branching value constraints.
        {
            // When the opcode is BEQ and we are branching, assert that a_eq_b is true.
            builder.when(local.is_beq * local.is_branching).assert_one(local.a_eq_b);

            // When the opcode is BEQ and we are not branching, assert that either a_gt_b or a_lt_b
            // is true.
            builder
                .when(local.is_beq)
                .when_not(local.is_branching)
                .assert_one(local.a_gt_b + local.a_lt_b);

            // When the opcode is BNE and we are branching, assert that either a_gt_b or a_lt_b is
            // true.
            builder.when(local.is_bne * local.is_branching).assert_one(local.a_gt_b + local.a_lt_b);

            // When the opcode is BNE and we are not branching, assert that a_eq_b is true.
            builder.when(local.is_bne).when_not(local.is_branching).assert_one(local.a_eq_b);

            // When the opcode is BLT or BLTU and we are branching, assert that a_lt_b is true.
            builder
                .when((local.is_blt + local.is_bltu) * local.is_branching)
                .assert_one(local.a_lt_b);

            // When the opcode is BLT or BLTU and we are not branching, assert that either a_eq_b
            // or a_gt_b is true.
            builder
                .when(local.is_blt + local.is_bltu)
                .when_not(local.is_branching)
                .assert_one(local.a_eq_b + local.a_gt_b);

            // When the opcode is BGE or BGEU and we are branching, assert that a_gt_b is true.
            builder
                .when((local.is_bge + local.is_bgeu) * local.is_branching)
                .assert_one(local.a_gt_b + local.a_eq_b);

            // When the opcode is BGE or BGEU and we are not branching, assert that either a_eq_b
            // or a_lt_b is true.
            builder
                .when(local.is_bge + local.is_bgeu)
                .when_not(local.is_branching)
                .assert_one(local.a_lt_b);
        }

        // When it's a branch instruction and a_eq_b, assert that a == b.
        builder.when(local.a_eq_b).assert_word_eq(*a_word, *b_word);

        let use_signed_comp = local.is_blt + local.is_bge;
        LtOperationSigned::<AB::F>::eval(
            builder,
            (*a_word).map(Into::into),
            (*b_word).map(Into::into),
            local.compare_operation,
            use_signed_comp.clone(),
            is_real.clone(),
        );
        let is_eq = AB::Expr::one() -
            (local.compare_operation.result.byte_flags[0] +
                local.compare_operation.result.byte_flags[1] +
                local.compare_operation.result.byte_flags[2] +
                local.compare_operation.result.byte_flags[3]);
        let is_less = local.compare_operation.result.result;
        builder.when(is_real.clone()).assert_eq(local.a_eq_b, is_eq.clone());
        builder.when(is_real.clone()).assert_eq(local.a_lt_b, is_less);
        builder.assert_eq(is_real.clone(), local.a_eq_b + local.a_lt_b + local.a_gt_b);
        // No need to constraint a_gt_b, when is_real, a_gt_b = 1 - a_eq_b - a_lt_b
    }
}
