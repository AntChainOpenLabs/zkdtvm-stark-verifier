use dt_derive::AlignedBorrow;
use dt_stark::Word;
use std::mem::size_of;

use crate::{
    adapter::{BTypeRegisterOp, CPUState},
    operations::{AddOperation, BabyBearWordRangeChecker, LtOperationSigned},
};

pub const NUM_BRANCH_COLS: usize = size_of::<BranchColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct BranchColumns<T> {
    pub cpu_state: CPUState<T>,
    ///op_a <- rs1, op_b <- rs2, op_c <- imm , if cond(op_a,op_b) next_pc = pc + op_c_value else
    /// next_pc = pc + 4
    pub mem_ops: BTypeRegisterOp<T>,

    /// The current program counter.
    pub pc: Word<T>,
    pub pc_range_checker: BabyBearWordRangeChecker<T>,
    /// pc + op_c_value (if cond && real)
    pub add_op: AddOperation<T>,
    /// next pc range checking: verifies add_op.value is within field range
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,

    /// Branch Instructions.
    pub is_beq: T,
    pub is_bne: T,
    pub is_blt: T,
    pub is_bge: T,
    pub is_bltu: T,
    pub is_bgeu: T,

    /// The is_branching column is equal to:
    ///
    /// > is_beq & a_eq_b ||
    /// > is_bne & (a_lt_b | a_gt_b) ||
    /// > (is_blt | is_bltu) & a_lt_b ||
    /// > (is_bge | is_bgeu) & (a_eq_b | a_gt_b)
    pub is_branching: T,

    /// Whether a equals b.
    pub a_eq_b: T,

    /// Whether a is greater than b.
    pub a_gt_b: T,

    /// Whether a is less than b.
    pub a_lt_b: T,
    /// comparison op
    pub compare_operation: LtOperationSigned<T>,
}
