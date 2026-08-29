use crate::{
    adapter::instruction::InstructionCols,
    air::{MemoryAirBuilder, ProgramAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols},
};
use dt_core_executor::{
    events::{ByteRecord, MemoryAccessPosition},
    RTypeRecord,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use serde::{Deserialize, Serialize};
/// A set of columns to read operations with op_a, op_b, op_c being registers.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct RTypeRegisterOp<T> {
    //A register
    pub op_a: T,
    // A operand read operation
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_a_zero: T,
    pub op_b: T,
    pub op_b_access: MemoryReadCols<T>,
    pub op_c: T,
    pub op_c_access: MemoryReadCols<T>,
}

impl<F: Field> RTypeRegisterOp<F> {
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, record: RTypeRecord) {
        self.op_a = F::from_canonical_u8(record.op_a);
        self.op_a_access.populate(record.a, blu_events);
        self.op_a_zero = F::from_bool(record.op_a == 0);
        self.op_b = F::from_canonical_u32(record.op_b);
        self.op_b_access.populate(
            record
                .b
                .read_record()
                .expect("op_b_access in rtype instruction,should be read operation"),
            blu_events,
        );
        self.op_c = F::from_canonical_u32(record.op_c);
        self.op_c_access.populate(
            record
                .c
                .read_record()
                .expect("op_c_access in rtype instruction,should be read operation"),
            blu_events,
        );
    }
    pub fn dummy(a: u32, b: u32, c: u32) -> Self {
        Self {
            op_a: F::from_canonical_u8(0),
            op_a_access: MemoryReadWriteCols::dummy(a),
            op_a_zero: F::from_bool(false),
            op_b: F::from_canonical_u32(0),
            op_b_access: MemoryReadCols::dummy(b),
            op_c: F::from_canonical_u32(0),
            op_c_access: MemoryReadCols::dummy(c),
        }
    }
}
impl<T> RTypeRegisterOp<T> {
    pub fn prev_op_a_value(&self) -> &Word<T> {
        &self.op_a_access.prev_value
    }
    pub fn op_a_value(&self) -> &Word<T> {
        self.op_a_access.value()
    }

    pub fn op_b_value(&self) -> &Word<T> {
        self.op_b_access.value()
    }

    pub fn op_c_value(&self) -> &Word<T> {
        self.op_c_access.value()
    }
}
impl<F: Field> RTypeRegisterOp<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder + MemoryAirBuilder + ProgramAirBuilder>(
        builder: &mut AB,
        shard: AB::Expr,
        clk: AB::Expr,
        pc: AB::Expr,
        opcode: impl Into<AB::Expr> + Clone,
        cols: RTypeRegisterOp<AB::Var>,
        //if is not real, is not syscall
        is_syscall: AB::Expr,
        is_real: AB::Expr,
    ) {
        // builder.assert_bool(is_real.clone());
        // builder.assert_bool(is_syscall.clone());
        builder.when(AB::Expr::one() - is_real.clone()).assert_zero(is_syscall.clone());

        let instruction = InstructionCols {
            opcode: opcode.clone().into(),
            op_a: cols.op_a.into(),
            op_b: Word::extend_expr::<AB>(cols.op_b.into()),
            op_c: Word::extend_expr::<AB>(cols.op_c.into()),
            op_a_0: cols.op_a_zero.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };

        builder.send_program(pc, instruction.clone(), is_real.clone());

        builder.eval_memory_access(
            shard.clone(),
            clk.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::B as u32),
            instruction.op_b[0].clone(),
            &cols.op_b_access,
            is_real.clone(),
        );

        builder.eval_memory_access(
            shard.clone(),
            clk.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::C as u32),
            instruction.op_c[0].clone(),
            &cols.op_c_access,
            is_real.clone(),
        );

        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_zero).assert_word_zero(*cols.op_a_value());
        builder.when(cols.op_a_zero).assert_one(is_real.clone());
        //if syscall, check op_a in syscall
        builder.eval_memory_access(
            shard,
            clk + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            instruction.op_a,
            &cols.op_a_access,
            is_real.clone() - is_syscall,
        );
        // builder.slice_range_check_u8(&cols.op_a_access.access.value.0, is_real);
    }
}
// ============================================================================

/// RTypeRegisterOp: 1 Program + 3 × 4 MemoryAccess = 13
pub(crate) const RTYPE_REGISTER_OP_NUM_INTERACTIONS: usize =
    crate::program::program_polyair::PROGRAM_NUM_INTERACTIONS +
        3 * crate::memory::polyair::MEMORY_READ_NUM_INTERACTIONS;

/// PolyAir gate constraints for RTypeRegisterOp.
///
/// Enforces:
///   - `op_a_zero => is_real = 1`
///   - `op_a_zero * op_a_value[i] = 0` for i=0..3 — x0 register must be 0
pub fn rtype_register_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    op_a_zero: AB::VarMaybeExt,
    op_a_value: [AB::VarMaybeExt; 4],
    is_real: AB::VarMaybeExt,
) {
    builder.when(op_a_zero.clone()).assert_one(is_real);
    // When op_a_zero = 1, the value must be zero
    for i in 0..4 {
        builder.when(op_a_zero.clone()).assert_zero(op_a_value[i].clone());
    }
}

/// Precompute denominators for RTypeRegisterOp interactions.
///
/// Composes: 1 program + 4 memory_read(op_b) + 4 memory_read(op_c) +
/// 4 memory_readwrite(op_a) = 13 interactions.
pub fn rtype_register_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    pc: AB::VarMaybeExt,
    opcode: AB::VarMaybeExt,
    op_a: AB::VarMaybeExt,
    op_b: AB::VarMaybeExt,
    op_c: AB::VarMaybeExt,
    op_a_zero: AB::VarMaybeExt,
    op_b_access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
    op_c_access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
    op_a_access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
    op_a_prev_value: &dt_stark::Word<AB::VarMaybeExt>,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
) {
    use dt_core_executor::events::MemoryAccessPosition;
    use p3_field::AbstractField;

    use crate::{
        memory::polyair::{memory_read_precompute_lc, memory_readwrite_precompute_lc},
        program::program_polyair::program_precompute_lc,
    };

    // #1: send_program (R-Type: scalar op_b/op_c zero-extended, imm_b=0, imm_c=0)
    let zero = AB::zero_maybe();
    program_precompute_lc(
        builder,
        pc,
        opcode,
        op_a.clone(),
        [op_b.clone(), zero.clone(), zero.clone(), zero.clone()],
        [op_c.clone(), zero.clone(), zero.clone(), zero.clone()],
        op_a_zero,
        zero.clone(), // imm_b = 0
        zero,         // imm_c = 0
    );

    // Compute per-access clk offsets
    let clk_b = clk.clone() +
        AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::B as u8));
    let clk_c = clk.clone() +
        AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::C as u8));
    let clk_a =
        clk + AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8));

    // #2-5: op_b memory read
    memory_read_precompute_lc(builder, op_b_access, op_b, shard.clone(), clk_b);
    // #6-9: op_c memory read
    memory_read_precompute_lc(builder, op_c_access, op_c, shard.clone(), clk_c);
    // #10-13: op_a memory read-write
    memory_readwrite_precompute_lc(builder, op_a_access, op_a_prev_value, op_a, shard, clk_a);
}

/// Declare multiplicities for RTypeRegisterOp's 13 interactions.
pub fn rtype_register_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    use crate::{
        memory::polyair::{memory_read_lookup, memory_readwrite_lookup},
        program::program_polyair::program_lookup,
    };

    // #1: program
    program_lookup(builder, is_real.clone());
    // #2-5: op_b memory read
    memory_read_lookup(builder, is_real.clone());
    // #6-9: op_c memory read
    memory_read_lookup(builder, is_real.clone());
    // #10-13: op_a memory read-write
    memory_readwrite_lookup(builder, is_real);
}
