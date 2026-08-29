use crate::{
    adapter::instruction::InstructionCols,
    air::{MemoryAirBuilder, ProgramAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols},
};
use dt_core_executor::{
    events::{ByteRecord, MemoryAccessPosition},
    JTypeRecord,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JTypeRegisterOp<T> {
    pub op_a: T,
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_a_zero: T,
    pub op_b_imm: Word<T>,
    pub op_c_imm: Word<T>,
}
impl<F: Field> JTypeRegisterOp<F> {
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, record: JTypeRecord) {
        self.op_a = F::from_canonical_u8(record.op_a);
        self.op_a_access.populate(record.a, blu_events);
        self.op_a_zero = F::from_bool(record.op_a == 0);
        self.op_b_imm = Word::from(record.op_b);
        self.op_c_imm = Word::from(record.op_c);
    }
}
impl<T> JTypeRegisterOp<T> {
    pub fn prev_op_a_value(&self) -> &Word<T> {
        &self.op_a_access.prev_value
    }
    pub fn op_a_value(&self) -> &Word<T> {
        self.op_a_access.value()
    }

    pub fn op_b_value(&self) -> &Word<T> {
        &self.op_b_imm
    }

    pub fn op_c_value(&self) -> &Word<T> {
        &self.op_c_imm
    }
}
impl<F: Field> JTypeRegisterOp<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder + MemoryAirBuilder + ProgramAirBuilder>(
        builder: &mut AB,
        shard: AB::Expr,
        clk: AB::Expr,
        pc: AB::Expr,
        opcode: impl Into<AB::Expr> + Clone,
        cols: JTypeRegisterOp<AB::Var>,
        is_real: AB::Expr,
    ) {
        // builder.assert_bool(is_real.clone());

        let instruction = InstructionCols {
            opcode: opcode.clone().into(),
            op_a: cols.op_a.into(),
            op_b: cols.op_b_imm.map(Into::into),
            op_c: cols.op_c_imm.map(Into::into),
            op_a_0: cols.op_a_zero.into(),
            imm_b: AB::Expr::one(),
            imm_c: AB::Expr::one(),
        };

        builder.send_program(pc, instruction.clone(), is_real.clone());

        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_zero).assert_word_zero(*cols.op_a_value());
        builder.when(cols.op_a_zero).assert_one(is_real.clone());
        //no syscall case
        builder.eval_memory_access(
            shard,
            clk + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            instruction.op_a,
            &cols.op_a_access,
            is_real,
        );
    }
}
// ============================================================================

/// JTypeRegisterOp: 1 Program + 4 op_a readwrite = 5
/// No op_b/op_c memory access (both are immediates).
pub(crate) const JTYPE_NUM_INTERACTIONS: usize =
    crate::program::program_polyair::PROGRAM_NUM_INTERACTIONS +
        crate::memory::polyair::MEMORY_READ_NUM_INTERACTIONS;

/// PolyAir gate constraints for JTypeRegisterOp.
pub fn jtype_register_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    op_a_zero: AB::VarMaybeExt,
    op_a_value: [AB::VarMaybeExt; 4],
    is_real: AB::VarMaybeExt,
) {
    builder.when(op_a_zero.clone()).assert_one(is_real);
    for i in 0..4 {
        builder.when(op_a_zero.clone()).assert_zero(op_a_value[i].clone());
    }
}

/// Precompute denominators for JTypeRegisterOp interactions.
///
/// Composes: 1 program + 4 memory_readwrite(op_a) = 5 interactions.
pub fn jtype_register_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    pc: AB::VarMaybeExt,
    opcode: AB::VarMaybeExt,
    op_a: AB::VarMaybeExt,
    op_b_imm: [AB::VarMaybeExt; 4],
    op_c_imm: [AB::VarMaybeExt; 4],
    op_a_zero: AB::VarMaybeExt,
    op_a_access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
    op_a_prev_value: &dt_stark::Word<AB::VarMaybeExt>,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
) {
    use dt_core_executor::events::MemoryAccessPosition;
    use p3_field::AbstractField;

    use crate::{
        memory::polyair::memory_readwrite_precompute_lc,
        program::program_polyair::program_precompute_lc,
    };

    // #1: send_program (J-Type: op_b=Word imm, op_c=Word imm, imm_b=1, imm_c=1)
    let one = AB::VarMaybeExt::from(AB::F::one());
    program_precompute_lc(
        builder,
        pc,
        opcode,
        op_a.clone(),
        op_b_imm,
        op_c_imm,
        op_a_zero,
        one.clone(), // imm_b = 1
        one,         // imm_c = 1
    );

    // #2-5: op_a memory read-write
    let clk_a =
        clk + AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8));
    memory_readwrite_precompute_lc(builder, op_a_access, op_a_prev_value, op_a, shard, clk_a);
}

/// Declare multiplicities for JTypeRegisterOp's 5 interactions.
pub fn jtype_register_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
) {
    use crate::{
        memory::polyair::memory_readwrite_lookup, program::program_polyair::program_lookup,
    };

    // #1: program
    program_lookup(builder, is_real.clone());
    // #2-5: op_a memory read-write
    memory_readwrite_lookup(builder, is_real);
}
