use crate::{
    adapter::instruction::InstructionCols,
    air::{MemoryAirBuilder, ProgramAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols},
};
use dt_core_executor::{
    events::{ByteRecord, MemoryAccessPosition, MemoryReadRecord},
    ALUTypeRecord,
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
pub struct ALUTypeRegisterOp<T> {
    //A register
    pub op_a: T,
    // A operand read operation
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_a_zero: T,
    pub op_b: T,
    pub op_b_access: MemoryReadCols<T>,
    pub op_c: Word<T>,
    pub op_c_access: MemoryReadCols<T>,
    pub imm_c: T,
}

impl<F: Field> ALUTypeRegisterOp<F> {
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, record: ALUTypeRecord) {
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
        self.op_c = Word::from(record.op_c);
        let imm_c = record.c.is_none();
        self.imm_c = F::from_bool(imm_c);

        if imm_c {
            // Dummy record for immediate operand: use (0,1) > (0,0) so diff_minus_one = 0 and no
            // underflow.
            let dummy_read_record = MemoryReadRecord {
                value: record.op_c,
                shard: 0,
                timestamp: 1,
                prev_shard: 0,
                prev_timestamp: 0,
            };
            // When imm_c is true, eval uses multiplicity 0 for op_c memory access,
            // so no byte lookups should be generated. Use a discarded Vec.
            let mut dummy_blu: Vec<dt_core_executor::events::ByteLookupEvent> = vec![];
            self.op_c_access.populate(dummy_read_record, &mut dummy_blu);
        } else {
            self.op_c_access.populate(record.c.unwrap().read_record().unwrap(), blu_events);
        };
    }
}
impl<T> ALUTypeRegisterOp<T> {
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
impl<F: Field> ALUTypeRegisterOp<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder + MemoryAirBuilder + ProgramAirBuilder>(
        builder: &mut AB,
        shard: AB::Expr,
        clk: AB::Expr,
        pc: AB::Expr,
        opcode: impl Into<AB::Expr> + Clone,
        cols: ALUTypeRegisterOp<AB::Var>,
        is_real: AB::Expr,
    ) {
        // builder.assert_bool(is_real.clone());
        // Assert that `imm_c` is zero if the operation is not real.
        // This is to ensure that the `op_c` read multiplicity is zero on padding rows.
        builder.when_not(is_real.clone()).assert_eq(cols.imm_c, AB::Expr::zero());
        let instruction = InstructionCols {
            opcode: opcode.clone().into(),
            op_a: cols.op_a.into(),
            op_b: Word::extend_expr::<AB>(cols.op_b.into()),
            op_c: cols.op_c.map(Into::into),
            op_a_0: cols.op_a_zero.into(),
            imm_b: AB::Expr::zero(),
            imm_c: cols.imm_c.into(),
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
            is_real.clone() - cols.imm_c,
        );
        builder.when(cols.imm_c).assert_word_eq(*cols.op_c_access.prev_value(), cols.op_c);
        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_zero).assert_word_zero(*cols.op_a_value());
        builder.when(cols.op_a_zero).assert_one(is_real.clone());
        //if syscall, check op_a in syscall
        builder.eval_memory_access(
            shard,
            clk + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            instruction.op_a,
            &cols.op_a_access,
            is_real.clone(),
        );
        // builder.slice_range_check_u8(&cols.op_a_access.access.value.0, is_real);
    }
}

/// ALUTypeRegisterOp: 1 Program + 4 op_b read + 4 op_c read + 4 op_a readwrite = 13
pub(crate) const ALU_TYPE_NUM_INTERACTIONS: usize =
    crate::program::program_polyair::PROGRAM_NUM_INTERACTIONS +
        3 * crate::memory::polyair::MEMORY_READ_NUM_INTERACTIONS;

/// PolyAir gate constraints for ALUTypeRegisterOp.
///
/// Enforces:
///   - `(1 - is_real) * imm_c = 0` — imm_c must be 0 on padding rows
///   - `op_a_zero => is_real = 1`
///   - `op_a_zero * op_a_value[i] = 0` for i=0..3 — x0 register must be 0
///   - `imm_c * (op_c_access_value[i] - op_c[i]) = 0` — when imm_c, prev_value matches op_c
pub fn alu_type_register_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    op_a_zero: AB::VarMaybeExt,
    op_a_value: [AB::VarMaybeExt; 4],
    imm_c: AB::VarMaybeExt,
    op_c_access_value: [AB::VarMaybeExt; 4],
    op_c: [AB::VarMaybeExt; 4],
    is_real: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    // imm_c must be 0 on padding rows
    builder.when_ne(is_real.clone(), one.clone()).assert_zero(imm_c.clone());
    builder.when(op_a_zero.clone()).assert_one(is_real);
    // When op_a_zero = 1, the value must be zero
    for i in 0..4 {
        builder.when(op_a_zero.clone()).assert_zero(op_a_value[i].clone());
    }
    // When imm_c = 1, op_c_access.value must match op_c (immediate consistency)
    for i in 0..4 {
        builder.when(imm_c.clone()).assert_zero(op_c_access_value[i].clone() - op_c[i].clone());
    }
}

/// Precompute denominators for ALUTypeRegisterOp interactions.
///
/// Composes: 1 program + 4 memory_read(op_b) + 4 memory_read(op_c) +
/// 4 memory_readwrite(op_a) = 13 interactions.
pub fn alu_type_register_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    pc: AB::VarMaybeExt,
    opcode: AB::VarMaybeExt,
    op_a: AB::VarMaybeExt,
    op_b: AB::VarMaybeExt,
    op_c: [AB::VarMaybeExt; 4],
    op_a_zero: AB::VarMaybeExt,
    imm_c: AB::VarMaybeExt,
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

    // #1: send_program (ALU-Type: op_b=scalar zero-extended, op_c=Word, imm_b=0, imm_c=col)
    let zero = AB::zero_maybe();
    program_precompute_lc(
        builder,
        pc,
        opcode,
        op_a.clone(),
        [op_b.clone(), zero.clone(), zero.clone(), zero],
        op_c.clone(),
        op_a_zero,
        AB::zero_maybe(), // imm_b = 0
        imm_c,
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
    memory_read_precompute_lc(builder, op_c_access, op_c[0].clone(), shard.clone(), clk_c);
    // #10-13: op_a memory read-write
    memory_readwrite_precompute_lc(builder, op_a_access, op_a_prev_value, op_a, shard, clk_a);
}

/// Declare multiplicities for ALUTypeRegisterOp's 13 interactions.
pub fn alu_type_register_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    imm_c: AB::VarMaybeExt,
) {
    use crate::{
        memory::polyair::{memory_read_lookup, memory_readwrite_lookup},
        program::program_polyair::program_lookup,
    };

    // #1: program
    program_lookup(builder, is_real.clone());
    // #2-5: op_b memory read
    memory_read_lookup(builder, is_real.clone());
    // #6-9: op_c memory read (zeroed when imm_c=1)
    memory_read_lookup(builder, is_real.clone() - imm_c);
    // #10-13: op_a memory read-write
    memory_readwrite_lookup(builder, is_real);
}
