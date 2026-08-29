use crate::{
    adapter::instruction::InstructionCols,
    air::{MemoryAirBuilder, ProgramAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols},
};
use dt_core_executor::{
    events::{ByteRecord, MemoryAccessPosition},
    AddiRecord,
};
use dt_derive::AlignedBorrow;
use dt_stark::{
    air::{DTAirBuilder, FullAirBuilder},
    Word,
};
use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};

/// Register operations for ADDI-like instructions where op_b may be a register or an immediate.
///
/// When `is_imm_b = 0` (normal ADDI): op_b is a register, `op_b_access` is populated.
/// When `is_imm_b = 1` (LUI-as-ADD): op_b is an immediate, `op_b_access` is zeroed/dummy.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddiRegisterOp<T> {
    pub op_a: T,
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_a_zero: T,
    pub op_b: T,
    pub op_b_access: MemoryReadCols<T>,
    pub op_c_imm: Word<T>,
    /// Whether op_b is an immediate (1) or a register (0).
    pub is_imm_b: T,
    /// Whether op_b is a register AND this is a real row.
    /// `is_reg_b = is_real * (1 - is_imm_b)`. Stored as a column to keep
    /// the multiplicity at degree 1 for the interaction builder.
    pub is_reg_b: T,
}

impl<F: Field> AddiRegisterOp<F> {
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, record: AddiRecord) {
        self.op_a = F::from_canonical_u8(record.op_a);
        self.op_a_access.populate(record.a, blu_events);
        self.op_a_zero = F::from_bool(record.op_a == 0);
        self.op_b = F::from_canonical_u32(record.op_b);
        self.is_imm_b = F::from_bool(record.imm_b);
        // is_reg_b = 1 when this is a real row AND b is a register (not immediate).
        // For real rows: !imm_b. For padding rows (not populated): stays 0.
        self.is_reg_b = F::from_bool(!record.imm_b);

        if let Some(b_record) = record.b {
            // Normal ADDI: op_b is a register
            self.op_b_access.populate(
                b_record.read_record().expect("op_b in addi instruction should be read operation"),
                blu_events,
            );
        }
        // When imm_b=true, op_b_access stays zeroed (default)

        self.op_c_imm = Word::from(record.op_c);
    }
}

impl<T> AddiRegisterOp<T> {
    pub fn prev_op_a_value(&self) -> &Word<T> {
        &self.op_a_access.prev_value
    }
    pub fn op_a_value(&self) -> &Word<T> {
        self.op_a_access.value()
    }

    /// Returns the op_b value as a Word.
    /// When is_imm_b=0, this is the register value from op_b_access.
    /// When is_imm_b=1, this is the immediate value from op_b (extended to Word).
    /// The constraint logic must select the correct source based on is_imm_b.
    pub fn op_b_value(&self) -> &Word<T> {
        // In trace generation, this returns the register value. The AIR constraints
        // must handle the is_imm_b=1 case separately.
        self.op_b_access.value()
    }

    pub fn op_c_value(&self) -> &Word<T> {
        &self.op_c_imm
    }
}

impl<F: Field> AddiRegisterOp<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: DTAirBuilder + MemoryAirBuilder + ProgramAirBuilder>(
        builder: &mut AB,
        shard: AB::Expr,
        clk: AB::Expr,
        pc: AB::Expr,
        opcode: impl Into<AB::Expr> + Clone,
        cols: AddiRegisterOp<AB::Var>,
        is_real: AB::Expr,
    ) {
        // is_imm_b is boolean
        builder.when(is_real.clone()).assert_bool(cols.is_imm_b);
        // is_reg_b is boolean
        builder.assert_bool(cols.is_reg_b);
        // Constrain: is_reg_b = is_real * (1 - is_imm_b)
        // Equivalent to: when is_real: is_reg_b = 1 - is_imm_b; when !is_real: is_reg_b = 0
        builder
            .when(is_real.clone())
            .assert_eq(cols.is_reg_b, AB::Expr::one() - cols.is_imm_b.into());
        builder.when(AB::Expr::one() - is_real.clone()).assert_zero(cols.is_reg_b);

        let instruction = InstructionCols {
            opcode: opcode.clone().into(),
            op_a: cols.op_a.into(),
            op_b: Word::extend_expr::<AB>(cols.op_b.into()),
            op_c: cols.op_c_imm.map(Into::into),
            op_a_0: cols.op_a_zero.into(),
            imm_b: cols.is_imm_b.into(),
            imm_c: AB::Expr::one(),
        };

        builder.send_program(pc, instruction.clone(), is_real.clone());

        // Only evaluate memory access for op_b when it's a register.
        // Use the column `is_reg_b` (degree 1) as multiplicity.
        builder.eval_memory_access(
            shard.clone(),
            clk.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::B as u32),
            instruction.op_b[0].clone(),
            &cols.op_b_access,
            cols.is_reg_b.into(),
        );

        // Assert that `op_a` is zero if `op_a_0` is true.
        builder.when(cols.op_a_zero).assert_word_zero(*cols.op_a_value());
        // if op_a_zero, is real
        builder.when(cols.op_a_zero).assert_one(is_real.clone());

        builder.eval_memory_access(
            shard,
            clk + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            instruction.op_a,
            &cols.op_a_access,
            is_real.clone(),
        );
    }
}
// ============================================================================

/// AddiRegisterOp: 1 Program + 4 op_b read + 4 op_a readwrite = 9
pub(crate) const ADDI_TYPE_NUM_INTERACTIONS: usize =
    crate::program::program_polyair::PROGRAM_NUM_INTERACTIONS +
        2 * crate::memory::polyair::MEMORY_READ_NUM_INTERACTIONS;

/// PolyAir gate constraints for AddiRegisterOp.
///
/// Enforces:
///   - `is_real * is_imm_b * (1 - is_imm_b) = 0` — is_imm_b boolean on real rows
///   - `is_reg_b * (1 - is_reg_b) = 0` — is_reg_b boolean always
///   - `is_real * (is_reg_b - (1 - is_imm_b)) = 0` — linkage on real rows
///   - `(1 - is_real) * is_reg_b = 0` — is_reg_b = 0 on padding
///   - `op_a_zero => is_real = 1`
///   - `op_a_zero * op_a_value[i] = 0` for i=0..3
pub fn addi_register_op_gate_constraints<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    op_a_zero: AB::VarMaybeExt,
    op_a_value: [AB::VarMaybeExt; 4],
    is_imm_b: AB::VarMaybeExt,
    is_reg_b: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) {
    let one = AB::one_maybe();
    // is_imm_b boolean when real
    builder.when(is_real.clone()).assert_zero(is_imm_b.clone() * (one.clone() - is_imm_b.clone()));
    // is_reg_b boolean always
    builder.assert_zero(is_reg_b.clone() * (one.clone() - is_reg_b.clone()));
    // linkage: when is_real: is_reg_b = 1 - is_imm_b
    builder.when(is_real.clone()).assert_zero(is_reg_b.clone() - (one.clone() - is_imm_b));
    // padding: is_reg_b = 0
    builder.when_ne(is_real.clone(), one.clone()).assert_zero(is_reg_b);
    builder.when(op_a_zero.clone()).assert_one(is_real);
    // When op_a_zero = 1, the value must be zero
    for i in 0..4 {
        builder.when(op_a_zero.clone()).assert_zero(op_a_value[i].clone());
    }
}

/// Precompute denominators for AddiRegisterOp interactions.
///
/// Composes: 1 program + 4 memory_read(op_b) + 4 memory_readwrite(op_a) = 9 interactions.
/// No op_c memory access (op_c is always an immediate).
pub fn addi_register_op_precompute_lc<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    pc: AB::VarMaybeExt,
    opcode: AB::VarMaybeExt,
    op_a: AB::VarMaybeExt,
    op_b: AB::VarMaybeExt,
    op_c_imm: [AB::VarMaybeExt; 4],
    op_a_zero: AB::VarMaybeExt,
    is_imm_b: AB::VarMaybeExt,
    op_b_access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
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

    // #1: send_program (ADDI-Type: op_b=scalar, op_c=Word imm, imm_b=is_imm_b, imm_c=1)
    let zero = AB::zero_maybe();
    let one = AB::VarMaybeExt::from(AB::F::one());
    program_precompute_lc(
        builder,
        pc,
        opcode,
        op_a.clone(),
        [op_b.clone(), zero.clone(), zero.clone(), zero],
        op_c_imm,
        op_a_zero,
        is_imm_b,
        one, // imm_c = 1
    );

    // Compute per-access clk offsets
    let clk_b = clk.clone() +
        AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::B as u8));
    let clk_a =
        clk + AB::VarMaybeExt::from(AB::F::from_canonical_u8(MemoryAccessPosition::A as u8));

    // #2-5: op_b memory read
    memory_read_precompute_lc(builder, op_b_access, op_b, shard.clone(), clk_b);
    // #6-9: op_a memory read-write
    memory_readwrite_precompute_lc(builder, op_a_access, op_a_prev_value, op_a, shard, clk_a);
}

/// Declare multiplicities for AddiRegisterOp's 9 interactions.
pub fn addi_register_op_lookup<AB: dt_stark::air::FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    is_reg_b: AB::VarMaybeExt,
) {
    use crate::{
        memory::polyair::{memory_read_lookup, memory_readwrite_lookup},
        program::program_polyair::program_lookup,
    };

    // #1: program
    program_lookup(builder, is_real.clone());
    // #2-5: op_b memory read (conditional on is_reg_b)
    memory_read_lookup(builder, is_reg_b);
    // #6-9: op_a memory read-write
    memory_readwrite_lookup(builder, is_real);
}
