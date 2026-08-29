//! PolyAir-optimized FullAir implementation for MulChip.
//!
//! This module provides a `FullAir` implementation that maps the real SP1 `MulCols`
//! layout to PolyAir's four-phase constraint model. It coexists with the original
//! `Air<AB>` implementation in `mod.rs` without modifying it.
//!
//! The implementation follows the PolyAir optimization pattern:
//! - `reserved_poly`: Full trace reserved (all columns needed for multiplication gate constraints
//!   in `eval()`)
//! - `precompute_lc`: All lookup RLC denominators are precomputed (uses `unsafe transmute` to
//!   access columns via `MulCols` struct field names instead of manual offset constants)
//! - `eval`: Gate constraints (multiplication carry chain, sign extension, result matching, shard
//!   equality, op_a_zero adapter constraints, one-hot selector)
//! - `lookup`: Send/recv multiplicity declarations
//!
//! ## Interaction Summary (32 total)
//!
//! From `MulChip::eval()` execution order:
//!
//! ### CPUState::eval() — 4 interactions
//! 1. recv(State)          — receive_state(shard, clk, pc)
//! 2. send(State)          — send_state(shard, clk+4, next_pc)
//! 3. send(Byte/U16Range)  — clk_0_16 range check
//! 4. send(Byte/BitRange)  — clk_16_28 range check (12 bits)
//!
//! ### MulOperation::eval() — 14 interactions (all mult = perform_calc)
//! 5. send(Byte/MSB)      — b_msb extraction
//! 6. send(Byte/MSB)      — c_msb extraction
//! 7. send(Byte/U8Range)  — product[0:2] range check
//! 8. send(Byte/U8Range)  — product[2:4] range check
//! 9. send(Byte/U8Range)  — product[4:6] range check
//! 10. send(Byte/U8Range)  — product[6:8] range check
//! 11. send(Byte/U16Range) — carry[0] range check
//! 12. send(Byte/U16Range) — carry[1] range check
//! 13. send(Byte/U16Range) — carry[2] range check
//! 14. send(Byte/U16Range) — carry[3] range check
//! 15. send(Byte/U16Range) — carry[4] range check
//! 16. send(Byte/U16Range) — carry[5] range check
//! 17. send(Byte/U16Range) — carry[6] range check
//! 18. send(Byte/U16Range) — carry[7] range check
//!
//! ### RTypeRegisterOp::eval() — 13 interactions
//! 19. send(Program)       — send_program
//! 20. send(Byte/U16Range) — op_b timestamp diff_16bit
//! 21. send(Byte/BitRange) — op_b timestamp diff_12bit
//! 22. send(Memory)        — op_b memory prev
//! 23. recv(Memory)        — op_b memory curr
//! 24. send(Byte/U16Range) — op_c timestamp diff_16bit
//! 25. send(Byte/BitRange) — op_c timestamp diff_12bit
//! 26. send(Memory)        — op_c memory prev
//! 27. recv(Memory)        — op_c memory curr
//! 28. send(Byte/U16Range) — op_a timestamp diff_16bit
//! 29. send(Byte/BitRange) — op_a timestamp diff_12bit
//! 30. send(Memory)        — op_a memory prev
//! 31. recv(Memory)        — op_a memory curr
//!
//! ### BitVec boolean constraint — 1 interaction (mult=is_real)
//! 32. send(BitVec) — 8 boolean expressions: [b_msb, c_msb, b_sign_extend, c_sign_extend, is_mul,
//!     is_mulh, is_mulhu, is_mulhsu] is_real: explicit bool gate in eval. sum_flags: implied by
//!     individual selector booleans + when(is_real).assert_one(sum_flags) in eval.

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{MulChip, MulCols, NUM_MUL_COLS};
use crate::{
    adapter::{
        register::r_type::{
            rtype_register_op_gate_constraints, rtype_register_op_lookup,
            rtype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    operations::{mul_op_gate_constraints, mul_op_lookup, mul_op_precompute_lc},
};

/// Public values index for execution shard.
const PV_EXECUTION_SHARD_IDX: usize = 44;

/// Maximum number of values in any single lookup interaction.
/// BitVec has 16 elements; Program send has 15.
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `MulCols<u8>` (NUM_MUL_COLS = 64).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1..4]   cpu_state.{clk_16_28, clk_0_16, pc}     ← precompute-only
//   [4]      mem_ops.op_a                            ← precompute-only
//   [5..9]   mem_ops.op_a_access.prev_value          ← precompute-only
//   [9..13]  mem_ops.op_a_access.access.value
//   [13..18] mem_ops.op_a_access.access.{ts fields}  ← precompute-only
//   [18]     mem_ops.op_a_zero
//   [19]     mem_ops.op_b                            ← precompute-only
//   [20..24] mem_ops.op_b_access.access.value
//   [24..29] mem_ops.op_b_access.access.{ts fields}  ← precompute-only
//   [29]     mem_ops.op_c                            ← precompute-only
//   [30..34] mem_ops.op_c_access.access.value
//   [34..39] mem_ops.op_c_access.access.{ts fields}  ← precompute-only
//   [39..47] mul_op.carry
//   [47..55] mul_op.product
//   [55]     mul_op.b_msb
//   [56]     mul_op.c_msb
//   [57]     mul_op.b_sign_extend
//   [58]     mul_op.c_sign_extend
//   [59]     is_mul
//   [60]     is_mulh
//   [61]     is_mulhu
//   [62]     is_mulhsu
//   [63]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_B_VALUE: usize = 20;
const COL_OP_C_ACCESS_VALUE: usize = 30;
const COL_MUL_CARRY: usize = 39;
const COL_MUL_PRODUCT: usize = 47;
const COL_MUL_B_MSB: usize = 55;
const COL_MUL_C_MSB: usize = 56;
const COL_MUL_B_SIGN_EXTEND: usize = 57;
const COL_MUL_C_SIGN_EXTEND: usize = 58;
const COL_IS_MUL: usize = 59;
const COL_IS_MULH: usize = 60;
const COL_IS_MULHU: usize = 61;
const COL_IS_MULHSU: usize = 62;
const COL_IS_REAL: usize = 63;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 39).
//
// Only fields read by `eval` or `lookup` are retained; clk/pc, op_a/op_b/op_c
// scalars, prev_value, and memory access timestamp fields are consumed in
// `precompute_lc` as lookup denominators and are not needed elsewhere.
//
//   [0]      is_real
//   [1]      cpu_state.shard
//   [2]      is_mul
//   [3]      is_mulh
//   [4]      is_mulhu
//   [5]      is_mulhsu
//   [6]      op_a_zero
//   [7..11]  op_a_access.access.value (Word)
//   [11..15] op_b_access.access.value (Word)
//   [15..19] op_c_access.access.value (Word)
//   [19..27] mul_op.product (LONG_WORD_SIZE = 8)
//   [27..35] mul_op.carry   (LONG_WORD_SIZE = 8)
//   [35]     mul_op.b_msb
//   [36]     mul_op.c_msb
//   [37]     mul_op.b_sign_extend
//   [38]     mul_op.c_sign_extend
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_CPU_SHARD: usize = 1;
const RES_IS_MUL: usize = 2;
const RES_IS_MULH: usize = 3;
const RES_IS_MULHU: usize = 4;
const RES_IS_MULHSU: usize = 5;
const RES_OP_A_ZERO: usize = 6;
const RES_OP_A_VALUE: usize = 7;
const RES_OP_B_VALUE: usize = 11;
const RES_OP_C_VALUE: usize = 15;
const RES_MUL_PRODUCT: usize = 19;
const RES_MUL_CARRY: usize = 27;
const RES_MUL_B_MSB: usize = 35;
const RES_MUL_C_MSB: usize = 36;
const RES_MUL_B_SIGN_EXTEND: usize = 37;
const RES_MUL_C_SIGN_EXTEND: usize = 38;
const RES_NUM_COLS: usize = 39;

// =============================================================================
// MulChipPolyAir wrapper type
// =============================================================================

/// PolyAir-optimized wrapper for the SP1 MulChip.
///
/// This type implements `FullAir` using the real `MulCols` column layout,
/// mapping SP1's constraint and interaction patterns to PolyAir's four-phase model.
#[derive(Default, Clone, Copy)]
pub struct MulChipPolyAir;

/// Compute the BitVec payload for a single MUL event, mirroring the
/// recurrence in `MulOperation::populate` (`operations/mul.rs:42-78`).
///
/// On `op_a_0=true` rows the base chip skips `MulOperation::populate`
/// (`alu/mul/mod.rs:178-186`), leaving `b_msb`, `c_msb`, `b_sign_extend`,
/// `c_sign_extend` as zero in the trace. The selectors are populated
/// unconditionally, so the trace's BitVec value on those rows is
/// `pack(0, 0, 0, 0, is_mul, is_mulh, is_mulhu, is_mulhsu)`.
///
/// Bit layout matches the order in `precompute_lc`:
///   bit 0: b_msb
///   bit 1: c_msb
///   bit 2: b_sign_extend
///   bit 3: c_sign_extend
///   bit 4: is_mul
///   bit 5: is_mulh
///   bit 6: is_mulhu
///   bit 7: is_mulhsu
#[inline]
fn mul_bitvec_value(b: u32, c: u32, opcode: Opcode, op_a_0: bool) -> u16 {
    use crate::operations::get_msb;

    let is_mul_bit: u16 = (opcode == Opcode::MUL) as u16;
    let is_mulh_bit: u16 = (opcode == Opcode::MULH) as u16;
    let is_mulhu_bit: u16 = (opcode == Opcode::MULHU) as u16;
    let is_mulhsu_bit: u16 = (opcode == Opcode::MULHSU) as u16;

    let (b_msb, c_msb, b_sign_ext, c_sign_ext): (u16, u16, u16, u16) = if op_a_0 {
        // MulOperation::populate skipped — trace mul_op fields are zero.
        (0, 0, 0, 0)
    } else {
        let b_msb = get_msb(b.to_le_bytes()) as u16;
        let c_msb = get_msb(c.to_le_bytes()) as u16;
        let b_sign_ext: u16 =
            if (opcode == Opcode::MULH || opcode == Opcode::MULHSU) && b_msb == 1 { 1 } else { 0 };
        let c_sign_ext: u16 = if opcode == Opcode::MULH && c_msb == 1 { 1 } else { 0 };
        (b_msb, c_msb, b_sign_ext, c_sign_ext)
    };

    b_msb |
        (c_msb << 1) |
        (b_sign_ext << 2) |
        (c_sign_ext << 3) |
        (is_mul_bit << 4) |
        (is_mulh_bit << 5) |
        (is_mulhu_bit << 6) |
        (is_mulhsu_bit << 7)
}

// =============================================================================
// FullAir implementation
// =============================================================================

impl<AB: FullAirBuilder> FullAir<AB> for MulChipPolyAir {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_IS_MUL));
        cols.push(PairCol::Main(COL_IS_MULH));
        cols.push(PairCol::Main(COL_IS_MULHU));
        cols.push(PairCol::Main(COL_IS_MULHSU));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_ACCESS_VALUE + i));
        }
        for i in 0..8 {
            cols.push(PairCol::Main(COL_MUL_PRODUCT + i));
        }
        for i in 0..8 {
            cols.push(PairCol::Main(COL_MUL_CARRY + i));
        }
        cols.push(PairCol::Main(COL_MUL_B_MSB));
        cols.push(PairCol::Main(COL_MUL_C_MSB));
        cols.push(PairCol::Main(COL_MUL_B_SIGN_EXTEND));
        cols.push(PairCol::Main(COL_MUL_C_SIGN_EXTEND));
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();

        // SAFETY: MulCols is #[repr(C)] and contains only fields of type T, [T; N],
        // or nested #[repr(C)] structs. The main trace slice has exactly NUM_MUL_COLS
        // elements.
        let local: &MulCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        // --- Derived values ---
        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();

        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        // Opcode: compute as weighted sum of one-hot flags
        let opcode = local.is_mul.clone() *
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::MUL as u32)) +
            local.is_mulh.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::MULH as u32)) +
            local.is_mulhu.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::MULHU as u32)) +
            local.is_mulhsu.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::MULHSU as u32));

        // =====================================================================
        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        // =====================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc.clone(),
            next_pc,
        );

        // =====================================================================
        // #5-18: MulOperation (2 MSB + 4 U8Range + 8 U16Range)
        // =====================================================================
        mul_op_precompute_lc(
            builder,
            local.mul_op.b_msb.clone(),
            local.mul_op.c_msb.clone(),
            local.mem_ops.op_b_access.access.value[3].clone(),
            local.mem_ops.op_c_access.access.value[3].clone(),
            &local.mul_op.product,
            &local.mul_op.carry,
        );

        // =====================================================================
        // #19-31: RTypeRegisterOp (1 program + 3×4 memory)
        // =====================================================================
        rtype_register_op_precompute_lc(
            builder,
            pc,
            opcode,
            local.mem_ops.op_a.clone(),
            local.mem_ops.op_b.clone(),
            local.mem_ops.op_c.clone(),
            local.mem_ops.op_a_zero.clone(),
            &local.mem_ops.op_b_access.access,
            &local.mem_ops.op_c_access.access,
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_a_access.prev_value,
            shard,
            clk,
        );

        // =====================================================================
        // #32: BitVec (8 bools: b/c_msb, b/c_sign_extend,
        //       is_mul, is_mulh, is_mulhu, is_mulhsu)
        // is_real removed — explicit bool gate in eval.
        // sum_flags removed — its boolean nature is implied by the 4 individual
        // selector booleans above (each in {0,1}) plus
        // when(is_real).assert_one(sum_flags) in eval, which together force
        // sum_flags = 1 on real rows. Original AIR never had assert_bool(sum_flags).
        bitvec_precompute_lc(
            builder,
            vec![
                local.mul_op.b_msb.clone(),
                local.mul_op.c_msb.clone(),
                local.mul_op.b_sign_extend.clone(),
                local.mul_op.c_sign_extend.clone(),
                local.is_mul.clone(),
                local.is_mulh.clone(),
                local.is_mulhu.clone(),
                local.is_mulhsu.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let is_mul = local[RES_IS_MUL].clone();
        let is_mulh = local[RES_IS_MULH].clone();
        let is_mulhu = local[RES_IS_MULHU].clone();
        let is_mulhsu = local[RES_IS_MULHSU].clone();

        // Boolean constraints (b_msb, c_msb, b_sign_extend, c_sign_extend,
        // is_mul, is_mulh, is_mulhu, is_mulhsu, sum_flags) are enforced by
        // BitVec #32 on real rows. `is_real` is no longer in the BitVec payload
        // (mult conditioning makes its inclusion redundant on real rows and
        // unenforced on padding); restate as an explicit gate.
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // =====================================================================
        // #1: CPUState gate constraints
        // =====================================================================
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // =====================================================================
        // #2: One-hot selector (chip-specific)
        // =====================================================================
        let sum_flags = is_mul.clone() + is_mulh.clone() + is_mulhu.clone() + is_mulhsu.clone();
        builder.when(is_real.clone()).assert_zero(sum_flags - AB::one_maybe());

        // =====================================================================
        // #3: MulOperation gate constraints (sign extension + carry chain)
        // =====================================================================
        // Original AIR gates MulOperation on perform_calc = is_real - op_a_zero,
        // so carry/product constraints are skipped when op_a_zero=1 (x0 register).
        let perform_calc = is_real.clone() - local[RES_OP_A_ZERO].clone();
        let is_b_signed = is_mulh.clone() + is_mulhsu.clone();
        let is_c_signed = is_mulh.clone();
        let op_b_value: [AB::VarMaybeExt; 4] = [
            local[RES_OP_B_VALUE].clone(),
            local[RES_OP_B_VALUE + 1].clone(),
            local[RES_OP_B_VALUE + 2].clone(),
            local[RES_OP_B_VALUE + 3].clone(),
        ];
        let op_c_value: [AB::VarMaybeExt; 4] = [
            local[RES_OP_C_VALUE].clone(),
            local[RES_OP_C_VALUE + 1].clone(),
            local[RES_OP_C_VALUE + 2].clone(),
            local[RES_OP_C_VALUE + 3].clone(),
        ];
        let product: [AB::VarMaybeExt; 8] =
            core::array::from_fn(|i| local[RES_MUL_PRODUCT + i].clone());
        let carry: [AB::VarMaybeExt; 8] =
            core::array::from_fn(|i| local[RES_MUL_CARRY + i].clone());
        mul_op_gate_constraints(
            builder,
            op_b_value,
            op_c_value,
            product.clone(),
            carry,
            local[RES_MUL_B_MSB].clone(),
            local[RES_MUL_C_MSB].clone(),
            local[RES_MUL_B_SIGN_EXTEND].clone(),
            local[RES_MUL_C_SIGN_EXTEND].clone(),
            is_b_signed,
            is_c_signed,
            perform_calc,
        );

        // =====================================================================
        // #4: Result matching (chip-specific)
        // =====================================================================
        // MUL: a == product[0..4]
        for i in 0..4 {
            builder
                .when(is_mul.clone())
                .when(is_real.clone())
                .assert_zero(local[RES_OP_A_VALUE + i].clone() - product[i].clone());
        }
        // MULH/MULHU/MULHSU: a == product[4..8]
        let is_upper = is_mulh + is_mulhu + is_mulhsu;
        for i in 0..4 {
            builder
                .when(is_upper.clone())
                .when(is_real.clone())
                .assert_zero(local[RES_OP_A_VALUE + i].clone() - product[i + 4].clone());
        }

        // =====================================================================
        // #5: RTypeRegisterOp gate constraints (op_a_zero)
        // =====================================================================
        let op_a_value: [AB::VarMaybeExt; 4] = [
            local[RES_OP_A_VALUE].clone(),
            local[RES_OP_A_VALUE + 1].clone(),
            local[RES_OP_A_VALUE + 2].clone(),
            local[RES_OP_A_VALUE + 3].clone(),
        ];
        rtype_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            op_a_value,
            is_real,
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let op_a_zero = local[RES_OP_A_ZERO].clone();

        // perform_calc = is_real - op_a_zero
        // MulOperation interactions use this multiplicity (original AIR passes
        // perform_calc as the is_real parameter to MulOperation::eval()).
        let perform_calc = is_real.clone() - op_a_zero;

        // Order matches precompute_lc exactly.

        // #1-4: CPUState (mult = is_real)
        cpu_state_lookup(builder, is_real.clone());

        // #5-18: MulOperation (mult = perform_calc)
        mul_op_lookup(builder, perform_calc);

        // #19-31: RTypeRegisterOp (mult = is_real)
        rtype_register_op_lookup(builder, is_real.clone());

        // #32: BitVec — emit only on real rows.
        // is_real ∈ {0, 1} by the explicit boolean gate added in `eval`,
        // so mult is non-negative and bounded.
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to MulChip)
// =============================================================================

impl<F: Field> BaseAir<F> for MulChipPolyAir {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<F: Field> MachineAir<F> for MulChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "MulPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        MulChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .mul_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse MulChip path: cpu_state, mem_ops, mul_op BLU
                    // (MSB + U8Range product + U16Range carry). mul_op.populate
                    // is skipped when op_a_0=true, matching the trace.
                    let mut row = [F::zero(); NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
                    MulChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec on every real row
                    // (mult = is_real). The helper accounts for the mul_op
                    // skip on op_a_0=true rows.
                    let value = mul_bitvec_value(event.b, event.c, event.opcode, event.op_a_0);
                    blu.add_bit_vec_lookup(value);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <MulChip as MachineAir<F>>::included(&MulChip, shard)
    }

    fn local_only(&self) -> bool {
        <MulChip as MachineAir<F>>::local_only(&MulChip)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Total number of lookup interactions:
    /// - 4 CPUState (recv_state, send_state, clk_u16, clk_bit)
    /// - 14 MulOperation (2×MSB, 4×product_u8, 8×carry_u16)
    /// - 13 RTypeRegisterOp (1×program, 3×(ts_u16, ts_bit, mem_send, mem_recv))
    /// - 1 Packed boolean bit lookup (8 bits)
    const NUM_LOOKUPS: usize = 32;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    use dt_core_executor::{ExecutionRecord, Executor};
    use dt_stark::{
        air::{
            full_air_builders::{
                collect_reserved_poly,
                evaluator::{
                    bound_var_main_prep, bound_var_mat, first_round_evaluation,
                    nonfirst_round_evaluation,
                },
                permutation::generate_permutation_trace_,
                precompute::{precompute_linear_combination, PrecomputeRowBuilder},
            },
            FullAir, MachineAir,
        },
        DTCoreOpts,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    use super::super::MulChip;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    // =========================================================================
    // Test Helper Functions
    // =========================================================================

    /// Build public values with execution_shard set at the expected index.
    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let required_max_beta_power = <MulChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&MulChipPolyAir);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn reducer() -> Vec<EF> {
        // Gate constraints: is_real_bool(1) + cpu_state(1) + one_hot(1) + mul_op(10)
        //                   + result_match(8) + rtype(5) = 26
        // Lookup batch: ceil(32/3) = 11
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 26;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS as u32).map(|i| ef(i + 1)).collect()
    }

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(air: &MulChipPolyAir, main: &RowMajorMatrix<F>) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <MulChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let row_binding = main.row_slice(row_idx);
            use std::ops::Deref;
            let row: &[F] = row_binding.deref();
            let reserved = collect_reserved_poly(row, &[], &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    /// Generate a sample trace with valid MUL operations.
    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = MulChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    // =========================================================================
    // Constraint satisfaction tests
    // =========================================================================

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = MulChipPolyAir;
        let main = sample_trace();
        let height = main.height();
        std::println!("trace height = {}, width = {}", height, main.width());
        assert!(height >= 2);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            NUM_PRECOMPUTED,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        let constraint_reducer = reducer();
        let global = EF::zero();

        let first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            F::one(),
            F::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(first.iter().all(|x| x.is_zero()), "first_round_evaluation failed: {:?}", first);

        let nonfirst = nonfirst_round_evaluation(
            &air,
            &public,
            &reserved,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            EF::one(),
            EF::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(
            nonfirst.iter().all(|x| x.is_zero()),
            "nonfirst_round_evaluation failed: {:?}",
            nonfirst
        );
    }

    fn random_mul_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace();
        let base_height = base.height();
        assert!(base_height >= 1, "sample trace must contain at least one row");
        assert!(
            target_height >= base_height,
            "target height {} smaller than sample trace height {}",
            target_height,
            base_height
        );
        if target_height == base_height {
            return base;
        }

        let last_row_start = (base_height - 1) * NUM_MUL_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_MUL_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_MUL_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_MUL_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = MulChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_mul_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc =
            <MulChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

        // --- Precompute phase ---
        let t_precompute = std::time::Instant::now();
        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            NUM_PRECOMPUTED,
        );
        let precompute_elapsed = t_precompute.elapsed();
        std::println!("  precompute_linear_combination: {:?}", precompute_elapsed);

        let t_perm = std::time::Instant::now();
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );
        let perm_elapsed = t_perm.elapsed();
        std::println!("  generate_permutation_trace_: {:?}", perm_elapsed);

        let mut precomputed = trim_rows(&precomputed_full, height);
        let mut permutation = trim_rows(&permutation_full, height);

        // --- Round 0: first_round_evaluation ---
        let t_total = std::time::Instant::now();
        let t_round = std::time::Instant::now();
        let _first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &precomputed,
            &permutation,
            alpha,
            &beta_powers,
            beta_septix,
            global,
            F::one(),
            F::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        std::println!("  round 0 (first_round): {:?}", t_round.elapsed());

        // --- Rounds 1..log_n-1: fold + nonfirst_round_evaluation ---
        let mut reserved = bound_var_main_prep(&main, None, &reserved_poly_desc, ef(42));
        precomputed = bound_var_mat(&precomputed_full, ef(42));
        permutation = bound_var_mat(&permutation_full, ef(42));
        let mut selector_first = EF::one() - ef(42);
        let mut selector_last = ef(42);

        for round in 1..log_n {
            let challenge = ef((round as u32) + 100);
            let t_round = std::time::Instant::now();

            let _nonfirst = nonfirst_round_evaluation(
                &air,
                &public,
                &reserved,
                &precomputed,
                &permutation,
                alpha,
                &beta_powers,
                beta_septix,
                global,
                selector_first,
                selector_last,
                local_sum,
                BATCH_SIZE,
                &constraint_reducer,
            );

            let round_elapsed = t_round.elapsed();
            std::println!("  round {} (nonfirst): {:?}", round, round_elapsed);

            if round < log_n - 1 {
                reserved = bound_var_mat(&reserved, challenge);
                precomputed = bound_var_mat(&precomputed, challenge);
                permutation = bound_var_mat(&permutation, challenge);
                selector_first *= EF::one() - challenge;
                selector_last *= challenge;
            }
        }

        let total_eval_elapsed = t_total.elapsed();
        std::println!("  ---");
        std::println!("  total precompute: {:?}", precompute_elapsed);
        std::println!("  total perm_gen:   {:?}", perm_elapsed);
        std::println!("  total eval ({} rounds): {:?}", log_n, total_eval_elapsed);
        std::println!(
            "  GRAND TOTAL (precompute + perm + eval): {:?}",
            precompute_elapsed + perm_elapsed + total_eval_elapsed
        );
    }

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    use dt_core_executor::{Instruction, Program};

    /// Mix of MUL / MULH / MULHU / MULHSU on positive and negative operands,
    /// with one writing to x0 to exercise the op_a_0 branch.
    fn simple_mul_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 7, false, true),
            Instruction::new(Opcode::ADD, 3, 0, 0xffff_ffff, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0x8000_0000, false, true),
            Instruction::new(Opcode::MUL, 10, 1, 2, false, false),
            Instruction::new(Opcode::MULH, 11, 3, 4, false, false),
            Instruction::new(Opcode::MULHU, 12, 1, 4, false, false),
            Instruction::new(Opcode::MULHSU, 13, 3, 2, false, false),
            // x0 destination — exercises op_a_0=true emit branch.
            Instruction::new(Opcode::MUL, 0, 1, 2, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_mul_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <MulChipPolyAir as MachineAir<F>>::generate_dependencies(
            &MulChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = shard.mul_events.len();
        assert!(expected > 0, "fixture must include mul events");
        assert_eq!(bitvec_total, expected, "BitVec BLU emit count must equal lookup send count");
    }

    /// Every MUL writes to x0 — BitVec should still emit (mult = is_real, not
    /// is_real - op_a_zero), and the helper must produce values matching the
    /// trace where mul_op fields are 0 due to skipped populate.
    fn only_x0_mul_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 7, false, true),
            Instruction::new(Opcode::MUL, 0, 1, 2, false, false),
            Instruction::new(Opcode::MULH, 0, 1, 2, false, false),
            Instruction::new(Opcode::MULHU, 0, 1, 2, false, false),
            Instruction::new(Opcode::MULHSU, 0, 1, 2, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_emitted_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_mul_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(
            shard.mul_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all mul events must be op_a_0=true",
        );
        let expected = shard.mul_events.len();
        assert!(expected > 0, "fixture must yield mul events");

        let mut deps = ExecutionRecord::default();
        <MulChipPolyAir as MachineAir<F>>::generate_dependencies(
            &MulChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_count: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();
        assert_eq!(bitvec_count, expected, "op_a_0=true rows must still emit BitVec for mul chip");
    }

    /// Sanity: helper value matches the trace recurrence for a known case.
    /// MULHSU with negative b (msb=1) and positive c: b_sign_extend=1, c_sign_extend=0.
    #[test]
    fn mul_bitvec_value_mulhsu_signed_b() {
        let b: u32 = 0x8000_0000; // msb=1
        let c: u32 = 0x7fff_ffff; // msb=0
        let v = mul_bitvec_value(b, c, Opcode::MULHSU, /* op_a_0 */ false);
        // bit 0: b_msb=1
        // bit 2: b_sign_extend=1 (mulh/mulhsu && b_msb)
        // bit 7: is_mulhsu=1
        let expected: u16 = 1 | (1 << 2) | (1 << 7);
        assert_eq!(v, expected, "got 0b{:08b}, expected 0b{:08b}", v, expected);
    }

    /// Sanity: when op_a_0=true, mul_op fields are zero in trace; helper
    /// returns only selector bits, regardless of operand signs.
    #[test]
    fn mul_bitvec_value_op_a_zero_zeros_mul_op_bits() {
        let b: u32 = 0x8000_0000;
        let c: u32 = 0x8000_0000;
        let v = mul_bitvec_value(b, c, Opcode::MULH, /* op_a_0 */ true);
        // bit 5: is_mulh=1. All mul_op bits zero (populate skipped).
        let expected: u16 = 1 << 5;
        assert_eq!(v, expected, "got 0b{:08b}, expected 0b{:08b}", v, expected);
    }
}
