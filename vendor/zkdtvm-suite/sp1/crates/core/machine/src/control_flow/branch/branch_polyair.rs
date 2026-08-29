//! PolyAir `FullAir` adapter for `BranchChip` (`air.rs`).
//!
//! Mirrors the `Air` implementation using the four-phase PolyAir model
//! (`reserved_poly` / `precompute_lc` / `eval` / `lookup`).

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};
use crate::{
    adapter::{
        register::b_type::{
            btype_register_op_gate_constraints, btype_register_op_lookup,
            btype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    operations::{
        add_op_gate_constraints, add_op_lookup, add_op_precompute_lc,
        baby_bear_range_check_gate_constraints, baby_bear_range_check_lookup,
        baby_bear_range_check_precompute_lc, lt_signed_gate_constraints, lt_signed_lookup,
        lt_signed_precompute_lc,
    },
};

/// BitVec has 16 elements; Program send has 15.
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `BranchColumns<u8>` (NUM_BRANCH_COLS = 59).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1..3]   cpu_state.{clk_16_28, clk_0_16}            ← precompute-only
//   [3]      cpu_state.pc
//   [4]      mem_ops.op_a                               ← precompute-only
//   [5..9]   mem_ops.op_a_access.access.value
//   [9..14]  mem_ops.op_a_access.access.{ts fields}     ← precompute-only
//   [14]     mem_ops.op_a_zero
//   [15]     mem_ops.op_b                               ← precompute-only
//   [16..20] mem_ops.op_b_access.access.value
//   [20..25] mem_ops.op_b_access.access.{ts fields}     ← precompute-only
//   [25..29] mem_ops.op_c_imm
//   [29..33] pc (Word)
//   [33]     pc_range_checker.most_sig_byte_lt_120
//   [34..38] add_op.value
//   [38]     next_pc_range_checker.most_sig_byte_lt_120
//   [39]     is_beq
//   [40]     is_bne
//   [41]     is_blt
//   [42]     is_bge
//   [43]     is_bltu
//   [44]     is_bgeu
//   [45]     is_branching
//   [46]     a_eq_b
//   [47]     a_gt_b
//   [48]     a_lt_b
//   [49..53] compare_operation.result.byte_flags
//   [53..55] compare_operation.result.comparison_bytes
//   [55]     compare_operation.result.not_eq_inv
//   [56]     compare_operation.result.result
//   [57]     compare_operation.b_msb
//   [58]     compare_operation.c_msb
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_CPU_PC: usize = 3;
const COL_OP_A_VALUE: usize = 5;
const COL_OP_A_ZERO: usize = 14;
const COL_OP_B_VALUE: usize = 16;
const COL_OP_C_IMM: usize = 25;
const COL_PC_WORD: usize = 29;
const COL_PC_RANGE_CHECKER: usize = 33;
const COL_ADD_OP_VALUE: usize = 34;
const COL_NEXT_PC_RANGE_CHECKER: usize = 38;
const COL_IS_BEQ: usize = 39;
const COL_IS_BNE: usize = 40;
const COL_IS_BLT: usize = 41;
const COL_IS_BGE: usize = 42;
const COL_IS_BLTU: usize = 43;
const COL_IS_BGEU: usize = 44;
const COL_IS_BRANCHING: usize = 45;
const COL_A_EQ_B: usize = 46;
const COL_A_GT_B: usize = 47;
const COL_A_LT_B: usize = 48;
const COL_BYTE_FLAGS: usize = 49;
const COL_COMPARISON_BYTES: usize = 53;
const COL_NOT_EQ_INV: usize = 55;
const COL_LT_RESULT: usize = 56;
const COL_B_MSB: usize = 57;
const COL_C_MSB: usize = 58;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 45).
//
// Only fields read by `eval` or `lookup` are retained.
//
//   [0..6]   is_beq, is_bne, is_blt, is_bge, is_bltu, is_bgeu
//   [6]      is_branching
//   [7]      cpu_state.shard
//   [8]      cpu_state.pc
//   [9]      op_a_zero
//   [10]     pc_range_checker.most_sig_byte_lt_120
//   [11]     next_pc_range_checker.most_sig_byte_lt_120
//   [12]     a_eq_b
//   [13]     a_gt_b
//   [14]     a_lt_b
//   [15]     not_eq_inv
//   [16]     compare_operation.result.result
//   [17]     b_msb
//   [18]     c_msb
//   [19..23] op_a_access.access.value
//   [23..27] op_b_access.access.value
//   [27..31] op_c_imm
//   [31..35] pc (Word)
//   [35..39] add_op.value
//   [39..43] byte_flags
//   [43..45] comparison_bytes
// ============================================================================

const RES_IS_BEQ: usize = 0;
const RES_IS_BNE: usize = 1;
const RES_IS_BLT: usize = 2;
const RES_IS_BGE: usize = 3;
const RES_IS_BLTU: usize = 4;
const RES_IS_BGEU: usize = 5;
const RES_IS_BRANCHING: usize = 6;
const RES_CPU_SHARD: usize = 7;
const RES_CPU_PC: usize = 8;
const RES_OP_A_ZERO: usize = 9;
const RES_PC_RANGE_CHECKER: usize = 10;
const RES_NEXT_PC_RANGE_CHECKER: usize = 11;
const RES_A_EQ_B: usize = 12;
const RES_A_GT_B: usize = 13;
const RES_A_LT_B: usize = 14;
const RES_NOT_EQ_INV: usize = 15;
const RES_LT_RESULT: usize = 16;
const RES_B_MSB: usize = 17;
const RES_C_MSB: usize = 18;
const RES_OP_A_VALUE: usize = 19;
const RES_OP_B_VALUE: usize = 23;
const RES_OP_C_IMM: usize = 27;
const RES_PC_WORD: usize = 31;
const RES_ADD_OP_VALUE: usize = 35;
const RES_BYTE_FLAGS: usize = 39;
const RES_COMPARISON_BYTES: usize = 43;
const RES_NUM_COLS: usize = 45;

#[derive(Default, Clone, Copy)]
pub struct BranchChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for BranchChipPolyAir {
    fn width(&self) -> usize {
        NUM_BRANCH_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_BEQ));
        cols.push(PairCol::Main(COL_IS_BNE));
        cols.push(PairCol::Main(COL_IS_BLT));
        cols.push(PairCol::Main(COL_IS_BGE));
        cols.push(PairCol::Main(COL_IS_BLTU));
        cols.push(PairCol::Main(COL_IS_BGEU));
        cols.push(PairCol::Main(COL_IS_BRANCHING));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_CPU_PC));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_PC_RANGE_CHECKER));
        cols.push(PairCol::Main(COL_NEXT_PC_RANGE_CHECKER));
        cols.push(PairCol::Main(COL_A_EQ_B));
        cols.push(PairCol::Main(COL_A_GT_B));
        cols.push(PairCol::Main(COL_A_LT_B));
        cols.push(PairCol::Main(COL_NOT_EQ_INV));
        cols.push(PairCol::Main(COL_LT_RESULT));
        cols.push(PairCol::Main(COL_B_MSB));
        cols.push(PairCol::Main(COL_C_MSB));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_IMM + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_PC_WORD + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_ADD_OP_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_BYTE_FLAGS + i));
        }
        for i in 0..2 {
            cols.push(PairCol::Main(COL_COMPARISON_BYTES + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &BranchColumns<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc_scalar = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let base_w = |i: u32| AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << (8 * i)));
        let add_val = &local.add_op.value;
        let next_pc_scalar = add_val[0].clone() * base_w(0) +
            add_val[1].clone() * base_w(1) +
            add_val[2].clone() * base_w(2) +
            add_val[3].clone() * base_w(3);

        let is_beq = local.is_beq.clone();
        let is_bne = local.is_bne.clone();
        let is_blt = local.is_blt.clone();
        let is_bge = local.is_bge.clone();
        let is_bltu = local.is_bltu.clone();
        let is_bgeu = local.is_bgeu.clone();

        let opcode_expr = is_beq.clone() *
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BEQ as u8)) +
            is_bne.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BNE as u8)) +
            is_blt.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BLT as u8)) +
            is_bge.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BGE as u8)) +
            is_bltu.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BLTU as u8)) +
            is_bgeu.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::BGEU as u8));

        let op_c_imm = &local.mem_ops.op_c_imm;
        let pc_word = &local.pc;

        // =====================================================================
        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        // =====================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc_scalar.clone(),
            next_pc_scalar,
        );

        // =====================================================================
        // #5-13: BTypeRegisterOp (1 program + 4 op_a read + 4 op_b read)
        // =====================================================================
        btype_register_op_precompute_lc(
            builder,
            pc_scalar,
            opcode_expr,
            local.mem_ops.op_a.clone(),
            local.mem_ops.op_b.clone(),
            [op_c_imm[0].clone(), op_c_imm[1].clone(), op_c_imm[2].clone(), op_c_imm[3].clone()],
            local.mem_ops.op_a_zero.clone(),
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_b_access.access,
            shard,
            clk,
        );

        let op_a_val = &local.mem_ops.op_a_access.access.value;
        let op_b_val = &local.mem_ops.op_b_access.access.value;

        // =====================================================================
        // #14-15: BabyBearWordRangeChecker (pc, add_op.value)
        // =====================================================================
        baby_bear_range_check_precompute_lc(
            builder,
            pc_word[3].clone(),
            local.pc_range_checker.most_sig_byte_lt_120.clone(),
        );
        baby_bear_range_check_precompute_lc(
            builder,
            add_val[3].clone(),
            local.next_pc_range_checker.most_sig_byte_lt_120.clone(),
        );

        // =====================================================================
        // #16-17: AddOperation U8Range (mult: is_branching)
        // =====================================================================
        add_op_precompute_lc(builder, add_val);

        // =====================================================================
        // #18-20: LtOperationSigned (2 MSB + 1 U8Range diff)
        // =====================================================================
        lt_signed_precompute_lc(
            builder,
            local.compare_operation.b_msb.clone(),
            local.compare_operation.c_msb.clone(),
            op_a_val[3].clone(),
            op_b_val[3].clone(),
            [
                local.compare_operation.result.comparison_bytes[0].clone(),
                local.compare_operation.result.comparison_bytes[1].clone(),
            ],
            local.compare_operation.result.result.clone(),
        );

        // =====================================================================
        // #21: BitVec (12 bools: is_beq, is_bne, is_blt, is_bge, is_bltu,
        //       is_bgeu, is_branching, byte_flags[0..3], result)
        // is_real (= sum of 6 selectors) and is_signed (= is_blt + is_bge)
        // are derived expressions, not raw witness columns; their booleanness
        // is implied by the individual selector booleans enforced here plus
        // the explicit is_real bool gate in `eval`.
        // =====================================================================
        let cmp = &local.compare_operation.result;
        bitvec_precompute_lc(
            builder,
            vec![
                is_beq,
                is_bne,
                is_blt,
                is_bge,
                is_bltu,
                is_bgeu,
                local.is_branching.clone(),
                cmp.byte_flags[0].clone(),
                cmp.byte_flags[1].clone(),
                cmp.byte_flags[2].clone(),
                cmp.byte_flags[3].clone(),
                cmp.result.clone(),
            ],
        );
    }

    /// Gate constraints, ordered to match the original `Air<AB>::eval()` in `air.rs`.
    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let one = AB::one_maybe();

        let is_beq = local[RES_IS_BEQ].clone();
        let is_bne = local[RES_IS_BNE].clone();
        let is_blt = local[RES_IS_BLT].clone();
        let is_bge = local[RES_IS_BGE].clone();
        let is_bltu = local[RES_IS_BLTU].clone();
        let is_bgeu = local[RES_IS_BGEU].clone();
        let is_branching = local[RES_IS_BRANCHING].clone();
        let a_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_A_VALUE + i].clone());
        let b_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_B_VALUE + i].clone());
        let pc_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_PC_WORD + i].clone());
        let add_value: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_ADD_OP_VALUE + i].clone());
        let op_c_imm: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_C_IMM + i].clone());
        let byte_flags: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_BYTE_FLAGS + i].clone());
        let comparison_bytes: [AB::VarMaybeExt; 2] =
            core::array::from_fn(|i| local[RES_COMPARISON_BYTES + i].clone());

        // ── air.rs L47-55: assert_bool × 8 ──────────────────────────────
        // The 6 selector booleans and is_branching/byte_flags/cmp.result
        // are enforced by BitVec #21 (conditioned on is_real).

        // ── air.rs L56-61: is_real ───────────────────────────────────────
        // Explicit bool gate for is_real (removed from BitVec payload).
        let is_real = is_beq.clone() +
            is_bne.clone() +
            is_blt.clone() +
            is_bge.clone() +
            is_bltu.clone() +
            is_bgeu.clone();
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));
        let not_branching = is_real.clone() - is_branching.clone();

        // ── air.rs L63-69: CPUState::eval() ─────────────────────────────
        let pv = builder.public();
        const PV_EXECUTION_SHARD_IDX: usize = 44;
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // ── air.rs L70-82: BTypeRegisterOp::eval() ─────────────────────
        btype_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            a_word.clone(),
            is_real.clone(),
        );

        // ── air.rs L83: assert_eq(cpu_state.pc, pc.reduce) ─────────────
        let base_w = |i: u32| AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << (8 * i)));
        let pc_reduced = pc_word[0].clone() * base_w(0) +
            pc_word[1].clone() * base_w(1) +
            pc_word[2].clone() * base_w(2) +
            pc_word[3].clone() * base_w(3);
        builder.assert_zero(pc_reduced - local[RES_CPU_PC].clone());

        // ── air.rs L86-93: BabyBearWordRangeChecker (pc) ────────────────
        baby_bear_range_check_gate_constraints(
            builder,
            pc_word.clone(),
            local[RES_PC_RANGE_CHECKER].clone(),
            is_real.clone(),
        );

        // ── air.rs L94-100: BabyBearWordRangeChecker (add_op.value) ─────
        baby_bear_range_check_gate_constraints(
            builder,
            add_value.clone(),
            local[RES_NEXT_PC_RANGE_CHECKER].clone(),
            is_real.clone(),
        );

        // ── air.rs L102-107: AddOperation::eval() ───────────────────────
        add_op_gate_constraints(
            builder,
            pc_word,
            op_c_imm,
            add_value.clone(),
            is_branching.clone(),
        );

        // ── air.rs L108-112: when(not_branching) add_op.value = pc + 4 ───
        builder.when(not_branching.clone()).assert_eq(
            add_value[0].clone() * base_w(0) +
                add_value[1].clone() * base_w(1) +
                add_value[2].clone() * base_w(2) +
                add_value[3].clone() * base_w(3),
            local[RES_CPU_PC].clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC)),
        );

        // ── air.rs L114: not_branching = is_real - is_branching ─────────
        builder.assert_zero(is_real.clone() - (is_branching.clone() + not_branching.clone()));

        // ── air.rs L118-155: Branching value constraints ────────────────
        let a_eq_b = local[RES_A_EQ_B].clone();
        let a_gt_b = local[RES_A_GT_B].clone();
        let a_lt_b = local[RES_A_LT_B].clone();
        builder.when(is_beq.clone() * is_branching.clone()).assert_one(a_eq_b.clone());
        builder
            .when(is_beq.clone() * (one.clone() - is_branching.clone()))
            .assert_one(a_gt_b.clone() + a_lt_b.clone());

        builder
            .when(is_bne.clone() * is_branching.clone())
            .assert_one(a_gt_b.clone() + a_lt_b.clone());
        builder
            .when(is_bne.clone() * (one.clone() - is_branching.clone()))
            .assert_one(a_eq_b.clone());

        builder
            .when((is_blt.clone() + is_bltu.clone()) * is_branching.clone())
            .assert_one(a_lt_b.clone());
        builder
            .when((is_blt.clone() + is_bltu.clone()) * (one.clone() - is_branching.clone()))
            .assert_one(a_eq_b.clone() + a_gt_b.clone());

        builder
            .when((is_bge.clone() + is_bgeu.clone()) * is_branching.clone())
            .assert_one(a_gt_b.clone() + a_eq_b.clone());
        builder
            .when((is_bge.clone() + is_bgeu.clone()) * (one.clone() - is_branching))
            .assert_one(a_lt_b.clone());

        // ── air.rs L158: when(a_eq_b) assert a == b ────────────────────
        for i in 0..4 {
            builder.when(a_eq_b.clone()).assert_zero(a_word[i].clone() - b_word[i].clone());
        }

        // ── air.rs L160-170: LtOperationSigned::eval() ─────────────────
        let is_signed = is_blt + is_bge;
        builder.when(one.clone() - is_real.clone()).assert_zero(is_signed.clone());

        let lt_result = local[RES_LT_RESULT].clone();
        let b_msb = local[RES_B_MSB].clone();
        let c_msb = local[RES_C_MSB].clone();
        let not_eq_inv = local[RES_NOT_EQ_INV].clone();
        lt_signed_gate_constraints(
            builder,
            a_word,
            b_word,
            b_msb,
            c_msb,
            byte_flags.clone(),
            comparison_bytes,
            not_eq_inv,
            lt_result.clone(),
            is_signed,
            is_real.clone(),
        );

        // ── air.rs L171-179: Link LT result to branch helper booleans ──
        let is_eq = one -
            (byte_flags[0].clone() +
                byte_flags[1].clone() +
                byte_flags[2].clone() +
                byte_flags[3].clone());
        let is_less = lt_result;
        builder.when(is_real.clone()).assert_eq(a_eq_b.clone(), is_eq);
        builder.when(is_real.clone()).assert_eq(a_lt_b.clone(), is_less);
        builder.assert_zero(is_real - a_eq_b - a_lt_b - a_gt_b);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_beq = local[RES_IS_BEQ].clone();
        let is_bne = local[RES_IS_BNE].clone();
        let is_blt = local[RES_IS_BLT].clone();
        let is_bge = local[RES_IS_BGE].clone();
        let is_bltu = local[RES_IS_BLTU].clone();
        let is_bgeu = local[RES_IS_BGEU].clone();
        let is_real = is_beq + is_bne + is_blt.clone() + is_bge.clone() + is_bltu + is_bgeu;
        let is_signed = is_blt + is_bge;
        let is_branching = local[RES_IS_BRANCHING].clone();

        // #1-4: CPUState
        cpu_state_lookup(builder, is_real.clone());
        // #5-13: BTypeRegisterOp
        btype_register_op_lookup(builder, is_real.clone());
        // #14-15: BabyBearWordRangeChecker (pc, next_pc)
        baby_bear_range_check_lookup(builder, is_real.clone());
        baby_bear_range_check_lookup(builder, is_real.clone());
        // #16-17: AddOperation U8Range
        add_op_lookup(builder, is_branching);
        // #18-20: LtOperationSigned (2 MSB + 1 U8Range diff)
        lt_signed_lookup(builder, is_signed, is_real.clone());
        // #21: BitVec boolean — mult = is_real, matching the conditioning of
        // every other lookup in the chip. On real rows is_real = 1 (exactly
        // one selector set), so BitVec enforces booleanness of all 14 payload
        // bits. On padding (PaddingRow::Zero, all selectors = 0), is_real = 0,
        // no send, and the payload bits are trivially boolean (all zero).
        bitvec_lookup(builder, is_real);
    }
}

/// Pack a slice of field-element bits (each 0 or 1) into a u16 value suitable
/// for `add_bit_vec_lookup`. Bit `i` of the result equals `bits[i]`.
fn pack_bits<F: Field>(bits: &[F]) -> u16 {
    debug_assert!(bits.len() <= 16, "BitVec payload exceeds 16 bits");
    let mut value: u16 = 0;
    for (i, b) in bits.iter().enumerate() {
        if b.is_one() {
            value |= 1u16 << i;
        }
    }
    value
}

// =============================================================================
// MachineAir implementation (delegation to BranchChip)
// =============================================================================

impl<F: Field> BaseAir<F> for BranchChipPolyAir {
    fn width(&self) -> usize {
        NUM_BRANCH_COLS
    }
}

impl<F: Field> MachineAir<F> for BranchChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "BranchPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        BranchChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        if input.branch_events.is_empty() {
            return;
        }

        let shard = input.execution_shard();
        let chunk_size = std::cmp::max(input.branch_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .branch_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    let mut row = [F::zero(); NUM_BRANCH_COLS];
                    let cols: &mut BranchColumns<F> = row.as_mut_slice().borrow_mut();
                    BranchChip.event_to_row(record, event, cols, &mut blu, shard);
                    // BitVec #21 (12 bits, in precompute_lc order):
                    //   0  is_beq            6  is_branching
                    //   1  is_bne            7  byte_flags[0]
                    //   2  is_blt            8  byte_flags[1]
                    //   3  is_bge            9  byte_flags[2]
                    //   4  is_bltu          10  byte_flags[3]
                    //   5  is_bgeu          11  cmp.result
                    blu.add_bit_vec_lookup(pack_bits(&[
                        cols.is_beq,
                        cols.is_bne,
                        cols.is_blt,
                        cols.is_bge,
                        cols.is_bltu,
                        cols.is_bgeu,
                        cols.is_branching,
                        cols.compare_operation.result.byte_flags[0],
                        cols.compare_operation.result.byte_flags[1],
                        cols.compare_operation.result.byte_flags[2],
                        cols.compare_operation.result.byte_flags[3],
                        cols.compare_operation.result.result,
                    ]));
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <BranchChip as MachineAir<F>>::included(&BranchChip, shard)
    }

    fn local_only(&self) -> bool {
        <BranchChip as MachineAir<F>>::local_only(&BranchChip)
    }
}

#[cfg(test)]
mod tests {
    use std;

    use super::*;

    /// Lookup interaction count (see module docs / interaction order in `precompute_lc`).
    const NUM_LOOKUPS: usize = 21;
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

    use super::super::BranchChip;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const PV_EXECUTION_SHARD_IDX: usize = 44;

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
        let n = <BranchChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&BranchChipPolyAir);
        (0..=n).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn reducer() -> Vec<EF> {
        // Gate constraints: 47 (46 original + 1 is_real bool gate)
        // Lookup batch: ceil(21/3) = 7
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 47;
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

    fn reserved_poly_matrix(
        air: &BranchChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <BranchChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = BranchChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_branch_first_and_nonfirst_round_evaluation_satisfied() {
        let air = BranchChipPolyAir;
        let main = sample_trace();
        let height = main.height();
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

    fn random_branch_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_BRANCH_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_BRANCH_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_BRANCH_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_BRANCH_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = BranchChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_branch_trace(log_n, seed);
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
            <BranchChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    /// Branch emits 1 BitVec lookup per real event with mult = is_real
    /// (= sum of selectors = 1 on real rows, 0 on padding). Total emission
    /// equals branch_events.len(). All payload columns are populated
    /// unconditionally by event_to_row, independent of op_a_0.
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <BranchChipPolyAir as MachineAir<F>>::generate_dependencies(
            &BranchChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = shard.branch_events.len();
        assert!(expected > 0, "fixture must include branch events");
        assert_eq!(bitvec_total, expected, "BitVec BLU count must equal event count");
    }
}
