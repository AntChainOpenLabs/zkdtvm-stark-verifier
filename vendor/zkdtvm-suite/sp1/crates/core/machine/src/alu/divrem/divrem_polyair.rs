//! PolyAir FullAir implementation for `DivRemChip`.
//!
//! The 43 original interactions remain lookup-enforced. Extra chip-level checks
//! use reserved columns plus `BitVec` and `U16Range` lookups for booleans and
//! derived carries. Memory timestamp internal gates remain framework-level.

use core::mem::transmute;

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    adapter::{
        register::r_type::{
            rtype_register_op_gate_constraints, rtype_register_op_lookup,
            rtype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{
        bitvec_lookup, bitvec_precompute_lc, ltu_lookup, ltu_precompute_lc, msb_lookup,
        msb_precompute_lc, slice_u8_range_lookup, slice_u8_range_precompute_lc,
    },
    operations::{
        add_op_gate_constraints, add_op_lookup, add_op_precompute_lc,
        assert_lt_bytes_gate_constraints, is_equal_word_gate_constraints,
        is_zero_word_gate_constraints, mul_op_gate_constraints, mul_op_lookup,
        mul_op_precompute_lc,
    },
    utils::indices_arr,
};

use super::{DivRemChip, DivRemCols, NUM_DIVREM_COLS};

/// Largest lookup payload: BitVec has 16 values (fixed width), which is the largest.
/// `send_program` has 15 values.
const MAX_LOOKUP_VALUES: usize = 16;

const fn make_col_map() -> DivRemCols<usize> {
    let arr = indices_arr::<NUM_DIVREM_COLS>();
    // SAFETY: DivRemCols is #[repr(C)] and can be mapped from a flat index array.
    unsafe { transmute::<[usize; NUM_DIVREM_COLS], DivRemCols<usize>>(arr) }
}

const DIVREM_COL_MAP: DivRemCols<usize> = make_col_map();
const COL_IS_REAL: usize = DIVREM_COL_MAP.is_real;
const COL_IS_DIV: usize = DIVREM_COL_MAP.is_div;
const COL_IS_DIVU: usize = DIVREM_COL_MAP.is_divu;
const COL_IS_REM: usize = DIVREM_COL_MAP.is_rem;
const COL_IS_REMU: usize = DIVREM_COL_MAP.is_remu;
const COL_B_MSB: usize = DIVREM_COL_MAP.b_msb;
const COL_REM_MSB: usize = DIVREM_COL_MAP.rem_msb;
const COL_C_MSB: usize = DIVREM_COL_MAP.c_msb;
const COL_B_NEG: usize = DIVREM_COL_MAP.b_neg;
const COL_REM_NEG: usize = DIVREM_COL_MAP.rem_neg;
const COL_C_NEG: usize = DIVREM_COL_MAP.c_neg;
const COL_REMAINDER_CHECK_MULT: usize = DIVREM_COL_MAP.remainder_check_multiplicity;
const COL_SHARD: usize = DIVREM_COL_MAP.cpu_state.shard;

// Reserved columns used by chip-level gates.
const COL_OP_A_ZERO: usize = DIVREM_COL_MAP.mem_ops.op_a_zero;
const COL_QUOTIENT: [usize; 4] = DIVREM_COL_MAP.quotient.0;
const COL_REMAINDER: [usize; 4] = DIVREM_COL_MAP.remainder.0;
const COL_OP_A_VALUE: [usize; 4] = DIVREM_COL_MAP.mem_ops.op_a_access.access.value.0;
const COL_OP_B_VALUE: [usize; 4] = DIVREM_COL_MAP.mem_ops.op_b_access.access.value.0;
const COL_OP_C_VALUE: [usize; 4] = DIVREM_COL_MAP.mem_ops.op_c_access.access.value.0;
const COL_IS_C_0_RESULT: usize = DIVREM_COL_MAP.is_c_0.result;

// Columns for `|remainder| < |c|`.
const COL_ABS_C: [usize; 4] = DIVREM_COL_MAP.abs_c.0;
const COL_ABS_REMAINDER: [usize; 4] = DIVREM_COL_MAP.abs_remainder.0;
const COL_MAX_ABS_C_OR_1: [usize; 4] = DIVREM_COL_MAP.max_abs_c_or_1.0;
const COL_BYTE_FLAGS: [usize; 4] = DIVREM_COL_MAP.remainder_lt_operation.byte_flags;
const COL_A_COMPARISON_BYTE: usize = DIVREM_COL_MAP.remainder_lt_operation.a_comparison_byte;
const COL_B_COMPARISON_BYTE: usize = DIVREM_COL_MAP.remainder_lt_operation.b_comparison_byte;
const COL_C_NEG_OP_VALUE: [usize; 4] = DIVREM_COL_MAP.c_neg_operation.value.0;
const COL_REM_NEG_OP_VALUE: [usize; 4] = DIVREM_COL_MAP.rem_neg_operation.value.0;

// Columns for `is_c_0` witness checks.
const COL_IS_C_0_BYTE_INV: [usize; 4] = [
    DIVREM_COL_MAP.is_c_0.is_zero_byte[0].inverse,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[1].inverse,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[2].inverse,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[3].inverse,
];
const COL_IS_C_0_BYTE_RESULT: [usize; 4] = [
    DIVREM_COL_MAP.is_c_0.is_zero_byte[0].result,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[1].result,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[2].result,
    DIVREM_COL_MAP.is_c_0.is_zero_byte[3].result,
];
const COL_IS_C_0_LOWER_HALF_ZERO: usize = DIVREM_COL_MAP.is_c_0.is_lower_half_zero;
const COL_IS_C_0_UPPER_HALF_ZERO: usize = DIVREM_COL_MAP.is_c_0.is_upper_half_zero;

// Columns for `is_overflow` witness checks.
const COL_IS_OVERFLOW: usize = DIVREM_COL_MAP.is_overflow;
const COL_OVERFLOW_B_BYTE_INV: [usize; 4] = [
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[0].inverse,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[1].inverse,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[2].inverse,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[3].inverse,
];
const COL_OVERFLOW_B_BYTE_RESULT: [usize; 4] = [
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[0].result,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[1].result,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[2].result,
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_zero_byte[3].result,
];
const COL_OVERFLOW_B_LOWER_HALF_ZERO: usize =
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_lower_half_zero;
const COL_OVERFLOW_B_UPPER_HALF_ZERO: usize =
    DIVREM_COL_MAP.is_overflow_b.is_diff_zero.is_upper_half_zero;
const COL_OVERFLOW_C_BYTE_INV: [usize; 4] = [
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[0].inverse,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[1].inverse,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[2].inverse,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[3].inverse,
];
const COL_OVERFLOW_C_BYTE_RESULT: [usize; 4] = [
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[0].result,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[1].result,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[2].result,
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_zero_byte[3].result,
];
const COL_OVERFLOW_C_LOWER_HALF_ZERO: usize =
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_lower_half_zero;
const COL_OVERFLOW_C_UPPER_HALF_ZERO: usize =
    DIVREM_COL_MAP.is_overflow_c.is_diff_zero.is_upper_half_zero;
const COL_OVERFLOW_B_RESULT: usize = DIVREM_COL_MAP.is_overflow_b.is_diff_zero.result;
const COL_OVERFLOW_C_RESULT: usize = DIVREM_COL_MAP.is_overflow_c.is_diff_zero.result;
const COL_ABS_C_ALU_EVENT: usize = DIVREM_COL_MAP.abs_c_alu_event;
const COL_ABS_REM_ALU_EVENT: usize = DIVREM_COL_MAP.abs_rem_alu_event;

// Columns for `c_times_quotient: MulOperation<T>`.
const COL_CTQ_PRODUCT: [usize; 8] = DIVREM_COL_MAP.c_times_quotient.product;
const COL_CTQ_CARRY: [usize; 8] = DIVREM_COL_MAP.c_times_quotient.carry;
const COL_CTQ_B_SIGN_EXTEND: usize = DIVREM_COL_MAP.c_times_quotient.b_sign_extend;
const COL_CTQ_C_SIGN_EXTEND: usize = DIVREM_COL_MAP.c_times_quotient.c_sign_extend;
const COL_CTQ_B_MSB: usize = DIVREM_COL_MAP.c_times_quotient.b_msb;
const COL_CTQ_C_MSB: usize = DIVREM_COL_MAP.c_times_quotient.c_msb;

// DivRem-owned carry witnesses for adding `remainder` into `c_times_quotient.product`.
const COL_DIVREM_CARRY: [usize; 8] = DIVREM_COL_MAP.carry;

/// Public values index for `execution_shard`.
const PV_EXECUTION_SHARD_IDX: usize = 44;

#[derive(Default, Clone, Copy)]
pub struct DivRemChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for DivRemChipPolyAir {
    fn width(&self) -> usize {
        NUM_DIVREM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Keep this ordered by the underlying DivRemCols trace offsets.
        vec![
            PairCol::Main(COL_SHARD),             // [0]   cpu_state.shard
            PairCol::Main(COL_OP_A_ZERO),         // [1]   mem_ops.op_a_zero
            PairCol::Main(COL_OP_A_VALUE[0]),     // [2]   mem_ops.op_a_access.access.value
            PairCol::Main(COL_OP_A_VALUE[1]),     // [3]
            PairCol::Main(COL_OP_A_VALUE[2]),     // [4]
            PairCol::Main(COL_OP_A_VALUE[3]),     // [5]
            PairCol::Main(COL_OP_B_VALUE[0]),     // [6]   mem_ops.op_b_access.access.value
            PairCol::Main(COL_OP_B_VALUE[1]),     // [7]
            PairCol::Main(COL_OP_B_VALUE[2]),     // [8]
            PairCol::Main(COL_OP_B_VALUE[3]),     // [9]
            PairCol::Main(COL_OP_C_VALUE[0]),     // [10]  mem_ops.op_c_access.access.value
            PairCol::Main(COL_OP_C_VALUE[1]),     // [11]
            PairCol::Main(COL_OP_C_VALUE[2]),     // [12]
            PairCol::Main(COL_OP_C_VALUE[3]),     // [13]
            PairCol::Main(COL_QUOTIENT[0]),       // [14]  quotient
            PairCol::Main(COL_QUOTIENT[1]),       // [15]
            PairCol::Main(COL_QUOTIENT[2]),       // [16]
            PairCol::Main(COL_QUOTIENT[3]),       // [17]
            PairCol::Main(COL_REMAINDER[0]),      // [18]  remainder
            PairCol::Main(COL_REMAINDER[1]),      // [19]
            PairCol::Main(COL_REMAINDER[2]),      // [20]
            PairCol::Main(COL_REMAINDER[3]),      // [21]
            PairCol::Main(COL_ABS_REMAINDER[0]),  // [22]  abs_remainder
            PairCol::Main(COL_ABS_REMAINDER[1]),  // [23]
            PairCol::Main(COL_ABS_REMAINDER[2]),  // [24]
            PairCol::Main(COL_ABS_REMAINDER[3]),  // [25]
            PairCol::Main(COL_ABS_C[0]),          // [26]  abs_c
            PairCol::Main(COL_ABS_C[1]),          // [27]
            PairCol::Main(COL_ABS_C[2]),          // [28]
            PairCol::Main(COL_ABS_C[3]),          // [29]
            PairCol::Main(COL_MAX_ABS_C_OR_1[0]), // [30]  max_abs_c_or_1
            PairCol::Main(COL_MAX_ABS_C_OR_1[1]), // [31]
            PairCol::Main(COL_MAX_ABS_C_OR_1[2]), // [32]
            PairCol::Main(COL_MAX_ABS_C_OR_1[3]), // [33]
            // IsZeroWord bytes are interleaved in trace order: inv, result, ...
            PairCol::Main(COL_IS_C_0_BYTE_INV[0]),     // [34]
            PairCol::Main(COL_IS_C_0_BYTE_RESULT[0]),  // [35]
            PairCol::Main(COL_IS_C_0_BYTE_INV[1]),     // [36]
            PairCol::Main(COL_IS_C_0_BYTE_RESULT[1]),  // [37]
            PairCol::Main(COL_IS_C_0_BYTE_INV[2]),     // [38]
            PairCol::Main(COL_IS_C_0_BYTE_RESULT[2]),  // [39]
            PairCol::Main(COL_IS_C_0_BYTE_INV[3]),     // [40]
            PairCol::Main(COL_IS_C_0_BYTE_RESULT[3]),  // [41]
            PairCol::Main(COL_IS_C_0_LOWER_HALF_ZERO), // [42]
            PairCol::Main(COL_IS_C_0_UPPER_HALF_ZERO), // [43]
            PairCol::Main(COL_IS_C_0_RESULT),          // [44]
            PairCol::Main(COL_C_NEG_OP_VALUE[0]),      // [45]  c_neg_operation.value
            PairCol::Main(COL_C_NEG_OP_VALUE[1]),      // [46]
            PairCol::Main(COL_C_NEG_OP_VALUE[2]),      // [47]
            PairCol::Main(COL_C_NEG_OP_VALUE[3]),      // [48]
            PairCol::Main(COL_REM_NEG_OP_VALUE[0]),    // [49]  rem_neg_operation.value
            PairCol::Main(COL_REM_NEG_OP_VALUE[1]),    // [50]
            PairCol::Main(COL_REM_NEG_OP_VALUE[2]),    // [51]
            PairCol::Main(COL_REM_NEG_OP_VALUE[3]),    // [52]
            PairCol::Main(COL_BYTE_FLAGS[0]),          // [53]  remainder_lt_operation.byte_flags
            PairCol::Main(COL_BYTE_FLAGS[1]),          // [54]
            PairCol::Main(COL_BYTE_FLAGS[2]),          // [55]
            PairCol::Main(COL_BYTE_FLAGS[3]),          // [56]
            PairCol::Main(COL_A_COMPARISON_BYTE),      // [57]
            PairCol::Main(COL_B_COMPARISON_BYTE),      // [58]
            PairCol::Main(COL_IS_DIV),                 // [59]
            PairCol::Main(COL_IS_DIVU),                // [60]
            PairCol::Main(COL_IS_REM),                 // [61]
            PairCol::Main(COL_IS_REMU),                // [62]
            PairCol::Main(COL_IS_OVERFLOW),            // [63]
            // is_overflow_b.is_diff_zero bytes are interleaved: inv, result, ...
            PairCol::Main(COL_OVERFLOW_B_BYTE_INV[0]), // [64]
            PairCol::Main(COL_OVERFLOW_B_BYTE_RESULT[0]), // [65]
            PairCol::Main(COL_OVERFLOW_B_BYTE_INV[1]), // [66]
            PairCol::Main(COL_OVERFLOW_B_BYTE_RESULT[1]), // [67]
            PairCol::Main(COL_OVERFLOW_B_BYTE_INV[2]), // [68]
            PairCol::Main(COL_OVERFLOW_B_BYTE_RESULT[2]), // [69]
            PairCol::Main(COL_OVERFLOW_B_BYTE_INV[3]), // [70]
            PairCol::Main(COL_OVERFLOW_B_BYTE_RESULT[3]), // [71]
            PairCol::Main(COL_OVERFLOW_B_LOWER_HALF_ZERO), // [72]
            PairCol::Main(COL_OVERFLOW_B_UPPER_HALF_ZERO), // [73]
            PairCol::Main(COL_OVERFLOW_B_RESULT),      // [74]
            // is_overflow_c.is_diff_zero bytes are interleaved: inv, result, ...
            PairCol::Main(COL_OVERFLOW_C_BYTE_INV[0]), // [75]
            PairCol::Main(COL_OVERFLOW_C_BYTE_RESULT[0]), // [76]
            PairCol::Main(COL_OVERFLOW_C_BYTE_INV[1]), // [77]
            PairCol::Main(COL_OVERFLOW_C_BYTE_RESULT[1]), // [78]
            PairCol::Main(COL_OVERFLOW_C_BYTE_INV[2]), // [79]
            PairCol::Main(COL_OVERFLOW_C_BYTE_RESULT[2]), // [80]
            PairCol::Main(COL_OVERFLOW_C_BYTE_INV[3]), // [81]
            PairCol::Main(COL_OVERFLOW_C_BYTE_RESULT[3]), // [82]
            PairCol::Main(COL_OVERFLOW_C_LOWER_HALF_ZERO), // [83]
            PairCol::Main(COL_OVERFLOW_C_UPPER_HALF_ZERO), // [84]
            PairCol::Main(COL_OVERFLOW_C_RESULT),      // [85]
            PairCol::Main(COL_B_MSB),                  // [86]
            PairCol::Main(COL_REM_MSB),                // [87]
            PairCol::Main(COL_C_MSB),                  // [88]
            PairCol::Main(COL_B_NEG),                  // [89]
            PairCol::Main(COL_REM_NEG),                // [90]
            PairCol::Main(COL_C_NEG),                  // [91]
            PairCol::Main(COL_ABS_C_ALU_EVENT),        // [92]
            PairCol::Main(COL_ABS_REM_ALU_EVENT),      // [93]
            PairCol::Main(COL_IS_REAL),                // [94]
            PairCol::Main(COL_REMAINDER_CHECK_MULT),   // [95]
            // c_times_quotient fields used by mul_op_gate_constraints.
            PairCol::Main(COL_CTQ_PRODUCT[0]),    // [96]
            PairCol::Main(COL_CTQ_PRODUCT[1]),    // [97]
            PairCol::Main(COL_CTQ_PRODUCT[2]),    // [98]
            PairCol::Main(COL_CTQ_PRODUCT[3]),    // [99]
            PairCol::Main(COL_CTQ_PRODUCT[4]),    // [100]
            PairCol::Main(COL_CTQ_PRODUCT[5]),    // [101]
            PairCol::Main(COL_CTQ_PRODUCT[6]),    // [102]
            PairCol::Main(COL_CTQ_PRODUCT[7]),    // [103]
            PairCol::Main(COL_CTQ_CARRY[0]),      // [104]
            PairCol::Main(COL_CTQ_CARRY[1]),      // [105]
            PairCol::Main(COL_CTQ_CARRY[2]),      // [106]
            PairCol::Main(COL_CTQ_CARRY[3]),      // [107]
            PairCol::Main(COL_CTQ_CARRY[4]),      // [108]
            PairCol::Main(COL_CTQ_CARRY[5]),      // [109]
            PairCol::Main(COL_CTQ_CARRY[6]),      // [110]
            PairCol::Main(COL_CTQ_CARRY[7]),      // [111]
            PairCol::Main(COL_CTQ_B_SIGN_EXTEND), // [112]
            PairCol::Main(COL_CTQ_C_SIGN_EXTEND), // [113]
            PairCol::Main(COL_CTQ_B_MSB),         // [114]
            PairCol::Main(COL_CTQ_C_MSB),         // [115]
            // DivRem-owned carry witnesses used by the carry-chain gate.
            PairCol::Main(COL_DIVREM_CARRY[0]), // [116]
            PairCol::Main(COL_DIVREM_CARRY[1]), // [117]
            PairCol::Main(COL_DIVREM_CARRY[2]), // [118]
            PairCol::Main(COL_DIVREM_CARRY[3]), // [119]
            PairCol::Main(COL_DIVREM_CARRY[4]), // [120]
            PairCol::Main(COL_DIVREM_CARRY[5]), // [121]
            PairCol::Main(COL_DIVREM_CARRY[6]), // [122]
            PairCol::Main(COL_DIVREM_CARRY[7]), // [123]
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let one = AB::VarMaybeExt::from(AB::F::one());

        // SAFETY: DivRemCols is #[repr(C)] and the main row has NUM_DIVREM_COLS columns.
        let local: &DivRemCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        // --- common derived values ---
        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        let opcode = local.is_divu.clone() *
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::DIVU as u8)) +
            local.is_remu.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::REMU as u8)) +
            local.is_div.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::DIV as u8)) +
            local.is_rem.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::REM as u8));

        let op_b_acc = &local.mem_ops.op_b_access.access;
        let op_c_acc = &local.mem_ops.op_c_access.access;
        let product = &local.c_times_quotient.product;

        // =========================================================================
        // #1–4: CPUState — 4 interactions
        // =========================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc.clone(),
            next_pc,
        );

        // =========================================================================
        // #5–17: RTypeRegisterOp — 13 interactions (1 program + 3×4 memory)
        // =========================================================================
        rtype_register_op_precompute_lc(
            builder,
            pc,
            opcode,
            local.mem_ops.op_a.clone(),
            local.mem_ops.op_b.clone(),
            local.mem_ops.op_c.clone(),
            local.mem_ops.op_a_zero.clone(),
            op_b_acc,
            op_c_acc,
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_a_access.prev_value,
            shard.clone(),
            clk,
        );

        // =========================================================================
        // #18–31: MulOperation — 14 interactions (2 MSB + 4 U8Range + 8 U16Range)
        // =========================================================================
        mul_op_precompute_lc(
            builder,
            local.c_times_quotient.b_msb.clone(),
            local.c_times_quotient.c_msb.clone(),
            op_c_acc.value[3].clone(),
            local.quotient[3].clone(),
            product,
            &local.c_times_quotient.carry,
        );

        // =========================================================================
        // #32: remainder_lt_operation (LTU) — 1 interaction
        // =========================================================================
        ltu_precompute_lc(
            builder,
            local.remainder_lt_operation.a_comparison_byte.clone(),
            local.remainder_lt_operation.b_comparison_byte.clone(),
        );

        // =========================================================================
        // #33–34: c_neg_operation (AddOperation) — 2 interactions
        // =========================================================================
        add_op_precompute_lc(builder, &local.c_neg_operation.value);

        // =========================================================================
        // #35–36: rem_neg_operation (AddOperation) — 2 interactions
        // =========================================================================
        add_op_precompute_lc(builder, &local.rem_neg_operation.value);

        // =========================================================================
        // #37–39: Explicit MSB checks (b, c, rem) — 3 interactions
        // =========================================================================
        msb_precompute_lc(builder, local.b_msb.clone(), op_b_acc.value[3].clone());
        msb_precompute_lc(builder, local.c_msb.clone(), op_c_acc.value[3].clone());
        msb_precompute_lc(builder, local.rem_msb.clone(), local.remainder[3].clone());

        // =========================================================================
        // #40–43: quotient/remainder U8Range — 4 interactions
        // =========================================================================
        slice_u8_range_precompute_lc(
            builder,
            &[
                local.quotient[0].clone(),
                local.quotient[1].clone(),
                local.quotient[2].clone(),
                local.quotient[3].clone(),
                local.remainder[0].clone(),
                local.remainder[1].clone(),
                local.remainder[2].clone(),
                local.remainder[3].clone(),
            ],
        );

        // =========================================================================
        // #44–45: Boolean constraints via BitVec (merged for efficiency)
        //
        // Boolean witnesses split across two BitVec lookups:
        //   #44: 15 opcode/control/sign bits
        //   #45: 8 sign-extension and overflow bits
        // =========================================================================

        // #44: BitVec [is_div, is_divu, is_rem, is_remu, op_a_zero, is_c_0,
        //              is_overflow, b_msb, rem_msb, c_msb, b_neg, rem_neg, c_neg,
        //              remainder_check_multiplicity, overflow_b_byte_result[0]] (15 bits)
        //
        // is_real is dropped from the payload because the BitVec mult is now
        // conditioned on it (see `lookup`); BitVec only enforces booleanness
        // when mult ≠ 0, so is_real's booleanness is instead asserted as an
        // explicit gate in `eval`.
        bitvec_precompute_lc(
            builder,
            vec![
                local.is_div.clone(),
                local.is_divu.clone(),
                local.is_rem.clone(),
                local.is_remu.clone(),
                local.mem_ops.op_a_zero.clone(),
                local.is_c_0.result.clone(),
                local.is_overflow.clone(),
                local.b_msb.clone(),
                local.rem_msb.clone(),
                local.c_msb.clone(),
                local.b_neg.clone(),
                local.rem_neg.clone(),
                local.c_neg.clone(),
                local.remainder_check_multiplicity.clone(),
                local.is_overflow_b.is_diff_zero.is_zero_byte[0].result.clone(),
            ],
        );

        // #45: BitVec [b_sign_extend, c_sign_extend, overflow_b_byte_result[1..3],
        //              overflow_b_lower, overflow_b_upper, overflow_c_byte_result[0]] (8 bits)
        //
        // DivRem carry witnesses are constrained in eval; keeping them out of
        // this payload keeps every precomputed LC degree-1.
        let sign_and_overflow_bits = vec![
            local.c_times_quotient.b_sign_extend.clone(),
            local.c_times_quotient.c_sign_extend.clone(),
            local.is_overflow_b.is_diff_zero.is_zero_byte[1].result.clone(),
            local.is_overflow_b.is_diff_zero.is_zero_byte[2].result.clone(),
            local.is_overflow_b.is_diff_zero.is_zero_byte[3].result.clone(),
            local.is_overflow_b.is_diff_zero.is_lower_half_zero.clone(),
            local.is_overflow_b.is_diff_zero.is_upper_half_zero.clone(),
            local.is_overflow_c.is_diff_zero.is_zero_byte[0].result.clone(),
        ];
        bitvec_precompute_lc(builder, sign_and_overflow_bits);

        // =========================================================================
        // #46: BitVec for AssertLtColsBytes byte_flags + IsZeroWord sub-results
        //      + is_overflow_c sub-results
        // =========================================================================
        bitvec_precompute_lc(
            builder,
            vec![
                local.remainder_lt_operation.byte_flags[0].clone(),
                local.remainder_lt_operation.byte_flags[1].clone(),
                local.remainder_lt_operation.byte_flags[2].clone(),
                local.remainder_lt_operation.byte_flags[3].clone(),
                local.is_c_0.is_zero_byte[0].result.clone(),
                local.is_c_0.is_zero_byte[1].result.clone(),
                local.is_c_0.is_zero_byte[2].result.clone(),
                local.is_c_0.is_zero_byte[3].result.clone(),
                local.is_c_0.is_lower_half_zero.clone(),
                local.is_c_0.is_upper_half_zero.clone(),
                // Overflow_c booleans packed into the remaining slots.
                local.is_overflow_c.is_diff_zero.is_zero_byte[1].result.clone(),
                local.is_overflow_c.is_diff_zero.is_zero_byte[2].result.clone(),
                local.is_overflow_c.is_diff_zero.is_zero_byte[3].result.clone(),
                local.is_overflow_c.is_diff_zero.is_lower_half_zero.clone(),
                local.is_overflow_c.is_diff_zero.is_upper_half_zero.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let one = AB::one_maybe();

        // Aliases follow reserved_poly order.
        let shard = local[0].clone();
        let op_a_zero = local[1].clone();
        let op_a_value: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[2 + i].clone());
        let op_b_value: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[6 + i].clone());
        let op_c_value: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[10 + i].clone());
        let quotient: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[14 + i].clone());
        let remainder: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[18 + i].clone());
        let abs_remainder: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[22 + i].clone());
        let abs_c: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[26 + i].clone());
        let max_abs_c_or_1: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[30 + i].clone());
        // Interleaved as inv0, result0, inv1, result1, ...
        let is_c_0_byte_inv: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[34 + 2 * i].clone());
        let is_c_0_byte_result: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[35 + 2 * i].clone());
        let is_c_0_lower_half_zero = local[42].clone();
        let is_c_0_upper_half_zero = local[43].clone();
        let is_c_0 = local[44].clone();
        let c_neg_op_value: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[45 + i].clone());
        let rem_neg_op_value: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[49 + i].clone());
        let byte_flags: [AB::VarMaybeExt; 4] = core::array::from_fn(|i| local[53 + i].clone());
        let a_comparison_byte = local[57].clone();
        let b_comparison_byte = local[58].clone();
        let is_div = local[59].clone();
        let is_divu = local[60].clone();
        let is_rem = local[61].clone();
        let is_remu = local[62].clone();
        let is_overflow = local[63].clone();
        // Interleaved as inv0, result0, ...
        let overflow_b_byte_inv: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[64 + 2 * i].clone());
        let overflow_b_byte_result: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[65 + 2 * i].clone());
        let overflow_b_lower = local[72].clone();
        let overflow_b_upper = local[73].clone();
        let overflow_b_result_col = local[74].clone();
        // Interleaved as inv0, result0, ...
        let overflow_c_byte_inv: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[75 + 2 * i].clone());
        let overflow_c_byte_result: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[76 + 2 * i].clone());
        let overflow_c_lower = local[83].clone();
        let overflow_c_upper = local[84].clone();
        let overflow_c_result_col = local[85].clone();
        let b_msb = local[86].clone();
        let rem_msb = local[87].clone();
        let c_msb = local[88].clone();
        let b_neg = local[89].clone();
        let rem_neg = local[90].clone();
        let c_neg = local[91].clone();
        let abs_c_alu_event = local[92].clone();
        let abs_rem_alu_event = local[93].clone();
        let is_real = local[94].clone();
        let remainder_check_mult = local[95].clone();

        // =================================================================
        // is_real boolean gate.
        // =================================================================
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // =================================================================
        // CPUState shard constraint
        // =================================================================
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(builder, shard, execution_shard, is_real.clone());

        // =================================================================
        // One-hot opcode selector
        // =================================================================
        builder.assert_eq(
            is_div.clone() + is_divu.clone() + is_rem.clone() + is_remu.clone(),
            one.clone(),
        );

        // =================================================================
        // is_signed_type linkage: *_neg = *_msb * (is_div + is_rem)
        // =================================================================
        let is_signed = is_div.clone() + is_rem.clone();
        builder.assert_eq(b_neg.clone(), b_msb * is_signed.clone());
        builder.assert_eq(rem_neg.clone(), rem_msb * is_signed.clone());
        builder.assert_eq(c_neg.clone(), c_msb * is_signed.clone());

        // =================================================================
        // is_overflow derivation (mod.rs:525-551)
        //
        // is_overflow_b: b_word == i32::MIN (0x80000000)
        // is_overflow_c: c_word == -1       (0xFFFFFFFF)
        // is_overflow = result_b * result_c * is_signed
        // =================================================================
        {
            let i32_min: [AB::VarMaybeExt; 4] = [
                AB::VarMaybeExt::from(AB::F::zero()),
                AB::VarMaybeExt::from(AB::F::zero()),
                AB::VarMaybeExt::from(AB::F::zero()),
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(0x80)),
            ];
            // The result column is materialized so downstream products stay low-degree.
            is_equal_word_gate_constraints(
                builder,
                op_b_value.clone(),
                i32_min,
                overflow_b_byte_inv,
                overflow_b_byte_result,
                overflow_b_lower,
                overflow_b_upper,
                overflow_b_result_col.clone(),
                is_real.clone(),
            );

            let neg_one: [AB::VarMaybeExt; 4] = [
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xFF)),
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xFF)),
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xFF)),
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xFF)),
            ];
            let overflow_c_result = overflow_c_result_col.clone();
            is_equal_word_gate_constraints(
                builder,
                op_c_value.clone(),
                neg_one,
                overflow_c_byte_inv,
                overflow_c_byte_result,
                overflow_c_lower,
                overflow_c_upper,
                overflow_c_result.clone(),
                is_real.clone(),
            );

            // is_overflow = result_b * result_c * is_signed.
            builder.assert_eq(
                is_overflow.clone(),
                overflow_b_result_col * overflow_c_result_col * is_signed.clone(),
            );
        }

        // =================================================================
        // Sign consistency rule 1: rem_neg implies b_neg.
        // =================================================================
        builder.when(rem_neg.clone()).assert_eq(b_neg.clone(), one.clone());

        // =================================================================
        // Sign consistency rule 2: rem > 0 implies b >= 0.
        // rem_byte_sum * (1 - rem_neg) * b_neg = 0
        // =================================================================
        let rem_byte_sum = remainder[0].clone() +
            remainder[1].clone() +
            remainder[2].clone() +
            remainder[3].clone();
        builder.when((one.clone() - rem_neg.clone()) * b_neg.clone()).assert_zero(rem_byte_sum);

        // =================================================================
        // Output selection (8 constraints) — mod.rs:613-622
        // =================================================================
        let op_a_not_0 = one.clone() - op_a_zero.clone();
        let is_div_type = is_div.clone() + is_divu.clone();
        let is_rem_type = is_rem.clone() + is_remu;
        for i in 0..4 {
            builder
                .when(op_a_not_0.clone())
                .when(is_div_type.clone())
                .assert_eq(quotient[i].clone(), op_a_value[i].clone());
            builder
                .when(op_a_not_0.clone())
                .when(is_rem_type.clone())
                .assert_eq(remainder[i].clone(), op_a_value[i].clone());
        }

        // =================================================================
        // Division-by-zero quotient rule (4 constraints) — mod.rs:661-665
        // =================================================================
        let u8_max = AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xff));
        for i in 0..4 {
            builder
                .when(is_c_0.clone() * is_div_type.clone())
                .assert_eq(quotient[i].clone(), u8_max.clone());
        }

        // =================================================================
        // is_c_0 — IsZeroWordOperation gate constraints
        //
        // Covers both directions:
        //   Direction 2: byte_result[i] = 1 - inverse[i] * c_word[i],
        //                half/full product chain
        //   Direction 1: byte_result[i] = 1 → c_word[i] = 0 (per-byte)
        //
        // Boolean constraints on byte_result, half_zero, is_c_0 are via BitVec #54.
        // =================================================================
        is_zero_word_gate_constraints(
            builder,
            op_c_value.clone(),
            is_c_0_byte_inv,
            is_c_0_byte_result,
            is_c_0_lower_half_zero,
            is_c_0_upper_half_zero,
            is_c_0.clone(),
            is_real.clone(),
        );

        // =================================================================
        // remainder_check_multiplicity exact (1 constraint) — mod.rs:702-705
        // =================================================================
        builder.assert_eq(
            remainder_check_mult.clone(),
            (one.clone() - is_c_0.clone()) * is_real.clone(),
        );

        // =================================================================
        // abs_c == c_word when not c_neg.
        // (1 - c_neg) * (abs_c[i] - c_word[i]) = 0
        // =================================================================
        for i in 0..4 {
            builder
                .when_ne(c_neg.clone(), one.clone())
                .assert_eq(abs_c[i].clone(), op_c_value[i].clone());
        }

        // =================================================================
        // abs_remainder == remainder when not rem_neg.
        // =================================================================
        for i in 0..4 {
            builder
                .when_ne(rem_neg.clone(), one.clone())
                .assert_eq(abs_remainder[i].clone(), remainder[i].clone());
        }

        // =================================================================
        // MulOperation gate constraints: c_word × quotient = c_times_quotient
        //
        // The carry chain is enforced as gates instead of derived LC lookups.
        // Reserved offsets [96..115] expose the c_times_quotient witness columns.
        // =================================================================
        {
            let ctq_product: [AB::VarMaybeExt; 8] = core::array::from_fn(|i| local[96 + i].clone());
            let ctq_carry: [AB::VarMaybeExt; 8] = core::array::from_fn(|i| local[104 + i].clone());
            let ctq_b_sign_extend = local[112].clone();
            let ctq_c_sign_extend = local[113].clone();
            let ctq_b_msb = local[114].clone();
            let ctq_c_msb = local[115].clone();
            let is_signed_mul_b = is_div.clone() + is_rem.clone();
            let is_signed_mul_c = is_div.clone() + is_rem.clone();
            mul_op_gate_constraints(
                builder,
                op_c_value.clone(),
                quotient.clone(),
                ctq_product,
                ctq_carry,
                ctq_b_msb,
                ctq_c_msb,
                ctq_b_sign_extend,
                ctq_c_sign_extend,
                is_signed_mul_b,
                is_signed_mul_c,
                is_real.clone(),
            );
        }

        // =================================================================
        // DivRem carry-chain gate (mirrors mod.rs:553-607)
        //
        // Adds `remainder` (sign-extended on the upper 4 bytes) into
        // `c_times_quotient.product`, propagates carry, and asserts the
        // resulting 8-limb value equals `b` (lower 4 bytes) and matches the
        // sign-extended/overflow pattern (upper 4 bytes).
        //
        // Carry witnesses live at reserved indices [116..123] and are boolean.
        // =================================================================
        {
            let divrem_carry: [AB::VarMaybeExt; 8] =
                core::array::from_fn(|i| local[116 + i].clone());
            let ctq_product: [AB::VarMaybeExt; 8] = core::array::from_fn(|i| local[96 + i].clone());
            let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8));
            let u8_max = AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xff));

            // Boolean range for each carry (mod.rs:755-757).
            for i in 0..8 {
                builder
                    .assert_zero(divrem_carry[i].clone() * (one.clone() - divrem_carry[i].clone()));
            }

            let sign_extension = rem_neg.clone() * u8_max.clone();
            for i in 0..8 {
                let mut acc = ctq_product[i].clone();
                if i < 4 {
                    acc = acc + remainder[i].clone();
                } else {
                    acc = acc + sign_extension.clone();
                }
                acc = acc - divrem_carry[i].clone() * base.clone();
                if i > 0 {
                    acc = acc + divrem_carry[i - 1].clone();
                }

                if i < 4 {
                    // Lower 4 bytes must match b (mod.rs:587).
                    builder.assert_eq(op_b_value[i].clone(), acc);
                } else {
                    // Upper 4 bytes (mod.rs:589-605):
                    //   when (1 - is_overflow) * b_neg:        acc == 0xff
                    //   when (1 - is_overflow) * (1 - b_neg):  acc == 0
                    //   when is_overflow:                       acc == 0
                    let not_overflow = one.clone() - is_overflow.clone();
                    builder
                        .when(not_overflow.clone())
                        .when(b_neg.clone())
                        .assert_eq(acc.clone(), u8_max.clone());
                    builder
                        .when(not_overflow.clone())
                        .when_ne(one.clone(), b_neg.clone())
                        .assert_zero(acc.clone());
                    builder.when(is_overflow.clone()).assert_zero(acc);
                }
            }
        }

        // =================================================================
        // AddOperation carry gates.
        //
        // c_neg_operation: c_word + abs_c = c_neg_op_value (mod 2^32)
        // rem_neg_operation: remainder + abs_remainder = rem_neg_op_value (mod 2^32)
        //
        // These are only populated (and thus only valid) when c_neg/rem_neg
        // is set, matching the original Air which uses abs_c_alu_event =
        // c_neg * is_real and abs_rem_alu_event = rem_neg * is_real.
        // =================================================================
        // Materialize these multiplicities so add_op_gate_constraints receives
        // degree-1 selectors.
        builder.assert_eq(abs_c_alu_event.clone(), c_neg.clone() * is_real.clone());
        builder.assert_eq(abs_rem_alu_event.clone(), rem_neg.clone() * is_real.clone());
        add_op_gate_constraints(
            builder,
            op_c_value,
            abs_c.clone(),
            c_neg_op_value,
            abs_c_alu_event,
        );
        add_op_gate_constraints(
            builder,
            remainder.clone(),
            abs_remainder.clone(),
            rem_neg_op_value,
            abs_rem_alu_event,
        );

        // =================================================================
        // max_abs_c_or_1 computation.
        // max(abs(c), 1) = abs_c * (1 - is_c_0) + 1 * is_c_0
        // =================================================================
        {
            // Byte 0: max_abs_c_or_1[0] = is_c_0 * 1 + (1 - is_c_0) * abs_c[0]
            let expected_0 =
                is_c_0.clone() * one.clone() + (one.clone() - is_c_0.clone()) * abs_c[0].clone();
            builder.assert_eq(max_abs_c_or_1[0].clone(), expected_0);
            // Bytes 1..3: max_abs_c_or_1[i] = (1 - is_c_0) * abs_c[i]
            for i in 1..4 {
                builder.assert_eq(
                    max_abs_c_or_1[i].clone(),
                    (one.clone() - is_c_0.clone()) * abs_c[i].clone(),
                );
            }
        }

        // =================================================================
        // AssertLtColsBytes internal gates (audit §11, item #7)
        //
        // Gated on remainder_check_mult (= (1-is_c_0)*is_real), so the
        // comparison is only enforced when c ≠ 0.
        // Boolean enforcement of byte_flags is via BitVec #54.
        // =================================================================
        assert_lt_bytes_gate_constraints(
            builder,
            abs_remainder,
            abs_c.clone(),
            byte_flags,
            a_comparison_byte.clone(),
            b_comparison_byte.clone(),
            remainder_check_mult.clone(),
        );

        // =================================================================
        // RTypeRegisterOp gates (5 constraints)
        // =================================================================
        rtype_register_op_gate_constraints(builder, op_a_zero, op_a_value, is_real);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[94].clone();
        let remainder_check_mult = local[95].clone();
        // Read materialized multiplicities so lookup selectors stay degree-1.
        let abs_c_alu_mult = local[92].clone();
        let abs_rem_alu_mult = local[93].clone();

        // #1–4: CPUState
        cpu_state_lookup(builder, is_real.clone());
        // #5–17: RTypeRegisterOp (1 program + 3×4 memory)
        rtype_register_op_lookup(builder, is_real.clone());
        // #18–31: MulOperation (2 MSB + 4 U8Range + 8 U16Range)
        mul_op_lookup(builder, is_real.clone());
        // #32: LTU
        ltu_lookup(builder, remainder_check_mult);
        // #33–34: c_neg_operation AddOp
        add_op_lookup(builder, abs_c_alu_mult);
        // #35–36: rem_neg_operation AddOp
        add_op_lookup(builder, abs_rem_alu_mult);
        // #37–39: explicit MSB (b, c, rem)
        msb_lookup(builder, is_real.clone());
        msb_lookup(builder, is_real.clone());
        msb_lookup(builder, is_real.clone());
        // #40–43: quotient/remainder U8Range (4 pairs)
        slice_u8_range_lookup(builder, is_real.clone(), 4);
        // #44–45: BitVec × 2 (mult = is_real)
        // All payload bits in #44 and #45 are derived from per-event populate
        // calls that run for every real row (independent of op_a_0), so the
        // conditioning is just `is_real`. On padding rows is_real = 0 ⇒ no
        // emission. is_real is enforced boolean by an explicit gate in `eval`.
        bitvec_lookup(builder, is_real.clone());
        bitvec_lookup(builder, is_real.clone());
        // #46: BitVec (mult = is_real)
        // byte_flags are populated only when remainder_check_mult = 1, but the
        // remaining bits (is_c_0, overflow_c internals) are populated for every
        // real row. We emit one BitVec per real row; on rows with c = 0 the
        // byte_flags bits are 0 (skipped populate), producing a valid pattern
        // that ByteChip's BitVec row still matches.
        bitvec_lookup(builder, is_real.clone());
    }
}

/// Pack a slice of field-element bits (each must be 0 or 1) into a u16 value
/// suitable for `add_bit_vec_lookup`. Bit `i` of the result equals `bits[i]`.
///
/// The trace columns that feed BitVec payloads in DivRem are all populated
/// with `F::from_bool(_)` or `F::from_canonical_u32(0 | 1)`, so the `is_one()`
/// check is exact on real rows. Padding rows are not iterated.
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
// MachineAir implementation (delegation to DivRemChip)
// =============================================================================

impl<F: Field> BaseAir<F> for DivRemChipPolyAir {
    fn width(&self) -> usize {
        NUM_DIVREM_COLS
    }
}

impl<F: Field> MachineAir<F> for DivRemChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "DivRemPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        DivRemChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::Borrow;
        use dt_core_executor::events::ByteRecord;

        if input.divrem_events.is_empty() {
            return;
        }

        // DivRem's base BLU emission happens inside `generate_trace` (the base
        // chip leaves `generate_dependencies` as default). The PolyAir prover
        // path calls `generate_dependencies` and `generate_trace` separately on
        // *different* records, so we must mirror the base BLU emission here by
        // running the base trace gen into `output`. We then reuse the produced
        // trace rows to compute the 3 BitVec payloads (PolyAir-only lookups
        // that the base chip never emits).
        let trace = DivRemChip.generate_trace(input, output).decompress();

        for row_idx in 0..input.divrem_events.len() {
            let row_start = row_idx * NUM_DIVREM_COLS;
            let row: &[F] = &trace.values[row_start..row_start + NUM_DIVREM_COLS];
            let cols: &DivRemCols<F> = row.borrow();

            // BitVec #44 (15 bits, in precompute_lc order — is_real dropped)
            output.add_bit_vec_lookup(pack_bits(&[
                cols.is_div,
                cols.is_divu,
                cols.is_rem,
                cols.is_remu,
                cols.mem_ops.op_a_zero,
                cols.is_c_0.result,
                cols.is_overflow,
                cols.b_msb,
                cols.rem_msb,
                cols.c_msb,
                cols.b_neg,
                cols.rem_neg,
                cols.c_neg,
                cols.remainder_check_multiplicity,
                cols.is_overflow_b.is_diff_zero.is_zero_byte[0].result,
            ]));

            // BitVec #45 (8 bits, in precompute_lc order)
            output.add_bit_vec_lookup(pack_bits(&[
                cols.c_times_quotient.b_sign_extend,
                cols.c_times_quotient.c_sign_extend,
                cols.is_overflow_b.is_diff_zero.is_zero_byte[1].result,
                cols.is_overflow_b.is_diff_zero.is_zero_byte[2].result,
                cols.is_overflow_b.is_diff_zero.is_zero_byte[3].result,
                cols.is_overflow_b.is_diff_zero.is_lower_half_zero,
                cols.is_overflow_b.is_diff_zero.is_upper_half_zero,
                cols.is_overflow_c.is_diff_zero.is_zero_byte[0].result,
            ]));

            // BitVec #54 (15 bits — byte_flags + is_c_0 internals + overflow_c internals)
            output.add_bit_vec_lookup(pack_bits(&[
                cols.remainder_lt_operation.byte_flags[0],
                cols.remainder_lt_operation.byte_flags[1],
                cols.remainder_lt_operation.byte_flags[2],
                cols.remainder_lt_operation.byte_flags[3],
                cols.is_c_0.is_zero_byte[0].result,
                cols.is_c_0.is_zero_byte[1].result,
                cols.is_c_0.is_zero_byte[2].result,
                cols.is_c_0.is_zero_byte[3].result,
                cols.is_c_0.is_lower_half_zero,
                cols.is_c_0.is_upper_half_zero,
                cols.is_overflow_c.is_diff_zero.is_zero_byte[1].result,
                cols.is_overflow_c.is_diff_zero.is_zero_byte[2].result,
                cols.is_overflow_c.is_diff_zero.is_zero_byte[3].result,
                cols.is_overflow_c.is_diff_zero.is_lower_half_zero,
                cols.is_overflow_c.is_diff_zero.is_upper_half_zero,
            ]));
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <DivRemChip as MachineAir<F>>::included(&DivRemChip, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        DivRemChip.padding_row()
    }

    fn local_only(&self) -> bool {
        <DivRemChip as MachineAir<F>>::local_only(&DivRemChip)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::DivRemChip, *};

    /// Total lookup interactions emitted by lookup().
    const NUM_LOOKUPS: usize = 46;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;
    use dt_core_executor::{ExecutionRecord, Executor, Instruction, Opcode, Program};
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
            MachineAir,
        },
        DTCoreOpts,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

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
        let required_max_beta_power = <DivRemChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&DivRemChipPolyAir);
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
        // Gate constraints: 135.
        // Lookup batch: ceil(46/3) = 16
        // Cumulative sum: 3
        // Total: 135 + 16 + 3 = 154
        const NUM_GATE_CONSTRAINTS: usize = 135;
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
        air: &DivRemChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <DivRemChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    fn simple_divrem_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, 100, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 7, false, true),
            Instruction::new(Opcode::DIV, 10, 5, 6, false, false),
            Instruction::new(Opcode::DIVU, 11, 5, 6, false, false),
            Instruction::new(Opcode::REM, 12, 5, 6, false, false),
            Instruction::new(Opcode::REMU, 13, 5, 6, false, false),
            Instruction::new(Opcode::ADD, 7, 0, 0xfffffff9, false, true),
            Instruction::new(Opcode::ADD, 8, 0, 0xfffffffd, false, true),
            Instruction::new(Opcode::DIV, 14, 7, 8, false, false),
            Instruction::new(Opcode::REM, 15, 7, 8, false, false),
            Instruction::new(Opcode::ADD, 9, 0, 0, false, true),
            Instruction::new(Opcode::DIVU, 16, 5, 9, false, false),
            Instruction::new(Opcode::REMU, 17, 5, 9, false, false),
            Instruction::new(Opcode::ADD, 18, 0, 0x80000000, false, true),
            Instruction::new(Opcode::ADD, 19, 0, 0xffffffff, false, true),
            Instruction::new(Opcode::DIV, 20, 18, 19, false, false),
            Instruction::new(Opcode::REM, 21, 18, 19, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        let program = simple_divrem_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = DivRemChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = DivRemChipPolyAir;
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

    fn random_divrem_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;

        let base = sample_trace();
        let base_height = base.height();
        assert!(base_height >= 1, "sample_trace returned empty trace");
        assert!(
            target_height >= base_height,
            "target 2^{} = {} is smaller than sample_trace height {}",
            log_n,
            target_height,
            base_height,
        );

        if target_height == base_height {
            return base;
        }

        let last_row_start = (base_height - 1) * NUM_DIVREM_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_DIVREM_COLS];

        let mut values = Vec::with_capacity(target_height * NUM_DIVREM_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }

        RowMajorMatrix::new(values, NUM_DIVREM_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = DivRemChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_divrem_trace(log_n, seed);
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
            <DivRemChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    fn only_x0_divrem_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, 100, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 7, false, true),
            Instruction::new(Opcode::DIV, 0, 5, 6, false, false),
            Instruction::new(Opcode::DIVU, 0, 5, 6, false, false),
            Instruction::new(Opcode::REM, 0, 5, 6, false, false),
            Instruction::new(Opcode::REMU, 0, 5, 6, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// DivRem emits 3 BitVec lookups per real event (one each for #44, #45, #54).
    /// All payload bits derive from columns populated for every real row
    /// (independent of op_a_0), so emit count equals 3 × divrem_events.len().
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_divrem_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <DivRemChipPolyAir as MachineAir<F>>::generate_dependencies(
            &DivRemChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = 3 * shard.divrem_events.len();
        assert!(expected > 0, "fixture must include divrem events");
        assert_eq!(
            bitvec_total, expected,
            "BitVec BLU emit count must equal 3 × event count (one per BitVec lookup)",
        );
    }

    /// Program where every DivRem writes to x0. BitVec still emits because
    /// the payload bits derive from columns populated regardless of op_a_0.
    #[test]
    fn bitvec_emitted_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_divrem_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(
            shard.divrem_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all divrem events must be op_a_0=true",
        );
        let expected = 3 * shard.divrem_events.len();
        assert!(expected > 0, "fixture must yield divrem events");

        let mut deps = ExecutionRecord::default();
        <DivRemChipPolyAir as MachineAir<F>>::generate_dependencies(
            &DivRemChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_count: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();
        assert_eq!(
            bitvec_count, expected,
            "op_a_0=true rows must still emit BitVec for DivRem chip",
        );
    }
}
