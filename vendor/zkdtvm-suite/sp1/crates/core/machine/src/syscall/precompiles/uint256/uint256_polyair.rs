//! PolyAir adaptation of Uint256MulChip.
//!
//! Bridges `Uint256MulCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (195 total)
//!
//!   #1  ..#95:  output FieldOpCols range checks (16 U8Range result + 16 U8Range carry + 63
//! U16Range witness)   #96 ..#98:  output_range_check FieldLtCols (1 LTU + 2 BitVec for 32
//! byte_flags)   #99 ..#130: x_memory readwrite (8 words × 4 interactions each)
//!   #131..#162: y_memory read (8 words × 4 interactions each)
//!   #163..#194: modulus_memory read (8 words × 4 interactions each)
//!   #195:       recv(Syscall)
//!
//!   Plus 6 precomputed β-evals → NUM_LOOKUPS + 6 precomputed total.
//!     [+0] output_witness_beta
//!     [+1] x_beta            (β eval of x_memory[*].prev_value)
//!     [+2] y_beta            (β eval of y_memory[*].access.value)
//!     [+3] result_beta       (β eval of output.result)
//!     [+4] carry_beta        (β eval of output.carry)
//!     [+5] diff_beta         (assert_all_eq(output.result, x_value))
//!
//! ## Boolean handling (≤3 → direct gate constraints)
//!   - is_real, modulus_is_zero.result, modulus_is_not_zero → gate assert_zero(x*(1-x))
//!
//! ## Dynamic modulus
//!   The effective modulus is `modulus_limbs * (1 - modulus_is_zero) + 2^256 * modulus_is_zero`.
//!   Gate constraints build the vanishing polynomial manually using `field_op_gate_constraints`.
//!
//! ## reserved_poly minimization
//!   Skipped (consumed only inside `precompute_lc` as lookup denominators or β-evals):
//!     - x_ptr, y_ptr                            (address inputs to memory LCs only)
//!     - x_memory[i].prev_value                  (collapsed into x_beta)
//!     - x_memory[i].access.value                (collapsed into diff_beta)
//!     - y_memory[i].access.value                (collapsed into y_beta)
//!     - output.carry                            (collapsed into carry_beta)
//!     - output.witness                          (collapsed into output_witness_beta)
//!   Kept: is_real, shard, clk, modulus_is_zero.{inverse,result}, modulus_is_not_zero,
//!         output.result (needed limb-wise by field_lt),
//!         modulus_memory[i].access.value (needed limb-wise by field_lt + IsZeroOp sum),
//!         output_range_check (full),
//!         {x, y, modulus}_memory[i] timestamp fields.

use std::ops::Deref;

use crate::{
    memory::{
        polyair::{
            memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
            memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
        },
        MemoryAccessCols,
    },
    operations::{
        field::{
            field_op::{
                field_op_beta_from_coeffs, field_op_gate_constraints, field_op_lookup,
                field_op_num_interactions, field_op_precompute_lc,
                field_op_precompute_witness_beta, FieldOpBetaConsts,
            },
            range::{
                field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
                field_lt_precompute_lc, FieldLtCols,
            },
        },
        is_zero::is_zero_op_gate_constraints,
    },
};
use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    params::{Limbs, NumLimbs},
    uint256::U256Field,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol, Polynomial},
    InteractionKind, Word,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::air::{Uint256MulCols, NUM_COLS, WORDS_FIELD_ELEMENT};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
///
/// = field_op_num_interactions<U256Field>  (95: 16 U8Range result + 16 U8Range carry + 63 U16Range
/// witness)
/// + field_lt_num_interactions<U256Field>  (3: 1 LTU + 2 BitVec)
/// + WORDS_FIELD_ELEMENT * 4              (32: x_memory readwrite)
/// + WORDS_FIELD_ELEMENT * 4              (32: y_memory read)
/// + WORDS_FIELD_ELEMENT * 4              (32: modulus_memory read)
/// + 1                                    (recv Syscall)
const NUM_LOOKUPS: usize = field_op_num_interactions::<U256Field>() +
    field_lt_num_interactions::<U256Field>() +
    WORDS_FIELD_ELEMENT * 4 +
    WORDS_FIELD_ELEMENT * 4 +
    WORDS_FIELD_ELEMENT * 4 +
    1;

/// Precomputed linear combinations: one per lookup + 5 β-evals + 1 assert_all_eq diff.
const NUM_PRECOMPUTED: usize = NUM_LOOKUPS + 6;
const OUTPUT_WITNESS_BETA_IDX: usize = NUM_LOOKUPS;
const X_BETA_IDX: usize = NUM_LOOKUPS + 1;
const Y_BETA_IDX: usize = NUM_LOOKUPS + 2;
const RESULT_BETA_IDX: usize = NUM_LOOKUPS + 3;
const CARRY_BETA_IDX: usize = NUM_LOOKUPS + 4;
const DIFF_BETA_IDX: usize = NUM_LOOKUPS + 5;

/// Maximum number of values in a single lookup payload.
/// BitVec payload from FieldLtCols byte_flags is 16 — the largest payload.
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets (byte index within Uint256MulCols<u8>).
//
// Layout (#[repr(C)], from air.rs):
//   [0]  shard
//   [1]  clk
//   [2]  x_ptr
//   [3]  y_ptr
//   [4 ..  4+ 8*13=108]   x_memory[8]           (MemoryWriteCols, 13 each)
//   [108..108+ 8* 9=180]  y_memory[8]           (MemoryReadCols,  9 each)
//   [180..180+ 8* 9=252]  modulus_memory[8]     (MemoryReadCols,  9 each)
//   [252..254]            modulus_is_zero       (inverse, result)
//   [254]                 modulus_is_not_zero
//   [255..255+127=382]    output                (FieldOpCols: result 32 + carry 32 + witness 63)
//   [382..382+34=416]     output_range_check    (FieldLtCols: byte_flags 32 + 2 comparison bytes)
//   [416]                 is_real
//   NUM_COLS = 417
// ============================================================================

const COL_SHARD: usize = 0;
const COL_CLK: usize = 1;
const COL_X_MEM_BASE: usize = 4;
const COL_Y_MEM_BASE: usize = COL_X_MEM_BASE + WORDS_FIELD_ELEMENT * MEM_WRITE_COLS_SIZE;
const COL_MOD_MEM_BASE: usize = COL_Y_MEM_BASE + WORDS_FIELD_ELEMENT * MEM_READ_COLS_SIZE;
const COL_MOD_IS_ZERO_INVERSE: usize = COL_MOD_MEM_BASE + WORDS_FIELD_ELEMENT * MEM_READ_COLS_SIZE;
const COL_MOD_IS_ZERO_RESULT: usize = COL_MOD_IS_ZERO_INVERSE + 1;
const COL_MOD_IS_NOT_ZERO: usize = COL_MOD_IS_ZERO_RESULT + 1;
const COL_OUTPUT_BASE: usize = COL_MOD_IS_NOT_ZERO + 1;
const COL_OUTPUT_RANGE_CHECK_BASE: usize = COL_OUTPUT_BASE + NB_LIMBS + NB_LIMBS + NB_WITNESS_LIMBS;
const COL_IS_REAL: usize = NUM_COLS - 1;

const NB_LIMBS: usize = 32; // <U256Field as FieldParameters>::NB_LIMBS
const NB_WITNESS_LIMBS: usize = 63; // <U256Field as NumLimbs>::Witness::USIZE

const MEM_READ_COLS_SIZE: usize = 9;
const MEM_WRITE_COLS_SIZE: usize = 13;
// MemoryReadCols offsets:
const MEM_READ_VALUE_OFF: usize = 0;
const MEM_READ_PREV_SHARD_OFF: usize = 4;
const MEM_READ_PREV_CLK_OFF: usize = 5;
const MEM_READ_COMPARE_CLK_OFF: usize = 6;
const MEM_READ_DIFF_16_OFF: usize = 7;
const MEM_READ_DIFF_12_OFF: usize = 8;
// MemoryWriteCols offsets (prev_value 0..4 + access at 4..13):
const MEM_WRITE_PREV_SHARD_OFF: usize = 8;
const MEM_WRITE_PREV_CLK_OFF: usize = 9;
const MEM_WRITE_COMPARE_CLK_OFF: usize = 10;
const MEM_WRITE_DIFF_16_OFF: usize = 11;
const MEM_WRITE_DIFF_12_OFF: usize = 12;

// ============================================================================
// reserved_poly slice layout.
//
//   [0]   is_real
//   [1]   shard
//   [2]   clk
//   [3]   modulus_is_zero.inverse
//   [4]   modulus_is_zero.result
//   [5]   modulus_is_not_zero
//   [6..38]    output.result (NB_LIMBS = 32)
//   [38..70]   modulus_memory[i].access.value  (8 × 4 = 32)
//   [70..102]  output_range_check.byte_flags   (32)
//   [102]      output_range_check.lhs_comparison_byte
//   [103]      output_range_check.rhs_comparison_byte
//   [104..144] x_memory[i] timestamps (5 each × 8 = 40)
//   [144..184] y_memory[i] timestamps (5 × 8 = 40)
//   [184..224] modulus_memory[i] timestamps (5 × 8 = 40)
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_SHARD: usize = 1;
const RES_CLK: usize = 2;
const RES_MOD_IS_ZERO_INVERSE: usize = 3;
const RES_MOD_IS_ZERO_RESULT: usize = 4;
const RES_MOD_IS_NOT_ZERO: usize = 5;
const RES_OUTPUT_RESULT_BASE: usize = 6;
const RES_MOD_MEM_VALUE_BASE: usize = RES_OUTPUT_RESULT_BASE + NB_LIMBS;
const RES_OR_BYTE_FLAGS_BASE: usize = RES_MOD_MEM_VALUE_BASE + WORDS_FIELD_ELEMENT * 4;
const RES_OR_LHS_BYTE: usize = RES_OR_BYTE_FLAGS_BASE + NB_LIMBS;
const RES_OR_RHS_BYTE: usize = RES_OR_LHS_BYTE + 1;
const RES_PER_MEM_TS: usize = 5;
const RES_X_MEM_BASE: usize = RES_OR_RHS_BYTE + 1;
const RES_Y_MEM_BASE: usize = RES_X_MEM_BASE + WORDS_FIELD_ELEMENT * RES_PER_MEM_TS;
const RES_MOD_MEM_TS_BASE: usize = RES_Y_MEM_BASE + WORDS_FIELD_ELEMENT * RES_PER_MEM_TS;
const RES_LEN: usize = RES_MOD_MEM_TS_BASE + WORDS_FIELD_ELEMENT * RES_PER_MEM_TS;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for Uint256MulChip.
#[derive(Clone, Copy, Default)]
pub struct Uint256MulPolyAir;

impl Uint256MulPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for Uint256MulPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<U256Field>(
            MAX_LOOKUP_VALUES,
        )
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // See "reserved_poly slice layout" comment at top of file.
        let mut cols: Vec<PairCol> = Vec::with_capacity(RES_LEN);

        // Scalars.
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_MOD_IS_ZERO_INVERSE));
        cols.push(PairCol::Main(COL_MOD_IS_ZERO_RESULT));
        cols.push(PairCol::Main(COL_MOD_IS_NOT_ZERO));

        // output.result limbs (needed limb-wise by field_lt_gate_constraints).
        for k in 0..NB_LIMBS {
            cols.push(PairCol::Main(COL_OUTPUT_BASE + k));
        }

        // modulus_memory[i].access.value (needed limb-wise by field_lt + IsZeroOp sum).
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_MOD_MEM_BASE + i * MEM_READ_COLS_SIZE;
            for k in 0..4 {
                cols.push(PairCol::Main(base + MEM_READ_VALUE_OFF + k));
            }
        }

        // output_range_check.byte_flags (32) + 2 comparison bytes.
        for k in 0..NB_LIMBS {
            cols.push(PairCol::Main(COL_OUTPUT_RANGE_CHECK_BASE + k));
        }
        cols.push(PairCol::Main(COL_OUTPUT_RANGE_CHECK_BASE + NB_LIMBS));
        cols.push(PairCol::Main(COL_OUTPUT_RANGE_CHECK_BASE + NB_LIMBS + 1));

        // x_memory timestamps (5 per access).
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_X_MEM_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_12_OFF));
        }

        // y_memory timestamps.
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_Y_MEM_BASE + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_READ_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_READ_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_12_OFF));
        }

        // modulus_memory timestamps.
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_MOD_MEM_BASE + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_READ_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_READ_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_12_OFF));
        }

        debug_assert_eq!(cols.len(), RES_LEN);
        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators + polynomial optimizations
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Uint256MulCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let x_ptr = local.x_ptr.clone();
        let y_ptr = local.y_ptr.clone();

        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));

        // =================================================================
        // #1..#95: output FieldOpCols range checks
        // =================================================================
        let output_result_limbs: Vec<AB::VarMaybeExt> =
            local.output.result.0.iter().cloned().collect();
        let output_carry_limbs: Vec<AB::VarMaybeExt> =
            local.output.carry.0.iter().cloned().collect();
        let output_witness_limbs: Vec<AB::VarMaybeExt> =
            local.output.witness.0.iter().cloned().collect();
        field_op_precompute_lc::<AB, U256Field>(
            builder,
            &output_result_limbs,
            &output_carry_limbs,
            &output_witness_limbs,
        );

        // =================================================================
        // #96..#98: output_range_check FieldLtCols (LTU + BitVec for byte_flags)
        // =================================================================
        {
            let flags: Vec<AB::VarMaybeExt> =
                local.output_range_check.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, U256Field>(
                builder,
                local.output_range_check.lhs_comparison_byte.clone(),
                local.output_range_check.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // =================================================================
        // #99..#130: x_memory readwrite (8 words × 4 interactions each)
        // x is written with the result; read at clk+1
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..WORDS_FIELD_ELEMENT {
            let addr = x_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.x_memory[i].access,
                &local.x_memory[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // #131..#162: y_memory read (8 words × 4 interactions each)
        // y is read at clk from y_ptr
        // =================================================================
        for i in 0..WORDS_FIELD_ELEMENT {
            let addr = y_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.y_memory[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #163..#194: modulus_memory read (8 words × 4 interactions each)
        // modulus is read at clk from y_ptr + WORDS_FIELD_ELEMENT * 4
        // =================================================================
        for i in 0..WORDS_FIELD_ELEMENT {
            let addr = y_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    ((WORDS_FIELD_ELEMENT + i) * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.modulus_memory[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #195: recv(Syscall)
        // =================================================================
        let syscall_id =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(SyscallCode::UINT256_MUL.syscall_id()));
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![shard.clone(), clk.clone(), syscall_id, x_ptr, y_ptr],
        ));

        // Keep non-lookup precomputations after all lookup denominators so permutation generation
        // still sees the first NUM_LOOKUPS entries as invertible lookup values.
        field_op_precompute_witness_beta::<AB, U256Field>(builder, &output_witness_limbs);

        // =================================================================
        // β-evaluations of gate operands moved out of `eval`.
        // =================================================================
        // x_beta: 32 limbs from x_memory[].prev_value.
        let x_limbs: Vec<AB::VarMaybeExt> = local.x_memory[..WORDS_FIELD_ELEMENT]
            .iter()
            .flat_map(|m| m.prev_value.0.iter().cloned())
            .collect();
        let x_beta = field_op_beta_from_coeffs(builder, &x_limbs);
        builder.retain_precomputed(x_beta);

        // y_beta: 32 limbs from y_memory[].access.value.
        let y_limbs: Vec<AB::VarMaybeExt> = local.y_memory[..WORDS_FIELD_ELEMENT]
            .iter()
            .flat_map(|m| m.access.value.0.iter().cloned())
            .collect();
        let y_beta = field_op_beta_from_coeffs(builder, &y_limbs);
        builder.retain_precomputed(y_beta);

        // result_beta: β eval of output.result.
        let result_beta = field_op_beta_from_coeffs(builder, &output_result_limbs);
        builder.retain_precomputed(result_beta);

        // carry_beta: β eval of output.carry.
        let carry_beta = field_op_beta_from_coeffs(builder, &output_carry_limbs);
        builder.retain_precomputed(carry_beta);

        // =================================================================
        // Polynomial optimization for assert_all_eq:
        // compute diff(β) = Σ (output.result[i] - x_value[i]) * β^i
        // =================================================================
        let x_value_limbs: Vec<AB::VarMaybeExt> = local.x_memory[..WORDS_FIELD_ELEMENT]
            .iter()
            .flat_map(|acc| acc.access.value.0.iter().cloned())
            .collect();

        let diff_coeffs: Vec<AB::VarMaybeExt> = local
            .output
            .result
            .0
            .iter()
            .zip(x_value_limbs.iter())
            .map(|(r, v)| r.clone() - v.clone())
            .collect();

        let diff_beta = {
            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            Polynomial::from_coefficients(&diff_coeffs).eval_with_powers(beta_powers, zero_ext)
        };
        builder.retain_precomputed(diff_beta);
    }

    // ========================================================================
    // Phase 2: eval — gate constraints (reserved_poly columns only)
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let beta_consts = FieldOpBetaConsts::<AB>::new::<U256Field>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let shard = local[RES_SHARD].clone();
        let clk = local[RES_CLK].clone();
        let modulus_is_zero_inverse = local[RES_MOD_IS_ZERO_INVERSE].clone();
        let modulus_is_zero_result = local[RES_MOD_IS_ZERO_RESULT].clone();
        let modulus_is_not_zero = local[RES_MOD_IS_NOT_ZERO].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);

        // Collect limbs that are needed limb-wise (kept in reserved_poly).
        let result_limbs: Vec<AB::VarMaybeExt> =
            (0..NB_LIMBS).map(|k| local[RES_OUTPUT_RESULT_BASE + k].clone()).collect();
        let modulus_limbs: Vec<AB::VarMaybeExt> =
            (0..NB_LIMBS).map(|k| local[RES_MOD_MEM_VALUE_BASE + k].clone()).collect();

        // -- air.rs L249-254: IsZeroOperation gate constraints --
        let modulus_byte_sum = modulus_limbs.iter().fold(zero, |acc, limb| acc + limb.clone());
        is_zero_op_gate_constraints(
            builder,
            modulus_byte_sum,
            modulus_is_zero_inverse,
            modulus_is_zero_result.clone(),
            is_real.clone(),
        );

        // -- air.rs L257-268: Dynamic modulus construction --
        // Pull pre-evaluated β-values from precomputed.
        let (output_witness_beta, x_beta, y_beta, result_beta, carry_beta) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            (
                pc[OUTPUT_WITNESS_BETA_IDX].clone(),
                pc[X_BETA_IDX].clone(),
                pc[Y_BETA_IDX].clone(),
                pc[RESULT_BETA_IDX].clone(),
                pc[CARRY_BETA_IDX].clone(),
            )
        };

        // effective_modulus = modulus_limbs * (1 - modulus_is_zero) ++ [modulus_is_zero] (33
        // coeffs)
        {
            let mut effective_modulus_coeffs: Vec<AB::VarMaybeExt> = modulus_limbs
                .iter()
                .map(|limb| limb.clone() * (one.clone() - modulus_is_zero_result.clone()))
                .collect();
            effective_modulus_coeffs.push(modulus_is_zero_result.clone());
            let modulus_beta = field_op_beta_from_coeffs(builder, &effective_modulus_coeffs);

            // vanishing = x * y - result - carry * modulus
            let vanishing_beta = x_beta * y_beta - result_beta - carry_beta * modulus_beta;
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                output_witness_beta,
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // -- air.rs L279-286: output_range_check gate constraints --
        {
            let byte_flags: Limbs<AB::VarMaybeExt, <U256Field as NumLimbs>::Limbs> =
                (0..NB_LIMBS).map(|k| local[RES_OR_BYTE_FLAGS_BASE + k].clone()).collect();
            let output_range_check: FieldLtCols<AB::VarMaybeExt, U256Field> = FieldLtCols {
                byte_flags,
                lhs_comparison_byte: local[RES_OR_LHS_BYTE].clone(),
                rhs_comparison_byte: local[RES_OR_RHS_BYTE].clone(),
            };
            field_lt_gate_constraints::<AB, U256Field>(
                builder,
                &result_limbs,
                &modulus_limbs,
                &output_range_check,
                modulus_is_not_zero.clone(),
            );
        }

        // -- air.rs L287-290: modulus_is_not_zero derivation --
        builder.assert_zero(
            modulus_is_not_zero - is_real.clone() * (one.clone() - modulus_is_zero_result),
        );

        // -- air.rs L293: assert_all_eq optimization via precomputed polynomial --
        {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let diff_beta = pc[DIFF_BETA_IDX].clone();
            builder.when(is_real.clone()).assert_zero_ext(diff_beta);
        }

        // Build a MemoryAccessCols from 5 timestamp slots in reserved_poly.
        let acc_from = |base: usize| MemoryAccessCols::<AB::VarMaybeExt> {
            value: zero_word.clone(),
            prev_shard: local[base].clone(),
            prev_clk: local[base + 1].clone(),
            compare_clk: local[base + 2].clone(),
            diff_16bit_limb: local[base + 3].clone(),
            diff_12bit_limb: local[base + 4].clone(),
        };

        // -- air.rs L296-310: memory timestamp gate constraints --
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        // x_memory: written at clk+1
        for i in 0..WORDS_FIELD_ELEMENT {
            let acc = acc_from(RES_X_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                write_clk.clone(),
                is_real.clone(),
            );
        }
        // y_memory: read at clk
        for i in 0..WORDS_FIELD_ELEMENT {
            let acc = acc_from(RES_Y_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }
        // modulus_memory: read at clk
        for i in 0..WORDS_FIELD_ELEMENT {
            let acc = acc_from(RES_MOD_MEM_TS_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }

        // -- air.rs L330: Boolean constraint for is_real --
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let modulus_is_not_zero = local[RES_MOD_IS_NOT_ZERO].clone();

        // #1..#95: output FieldOpCols range checks
        field_op_lookup::<AB, U256Field>(builder, is_real.clone());

        // #96..#98: output_range_check LTU + BitVec
        field_lt_lookup::<AB, U256Field>(builder, modulus_is_not_zero);

        // #99..#130: x_memory readwrites
        for _ in 0..WORDS_FIELD_ELEMENT {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #131..#162: y_memory reads
        for _ in 0..WORDS_FIELD_ELEMENT {
            memory_read_lookup(builder, is_real.clone());
        }

        // #163..#194: modulus_memory reads
        for _ in 0..WORDS_FIELD_ELEMENT {
            memory_read_lookup(builder, is_real.clone());
        }

        // #195: recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use crate::syscall::precompiles::uint256::Uint256MulChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field> BaseAir<F> for Uint256MulPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for Uint256MulPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <Uint256MulChip as MachineAir<F>>::name(&Uint256MulChip) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Uint256MulChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use crate::utils::{words_to_bytes_le, zeroed_f_vec};
        use dt_core_executor::events::{ByteLookupEvent, PrecompileEvent};
        use num::{BigUint, Zero};
        use std::{borrow::BorrowMut, mem::size_of};

        <Uint256MulChip as MachineAir<F>>::generate_dependencies(&Uint256MulChip, input, output);

        let events = input.get_precompile_events(SyscallCode::UINT256_MUL);
        for (_, event) in events {
            let PrecompileEvent::Uint256Mul(event) = event else { unreachable!() };
            let modulus = BigUint::from_bytes_le(&words_to_bytes_le::<32>(&event.modulus));
            if modulus.is_zero() {
                continue;
            }
            let x = BigUint::from_bytes_le(&words_to_bytes_le::<32>(&event.x));
            let y = BigUint::from_bytes_le(&words_to_bytes_le::<32>(&event.y));
            let result = (&x * &y) % &modulus;
            let lt_size = size_of::<FieldLtCols<u8, U256Field>>();
            let mut lt_row = zeroed_f_vec(lt_size);
            let lt_cols: &mut FieldLtCols<F, U256Field> = lt_row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            lt_cols.populate(&mut ignored_blu, &result, &modulus);
            add_field_lt_bitvec_lookups::<F, U256Field>(output, lt_cols);
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <Uint256MulChip as MachineAir<F>>::included(&Uint256MulChip, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        Uint256MulChip.padding_row()
    }

    fn local_only(&self) -> bool {
        <Uint256MulChip as MachineAir<F>>::local_only(&Uint256MulChip)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 3;

    use crate::syscall::precompiles::uint256::Uint256MulChip;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::params::{FieldParameters, NumLimbs};
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
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::ops::Deref;
    use test_artifacts::UINT256_MUL_ELF;
    use typenum::Unsigned;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    /// BabyBear modulus: 15 * 2^27 + 1
    const BABYBEAR_MODULUS: u32 = 2013265921;

    fn random_f(rng: &mut StdRng) -> F {
        let value = rng.gen_range(0..BABYBEAR_MODULUS);
        F::from_canonical_u32(value)
    }

    fn random_ef(rng: &mut StdRng) -> EF {
        let values: [F; 4] = [random_f(rng), random_f(rng), random_f(rng), random_f(rng)];
        EF::from_base_slice(&values)
    }

    fn challenge_beta_with_seed(seed: u64) -> EF {
        let mut rng = StdRng::seed_from_u64(seed);
        random_ef(&mut rng)
    }

    fn beta_powers(air: &Uint256MulPolyAir, beta: EF) -> Vec<EF> {
        let max = <Uint256MulPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(air);
        (0..=max).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(
        air: &Uint256MulPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <Uint256MulPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let empty_prep: Vec<F> = vec![];
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            let main_row: &[F] = Deref::deref(&main_binding);
            let reserved = collect_reserved_poly(main_row, &empty_prep, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    /// Build a real trace from the UINT256_MUL test ELF.
    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = Program::from(UINT256_MUL_ELF).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::UINT256_MUL).is_empty() {
                continue;
            }

            let mut uint_shard = ExecutionRecord::new(shard.program.clone());
            uint_shard.precompile_events = shard.precompile_events.clone();

            let chip = Uint256MulChip::new();
            return Some(
                chip.generate_trace(&uint_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_uint256_mul_polyair_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No Uint256Mul trace found -- skipping test");
                return;
            }
        };

        let air = Uint256MulPolyAir::new();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let bp = beta_powers(&air, beta);
        let bs = beta_septix(beta);
        let public: Vec<F> = vec![];

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
            NUM_PRECOMPUTED,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &bp,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        // Conservative upper bound for gate constraints.
        let nb_limbs = <U256Field as FieldParameters>::NB_LIMBS; // 32
        let nb_witness = <U256Field as NumLimbs>::Witness::USIZE; // 63
        let num_gate_constraints =
            (nb_witness + 1)                  // FieldOpCols vanishing polynomial constraints
            + 1                                // IsZeroOp: result = 1 - inverse * a
            + 1                                // IsZeroOp: result * a = 0
            + (nb_limbs + 3)                   // FieldLtCols gate constraints
            + 1                                // modulus_is_not_zero derivation
            + 1                                // assert_zero_ext for diff_beta
            + WORDS_FIELD_ELEMENT * 3 * 3      // memory timestamp (x, y, mod)
            + 2                                // is_real bool + modulus_is_zero bool
            ;
        let num_reducer = num_gate_constraints + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(reducer_seed);
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();

        let first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &precomputed,
            &permutation,
            alpha,
            &bp,
            bs,
            global,
            F::one(),
            F::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(
            first.iter().all(|x| x.is_zero()),
            "first_round non-zero at indices: {:?}",
            first
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(10)
                .collect::<Vec<_>>()
        );

        let nonfirst = nonfirst_round_evaluation(
            &air,
            &public,
            &reserved,
            &precomputed,
            &permutation,
            alpha,
            &bp,
            bs,
            global,
            EF::one(),
            EF::one(),
            local_sum,
            BATCH_SIZE,
            &constraint_reducer,
        );
        assert!(
            nonfirst.iter().all(|x| x.is_zero()),
            "nonfirst_round non-zero at indices: {:?}",
            nonfirst
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(10)
                .collect::<Vec<_>>()
        );
    }

    /// Generate a random Uint256Mul trace for performance testing.
    fn random_uint256_mul_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace().expect("sample trace should exist");
        let base_height = base.height();
        let width = base.width();

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

        let last_row_start = (base_height - 1) * width;
        let last_row = &base.values[last_row_start..last_row_start + width];
        let mut values = Vec::with_capacity(target_height * width);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }

        RowMajorMatrix::new(values, width)
    }

    /// Multi-round sumcheck benchmark for Uint256MulPolyAir.
    ///
    /// Runs a complete `log_n`-round sumcheck:
    ///   Round 0: first_round_evaluation (base-field trace)
    ///   Rounds 1..log_n-1: bound_var_* folding + nonfirst_round_evaluation
    ///
    /// This measures the **total** sumcheck proving time, where PolyAir's
    /// precompute optimization should show cumulative benefits.
    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = Uint256MulPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::UINT256_MUL_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_uint256_mul_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let bp = beta_powers(&air, beta);
        let bs = beta_septix(beta);
        let public: Vec<F> = vec![];

        // Conservative upper bound for constraint reducer (matches constraint check test).
        let nb_limbs = <U256Field as FieldParameters>::NB_LIMBS;
        let nb_witness = <U256Field as NumLimbs>::Witness::USIZE;
        let num_gate_constraints = (nb_witness + 1)                                // FieldOpCols vanishing polynomial constraints
            + 1                                             // IsZeroOp: result = 1 - inverse * a
            + 1                                             // IsZeroOp: result * a = 0
            + (nb_limbs + 3)                                // FieldLtCols gate constraints
            + 1                                             // modulus_is_not_zero derivation
            + 1                                             // assert_zero_ext for diff_beta
            + WORDS_FIELD_ELEMENT * 3 * 3                   // memory timestamp (x, y, mod)
            + 2; // is_real bool + modulus_is_zero bool
        let num_reducer = num_gate_constraints + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc =
            <Uint256MulPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

        // --- Precompute phase ---
        let t_precompute = std::time::Instant::now();
        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
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
            &bp,
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
            &bp,
            bs,
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
                &bp,
                bs,
                global,
                selector_first,
                selector_last,
                local_sum,
                BATCH_SIZE,
                &constraint_reducer,
            );

            let round_elapsed = t_round.elapsed();
            std::println!("  round {} (nonfirst): {:?}", round, round_elapsed);

            // Fold for next round (skip on last round)
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
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl Uint256MulPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
