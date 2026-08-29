//! PolyAir adaptation of U256x2048MulChip.
//!
//! Bridges `U256x2048MulCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (1345 total)
//!
//!   #1   ..#760:  8 × FieldOpCols range checks (each: 16 U8Range result + 16 U8Range carry + 63
//! U16Range witness)   #761 ..#764:  lo_ptr_memory read (1 word × 4 interactions)
//!   #765 ..#768:  hi_ptr_memory read (1 word × 4 interactions)
//!   #769 ..#800:  a_memory read (8 words × 4 interactions)
//!   #801 ..#1056: b_memory read (64 words × 4 interactions)
//!   #1057..#1312: lo_memory write (64 words × 4 interactions)
//!   #1313..#1344: hi_memory write (8 words × 4 interactions)
//!   #1345:        recv(Syscall)
//!
//!   Plus 42 precomputed polynomial optimizations → NUM_LOOKUPS + 42 precomputed total.
//!     [+0..+8]   witness_beta[0..8]              (one per output FieldOpCols)
//!     [+8]       a_beta                          (β eval of a_memory limbs)
//!     [+9..+17]  b_beta[0..8]                    (β eval of each 32-limb b chunk)
//!     [+17..+25] result_beta[0..8]               (β eval of each output.result)
//!     [+25..+33] carry_beta[0..8]                (β eval of each output.carry)
//!     [+33..+41] result_diff_beta[0..8]          (assert_all_eq(result, lo_chunk))
//!     [+41]      hi_diff_beta                    (assert_all_eq(carry[7], hi_value))
//!
//! ## Boolean handling (1 boolean → direct gate constraint)
//!   - is_real → gate assert_zero(x*(1-x))
//!
//! ## Fixed modulus
//!   The modulus is always 2^256 (33 coefficients: [0]*32 + [1]).
//!   Gate constraints build vanishing polynomials for each eval_mul_and_carry call.
//!
//! ## reserved_poly minimization
//!   Only the columns actually consumed by `eval` and `lookup` are reserved.
//!   Skipped (consumed only inside `precompute_lc` as lookup denominators or β-evals):
//!     - a_ptr, b_ptr                           (address inputs to memory LCs only)
//!     - a_memory[i].access.value               (collapsed into a_beta)
//!     - b_memory[i].access.value               (collapsed into b_beta[chunk])
//!     - lo_memory[i].prev_value, .access.value (lookup LC + result_diff_beta only)
//!     - hi_memory[i].prev_value, .access.value (lookup LC + hi_diff_beta only)
//!     - outputs[i].result                      (collapsed into result_beta[i])
//!     - outputs[i].carry                       (collapsed into carry_beta[i])
//!     - outputs[i].witness                     (collapsed into witness_beta[i])
//!   Kept: is_real, shard, clk, lo_ptr, hi_ptr,
//!         lo_ptr_memory (full: value used for reduce() + timestamps),
//!         hi_ptr_memory (full),
//!         {a, b, lo, hi}_memory[i] timestamp fields only.

use std::ops::Deref;

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::uint256::U256Field;
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol, Polynomial},
    InteractionKind, Word,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    memory::{
        polyair::{
            memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
            memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
        },
        MemoryAccessCols,
    },
    operations::field::field_op::{
        field_op_beta_from_coeffs, field_op_gate_constraints, field_op_lookup,
        field_op_num_interactions, field_op_precompute_lc, field_op_precompute_witness_beta,
        FieldOpBetaConsts,
    },
};

use super::air::{U256x2048MulCols, HI_REGISTER, LO_REGISTER, NUM_COLS, WORDS_FIELD_ELEMENT};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
const NUM_LOOKUPS: usize = 8 * field_op_num_interactions::<U256Field>() // 8 × 95 = 760
    + 4                                                                 // lo_ptr_memory read
    + 4                                                                 // hi_ptr_memory read
    + WORDS_FIELD_ELEMENT * 4                                           // a_memory read (8×4=32)
    + WORDS_FIELD_ELEMENT * 8 * 4                                       // b_memory read (64×4=256)
    + WORDS_FIELD_ELEMENT * 8 * 4                                       // lo_memory write (64×4=256)
    + WORDS_FIELD_ELEMENT * 4                                           // hi_memory write (8×4=32)
    + 1; // recv(Syscall)

/// Precomputed linear combinations:
/// - one per lookup
/// - 8 witness(beta) values for the 8 FieldOp rows
/// - 1 a_beta + 8 b_beta + 8 result_beta + 8 carry_beta (operands of the 8 gates)
/// - 9 diff(beta) values for the assert_all_eq optimizations
const NUM_POLY_OPTS: usize = 8 + 1 + 8 + 8 + 8 + 9;
const NUM_PRECOMPUTED: usize = NUM_LOOKUPS + NUM_POLY_OPTS;

/// Indices (relative to NUM_LOOKUPS) inside the precomputed slice.
const PC_WITNESS_BETA_BASE: usize = 0; // 8 entries
const PC_A_BETA: usize = 8;
const PC_B_BETA_BASE: usize = 9; // 8 entries
const PC_RESULT_BETA_BASE: usize = 17; // 8 entries
const PC_CARRY_BETA_BASE: usize = 25; // 8 entries
const PC_RESULT_DIFF_BETA_BASE: usize = 33; // 8 entries
const PC_HI_DIFF_BETA: usize = 41;

/// Maximum number of values in a single lookup payload.
/// Memory payloads are 7 (shard, clk, addr, value[0..3]), byte payloads are 5.
/// The assert_all_eq polynomial evaluations use NB_LIMBS=32 powers of beta.
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets (byte index within U256x2048MulCols<u8>)
//
// Layout (#[repr(C)], from air.rs):
//   [0]  shard
//   [1]  clk
//   [2]  a_ptr
//   [3]  b_ptr
//   [4]  lo_ptr
//   [5]  hi_ptr
//   [6 ..]  lo_ptr_memory          (MemoryReadCols, 9 bytes)
//   [..]    hi_ptr_memory          (MemoryReadCols, 9 bytes)
//   [..]    a_memory[8]            (8 × MemoryReadCols  = 72 bytes)
//   [..]    b_memory[64]           (64 × MemoryReadCols = 576 bytes)
//   [..]    lo_memory[64]          (64 × MemoryWriteCols, 13 bytes each = 832 bytes)
//   [..]    hi_memory[8]           (8 × MemoryWriteCols = 104 bytes)
//   [..]    outputs[8]             (8 × FieldOpCols, 127 bytes each = 1016 bytes)
//   [last]  is_real
// ============================================================================

const COL_SHARD: usize = 0;
const COL_CLK: usize = 1;
const COL_LO_PTR: usize = 4;
const COL_HI_PTR: usize = 5;
const COL_LO_PTR_MEM_BASE: usize = 6;
const COL_HI_PTR_MEM_BASE: usize = COL_LO_PTR_MEM_BASE + MEM_READ_COLS_SIZE;
const COL_A_MEM_BASE: usize = COL_HI_PTR_MEM_BASE + MEM_READ_COLS_SIZE;
const COL_B_MEM_BASE: usize = COL_A_MEM_BASE + WORDS_FIELD_ELEMENT * MEM_READ_COLS_SIZE;
const COL_LO_MEM_BASE: usize = COL_B_MEM_BASE + WORDS_FIELD_ELEMENT * 8 * MEM_READ_COLS_SIZE;
const COL_HI_MEM_BASE: usize = COL_LO_MEM_BASE + WORDS_FIELD_ELEMENT * 8 * MEM_WRITE_COLS_SIZE;
const COL_IS_REAL: usize = NUM_COLS - 1;

const MEM_READ_COLS_SIZE: usize = 9;
const MEM_WRITE_COLS_SIZE: usize = 13;
// Offsets within MemoryReadCols { access: MemoryAccessCols }:
const MEM_READ_VALUE_OFF: usize = 0; // 4 bytes
const MEM_READ_PREV_SHARD_OFF: usize = 4;
const MEM_READ_PREV_CLK_OFF: usize = 5;
const MEM_READ_COMPARE_CLK_OFF: usize = 6;
const MEM_READ_DIFF_16_OFF: usize = 7;
const MEM_READ_DIFF_12_OFF: usize = 8;
// Offsets within MemoryWriteCols { prev_value, access }:
//   prev_value occupies bytes 0..4, access.value occupies bytes 4..8 — both
//   stripped from reserved_poly; only timestamp fields below are referenced.
const MEM_WRITE_PREV_SHARD_OFF: usize = 8;
const MEM_WRITE_PREV_CLK_OFF: usize = 9;
const MEM_WRITE_COMPARE_CLK_OFF: usize = 10;
const MEM_WRITE_DIFF_16_OFF: usize = 11;
const MEM_WRITE_DIFF_12_OFF: usize = 12;

// ============================================================================
// reserved_poly slice layout (positions in the reserved row).
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  lo_ptr
//   [4]  hi_ptr
//   [5..9]   lo_ptr_memory.access.value (4)
//   [9..14]  lo_ptr_memory timestamps   (5: prev_shard, prev_clk, compare_clk, diff_16, diff_12)
//   [14..18] hi_ptr_memory.access.value (4)
//   [18..23] hi_ptr_memory timestamps   (5)
//   [23 + i*5 ..]  a_memory[i] timestamps only (5 each, 8 entries)
//   [..]           b_memory[i] timestamps (5 × 64)
//   [..]           lo_memory[i] timestamps (5 × 64)
//   [..]           hi_memory[i] timestamps (5 × 8)
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_SHARD: usize = 1;
const RES_CLK: usize = 2;
const RES_LO_PTR: usize = 3;
const RES_HI_PTR: usize = 4;
const RES_LO_PTR_MEM_VALUE_BASE: usize = 5;
const RES_LO_PTR_MEM_TS_BASE: usize = 9;
const RES_HI_PTR_MEM_VALUE_BASE: usize = 14;
const RES_HI_PTR_MEM_TS_BASE: usize = 18;
const RES_PER_MEM_TS: usize = 5;
const RES_A_MEM_BASE: usize = 23;
const RES_B_MEM_BASE: usize = RES_A_MEM_BASE + WORDS_FIELD_ELEMENT * RES_PER_MEM_TS;
const RES_LO_MEM_BASE: usize = RES_B_MEM_BASE + WORDS_FIELD_ELEMENT * 8 * RES_PER_MEM_TS;
const RES_HI_MEM_BASE: usize = RES_LO_MEM_BASE + WORDS_FIELD_ELEMENT * 8 * RES_PER_MEM_TS;
const RES_LEN: usize = RES_HI_MEM_BASE + WORDS_FIELD_ELEMENT * RES_PER_MEM_TS;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for U256x2048MulChip.
#[derive(Clone, Copy, Default)]
pub struct U256x2048MulPolyAir;

impl U256x2048MulPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for U256x2048MulPolyAir
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

        // Scalars: is_real, shard, clk, lo_ptr, hi_ptr.
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_LO_PTR));
        cols.push(PairCol::Main(COL_HI_PTR));

        // lo_ptr_memory: keep full (value used in reduce() + timestamps).
        for k in 0..4 {
            cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_VALUE_OFF + k));
        }
        cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_PREV_SHARD_OFF));
        cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_PREV_CLK_OFF));
        cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_COMPARE_CLK_OFF));
        cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_DIFF_16_OFF));
        cols.push(PairCol::Main(COL_LO_PTR_MEM_BASE + MEM_READ_DIFF_12_OFF));

        // hi_ptr_memory: keep full.
        for k in 0..4 {
            cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_VALUE_OFF + k));
        }
        cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_PREV_SHARD_OFF));
        cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_PREV_CLK_OFF));
        cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_COMPARE_CLK_OFF));
        cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_DIFF_16_OFF));
        cols.push(PairCol::Main(COL_HI_PTR_MEM_BASE + MEM_READ_DIFF_12_OFF));

        // a_memory: timestamps only (value collapsed into a_beta).
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_A_MEM_BASE + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_READ_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_READ_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_12_OFF));
        }

        // b_memory: timestamps only (value collapsed into b_beta[chunk]).
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let base = COL_B_MEM_BASE + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_READ_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_READ_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_READ_DIFF_12_OFF));
        }

        // lo_memory: timestamps only (prev_value/value consumed in precompute_lc only).
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let base = COL_LO_MEM_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_12_OFF));
        }

        // hi_memory: timestamps only.
        for i in 0..WORDS_FIELD_ELEMENT {
            let base = COL_HI_MEM_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_WRITE_DIFF_12_OFF));
        }

        debug_assert_eq!(cols.len(), RES_LEN);
        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &U256x2048MulCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let a_ptr = local.a_ptr.clone();
        let b_ptr = local.b_ptr.clone();
        let lo_ptr = local.lo_ptr.clone();
        let hi_ptr = local.hi_ptr.clone();

        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));

        let outputs = [
            &local.a_mul_b1,
            &local.ab2_plus_carry,
            &local.ab3_plus_carry,
            &local.ab4_plus_carry,
            &local.ab5_plus_carry,
            &local.ab6_plus_carry,
            &local.ab7_plus_carry,
            &local.ab8_plus_carry,
        ];

        // =================================================================
        // #1..#760: 8 × FieldOpCols range checks (95 each)
        // =================================================================
        for output in &outputs {
            field_op_precompute_lc::<AB, U256Field>(
                builder,
                &output.result.0.iter().cloned().collect::<Vec<_>>(),
                &output.carry.0.iter().cloned().collect::<Vec<_>>(),
                &output.witness.0.iter().cloned().collect::<Vec<_>>(),
            );
        }

        // =================================================================
        // #761..#764: lo_ptr_memory read (1 word × 4 interactions)
        // Read at clk from LO_REGISTER
        // =================================================================
        {
            let addr = AB::VarMaybeExt::from(AB::F::from_canonical_u32(LO_REGISTER));
            memory_read_precompute_lc(
                builder,
                &local.lo_ptr_memory.access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #765..#768: hi_ptr_memory read (1 word × 4 interactions)
        // Read at clk from HI_REGISTER
        // =================================================================
        {
            let addr = AB::VarMaybeExt::from(AB::F::from_canonical_u32(HI_REGISTER));
            memory_read_precompute_lc(
                builder,
                &local.hi_ptr_memory.access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #769..#800: a_memory read (8 words × 4 interactions)
        // Read at clk from a_ptr + offset
        // =================================================================
        for i in 0..WORDS_FIELD_ELEMENT {
            let addr = a_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.a_memory[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #801..#1056: b_memory read (64 words × 4 interactions)
        // Read at clk from b_ptr + offset
        // =================================================================
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let addr = b_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.b_memory[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #1057..#1312: lo_memory write (64 words × 4 interactions)
        // Written at clk+1 from lo_ptr + offset
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let addr = lo_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.lo_memory[i].access,
                &local.lo_memory[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // #1313..#1344: hi_memory write (8 words × 4 interactions)
        // Written at clk+1 from hi_ptr + offset
        // =================================================================
        for i in 0..WORDS_FIELD_ELEMENT {
            let addr = hi_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.hi_memory[i].access,
                &local.hi_memory[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // #1345: recv(Syscall)
        // =================================================================
        let syscall_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
            SyscallCode::U256XU2048_MUL.syscall_id(),
        ));
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![shard.clone(), clk.clone(), syscall_id, a_ptr, b_ptr],
        ));

        for output in &outputs {
            field_op_precompute_witness_beta::<AB, U256Field>(
                builder,
                &output.witness.0.iter().cloned().collect::<Vec<_>>(),
            );
        }

        // =================================================================
        // β-evaluations of gate operands (a, b chunks, result, carry).
        // Moving these out of `eval` removes the corresponding limb columns
        // from `reserved_poly`.
        // =================================================================

        // a_beta: 32 limbs from a_memory[0..8].access.value.
        let a_limbs: Vec<AB::VarMaybeExt> = local.a_memory[..WORDS_FIELD_ELEMENT]
            .iter()
            .flat_map(|m| m.access.value.0.iter().cloned())
            .collect();
        let a_beta = field_op_beta_from_coeffs(builder, &a_limbs);
        builder.retain_precomputed(a_beta);

        // b_beta[0..8]: each 32-limb chunk of b_memory.
        for chunk_idx in 0..8 {
            let b_chunk: Vec<AB::VarMaybeExt> = local.b_memory
                [chunk_idx * WORDS_FIELD_ELEMENT..(chunk_idx + 1) * WORDS_FIELD_ELEMENT]
                .iter()
                .flat_map(|m| m.access.value.0.iter().cloned())
                .collect();
            let b_beta = field_op_beta_from_coeffs(builder, &b_chunk);
            builder.retain_precomputed(b_beta);
        }

        // result_beta[0..8]: each output.result.
        for output in &outputs {
            let result_beta = field_op_beta_from_coeffs(
                builder,
                &output.result.0.iter().cloned().collect::<Vec<_>>(),
            );
            builder.retain_precomputed(result_beta);
        }

        // carry_beta[0..8]: each output.carry.
        for output in &outputs {
            let carry_beta = field_op_beta_from_coeffs(
                builder,
                &output.carry.0.iter().cloned().collect::<Vec<_>>(),
            );
            builder.retain_precomputed(carry_beta);
        }

        // =================================================================
        // Polynomial optimizations for assert_all_eq (9 total)
        // Each optimization is computed in a local block to avoid borrow conflicts.
        // =================================================================

        // 8 × assert_all_eq(outputs[i].result, lo_memory_chunk[i])
        for i in 0..8 {
            let lo_chunk_limbs: Vec<AB::VarMaybeExt> = local.lo_memory
                [i * WORDS_FIELD_ELEMENT..(i + 1) * WORDS_FIELD_ELEMENT]
                .iter()
                .flat_map(|m| m.access.value.0.iter().cloned())
                .collect();

            let diff_coeffs: Vec<AB::VarMaybeExt> = outputs[i]
                .result
                .0
                .iter()
                .zip(lo_chunk_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();

            let diff_beta = {
                let beta_powers = builder.beta_powers();
                let zero_ext = AB::from_ef(AB::EF::zero());
                Polynomial::from_coefficients(&diff_coeffs).eval_with_powers(beta_powers, zero_ext)
            };
            builder.retain_precomputed(diff_beta);
        }

        // 1 × assert_all_eq(outputs[7].carry, value_as_limbs(hi_memory))
        {
            let hi_value_limbs: Vec<AB::VarMaybeExt> = local.hi_memory[..WORDS_FIELD_ELEMENT]
                .iter()
                .flat_map(|m| m.access.value.0.iter().cloned())
                .collect();

            let diff_coeffs: Vec<AB::VarMaybeExt> = outputs[7]
                .carry
                .0
                .iter()
                .zip(hi_value_limbs.iter())
                .map(|(c, v)| c.clone() - v.clone())
                .collect();

            let diff_beta = {
                let beta_powers = builder.beta_powers();
                let zero_ext = AB::from_ef(AB::EF::zero());
                Polynomial::from_coefficients(&diff_coeffs).eval_with_powers(beta_powers, zero_ext)
            };
            builder.retain_precomputed(diff_beta);
        }
    }

    // ========================================================================
    // Phase 2: eval — gate constraints
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let beta_consts = FieldOpBetaConsts::<AB>::new::<U256Field>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let shard = local[RES_SHARD].clone();
        let clk = local[RES_CLK].clone();
        let lo_ptr = local[RES_LO_PTR].clone();
        let hi_ptr = local[RES_HI_PTR].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero]);

        // -- air.rs L266: is_real boolean --
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // Pull β-evaluations of all gate operands from precomputed.
        let (witness_betas, a_beta, b_betas, result_betas, carry_betas) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let witness_betas: Vec<AB::VarExt> =
                (0..8).map(|i| pc[NUM_LOOKUPS + PC_WITNESS_BETA_BASE + i].clone()).collect();
            let a_beta = pc[NUM_LOOKUPS + PC_A_BETA].clone();
            let b_betas: Vec<AB::VarExt> =
                (0..8).map(|i| pc[NUM_LOOKUPS + PC_B_BETA_BASE + i].clone()).collect();
            let result_betas: Vec<AB::VarExt> =
                (0..8).map(|i| pc[NUM_LOOKUPS + PC_RESULT_BETA_BASE + i].clone()).collect();
            let carry_betas: Vec<AB::VarExt> =
                (0..8).map(|i| pc[NUM_LOOKUPS + PC_CARRY_BETA_BASE + i].clone()).collect();
            (witness_betas, a_beta, b_betas, result_betas, carry_betas)
        };

        // Fixed modulus 2^256 = x^32 in base-2^8 limbs → at β it evaluates to β^32.
        let modulus_beta = builder.beta_powers()
            [<U256Field as dt_curves::params::FieldParameters>::NB_LIMBS]
            .clone();

        // -- air.rs L340-357: eval_mul_and_carry gate constraints --
        // outputs[0]: vanishing = a * b[0] - result - carry * modulus.
        {
            let vanishing_beta = a_beta.clone() * b_betas[0].clone() -
                result_betas[0].clone() -
                carry_betas[0].clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[0].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }
        // outputs[1..7]: vanishing = a * b[i] + carry[i-1] - result - carry * modulus.
        for i in 1..8 {
            let vanishing_beta = a_beta.clone() * b_betas[i].clone() + carry_betas[i - 1].clone() -
                result_betas[i].clone() -
                carry_betas[i].clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[i].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // -- air.rs L369-371: assert_all_eq polynomial optimizations --
        {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();

            // 8 × assert_all_eq(outputs[i].result, lo_memory_chunk[i])
            for i in 0..8 {
                let diff_beta = pc[NUM_LOOKUPS + PC_RESULT_DIFF_BETA_BASE + i].clone();
                builder.when(is_real.clone()).assert_zero_ext(diff_beta);
            }
            // assert_all_eq(outputs[7].carry, value_as_limbs(hi_memory))
            let hi_diff_beta = pc[NUM_LOOKUPS + PC_HI_DIFF_BETA].clone();
            builder.when(is_real.clone()).assert_zero_ext(hi_diff_beta);
        }

        // -- air.rs L384-388: lo_ptr == lo_ptr_memory.value().reduce() --
        {
            let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
            let v0 = local[RES_LO_PTR_MEM_VALUE_BASE].clone();
            let v1 = local[RES_LO_PTR_MEM_VALUE_BASE + 1].clone();
            let v2 = local[RES_LO_PTR_MEM_VALUE_BASE + 2].clone();
            let v3 = local[RES_LO_PTR_MEM_VALUE_BASE + 3].clone();
            let reduced = v0 +
                v1 * base.clone() +
                v2 * base.clone() * base.clone() +
                v3 * base.clone() * base.clone() * base;
            builder.when(is_real.clone()).assert_eq(lo_ptr, reduced);
        }

        // -- air.rs L391-394: hi_ptr == hi_ptr_memory.value().reduce() --
        {
            let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
            let v0 = local[RES_HI_PTR_MEM_VALUE_BASE].clone();
            let v1 = local[RES_HI_PTR_MEM_VALUE_BASE + 1].clone();
            let v2 = local[RES_HI_PTR_MEM_VALUE_BASE + 2].clone();
            let v3 = local[RES_HI_PTR_MEM_VALUE_BASE + 3].clone();
            let reduced = v0 +
                v1 * base.clone() +
                v2 * base.clone() * base.clone() +
                v3 * base.clone() * base.clone() * base;
            builder.when(is_real.clone()).assert_eq(hi_ptr, reduced);
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

        // -- Memory timestamp gate constraints --
        // lo_ptr_memory: read at clk
        memory_timestamp_gate_constraints(
            builder,
            &acc_from(RES_LO_PTR_MEM_TS_BASE),
            shard.clone(),
            clk.clone(),
            is_real.clone(),
        );
        // hi_ptr_memory: read at clk
        memory_timestamp_gate_constraints(
            builder,
            &acc_from(RES_HI_PTR_MEM_TS_BASE),
            shard.clone(),
            clk.clone(),
            is_real.clone(),
        );
        // a_memory: read at clk
        for i in 0..WORDS_FIELD_ELEMENT {
            let acc = acc_from(RES_A_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }
        // b_memory: read at clk
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let acc = acc_from(RES_B_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }
        // lo_memory / hi_memory: written at clk+1
        let write_clk = clk + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..(WORDS_FIELD_ELEMENT * 8) {
            let acc = acc_from(RES_LO_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                write_clk.clone(),
                is_real.clone(),
            );
        }
        for i in 0..WORDS_FIELD_ELEMENT {
            let acc = acc_from(RES_HI_MEM_BASE + i * RES_PER_MEM_TS);
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                write_clk.clone(),
                is_real.clone(),
            );
        }
    }

    // ========================================================================
    // Phase 3: lookup
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        // #1..#760: 8 × FieldOpCols range checks
        for _ in 0..8 {
            field_op_lookup::<AB, U256Field>(builder, is_real.clone());
        }

        // #761..#764: lo_ptr_memory read
        memory_read_lookup(builder, is_real.clone());

        // #765..#768: hi_ptr_memory read
        memory_read_lookup(builder, is_real.clone());

        // #769..#800: a_memory read (8 words)
        for _ in 0..WORDS_FIELD_ELEMENT {
            memory_read_lookup(builder, is_real.clone());
        }

        // #801..#1056: b_memory read (64 words)
        for _ in 0..(WORDS_FIELD_ELEMENT * 8) {
            memory_read_lookup(builder, is_real.clone());
        }

        // #1057..#1312: lo_memory write (64 words)
        for _ in 0..(WORDS_FIELD_ELEMENT * 8) {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #1313..#1344: hi_memory write (8 words)
        for _ in 0..WORDS_FIELD_ELEMENT {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #1345: recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use crate::syscall::precompiles::u256x2048_mul::U256x2048MulChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for U256x2048MulPolyAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<F: Field> MachineAir<F> for U256x2048MulPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <U256x2048MulChip as MachineAir<F>>::name(&U256x2048MulChip) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        U256x2048MulChip.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <U256x2048MulChip as MachineAir<F>>::included(&U256x2048MulChip, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        U256x2048MulChip.padding_row()
    }

    fn local_only(&self) -> bool {
        <U256x2048MulChip as MachineAir<F>>::local_only(&U256x2048MulChip)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 3;

    use crate::syscall::precompiles::u256x2048_mul::U256x2048MulChip;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::params::FieldParameters;
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
    use test_artifacts::U256XU2048_MUL_ELF;

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

    fn beta_powers(air: &U256x2048MulPolyAir, beta: EF) -> Vec<EF> {
        let max = <U256x2048MulPolyAir as FullAir<
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
        air: &U256x2048MulPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <U256x2048MulPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(
                air,
            );
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

    /// Build a real trace from the U256XU2048_MUL test ELF.
    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = Program::from(U256XU2048_MUL_ELF).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::U256XU2048_MUL).is_empty() {
                continue;
            }

            let mut sub_shard = ExecutionRecord::new(shard.program.clone());
            sub_shard.precompile_events = shard.precompile_events.clone();

            let chip = U256x2048MulChip::new();
            return Some(
                chip.generate_trace(&sub_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_u256x2048_mul_polyair_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No U256x2048Mul trace found -- skipping test");
                return;
            }
        };

        let air = U256x2048MulPolyAir::new();
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
        // vanishing polynomial constraints per FieldOpCols
        let field_op_gate_count = <U256Field as FieldParameters>::NB_WITNESS_LIMBS + 1;
        let num_gate_constraints =
            8 * field_op_gate_count                           // 8 × eval_mul_and_carry
            + 9                                                // 9 × assert_zero_ext for polynomial optimizations
            + 2                                                // lo_ptr, hi_ptr assert_eq
            + 1                                                // is_real boolean
            + (2 + WORDS_FIELD_ELEMENT + WORDS_FIELD_ELEMENT * 8
               + WORDS_FIELD_ELEMENT * 8 + WORDS_FIELD_ELEMENT) * 3  // memory timestamp constraints
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

    /// Generate a random U256x2048Mul trace for performance testing.
    fn random_u256x2048_mul_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

    /// Multi-round sumcheck benchmark for U256x2048MulPolyAir.
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
        let air = U256x2048MulPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::U256X2048_MUL_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_u256x2048_mul_trace(log_n, seed);
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
        let field_op_gate_count = <U256Field as FieldParameters>::NB_WITNESS_LIMBS + 1;
        let num_gate_constraints =
            8 * field_op_gate_count                           // 8 × eval_mul_and_carry
            + 9                                                // 9 × assert_zero_ext for polynomial optimizations
            + 2                                                // lo_ptr, hi_ptr assert_eq
            + 1                                                // is_real boolean
            + (2 + WORDS_FIELD_ELEMENT + WORDS_FIELD_ELEMENT * 8
               + WORDS_FIELD_ELEMENT * 8 + WORDS_FIELD_ELEMENT) * 3  // memory timestamp constraints
            ;
        let num_reducer = num_gate_constraints + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc = <U256x2048MulPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(&air);

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
impl U256x2048MulPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
