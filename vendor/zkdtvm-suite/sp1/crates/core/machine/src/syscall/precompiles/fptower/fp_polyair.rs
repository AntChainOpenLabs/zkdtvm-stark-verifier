//! PolyAir adaptation of FpOpChip.
//!
//! Bridges `FpOpCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary
//!
//!   Phase 1 (precompute_lc):
//!     #1 .. field_op_num:     output FieldOpCols range checks (U8Range + U8Range + U16Range)
//!     #next .. +field_lt_num: output_range LTU + BitVec
//!     #next .. +WordsFieldElement*4: y_access memory_read (4 each)
//!     #next .. +WordsFieldElement*4: x_access memory_readwrite (4 each)
//!     #next:   recv(Syscall)
//!     #next:   output.witness(β)
//!     #next:   x_access.prev_value(β)
//!     #next:   y_access.value(β)
//!     #next:   output.result(β)
//!     #next:   output.carry(β)
//!     #next:   assert_all_eq polynomial optimization (precomputed diff(β))
//!
//!   Phase 2 (eval): gate constraints (incl. booleans for is_add, is_sub, is_mul, is_real)
//!   Phase 3 (lookup): send/recv multiplicities

use std::{marker::PhantomData, ops::Deref};

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    params::{Limbs, NumLimbs, NumWords},
    weierstrass::{FieldType, FpOpField},
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol, Polynomial},
    InteractionKind, Word,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use typenum::Unsigned;

use crate::{
    memory::{
        polyair::{
            memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
            memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
        },
        MemoryAccessCols,
    },
    operations::field::{
        field_op::{
            field_op_beta_from_coeffs, field_op_lookup, field_op_num_interactions,
            field_op_precompute_lc, field_op_precompute_witness_beta,
            field_op_variable_gate_constraints, FieldOpBetaConsts,
        },
        range::{
            field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
            field_lt_precompute_lc, FieldLtCols,
        },
    },
};

use super::fp::{num_fp_cols, FpOpCols};

// ============================================================================
// Constants (computed from type parameters)
// ============================================================================

/// Compute total lookup interactions for FpOpChip<P>.
///
/// = field_op_num_interactions<P>           (output range checks)
/// + field_lt_num_interactions<P>           (output_range: LTU + BitVec)
/// + WordsFieldElement * 4                 (y_access memory_read)
/// + WordsFieldElement * 4                 (x_access memory_readwrite)
/// + 1                                     (recv Syscall)
const fn num_lookups<P: FpOpField>() -> usize {
    field_op_num_interactions::<P>() +
        field_lt_num_interactions::<P>() +
        <P as NumWords>::WordsFieldElement::USIZE * 4 +
        <P as NumWords>::WordsFieldElement::USIZE * 4 +
        1
}

/// Precomputed values: one per lookup + witness/input/output beta caches + `diff(beta)` for
/// the `assert_all_eq` optimization.
pub(crate) const fn num_precomputed<P: FpOpField>() -> usize {
    num_lookups::<P>() + 6
}

// ============================================================================
// Column offsets within `FpOpCols<u8, P>` (1 byte == 1 column index)
//
// Layout (#[repr(C)], from fp.rs):
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  is_add
//   [4]  is_sub
//   [5]  is_mul
//   [6]  x_ptr            ← precompute-only (skipped from reserved_poly)
//   [7]  y_ptr            ← precompute-only (skipped from reserved_poly)
//   [8 + i*13 ..]                      x_access[i] = MemoryWriteCols
//     +0..+4   prev_value
//     +4..+8   access.value           ← precompute-only (skipped)
//     +8       access.prev_shard
//     +9       access.prev_clk
//     +10      access.compare_clk
//     +11      access.diff_16bit_limb
//     +12      access.diff_12bit_limb
//   [8 + W*13 + i*9 ..]                y_access[i] = MemoryReadCols (full 9 bytes used)
//   [8 + W*22 .. + L]                  output.result
//   [.. + L]                           output.carry
//   [.. + W_W]                         output.witness        ← precompute-only (skipped)
//   [.. + L]                           output_range.byte_flags
//   [+0]                               output_range.lhs_comparison_byte
//   [+1]                               output_range.rhs_comparison_byte
// ============================================================================

const COL_IS_REAL: usize = 0;
const COL_SHARD: usize = 1;
const COL_CLK: usize = 2;
const COL_IS_ADD: usize = 3;
const COL_IS_SUB: usize = 4;
const COL_IS_MUL: usize = 5;
const COL_X_ACCESS_BASE: usize = 8;
const MEM_WRITE_COLS_SIZE: usize = 13;
const MEM_READ_COLS_SIZE: usize = 9;
const MEM_ACCESS_PREV_SHARD_OFF: usize = 4;
const MEM_ACCESS_PREV_CLK_OFF: usize = 5;
const MEM_ACCESS_COMPARE_CLK_OFF: usize = 6;
const MEM_ACCESS_DIFF_16_OFF: usize = 7;
const MEM_ACCESS_DIFF_12_OFF: usize = 8;

#[inline]
fn col_y_access_base<P: FpOpField>() -> usize {
    let w = <P as NumWords>::WordsFieldElement::USIZE;
    COL_X_ACCESS_BASE + w * MEM_WRITE_COLS_SIZE
}

#[inline]
fn col_output_base<P: FpOpField>() -> usize {
    let w = <P as NumWords>::WordsFieldElement::USIZE;
    COL_X_ACCESS_BASE + w * (MEM_WRITE_COLS_SIZE + MEM_READ_COLS_SIZE)
}

#[inline]
fn col_output_range_base<P: FpOpField>() -> usize {
    col_output_base::<P>() + 2 * P::NB_LIMBS + P::NB_WITNESS_LIMBS
}

// ============================================================================
// Layout of the reserved_poly row (positions in the reserved slice).
// This is the order columns are emitted by `reserved_poly()`.
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  is_add
//   [4]  is_sub
//   [5]  is_mul
//   [6 + i*5 + 0]      x_access[i].access.prev_shard
//   [6 + i*5 + 1]      x_access[i].access.prev_clk
//   [6 + i*5 + 2]      x_access[i].access.compare_clk
//   [6 + i*5 + 3]      x_access[i].access.diff_16bit_limb
//   [6 + i*5 + 4]      x_access[i].access.diff_12bit_limb
//   [6 + W*5 + i*5 + 0]        y_access[i].access.prev_shard
//   [6 + W*5 + i*5 + 1]        y_access[i].access.prev_clk
//   [6 + W*5 + i*5 + 2]        y_access[i].access.compare_clk
//   [6 + W*5 + i*5 + 3]        y_access[i].access.diff_16bit_limb
//   [6 + W*5 + i*5 + 4]        y_access[i].access.diff_12bit_limb
//   [6 + 2*W*5 + 0..+L]   output.result
//   [..      + 0..+L]     output_range.byte_flags
//   [..      + 0]         output_range.lhs_comparison_byte
//   [..      + 1]         output_range.rhs_comparison_byte
//
// NOTE: x_access[i].prev_value and y_access[i].access.value are NOT in
// reserved_poly — they are consumed as β-evaluations (p_beta, q_beta)
// computed in precompute_lc and stored in the precomputed slice.
// ============================================================================

const RES_NUM_SCALAR: usize = 6;
const RES_PER_X_ACCESS: usize = 5; // 5 timestamp fields only (prev_value removed)
const RES_PER_Y_ACCESS: usize = 5; // 5 timestamp fields only (access.value removed)

#[inline]
fn res_x_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_X_ACCESS
}
#[inline]
fn res_y_access_base<P: FpOpField>(i: usize) -> usize {
    let w = <P as NumWords>::WordsFieldElement::USIZE;
    RES_NUM_SCALAR + w * RES_PER_X_ACCESS + i * RES_PER_Y_ACCESS
}
#[inline]
fn res_output_result_base<P: FpOpField>() -> usize {
    let w = <P as NumWords>::WordsFieldElement::USIZE;
    RES_NUM_SCALAR + w * (RES_PER_X_ACCESS + RES_PER_Y_ACCESS)
}
#[inline]
fn res_output_range_base<P: FpOpField>() -> usize {
    res_output_result_base::<P>() + P::NB_LIMBS
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for FpOpChip.
#[derive(Clone, Copy)]
pub struct FpOpPolyAir<P: FpOpField> {
    _marker: PhantomData<P>,
}

impl<P: FpOpField> Default for FpOpPolyAir<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: FpOpField> FpOpPolyAir<P> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<P: FpOpField, AB: FullAirBuilder> FullAir<AB> for FpOpPolyAir<P>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        num_fp_cols::<P>()
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<P>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - x_ptr, y_ptr           (only address inputs to memory/syscall LCs)
        //   - x_access[i].prev_value (consumed as p_beta in precompute_lc)
        //   - x_access[i].access.value (consistency vs output.result is enforced via the
        //     precomputed diff(β) polynomial)
        //   - y_access[i].access.value (consumed as q_beta in precompute_lc)
        //   - output.witness         (collapsed into precomputed witness(β))
        //   - output.carry           (collapsed into precomputed carry(β))
        let w = <P as NumWords>::WordsFieldElement::USIZE;
        let l = P::NB_LIMBS;

        let mut cols: Vec<PairCol> =
            Vec::with_capacity(RES_NUM_SCALAR + 2 * w * RES_PER_X_ACCESS + 2 * l + 2);

        // Scalars (skip x_ptr, y_ptr at indices 6, 7).
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_IS_ADD));
        cols.push(PairCol::Main(COL_IS_SUB));
        cols.push(PairCol::Main(COL_IS_MUL));

        // x_access[i]: 5 timestamp fields only. Skip prev_value (4 cols) and access.value (4 cols).
        for i in 0..w {
            let base = COL_X_ACCESS_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_12_OFF));
        }

        // y_access[i]: 5 timestamp fields only. Skip access.value (4 cols).
        let y_base_main = col_y_access_base::<P>();
        for i in 0..w {
            let base = y_base_main + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_12_OFF));
        }

        // output.result (skip output.carry/output.witness).
        let out_base = col_output_base::<P>();
        for k in 0..l {
            cols.push(PairCol::Main(out_base + k));
        }

        // output_range (all of it: byte_flags + 2 comparison bytes).
        let or_base = col_output_range_base::<P>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(or_base + k));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators + polynomial optimizations
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &FpOpCols<AB::VarMaybeExt, P> = unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let x_ptr = local.x_ptr.clone();
        let y_ptr = local.y_ptr.clone();
        let is_add = local.is_add.clone();
        let is_sub = local.is_sub.clone();
        let is_mul = local.is_mul.clone();
        let is_real = local.is_real.clone();

        let num_words_field_element = <P as NumLimbs>::Limbs::USIZE / 4;
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));

        // =================================================================
        // output FieldOpCols range checks
        // =================================================================
        field_op_precompute_lc::<AB, P>(
            builder,
            &local.output.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.output.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.output.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // =================================================================
        // output_range: LTU + BitVec for byte_flags
        // =================================================================
        {
            let flags: Vec<AB::VarMaybeExt> =
                local.output_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, P>(
                builder,
                local.output_range.lhs_comparison_byte.clone(),
                local.output_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // =================================================================
        // y_access: WordsFieldElement memory_read (4 interactions each)
        // =================================================================
        for i in 0..<P as NumWords>::WordsFieldElement::USIZE {
            let addr = y_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.y_access[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // x_access: WordsFieldElement memory_readwrite (4 interactions each)
        // We read p at clk+1 since p, q could be the same.
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..<P as NumWords>::WordsFieldElement::USIZE {
            let addr = x_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.x_access[i].access,
                &local.x_access[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // recv(Syscall) — syscall registration
        // Syscall ID is computed from operation selectors:
        //   syscall_id = is_add * add_id + is_sub * sub_id + is_mul * mul_id
        // =================================================================
        let (add_id_val, sub_id_val, mul_id_val) = match P::FIELD_TYPE {
            FieldType::Bn254 => (
                SyscallCode::BN254_FP_ADD.syscall_id(),
                SyscallCode::BN254_FP_SUB.syscall_id(),
                SyscallCode::BN254_FP_MUL.syscall_id(),
            ),
            FieldType::Bls12381 => (
                SyscallCode::BLS12381_FP_ADD.syscall_id(),
                SyscallCode::BLS12381_FP_SUB.syscall_id(),
                SyscallCode::BLS12381_FP_MUL.syscall_id(),
            ),
        };
        let add_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(add_id_val));
        let sub_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(sub_id_val));
        let mul_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(mul_id_val));
        let syscall_id = is_add * add_id + is_sub * sub_id + is_mul * mul_id;

        builder.retain_precomputed(
            builder.lookup_denominator(syscall_kind, vec![shard, clk, syscall_id, x_ptr, y_ptr]),
        );

        field_op_precompute_witness_beta::<AB, P>(
            builder,
            &local.output.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        let p_beta = field_op_beta_from_coeffs(
            builder,
            &local
                .x_access
                .iter()
                .flat_map(|acc| acc.prev_value.0.iter().cloned())
                .collect::<Vec<_>>(),
        );
        builder.retain_precomputed(p_beta);

        let q_beta = field_op_beta_from_coeffs(
            builder,
            &local
                .y_access
                .iter()
                .flat_map(|acc| acc.access.value.0.iter().cloned())
                .collect::<Vec<_>>(),
        );
        builder.retain_precomputed(q_beta);

        let result_beta = field_op_beta_from_coeffs(
            builder,
            &local.output.result.0.iter().cloned().collect::<Vec<_>>(),
        );
        builder.retain_precomputed(result_beta);

        let carry_beta = field_op_beta_from_coeffs(
            builder,
            &local.output.carry.0.iter().cloned().collect::<Vec<_>>(),
        );
        builder.retain_precomputed(carry_beta);

        // =================================================================
        // Polynomial optimization for assert_all_eq:
        // compute diff(β) = Σ (output.result[i] - x_value[i]) * β^i
        // =================================================================
        let diff_coeffs: Vec<AB::VarMaybeExt> = local
            .output
            .result
            .0
            .iter()
            .zip(
                local.x_access[..num_words_field_element]
                    .iter()
                    .flat_map(|acc| acc.access.value.0.iter()),
            )
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
        let beta_consts = FieldOpBetaConsts::<AB>::new::<P>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[COL_IS_REAL].clone();
        let shard = local[COL_SHARD].clone();
        let clk = local[COL_CLK].clone();
        let is_add = local[COL_IS_ADD].clone();
        let is_sub = local[COL_IS_SUB].clone();
        let is_mul = local[COL_IS_MUL].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
        let w = <P as NumWords>::WordsFieldElement::USIZE;
        let l = P::NB_LIMBS;

        let (output_witness_beta, p_beta, q_beta, result_beta, carry_beta) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let base = num_lookups::<P>();
            (
                pc[base].clone(),
                pc[base + 1].clone(),
                pc[base + 2].clone(),
                pc[base + 3].clone(),
                pc[base + 4].clone(),
            )
        };

        // -- Boolean constraints for is_add, is_sub, is_mul, is_real --
        builder.assert_zero(is_add.clone() * (one.clone() - is_add.clone()));
        builder.assert_zero(is_sub.clone() * (one.clone() - is_sub.clone()));
        builder.assert_zero(is_mul.clone() * (one.clone() - is_mul.clone()));
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // -- One-hot selector constraint: is_add + is_sub + is_mul = 1 --
        builder.assert_zero(is_add.clone() + is_sub.clone() + is_mul.clone() - one);

        // -- output.eval_variable gate constraints --
        let res_out = res_output_result_base::<P>();
        field_op_variable_gate_constraints::<AB, P>(
            builder,
            p_beta,
            q_beta,
            result_beta,
            carry_beta,
            output_witness_beta,
            is_add,
            is_sub,
            is_mul,
            zero,
            &beta_consts,
        );

        // -- assert_all_eq optimization: use precomputed polynomial value --
        // The last precomputed value is diff(β).
        {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let total_precomputed = num_precomputed::<P>();

            let diff_beta = pc[total_precomputed - 1].clone();

            // when(is_real): output.result == x_value ⟺ diff(β) == 0
            builder.when(is_real.clone()).assert_zero_ext(diff_beta);
        }

        // -- output_range.eval gate constraints --
        {
            let modulus_limbs: Vec<AB::VarMaybeExt> = P::MODULUS
                .iter()
                .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
                .collect();
            let result_limbs_vec: Vec<AB::VarMaybeExt> =
                (0..l).map(|k| local[res_out + k].clone()).collect();
            let or_base = res_output_range_base::<P>();
            let byte_flags: Limbs<AB::VarMaybeExt, <P as NumLimbs>::Limbs> =
                (0..l).map(|k| local[or_base + k].clone()).collect();
            let output_range: FieldLtCols<AB::VarMaybeExt, P> = FieldLtCols {
                byte_flags,
                lhs_comparison_byte: local[or_base + l].clone(),
                rhs_comparison_byte: local[or_base + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, P>(
                builder,
                &result_limbs_vec,
                &modulus_limbs,
                &output_range,
                is_real.clone(),
            );
        }

        // -- memory timestamp constraints --
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());

        // y_access: read at clk
        for i in 0..w {
            let base = res_y_access_base::<P>(i);
            let acc = MemoryAccessCols::<AB::VarMaybeExt> {
                value: zero_word.clone(),
                prev_shard: local[base].clone(),
                prev_clk: local[base + 1].clone(),
                compare_clk: local[base + 2].clone(),
                diff_16bit_limb: local[base + 3].clone(),
                diff_12bit_limb: local[base + 4].clone(),
            };
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }

        // x_access: write at clk+1.
        for i in 0..w {
            let base = res_x_access_base(i);
            let acc = MemoryAccessCols::<AB::VarMaybeExt> {
                value: zero_word.clone(),
                prev_shard: local[base].clone(),
                prev_clk: local[base + 1].clone(),
                compare_clk: local[base + 2].clone(),
                diff_16bit_limb: local[base + 3].clone(),
                diff_12bit_limb: local[base + 4].clone(),
            };
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
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[COL_IS_REAL].clone();

        // output FieldOpCols range checks
        field_op_lookup::<AB, P>(builder, is_real.clone());

        // output_range LTU + BitVec
        field_lt_lookup::<AB, P>(builder, is_real.clone());

        // y_access memory reads
        for _ in 0..<P as NumWords>::WordsFieldElement::USIZE {
            memory_read_lookup(builder, is_real.clone());
        }

        // x_access memory readwrites
        for _ in 0..<P as NumWords>::WordsFieldElement::USIZE {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::fp::FpOpChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, P: FpOpField> BaseAir<F> for FpOpPolyAir<P> {
    fn width(&self) -> usize {
        num_fp_cols::<P>()
    }
}

impl<F: Field, P: FpOpField> MachineAir<F> for FpOpPolyAir<P> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <FpOpChip<P> as MachineAir<F>>::name(&FpOpChip::<P>::new()) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        FpOpChip::<P>::new().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use crate::utils::{words_to_bytes_le_vec, zeroed_f_vec};
        use dt_core_executor::events::{ByteLookupEvent, PrecompileEvent};
        use num::BigUint;
        use std::borrow::BorrowMut;

        <FpOpChip<P> as MachineAir<F>>::generate_dependencies(&FpOpChip::<P>::new(), input, output);

        let events = match P::FIELD_TYPE {
            FieldType::Bn254 => input.get_precompile_events(SyscallCode::BN254_FP_ADD),
            FieldType::Bls12381 => input.get_precompile_events(SyscallCode::BLS12381_FP_ADD),
        };
        for (_, event) in events {
            let event = match (P::FIELD_TYPE, event) {
                (FieldType::Bn254, PrecompileEvent::Bn254Fp(event)) => event,
                (FieldType::Bls12381, PrecompileEvent::Bls12381Fp(event)) => event,
                _ => unreachable!(),
            };
            let p = BigUint::from_bytes_le(&words_to_bytes_le_vec(&event.x));
            let q = BigUint::from_bytes_le(&words_to_bytes_le_vec(&event.y));
            let mut row = zeroed_f_vec(num_fp_cols::<P>());
            let cols: &mut super::fp::FpOpCols<F, P> = row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            FpOpChip::<P>::populate_field_ops(&mut ignored_blu, cols, p, q, event.op);
            add_field_lt_bitvec_lookups::<F, P>(output, &cols.output_range);
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <FpOpChip<P> as MachineAir<F>>::included(&FpOpChip::<P>::new(), shard)
    }

    fn padding_row(&self) -> Vec<F> {
        FpOpChip::<P>::new().padding_row()
    }

    fn local_only(&self) -> bool {
        <FpOpChip<P> as MachineAir<F>>::local_only(&FpOpChip::<P>::new())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::{
        params::FieldParameters,
        weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField},
    };
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
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, Field};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use std::ops::Deref;
    use test_artifacts::{BLS12381_FP_ELF, BN254_FP_ELF};

    use super::super::fp::FpOpChip;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const BATCH_SIZE: usize = 3;

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

    fn beta_powers_for<P: FpOpField>(air: &FpOpPolyAir<P>, beta: EF) -> Vec<EF> {
        let max = <FpOpPolyAir<P> as FullAir<
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

    fn reserved_poly_matrix<P: FpOpField>(
        air: &FpOpPolyAir<P>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <FpOpPolyAir<P> as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    #[test]
    fn test_fp_op_num_precomputed_accounts_for_beta_caches() {
        type P = Bn254BaseField;
        assert_eq!(num_precomputed::<P>(), num_lookups::<P>() + 6);
    }

    #[test]
    fn test_fp_op_reserved_poly_drops_output_carry() {
        type P = Bn254BaseField;
        let air = FpOpPolyAir::<P>::new();
        let reserved =
            <FpOpPolyAir<P> as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);
        let w = <P as NumWords>::WordsFieldElement::USIZE;
        let l = P::NB_LIMBS;
        assert_eq!(reserved.len(), RES_NUM_SCALAR + 2 * w * RES_PER_X_ACCESS + 2 * l + 2);
    }

    /// Build a real trace from a test ELF for the given field type.
    /// FpOpChip coalesces all FP operations (add/sub/mul) under the Add syscall code.
    fn sample_trace_for<P: FpOpField>(elf: &[u8]) -> Option<RowMajorMatrix<F>> {
        let syscall_code = match P::FIELD_TYPE {
            FieldType::Bn254 => dt_core_executor::syscalls::SyscallCode::BN254_FP_ADD,
            FieldType::Bls12381 => dt_core_executor::syscalls::SyscallCode::BLS12381_FP_ADD,
        };

        let program = Program::from(elf).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(syscall_code).is_empty() {
                continue;
            }

            let mut fp_shard = ExecutionRecord::new(shard.program.clone());
            fp_shard.precompile_events = shard.precompile_events.clone();

            let chip = FpOpChip::<P>::new();
            return Some(
                chip.generate_trace(&fp_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    /// Run the full constraint satisfaction check for a given field type P.
    fn run_constraint_check<P: FpOpField>(main: RowMajorMatrix<F>) {
        let air = FpOpPolyAir::<P>::new();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let beta_powers = beta_powers_for(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = num_lookups::<P>();
        let total_precomputed = num_precomputed::<P>();

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            total_precomputed,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            total_lookups,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        // Conservative upper bound for gate constraints.
        let num_gate_constraints = (2 * P::NB_LIMBS - 1)     // FieldOpCols variable vanishing
            + 1                                                // one-hot selector
            + 1                                                // assert_zero_ext for diff_beta
            + (P::NB_LIMBS + 3)                                // FieldLtCols gate constraints
            + <P as NumWords>::WordsFieldElement::USIZE * 2 * 3 // memory timestamp
            + 1  // is_real bool
            + 3; // is_add, is_sub, is_mul bool
        let num_reducer = num_gate_constraints + total_lookups.div_ceil(BATCH_SIZE) + 3;
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
            &beta_powers,
            beta_septix,
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

    #[test]
    fn test_fp_op_bn254_constraint_check() {
        type P = Bn254BaseField;
        let main = match sample_trace_for::<P>(BN254_FP_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bn254FpOp trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<P>(main);
    }

    #[test]
    fn test_fp_op_bls12381_constraint_check() {
        use dt_curves::weierstrass::bls12_381::Bls12381BaseField;
        type P = Bls12381BaseField;
        let main = match sample_trace_for::<P>(BLS12381_FP_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bls12381FpOp trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<P>(main);
    }

    fn random_fp_trace<P: FpOpField>(log_n: usize, _seed: u64, elf: &[u8]) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace_for::<P>(elf).expect("sample trace should exist");
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
        let width = base.width();
        let last_row_start = (base_height - 1) * width;
        let last_row = &base.values[last_row_start..last_row_start + width];
        let mut values = Vec::with_capacity(target_height * width);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, width)
    }

    fn do_perf_multi_round_sumcheck<P: FpOpField>(elf: &[u8]) {
        use crate::syscall::precompiles::perf_test_defaults;
        use dt_curves::weierstrass::FieldType;

        let air = FpOpPolyAir::<P>::new();
        let default_log_n = match P::FIELD_TYPE {
            FieldType::Bn254 => perf_test_defaults::BN254_FP_OP_LOG_N,
            FieldType::Bls12381 => perf_test_defaults::BLS12381_FP_OP_LOG_N,
        };
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_log_n);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_fp_trace::<P>(log_n, seed, elf);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let beta_powers = beta_powers_for(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = num_lookups::<P>();
        let total_precomputed = num_precomputed::<P>();
        let num_gate_constraints = (2 * P::NB_LIMBS - 1)
            + 1
            + 1
            + (P::NB_LIMBS + 3)
            + <P as NumWords>::WordsFieldElement::USIZE * 2 * 3
            + 1  // is_real bool
            + 3; // is_add, is_sub, is_mul bool
        let num_reducer = num_gate_constraints + total_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc =
            <FpOpPolyAir<P> as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

        // Precompute
        let t_precompute = std::time::Instant::now();
        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            total_precomputed,
        );
        let precompute_elapsed = t_precompute.elapsed();
        std::println!("  precompute_linear_combination: {:?}", precompute_elapsed);

        // Permutation
        let t_perm = std::time::Instant::now();
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            total_lookups,
        );
        let perm_elapsed = t_perm.elapsed();
        std::println!("  generate_permutation_trace_: {:?}", perm_elapsed);

        let mut precomputed = trim_rows(&precomputed_full, height);
        let mut permutation = trim_rows(&permutation_full, height);

        // Round 0
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

        // Rounds 1..log_n-1
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
            std::println!("  round {} (nonfirst): {:?}", round, t_round.elapsed());

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

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bn254() {
        do_perf_multi_round_sumcheck::<Bn254BaseField>(BN254_FP_ELF);
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bls12381() {
        do_perf_multi_round_sumcheck::<Bls12381BaseField>(BLS12381_FP_ELF);
    }
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl<P: FpOpField> FpOpPolyAir<P> {
    pub const fn num_lookups(&self) -> usize {
        num_lookups::<P>()
    }
    pub const fn num_precomputed(&self) -> usize {
        num_precomputed::<P>()
    }
}
