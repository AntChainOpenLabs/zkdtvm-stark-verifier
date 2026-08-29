//! PolyAir adaptation of Fp2AddSubAssignChip.
//!
//! Bridges `Fp2AddSubAssignCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary
//!
//!   Phase 1 (precompute_lc):
//!     #1 .. field_add_op_num:  c0 FieldAddOp range checks (U8Range + U16Range)
//!     #next .. +field_add_op_num:  c1 FieldAddOp range checks
//!     #next .. +field_lt_num:  c0_range LTU + BitVec
//!     #next .. +field_lt_num:  c1_range LTU + BitVec
//!     #next .. +WordsCurvePoint*4: y_access memory_read (4 each)
//!     #next .. +WordsCurvePoint*4: x_access memory_readwrite (4 each)
//!     #next:   recv(Syscall)
//!     #next:   c0.witness(β)
//!     #next:   c1.witness(β)
//!     #next:   p_x(β)   (β-evaluation of x_access[..N].prev_value)
//!     #next:   p_y(β)   (β-evaluation of x_access[N..].prev_value)
//!     #next:   q_x(β)   (β-evaluation of y_access[..N].access.value)
//!     #next:   q_y(β)   (β-evaluation of y_access[N..].access.value)
//!     #next:   assert_all_eq c0 polynomial optimization (precomputed diff(β))
//!     #next:   assert_all_eq c1 polynomial optimization (precomputed diff(β))
//!
//!   Phase 2 (eval): gate constraints
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
    memory::polyair::{
        memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
        memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
    },
    operations::field::{
        field_add_op::{
            field_add_op_lookup, field_add_op_num_interactions, field_add_op_precompute_lc,
            field_add_op_variable_gate_constraints_from_betas,
        },
        field_op::{
            field_op_beta_from_coeffs, field_op_precompute_witness_beta, FieldOpBetaConsts,
        },
        range::{
            field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
            field_lt_precompute_lc,
        },
    },
};

use crate::{memory::MemoryAccessCols, operations::field::range::FieldLtCols};

use super::fp2_addsub::{num_fp2_addsub_cols, Fp2AddSubAssignCols};

// ============================================================================
// Constants (computed from type parameters)
// ============================================================================

/// Compute total lookup interactions for Fp2AddSubAssignChip<P>.
///
/// = 2 * field_add_op_num_interactions<P>  (c0, c1 range checks)
/// + 2 * field_lt_num_interactions<P>      (c0_range, c1_range: LTU + BitVec)
/// + WordsCurvePoint * 4                   (y_access memory_read)
/// + WordsCurvePoint * 4                   (x_access memory_readwrite)
/// + 1                                     (recv Syscall)
const fn num_lookups<P: FpOpField>() -> usize {
    2 * field_add_op_num_interactions::<P>() +
        2 * field_lt_num_interactions::<P>() +
        <P as NumWords>::WordsCurvePoint::USIZE * 4 +
        <P as NumWords>::WordsCurvePoint::USIZE * 4 +
        1
}

/// Precomputed values: one per lookup + `c0.witness(beta)` + `c1.witness(beta)`
/// + `p_x(beta)` + `p_y(beta)` + `q_x(beta)` + `q_y(beta)`
/// + two `diff(beta)` values for the `assert_all_eq` optimizations.
const fn num_precomputed<P: FpOpField>() -> usize {
    num_lookups::<P>() + 8
}

/// Compute total gate constraints for `Fp2AddSubPolyAir<P>`.
///
/// This must stay in sync with `eval()`:
/// - 1 × assert_bool(is_add)
/// - 2 × field_add_op_variable_gate_constraints => 2 * (NB_LIMBS + 1)
/// - 2 × assert_zero_ext for c0/c1 polynomial linkage
/// - 2 × field_lt_gate_constraints => 2 * (NB_LIMBS + 3)
/// - 2 * WordsCurvePoint memory timestamp constraints, each contributes 3 constraints
/// - 1 × assert_bool(is_real)
const fn num_gate_constraints<P: FpOpField>() -> usize {
    1 + 2 * (P::NB_LIMBS + 1) +
        2 +
        2 * (P::NB_LIMBS + 3) +
        <P as NumWords>::WordsCurvePoint::USIZE * 2 * 3 +
        1
}

// ============================================================================
// Column offsets within Fp2AddSubAssignCols<u8, P> (1 byte = 1 column index)
//
// Layout (#[repr(C)]):
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  is_add
//   [4]  x_ptr            ← precompute-only (skipped from reserved_poly)
//   [5]  y_ptr            ← precompute-only (skipped from reserved_poly)
//   [6 + i*13 ..]         x_access[i] = MemoryWriteCols
//     +0..+4   prev_value
//     +4..+8   access.value           ← precompute-only (skipped)
//     +8       access.prev_shard
//     +9       access.prev_clk
//     +10      access.compare_clk
//     +11      access.diff_16bit_limb
//     +12      access.diff_12bit_limb
//   [6 + WCP*13 + i*9 ..] y_access[i] = MemoryReadCols (full 9 cols used)
//   [.. + L]   c0.result
//   [.. + 1]   c0.carry
//   [.. + AW]  c0.witness             ← precompute-only (skipped)
//   [.. + L]   c1.result
//   [.. + 1]   c1.carry
//   [.. + AW]  c1.witness             ← precompute-only (skipped)
//   [.. + L]   c0_range.byte_flags
//   [+0]       c0_range.lhs_comparison_byte
//   [+1]       c0_range.rhs_comparison_byte
//   [.. + L]   c1_range.byte_flags
//   [+0]       c1_range.lhs_comparison_byte
//   [+1]       c1_range.rhs_comparison_byte
// ============================================================================

const COL_IS_REAL: usize = 0;
const COL_SHARD: usize = 1;
const COL_CLK: usize = 2;
const COL_IS_ADD: usize = 3;
const COL_X_ACCESS_BASE: usize = 6;
const MEM_WRITE_COLS_SIZE: usize = 13;
const MEM_READ_COLS_SIZE: usize = 9;
const MEM_ACCESS_PREV_SHARD_OFF: usize = 4;
const MEM_ACCESS_PREV_CLK_OFF: usize = 5;
const MEM_ACCESS_COMPARE_CLK_OFF: usize = 6;
const MEM_ACCESS_DIFF_16_OFF: usize = 7;
const MEM_ACCESS_DIFF_12_OFF: usize = 8;

#[inline]
fn col_y_access_base<P: FpOpField>() -> usize {
    COL_X_ACCESS_BASE + <P as NumWords>::WordsCurvePoint::USIZE * MEM_WRITE_COLS_SIZE
}

#[inline]
fn col_c0_base<P: FpOpField>() -> usize {
    col_y_access_base::<P>() + <P as NumWords>::WordsCurvePoint::USIZE * MEM_READ_COLS_SIZE
}

#[inline]
fn col_c1_base<P: FpOpField>() -> usize {
    col_c0_base::<P>() + P::NB_LIMBS + 1 + P::NB_ADD_WITNESS_LIMBS
}

#[inline]
fn col_c0_range_base<P: FpOpField>() -> usize {
    col_c1_base::<P>() + P::NB_LIMBS + 1 + P::NB_ADD_WITNESS_LIMBS
}

#[inline]
fn col_c1_range_base<P: FpOpField>() -> usize {
    col_c0_range_base::<P>() + P::NB_LIMBS + 2
}

// ============================================================================
// Reserved-poly row layout (positions in the reserved slice).
// Order matches reserved_poly() emission order.
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  is_add
//   [4 + i*5 + 0]      x_access[i].access.prev_shard
//   [4 + i*5 + 1]      x_access[i].access.prev_clk
//   [4 + i*5 + 2]      x_access[i].access.compare_clk
//   [4 + i*5 + 3]      x_access[i].access.diff_16bit_limb
//   [4 + i*5 + 4]      x_access[i].access.diff_12bit_limb
//   [4 + WCP*5 + i*5 + 0]      y_access[i].access.prev_shard
//   [4 + WCP*5 + i*5 + 1]      y_access[i].access.prev_clk
//   [4 + WCP*5 + i*5 + 2]      y_access[i].access.compare_clk
//   [4 + WCP*5 + i*5 + 3]      y_access[i].access.diff_16bit_limb
//   [4 + WCP*5 + i*5 + 4]      y_access[i].access.diff_12bit_limb
//   [4 + 2*WCP*5 + 0..+L]  c0.result
//   [..          + 0]       c0.carry
//   [..          + 0..+L]   c1.result
//   [..          + 0]       c1.carry
//   [..          + 0..+L]   c0_range.byte_flags
//   [..          + 0]       c0_range.lhs_comparison_byte
//   [..          + 1]       c0_range.rhs_comparison_byte
//   [..          + 0..+L]   c1_range.byte_flags
//   [..          + 0]       c1_range.lhs_comparison_byte
//   [..          + 1]       c1_range.rhs_comparison_byte
//
// NOTE: x_access[i].prev_value and y_access[i].access.value are NOT in
// reserved_poly — they are consumed as β-evaluations (p_x_beta, p_y_beta,
// q_x_beta, q_y_beta) computed in precompute_lc and stored in the precomputed slice.
// ============================================================================

const RES_NUM_SCALAR: usize = 4;
const RES_PER_ACCESS: usize = 5; // 5 timestamp fields only (prev_value/access.value removed)

#[inline]
fn res_x_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_ACCESS
}
#[inline]
fn res_y_access_base<P: FpOpField>(i: usize) -> usize {
    let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
    RES_NUM_SCALAR + wcp * RES_PER_ACCESS + i * RES_PER_ACCESS
}
#[inline]
fn res_c0_result_base<P: FpOpField>() -> usize {
    let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
    RES_NUM_SCALAR + wcp * 2 * RES_PER_ACCESS
}
#[inline]
fn res_c0_carry<P: FpOpField>() -> usize {
    res_c0_result_base::<P>() + P::NB_LIMBS
}
#[inline]
fn res_c1_result_base<P: FpOpField>() -> usize {
    res_c0_carry::<P>() + 1
}
#[inline]
fn res_c1_carry<P: FpOpField>() -> usize {
    res_c1_result_base::<P>() + P::NB_LIMBS
}
#[inline]
fn res_c0_range_base<P: FpOpField>() -> usize {
    res_c1_carry::<P>() + 1
}
#[inline]
fn res_c1_range_base<P: FpOpField>() -> usize {
    res_c0_range_base::<P>() + P::NB_LIMBS + 2
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for Fp2AddSubAssignChip.
#[derive(Clone, Copy)]
pub struct Fp2AddSubPolyAir<P: FpOpField> {
    _marker: PhantomData<P>,
}

impl<P: FpOpField> Default for Fp2AddSubPolyAir<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: FpOpField> Fp2AddSubPolyAir<P> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<P: FpOpField, AB: FullAirBuilder> FullAir<AB> for Fp2AddSubPolyAir<P>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        num_fp2_addsub_cols::<P>()
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<P>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - x_ptr, y_ptr           (only address inputs to memory/syscall LCs)
        //   - x_access[i].prev_value (consumed as p_x_beta/p_y_beta in precompute_lc)
        //   - x_access[i].access.value (consistency vs c0/c1.result is enforced via precomputed
        //     diff(β) polynomial)
        //   - y_access[i].access.value (consumed as q_x_beta/q_y_beta in precompute_lc)
        //   - c0.witness, c1.witness (collapsed into precomputed witness(β))
        let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
        let l = P::NB_LIMBS;

        let mut cols: Vec<PairCol> = Vec::with_capacity(
            RES_NUM_SCALAR + 2 * wcp * RES_PER_ACCESS + 2 * (l + 1) + 2 * (l + 2),
        );

        // Scalars (skip x_ptr at index 4, y_ptr at index 5).
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_IS_ADD));

        // x_access[i]: 5 timestamp fields only. Skip prev_value (4 cols) and access.value (4 cols).
        for i in 0..wcp {
            let base = COL_X_ACCESS_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_12_OFF));
        }

        // y_access[i]: 5 timestamp fields only. Skip access.value (4 cols).
        let y_base_main = col_y_access_base::<P>();
        for i in 0..wcp {
            let base = y_base_main + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_12_OFF));
        }

        // c0.result and c0.carry (skip c0.witness).
        let c0_base = col_c0_base::<P>();
        for k in 0..l {
            cols.push(PairCol::Main(c0_base + k));
        }
        cols.push(PairCol::Main(c0_base + l));

        // c1.result and c1.carry (skip c1.witness).
        let c1_base = col_c1_base::<P>();
        for k in 0..l {
            cols.push(PairCol::Main(c1_base + k));
        }
        cols.push(PairCol::Main(c1_base + l));

        // c0_range (all: byte_flags + 2 comparison bytes).
        let c0r_base = col_c0_range_base::<P>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(c0r_base + k));
        }

        // c1_range (all: byte_flags + 2 comparison bytes).
        let c1r_base = col_c1_range_base::<P>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(c1r_base + k));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators + polynomial optimizations
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Fp2AddSubAssignCols<AB::VarMaybeExt, P> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let x_ptr = local.x_ptr.clone();
        let y_ptr = local.y_ptr.clone();

        let num_words_field_element = <P as NumLimbs>::Limbs::USIZE / 4;
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));

        // =================================================================
        // c0 FieldAddOp range checks
        // =================================================================
        field_add_op_precompute_lc::<AB, P>(
            builder,
            &local.c0.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.c0.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // =================================================================
        // c1 FieldAddOp range checks
        // =================================================================
        field_add_op_precompute_lc::<AB, P>(
            builder,
            &local.c1.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.c1.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // =================================================================
        // c0_range: LTU + BitVec for byte_flags
        // =================================================================
        {
            let c0_flags: Vec<AB::VarMaybeExt> =
                local.c0_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, P>(
                builder,
                local.c0_range.lhs_comparison_byte.clone(),
                local.c0_range.rhs_comparison_byte.clone(),
                &c0_flags,
            );
        }

        // =================================================================
        // c1_range: LTU + BitVec for byte_flags
        // =================================================================
        {
            let c1_flags: Vec<AB::VarMaybeExt> =
                local.c1_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, P>(
                builder,
                local.c1_range.lhs_comparison_byte.clone(),
                local.c1_range.rhs_comparison_byte.clone(),
                &c1_flags,
            );
        }

        // =================================================================
        // y_access: WordsCurvePoint memory_read (4 interactions each)
        // =================================================================
        for i in 0..<P as NumWords>::WordsCurvePoint::USIZE {
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
        // x_access: WordsCurvePoint memory_readwrite (4 interactions each)
        // We read p at clk+1 since p, q could be the same.
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..<P as NumWords>::WordsCurvePoint::USIZE {
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
        // =================================================================
        let (add_syscall_id, sub_syscall_id) = match P::FIELD_TYPE {
            FieldType::Bn254 => {
                (SyscallCode::BN254_FP2_ADD.syscall_id(), SyscallCode::BN254_FP2_SUB.syscall_id())
            }
            FieldType::Bls12381 => (
                SyscallCode::BLS12381_FP2_ADD.syscall_id(),
                SyscallCode::BLS12381_FP2_SUB.syscall_id(),
            ),
        };
        let add_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(add_syscall_id));
        let sub_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(sub_syscall_id));
        // syscall_id = is_add * add_id + (1 - is_add) * sub_id
        let syscall_id = local.is_add.clone() * add_id +
            (AB::VarMaybeExt::from(AB::F::one()) - local.is_add.clone()) * sub_id;

        builder.retain_precomputed(
            builder.lookup_denominator(syscall_kind, vec![shard, clk, syscall_id, x_ptr, y_ptr]),
        );

        field_op_precompute_witness_beta::<AB, P>(
            builder,
            &local.c0.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, P>(
            builder,
            &local.c1.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // =================================================================
        // Beta-evaluations of operand limbs (p_x, p_y from x_access.prev_value;
        // q_x, q_y from y_access.access.value).
        // These replace reading individual limb columns from reserved_poly
        // during eval — instead, eval reads the precomputed scalar directly.
        // =================================================================
        let p_x_coeffs: Vec<AB::VarMaybeExt> = local.x_access[..num_words_field_element]
            .iter()
            .flat_map(|a| a.prev_value.0.iter().cloned())
            .collect();
        let p_y_coeffs: Vec<AB::VarMaybeExt> = local.x_access[num_words_field_element..]
            .iter()
            .flat_map(|a| a.prev_value.0.iter().cloned())
            .collect();
        let q_x_coeffs: Vec<AB::VarMaybeExt> = local.y_access[..num_words_field_element]
            .iter()
            .flat_map(|a| a.access.value.0.iter().cloned())
            .collect();
        let q_y_coeffs: Vec<AB::VarMaybeExt> = local.y_access[num_words_field_element..]
            .iter()
            .flat_map(|a| a.access.value.0.iter().cloned())
            .collect();

        let p_x_beta = field_op_beta_from_coeffs::<AB>(builder, &p_x_coeffs);
        let p_y_beta = field_op_beta_from_coeffs::<AB>(builder, &p_y_coeffs);
        let q_x_beta = field_op_beta_from_coeffs::<AB>(builder, &q_x_coeffs);
        let q_y_beta = field_op_beta_from_coeffs::<AB>(builder, &q_y_coeffs);
        builder.retain_precomputed(p_x_beta);
        builder.retain_precomputed(p_y_beta);
        builder.retain_precomputed(q_x_beta);
        builder.retain_precomputed(q_y_beta);

        // =================================================================
        // Polynomial optimization for assert_all_eq:
        // Instead of NB_LIMBS individual assert_eq constraints per component,
        // compute diff(β) = Σ (c_result[i] - x_value[i]) * β^i
        // and retain the result. In eval, a single assert_zero_ext suffices.
        //
        // Pre-copy diff coefficients before borrowing beta_powers to
        // avoid holding an immutable borrow across retain_precomputed.
        // =================================================================
        let c0_diff_coeffs: Vec<AB::VarMaybeExt> = local
            .c0
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

        let c1_diff_coeffs: Vec<AB::VarMaybeExt> = local
            .c1
            .result
            .0
            .iter()
            .zip(
                local.x_access[num_words_field_element..]
                    .iter()
                    .flat_map(|acc| acc.access.value.0.iter()),
            )
            .map(|(r, v)| r.clone() - v.clone())
            .collect();

        // Now borrow beta_powers, compute both polynomial evaluations, drop the borrow.
        let (c0_diff_beta, c1_diff_beta) = {
            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            let c0 = Polynomial::from_coefficients(&c0_diff_coeffs)
                .eval_with_powers(beta_powers, zero_ext.clone());
            let c1 = Polynomial::from_coefficients(&c1_diff_coeffs)
                .eval_with_powers(beta_powers, zero_ext);
            (c0, c1)
        };
        builder.retain_precomputed(c0_diff_beta);
        builder.retain_precomputed(c1_diff_beta);
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
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero]);
        let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
        let l = P::NB_LIMBS;

        // -- Read all precomputed beta-scalars in one borrow --
        let (
            c0_witness_beta,
            c1_witness_beta,
            p_x_beta,
            p_y_beta,
            q_x_beta,
            q_y_beta,
            c0_diff_beta,
            c1_diff_beta,
        ) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let nl = num_lookups::<P>();
            (
                pc[nl].clone(),     // c0_witness_beta
                pc[nl + 1].clone(), // c1_witness_beta
                pc[nl + 2].clone(), // p_x_beta
                pc[nl + 3].clone(), // p_y_beta
                pc[nl + 4].clone(), // q_x_beta
                pc[nl + 5].clone(), // q_y_beta
                pc[nl + 6].clone(), // c0_diff_beta
                pc[nl + 7].clone(), // c1_diff_beta
            )
        };

        // -- assert_bool(is_add) --
        builder.assert_zero(is_add.clone() * (one.clone() - is_add.clone()));

        // Extract c0 add/sub result from reserved positions.
        let res_c0 = res_c0_result_base::<P>();
        let c0_result: Limbs<AB::VarMaybeExt, <P as NumLimbs>::Limbs> =
            (0..l).map(|k| local[res_c0 + k].clone()).collect();
        let c0_carry = local[res_c0_carry::<P>()].clone();

        field_add_op_variable_gate_constraints_from_betas::<AB, P>(
            builder,
            p_x_beta,
            q_x_beta,
            &c0_result,
            c0_carry,
            c0_witness_beta,
            is_add.clone(),
            &beta_consts,
        );

        // Extract c1 add/sub result from reserved positions.
        let res_c1 = res_c1_result_base::<P>();
        let c1_result: Limbs<AB::VarMaybeExt, <P as NumLimbs>::Limbs> =
            (0..l).map(|k| local[res_c1 + k].clone()).collect();
        let c1_carry = local[res_c1_carry::<P>()].clone();

        field_add_op_variable_gate_constraints_from_betas::<AB, P>(
            builder,
            p_y_beta,
            q_y_beta,
            &c1_result,
            c1_carry,
            c1_witness_beta,
            is_add,
            &beta_consts,
        );

        // -- assert_all_eq optimization: use precomputed polynomial values --
        {
            builder.when(is_real.clone()).assert_zero_ext(c0_diff_beta);
            builder.when(is_real.clone()).assert_zero_ext(c1_diff_beta);
        }

        // -- c0_range / c1_range gate constraints --
        {
            let modulus_limbs: Vec<AB::VarMaybeExt> = P::MODULUS
                .iter()
                .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
                .collect();

            let c0_result_limbs: Vec<AB::VarMaybeExt> =
                (0..l).map(|k| local[res_c0 + k].clone()).collect();
            let c0r = res_c0_range_base::<P>();
            let c0_range = FieldLtCols::<AB::VarMaybeExt, P> {
                byte_flags: (0..l).map(|k| local[c0r + k].clone()).collect(),
                lhs_comparison_byte: local[c0r + l].clone(),
                rhs_comparison_byte: local[c0r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, P>(
                builder,
                &c0_result_limbs,
                &modulus_limbs,
                &c0_range,
                is_real.clone(),
            );

            let c1_result_limbs: Vec<AB::VarMaybeExt> =
                (0..l).map(|k| local[res_c1 + k].clone()).collect();
            let c1r = res_c1_range_base::<P>();
            let c1_range = FieldLtCols::<AB::VarMaybeExt, P> {
                byte_flags: (0..l).map(|k| local[c1r + k].clone()).collect(),
                lhs_comparison_byte: local[c1r + l].clone(),
                rhs_comparison_byte: local[c1r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, P>(
                builder,
                &c1_result_limbs,
                &modulus_limbs,
                &c1_range,
                is_real.clone(),
            );
        }

        // -- memory timestamp constraints --
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());

        // y_access: read at clk
        for i in 0..wcp {
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

        // x_access: write at clk+1
        for i in 0..wcp {
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

        // -- assert_bool(is_real) --
        builder.assert_zero(is_real.clone() * (one - is_real));
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[COL_IS_REAL].clone();

        // c0 FieldAddOp range checks
        field_add_op_lookup::<AB, P>(builder, is_real.clone());

        // c1 FieldAddOp range checks
        field_add_op_lookup::<AB, P>(builder, is_real.clone());

        // c0_range LTU + BitVec
        field_lt_lookup::<AB, P>(builder, is_real.clone());

        // c1_range LTU + BitVec
        field_lt_lookup::<AB, P>(builder, is_real.clone());

        // y_access memory reads
        for _ in 0..<P as NumWords>::WordsCurvePoint::USIZE {
            memory_read_lookup(builder, is_real.clone());
        }

        // x_access memory readwrites
        for _ in 0..<P as NumWords>::WordsCurvePoint::USIZE {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::fp2_addsub::Fp2AddSubAssignChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, P: FpOpField> BaseAir<F> for Fp2AddSubPolyAir<P> {
    fn width(&self) -> usize {
        num_fp2_addsub_cols::<P>()
    }
}

impl<F: Field, P: FpOpField> MachineAir<F> for Fp2AddSubPolyAir<P> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <Fp2AddSubAssignChip<P> as MachineAir<F>>::name(&Fp2AddSubAssignChip::<P>::new()) +
            "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Fp2AddSubAssignChip::<P>::new().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use crate::utils::{words_to_bytes_le_vec, zeroed_f_vec};
        use dt_core_executor::events::{ByteLookupEvent, PrecompileEvent};
        use num::BigUint;
        use std::borrow::BorrowMut;

        // Default base-chip generate_dependencies (no-op for Fp2AddSubAssignChip, but
        // call it for parity with the other PolyAir adapters in case it gains BLU emission later).
        <Fp2AddSubAssignChip<P> as MachineAir<F>>::generate_dependencies(
            &Fp2AddSubAssignChip::<P>::new(),
            input,
            output,
        );

        // PolyAir-only BitVec lookups for c0_range and c1_range (from field_lt_precompute_lc).
        // Base chip coalesces ADD/SUB events onto the ADD syscall id (see fp2_addsub.rs::included),
        // so we iterate that single list.
        let events = match P::FIELD_TYPE {
            FieldType::Bn254 => input.get_precompile_events(SyscallCode::BN254_FP2_ADD),
            FieldType::Bls12381 => input.get_precompile_events(SyscallCode::BLS12381_FP2_ADD),
        };
        for (_, event) in events {
            let event = match (P::FIELD_TYPE, event) {
                (FieldType::Bn254, PrecompileEvent::Bn254Fp2AddSub(event)) => event,
                (FieldType::Bls12381, PrecompileEvent::Bls12381Fp2AddSub(event)) => event,
                _ => unreachable!(),
            };

            let p = &event.x;
            let q = &event.y;
            let p_x = BigUint::from_bytes_le(&words_to_bytes_le_vec(&p[..p.len() / 2]));
            let p_y = BigUint::from_bytes_le(&words_to_bytes_le_vec(&p[p.len() / 2..]));
            let q_x = BigUint::from_bytes_le(&words_to_bytes_le_vec(&q[..q.len() / 2]));
            let q_y = BigUint::from_bytes_le(&words_to_bytes_le_vec(&q[q.len() / 2..]));

            let mut row = zeroed_f_vec(num_fp2_addsub_cols::<P>());
            let cols: &mut Fp2AddSubAssignCols<F, P> = row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            Fp2AddSubAssignChip::<P>::populate_field_ops(
                &mut ignored_blu,
                cols,
                p_x,
                p_y,
                q_x,
                q_y,
                event.op,
            );
            add_field_lt_bitvec_lookups::<F, P>(output, &cols.c0_range);
            add_field_lt_bitvec_lookups::<F, P>(output, &cols.c1_range);
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <Fp2AddSubAssignChip<P> as MachineAir<F>>::included(&Fp2AddSubAssignChip::<P>::new(), shard)
    }

    fn padding_row(&self) -> Vec<F> {
        Fp2AddSubAssignChip::<P>::new().padding_row()
    }

    fn local_only(&self) -> bool {
        <Fp2AddSubAssignChip<P> as MachineAir<F>>::local_only(&Fp2AddSubAssignChip::<P>::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField};
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
    use test_artifacts::{BLS12381_FP2_ADDSUB_ELF, BN254_FP2_ADDSUB_ELF};

    use super::super::fp2_addsub::Fp2AddSubAssignChip;
    use crate::syscall::precompiles::perf_test_defaults;
    use dt_core_executor::syscalls::SyscallCode;
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

    fn beta_powers_for<P: FpOpField>(air: &Fp2AddSubPolyAir<P>, beta: EF) -> Vec<EF> {
        let max = <Fp2AddSubPolyAir<P> as FullAir<
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
        air: &Fp2AddSubPolyAir<P>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <Fp2AddSubPolyAir<P> as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(
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

    /// Build a real trace from a test ELF for the given field type and syscall code.
    fn sample_trace_for<P: FpOpField>(
        elf: &[u8],
        syscall_code: dt_core_executor::syscalls::SyscallCode,
    ) -> Option<RowMajorMatrix<F>> {
        let program = Program::from(elf).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(syscall_code).is_empty() {
                continue;
            }

            // Build a minimal record containing only fp2_addsub precompile events.
            let mut fp2_shard = ExecutionRecord::new(shard.program.clone());
            fp2_shard.precompile_events = shard.precompile_events.clone();

            let chip = Fp2AddSubAssignChip::<P>::new();
            return Some(
                chip.generate_trace(&fp2_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    /// Run the full constraint satisfaction check for a given field type P.
    fn run_constraint_check<P: FpOpField>(main: RowMajorMatrix<F>) {
        let air = Fp2AddSubPolyAir::<P>::new();
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

        let num_reducer = num_gate_constraints::<P>() + total_lookups.div_ceil(BATCH_SIZE) + 3;
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
    fn test_fp2_addsub_bn254_constraint_check() {
        type P = Bn254BaseField;
        let main = match sample_trace_for::<P>(
            BN254_FP2_ADDSUB_ELF,
            dt_core_executor::syscalls::SyscallCode::BN254_FP2_ADD,
        ) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bn254Fp2AddSub trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<P>(main);
    }

    #[test]
    fn test_fp2_addsub_bls12381_constraint_check() {
        type P = Bls12381BaseField;
        let main = match sample_trace_for::<P>(
            BLS12381_FP2_ADDSUB_ELF,
            dt_core_executor::syscalls::SyscallCode::BLS12381_FP2_ADD,
        ) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bls12381Fp2AddSub trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<P>(main);
    }

    fn random_fp2_addsub_trace<P: FpOpField>(
        log_n: usize,
        _seed: u64,
        elf: &[u8],
        syscall_code: SyscallCode,
    ) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace_for::<P>(elf, syscall_code).expect("sample trace should exist");
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

    fn do_perf_multi_round_sumcheck<P: FpOpField>(elf: &[u8], syscall_code: SyscallCode) {
        let air = Fp2AddSubPolyAir::<P>::new();
        let default_log_n = match P::FIELD_TYPE {
            FieldType::Bn254 => perf_test_defaults::BN254_FP2_ADDSUB_LOG_N,
            FieldType::Bls12381 => perf_test_defaults::BLS12381_FP2_ADDSUB_LOG_N,
        };
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_log_n);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_fp2_addsub_trace::<P>(log_n, seed, elf, syscall_code);
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
        let num_reducer = num_gate_constraints::<P>() + total_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc = <Fp2AddSubPolyAir<P> as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(&air);

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
        do_perf_multi_round_sumcheck::<Bn254BaseField>(
            BN254_FP2_ADDSUB_ELF,
            SyscallCode::BN254_FP2_ADD,
        );
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bls12381() {
        do_perf_multi_round_sumcheck::<Bls12381BaseField>(
            BLS12381_FP2_ADDSUB_ELF,
            SyscallCode::BLS12381_FP2_ADD,
        );
    }
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl<P: FpOpField> Fp2AddSubPolyAir<P> {
    pub const fn num_lookups(&self) -> usize {
        num_lookups::<P>()
    }
    pub const fn num_precomputed(&self) -> usize {
        num_precomputed::<P>()
    }
}
