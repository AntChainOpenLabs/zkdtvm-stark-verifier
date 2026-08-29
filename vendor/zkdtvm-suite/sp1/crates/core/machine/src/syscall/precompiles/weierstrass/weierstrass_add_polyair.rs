//! PolyAir adaptation of WeierstrassAddAssignChip.
//!
//! Bridges `WeierstrassAddAssignCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (generic over E::BaseField = P)
//!
//!   6 × FieldAddOpCols range checks:
//!     slope_numerator(Sub), slope_denominator(Sub), p_x_plus_q_x(Add),
//!     x3_ins(Sub), p_x_minus_x(Sub), y3_ins(Sub)
//!   4 × FieldOpCols range checks:
//!     inverse_check(Div), slope(Div), slope_squared(Sqr),
//!     slope_times_p_x_minus_x(Mul)
//!   2 × FieldLtCols: x3_range, y3_range (LTU + BitVec)
//!   WordsCurvePoint × 4: q_access memory_read
//!   WordsCurvePoint × 4: p_access memory_readwrite
//!   1: recv(Syscall)
//!   10: witness(β) precomputations
//!
//!   Plus 2 polynomial optimizations for assert_all_eq (x3→p_access, y3→p_access).
//!
//! ## Boolean handling
//!   - is_real: 1 boolean → direct gate constraint
//!   - 6 FieldAddOpCols carries: handled directly inside `field_add_op_{add,sub}_gate_constraints`

use std::{marker::PhantomData, ops::Deref};

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    params::{FieldParameters, Limbs, NumLimbs, NumWords},
    weierstrass::WeierstrassParameters,
    CurveType, EllipticCurve,
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
            field_add_op_variable_gate_constraints_all_betas,
        },
        field_op::{
            field_op_beta_from_coeffs, field_op_gate_constraints, field_op_lookup,
            field_op_mul_gate_constraints_all_betas, field_op_num_interactions,
            field_op_precompute_lc, field_op_precompute_witness_beta, FieldOpBetaConsts,
        },
        range::{
            field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
            field_lt_precompute_lc,
        },
    },
};

use crate::{memory::MemoryAccessCols, operations::field::range::FieldLtCols};

use super::weierstrass_add::{num_weierstrass_add_cols, WeierstrassAddAssignCols};

// ============================================================================
// Constants (computed from type parameters)
// ============================================================================

/// Compute total lookup interactions for WeierstrassAddAssignChip<E>.
///
/// = 6 * field_add_op_num_interactions<P>  (slope_numerator, slope_denominator, p_x_plus_q_x,
///                                          x3_ins, p_x_minus_x, y3_ins)
/// + 4 * field_op_num_interactions<P>      (inverse_check, slope, slope_squared,
///   slope_times_p_x_minus_x)
/// + 2 * field_lt_num_interactions<P>      (x3_range, y3_range)
/// + WordsCurvePoint * 4                   (q_access memory_read)
/// + WordsCurvePoint * 4                   (p_access memory_readwrite)
/// + 1                                     (recv Syscall)
const fn num_lookups<P: FieldParameters + NumWords>() -> usize {
    6 * field_add_op_num_interactions::<P>() +
        4 * field_op_num_interactions::<P>() +
        2 * field_lt_num_interactions::<P>() +
        <P as NumWords>::WordsCurvePoint::USIZE * 4 +
        <P as NumWords>::WordsCurvePoint::USIZE * 4 +
        1
}

/// Precomputed values: one per lookup + ten `witness(beta)` values
/// + four operand betas (p_x, p_y, q_x, q_y)
/// + two `diff(beta)` values for the `assert_all_eq` optimizations
/// + 12 result/carry β-evaluations appended last (4 FieldAddOps result_β only
///   + 4 FieldOps result_β & carry_β for slope_num/slope_denom/p_x_plus_q_x/
///   p_x_minus_x and inverse_check/slope/slope_sq/slope_times respectively).
const fn num_precomputed<P: FieldParameters + NumWords>() -> usize {
    num_lookups::<P>() + 28
}

// ============================================================================
// Column offsets within WeierstrassAddAssignCols<u8, P>
//
// Layout (#[repr(C)]):
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  p_ptr            ← precompute-only (skipped)
//   [4]  q_ptr            ← precompute-only (skipped)
//   [5 + i*13]            p_access[i] = MemoryWriteCols (13 cols each)
//   [5 + WCP*13 + i*9]    q_access[i] = MemoryReadCols (9 cols each)
//   Then: slope_denom, inverse_check, slope_num, slope, slope_sq,
//         p_x_plus_q_x, x3_ins, p_x_minus_x, y3_ins,
//         slope_times_p_x_minus_x, x3_range, y3_range
// ============================================================================

const COL_IS_REAL: usize = 0;
const COL_SHARD: usize = 1;
const COL_CLK: usize = 2;
const COL_P_ACCESS_BASE: usize = 5;
const MEM_WRITE_COLS_SIZE: usize = 13;
const MEM_READ_COLS_SIZE: usize = 9;
const MEM_ACCESS_PREV_SHARD_OFF: usize = 4;
const MEM_ACCESS_PREV_CLK_OFF: usize = 5;
const MEM_ACCESS_COMPARE_CLK_OFF: usize = 6;
const MEM_ACCESS_DIFF_16_OFF: usize = 7;
const MEM_ACCESS_DIFF_12_OFF: usize = 8;

#[inline]
fn col_q_access_base<P: FieldParameters + NumWords>() -> usize {
    COL_P_ACCESS_BASE + <P as NumWords>::WordsCurvePoint::USIZE * MEM_WRITE_COLS_SIZE
}

#[inline]
fn field_op_cols_size<P: FieldParameters>() -> usize {
    P::NB_LIMBS + P::NB_LIMBS + P::NB_WITNESS_LIMBS
}

#[inline]
fn field_add_op_cols_size<P: FieldParameters>() -> usize {
    P::NB_LIMBS + 1 + P::NB_ADD_WITNESS_LIMBS
}

#[inline]
fn col_slope_denom_base<P: FieldParameters + NumWords>() -> usize {
    col_q_access_base::<P>() + <P as NumWords>::WordsCurvePoint::USIZE * MEM_READ_COLS_SIZE
}

#[inline]
fn col_inverse_check_base<P: FieldParameters + NumWords>() -> usize {
    col_slope_denom_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_slope_num_base<P: FieldParameters + NumWords>() -> usize {
    col_inverse_check_base::<P>() + field_op_cols_size::<P>()
}

#[inline]
fn col_slope_base<P: FieldParameters + NumWords>() -> usize {
    col_slope_num_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_slope_sq_base<P: FieldParameters + NumWords>() -> usize {
    col_slope_base::<P>() + field_op_cols_size::<P>()
}

#[inline]
fn col_p_x_plus_q_x_base<P: FieldParameters + NumWords>() -> usize {
    col_slope_sq_base::<P>() + field_op_cols_size::<P>()
}

#[inline]
fn col_x3_ins_base<P: FieldParameters + NumWords>() -> usize {
    col_p_x_plus_q_x_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_p_x_minus_x_base<P: FieldParameters + NumWords>() -> usize {
    col_x3_ins_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_y3_ins_base<P: FieldParameters + NumWords>() -> usize {
    col_p_x_minus_x_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_slope_times_base<P: FieldParameters + NumWords>() -> usize {
    col_y3_ins_base::<P>() + field_add_op_cols_size::<P>()
}

#[inline]
fn col_x3_range_base<P: FieldParameters + NumWords>() -> usize {
    col_slope_times_base::<P>() + field_op_cols_size::<P>()
}

#[inline]
fn col_y3_range_base<P: FieldParameters + NumWords>() -> usize {
    col_x3_range_base::<P>() + P::NB_LIMBS + 2
}

// ============================================================================
// Reserved-poly row layout (positions in the reserved slice).
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3 + i*5]                   p_access[i]: timestamps(5) only
//   [3 + WCP*5 + i*5]           q_access[i]: timestamps(5) only
//   Then field ops (result+carry only, witness skipped):
//     slope_denom(L+1), inverse_check(2L), slope_num(L+1), slope(2L),
//     slope_sq(2L), p_x_plus_q_x(L+1), x3_ins(L+1), p_x_minus_x(L+1),
//     y3_ins(L+1), slope_times(2L), x3_range(L+2), y3_range(L+2)
//
// NOTE: p_access[i].prev_value and q_access[i].access.value are NOT in
// reserved_poly — they are consumed as β-evaluations (p_x(β), p_y(β),
// q_x(β), q_y(β)) computed in precompute_lc and stored in the precomputed slice.
// ============================================================================

const RES_NUM_SCALAR: usize = 3;
const RES_PER_P_ACCESS: usize = 5; // timestamps only (prev_value removed)
const RES_PER_Q_ACCESS: usize = 5; // timestamps only (access.value removed)

#[inline]
fn res_p_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_P_ACCESS
}
#[inline]
fn res_q_access_base<P: FieldParameters + NumWords>(i: usize) -> usize {
    let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
    RES_NUM_SCALAR + wcp * RES_PER_P_ACCESS + i * RES_PER_Q_ACCESS
}
#[inline]
fn res_ops_start<P: FieldParameters + NumWords>() -> usize {
    let wcp = <P as NumWords>::WordsCurvePoint::USIZE;
    RES_NUM_SCALAR + wcp * (RES_PER_P_ACCESS + RES_PER_Q_ACCESS)
}

// The 12 operation blocks in reserved order. Each offset is relative to res_ops_start.
// slope_denom: FieldAddOpCols → L+1
// inverse_check: FieldOpCols → 2L
// slope_num: FieldAddOpCols → L+1
// slope: FieldOpCols → 2L
// slope_sq: FieldOpCols → 2L
// p_x_plus_q_x: FieldAddOpCols → L+1
// x3_ins: FieldAddOpCols → L+1
// p_x_minus_x: FieldAddOpCols → L+1
// y3_ins: FieldAddOpCols → L+1
// slope_times: FieldOpCols → 2L
// x3_range: FieldLtCols → L+2
// y3_range: FieldLtCols → L+2

// Layout after β-eval optimization:
//   slope_denominator.carry (1)        ← FieldAddOp, only carry kept (boolean)
//   slope_numerator.carry   (1)        ← FieldAddOp
//   p_x_plus_q_x.carry      (1)        ← FieldAddOp
//   x3_ins (L+1)                       ← FieldAddOp, result+carry kept (FieldLt input)
//   p_x_minus_x.carry       (1)        ← FieldAddOp
//   y3_ins (L+1)                       ← FieldAddOp, result+carry kept (FieldLt input)
//   x3_range (L+2)
//   y3_range (L+2)
//
// inverse_check / slope / slope_sq / slope_times: entire FieldOpCols dropped
// (result+carry both precomputed; no FieldLt or other limb consumer).
#[inline]
fn res_slope_denom_carry<P: FieldParameters + NumWords>() -> usize {
    res_ops_start::<P>()
}
#[inline]
fn res_slope_num_carry<P: FieldParameters + NumWords>() -> usize {
    res_slope_denom_carry::<P>() + 1
}
#[inline]
fn res_p_x_plus_q_x_carry<P: FieldParameters + NumWords>() -> usize {
    res_slope_num_carry::<P>() + 1
}
#[inline]
fn res_x3_ins_base<P: FieldParameters + NumWords>() -> usize {
    res_p_x_plus_q_x_carry::<P>() + 1
}
#[inline]
fn res_p_x_minus_x_carry<P: FieldParameters + NumWords>() -> usize {
    res_x3_ins_base::<P>() + P::NB_LIMBS + 1
}
#[inline]
fn res_y3_ins_base<P: FieldParameters + NumWords>() -> usize {
    res_p_x_minus_x_carry::<P>() + 1
}
#[inline]
fn res_x3_range_base<P: FieldParameters + NumWords>() -> usize {
    res_y3_ins_base::<P>() + P::NB_LIMBS + 1
}
#[inline]
fn res_y3_range_base<P: FieldParameters + NumWords>() -> usize {
    res_x3_range_base::<P>() + P::NB_LIMBS + 2
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for WeierstrassAddAssignChip.
#[derive(Clone, Copy)]
pub struct WeierstrassAddPolyAir<E: EllipticCurve> {
    _marker: PhantomData<E>,
}

impl<E: EllipticCurve> Default for WeierstrassAddPolyAir<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: EllipticCurve> WeierstrassAddPolyAir<E> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<E: EllipticCurve + WeierstrassParameters, AB: FullAirBuilder> FullAir<AB>
    for WeierstrassAddPolyAir<E>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        num_weierstrass_add_cols::<E::BaseField>()
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<E::BaseField>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - p_ptr, q_ptr               (precompute-only: memory addresses, syscall LC)
        //   - p_access[i].prev_value     (consumed as p_x(β)/p_y(β) in precompute_lc)
        //   - p_access[i].access.value   (precompute-only: diff(β) polynomial)
        //   - q_access[i].access.value   (consumed as q_x(β)/q_y(β) in precompute_lc)
        //   - all FieldOpCols.witness     (precompute-only: witness(β))
        //   - all FieldAddOpCols.witness  (precompute-only: witness(β))
        let wcp = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;
        let l = <E::BaseField as FieldParameters>::NB_LIMBS;

        let mut cols: Vec<PairCol> = Vec::new();

        // Scalars
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));

        // p_access[i]: 5 timestamp fields only. Skip prev_value(4) and access.value(4).
        for i in 0..wcp {
            let base = COL_P_ACCESS_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_12_OFF));
        }

        // q_access[i]: 5 timestamp fields only. Skip access.value(4).
        let q_base_main = col_q_access_base::<E::BaseField>();
        for i in 0..wcp {
            let base = q_base_main + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_12_OFF));
        }

        // FieldAddOps slope_denom/slope_num/p_x_plus_q_x/p_x_minus_x: keep ONLY carry
        // (1 byte each, needed for boolean assertion `carry * (1-carry) = 0`).
        // Their result limbs are precomputed in precompute_lc as result(β).
        cols.push(PairCol::Main(col_slope_denom_base::<E::BaseField>() + l));
        cols.push(PairCol::Main(col_slope_num_base::<E::BaseField>() + l));
        cols.push(PairCol::Main(col_p_x_plus_q_x_base::<E::BaseField>() + l));
        // x3_ins: result(L) + carry(1) — result limbs feed FieldLt and Sub(p_x, x3).
        let x3_ins_b = col_x3_ins_base::<E::BaseField>();
        for k in 0..(l + 1) {
            cols.push(PairCol::Main(x3_ins_b + k));
        }
        cols.push(PairCol::Main(col_p_x_minus_x_base::<E::BaseField>() + l));
        // y3_ins: result(L) + carry(1) — result limbs feed FieldLt.
        let y3_ins_b = col_y3_ins_base::<E::BaseField>();
        for k in 0..(l + 1) {
            cols.push(PairCol::Main(y3_ins_b + k));
        }
        // inverse_check / slope / slope_sq / slope_times: entirely skipped
        // (result_β + carry_β both precomputed; no limb consumer in eval).

        // x3_range, y3_range (full FieldLtCols)
        let x3r = col_x3_range_base::<E::BaseField>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(x3r + k));
        }
        let y3r = col_y3_range_base::<E::BaseField>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(y3r + k));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &WeierstrassAddAssignCols<AB::VarMaybeExt, E::BaseField> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let p_ptr = local.p_ptr.clone();
        let q_ptr = local.q_ptr.clone();

        let num_words_field_element = <E::BaseField as NumLimbs>::Limbs::USIZE / 4;

        // ── slope_numerator (FieldAddOpCols, Sub: q_y - p_y) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.slope_numerator.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── slope_denominator (FieldAddOpCols, Sub: q_x - p_x) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.slope_denominator.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_denominator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── inverse_check (FieldOpCols, Div: 1 / slope_denom) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.inverse_check.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.inverse_check.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.inverse_check.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── slope (FieldOpCols, Div: slope_num / slope_denom) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.slope.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── slope_squared (FieldOpCols, Sqr: slope^2) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.slope_squared.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_squared.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_squared.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── p_x_plus_q_x (FieldAddOpCols, Add: p_x + q_x) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.p_x_plus_q_x.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.p_x_plus_q_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x3_ins (FieldAddOpCols, Sub: slope^2 - (p_x+q_x)) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.x3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── p_x_minus_x (FieldAddOpCols, Sub: p_x - x3) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.p_x_minus_x.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.p_x_minus_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── slope_times_p_x_minus_x (FieldOpCols, Mul) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.slope_times_p_x_minus_x.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_times_p_x_minus_x.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.slope_times_p_x_minus_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── y3_ins (FieldAddOpCols, Sub: slope*(p_x-x3) - p_y) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.y3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.y3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x3_range (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.x3_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, E::BaseField>(
                builder,
                local.x3_range.lhs_comparison_byte.clone(),
                local.x3_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── y3_range (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.y3_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, E::BaseField>(
                builder,
                local.y3_range.lhs_comparison_byte.clone(),
                local.y3_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── q_access: memory_read (WordsCurvePoint × 4 interactions) ──
        for i in 0..<E::BaseField as NumWords>::WordsCurvePoint::USIZE {
            let addr = q_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.q_access[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // ── p_access: memory_readwrite (WordsCurvePoint × 4 interactions) ──
        // We read/write p at clk+1 since p, q could be the same.
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..<E::BaseField as NumWords>::WordsCurvePoint::USIZE {
            let addr = p_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.p_access[i].access,
                &local.p_access[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // ── recv(Syscall) ──
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let syscall_id_felt = match E::CURVE_TYPE {
            CurveType::Secp256k1 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::SECP256K1_ADD.syscall_id(),
            )),
            CurveType::Secp256r1 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::SECP256R1_ADD.syscall_id(),
            )),
            CurveType::Bn254 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::BN254_ADD.syscall_id(),
            )),
            CurveType::Bls12381 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::BLS12381_ADD.syscall_id(),
            )),
            _ => panic!("Unsupported curve"),
        };
        builder.retain_precomputed(
            builder
                .lookup_denominator(syscall_kind, vec![shard, clk, syscall_id_felt, p_ptr, q_ptr]),
        );

        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.slope_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.slope_denominator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.inverse_check.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.slope.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.slope_squared.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.p_x_plus_q_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.x3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.p_x_minus_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.slope_times_p_x_minus_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.y3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── Precompute operand β-evaluations (p_access.prev_value → p_x/p_y,
        //    q_access.access.value → q_x/q_y). These replace reading individual
        //    limb columns from reserved_poly during eval. ──
        let p_x_coeffs: Vec<AB::VarMaybeExt> = local.p_access[..num_words_field_element]
            .iter()
            .flat_map(|acc| acc.prev_value.0.iter().cloned())
            .collect();
        let p_y_coeffs: Vec<AB::VarMaybeExt> = local.p_access[num_words_field_element..]
            .iter()
            .flat_map(|acc| acc.prev_value.0.iter().cloned())
            .collect();
        let q_x_coeffs: Vec<AB::VarMaybeExt> = local.q_access[..num_words_field_element]
            .iter()
            .flat_map(|acc| acc.access.value.0.iter().cloned())
            .collect();
        let q_y_coeffs: Vec<AB::VarMaybeExt> = local.q_access[num_words_field_element..]
            .iter()
            .flat_map(|acc| acc.access.value.0.iter().cloned())
            .collect();

        let p_x_beta = field_op_beta_from_coeffs::<AB>(builder, &p_x_coeffs);
        let p_y_beta = field_op_beta_from_coeffs::<AB>(builder, &p_y_coeffs);
        let q_x_beta = field_op_beta_from_coeffs::<AB>(builder, &q_x_coeffs);
        let q_y_beta = field_op_beta_from_coeffs::<AB>(builder, &q_y_coeffs);
        builder.retain_precomputed(p_x_beta);
        builder.retain_precomputed(p_y_beta);
        builder.retain_precomputed(q_x_beta);
        builder.retain_precomputed(q_y_beta);

        // ── Polynomial optimizations for assert_all_eq ──
        // x3_ins.result[i] == p_access[i/4].value()[i%4] for all NB_LIMBS
        {
            let x_value_limbs: Vec<AB::VarMaybeExt> = local.p_access[..num_words_field_element]
                .iter()
                .flat_map(|acc| acc.access.value.0.iter().cloned())
                .collect();
            let diff_coeffs: Vec<AB::VarMaybeExt> = local
                .x3_ins
                .result
                .0
                .iter()
                .zip(x_value_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();

            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            let diff_beta =
                Polynomial::from_coefficients(&diff_coeffs).eval_with_powers(beta_powers, zero_ext);
            builder.retain_precomputed(diff_beta);
        }

        // y3_ins.result[i] == p_access[num_words_field_element + i/4].value()[i%4]
        {
            let y_value_limbs: Vec<AB::VarMaybeExt> = local.p_access[num_words_field_element..]
                .iter()
                .flat_map(|acc| acc.access.value.0.iter().cloned())
                .collect();
            let diff_coeffs: Vec<AB::VarMaybeExt> = local
                .y3_ins
                .result
                .0
                .iter()
                .zip(y_value_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();

            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            let diff_beta =
                Polynomial::from_coefficients(&diff_coeffs).eval_with_powers(beta_powers, zero_ext);
            builder.retain_precomputed(diff_beta);
        }

        // ── Precompute result(β) for 4 FieldAddOps whose result limbs are not in
        // reserved_poly. Order: slope_num.r, slope_denom.r, p_x_plus_q_x.r,
        // p_x_minus_x.r. carry stays as a limb in reserved_poly (boolean check).
        for cols_result in [
            &local.slope_numerator.result,
            &local.slope_denominator.result,
            &local.p_x_plus_q_x.result,
            &local.p_x_minus_x.result,
        ] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &cols_result.0.iter().cloned().collect::<Vec<_>>(),
            ));
        }

        // ── Precompute result(β) + carry(β) for 4 FieldOps (FieldOpCols).
        // Order: inverse_check, slope, slope_squared, slope_times_p_x_minus_x.
        for op in [
            &local.inverse_check,
            &local.slope,
            &local.slope_squared,
            &local.slope_times_p_x_minus_x,
        ] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &op.result.0.iter().cloned().collect::<Vec<_>>(),
            ));
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &op.carry.0.iter().cloned().collect::<Vec<_>>(),
            ));
        }
    }

    // ========================================================================
    // Phase 2: eval — gate constraints
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let beta_consts = FieldOpBetaConsts::<AB>::new::<E::BaseField>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[COL_IS_REAL].clone();
        let shard = local[COL_SHARD].clone();
        let clk = local[COL_CLK].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
        let wcp = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;
        let l = <E::BaseField as FieldParameters>::NB_LIMBS;

        // -- Read all precomputed values in one borrow --
        // Order matches precompute_lc retain order:
        //   [0..10]   witness_betas (10 ops)
        //   [10..14]  operand betas (p_x, p_y, q_x, q_y)
        //   [14..16]  diff_betas (x3, y3 — read separately below)
        //   [16..20]  4 FieldAddOp result_β (slope_num, slope_denom, p_x_plus_q_x, p_x_minus_x)
        //   [20..28]  4 FieldOp result_β + carry_β (inverse_check, slope, slope_sq, slope_times)
        let (
            witness_betas,
            p_x_beta,
            p_y_beta,
            q_x_beta,
            q_y_beta,
            slope_num_r,
            slope_denom_r,
            p_x_plus_q_x_r,
            p_x_minus_x_r,
            inv_r,
            inv_c,
            slope_r,
            slope_c,
            slope_sq_r,
            slope_sq_c,
            slope_times_r,
            slope_times_c,
        ) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let start = num_lookups::<E::BaseField>();
            (
                vec![
                    pc[start].clone(),     // slope_numerator
                    pc[start + 1].clone(), // slope_denominator
                    pc[start + 2].clone(), // inverse_check
                    pc[start + 3].clone(), // slope
                    pc[start + 4].clone(), // slope_squared
                    pc[start + 5].clone(), // p_x_plus_q_x
                    pc[start + 6].clone(), // x3_ins
                    pc[start + 7].clone(), // p_x_minus_x
                    pc[start + 8].clone(), // slope_times
                    pc[start + 9].clone(), // y3_ins
                ],
                pc[start + 10].clone(), // p_x_beta
                pc[start + 11].clone(), // p_y_beta
                pc[start + 12].clone(), // q_x_beta
                pc[start + 13].clone(), // q_y_beta
                pc[start + 16].clone(), // slope_num.result_β
                pc[start + 17].clone(), // slope_denom.result_β
                pc[start + 18].clone(), // p_x_plus_q_x.result_β
                pc[start + 19].clone(), // p_x_minus_x.result_β
                pc[start + 20].clone(), // inverse_check.result_β
                pc[start + 21].clone(), // inverse_check.carry_β
                pc[start + 22].clone(), // slope.result_β
                pc[start + 23].clone(), // slope.carry_β
                pc[start + 24].clone(), // slope_squared.result_β
                pc[start + 25].clone(), // slope_squared.carry_β
                pc[start + 26].clone(), // slope_times.result_β
                pc[start + 27].clone(), // slope_times.carry_β
            )
        };

        // Reserved-poly reads for FieldAddOps whose ONLY surviving limb is carry.
        let slope_num_carry = local[res_slope_num_carry::<E::BaseField>()].clone();
        let slope_denom_carry = local[res_slope_denom_carry::<E::BaseField>()].clone();
        let p_x_plus_q_x_carry = local[res_p_x_plus_q_x_carry::<E::BaseField>()].clone();
        let p_x_minus_x_carry = local[res_p_x_minus_x_carry::<E::BaseField>()].clone();

        // x3_ins / y3_ins keep result+carry limbs (feed FieldLt).
        let x3_ins_base = res_x3_ins_base::<E::BaseField>();
        let x3_ins_result: Limbs<AB::VarMaybeExt, <E::BaseField as NumLimbs>::Limbs> =
            (0..l).map(|k| local[x3_ins_base + k].clone()).collect();
        let x3_ins_carry = local[x3_ins_base + l].clone();
        let y3_ins_base = res_y3_ins_base::<E::BaseField>();
        let y3_ins_result: Limbs<AB::VarMaybeExt, <E::BaseField as NumLimbs>::Limbs> =
            (0..l).map(|k| local[y3_ins_base + k].clone()).collect();
        let y3_ins_carry = local[y3_ins_base + l].clone();

        // ── slope_numerator: Sub(q_y, p_y) ──
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            q_y_beta.clone(),
            p_y_beta.clone(),
            slope_num_r.clone(),
            slope_num_carry,
            witness_betas[0].clone(),
            zero.clone(),
            &beta_consts,
        );

        // ── slope_denominator: Sub(q_x, p_x) ──
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            q_x_beta.clone(),
            p_x_beta.clone(),
            slope_denom_r.clone(),
            slope_denom_carry,
            witness_betas[1].clone(),
            zero.clone(),
            &beta_consts,
        );

        // ── inverse_check: Div(1, slope_denom) — `inv.r * slope_denom.r = 1 (mod p)` ──
        {
            let mut one_coeffs = vec![
                AB::VarMaybeExt::from(AB::F::zero());
                <E::BaseField as NumLimbs>::Limbs::USIZE
            ];
            one_coeffs[0] = AB::VarMaybeExt::from(AB::F::one());
            let one_beta = field_op_beta_from_coeffs(builder, &one_coeffs);
            let vanishing_beta = inv_r.clone() * slope_denom_r.clone() -
                one_beta -
                inv_c * beta_consts.modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[2].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── slope: Div(slope_num, slope_denom) — `slope.r * slope_denom.r = slope_num.r (mod p)`
        // ──
        {
            let vanishing_beta = slope_r.clone() * slope_denom_r -
                slope_num_r -
                slope_c * beta_consts.modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[3].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── slope_squared: Sqr(slope) — `slope.r * slope.r = slope_sq.r + slope_sq.c * p` ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            slope_r.clone(),
            slope_r.clone(),
            slope_sq_r.clone(),
            slope_sq_c,
            witness_betas[4].clone(),
            &beta_consts,
        );

        // ── p_x_plus_q_x: Add(p_x, q_x) ──
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            p_x_beta.clone(),
            q_x_beta.clone(),
            p_x_plus_q_x_r.clone(),
            p_x_plus_q_x_carry,
            witness_betas[5].clone(),
            one.clone(),
            &beta_consts,
        );

        // ── x3_ins: Sub(slope_squared.result, p_x_plus_q_x.result) ──
        // x3_ins.result still in reserved_poly (feeds FieldLt + p_x_minus_x via β-Horner).
        let x3_ins_result_beta = field_op_beta_from_coeffs(
            builder,
            &x3_ins_result.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            slope_sq_r,
            p_x_plus_q_x_r,
            x3_ins_result_beta.clone(),
            x3_ins_carry,
            witness_betas[6].clone(),
            zero.clone(),
            &beta_consts,
        );

        // ── p_x_minus_x: Sub(p_x, x3) ──
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            p_x_beta.clone(),
            x3_ins_result_beta,
            p_x_minus_x_r.clone(),
            p_x_minus_x_carry,
            witness_betas[7].clone(),
            zero.clone(),
            &beta_consts,
        );

        // ── slope_times_p_x_minus_x: Mul(slope, p_x_minus_x.result) ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            slope_r,
            p_x_minus_x_r,
            slope_times_r.clone(),
            slope_times_c,
            witness_betas[8].clone(),
            &beta_consts,
        );

        // ── y3_ins: Sub(slope_times.result, p_y) ──
        // y3_ins.result still in reserved_poly (feeds FieldLt).
        let y3_ins_result_beta = field_op_beta_from_coeffs(
            builder,
            &y3_ins_result.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_add_op_variable_gate_constraints_all_betas::<AB>(
            builder,
            slope_times_r,
            p_y_beta.clone(),
            y3_ins_result_beta,
            y3_ins_carry,
            witness_betas[9].clone(),
            zero,
            &beta_consts,
        );

        // ── x3_range / y3_range gate constraints ──
        {
            let modulus_limbs: Vec<AB::VarMaybeExt> = <E::BaseField as FieldParameters>::MODULUS
                .iter()
                .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
                .collect();

            let x3r = res_x3_range_base::<E::BaseField>();
            let x3_result_limbs: Vec<AB::VarMaybeExt> = x3_ins_result.0.iter().cloned().collect();
            let x3_range = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                byte_flags: (0..l).map(|k| local[x3r + k].clone()).collect(),
                lhs_comparison_byte: local[x3r + l].clone(),
                rhs_comparison_byte: local[x3r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, E::BaseField>(
                builder,
                &x3_result_limbs,
                &modulus_limbs,
                &x3_range,
                is_real.clone(),
            );

            let y3r = res_y3_range_base::<E::BaseField>();
            let y3_result_limbs: Vec<AB::VarMaybeExt> = y3_ins_result.0.iter().cloned().collect();
            let y3_range = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                byte_flags: (0..l).map(|k| local[y3r + k].clone()).collect(),
                lhs_comparison_byte: local[y3r + l].clone(),
                rhs_comparison_byte: local[y3r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, E::BaseField>(
                builder,
                &y3_result_limbs,
                &modulus_limbs,
                &y3_range,
                is_real.clone(),
            );
        }

        // ── assert_all_eq polynomial optimizations ──
        // diff_betas are at fixed positions [start+14, start+15] in precompute_lc;
        // the 12 result/carry β-evals retained AFTER them shift `total_precomputed - N`
        // away from these slots, so we index explicitly instead.
        {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let start = num_lookups::<E::BaseField>();

            let x3_diff_beta = pc[start + 14].clone();
            builder.when(is_real.clone()).assert_zero_ext(x3_diff_beta);

            let y3_diff_beta = pc[start + 15].clone();
            builder.when(is_real.clone()).assert_zero_ext(y3_diff_beta);
        }

        // ── memory timestamp constraints ──
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());

        for i in 0..wcp {
            let base = res_q_access_base::<E::BaseField>(i);
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

        for i in 0..wcp {
            let base = res_p_access_base(i);
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

        // ── is_real boolean ──
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

        // slope_numerator
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // slope_denominator
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // inverse_check
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // slope
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // slope_squared
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // p_x_plus_q_x
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // x3_ins
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // p_x_minus_x
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // slope_times_p_x_minus_x
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // y3_ins
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());

        // x3_range
        field_lt_lookup::<AB, E::BaseField>(builder, is_real.clone());
        // y3_range
        field_lt_lookup::<AB, E::BaseField>(builder, is_real.clone());

        // q_access memory reads
        for _ in 0..<E::BaseField as NumWords>::WordsCurvePoint::USIZE {
            memory_read_lookup(builder, is_real.clone());
        }

        // p_access memory readwrites
        for _ in 0..<E::BaseField as NumWords>::WordsCurvePoint::USIZE {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::weierstrass_add::WeierstrassAddAssignChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, E: EllipticCurve> BaseAir<F> for WeierstrassAddPolyAir<E> {
    fn width(&self) -> usize {
        num_weierstrass_add_cols::<E::BaseField>()
    }
}

impl<F: Field, E: EllipticCurve + WeierstrassParameters> MachineAir<F>
    for WeierstrassAddPolyAir<E>
{
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <WeierstrassAddAssignChip<E> as MachineAir<F>>::name(&WeierstrassAddAssignChip::<E>::new()) +
            "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use crate::utils::zeroed_f_vec;
        use dt_core_executor::events::{ByteLookupEvent, PrecompileEvent};
        use std::borrow::BorrowMut;

        <WeierstrassAddAssignChip<E> as MachineAir<F>>::generate_dependencies(
            &WeierstrassAddAssignChip::<E>::new(),
            input,
            output,
        );

        let events = match E::CURVE_TYPE {
            CurveType::Secp256k1 => input.get_precompile_events(SyscallCode::SECP256K1_ADD),
            CurveType::Secp256r1 => input.get_precompile_events(SyscallCode::SECP256R1_ADD),
            CurveType::Bn254 => input.get_precompile_events(SyscallCode::BN254_ADD),
            CurveType::Bls12381 => input.get_precompile_events(SyscallCode::BLS12381_ADD),
            _ => panic!("Unsupported curve"),
        };
        let num_cols = num_weierstrass_add_cols::<E::BaseField>();
        for (_, event) in events {
            let event = match event {
                PrecompileEvent::Secp256k1Add(e) |
                PrecompileEvent::Secp256r1Add(e) |
                PrecompileEvent::Bn254Add(e) |
                PrecompileEvent::Bls12381Add(e) => e,
                _ => unreachable!(),
            };
            let mut row = zeroed_f_vec(num_cols);
            let cols: &mut WeierstrassAddAssignCols<F, E::BaseField> =
                row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            WeierstrassAddAssignChip::<E>::populate_row(event, cols, &mut ignored_blu);
            add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &cols.x3_range);
            add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &cols.y3_range);
        }
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WeierstrassAddAssignChip::<E>::new().generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <WeierstrassAddAssignChip<E> as MachineAir<F>>::included(
            &WeierstrassAddAssignChip::<E>::new(),
            shard,
        )
    }

    fn padding_row(&self) -> Vec<F> {
        WeierstrassAddAssignChip::<E>::new().padding_row()
    }

    fn local_only(&self) -> bool {
        <WeierstrassAddAssignChip<E> as MachineAir<F>>::local_only(
            &WeierstrassAddAssignChip::<E>::new(),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::weierstrass::{
        bls12_381::Bls12381Parameters, bn254::Bn254Parameters, secp256k1::Secp256k1Parameters,
    };
    type Secp256k1 = dt_curves::weierstrass::SwCurve<Secp256k1Parameters>;
    type Bn254 = dt_curves::weierstrass::SwCurve<Bn254Parameters>;
    type Bls12381 = dt_curves::weierstrass::SwCurve<Bls12381Parameters>;
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
    use std::ops::Deref;
    use test_artifacts::{BLS12381_MUL_ELF, BN254_MUL_ELF, SECP256K1_MUL_ELF};

    use super::super::weierstrass_add::WeierstrassAddAssignChip;
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

    fn beta_powers_for<E: EllipticCurve + WeierstrassParameters>(
        air: &WeierstrassAddPolyAir<E>,
        beta: EF,
    ) -> Vec<EF> {
        let max = <WeierstrassAddPolyAir<E> as FullAir<
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

    fn reserved_poly_matrix<E: EllipticCurve + WeierstrassParameters>(
        air: &WeierstrassAddPolyAir<E>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <WeierstrassAddPolyAir<E> as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(air);
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

    /// Build a real trace for WeierstrassAddAssign from a test ELF.
    fn sample_trace_for<E: EllipticCurve + WeierstrassParameters>(
        elf: &[u8],
    ) -> Option<RowMajorMatrix<F>> {
        let syscall_code = match E::CURVE_TYPE {
            CurveType::Secp256k1 => SyscallCode::SECP256K1_ADD,
            CurveType::Secp256r1 => SyscallCode::SECP256R1_ADD,
            CurveType::Bn254 => SyscallCode::BN254_ADD,
            CurveType::Bls12381 => SyscallCode::BLS12381_ADD,
            _ => panic!("Unsupported curve"),
        };

        let program = Program::from(elf).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(syscall_code).is_empty() {
                continue;
            }

            let mut ec_shard = ExecutionRecord::new(shard.program.clone());
            ec_shard.precompile_events = shard.precompile_events.clone();

            let chip = WeierstrassAddAssignChip::<E>::new();
            return Some(
                chip.generate_trace(&ec_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    /// Run full constraint satisfaction check for a given curve type E.
    fn run_constraint_check<E: EllipticCurve + WeierstrassParameters>(main: RowMajorMatrix<F>) {
        let air = WeierstrassAddPolyAir::<E>::new();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let bp = beta_powers_for(&air, beta);
        let bs = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = num_lookups::<E::BaseField>();
        let total_precomputed = num_precomputed::<E::BaseField>();

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
            total_precomputed,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &bp,
            BATCH_SIZE,
            total_lookups,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        // Conservative upper bound for gate constraints.
        let nb_limbs = <E::BaseField as FieldParameters>::NB_LIMBS;
        // FieldOpCols: vanishing polynomial → 2*NB_LIMBS - 1 constraints each
        // FieldAddOpCols: similar but smaller witness
        // FieldLtCols: NB_LIMBS + 3 constraints each
        let field_op_vanishing = 2 * nb_limbs - 1;
        let field_add_op_vanishing = nb_limbs; // approximate
        let num_gate_constraints = 6 * (field_add_op_vanishing + 1)    // 6 FieldAddOpCols (+ carry boolean)
            + 4 * field_op_vanishing            // 4 FieldOpCols
            + 2 * (nb_limbs + 3)                // 2 FieldLtCols
            + 2                                  // 2 assert_zero_ext for poly opts
            + <E::BaseField as NumWords>::WordsCurvePoint::USIZE * 2 * 3 // memory timestamp
            + 1; // is_real boolean
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

    #[test]
    fn test_weierstrass_add_secp256k1_constraint_check() {
        // secp256k1_mul internally calls add, so we get add events from the mul ELF
        type E = Secp256k1;
        let main = match sample_trace_for::<E>(SECP256K1_MUL_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No Secp256k1Add trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main);
    }

    #[test]
    fn test_weierstrass_add_bn254_constraint_check() {
        type E = Bn254;
        let main = match sample_trace_for::<E>(BN254_MUL_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bn254Add trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main);
    }

    #[test]
    fn test_weierstrass_add_bls12381_constraint_check() {
        type E = Bls12381;
        let main = match sample_trace_for::<E>(BLS12381_MUL_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bls12381Add trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main);
    }

    /// Generate a random WeierstrassAdd trace for performance testing.
    fn random_weierstrass_add_trace<E: EllipticCurve + WeierstrassParameters>(
        log_n: usize,
        _seed: u64,
        elf: &[u8],
    ) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace_for::<E>(elf).expect("sample trace should exist");
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

    /// Shared multi-round sumcheck benchmark logic for WeierstrassAddPolyAir.
    fn do_perf_multi_round_sumcheck_add<E: EllipticCurve + WeierstrassParameters>(
        air: WeierstrassAddPolyAir<E>,
        main: RowMajorMatrix<F>,
    ) {
        let height = main.height();
        assert!(height >= 2);
        let log_n = height.trailing_zeros() as usize;
        std::println!("perf_multi_round: log_n={}, h={}", log_n, height);

        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let bp = beta_powers_for(&air, beta);
        let bs = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = num_lookups::<E::BaseField>();
        let total_precomputed = num_precomputed::<E::BaseField>();

        // Conservative upper bound for constraint reducer (matches constraint check test).
        let nb_limbs = <E::BaseField as FieldParameters>::NB_LIMBS;
        let field_op_vanishing = 2 * nb_limbs - 1;
        let field_add_op_vanishing = nb_limbs;
        let num_gate_constraints = 6 * (field_add_op_vanishing + 1)    // 6 FieldAddOpCols (+ carry boolean)
            + 4 * field_op_vanishing                                    // 4 FieldOpCols
            + 2 * (nb_limbs + 3)                                        // 2 FieldLtCols
            + 2                                                          // 2 assert_zero_ext for poly opts
            + <E::BaseField as NumWords>::WordsCurvePoint::USIZE * 2 * 3 // memory timestamp
            + 1; // is_real boolean
        let num_reducer = num_gate_constraints + total_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(reducer_seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc = <WeierstrassAddPolyAir<E> as FullAir<
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
            total_precomputed,
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
            total_lookups,
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

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_secp256k1() {
        type E = Secp256k1;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SECP256K1_ADD_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_weierstrass_add_trace::<E>(log_n, seed, SECP256K1_MUL_ELF);
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassAddPolyAir::<E>::new();
        do_perf_multi_round_sumcheck_add(air, main);
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bn254() {
        type E = Bn254;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::BN254_ADD_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_weierstrass_add_trace::<E>(log_n, seed, BN254_MUL_ELF);
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassAddPolyAir::<E>::new();
        do_perf_multi_round_sumcheck_add(air, main);
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bls12381() {
        type E = Bls12381;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::BLS12381_ADD_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_weierstrass_add_trace::<E>(log_n, seed, BLS12381_MUL_ELF);
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassAddPolyAir::<E>::new();
        do_perf_multi_round_sumcheck_add(air, main);
    }
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl<E: EllipticCurve> WeierstrassAddPolyAir<E> {
    pub const fn num_lookups(&self) -> usize {
        num_lookups::<E::BaseField>()
    }
    pub const fn num_precomputed(&self) -> usize {
        num_precomputed::<E::BaseField>()
    }
}
