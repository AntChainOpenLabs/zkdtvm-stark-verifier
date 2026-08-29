//! PolyAir adaptation of EdAddAssignChip.
//!
//! Bridges `EdAddAssignCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (Ed25519BaseField: NB_LIMBS=32, NB_WITNESS_LIMBS=62)
//!
//! Each FieldOpCols/FieldInnerProductCols/FieldDenCols instance generates
//! `field_op_num_interactions<Ed25519BaseField>()` = 32 + 62 = 94 interactions.
//! Each FieldLtCols generates `field_lt_num_interactions<Ed25519BaseField>()` = 3 interactions.
//!
//!   Phase 1 (precompute_lc):
//!     #1  ..  #94:  x3_numerator (FieldInnerProductCols) range checks
//!     #95 .. #188:  y3_numerator (FieldInnerProductCols) range checks
//!     #189 .. #282: x1_mul_y1 (FieldOpCols, Mul)
//!     #283 .. #376: x2_mul_y2 (FieldOpCols, Mul)
//!     #377 .. #470: f (FieldOpCols, Mul)
//!     #471 .. #564: d_mul_f (FieldOpCols, Mul)
//!     #565 .. #658: x3_ins (FieldDenCols)
//!     #659 .. #752: y3_ins (FieldDenCols)
//!     #753 .. #755: x3_range (FieldLtCols: 1 LTU + 2 BitVec)
//!     #756 .. #758: y3_range (FieldLtCols: 1 LTU + 2 BitVec)
//!     #759 .. #822: q_access memory_read (16 × 4 = 64)
//!     #823 .. #886: p_access memory_readwrite (16 × 4 = 64)
//!     #887:         recv(Syscall)
//!
//!   Plus 2 polynomial optimizations for assert_all_eq (x3→p_access, y3→p_access).
//!   Plus 4 operand β-evaluations: x1(β), y1(β), x2(β), y2(β).
//!
//!   Phase 2 (eval): gate constraints
//!   Phase 3 (lookup): send/recv multiplicities
//!
//! ## Boolean handling
//!   - is_real: 1 boolean → direct gate constraint (≤3 threshold)

use std::{marker::PhantomData, ops::Deref};

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    edwards::{ed25519::Ed25519BaseField, EdwardsParameters, NUM_LIMBS, WORDS_CURVE_POINT},
    params::{FieldParameters, NumLimbs},
    EllipticCurve,
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
        field_den::{field_den_lookup, field_den_precompute_lc},
        field_inner_product::{field_inner_product_lookup, field_inner_product_precompute_lc},
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

use super::ed_add::{EdAddAssignCols, NUM_ED_ADD_COLS};

// ============================================================================
// Constants
// ============================================================================

/// Total lookup interactions for EdAddAssignChip.
///
/// = 2 * field_op_num_interactions (x3_numerator, y3_numerator: FieldInnerProductCols)
/// + 4 * field_op_num_interactions (x1_mul_y1, x2_mul_y2, f, d_mul_f: FieldOpCols Mul)
/// + 2 * field_op_num_interactions (x3_ins, y3_ins: FieldDenCols)
/// + 2 * field_lt_num_interactions (x3_range, y3_range)
/// + WORDS_CURVE_POINT * 4          (q_access memory_read)
/// + WORDS_CURVE_POINT * 4          (p_access memory_readwrite)
/// + 1                              (recv Syscall)
const NUM_LOOKUPS: usize = 8 * field_op_num_interactions::<Ed25519BaseField>() +
    2 * field_lt_num_interactions::<Ed25519BaseField>() +
    WORDS_CURVE_POINT * 4 +
    WORDS_CURVE_POINT * 4 +
    1;

/// Precomputed linear combinations:
/// one per lookup + 8 witness_betas + 4 operand betas (x1, y1, x2, y2)
/// + 16 result/carry betas (x3_num, y3_num, x1_mul_y1, x2_mul_y2, f, d_mul_f, x3_ins, y3_ins)
/// + 2 diff_betas.
const NUM_PRECOMPUTED: usize = NUM_LOOKUPS + 30;

// ============================================================================
// Column offsets within EdAddAssignCols<u8>
//
// Layout (#[repr(C)]):
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  p_ptr            ← precompute-only (skipped)
//   [4]  q_ptr            ← precompute-only (skipped)
//   [5 + i*13]            p_access[i] = MemoryWriteCols (13 cols each)
//     +0..+4   prev_value
//     +4..+8   access.value           ← precompute-only (skipped from reserved_poly)
//     +8       access.prev_shard
//     +9       access.prev_clk
//     +10      access.compare_clk
//     +11      access.diff_16bit_limb
//     +12      access.diff_12bit_limb
//   [5 + WCP*13 + i*9]    q_access[i] = MemoryReadCols (9 cols, all needed)
//   Then: x3_numerator(L+L+W), y3_numerator, x1_mul_y1, x2_mul_y2, f, d_mul_f,
//         x3_ins, y3_ins, x3_range(L+2), y3_range(L+2)
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
fn field_op_cols_size() -> usize {
    NUM_LIMBS + NUM_LIMBS + Ed25519BaseField::NB_WITNESS_LIMBS
}

#[inline]
fn col_q_access_base() -> usize {
    COL_P_ACCESS_BASE + WORDS_CURVE_POINT * MEM_WRITE_COLS_SIZE
}

#[inline]
fn col_x3_numerator_base() -> usize {
    col_q_access_base() + WORDS_CURVE_POINT * MEM_READ_COLS_SIZE
}

#[inline]
fn col_y3_numerator_base() -> usize {
    col_x3_numerator_base() + field_op_cols_size()
}

#[inline]
fn col_x1_mul_y1_base() -> usize {
    col_y3_numerator_base() + field_op_cols_size()
}

#[inline]
fn col_x2_mul_y2_base() -> usize {
    col_x1_mul_y1_base() + field_op_cols_size()
}

#[inline]
fn col_f_base() -> usize {
    col_x2_mul_y2_base() + field_op_cols_size()
}

#[inline]
fn col_d_mul_f_base() -> usize {
    col_f_base() + field_op_cols_size()
}

#[inline]
fn col_x3_ins_base() -> usize {
    col_d_mul_f_base() + field_op_cols_size()
}

#[inline]
fn col_y3_ins_base() -> usize {
    col_x3_ins_base() + field_op_cols_size()
}

#[inline]
fn col_x3_range_base() -> usize {
    col_y3_ins_base() + field_op_cols_size()
}

#[inline]
fn col_y3_range_base() -> usize {
    col_x3_range_base() + NUM_LIMBS + 2
}

// ============================================================================
// Reserved-poly row layout (positions in the reserved slice).
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3 + i*5 + 0]            p_access[i].access.prev_shard
//   [3 + i*5 + 1]            p_access[i].access.prev_clk
//   [3 + i*5 + 2]            p_access[i].access.compare_clk
//   [3 + i*5 + 3]            p_access[i].access.diff_16bit_limb
//   [3 + i*5 + 4]            p_access[i].access.diff_12bit_limb
//   [3 + WCP*5 + i*5 + 0]    q_access[i].access.prev_shard
//   [3 + WCP*5 + i*5 + 1]    q_access[i].access.prev_clk
//   [3 + WCP*5 + i*5 + 2]    q_access[i].access.compare_clk
//   [3 + WCP*5 + i*5 + 3]    q_access[i].access.diff_16bit_limb
//   [3 + WCP*5 + i*5 + 4]    q_access[i].access.diff_12bit_limb
//   Then field ops (result/carry for x1_mul_y1, x2_mul_y2, f, d_mul_f are
//   NOT in reserved_poly — consumed as β-evaluations in precompute_lc):
//     x3_ins(result(L))
//     y3_ins(result(L))
//   x3_range(L+2), y3_range(L+2)
//
// NOTE: p_access[i].prev_value and q_access[i].access.value are NOT in
// reserved_poly — they are consumed as β-evaluations (x1_beta, y1_beta,
// x2_beta, y2_beta) computed in precompute_lc and stored in the precomputed slice.
// ============================================================================

const RES_NUM_SCALAR: usize = 3;
const RES_PER_ACCESS: usize = 5; // 5 timestamp fields only (prev_value/access.value removed)

#[inline]
fn res_p_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_ACCESS
}

#[inline]
fn res_q_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + WORDS_CURVE_POINT * RES_PER_ACCESS + i * RES_PER_ACCESS
}

#[inline]
fn res_ops_start() -> usize {
    RES_NUM_SCALAR + WORDS_CURVE_POINT * 2 * RES_PER_ACCESS
}

#[inline]
fn res_x3_ins_base() -> usize {
    res_ops_start()
}

#[inline]
fn res_y3_ins_base() -> usize {
    res_x3_ins_base() + NUM_LIMBS
}

#[inline]
fn res_x3_range_base() -> usize {
    res_y3_ins_base() + NUM_LIMBS
}

#[inline]
fn res_y3_range_base() -> usize {
    res_x3_range_base() + NUM_LIMBS + 2
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for EdAddAssignChip.
#[derive(Clone, Copy)]
pub struct EdAddAssignPolyAir<E: EllipticCurve> {
    _marker: PhantomData<E>,
}

impl<E: EllipticCurve> Default for EdAddAssignPolyAir<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: EllipticCurve> EdAddAssignPolyAir<E> {
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<E: EllipticCurve + EdwardsParameters, AB: FullAirBuilder> FullAir<AB> for EdAddAssignPolyAir<E>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_ED_ADD_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<Ed25519BaseField>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - p_ptr, q_ptr               (precompute-only: memory addresses, syscall LC)
        //   - p_access[i].prev_value     (consumed as x1_beta/y1_beta in precompute_lc)
        //   - p_access[i].access.value   (precompute-only: diff(β) polynomial)
        //   - q_access[i].access.value   (consumed as x2_beta/y2_beta in precompute_lc)
        //   - all FieldOpCols/InnerProduct/Den .witness (precompute-only: witness(β))
        let l = NUM_LIMBS;
        let mut cols: Vec<PairCol> = Vec::new();

        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));

        // p_access[i]: 5 timestamp fields only. Skip prev_value (4 cols) and access.value (4 cols).
        for i in 0..WORDS_CURVE_POINT {
            let base = COL_P_ACCESS_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_12_OFF));
        }

        // q_access[i]: 5 timestamp fields only. Skip access.value (4 cols).
        let q_base_main = col_q_access_base();
        for i in 0..WORDS_CURVE_POINT {
            let base = q_base_main + i * MEM_READ_COLS_SIZE;
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_ACCESS_DIFF_12_OFF));
        }

        // For x3_numerator/y3_numerator AND x1_mul_y1/x2_mul_y2/f/d_mul_f we retain
        // result/carry(β) in precompute_lc, so their trace limbs are not needed in
        // reserved_poly. None of their results flow to FieldLt (only d_mul_f.result
        // feeds x3_ins/y3_ins via β-eval).

        // For x3_ins/y3_ins we keep only result limbs: carry(β) is precomputed,
        // but result limbs are still needed by range and assert_all_eq constraints.
        for base_fn in [col_x3_ins_base, col_y3_ins_base] {
            let base = base_fn();
            for k in 0..l {
                cols.push(PairCol::Main(base + k));
            }
        }

        let x3r = col_x3_range_base();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(x3r + k));
        }
        let y3r = col_y3_range_base();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(y3r + k));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators + polynomial optimizations
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &EdAddAssignCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let p_ptr = local.p_ptr.clone();
        let q_ptr = local.q_ptr.clone();

        let num_words_field_element = <Ed25519BaseField as NumLimbs>::Limbs::USIZE / 4;

        // ── x3_numerator (FieldInnerProductCols) — 94 interactions ──
        field_inner_product_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.x3_numerator.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x3_numerator.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x3_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── y3_numerator (FieldInnerProductCols) — 94 interactions ──
        field_inner_product_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.y3_numerator.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.y3_numerator.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.y3_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x1_mul_y1 (FieldOpCols, Mul) — 94 interactions ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.x1_mul_y1.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x1_mul_y1.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x1_mul_y1.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x2_mul_y2 (FieldOpCols, Mul) — 94 interactions ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.x2_mul_y2.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x2_mul_y2.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x2_mul_y2.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── f (FieldOpCols, Mul: x1_mul_y1 * x2_mul_y2) — 94 interactions ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.f.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.f.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.f.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── d_mul_f (FieldOpCols, Mul: f * d_const) — 94 interactions ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.d_mul_f.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.d_mul_f.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.d_mul_f.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x3_ins (FieldDenCols) — 94 interactions ──
        field_den_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.x3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x3_ins.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── y3_ins (FieldDenCols) — 94 interactions ──
        field_den_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.y3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.y3_ins.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.y3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x3_range (FieldLtCols) — 3 interactions (1 LTU + 2 BitVec) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.x3_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, Ed25519BaseField>(
                builder,
                local.x3_range.lhs_comparison_byte.clone(),
                local.x3_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── y3_range (FieldLtCols) — 3 interactions (1 LTU + 2 BitVec) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.y3_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, Ed25519BaseField>(
                builder,
                local.y3_range.lhs_comparison_byte.clone(),
                local.y3_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── q_access: memory_read (WORDS_CURVE_POINT × 4 interactions) ──
        for i in 0..WORDS_CURVE_POINT {
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

        // ── p_access: memory_readwrite (WORDS_CURVE_POINT × 4 interactions) ──
        // p_access is at clk+1 since p, q could be the same
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..WORDS_CURVE_POINT {
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
        let syscall_id =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(SyscallCode::ED_ADD.syscall_id()));
        builder.retain_precomputed(
            builder.lookup_denominator(syscall_kind, vec![shard, clk, syscall_id, p_ptr, q_ptr]),
        );

        // ── Precompute witness(β) for each field op (used in eval gate constraints) ──
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.x3_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.y3_numerator.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.x1_mul_y1.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.x2_mul_y2.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.f.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.d_mul_f.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.x3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.y3_ins.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── Precompute operand β-evaluations (p_access.prev_value → x1/y1,
        //    q_access.access.value → x2/y2). These replace reading individual
        //    limb columns from reserved_poly during eval. ──
        let x1_coeffs: Vec<AB::VarMaybeExt> = local.p_access[..num_words_field_element]
            .iter()
            .flat_map(|acc| acc.prev_value.0.iter().cloned())
            .collect();
        let y1_coeffs: Vec<AB::VarMaybeExt> = local.p_access[num_words_field_element..]
            .iter()
            .flat_map(|acc| acc.prev_value.0.iter().cloned())
            .collect();
        let x2_coeffs: Vec<AB::VarMaybeExt> = local.q_access[..num_words_field_element]
            .iter()
            .flat_map(|acc| acc.access.value.0.iter().cloned())
            .collect();
        let y2_coeffs: Vec<AB::VarMaybeExt> = local.q_access[num_words_field_element..]
            .iter()
            .flat_map(|acc| acc.access.value.0.iter().cloned())
            .collect();

        let x1_beta = field_op_beta_from_coeffs::<AB>(builder, &x1_coeffs);
        let y1_beta = field_op_beta_from_coeffs::<AB>(builder, &y1_coeffs);
        let x2_beta = field_op_beta_from_coeffs::<AB>(builder, &x2_coeffs);
        let y2_beta = field_op_beta_from_coeffs::<AB>(builder, &y2_coeffs);
        builder.retain_precomputed(x1_beta);
        builder.retain_precomputed(y1_beta);
        builder.retain_precomputed(x2_beta);
        builder.retain_precomputed(y2_beta);

        // ── Precompute result/carry(β) for constraints that no longer read those limbs
        // from reserved_poly. Keep these after witness(β) so the first NUM_LOOKUPS
        // entries remain permutation-compatible.
        //
        // Layout (16 total, indices relative to cached_betas[]):
        //   [0..2]   x3_numerator  (result, carry)
        //   [2..4]   y3_numerator  (result, carry)
        //   [4..6]   x1_mul_y1     (result, carry)
        //   [6..8]   x2_mul_y2     (result, carry)
        //   [8..10]  f             (result, carry)
        //   [10..12] d_mul_f       (result, carry)
        //   [12..14] x3_ins        (result, carry)
        //   [14..16] y3_ins        (result, carry)
        for cols in [&local.x3_numerator, &local.y3_numerator] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &cols.result.0.iter().cloned().collect::<Vec<_>>(),
            ));
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &cols.carry.0.iter().cloned().collect::<Vec<_>>(),
            ));
        }
        for cols in [&local.x1_mul_y1, &local.x2_mul_y2, &local.f, &local.d_mul_f] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &cols.result.0.iter().cloned().collect::<Vec<_>>(),
            ));
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &cols.carry.0.iter().cloned().collect::<Vec<_>>(),
            ));
        }
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.x3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
        ));
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.x3_ins.carry.0.iter().cloned().collect::<Vec<_>>(),
        ));
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.y3_ins.result.0.iter().cloned().collect::<Vec<_>>(),
        ));
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.y3_ins.carry.0.iter().cloned().collect::<Vec<_>>(),
        ));

        // ── Polynomial optimizations for assert_all_eq ──
        // x3_ins.result[i] == p_access[i/4].value()[i%4] for all NUM_LIMBS
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
    }

    // ========================================================================
    // Phase 2: eval — gate constraints (reserved_poly columns only)
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let beta_consts = FieldOpBetaConsts::<AB>::new::<Ed25519BaseField>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[COL_IS_REAL].clone();
        let shard = local[COL_SHARD].clone();
        let clk = local[COL_CLK].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero]);
        let l = NUM_LIMBS;

        // -- Read all precomputed values in one borrow --
        // Order must match precompute_lc retain order:
        //   [0..8]    witness_betas
        //   [8..12]   operand betas (x1, y1, x2, y2)
        //   [12..28]  cached_betas (result/carry for all 8 field ops)
        //   [28..30]  diff_betas (read separately below)
        let (witness_betas, x1_beta, y1_beta, x2_beta, y2_beta, cached_betas) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let start = NUM_LOOKUPS;
            (
                vec![
                    pc[start].clone(),     // x3_numerator
                    pc[start + 1].clone(), // y3_numerator
                    pc[start + 2].clone(), // x1_mul_y1
                    pc[start + 3].clone(), // x2_mul_y2
                    pc[start + 4].clone(), // f
                    pc[start + 5].clone(), // d_mul_f
                    pc[start + 6].clone(), // x3_ins
                    pc[start + 7].clone(), // y3_ins
                ],
                pc[start + 8].clone(),  // x1_beta
                pc[start + 9].clone(),  // y1_beta
                pc[start + 10].clone(), // x2_beta
                pc[start + 11].clone(), // y2_beta
                vec![
                    pc[start + 12].clone(), // x3_numerator.result_beta
                    pc[start + 13].clone(), // x3_numerator.carry_beta
                    pc[start + 14].clone(), // y3_numerator.result_beta
                    pc[start + 15].clone(), // y3_numerator.carry_beta
                    pc[start + 16].clone(), // x1_mul_y1.result_beta
                    pc[start + 17].clone(), // x1_mul_y1.carry_beta
                    pc[start + 18].clone(), // x2_mul_y2.result_beta
                    pc[start + 19].clone(), // x2_mul_y2.carry_beta
                    pc[start + 20].clone(), // f.result_beta
                    pc[start + 21].clone(), // f.carry_beta
                    pc[start + 22].clone(), // d_mul_f.result_beta
                    pc[start + 23].clone(), // d_mul_f.carry_beta
                    pc[start + 24].clone(), // x3_ins.result_beta
                    pc[start + 25].clone(), // x3_ins.carry_beta
                    pc[start + 26].clone(), // y3_ins.result_beta
                    pc[start + 27].clone(), // y3_ins.carry_beta
                ],
            )
        };

        let modulus_beta = beta_consts.modulus_beta.clone();

        // ── x3_numerator: InnerProduct([x1, x2], [y2, y1]) ──
        {
            let vanishing_beta = x1_beta.clone() * y2_beta.clone() +
                x2_beta.clone() * y1_beta.clone() -
                cached_betas[0].clone() -
                cached_betas[1].clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[0].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── y3_numerator: InnerProduct([y1, x1], [y2, x2]) ──
        {
            let vanishing_beta = y1_beta.clone() * y2_beta.clone() +
                x1_beta.clone() * x2_beta.clone() -
                cached_betas[2].clone() -
                cached_betas[3].clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[1].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── x1_mul_y1: Mul(x1, y1) ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            x1_beta,
            y1_beta,
            cached_betas[4].clone(),
            cached_betas[5].clone(),
            witness_betas[2].clone(),
            &beta_consts,
        );

        // ── x2_mul_y2: Mul(x2, y2) ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            x2_beta,
            y2_beta,
            cached_betas[6].clone(),
            cached_betas[7].clone(),
            witness_betas[3].clone(),
            &beta_consts,
        );

        // ── f: Mul(x1_mul_y1.result, x2_mul_y2.result) ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            cached_betas[4].clone(),
            cached_betas[6].clone(),
            cached_betas[8].clone(),
            cached_betas[9].clone(),
            witness_betas[4].clone(),
            &beta_consts,
        );

        // ── d_mul_f: Mul(f.result, d_const) ──
        {
            let d_const_limbs: Vec<AB::VarMaybeExt> =
                E::D.iter().map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x))).collect();
            let d_const_beta = field_op_beta_from_coeffs::<AB>(builder, &d_const_limbs);
            field_op_mul_gate_constraints_all_betas::<AB>(
                builder,
                cached_betas[8].clone(),
                d_const_beta,
                cached_betas[10].clone(),
                cached_betas[11].clone(),
                witness_betas[5].clone(),
                &beta_consts,
            );
        }

        // ── x3_ins: FieldDen(x3_numerator.result, d_mul_f.result, sign=true) ──
        // sign=true: d_mul_f * x3_ins_result + x3_ins_result - x3_num_result - x3_ins_carry *
        // modulus = 0
        {
            let vanishing_beta = cached_betas[10].clone() * cached_betas[12].clone() +
                cached_betas[12].clone() -
                cached_betas[0].clone() -
                cached_betas[13].clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[6].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── y3_ins: FieldDen(y3_numerator.result, d_mul_f.result, sign=false) ──
        // sign=false: d_mul_f * y3_ins_result + y3_num_result - y3_ins_result - y3_ins_carry *
        // modulus = 0
        {
            let vanishing_beta = cached_betas[10].clone() * cached_betas[14].clone() +
                cached_betas[2].clone() -
                cached_betas[14].clone() -
                cached_betas[15].clone() * modulus_beta;
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[7].clone(),
                beta_consts.beta_minus_limb_shift,
            );
        }

        // ── x3_range / y3_range gate constraints ──
        {
            let modulus_limbs: Vec<AB::VarMaybeExt> = Ed25519BaseField::MODULUS
                .iter()
                .map(|&x| AB::VarMaybeExt::from(AB::F::from_canonical_u8(x)))
                .collect();

            let x3r = res_x3_range_base();
            let x3_result: Vec<AB::VarMaybeExt> =
                (0..l).map(|k| local[res_x3_ins_base() + k].clone()).collect();
            let x3_range = FieldLtCols::<AB::VarMaybeExt, Ed25519BaseField> {
                byte_flags: (0..l).map(|k| local[x3r + k].clone()).collect(),
                lhs_comparison_byte: local[x3r + l].clone(),
                rhs_comparison_byte: local[x3r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, Ed25519BaseField>(
                builder,
                &x3_result,
                &modulus_limbs,
                &x3_range,
                is_real.clone(),
            );

            let y3r = res_y3_range_base();
            let y3_result: Vec<AB::VarMaybeExt> =
                (0..l).map(|k| local[res_y3_ins_base() + k].clone()).collect();
            let y3_range = FieldLtCols::<AB::VarMaybeExt, Ed25519BaseField> {
                byte_flags: (0..l).map(|k| local[y3r + k].clone()).collect(),
                lhs_comparison_byte: local[y3r + l].clone(),
                rhs_comparison_byte: local[y3r + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, Ed25519BaseField>(
                builder,
                &y3_result,
                &modulus_limbs,
                &y3_range,
                is_real.clone(),
            );
        }

        // ── assert_all_eq polynomial optimizations ──
        {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();

            let x3_diff_beta = pc[NUM_PRECOMPUTED - 2].clone();
            builder.when(is_real.clone()).assert_zero_ext(x3_diff_beta);

            let y3_diff_beta = pc[NUM_PRECOMPUTED - 1].clone();
            builder.when(is_real.clone()).assert_zero_ext(y3_diff_beta);
        }

        // ── memory timestamp constraints ──
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());

        for i in 0..WORDS_CURVE_POINT {
            let base = res_q_access_base(i);
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

        for i in 0..WORDS_CURVE_POINT {
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

        // x3_numerator (FieldInnerProductCols)
        field_inner_product_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // y3_numerator (FieldInnerProductCols)
        field_inner_product_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // x1_mul_y1 (FieldOpCols, Mul)
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // x2_mul_y2 (FieldOpCols, Mul)
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // f (FieldOpCols, Mul)
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // d_mul_f (FieldOpCols, Mul)
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // x3_ins (FieldDenCols)
        field_den_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // y3_ins (FieldDenCols)
        field_den_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());

        // x3_range (FieldLtCols)
        field_lt_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        // y3_range (FieldLtCols)
        field_lt_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());

        // q_access memory reads
        for _ in 0..WORDS_CURVE_POINT {
            memory_read_lookup(builder, is_real.clone());
        }

        // p_access memory readwrites
        for _ in 0..WORDS_CURVE_POINT {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::ed_add::EdAddAssignChip;
use dt_core_executor::{
    events::{ByteLookupEvent, PrecompileEvent},
    ExecutionRecord, Program,
};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;
use std::borrow::BorrowMut;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, E: EllipticCurve> BaseAir<F> for EdAddAssignPolyAir<E> {
    fn width(&self) -> usize {
        NUM_ED_ADD_COLS
    }
}

impl<F: Field, E: EllipticCurve + EdwardsParameters> MachineAir<F> for EdAddAssignPolyAir<E> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <EdAddAssignChip<E> as MachineAir<F>>::name(&EdAddAssignChip::<E>::new()) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        EdAddAssignChip::<E>::new().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <EdAddAssignChip<E> as MachineAir<F>>::generate_dependencies(
            &EdAddAssignChip::<E>::new(),
            input,
            output,
        );

        // Emit PolyAir-only BitVec lookups for x3_range and y3_range (from field_lt_precompute_lc).
        let events = input.get_precompile_events(SyscallCode::ED_ADD);
        for (_, event) in events {
            let PrecompileEvent::EdAdd(event) = event else { unreachable!() };
            let mut row = [F::zero(); NUM_ED_ADD_COLS];
            let cols: &mut EdAddAssignCols<F> = row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            EdAddAssignChip::<E>::new().event_to_row(event, cols, &mut ignored_blu);
            add_field_lt_bitvec_lookups::<F, Ed25519BaseField>(output, &cols.x3_range);
            add_field_lt_bitvec_lookups::<F, Ed25519BaseField>(output, &cols.y3_range);
        }
    }

    fn padding_row(&self) -> Vec<F> {
        EdAddAssignChip::<E>::new().padding_row()
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <EdAddAssignChip<E> as MachineAir<F>>::included(&EdAddAssignChip::<E>::new(), shard)
    }

    fn local_only(&self) -> bool {
        <EdAddAssignChip<E> as MachineAir<F>>::local_only(&EdAddAssignChip::<E>::new())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::{edwards::ed25519::Ed25519, params::NumWords};
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
    use test_artifacts::ED_ADD_ELF;

    use super::super::ed_add::EdAddAssignChip;
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

    fn beta_powers_for(air: &EdAddAssignPolyAir<Ed25519>, beta: EF) -> Vec<EF> {
        let max = <EdAddAssignPolyAir<Ed25519> as FullAir<
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
        air: &EdAddAssignPolyAir<Ed25519>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <EdAddAssignPolyAir<Ed25519> as FullAir<
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

    /// Build a real trace for EdAddAssign from a test ELF.
    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = Program::from(ED_ADD_ELF).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::ED_ADD).is_empty() {
                continue;
            }

            let mut ec_shard = ExecutionRecord::new(shard.program.clone());
            ec_shard.precompile_events = shard.precompile_events.clone();

            let chip = EdAddAssignChip::<Ed25519>::new();
            return Some(
                chip.generate_trace(&ec_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    /// Run full constraint satisfaction check.
    fn run_constraint_check(main: RowMajorMatrix<F>) {
        let air = EdAddAssignPolyAir::<Ed25519>::new();
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
        let total_lookups = NUM_LOOKUPS;
        let total_precomputed = NUM_PRECOMPUTED;

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

        let nb_limbs = <Ed25519BaseField as FieldParameters>::NB_LIMBS;
        let words_field_element = <Ed25519BaseField as NumWords>::WordsFieldElement::USIZE;
        let field_op_vanishing = 2 * nb_limbs - 1;

        let num_gate_constraints =
            2 * (nb_limbs + 3) + 6 * field_op_vanishing + 2 + words_field_element * 2 * 3 + 2;

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
    fn test_ed_add_polyair_constraint_satisfaction() {
        let main = sample_trace().expect("Should find EdAdd events in ED_ADD_ELF");
        run_constraint_check(main);
    }

    fn random_ed_add_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace().expect("Should find EdAdd events in ED_ADD_ELF");
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

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = EdAddAssignPolyAir::<Ed25519>::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::ED_ADD_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_ed_add_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers_for(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = NUM_LOOKUPS;
        let total_precomputed = NUM_PRECOMPUTED;

        let nb_limbs = <Ed25519BaseField as FieldParameters>::NB_LIMBS;
        let words_field_element = <Ed25519BaseField as NumWords>::WordsFieldElement::USIZE;
        let field_op_vanishing = 2 * nb_limbs - 1;
        let num_gate_constraints =
            2 * (nb_limbs + 3) + 6 * field_op_vanishing + 2 + words_field_element * 2 * 3 + 2;
        let num_reducer = num_gate_constraints + total_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();

        let global = EF::zero();
        let reserved_poly_desc = <EdAddAssignPolyAir<Ed25519> as FullAir<
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
            &beta_powers,
            beta_septix,
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
            &beta_powers,
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
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl<E: EllipticCurve> EdAddAssignPolyAir<E> {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
