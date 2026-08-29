//! PolyAir adaptation of WeierstrassDecompressChip.
//!
//! Bridges `WeierstrassDecompressCols` constraints to PolyAir's `FullAir` four-phase model.
//! Supports both `LeastSignificantBit` and `Lexicographic` sign choice rules.
//!
//! ## Interaction Summary (generic over E::BaseField = P)
//!
//! ### Base interactions (both modes):
//!   1 × FieldLtCols: range_x (x < modulus)
//!   2 × FieldOpCols: x_2 (Sqr), x_3 (Mul)
//!   1 × FieldInnerProductCols: ax_plus_b (= field_op interactions)
//!   1 × FieldAddOpCols: x_3_plus_b_plus_ax (Add)
//!   1 × FieldSqrtCols: y (= field_op + field_lt + NB_LIMBS/2 + 1 AND)
//!   1 × FieldAddOpCols: neg_y (Sub: 0 - sqrt_y)
//!   1 × FieldLtCols: neg_y_range_check
//!   WordsFieldElement × 4: x_access memory_read
//!   WordsFieldElement × 4: y_access memory_readwrite
//!   1: recv(Syscall)
//!
//! ### Additional for Lexicographic mode:
//!   2 × FieldLtCols: comparison_lt_cols (called twice with different mults)
//!
//! ### Extra precomputed polynomials:
//!   6 × witness(beta): x_2, x_3, ax_plus_b, x_3_plus_b_plus_ax, neg_y, y.multiplication
//!   2 × diff(beta): sqrt_y - y_access.value, neg_y - y_access.value
//!
//! ## Boolean handling
//!   - is_real, sign_bit: direct gate constraints
//!   - lsb: direct gate constraints in this file because FieldSqrt is inlined in `eval`
//!   For Lexicographic mode:
//!   - is_y_eq_sqrt_y_result, when_sqrt_y_res_is_lt, when_neg_y_res_is_lt: direct gate

use std::{marker::PhantomData, mem::size_of, ops::Deref};

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    params::{FieldParameters, NumLimbs, NumWords},
    weierstrass::WeierstrassParameters,
    CurveType, EllipticCurve,
};
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol, Polynomial},
    InteractionKind, Word,
};
use num::BigUint;
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
        field_add_op::{
            field_add_op_lookup, field_add_op_num_interactions, field_add_op_precompute_lc,
            field_add_op_sub_gate_constraints,
        },
        field_inner_product::{
            field_inner_product_lookup, field_inner_product_num_interactions,
            field_inner_product_precompute_lc,
        },
        field_op::{
            field_op_beta_from_coeffs, field_op_gate_constraints, field_op_lookup,
            field_op_mul_gate_constraints_all_betas, field_op_num_interactions,
            field_op_precompute_lc, field_op_precompute_witness_beta, FieldOpBetaConsts,
        },
        field_sqrt::{field_sqrt_lookup, field_sqrt_num_interactions, field_sqrt_precompute_lc},
        range::{
            field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
            field_lt_precompute_lc, FieldLtCols,
        },
    },
};

use super::weierstrass_decompress::{
    num_weierstrass_decompress_cols, LexicographicChoiceCols, SignChoiceRule,
    WeierstrassDecompressCols,
};

// ============================================================================
// Constants (computed from type parameters)
// ============================================================================

/// Compute total lookup interactions for WeierstrassDecompressChip<E> base (LSB mode).
const fn num_base_lookups<P: FieldParameters + NumWords>() -> usize {
    field_lt_num_interactions::<P>()                   // range_x
    + 2 * field_op_num_interactions::<P>()             // x_2, x_3
    + field_inner_product_num_interactions::<P>()       // ax_plus_b
    + field_add_op_num_interactions::<P>()              // x_3_plus_b_plus_ax
    + field_sqrt_num_interactions::<P>()                // y
    + field_add_op_num_interactions::<P>()              // neg_y
    + field_lt_num_interactions::<P>()                  // neg_y_range_check
    + <P as NumWords>::WordsFieldElement::USIZE * 4    // x_access memory_read
    + <P as NumWords>::WordsFieldElement::USIZE * 4    // y_access memory_readwrite
    + 1 // recv(Syscall)
}

/// Additional lookups for Lexicographic mode: 2 × FieldLtCols.
const fn num_lex_lookups<P: FieldParameters + NumWords>() -> usize {
    2 * field_lt_num_interactions::<P>()
}

/// Precomputed linear combinations for LSB mode.
/// Layout: +6 witness(β) +2 ax_plus_b β +6 inner-op result/carry β +2 diff(β).
const fn num_lsb_precomputed<P: FieldParameters + NumWords>() -> usize {
    num_base_lookups::<P>() + 16
}

/// Precomputed linear combinations for Lexicographic mode.
const fn num_lex_precomputed<P: FieldParameters + NumWords>() -> usize {
    num_base_lookups::<P>() + num_lex_lookups::<P>() + 16
}

// ============================================================================
// Column layout constants
// ============================================================================

// Main trace scalar offsets
const COL_IS_REAL: usize = 0;
const COL_SHARD: usize = 1;
const COL_CLK: usize = 2;
// col 3 = ptr (precompute-only, skipped from reserved_poly)
const COL_SIGN_BIT: usize = 4;
const COL_X_ACCESS_BASE: usize = 5;

const MEM_READ_COLS_SIZE: usize = 9;
const MEM_READWRITE_COLS_SIZE: usize = 13;
const MEM_READWRITE_PREV_VALUE_SIZE: usize = 4;
const MEM_ACCESS_PREV_SHARD_OFF: usize = 4;
const MEM_ACCESS_PREV_CLK_OFF: usize = 5;
const MEM_ACCESS_COMPARE_CLK_OFF: usize = 6;
const MEM_ACCESS_DIFF_16_OFF: usize = 7;
const MEM_ACCESS_DIFF_12_OFF: usize = 8;

// Reserved-poly scalar indices (ptr is skipped, so sign_bit shifts to 3)
const RES_IS_REAL: usize = 0;
const RES_SHARD: usize = 1;
const RES_CLK: usize = 2;
const RES_SIGN_BIT: usize = 3;
const RES_NUM_SCALAR: usize = 4;
const RES_PER_X_ACCESS: usize = 9; // access.value(4) + timestamps(5) — value needed by field_lt
const RES_PER_Y_ACCESS: usize = 5; // timestamps only — value consumed in precompute_lc (diff betas)

// Main-trace offset helpers (generic over P)
#[inline]
fn wfe<P: NumWords>() -> usize {
    <P as NumWords>::WordsFieldElement::USIZE
}

#[inline]
fn col_y_access_base<P: NumWords>() -> usize {
    COL_X_ACCESS_BASE + wfe::<P>() * MEM_READ_COLS_SIZE
}

#[inline]
fn col_range_x_base<P: NumWords>() -> usize {
    col_y_access_base::<P>() + wfe::<P>() * MEM_READWRITE_COLS_SIZE
}

#[inline]
fn col_neg_y_rc_base<P: FieldParameters + NumWords>() -> usize {
    col_range_x_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn fop_size<P: FieldParameters>() -> usize {
    P::NB_LIMBS + P::NB_LIMBS + P::NB_WITNESS_LIMBS
}

#[inline]
fn fadd_size<P: FieldParameters>() -> usize {
    P::NB_LIMBS + 1 + P::NB_ADD_WITNESS_LIMBS
}

#[inline]
fn col_x_2_base<P: FieldParameters + NumWords>() -> usize {
    col_neg_y_rc_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn col_x_3_base<P: FieldParameters + NumWords>() -> usize {
    col_x_2_base::<P>() + fop_size::<P>()
}

#[inline]
fn col_axb_base<P: FieldParameters + NumWords>() -> usize {
    col_x_3_base::<P>() + fop_size::<P>()
}

#[inline]
fn col_x3bax_base<P: FieldParameters + NumWords>() -> usize {
    col_axb_base::<P>() + fop_size::<P>()
}

#[inline]
fn col_y_base<P: FieldParameters + NumWords>() -> usize {
    col_x3bax_base::<P>() + fadd_size::<P>()
}

#[inline]
fn col_y_range_base<P: FieldParameters + NumWords>() -> usize {
    col_y_base::<P>() + fop_size::<P>()
}

#[inline]
fn col_y_lsb<P: FieldParameters + NumWords>() -> usize {
    col_y_range_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn col_neg_y_base<P: FieldParameters + NumWords>() -> usize {
    col_y_lsb::<P>() + 1
}

// Reserved-poly offset helpers (generic over P)
#[inline]
fn res_x_access_base<P: NumWords>(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_X_ACCESS
}

#[inline]
fn res_y_access_base<P: NumWords>(i: usize) -> usize {
    RES_NUM_SCALAR + wfe::<P>() * RES_PER_X_ACCESS + i * RES_PER_Y_ACCESS
}

#[inline]
fn res_ops_start<P: NumWords>() -> usize {
    RES_NUM_SCALAR + wfe::<P>() * RES_PER_X_ACCESS + wfe::<P>() * RES_PER_Y_ACCESS
}

#[inline]
fn res_range_x_base<P: NumWords>() -> usize {
    res_ops_start::<P>()
}

#[inline]
fn res_neg_y_rc_base<P: FieldParameters + NumWords>() -> usize {
    res_range_x_base::<P>() + P::NB_LIMBS + 2
}

// Reserved-poly layout after β-eval optimization:
//   range_x          (L+2)         FieldLtCols
//   neg_y_rc         (L+2)         FieldLtCols
//   x3bax_carry      (1)           FieldAddOp carry (boolean)
//   y_mul_result     (L)           FieldOp result (= sqrt_y, feeds FieldLt + Sub)
//   y_range          (L+2)         FieldLtCols
//   y_lsb            (1)
//   neg_y            (L+1)         FieldAddOp result+carry (feeds FieldLt)
//
// Dropped from reserved_poly: x_2 (2L), x_3 (2L), x3bax.result (L),
// y_mul.carry (L). All precomputed as β-evals.
#[inline]
fn res_x3bax_carry<P: FieldParameters + NumWords>() -> usize {
    res_neg_y_rc_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn res_y_mul_base<P: FieldParameters + NumWords>() -> usize {
    res_x3bax_carry::<P>() + 1
}

#[inline]
fn res_y_range_base<P: FieldParameters + NumWords>() -> usize {
    res_y_mul_base::<P>() + P::NB_LIMBS
}

#[inline]
fn res_y_lsb<P: FieldParameters + NumWords>() -> usize {
    res_y_range_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn res_neg_y_base<P: FieldParameters + NumWords>() -> usize {
    res_y_lsb::<P>() + 1
}

#[inline]
fn res_base_width<P: FieldParameters + NumWords>() -> usize {
    res_neg_y_base::<P>() + P::NB_LIMBS + 1
}

// Lexicographic reserved-poly offsets (appended after base)
#[inline]
fn res_comparison_lt_base<P: FieldParameters + NumWords>() -> usize {
    res_base_width::<P>()
}

#[inline]
fn res_is_y_eq<P: FieldParameters + NumWords>() -> usize {
    res_comparison_lt_base::<P>() + P::NB_LIMBS + 2
}

#[inline]
fn res_when_sqrt_lt<P: FieldParameters + NumWords>() -> usize {
    res_is_y_eq::<P>() + 1
}

#[inline]
fn res_when_neg_lt<P: FieldParameters + NumWords>() -> usize {
    res_when_sqrt_lt::<P>() + 1
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for WeierstrassDecompressChip.
#[derive(Clone, Copy)]
pub struct WeierstrassDecompressPolyAir<E: EllipticCurve> {
    sign_rule: SignChoiceRule,
    _marker: PhantomData<E>,
}

impl<E: EllipticCurve> WeierstrassDecompressPolyAir<E> {
    pub fn new(sign_rule: SignChoiceRule) -> Self {
        Self { sign_rule, _marker: PhantomData }
    }

    fn is_lexicographic(&self) -> bool {
        matches!(self.sign_rule, SignChoiceRule::Lexicographic)
    }
}

impl<E: EllipticCurve + WeierstrassParameters, AB: FullAirBuilder> FullAir<AB>
    for WeierstrassDecompressPolyAir<E>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        let is_lex = self.is_lexicographic();
        num_weierstrass_decompress_cols::<E::BaseField>() +
            if is_lex { size_of::<LexicographicChoiceCols<u8, E::BaseField>>() } else { 0 }
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<E::BaseField>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - ptr                          (precompute-only: memory address, syscall LC)
        //   - y_access[i].access.value     (consumed as y_value(β) in precompute_lc for diff(β)
        //     linkage)
        //   - y_access[i].prev_value       (consumed in precompute_lc for memory_readwrite)
        //   - ax_plus_b (all limbs)        (result/carry(β) retained in precompute_lc)
        //   - all FieldOpCols.witness      (precompute-only: witness(β))
        //   - all FieldAddOpCols.witness   (precompute-only: witness(β))
        let l = <E::BaseField as FieldParameters>::NB_LIMBS;
        let w = wfe::<E::BaseField>();
        let mut cols: Vec<PairCol> = Vec::new();

        // Scalars: is_real, shard, clk, sign_bit (skip ptr)
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_SIGN_BIT));

        // x_access: full MemoryReadCols (9 cols each)
        for i in 0..w {
            let base = COL_X_ACCESS_BASE + i * MEM_READ_COLS_SIZE;
            for k in 0..MEM_READ_COLS_SIZE {
                cols.push(PairCol::Main(base + k));
            }
        }

        // y_access: timestamps(5) only. access.value consumed in precompute_lc (diff betas), not
        // eval.
        let ya_base = col_y_access_base::<E::BaseField>();
        for i in 0..w {
            let base = ya_base + i * MEM_READWRITE_COLS_SIZE;
            cols.push(PairCol::Main(
                base + MEM_READWRITE_PREV_VALUE_SIZE + MEM_ACCESS_PREV_SHARD_OFF,
            ));
            cols.push(PairCol::Main(
                base + MEM_READWRITE_PREV_VALUE_SIZE + MEM_ACCESS_PREV_CLK_OFF,
            ));
            cols.push(PairCol::Main(
                base + MEM_READWRITE_PREV_VALUE_SIZE + MEM_ACCESS_COMPARE_CLK_OFF,
            ));
            cols.push(PairCol::Main(base + MEM_READWRITE_PREV_VALUE_SIZE + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + MEM_READWRITE_PREV_VALUE_SIZE + MEM_ACCESS_DIFF_12_OFF));
        }

        // range_x: FieldLtCols (l+2 cols)
        let rx = col_range_x_base::<E::BaseField>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(rx + k));
        }

        // neg_y_range_check: FieldLtCols (l+2 cols)
        let nyrc = col_neg_y_rc_base::<E::BaseField>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(nyrc + k));
        }

        // x_2 / x_3: dropped entirely. result_β + carry_β precomputed in precompute_lc.
        // ax_plus_b: dropped entirely (result/carry(β) precomputed).

        // x_3_plus_b_plus_ax: keep only carry(1) for boolean check. result_β precomputed.
        cols.push(PairCol::Main(col_x3bax_base::<E::BaseField>() + l));

        // y.multiplication: keep only result(l) (= sqrt_y, feeds FieldLt + Sub).
        // carry_β precomputed; witness skipped.
        let ym = col_y_base::<E::BaseField>();
        for k in 0..l {
            cols.push(PairCol::Main(ym + k));
        }

        // y.range: FieldLtCols (l+2 cols)
        let yr = col_y_range_base::<E::BaseField>();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(yr + k));
        }

        // y.lsb
        cols.push(PairCol::Main(col_y_lsb::<E::BaseField>()));

        // neg_y: result(l) + carry(1), skip witness
        let ny = col_neg_y_base::<E::BaseField>();
        for k in 0..(l + 1) {
            cols.push(PairCol::Main(ny + k));
        }

        // Lexicographic mode extras
        let is_lex = self.is_lexicographic();
        if is_lex {
            let weierstrass_cols = num_weierstrass_decompress_cols::<E::BaseField>();
            // comparison_lt_cols: FieldLtCols (l+2 cols)
            for k in 0..(l + 2) {
                cols.push(PairCol::Main(weierstrass_cols + k));
            }
            // is_y_eq_sqrt_y_result, when_sqrt_y_res_is_lt, when_neg_y_res_is_lt
            let lex_scalars_base = weierstrass_cols + l + 2;
            cols.push(PairCol::Main(lex_scalars_base));
            cols.push(PairCol::Main(lex_scalars_base + 1));
            cols.push(PairCol::Main(lex_scalars_base + 2));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let weierstrass_cols = num_weierstrass_decompress_cols::<E::BaseField>();
        // Save raw pointer to main trace then drop the borrow so builder can be mutably borrowed.
        let main_ptr = {
            let main = builder.main();
            main.as_ptr() as *const AB::VarMaybeExt
        };
        let local: &WeierstrassDecompressCols<AB::VarMaybeExt, E::BaseField> =
            unsafe { core::mem::transmute(main_ptr) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let ptr = local.ptr.clone();

        let num_limbs = <E::BaseField as NumLimbs>::Limbs::USIZE;

        // ── range_x (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.range_x.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, E::BaseField>(
                builder,
                local.range_x.lhs_comparison_byte.clone(),
                local.range_x.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── x_2 (FieldOpCols, Sqr) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.x_2.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x_2.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x_2.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x_3 (FieldOpCols, Mul) ──
        field_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.x_3.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x_3.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.x_3.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── ax_plus_b (FieldInnerProductCols) ──
        field_inner_product_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.ax_plus_b.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.ax_plus_b.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.ax_plus_b.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x_3_plus_b_plus_ax (FieldAddOpCols, Add) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.x_3_plus_b_plus_ax.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.x_3_plus_b_plus_ax.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── y (FieldSqrtCols) ──
        // a_limbs = x_3_plus_b_plus_ax.result = y² (the actual multiplication input before the
        // hack-overwrite)
        let y_a_limbs: Vec<AB::VarMaybeExt> =
            local.x_3_plus_b_plus_ax.result.0.iter().cloned().collect();
        field_sqrt_precompute_lc::<AB, E::BaseField>(builder, &local.y, &y_a_limbs);

        // ── neg_y (FieldAddOpCols, Sub) ──
        field_add_op_precompute_lc::<AB, E::BaseField>(
            builder,
            &local.neg_y.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.neg_y.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── neg_y_range_check (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> =
                local.neg_y_range_check.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, E::BaseField>(
                builder,
                local.neg_y_range_check.lhs_comparison_byte.clone(),
                local.neg_y_range_check.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── x_access: memory_read (WordsFieldElement × 4 interactions) ──
        for i in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            let addr = ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i as u32) * 4 + num_limbs as u32,
                ));
            memory_read_precompute_lc(
                builder,
                &local.x_access[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // ── y_access: memory_readwrite (WordsFieldElement × 4 interactions) ──
        for i in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            let addr =
                ptr.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32((i as u32) * 4));
            memory_readwrite_precompute_lc(
                builder,
                &local.y_access[i].access,
                &local.y_access[i].prev_value,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // ── recv(Syscall) ──
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let syscall_id_felt = match E::CURVE_TYPE {
            CurveType::Secp256k1 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::SECP256K1_DECOMPRESS.syscall_id(),
            )),
            CurveType::Secp256r1 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::SECP256R1_DECOMPRESS.syscall_id(),
            )),
            CurveType::Bls12381 => AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                SyscallCode::BLS12381_DECOMPRESS.syscall_id(),
            )),
            _ => panic!("Unsupported curve"),
        };
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![shard, clk, syscall_id_felt, ptr, local.sign_bit.clone()],
        ));

        // ── Lexicographic mode: additional comparison lookups ──
        let is_lex = self.is_lexicographic();
        if is_lex {
            let choice_cols: &LexicographicChoiceCols<AB::VarMaybeExt, E::BaseField> = unsafe {
                &*(main_ptr.add(weierstrass_cols)
                    as *const LexicographicChoiceCols<AB::VarMaybeExt, E::BaseField>)
            };

            // comparison_lt_cols used for sqrt_y < neg_y (mult: when_sqrt_y_res_is_lt)
            {
                let flags: Vec<AB::VarMaybeExt> =
                    choice_cols.comparison_lt_cols.byte_flags.0.iter().cloned().collect();
                field_lt_precompute_lc::<AB, E::BaseField>(
                    builder,
                    choice_cols.comparison_lt_cols.lhs_comparison_byte.clone(),
                    choice_cols.comparison_lt_cols.rhs_comparison_byte.clone(),
                    &flags,
                );
            }

            // comparison_lt_cols used for neg_y < sqrt_y (mult: when_neg_y_res_is_lt)
            // Same columns, same denominator, but different multiplicity in lookup
            {
                let flags: Vec<AB::VarMaybeExt> =
                    choice_cols.comparison_lt_cols.byte_flags.0.iter().cloned().collect();
                field_lt_precompute_lc::<AB, E::BaseField>(
                    builder,
                    choice_cols.comparison_lt_cols.lhs_comparison_byte.clone(),
                    choice_cols.comparison_lt_cols.rhs_comparison_byte.clone(),
                    &flags,
                );
            }
        }

        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.x_2.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.x_3.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.ax_plus_b.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.x_3_plus_b_plus_ax.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.neg_y.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, E::BaseField>(
            builder,
            &local.y.multiplication.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.ax_plus_b.result.0.iter().cloned().collect::<Vec<_>>(),
        ));
        builder.retain_precomputed(field_op_beta_from_coeffs(
            builder,
            &local.ax_plus_b.carry.0.iter().cloned().collect::<Vec<_>>(),
        ));

        // ── Precompute β-evals for inner ops whose trace limbs are not in
        //    reserved_poly. Order matches eval read positions [start+8..start+14]:
        //      x_2.result_β, x_2.carry_β,
        //      x_3.result_β, x_3.carry_β,
        //      x_3_plus_b_plus_ax.result_β,
        //      y.multiplication.carry_β.
        for limbs in [
            &local.x_2.result.0[..],
            &local.x_2.carry.0[..],
            &local.x_3.result.0[..],
            &local.x_3.carry.0[..],
            &local.x_3_plus_b_plus_ax.result.0[..],
            &local.y.multiplication.carry.0[..],
        ] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &limbs.iter().cloned().collect::<Vec<_>>(),
            ));
        }

        // ── Polynomial optimizations for y linkage (y_access matches sqrt_y or neg_y) ──
        // We use a single polynomial check: for LSB mode, the linkage depends on sign_bit vs lsb.
        // For simplicity, we compute diff(β) for sqrt_y vs y_access and neg_y vs y_access.
        {
            let sqrt_y_limbs: Vec<AB::VarMaybeExt> =
                local.y.multiplication.result.0.iter().cloned().collect();
            let y_value_limbs: Vec<AB::VarMaybeExt> =
                local.y_access.iter().flat_map(|acc| acc.access.value.0.iter().cloned()).collect();

            // diff_sqrt = sqrt_y - y_access_value
            let diff_sqrt_coeffs: Vec<AB::VarMaybeExt> = sqrt_y_limbs
                .iter()
                .zip(y_value_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();
            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            let diff_sqrt_beta = Polynomial::from_coefficients(&diff_sqrt_coeffs)
                .eval_with_powers(beta_powers, zero_ext.clone());
            builder.retain_precomputed(diff_sqrt_beta);

            // diff_neg = neg_y - y_access_value
            let neg_y_limbs: Vec<AB::VarMaybeExt> = local.neg_y.result.0.iter().cloned().collect();
            let diff_neg_coeffs: Vec<AB::VarMaybeExt> = neg_y_limbs
                .iter()
                .zip(y_value_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();
            let beta_powers = builder.beta_powers();
            let diff_neg_beta = Polynomial::from_coefficients(&diff_neg_coeffs)
                .eval_with_powers(beta_powers, zero_ext);
            builder.retain_precomputed(diff_neg_beta);
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

        let is_real = local[RES_IS_REAL].clone();
        let shard = local[RES_SHARD].clone();
        let clk = local[RES_CLK].clone();
        let sign_bit = local[RES_SIGN_BIT].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
        let l = <E::BaseField as FieldParameters>::NB_LIMBS;
        let w = wfe::<E::BaseField>();
        let is_lex = self.is_lexicographic();

        // Precompute layout (start = base or base+lex lookups):
        //   [0..6]   witness_betas (x_2, x_3, ax_plus_b, x3bax, neg_y, y.mul)
        //   [6..8]   ax_plus_b result_β + carry_β
        //   [8..10]  x_2 result_β + carry_β
        //   [10..12] x_3 result_β + carry_β
        //   [12]     x_3_plus_b_plus_ax.result_β
        //   [13]     y.multiplication.carry_β
        //   [tail..] 2 diff_betas (accessed via total_precomputed-2/-1)
        let (witness_betas, axb_betas, x2_r, x2_c, x3_r, x3_c, x3bax_r, y_mul_c) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let start = if is_lex {
                num_base_lookups::<E::BaseField>() + num_lex_lookups::<E::BaseField>()
            } else {
                num_base_lookups::<E::BaseField>()
            };
            (
                vec![
                    pc[start].clone(),     // x_2
                    pc[start + 1].clone(), // x_3
                    pc[start + 2].clone(), // ax_plus_b
                    pc[start + 3].clone(), // x_3_plus_b_plus_ax
                    pc[start + 4].clone(), // neg_y
                    pc[start + 5].clone(), // y.multiplication
                ],
                (
                    pc[start + 6].clone(), // ax_plus_b.result_beta
                    pc[start + 7].clone(), // ax_plus_b.carry_beta
                ),
                pc[start + 8].clone(),  // x_2.result_β
                pc[start + 9].clone(),  // x_2.carry_β
                pc[start + 10].clone(), // x_3.result_β
                pc[start + 11].clone(), // x_3.carry_β
                pc[start + 12].clone(), // x_3_plus_b_plus_ax.result_β
                pc[start + 13].clone(), // y.multiplication.carry_β
            )
        };

        // -- Extract x_limbs from x_access value columns --
        let x_limbs: Vec<AB::VarMaybeExt> = (0..w)
            .flat_map(|i| {
                let base = res_x_access_base::<E::BaseField>(i);
                (0..4).map(move |k| local[base + k].clone())
            })
            .collect();

        let modulus_limbs: Vec<AB::VarMaybeExt> = <E::BaseField as FieldParameters>::MODULUS
            .iter()
            .map(|&byte| AB::VarMaybeExt::from(AB::F::from_canonical_u8(byte)))
            .collect();
        let modulus_beta = beta_consts.modulus_beta.clone();
        let x_beta = field_op_beta_from_coeffs(builder, &x_limbs);

        // ── range_x: FieldLtCols gate constraints (x < modulus) ──
        {
            let rx = res_range_x_base::<E::BaseField>();
            let range_x = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                byte_flags: (0..l).map(|k| local[rx + k].clone()).collect(),
                lhs_comparison_byte: local[rx + l].clone(),
                rhs_comparison_byte: local[rx + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, E::BaseField>(
                builder,
                &x_limbs,
                &modulus_limbs,
                &range_x,
                is_real.clone(),
            );
        }

        // ── x_2: Sqr(x, x) — all βs precomputed, no trace limbs ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            x_beta.clone(),
            x_beta.clone(),
            x2_r.clone(),
            x2_c,
            witness_betas[0].clone(),
            &beta_consts,
        );

        // ── x_3: Mul(x_2.result, x) — all βs precomputed ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            x2_r,
            x_beta.clone(),
            x3_r.clone(),
            x3_c,
            witness_betas[1].clone(),
            &beta_consts,
        );

        // ── ax_plus_b: InnerProduct([a_const, b_const], [x, 1]) ──
        {
            let a_const_limbs: Vec<AB::VarMaybeExt> =
                <E::BaseField as FieldParameters>::to_limbs_field::<AB::F, _>(&E::a_int())
                    .0
                    .iter()
                    .map(|&f| AB::VarMaybeExt::from(f))
                    .collect();
            let b_const_limbs: Vec<AB::VarMaybeExt> =
                <E::BaseField as FieldParameters>::to_limbs_field::<AB::F, _>(&E::b_int())
                    .0
                    .iter()
                    .map(|&f| AB::VarMaybeExt::from(f))
                    .collect();
            let one_limbs: Vec<AB::VarMaybeExt> =
                <E::BaseField as FieldParameters>::to_limbs_field::<AB::F, _>(&BigUint::from(1u32))
                    .0
                    .iter()
                    .map(|&f| AB::VarMaybeExt::from(f))
                    .collect();
            let a_beta = field_op_beta_from_coeffs(builder, &a_const_limbs);
            let b_beta = field_op_beta_from_coeffs(builder, &b_const_limbs);
            let one_beta = field_op_beta_from_coeffs(builder, &one_limbs);
            let vanishing_beta = a_beta * x_beta.clone() + b_beta * one_beta -
                axb_betas.0.clone() -
                axb_betas.1.clone() * modulus_beta.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[2].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
        }

        // ── x_3_plus_b_plus_ax: Add(x_3.result, ax_plus_b.result) — all βs precomputed.
        // Only the FieldAddOp carry stays in reserved_poly (boolean check).
        let x3bax_carry = local[res_x3bax_carry::<E::BaseField>()].clone();
        {
            let vanishing_beta = x3_r + axb_betas.0.clone() -
                x3bax_r.clone() -
                modulus_beta.clone() * x3bax_carry.clone();
            field_op_gate_constraints::<AB>(
                builder,
                vanishing_beta,
                witness_betas[3].clone(),
                beta_consts.beta_minus_limb_shift.clone(),
            );
            builder.assert_zero(x3bax_carry.clone() * (one.clone() - x3bax_carry));
        }

        // ── y: FieldSqrt gate constraints (inlined to use precomputed witness_beta) ──
        // y.multiplication.result limbs (= sqrt_y) stay in reserved_poly (feed FieldLt + Sub).
        // y.multiplication.carry_β is precomputed.
        let sqrt_y_limbs: Vec<AB::VarMaybeExt> = {
            let ym = res_y_mul_base::<E::BaseField>();
            (0..l).map(|k| local[ym + k].clone()).collect()
        };
        {
            // 1. Mul verification: sqrt * sqrt = input (x_3_plus_b_plus_ax.result)
            // sqrt_y_β is computed via Horner (sqrt limbs needed elsewhere); result side
            // uses precomputed x3bax_r as the "result" β; carry uses precomputed y_mul_c.
            let sqrt_y_beta = field_op_beta_from_coeffs(builder, &sqrt_y_limbs);
            field_op_mul_gate_constraints_all_betas::<AB>(
                builder,
                sqrt_y_beta.clone(),
                sqrt_y_beta,
                x3bax_r,
                y_mul_c,
                witness_betas[5].clone(),
                &beta_consts,
            );

            // 2. Range check: sqrt < modulus
            let yr = res_y_range_base::<E::BaseField>();
            let y_range = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                byte_flags: (0..l).map(|k| local[yr + k].clone()).collect(),
                lhs_comparison_byte: local[yr + l].clone(),
                rhs_comparison_byte: local[yr + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, E::BaseField>(
                builder,
                &sqrt_y_limbs,
                &modulus_limbs,
                &y_range,
                is_real.clone(),
            );

            // 3. assert_bool(lsb)
            let lsb = local[res_y_lsb::<E::BaseField>()].clone();
            builder.assert_zero(lsb.clone() * (one.clone() - lsb.clone()));
            // 4. when(is_real).assert_eq(lsb, is_odd) — is_odd = lsb, so vacuous
            builder.assert_zero(is_real.clone() * (lsb.clone() - lsb));
        }

        // ── neg_y: Sub(0, sqrt_y) ──
        // neg_y.result limbs stay in reserved_poly (feed FieldLt + lex comparison).
        let neg_y_op = {
            use dt_curves::params::Limbs;
            let base = res_neg_y_base::<E::BaseField>();
            let result: Limbs<AB::VarMaybeExt, <E::BaseField as NumLimbs>::Limbs> =
                (0..l).map(|k| local[base + k].clone().clone()).collect();
            let carry = local[base + l].clone();
            let witness: Limbs<AB::VarMaybeExt, <E::BaseField as NumLimbs>::AddWitness> =
                std::iter::repeat_with(|| zero.clone())
                    .take(<E::BaseField as FieldParameters>::NB_ADD_WITNESS_LIMBS)
                    .collect();
            crate::operations::field::field_add_op::FieldAddOpCols { result, carry, witness }
        };
        {
            let zero_limbs: Vec<AB::VarMaybeExt> = vec![zero; l];
            field_add_op_sub_gate_constraints::<AB, E::BaseField>(
                builder,
                &zero_limbs,
                &sqrt_y_limbs,
                &neg_y_op,
                witness_betas[4].clone(),
                &beta_consts,
            );
        }

        // ── neg_y_range_check: FieldLtCols gate constraints (neg_y < modulus) ──
        {
            let neg_y_limbs: Vec<AB::VarMaybeExt> = neg_y_op.result.0.iter().cloned().collect();
            let nrc = res_neg_y_rc_base::<E::BaseField>();
            let neg_y_range = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                byte_flags: (0..l).map(|k| local[nrc + k].clone()).collect(),
                lhs_comparison_byte: local[nrc + l].clone(),
                rhs_comparison_byte: local[nrc + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, E::BaseField>(
                builder,
                &neg_y_limbs,
                &modulus_limbs,
                &neg_y_range,
                is_real.clone(),
            );
        }

        // ── y linkage constraints ──
        let total_precomputed = if is_lex {
            num_lex_precomputed::<E::BaseField>()
        } else {
            num_lsb_precomputed::<E::BaseField>()
        };

        let precomputed = builder.precomputed();
        let pc_binding = precomputed.row_slice(0);
        let pc: &[AB::VarExt] = pc_binding.deref();
        let diff_sqrt_beta = pc[total_precomputed - 2].clone();
        let diff_neg_beta = pc[total_precomputed - 1].clone();

        let lsb = local[res_y_lsb::<E::BaseField>()].clone();

        match self.sign_rule {
            SignChoiceRule::LeastSignificantBit => {
                // Linear selectors matching the original when_ne constraints.
                // Lowers `is_real · sel · diff` from degree 3 (XOR form) to degree 2.
                let when_eq = lsb.clone() + sign_bit.clone() - one.clone();
                let when_neq = lsb - sign_bit.clone();

                builder.when(is_real.clone()).when(when_eq).assert_zero_ext(diff_sqrt_beta);
                builder.when(is_real.clone()).when(when_neq).assert_zero_ext(diff_neg_beta);
            }
            SignChoiceRule::Lexicographic => {
                let is_y_eq = local[res_is_y_eq::<E::BaseField>()].clone();
                let when_sqrt_lt = local[res_when_sqrt_lt::<E::BaseField>()].clone();
                let when_neg_lt = local[res_when_neg_lt::<E::BaseField>()].clone();

                builder.assert_zero(is_y_eq.clone() * (one.clone() - is_y_eq.clone()));
                builder.assert_zero(when_sqrt_lt.clone() * (one.clone() - when_sqrt_lt.clone()));
                builder.assert_zero(when_neg_lt.clone() * (one.clone() - when_neg_lt.clone()));

                builder.assert_zero(
                    is_real.clone() * (when_sqrt_lt.clone() + when_neg_lt.clone() - one.clone()),
                );

                builder.when(is_real.clone()).when(is_y_eq.clone()).assert_zero_ext(diff_sqrt_beta);
                builder
                    .when(is_real.clone())
                    .when(one.clone() - is_y_eq.clone())
                    .assert_zero_ext(diff_neg_beta);

                builder.assert_zero((one.clone() - is_real.clone()) * when_sqrt_lt.clone());
                builder.assert_zero((one.clone() - is_real.clone()) * when_neg_lt.clone());

                builder.assert_zero(
                    is_real.clone() * sign_bit.clone() * (is_y_eq.clone() - when_neg_lt.clone()),
                );
                builder.assert_zero(
                    is_real.clone() *
                        (one.clone() - sign_bit.clone()) *
                        (is_y_eq - when_sqrt_lt.clone()),
                );

                // comparison_lt_cols gate constraints
                let neg_y_limbs: Vec<AB::VarMaybeExt> = neg_y_op.result.0.iter().cloned().collect();

                let cmp_base = res_comparison_lt_base::<E::BaseField>();
                let comparison_lt = FieldLtCols::<AB::VarMaybeExt, E::BaseField> {
                    byte_flags: (0..l).map(|k| local[cmp_base + k].clone()).collect(),
                    lhs_comparison_byte: local[cmp_base + l].clone(),
                    rhs_comparison_byte: local[cmp_base + l + 1].clone(),
                };

                field_lt_gate_constraints::<AB, E::BaseField>(
                    builder,
                    &sqrt_y_limbs,
                    &neg_y_limbs,
                    &comparison_lt,
                    when_sqrt_lt,
                );

                field_lt_gate_constraints::<AB, E::BaseField>(
                    builder,
                    &neg_y_limbs,
                    &sqrt_y_limbs,
                    &comparison_lt,
                    when_neg_lt,
                );
            }
        }

        // ── memory timestamp constraints ──
        for i in 0..w {
            let base = res_x_access_base::<E::BaseField>(i);
            let acc = MemoryAccessCols::<AB::VarMaybeExt> {
                value: zero_word.clone(),
                prev_shard: local[base + 4].clone(),
                prev_clk: local[base + 5].clone(),
                compare_clk: local[base + 6].clone(),
                diff_16bit_limb: local[base + 7].clone(),
                diff_12bit_limb: local[base + 8].clone(),
            };
            memory_timestamp_gate_constraints(
                builder,
                &acc,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }
        for i in 0..w {
            let base = res_y_access_base::<E::BaseField>(i);
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

        // ── Boolean constraints ──
        builder.assert_zero(is_real.clone() * (one.clone() - is_real));
        builder.assert_zero(sign_bit.clone() * (one - sign_bit));
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        // Must match precompute_lc order exactly!
        field_lt_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_inner_product_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_sqrt_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_add_op_lookup::<AB, E::BaseField>(builder, is_real.clone());
        field_lt_lookup::<AB, E::BaseField>(builder, is_real.clone());

        for _ in 0..wfe::<E::BaseField>() {
            memory_read_lookup(builder, is_real.clone());
        }
        for _ in 0..wfe::<E::BaseField>() {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        builder.recv(is_real);

        if self.is_lexicographic() {
            let when_sqrt_lt = local[res_when_sqrt_lt::<E::BaseField>()].clone();
            let when_neg_lt = local[res_when_neg_lt::<E::BaseField>()].clone();
            field_lt_lookup::<AB, E::BaseField>(builder, when_sqrt_lt);
            field_lt_lookup::<AB, E::BaseField>(builder, when_neg_lt);
        }
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::weierstrass_decompress::WeierstrassDecompressChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field, E: EllipticCurve> BaseAir<F> for WeierstrassDecompressPolyAir<E> {
    fn width(&self) -> usize {
        let is_lex = self.is_lexicographic();
        num_weierstrass_decompress_cols::<E::BaseField>() +
            if is_lex {
                std::mem::size_of::<LexicographicChoiceCols<u8, E::BaseField>>()
            } else {
                0
            }
    }
}

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, E: EllipticCurve + WeierstrassParameters> MachineAir<F>
    for WeierstrassDecompressPolyAir<E>
{
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        let c = WeierstrassDecompressChip::<E>::new(self.sign_rule);
        <WeierstrassDecompressChip<E> as MachineAir<F>>::name(&c) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        WeierstrassDecompressChip::<E>::new(self.sign_rule).generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use dt_core_executor::events::{ByteLookupEvent, PrecompileEvent};
        use num::BigUint;
        use std::borrow::BorrowMut;

        <WeierstrassDecompressChip<E> as MachineAir<F>>::generate_dependencies(
            &WeierstrassDecompressChip::<E>::new(self.sign_rule),
            input,
            output,
        );

        let events = match E::CURVE_TYPE {
            CurveType::Secp256k1 => input.get_precompile_events(SyscallCode::SECP256K1_DECOMPRESS),
            CurveType::Secp256r1 => input.get_precompile_events(SyscallCode::SECP256R1_DECOMPRESS),
            CurveType::Bls12381 => input.get_precompile_events(SyscallCode::BLS12381_DECOMPRESS),
            _ => panic!("Unsupported curve"),
        };

        let weierstrass_width = num_weierstrass_decompress_cols::<E::BaseField>();
        let lex_extra = if self.is_lexicographic() {
            std::mem::size_of::<LexicographicChoiceCols<u8, E::BaseField>>()
        } else {
            0
        };
        let width = weierstrass_width + lex_extra;
        let modulus = E::BaseField::modulus();

        for (_, event) in events {
            let event = match event {
                PrecompileEvent::Secp256k1Decompress(e) |
                PrecompileEvent::Secp256r1Decompress(e) |
                PrecompileEvent::Bls12381Decompress(e) => e,
                _ => unreachable!(),
            };

            let x = BigUint::from_bytes_le(&event.x_bytes);
            let mut row = crate::utils::zeroed_f_vec(width);
            let cols: &mut super::weierstrass_decompress::WeierstrassDecompressCols<
                F,
                E::BaseField,
            > = row[0..weierstrass_width].borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            WeierstrassDecompressChip::<E>::populate_field_ops(&mut ignored_blu, cols, x);

            // range_x, y.range (via field_sqrt), neg_y_range_check
            add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &cols.range_x);
            add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &cols.y.range);
            add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &cols.neg_y_range_check);

            // Lexicographic: comparison_lt_cols — exactly one of when_sqrt_lt / when_neg_lt
            // fires per real row, so we emit one set of BitVec lookups per chunk.
            if self.is_lexicographic() {
                let decompressed_y = BigUint::from_bytes_le(&event.decompressed_y_bytes);
                let neg_y = &modulus - &decompressed_y;

                let lt_size = std::mem::size_of::<
                    super::weierstrass_decompress::LexicographicChoiceCols<u8, E::BaseField>,
                >();
                let mut lt_row = crate::utils::zeroed_f_vec(lt_size);
                let lt_cols: &mut LexicographicChoiceCols<F, E::BaseField> =
                    lt_row.as_mut_slice().borrow_mut();

                if event.sign_bit {
                    lt_cols.comparison_lt_cols.populate(&mut ignored_blu, &neg_y, &decompressed_y);
                } else {
                    lt_cols.comparison_lt_cols.populate(&mut ignored_blu, &decompressed_y, &neg_y);
                }
                add_field_lt_bitvec_lookups::<F, E::BaseField>(output, &lt_cols.comparison_lt_cols);
            }
        }
    }

    fn included(&self, shard: &Self::Record) -> bool {
        let c = WeierstrassDecompressChip::<E>::new(self.sign_rule);
        <WeierstrassDecompressChip<E> as MachineAir<F>>::included(&c, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        WeierstrassDecompressChip::<E>::new(self.sign_rule).padding_row()
    }

    fn local_only(&self) -> bool {
        let c = WeierstrassDecompressChip::<E>::new(self.sign_rule);
        <WeierstrassDecompressChip<E> as MachineAir<F>>::local_only(&c)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use amcl::{bls381::bls381::basic::key_pair_generate_g2, rand::RAND};
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::weierstrass::{
        bls12_381::Bls12381Parameters, secp256k1::Secp256k1Parameters,
        secp256r1::Secp256r1Parameters,
    };
    type Secp256k1 = dt_curves::weierstrass::SwCurve<Secp256k1Parameters>;
    type Secp256r1 = dt_curves::weierstrass::SwCurve<Secp256r1Parameters>;
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
    use elliptic_curve::sec1::ToEncodedPoint;
    use p3_baby_bear::BabyBear;
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::thread_rng;
    use std::ops::Deref;
    use test_artifacts::{
        BLS12381_DECOMPRESS_ELF, SECP256K1_DECOMPRESS_ELF, SECP256R1_DECOMPRESS_ELF,
    };

    use super::super::weierstrass_decompress::WeierstrassDecompressChip;
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
        air: &WeierstrassDecompressPolyAir<E>,
        beta: EF,
    ) -> Vec<EF> {
        let max = <WeierstrassDecompressPolyAir<E> as FullAir<
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
        air: &WeierstrassDecompressPolyAir<E>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <WeierstrassDecompressPolyAir<E> as FullAir<
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

    fn sample_compressed_input<E: EllipticCurve + WeierstrassParameters>() -> Vec<u8> {
        match E::CURVE_TYPE {
            CurveType::Secp256k1 => {
                let secret_key = k256::SecretKey::random(&mut thread_rng());
                let public_key = secret_key.public_key();
                public_key.to_sec1_bytes().to_vec()
            }
            CurveType::Secp256r1 => {
                let secret_key = p256::SecretKey::random(&mut thread_rng());
                let public_key = secret_key.public_key();
                public_key.to_encoded_point(true).as_bytes().to_vec()
            }
            CurveType::Bls12381 => {
                let mut rand = RAND::new();
                let seed = (0..100).map(|i| i as u8).collect::<Vec<_>>();
                rand.seed(seed.len(), &seed);
                let (_, compressed) = key_pair_generate_g2(&mut rand);
                compressed.to_vec()
            }
            _ => panic!("Unsupported curve"),
        }
    }

    fn sample_trace_for<E: EllipticCurve + WeierstrassParameters>(
        elf: &[u8],
        sign_rule: SignChoiceRule,
    ) -> Option<RowMajorMatrix<F>> {
        let syscall_code = match E::CURVE_TYPE {
            CurveType::Secp256k1 => SyscallCode::SECP256K1_DECOMPRESS,
            CurveType::Secp256r1 => SyscallCode::SECP256R1_DECOMPRESS,
            CurveType::Bls12381 => SyscallCode::BLS12381_DECOMPRESS,
            _ => panic!("Unsupported curve"),
        };

        let program = Program::from(elf).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        let compressed = sample_compressed_input::<E>();
        runtime.write_stdin_slice(&compressed);
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(syscall_code).is_empty() {
                continue;
            }

            let mut ec_shard = ExecutionRecord::new(shard.program.clone());
            ec_shard.precompile_events = shard.precompile_events.clone();

            let chip = WeierstrassDecompressChip::<E>::new(sign_rule);
            return Some(
                chip.generate_trace(&ec_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    fn run_constraint_check<E: EllipticCurve + WeierstrassParameters>(
        main: RowMajorMatrix<F>,
        sign_rule: SignChoiceRule,
    ) {
        let air = WeierstrassDecompressPolyAir::<E>::new(sign_rule);
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
        let is_lex = air.is_lexicographic();
        let total_lookups = if is_lex {
            num_base_lookups::<E::BaseField>() + num_lex_lookups::<E::BaseField>()
        } else {
            num_base_lookups::<E::BaseField>()
        };
        let total_precomputed = if is_lex {
            num_lex_precomputed::<E::BaseField>()
        } else {
            num_lsb_precomputed::<E::BaseField>()
        };

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

        // Exact gate-constraint count.
        let nb_limbs = <E::BaseField as FieldParameters>::NB_LIMBS;
        let nb_witness = <E::BaseField as FieldParameters>::NB_WITNESS_LIMBS;
        let nb_add_witness = <E::BaseField as FieldParameters>::NB_ADD_WITNESS_LIMBS;
        let words_field_element = <E::BaseField as NumWords>::WordsFieldElement::USIZE;

        // FieldLtCols helper emits: NB_LIMBS + 3 constraints.
        let field_lt_gate_count = nb_limbs + 3;
        // field_op_gate_constraints emits: witness_len + 1 constraints.
        let field_op_gate_count = nb_witness + 1;
        // FieldAddOp fixed add/sub: field_op_gate_constraints(add_witness) + carry boolean.
        let field_add_gate_count = (nb_add_witness + 1) + 1;

        // Base path (both LSB and lexicographic):
        // - range_x, neg_y_range_check, and FieldSqrt range: 3 × FieldLtCols
        // - x_2, x_3, ax_plus_b, and FieldSqrt mul-check: 4 × FieldOp-like checks
        // - x_3_plus_b_plus_ax, neg_y: 2 × FieldAddOp checks
        // FieldSqrtCols local constraints: assert_bool(lsb), when(is_real) assert_eq(lsb, sign)
        // - y linkage: 2 assert_zero_ext (LSB or Lex branch both enforce two)
        // - memory timestamp constraints: 2 * WordsFieldElement accesses, 3 each
        // - is_real/sign_bit booleans: 2
        let mut num_gate_constraints = 3 * field_lt_gate_count +
            4 * field_op_gate_count +
            2 * field_add_gate_count +
            2 +
            2 +
            words_field_element * 2 * 3 +
            2;

        // Lexicographic-only extras:
        // - 3 flag booleans + 1 disjointness
        // - 2 conditional-zero constraints for !is_real
        // - 2 sign-bit linkage constraints
        // - 2 additional FieldLtCols (sqrt<neg and neg<sqrt)
        if is_lex {
            num_gate_constraints += 8 + 2 * field_lt_gate_count;
        }

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
    fn test_weierstrass_decompress_secp256k1_constraint_check() {
        type E = Secp256k1;
        let sign_rule = SignChoiceRule::LeastSignificantBit;
        let main = match sample_trace_for::<E>(SECP256K1_DECOMPRESS_ELF, sign_rule) {
            Some(trace) => trace,
            None => {
                eprintln!("No Secp256k1Decompress trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main, sign_rule);
    }

    #[test]
    fn test_weierstrass_decompress_bls12381_constraint_check() {
        type E = Bls12381;
        let sign_rule = SignChoiceRule::Lexicographic;
        let main = match sample_trace_for::<E>(BLS12381_DECOMPRESS_ELF, sign_rule) {
            Some(trace) => trace,
            None => {
                eprintln!("No Bls12381Decompress trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main, sign_rule);
    }

    #[test]
    fn test_weierstrass_decompress_secp256r1_constraint_check() {
        type E = Secp256r1;
        let sign_rule = SignChoiceRule::LeastSignificantBit;
        let main = match sample_trace_for::<E>(SECP256R1_DECOMPRESS_ELF, sign_rule) {
            Some(trace) => trace,
            None => {
                eprintln!("No Secp256r1Decompress trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main, sign_rule);
    }

    /// Generate a random WeierstrassDecompress trace for performance testing.
    fn random_weierstrass_decompress_trace<E: EllipticCurve + WeierstrassParameters>(
        log_n: usize,
        _seed: u64,
        elf: &[u8],
        sign_rule: SignChoiceRule,
    ) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace_for::<E>(elf, sign_rule).expect("sample trace should exist");
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

    /// Shared multi-round sumcheck benchmark logic for WeierstrassDecompressPolyAir.
    fn do_perf_multi_round_sumcheck_decompress<E: EllipticCurve + WeierstrassParameters>(
        air: WeierstrassDecompressPolyAir<E>,
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
        let is_lex = air.is_lexicographic();
        let total_lookups = if is_lex {
            num_base_lookups::<E::BaseField>() + num_lex_lookups::<E::BaseField>()
        } else {
            num_base_lookups::<E::BaseField>()
        };
        let total_precomputed = if is_lex {
            num_lex_precomputed::<E::BaseField>()
        } else {
            num_lsb_precomputed::<E::BaseField>()
        };

        // Conservative upper bound for constraint reducer (matches constraint check test).
        let nb_limbs = <E::BaseField as FieldParameters>::NB_LIMBS;
        let nb_witness = <E::BaseField as FieldParameters>::NB_WITNESS_LIMBS;
        let nb_add_witness = <E::BaseField as FieldParameters>::NB_ADD_WITNESS_LIMBS;
        let words_field_element = <E::BaseField as NumWords>::WordsFieldElement::USIZE;

        let field_lt_gate_count = nb_limbs + 3;
        let field_op_gate_count = nb_witness + 1;
        let field_add_gate_count = (nb_add_witness + 1) + 1;

        let mut num_gate_constraints = 3 * field_lt_gate_count +
            4 * field_op_gate_count +
            2 * field_add_gate_count +
            2 +
            2 +
            words_field_element * 2 * 3 +
            2;

        if is_lex {
            num_gate_constraints += 8 + 2 * field_lt_gate_count;
        }

        let num_reducer = num_gate_constraints + total_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(reducer_seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc = <WeierstrassDecompressPolyAir<E> as FullAir<
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
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SECP256K1_DECOMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);
        let sign_rule = SignChoiceRule::LeastSignificantBit;

        let main = random_weierstrass_decompress_trace::<E>(
            log_n,
            seed,
            SECP256K1_DECOMPRESS_ELF,
            sign_rule,
        );
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassDecompressPolyAir::<E>::new(sign_rule);
        do_perf_multi_round_sumcheck_decompress(air, main);
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_bls12381() {
        type E = Bls12381;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::BLS12381_DECOMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);
        let sign_rule = SignChoiceRule::Lexicographic;

        let main = random_weierstrass_decompress_trace::<E>(
            log_n,
            seed,
            BLS12381_DECOMPRESS_ELF,
            sign_rule,
        );
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassDecompressPolyAir::<E>::new(sign_rule);
        do_perf_multi_round_sumcheck_decompress(air, main);
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck_secp256r1() {
        type E = Secp256r1;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SECP256R1_DECOMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);
        let sign_rule = SignChoiceRule::LeastSignificantBit;

        let main = random_weierstrass_decompress_trace::<E>(
            log_n,
            seed,
            SECP256R1_DECOMPRESS_ELF,
            sign_rule,
        );
        assert_eq!(main.height(), 1 << log_n);
        let air = WeierstrassDecompressPolyAir::<E>::new(sign_rule);
        do_perf_multi_round_sumcheck_decompress(air, main);
    }
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
// Counts depend on `sign_rule` (LSB vs Lexicographic), so these cannot be `const fn`.
impl<E: EllipticCurve> WeierstrassDecompressPolyAir<E> {
    pub fn num_lookups(&self) -> usize {
        if self.is_lexicographic() {
            num_base_lookups::<E::BaseField>() + num_lex_lookups::<E::BaseField>()
        } else {
            num_base_lookups::<E::BaseField>()
        }
    }

    pub fn num_precomputed(&self) -> usize {
        if self.is_lexicographic() {
            num_lex_precomputed::<E::BaseField>()
        } else {
            num_lsb_precomputed::<E::BaseField>()
        }
    }
}
