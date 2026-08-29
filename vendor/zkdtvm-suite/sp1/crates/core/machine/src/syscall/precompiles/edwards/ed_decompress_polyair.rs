use std::{marker::PhantomData, ops::Deref};

use dt_core_executor::syscalls::SyscallCode;
use dt_curves::{
    edwards::{ed25519::Ed25519BaseField, EdwardsParameters, NUM_LIMBS},
    params::{FieldParameters, NumWords},
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
            field_op_add_gate_constraints_all_betas, field_op_beta_from_coeffs,
            field_op_div_gate_constraints_all_betas, field_op_lookup,
            field_op_mul_gate_constraints_all_betas, field_op_num_interactions,
            field_op_precompute_lc, field_op_precompute_witness_beta,
            field_op_sub_gate_constraints_all_betas, FieldOpBetaConsts,
        },
        field_sqrt::{field_sqrt_lookup, field_sqrt_num_interactions, field_sqrt_precompute_lc},
        range::{
            field_lt_gate_constraints, field_lt_lookup, field_lt_num_interactions,
            field_lt_precompute_lc, FieldLtCols,
        },
    },
};

use super::ed_decompress::{EdDecompressCols, NUM_ED_DECOMPRESS_COLS};

// ============================================================================
// Constants
// ============================================================================

const fn num_lookups<P: FieldParameters + NumWords>() -> usize {
    field_lt_num_interactions::<P>()                   // y_range
    + field_op_num_interactions::<P>()                 // yy
    + field_op_num_interactions::<P>()                 // u
    + field_op_num_interactions::<P>()                 // dyy
    + field_op_num_interactions::<P>()                 // v
    + field_op_num_interactions::<P>()                 // u_div_v
    + field_sqrt_num_interactions::<P>()               // x
    + field_op_num_interactions::<P>()                 // neg_x
    + field_lt_num_interactions::<P>()                 // neg_x_range
    + <P as NumWords>::WordsFieldElement::USIZE * 4    // x_access memory_write
    + <P as NumWords>::WordsFieldElement::USIZE * 4    // y_access memory_read
    + 1 // recv(Syscall)
}

/// Precomputed linear combinations: one per lookup + 7 witness_betas
/// + 11 result/carry β-evaluations (yy/u/dyy/v/u_div_v result_β + carry_β each = 10; x_mul.carry_β
///   = 1) + 2 diff_betas.
const fn num_precomputed<P: FieldParameters + NumWords>() -> usize {
    num_lookups::<P>() + 20
}

// ============================================================================
// Column offsets within EdDecompressCols<u8>
//
// Layout (#[repr(C)]):
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  ptr              ← precompute-only (skipped)
//   [4]  sign
//   [5 + i*13]            x_access[i] = MemoryWriteCols (13 cols each)
//     +0..+4   prev_value
//     +4..+8   access.value           ← precompute-only (skipped from reserved_poly)
//     +8       access.prev_shard
//     +9       access.prev_clk
//     +10      access.compare_clk
//     +11      access.diff_16bit_limb
//     +12      access.diff_12bit_limb
//   [5 + WFE*13 + i*9]   y_access[i] = MemoryReadCols (9 cols, all needed)
//   Then: neg_x_range, y_range, yy, u, dyy, v, u_div_v, x(FieldSqrt), neg_x
// ============================================================================

const COL_IS_REAL: usize = 0;
const COL_SHARD: usize = 1;
const COL_CLK: usize = 2;
// col 3 = ptr (precompute-only, skipped from reserved_poly)
const COL_SIGN: usize = 4;
const COL_X_ACCESS_BASE: usize = 5;

// Reserved-poly indices for scalars (ptr is skipped, so sign shifts to 3)
const RES_IS_REAL: usize = 0;
const RES_SHARD: usize = 1;
const RES_CLK: usize = 2;
const RES_SIGN: usize = 3;
const MEM_WRITE_COLS_SIZE: usize = 13;
const MEM_READ_COLS_SIZE: usize = 9;
const MEM_ACCESS_PREV_SHARD_OFF: usize = 4;
const MEM_ACCESS_PREV_CLK_OFF: usize = 5;
const MEM_ACCESS_COMPARE_CLK_OFF: usize = 6;
const MEM_ACCESS_DIFF_16_OFF: usize = 7;
const MEM_ACCESS_DIFF_12_OFF: usize = 8;
const WORDS_FE: usize = 8;

#[inline]
fn field_op_cols_size() -> usize {
    NUM_LIMBS + NUM_LIMBS + Ed25519BaseField::NB_WITNESS_LIMBS
}

#[inline]
fn col_y_access_base() -> usize {
    COL_X_ACCESS_BASE + WORDS_FE * MEM_WRITE_COLS_SIZE
}

#[inline]
fn col_neg_x_range_base() -> usize {
    col_y_access_base() + WORDS_FE * MEM_READ_COLS_SIZE
}

#[inline]
fn col_y_range_base() -> usize {
    col_neg_x_range_base() + NUM_LIMBS + 2
}

#[inline]
fn col_yy_base() -> usize {
    col_y_range_base() + NUM_LIMBS + 2
}

#[inline]
fn col_u_base() -> usize {
    col_yy_base() + field_op_cols_size()
}

#[inline]
fn col_dyy_base() -> usize {
    col_u_base() + field_op_cols_size()
}

#[inline]
fn col_v_base() -> usize {
    col_dyy_base() + field_op_cols_size()
}

#[inline]
fn col_u_div_v_base() -> usize {
    col_v_base() + field_op_cols_size()
}

#[inline]
fn col_x_mul_base() -> usize {
    col_u_div_v_base() + field_op_cols_size()
}

#[inline]
fn col_x_range_base() -> usize {
    col_x_mul_base() + field_op_cols_size()
}

#[inline]
fn col_x_lsb() -> usize {
    col_x_range_base() + NUM_LIMBS + 2
}

#[inline]
fn col_neg_x_base() -> usize {
    col_x_lsb() + 1
}

// ============================================================================
// Reserved-poly row layout (positions in the reserved slice).
//
//   [0]  is_real
//   [1]  shard
//   [2]  clk
//   [3]  sign
//   [4 + i*5]              x_access[i]: timestamps(5) only
//   [4 + WFE*5 + i*9]     y_access[i]: access.value(4) + timestamps(5)
//   Then: y_range(L+2), yy(2L), u(2L), dyy(2L), v(2L), u_div_v(2L),
//         x.mul(2L), x.range(L+2), x.lsb(1), neg_x(2L), neg_x_range(L+2)
//
// NOTE: x_access[i].prev_value is NOT in reserved_poly — it is consumed
// only in precompute_lc (memory_readwrite_precompute_lc and diff_x/diff_neg_x
// polynomial optimization), not in eval.
// ============================================================================

const RES_NUM_SCALAR: usize = 4;
const RES_PER_X_ACCESS: usize = 5; // timestamps only (prev_value removed)
const RES_PER_Y_ACCESS: usize = 9; // access.value + timestamps (field_lt needs limbs)

#[inline]
fn res_x_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + i * RES_PER_X_ACCESS
}

#[inline]
fn res_y_access_base(i: usize) -> usize {
    RES_NUM_SCALAR + WORDS_FE * RES_PER_X_ACCESS + i * RES_PER_Y_ACCESS
}

#[inline]
fn res_ops_start() -> usize {
    RES_NUM_SCALAR + WORDS_FE * RES_PER_X_ACCESS + WORDS_FE * RES_PER_Y_ACCESS
}

#[inline]
fn res_y_range_base() -> usize {
    res_ops_start()
}

// Reserved-poly layout after β-eval optimization:
//   y_range          (L+2)  FieldLt
//   x_mul_result     (L)    FieldOp result (= sqrt(u_div_v); feeds FieldLt + Sub)
//   x_range          (L+2)  FieldLt
//   x_lsb            (1)
//   neg_x            (2L)   FieldOp result+carry (feeds FieldLt)
//   neg_x_range      (L+2)  FieldLt
//
// Dropped from reserved_poly: yy, u, dyy, v, u_div_v (entirely) and x_mul.carry.
// All precomputed as β-evals.
#[inline]
fn res_x_mul_base() -> usize {
    res_y_range_base() + NUM_LIMBS + 2
}

#[inline]
fn res_x_range_base() -> usize {
    res_x_mul_base() + NUM_LIMBS
}

#[inline]
fn res_x_lsb() -> usize {
    res_x_range_base() + NUM_LIMBS + 2
}

#[inline]
fn res_neg_x_base() -> usize {
    res_x_lsb() + 1
}

#[inline]
fn res_neg_x_range_base() -> usize {
    res_neg_x_base() + 2 * NUM_LIMBS
}

// ============================================================================
// PolyAir wrapper
// ============================================================================

#[derive(Clone, Copy)]
pub struct EdDecompressPolyAir<E: EdwardsParameters> {
    _marker: PhantomData<E>,
}

impl<E: EdwardsParameters> EdDecompressPolyAir<E> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<E: EdwardsParameters, AB: FullAirBuilder> FullAir<AB> for EdDecompressPolyAir<E>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_ED_DECOMPRESS_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        crate::syscall::precompiles::required_max_beta_power_for_field::<E::BaseField>(16)
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only reserve columns actually read by `eval` / `lookup`. Skipped:
        //   - ptr                          (precompute-only: memory address, syscall LC)
        //   - x_access[i].prev_value       (consumed in precompute_lc for memory_readwrite)
        //   - x_access[i].access.value     (consumed as x_value(β) in precompute_lc for diff(β) x
        //     linkage)
        //   - all FieldOpCols.witness      (precompute-only: witness(β))
        let l = NUM_LIMBS;
        let mut cols: Vec<PairCol> = Vec::new();

        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_SHARD));
        cols.push(PairCol::Main(COL_CLK));
        cols.push(PairCol::Main(COL_SIGN));

        // x_access: timestamps(5) only. prev_value consumed in precompute_lc, not eval.
        for i in 0..WORDS_FE {
            let base = COL_X_ACCESS_BASE + i * MEM_WRITE_COLS_SIZE;
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_SHARD_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_PREV_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_COMPARE_CLK_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_16_OFF));
            cols.push(PairCol::Main(base + 4 + MEM_ACCESS_DIFF_12_OFF));
        }

        // y_access: full 9 cols each
        let y_base = col_y_access_base();
        for i in 0..WORDS_FE {
            let base = y_base + i * MEM_READ_COLS_SIZE;
            for k in 0..MEM_READ_COLS_SIZE {
                cols.push(PairCol::Main(base + k));
            }
        }

        // y_range: byte_flags(L) + lhs + rhs
        let yr = col_y_range_base();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(yr + k));
        }

        // yy / u / dyy / v / u_div_v: dropped entirely from reserved_poly.
        // Their result+carry β-evals are precomputed in precompute_lc.

        // x.multiplication: keep ONLY result(L) (= sqrt(u_div_v); feeds FieldLt + Sub).
        // carry_β is precomputed.
        let xm = col_x_mul_base();
        for k in 0..l {
            cols.push(PairCol::Main(xm + k));
        }

        // x.range: byte_flags(L) + lhs + rhs
        let xr = col_x_range_base();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(xr + k));
        }

        // x.lsb
        cols.push(PairCol::Main(col_x_lsb()));

        // neg_x: result(L) + carry(L), skip witness
        let nx = col_neg_x_base();
        for k in 0..(2 * l) {
            cols.push(PairCol::Main(nx + k));
        }

        // neg_x_range: byte_flags(L) + lhs + rhs
        let nxr = col_neg_x_range_base();
        for k in 0..(l + 2) {
            cols.push(PairCol::Main(nxr + k));
        }

        cols
    }

    // ========================================================================
    // Phase 1: precompute_lc
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main_ptr = {
            let main = builder.main();
            main.as_ptr() as *const AB::VarMaybeExt
        };
        let local: &EdDecompressCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main_ptr) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let ptr = local.ptr.clone();

        // ── y_range (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> = local.y_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, Ed25519BaseField>(
                builder,
                local.y_range.lhs_comparison_byte.clone(),
                local.y_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── yy (FieldOpCols, Mul) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.yy.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.yy.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.yy.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── u (FieldOpCols, Sub) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.u.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.u.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.u.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── dyy (FieldOpCols, Mul) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.dyy.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.dyy.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.dyy.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── v (FieldOpCols, Add) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.v.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.v.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.v.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── u_div_v (FieldOpCols, Div) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.u_div_v.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.u_div_v.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.u_div_v.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── x (FieldSqrtCols) ──
        // a_limbs = u_div_v.result = sqrt² (the actual multiplication input before the
        // hack-overwrite)
        let x_a_limbs: Vec<AB::VarMaybeExt> = local.u_div_v.result.0.iter().cloned().collect();
        field_sqrt_precompute_lc::<AB, Ed25519BaseField>(builder, &local.x, &x_a_limbs);

        // ── neg_x (FieldOpCols, Sub) ──
        field_op_precompute_lc::<AB, Ed25519BaseField>(
            builder,
            &local.neg_x.result.0.iter().cloned().collect::<Vec<_>>(),
            &local.neg_x.carry.0.iter().cloned().collect::<Vec<_>>(),
            &local.neg_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── neg_x_range (FieldLtCols) ──
        {
            let flags: Vec<AB::VarMaybeExt> =
                local.neg_x_range.byte_flags.0.iter().cloned().collect();
            field_lt_precompute_lc::<AB, Ed25519BaseField>(
                builder,
                local.neg_x_range.lhs_comparison_byte.clone(),
                local.neg_x_range.rhs_comparison_byte.clone(),
                &flags,
            );
        }

        // ── x_access: memory_write (WordsFieldElement × 4 interactions) ──
        for i in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            let addr =
                ptr.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32((i as u32) * 4));
            memory_readwrite_precompute_lc(
                builder,
                &local.x_access[i].access,
                &local.x_access[i].prev_value,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // ── y_access: memory_read (WordsFieldElement × 4 interactions) ──
        for i in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            let addr =
                ptr.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32((i as u32) * 4 + 32));
            memory_read_precompute_lc(
                builder,
                &local.y_access[i].access,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // ── recv(Syscall) ──
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let syscall_id_felt = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
            SyscallCode::ED_DECOMPRESS.syscall_id(),
        ));
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![shard, clk, syscall_id_felt, ptr, local.sign.clone()],
        ));

        // ── Precompute witness(β) for each field op (used in eval gate constraints) ──
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.yy.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.u.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.dyy.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.v.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.u_div_v.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.x.multiplication.witness.0.iter().cloned().collect::<Vec<_>>(),
        );
        field_op_precompute_witness_beta::<AB, Ed25519BaseField>(
            builder,
            &local.neg_x.witness.0.iter().cloned().collect::<Vec<_>>(),
        );

        // ── Precompute β-evals for inner FieldOps whose result+carry limbs are not
        // in reserved_poly. Order matches eval read positions [start+7..start+18]:
        //   yy.r, yy.c, u.r, u.c, dyy.r, dyy.c, v.r, v.c, u_div_v.r, u_div_v.c,
        //   x.multiplication.carry_β.
        for limbs in [
            &local.yy.result.0[..],
            &local.yy.carry.0[..],
            &local.u.result.0[..],
            &local.u.carry.0[..],
            &local.dyy.result.0[..],
            &local.dyy.carry.0[..],
            &local.v.result.0[..],
            &local.v.carry.0[..],
            &local.u_div_v.result.0[..],
            &local.u_div_v.carry.0[..],
            &local.x.multiplication.carry.0[..],
        ] {
            builder.retain_precomputed(field_op_beta_from_coeffs(
                builder,
                &limbs.iter().cloned().collect::<Vec<_>>(),
            ));
        }

        // ── Polynomial optimizations for x linkage ──
        {
            let x_limbs: Vec<AB::VarMaybeExt> =
                local.x.multiplication.result.0.iter().cloned().collect();
            let x_access_limbs: Vec<AB::VarMaybeExt> =
                local.x_access.iter().flat_map(|acc| acc.access.value.0.iter().cloned()).collect();

            let diff_x_coeffs: Vec<AB::VarMaybeExt> = x_limbs
                .iter()
                .zip(x_access_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();
            let beta_powers = builder.beta_powers();
            let zero_ext = AB::from_ef(AB::EF::zero());
            let diff_x_beta = Polynomial::from_coefficients(&diff_x_coeffs)
                .eval_with_powers(beta_powers, zero_ext.clone());
            builder.retain_precomputed(diff_x_beta);

            let neg_x_limbs: Vec<AB::VarMaybeExt> = local.neg_x.result.0.iter().cloned().collect();
            let diff_neg_x_coeffs: Vec<AB::VarMaybeExt> = neg_x_limbs
                .iter()
                .zip(x_access_limbs.iter())
                .map(|(r, v)| r.clone() - v.clone())
                .collect();
            let beta_powers = builder.beta_powers();
            let diff_neg_x_beta = Polynomial::from_coefficients(&diff_neg_x_coeffs)
                .eval_with_powers(beta_powers, zero_ext);
            builder.retain_precomputed(diff_neg_x_beta);
        }
    }

    // ========================================================================
    // Phase 2: eval — gate constraints
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let beta_consts = FieldOpBetaConsts::<AB>::new::<Ed25519BaseField>(builder);
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let shard = local[RES_SHARD].clone();
        let clk = local[RES_CLK].clone();
        let sign = local[RES_SIGN].clone();
        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let zero_word = Word([zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
        let l = NUM_LIMBS;

        // Precompute layout (start = num_lookups):
        //   [0..7]   witness_betas (yy, u, dyy, v, u_div_v, x.mul, neg_x)
        //   [7..17]  10 β: yy.r/c, u.r/c, dyy.r/c, v.r/c, u_div_v.r/c
        //   [17]     x.multiplication.carry_β
        //   [tail..] 2 diff_betas (accessed via total_precomputed-2/-1)
        let (witness_betas, yy_r, yy_c, u_r, u_c, dyy_r, dyy_c, v_r, v_c, udv_r, udv_c, x_mul_c) = {
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let start = num_lookups::<E::BaseField>();
            (
                vec![
                    pc[start].clone(),     // yy
                    pc[start + 1].clone(), // u
                    pc[start + 2].clone(), // dyy
                    pc[start + 3].clone(), // v
                    pc[start + 4].clone(), // u_div_v
                    pc[start + 5].clone(), // x.multiplication
                    pc[start + 6].clone(), // neg_x
                ],
                pc[start + 7].clone(),  // yy.r
                pc[start + 8].clone(),  // yy.c
                pc[start + 9].clone(),  // u.r
                pc[start + 10].clone(), // u.c
                pc[start + 11].clone(), // dyy.r
                pc[start + 12].clone(), // dyy.c
                pc[start + 13].clone(), // v.r
                pc[start + 14].clone(), // v.c
                pc[start + 15].clone(), // u_div_v.r
                pc[start + 16].clone(), // u_div_v.c
                pc[start + 17].clone(), // x.multiplication.carry_β
            )
        };

        // -- Extract input y limbs from y_access value columns --
        let y_limbs: Vec<AB::VarMaybeExt> = (0..WORDS_FE)
            .flat_map(|i| {
                let base = res_y_access_base(i);
                (0..4).map(move |k| local[base + k].clone())
            })
            .collect();

        // ── y_range: FieldLtCols gate constraints (y < modulus) ──
        let modulus_limbs: Vec<AB::VarMaybeExt> = Ed25519BaseField::MODULUS
            .iter()
            .map(|&byte| AB::VarMaybeExt::from(AB::F::from_canonical_u8(byte)))
            .collect();
        {
            let yr = res_y_range_base();
            let y_range = FieldLtCols::<AB::VarMaybeExt, Ed25519BaseField> {
                byte_flags: (0..l).map(|k| local[yr + k].clone()).collect(),
                lhs_comparison_byte: local[yr + l].clone(),
                rhs_comparison_byte: local[yr + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, Ed25519BaseField>(
                builder,
                &y_limbs,
                &modulus_limbs,
                &y_range,
                is_real.clone(),
            );
        }

        let y_beta = field_op_beta_from_coeffs(builder, &y_limbs);

        // ── yy: Sqr(y) — all βs precomputed ──
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            y_beta.clone(),
            y_beta,
            yy_r.clone(),
            yy_c,
            witness_betas[0].clone(),
            &beta_consts,
        );

        // ── u: Sub(yy, 1) ──
        let mut one_limbs = vec![zero.clone(); l];
        one_limbs[0] = one.clone();
        let one_beta = field_op_beta_from_coeffs(builder, &one_limbs);
        field_op_sub_gate_constraints_all_betas::<AB>(
            builder,
            yy_r.clone(),
            one_beta.clone(),
            u_r.clone(),
            u_c,
            witness_betas[1].clone(),
            &beta_consts,
        );

        // ── dyy: Mul(d, yy) ──
        let d_limbs: Vec<AB::VarMaybeExt> =
            <E::BaseField as FieldParameters>::to_limbs_field::<AB::F, _>(&E::d_biguint())
                .0
                .iter()
                .map(|&f| AB::VarMaybeExt::from(f))
                .collect();
        let d_beta = field_op_beta_from_coeffs(builder, &d_limbs);
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            d_beta,
            yy_r,
            dyy_r.clone(),
            dyy_c,
            witness_betas[2].clone(),
            &beta_consts,
        );

        // ── v: Add(1, dyy) ──
        field_op_add_gate_constraints_all_betas::<AB>(
            builder,
            one_beta,
            dyy_r,
            v_r.clone(),
            v_c,
            witness_betas[3].clone(),
            &beta_consts,
        );

        // ── u_div_v: Div(u, v) ──
        field_op_div_gate_constraints_all_betas::<AB>(
            builder,
            u_r,
            v_r,
            udv_r.clone(),
            udv_c,
            witness_betas[4].clone(),
            &beta_consts,
        );

        // ── x: FieldSqrt gate constraints — sqrt_limbs (= x.mul.result) stay in
        // reserved_poly (feed FieldLt + Sub for neg_x). Use precomputed u_div_v.r
        // as the "result" β slot of the mul gate; x.mul.carry_β is precomputed.
        let sqrt_limbs: Vec<AB::VarMaybeExt> =
            (0..l).map(|k| local[res_x_mul_base() + k].clone()).collect();
        let sqrt_beta = field_op_beta_from_coeffs(builder, &sqrt_limbs);
        field_op_mul_gate_constraints_all_betas::<AB>(
            builder,
            sqrt_beta.clone(),
            sqrt_beta,
            udv_r,
            x_mul_c,
            witness_betas[5].clone(),
            &beta_consts,
        );
        {
            let xr = res_x_range_base();
            let x_range = FieldLtCols::<AB::VarMaybeExt, Ed25519BaseField> {
                byte_flags: (0..l).map(|k| local[xr + k].clone()).collect(),
                lhs_comparison_byte: local[xr + l].clone(),
                rhs_comparison_byte: local[xr + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, Ed25519BaseField>(
                builder,
                &sqrt_limbs,
                &modulus_limbs,
                &x_range,
                is_real.clone(),
            );
        }
        {
            let lsb = local[res_x_lsb()].clone();
            builder.assert_zero(lsb.clone() * (one.clone() - lsb.clone()));
            builder.assert_zero(is_real.clone() * (lsb - zero.clone()));
        }

        // ── neg_x: Sub(0, x) — neg_x.result limbs stay in reserved_poly (FieldLt) ──
        // We compute neg_x.r/c β-evals inline here (no extra precompute slot — only
        // saves 1 Horner since result limbs are still in reserved_poly).
        let neg_x_base_off = res_neg_x_base();
        let neg_x_result_limbs: Vec<AB::VarMaybeExt> =
            (0..l).map(|k| local[neg_x_base_off + k].clone()).collect();
        let neg_x_carry_limbs: Vec<AB::VarMaybeExt> =
            (0..l).map(|k| local[neg_x_base_off + l + k].clone()).collect();
        let neg_x_result_beta = field_op_beta_from_coeffs(builder, &neg_x_result_limbs);
        let neg_x_carry_beta = field_op_beta_from_coeffs(builder, &neg_x_carry_limbs);
        {
            let zero_limbs = vec![zero; l];
            let zero_beta = field_op_beta_from_coeffs(builder, &zero_limbs);
            let sqrt_beta_for_sub = field_op_beta_from_coeffs(builder, &sqrt_limbs);
            field_op_sub_gate_constraints_all_betas::<AB>(
                builder,
                zero_beta,
                sqrt_beta_for_sub,
                neg_x_result_beta,
                neg_x_carry_beta,
                witness_betas[6].clone(),
                &beta_consts,
            );
        }

        // ── neg_x_range: FieldLtCols gate constraints (neg_x < modulus) ──
        {
            let neg_x_limbs: Vec<AB::VarMaybeExt> = neg_x_result_limbs.clone();
            let nxr = res_neg_x_range_base();
            let neg_x_range = FieldLtCols::<AB::VarMaybeExt, Ed25519BaseField> {
                byte_flags: (0..l).map(|k| local[nxr + k].clone()).collect(),
                lhs_comparison_byte: local[nxr + l].clone(),
                rhs_comparison_byte: local[nxr + l + 1].clone(),
            };
            field_lt_gate_constraints::<AB, Ed25519BaseField>(
                builder,
                &neg_x_limbs,
                &modulus_limbs,
                &neg_x_range,
                is_real.clone(),
            );
        }

        // ── x linkage constraints ──
        {
            let total_precomputed = num_precomputed::<E::BaseField>();
            let precomputed = builder.precomputed();
            let pc_binding = precomputed.row_slice(0);
            let pc: &[AB::VarExt] = pc_binding.deref();
            let diff_x_beta = pc[total_precomputed - 2].clone();
            let diff_neg_x_beta = pc[total_precomputed - 1].clone();

            builder.when(is_real.clone()).when(sign.clone()).assert_zero_ext(diff_neg_x_beta);
            builder
                .when(is_real.clone())
                .when(one.clone() - sign.clone())
                .assert_zero_ext(diff_x_beta);
        }

        // ── memory timestamp constraints ──
        for i in 0..WORDS_FE {
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
                clk.clone(),
                is_real.clone(),
            );
        }
        for i in 0..WORDS_FE {
            let base = res_y_access_base(i);
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

        // ── Boolean constraints ──
        builder.assert_zero(is_real.clone() * (one.clone() - is_real));
        builder.assert_zero(sign.clone() * (one - sign));
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        field_lt_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_sqrt_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_op_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());
        field_lt_lookup::<AB, Ed25519BaseField>(builder, is_real.clone());

        for _ in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        for _ in 0..<E::BaseField as NumWords>::WordsFieldElement::USIZE {
            memory_read_lookup(builder, is_real.clone());
        }

        builder.recv(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::ed_decompress::EdDecompressChip;
use dt_core_executor::{
    events::{ByteLookupEvent, PrecompileEvent},
    ExecutionRecord, Program,
};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use num::BigUint;
use p3_air::BaseAir;
use p3_field::Field;
use std::borrow::BorrowMut;

use crate::syscall::precompiles::add_field_lt_bitvec_lookups;

impl<F: Field, E: EdwardsParameters> BaseAir<F> for EdDecompressPolyAir<E> {
    fn width(&self) -> usize {
        NUM_ED_DECOMPRESS_COLS
    }
}

impl<F: Field, E: EdwardsParameters> MachineAir<F> for EdDecompressPolyAir<E> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <EdDecompressChip<E> as MachineAir<F>>::name(&EdDecompressChip::<E>::new()) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        EdDecompressChip::<E>::new().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        // Standard BLU from base chip (memory accesses, field ops, etc.)
        <EdDecompressChip<E> as MachineAir<F>>::generate_dependencies(
            &EdDecompressChip::<E>::new(),
            input,
            output,
        );

        // PolyAir-only BitVec lookups for y_range, x.range, neg_x_range.
        let events = input.get_precompile_events(SyscallCode::ED_DECOMPRESS);
        for (_, event) in events {
            let PrecompileEvent::EdDecompress(event) = event else { unreachable!() };
            let y = BigUint::from_bytes_le(&event.y_bytes);
            let mut row = [F::zero(); NUM_ED_DECOMPRESS_COLS];
            let cols: &mut EdDecompressCols<F> = row.as_mut_slice().borrow_mut();
            let mut ignored_blu: Vec<ByteLookupEvent> = Vec::new();
            cols.populate_field_ops::<E>(&mut ignored_blu, &y);
            add_field_lt_bitvec_lookups::<F, Ed25519BaseField>(output, &cols.y_range);
            add_field_lt_bitvec_lookups::<F, Ed25519BaseField>(output, &cols.x.range);
            add_field_lt_bitvec_lookups::<F, Ed25519BaseField>(output, &cols.neg_x_range);
        }
    }

    fn padding_row(&self) -> Vec<F> {
        EdDecompressChip::<E>::new().padding_row()
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <EdDecompressChip<E> as MachineAir<F>>::included(&EdDecompressChip::<E>::new(), shard)
    }

    fn local_only(&self) -> bool {
        <EdDecompressChip<E> as MachineAir<F>>::local_only(&EdDecompressChip::<E>::new())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use dt_core_executor::{ExecutionRecord, Executor, Program};
    use dt_curves::edwards::ed25519::Ed25519Parameters;
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
    use test_artifacts::ED_DECOMPRESS_ELF;

    use super::super::ed_decompress::EdDecompressChip;
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

    fn beta_powers_for<E: EdwardsParameters>(air: &EdDecompressPolyAir<E>, beta: EF) -> Vec<EF> {
        let max = <EdDecompressPolyAir<E> as FullAir<
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

    fn reserved_poly_matrix<E: EdwardsParameters>(
        air: &EdDecompressPolyAir<E>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <EdDecompressPolyAir<E> as FullAir<
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

    fn sample_trace_for<E: EdwardsParameters>(elf: &[u8]) -> Option<RowMajorMatrix<F>> {
        let program = Program::from(elf).unwrap();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::ED_DECOMPRESS).is_empty() {
                continue;
            }

            let mut ec_shard = ExecutionRecord::new(shard.program.clone());
            ec_shard.precompile_events = shard.precompile_events.clone();

            let chip = EdDecompressChip::<E>::new();
            return Some(
                chip.generate_trace(&ec_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    fn run_constraint_check<E: EdwardsParameters>(main: RowMajorMatrix<F>) {
        let air = EdDecompressPolyAir::<E>::new();
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

        let nb_limbs = <E::BaseField as FieldParameters>::NB_LIMBS;
        let words_field_element = <E::BaseField as NumWords>::WordsFieldElement::USIZE;
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
    fn test_ed_decompress_constraint_check() {
        type E = Ed25519Parameters;
        let main = match sample_trace_for::<E>(ED_DECOMPRESS_ELF) {
            Some(trace) => trace,
            None => {
                eprintln!("No EdDecompress trace found -- skipping test");
                return;
            }
        };
        run_constraint_check::<E>(main);
    }

    fn random_ed_decompress_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace_for::<Ed25519Parameters>(ED_DECOMPRESS_ELF)
            .expect("Should find EdDecompress events in ED_DECOMPRESS_ELF");
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
        type E = Ed25519Parameters;
        let air = EdDecompressPolyAir::<E>::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::ED_DECOMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_ed_decompress_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers_for(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let total_lookups = num_lookups::<Ed25519BaseField>();
        let total_precomputed = num_precomputed::<Ed25519BaseField>();

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
        let reserved_poly_desc = <EdDecompressPolyAir<E> as FullAir<
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
impl<E: EdwardsParameters> EdDecompressPolyAir<E> {
    pub const fn num_lookups(&self) -> usize {
        num_lookups::<E::BaseField>()
    }
    pub const fn num_precomputed(&self) -> usize {
        num_precomputed::<E::BaseField>()
    }
}
