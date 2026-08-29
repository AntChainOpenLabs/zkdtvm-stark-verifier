//! PolyAir adaptation of ShaCompressChip.
//!
//! Bridges `ShaCompressCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (per row = 1 round)
//!
//! Each of the 64 rows (rounds) has:
//!   1. recv(ShaCompress) — [shard, clk, w_ptr, i, a, b, c, d, e, f, g, h]
//!   2. send(ShaCompress) — [shard, clk, w_ptr, i+1, new_a, b, c, d, new_e, f, g, h]
//!   3. memory_read — w[i] access
//!   4-89. BitVec/FixedRotateRight/XorN/AndN/AddN helper lookups
//!
//! ## Gate Constraints
//!
//!   - is_real boolean gate
//!   - 2× BitVec boolean packs for one-hot/sum
//!   - i reconstruction from one-hot
//!   - K constant selection via one-hot
//!   - FixedRotateRight, XorN, AndN, AddN, Not operations

use std::ops::Deref;

use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    memory::polyair::{
        memory_read_lookup, memory_read_precompute_lc, memory_timestamp_gate_constraints,
    },
    operations_dt::{
        add_n_lookup, add_n_precompute_lc, add_n_without_result_gate_constraints, and_n_lookup,
        and_n_precompute_lc, compact_word_to_arr, fixed_rotate_right_gate_constraints,
        fixed_rotate_right_lookup, fixed_rotate_right_precompute_lc, fixed_rotate_right_result,
        word_to_compact, xor_n_lookup, xor_n_precompute_lc, CompactWord,
    },
};

use super::{
    columns::{ShaCompressCols, NUM_SHA_COMPRESS_COLS},
    SHA_COMPRESS_K,
};

// ============================================================================
// Constants
// ============================================================================

/// Total lookup interactions per row.
///
/// 2 ShaCompress interactions
/// + 4 memory_read interactions
/// + 2 BitVec
/// + 6 FixedRotateRight * 4
/// + XorN(3,2,3,3) = 8 + 4 + 8 + 8
/// + 5 AndN(2) = 20
/// + AddN(5,2,2,2) = 3 + 2 + 2 + 2
const NUM_LOOKUPS_PER_ROW: usize = 89;

/// Maximum payload size: ShaCompress has 4 + 8*2 = 20 values
const MAX_LOOKUP_VALUES: usize = 20;

/// LogUp batch size
const BATCH_SIZE: usize = 3;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for ShaCompressChip.
#[derive(Clone, Copy, Default)]
pub struct ShaCompressPolyAir;

impl ShaCompressPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ShaCompressPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_SHA_COMPRESS_COLS).map(PairCol::Main).collect()
    }

    // ========================================================================
    // Phase 1: precompute_lc
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShaCompressCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let w_ptr = local.w_ptr.clone();
        let i = local.i.clone();

        let sha_compress_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
            InteractionKind::ShaCompress as usize,
        ));

        // Build one-hot for K selection
        let low_sum: AB::VarMaybeExt =
            local.i_low_one_hot.iter().cloned().fold(AB::zero_maybe(), |acc, x| acc + x);
        let high_sum: AB::VarMaybeExt =
            local.i_high_one_hot.iter().cloned().fold(AB::zero_maybe(), |acc, x| acc + x);

        // =====================================================================
        // #1: recv(ShaCompress) — [shard, clk, w_ptr, i, a, b, c, d, e, f, g, h]
        // =====================================================================
        let mut recv_values = vec![shard.clone(), clk.clone(), w_ptr.clone(), i.clone()];
        // Add a-h as compact words (2 limbs each)
        for word in [&local.a, &local.b, &local.c, &local.d, &local.e, &local.f, &local.g, &local.h]
        {
            recv_values.push(word.0[0].clone());
            recv_values.push(word.0[1].clone());
        }
        builder
            .retain_precomputed(builder.lookup_denominator(sha_compress_kind.clone(), recv_values));

        // =====================================================================
        // #2: send(ShaCompress) — [shard, clk, w_ptr, i+1, new_a, b, c, d, new_e, f, g, h]
        // =====================================================================
        let i_plus_one = i.clone() + AB::VarMaybeExt::from(AB::F::one());
        let mut send_values = vec![shard.clone(), clk.clone(), w_ptr.clone(), i_plus_one];
        // new_a = temp1_add_temp2.value, new_e = d_add_temp1.value
        for word in [
            &local.temp1_add_temp2.value,
            &local.a,
            &local.b,
            &local.c,
            &local.d_add_temp1.value,
            &local.e,
            &local.f,
            &local.g,
        ] {
            send_values.push(word.0[0].clone());
            send_values.push(word.0[1].clone());
        }
        builder.retain_precomputed(builder.lookup_denominator(sha_compress_kind, send_values));

        // =====================================================================
        // #3: memory_read — w[i] access
        // =====================================================================
        let addr = w_ptr + i * AB::VarMaybeExt::from(AB::F::from_canonical_u32(4));
        memory_read_precompute_lc(
            builder,
            &local.w_access.access,
            addr,
            shard.clone(),
            clk.clone(),
        );
        bitvec_precompute_lc(
            builder,
            local.i_low_one_hot.iter().chain(local.i_high_one_hot.iter()).cloned().collect(),
        );
        bitvec_precompute_lc(builder, vec![low_sum, high_sum]);

        // =====================================================================
        // Operation precompute_lc calls
        // =====================================================================

        // FixedRotateRight operations
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.e_rr_6,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            6,
        );
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.e_rr_11,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            11,
        );
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.e_rr_25,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            25,
        );
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.a_rr_2,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            2,
        );
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.a_rr_13,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            13,
        );
        fixed_rotate_right_precompute_lc::<AB>(
            builder,
            &local.a_rr_22,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            22,
        );

        // XorN operations
        // S1: e_rr_6 ^ e_rr_11 ^ e_rr_25 (N=3, so 2 intermediate results)
        let e_rr_6_result = fixed_rotate_right_result::<AB>(&local.e_rr_6, &local.e, 6);
        let e_rr_11_result = fixed_rotate_right_result::<AB>(&local.e_rr_11, &local.e, 11);
        let e_rr_25_result = fixed_rotate_right_result::<AB>(&local.e_rr_25, &local.e, 25);
        let e_rr_6_arr = compact_word_to_arr::<AB>(&e_rr_6_result, &local.e_rr_6_witness);
        let e_rr_11_arr = compact_word_to_arr::<AB>(&e_rr_11_result, &local.e_rr_11_witness);
        let e_rr_25_arr = compact_word_to_arr::<AB>(&e_rr_25_result, &local.e_rr_25_witness);
        // s1.value has 2 Word results: [e_rr_6 ^ e_rr_11, prev ^ e_rr_25]
        let s1_result_0: [AB::VarMaybeExt; 4] = local.s1.value[0].0.clone().map(|v| v.into());
        let s1_result_1: [AB::VarMaybeExt; 4] = local.s1.value[1].0.clone().map(|v| v.into());
        xor_n_precompute_lc::<AB>(
            builder,
            &[s1_result_0.clone(), s1_result_1],
            &[e_rr_6_arr, s1_result_0.clone()],
            &[e_rr_11_arr, e_rr_25_arr],
        );

        // ch: (e & f) ^ ((~e) & g) (N=2, so 1 intermediate result)
        let e_and_f_result_word = local.e_and_f.value[0].clone();
        let e_not_and_g_result_word = local.e_not_and_g.value[0].clone();
        let e_and_f_arr: [AB::VarMaybeExt; 4] = e_and_f_result_word.0.map(|v| v.into());
        let e_not_and_g_arr: [AB::VarMaybeExt; 4] = e_not_and_g_result_word.0.map(|v| v.into());
        let ch_result_0: [AB::VarMaybeExt; 4] = local.ch.value[0].0.clone().map(|v| v.into());
        xor_n_precompute_lc::<AB>(builder, &[ch_result_0], &[e_and_f_arr], &[e_not_and_g_arr]);

        // S0: a_rr_2 ^ a_rr_13 ^ a_rr_22 (N=3, so 2 intermediate results)
        let a_rr_2_result = fixed_rotate_right_result::<AB>(&local.a_rr_2, &local.a, 2);
        let a_rr_13_result = fixed_rotate_right_result::<AB>(&local.a_rr_13, &local.a, 13);
        let a_rr_22_result = fixed_rotate_right_result::<AB>(&local.a_rr_22, &local.a, 22);
        let a_rr_2_arr = compact_word_to_arr::<AB>(&a_rr_2_result, &local.a_rr_2_witness);
        let a_rr_13_arr = compact_word_to_arr::<AB>(&a_rr_13_result, &local.a_rr_13_witness);
        let a_rr_22_arr = compact_word_to_arr::<AB>(&a_rr_22_result, &local.a_rr_22_witness);
        let s0_result_0: [AB::VarMaybeExt; 4] = local.s0.value[0].0.clone().map(|v| v.into());
        let s0_result_1: [AB::VarMaybeExt; 4] = local.s0.value[1].0.clone().map(|v| v.into());
        xor_n_precompute_lc::<AB>(
            builder,
            &[s0_result_0.clone(), s0_result_1],
            &[a_rr_2_arr, s0_result_0.clone()],
            &[a_rr_13_arr, a_rr_22_arr],
        );

        // maj: (a & b) ^ (a & c) ^ (b & c) (N=3, so 2 intermediate results)
        let a_and_b_result_word = local.a_and_b.value[0].clone();
        let a_and_c_result_word = local.a_and_c.value[0].clone();
        let b_and_c_result_word = local.b_and_c.value[0].clone();
        let a_and_b_arr: [AB::VarMaybeExt; 4] = a_and_b_result_word.0.map(|v| v.into());
        let a_and_c_arr: [AB::VarMaybeExt; 4] = a_and_c_result_word.0.map(|v| v.into());
        let b_and_c_arr: [AB::VarMaybeExt; 4] = b_and_c_result_word.0.map(|v| v.into());
        let maj_result_0: [AB::VarMaybeExt; 4] = local.maj.value[0].0.clone().map(|v| v.into());
        let maj_result_1: [AB::VarMaybeExt; 4] = local.maj.value[1].0.clone().map(|v| v.into());
        xor_n_precompute_lc::<AB>(
            builder,
            &[maj_result_0.clone(), maj_result_1],
            &[a_and_b_arr, maj_result_0.clone()],
            &[a_and_c_arr, b_and_c_arr],
        );

        // AndN operations (N=2 for all, so 1 result each)
        // e & f
        let e_arr = compact_word_to_arr::<AB>(&local.e, &local.e_witness);
        let f_arr = compact_word_to_arr::<AB>(&local.f, &local.f_witness);
        let e_and_f_result: [AB::VarMaybeExt; 4] =
            local.e_and_f.value[0].0.clone().map(|v| v.into());
        and_n_precompute_lc::<AB>(builder, &[e_and_f_result], &[e_arr.clone()], &[f_arr]);

        // e_not & g
        let e_not_arr: [AB::VarMaybeExt; 4] = std::array::from_fn(|idx| {
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(u8::MAX as u32)) - e_arr[idx].clone()
        });
        let g_arr = compact_word_to_arr::<AB>(&local.g, &local.g_witness);
        let e_not_and_g_result: [AB::VarMaybeExt; 4] =
            local.e_not_and_g.value[0].0.clone().map(|v| v.into());
        and_n_precompute_lc::<AB>(builder, &[e_not_and_g_result], &[e_not_arr], &[g_arr]);

        // a & b
        let a_arr = compact_word_to_arr::<AB>(&local.a, &local.a_witness);
        let b_arr = compact_word_to_arr::<AB>(&local.b, &local.b_witness);
        let a_and_b_result: [AB::VarMaybeExt; 4] =
            local.a_and_b.value[0].0.clone().map(|v| v.into());
        and_n_precompute_lc::<AB>(builder, &[a_and_b_result], &[a_arr.clone()], &[b_arr]);

        // a & c
        let c_arr = compact_word_to_arr::<AB>(&local.c, &local.c_witness);
        let a_and_c_result: [AB::VarMaybeExt; 4] =
            local.a_and_c.value[0].0.clone().map(|v| v.into());
        and_n_precompute_lc::<AB>(builder, &[a_and_c_result], &[a_arr], &[c_arr.clone()]);

        // b & c
        let b_arr = compact_word_to_arr::<AB>(&local.b, &local.b_witness);
        let b_and_c_result: [AB::VarMaybeExt; 4] =
            local.b_and_c.value[0].0.clone().map(|v| v.into());
        and_n_precompute_lc::<AB>(builder, &[b_and_c_result], &[b_arr], &[c_arr]);

        // AddN operations
        // temp1 = h + s1 + ch + k + w (N=5)
        let h_compact = CompactWord([local.h.0[0].clone(), local.h.0[1].clone()]);
        let s1_final = word_to_compact::<AB>(&local.s1.value[1]);
        let ch_final = word_to_compact::<AB>(&local.ch.value[0]);
        let k_compact = CompactWord([local.k.0[0].clone(), local.k.0[1].clone()]);
        let w_compact = word_to_compact::<AB>(&local.w_access.access.value);
        add_n_precompute_lc::<AB>(
            builder,
            &[h_compact, s1_final, ch_final, k_compact, w_compact],
            CompactWord([local.temp1.value.0[0].clone(), local.temp1.value.0[1].clone()]),
        );

        // temp2 = s0 + maj (N=2)
        let s0_final = word_to_compact::<AB>(&local.s0.value[1]);
        let maj_final = word_to_compact::<AB>(&local.maj.value[1]);
        add_n_precompute_lc::<AB>(
            builder,
            &[s0_final, maj_final],
            CompactWord([local.temp2.value.0[0].clone(), local.temp2.value.0[1].clone()]),
        );

        // d_add_temp1 = d + temp1 (N=2)
        let d_compact = CompactWord([local.d.0[0].clone(), local.d.0[1].clone()]);
        let temp1_compact =
            CompactWord([local.temp1.value.0[0].clone(), local.temp1.value.0[1].clone()]);
        add_n_precompute_lc::<AB>(
            builder,
            &[d_compact, temp1_compact.clone()],
            CompactWord([
                local.d_add_temp1.value.0[0].clone(),
                local.d_add_temp1.value.0[1].clone(),
            ]),
        );

        // temp1_add_temp2 = temp1 + temp2 (N=2)
        let temp2_compact =
            CompactWord([local.temp2.value.0[0].clone(), local.temp2.value.0[1].clone()]);
        add_n_precompute_lc::<AB>(
            builder,
            &[temp1_compact, temp2_compact],
            CompactWord([
                local.temp1_add_temp2.value.0[0].clone(),
                local.temp1_add_temp2.value.0[1].clone(),
            ]),
        );
    }

    // ========================================================================
    // Phase 2: eval — gate constraints
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &ShaCompressCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const ShaCompressCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();
        let one = AB::one_maybe();

        // is_real boolean gate (replaces implicit BitVec enforcement)
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // ── air.rs L59-74: i reconstruction from one-hot ──
        let i_reconstructed: AB::VarMaybeExt = local
            .i_low_one_hot
            .iter()
            .enumerate()
            .map(|(idx, b)| {
                b.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(idx as u32))
            })
            .chain(local.i_high_one_hot.iter().enumerate().map(|(idx, b)| {
                b.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32((idx << 3) as u32))
            }))
            .fold(AB::zero_maybe(), |acc, x| acc + x);
        builder.when(is_real.clone()).assert_zero(local.i.clone() - i_reconstructed);

        // ── air.rs L140-158: K constant selection via one-hot ──
        let one_hot: [AB::VarMaybeExt; 64] = std::array::from_fn(|idx| {
            let low_idx = idx & 0x7;
            let high_idx = idx >> 3;
            local.i_low_one_hot[low_idx].clone() * local.i_high_one_hot[high_idx].clone()
        });

        // Verify k[0] (low 16 bits)
        let k_low: AB::VarMaybeExt = one_hot
            .iter()
            .zip(SHA_COMPRESS_K.iter())
            .map(|(b, k)| b.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(k & 0xFFFF)))
            .fold(AB::zero_maybe(), |acc, x| acc + x);
        builder.assert_zero(local.k.0[0].clone() - k_low);

        // Verify k[1] (high 16 bits)
        let k_high: AB::VarMaybeExt = one_hot
            .iter()
            .zip(SHA_COMPRESS_K.iter())
            .map(|(b, k)| b.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(k >> 16)))
            .fold(AB::zero_maybe(), |acc, x| acc + x);
        builder.assert_zero(local.k.0[1].clone() - k_high);

        // ── Memory timestamp constraints ──
        memory_timestamp_gate_constraints(
            builder,
            &local.w_access.access,
            local.shard.clone(),
            local.clk.clone(),
            is_real.clone(),
        );

        // ── Compression operations ──

        // S1 calculation: e_rr_6 ^ e_rr_11 ^ e_rr_25
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.e_rr_6,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            6,
        );
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.e_rr_11,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            11,
        );
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.e_rr_25,
            CompactWord([local.e.0[0].clone(), local.e.0[1].clone()]),
            25,
        );

        // S0 calculation: a_rr_2 ^ a_rr_13 ^ a_rr_22
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.a_rr_2,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            2,
        );
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.a_rr_13,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            13,
        );
        fixed_rotate_right_gate_constraints::<AB>(
            builder,
            &local.a_rr_22,
            CompactWord([local.a.0[0].clone(), local.a.0[1].clone()]),
            22,
        );

        // temp1 := h + S1 + ch + k[i] + w[i]
        let h_compact = CompactWord(local.h.0.clone());
        let s1_compact = word_to_compact::<AB>(&local.s1.value[local.s1.value.len() - 1]);
        let ch_compact = word_to_compact::<AB>(&local.ch.value[local.ch.value.len() - 1]);
        let k_compact = CompactWord(local.k.0.clone());
        let w_compact = word_to_compact::<AB>(&local.w_access.access.value);

        add_n_without_result_gate_constraints::<AB>(
            builder,
            &[h_compact, s1_compact, ch_compact, k_compact, w_compact],
            CompactWord(local.temp1.value.0.clone()),
            is_real.clone(),
        );

        // temp2 := S0 + maj
        let s0_compact = word_to_compact::<AB>(&local.s0.value[local.s0.value.len() - 1]);
        let maj_compact = word_to_compact::<AB>(&local.maj.value[local.maj.value.len() - 1]);

        add_n_without_result_gate_constraints::<AB>(
            builder,
            &[s0_compact, maj_compact],
            CompactWord(local.temp2.value.0.clone()),
            is_real.clone(),
        );

        // d_add_temp1 := d + temp1 (new e)
        let d_compact = CompactWord(local.d.0.clone());
        let temp1_compact = CompactWord(local.temp1.value.0.clone());

        add_n_without_result_gate_constraints::<AB>(
            builder,
            &[d_compact, temp1_compact.clone()],
            CompactWord(local.d_add_temp1.value.0.clone()),
            is_real.clone(),
        );

        // temp1_add_temp2 := temp1 + temp2 (new a)
        let temp2_compact = CompactWord(local.temp2.value.0.clone());

        add_n_without_result_gate_constraints::<AB>(
            builder,
            &[temp1_compact, temp2_compact],
            CompactWord(local.temp1_add_temp2.value.0.clone()),
            is_real.clone(),
        );
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &ShaCompressCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const ShaCompressCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // #1: recv(ShaCompress)
        builder.recv(is_real.clone());

        // #2: send(ShaCompress)
        builder.send(is_real.clone());

        // #3: memory_read
        memory_read_lookup(builder, is_real.clone());

        // #4-5: BitVec boolean packs for one-hot/sum (mult = is_real)
        bitvec_lookup(builder, is_real.clone());
        bitvec_lookup(builder, is_real.clone());

        // Operation lookups
        // 6 FixedRotateRight operations: e_rr_6, e_rr_11, e_rr_25, a_rr_2, a_rr_13, a_rr_22
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 6);
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 11);
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 25);
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 2);
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 13);
        fixed_rotate_right_lookup::<AB>(builder, is_real.clone(), 22);

        // 4 XorN operations: S1 (3 inputs), ch (2 inputs), S0 (3 inputs), maj (3 inputs)
        xor_n_lookup::<AB>(builder, is_real.clone(), 3); // S1
        xor_n_lookup::<AB>(builder, is_real.clone(), 2); // ch
        xor_n_lookup::<AB>(builder, is_real.clone(), 3); // S0
        xor_n_lookup::<AB>(builder, is_real.clone(), 3); // maj

        // 5 AndN operations: e&f, e_not&g, a&b, a&c, b&c (all 2 inputs)
        and_n_lookup::<AB>(builder, is_real.clone(), 2);
        and_n_lookup::<AB>(builder, is_real.clone(), 2);
        and_n_lookup::<AB>(builder, is_real.clone(), 2);
        and_n_lookup::<AB>(builder, is_real.clone(), 2);
        and_n_lookup::<AB>(builder, is_real.clone(), 2);

        // 4 AddN operations: temp1 (5 inputs), temp2 (2 inputs), d_add_temp1 (2 inputs),
        // temp1_add_temp2 (2 inputs)
        add_n_lookup::<AB>(builder, is_real.clone(), 5);
        add_n_lookup::<AB>(builder, is_real.clone(), 2);
        add_n_lookup::<AB>(builder, is_real.clone(), 2);
        add_n_lookup::<AB>(builder, is_real.clone(), 2);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::ShaCompressChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for ShaCompressPolyAir {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_COLS
    }
}

impl<F: Field> MachineAir<F> for ShaCompressPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <ShaCompressChip as MachineAir<F>>::name(&ShaCompressChip {}) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShaCompressChip {}.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use dt_core_executor::{
            events::{ByteLookupEvent, ByteRecord},
            syscalls::SyscallCode,
        };
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        // [1] Base chip: memory read BLUs + all operation BLUs
        <ShaCompressChip as MachineAir<F>>::generate_dependencies(
            &ShaCompressChip {},
            input,
            output,
        );

        // [2] PolyAir-only: 2 BitVec BLUs per real row (mult = is_real)
        //
        // BitVec #1 payload: i_low_one_hot[8] ++ i_high_one_hot[8]
        //   For round i: bit (i & 7) and bit (8 + (i >> 3)) are set.
        //   value = (1 << (i & 7)) | (1 << (8 + (i >> 3)))
        //
        // BitVec #2 payload: [low_sum, high_sum]
        //   Both sums are always 1 for real rows → value = 0b11 = 3.
        let events = input.get_precompile_events(SyscallCode::SHA_COMPRESS);
        if events.is_empty() {
            return;
        }

        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);
        let blu_batches = events
            .chunks(chunk_size)
            .par_bridge()
            .map(|chunk| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for _ in chunk {
                    for i in 0u16..64 {
                        blu.add_bit_vec_lookup((1u16 << (i & 7)) | (1u16 << (8 + (i >> 3))));
                        blu.add_bit_vec_lookup(3u16);
                    }
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShaCompressChip as MachineAir<F>>::included(&ShaCompressChip {}, shard)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        programs::tests::sha_compress_program,
        syscall::precompiles::sha256::compress_dt::ShaCompressChip,
    };
    use dt_core_executor::{syscalls::SyscallCode, ExecutionRecord, Executor};
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

    fn beta_powers_for(air: &ShaCompressPolyAir, beta: EF) -> Vec<EF> {
        let max = <ShaCompressPolyAir as FullAir<
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
        air: &ShaCompressPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <ShaCompressPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let empty_prep: Vec<F> = vec![];
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            let main_row: &[F] = core::ops::Deref::deref(&main_binding);
            let reserved = collect_reserved_poly(main_row, &empty_prep, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = sha_compress_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty() {
                continue;
            }

            let mut sub_shard = ExecutionRecord::new(shard.program.clone());
            sub_shard.precompile_events = shard.precompile_events.clone();

            let chip = ShaCompressChip::new();
            return Some(
                chip.generate_trace(&sub_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_sha_compress_polyair_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No ShaCompress trace found — skipping test");
                return;
            }
        };

        let air = ShaCompressPolyAir::new();
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

        // `precompute_linear_combination` and `generate_permutation_trace_` both expect
        // per-row capacities, not whole-trace totals.
        let num_lookups = NUM_LOOKUPS_PER_ROW;
        let num_precomputed = NUM_LOOKUPS_PER_ROW;

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
            num_precomputed,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &precomputed_full,
            alpha,
            &bp,
            BATCH_SIZE,
            num_lookups,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        let num_gate_constraints = 1 + 2 + 3 + 6 + 1; // +1 for is_real bool gate
        let num_reducer = num_gate_constraints + NUM_LOOKUPS_PER_ROW.div_ceil(BATCH_SIZE) + 3;
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
            "ShaCompress first_round non-zero at indices: {:?}",
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
            "ShaCompress nonfirst_round non-zero at indices: {:?}",
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
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::{syscalls::SyscallCode, ByteOpcode, ExecutionRecord};

        let program = sha_compress_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        let (sha_shard, num_events) = runtime
            .records
            .iter()
            .find_map(|r| {
                let shard = r.as_ref();
                let events = shard.get_precompile_events(SyscallCode::SHA_COMPRESS);
                if events.is_empty() {
                    return None;
                }
                let n = events.len();
                let mut s = ExecutionRecord::new(shard.program.clone());
                s.precompile_events = shard.precompile_events.clone();
                Some((s, n))
            })
            .expect("no sha compress events in test fixture");

        assert!(num_events > 0, "fixture must contain sha compress events");

        let mut deps = ExecutionRecord::default();
        <ShaCompressPolyAir as MachineAir<F>>::generate_dependencies(
            &ShaCompressPolyAir,
            &sha_shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        // 2 BitVec BLUs per round × 64 rounds per event × num_events
        let expected = num_events * 64 * 2;
        assert_eq!(bitvec_total, expected, "BitVec BLU count must match send count");
    }

    fn random_sha_compress_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace().expect("sample trace should exist");
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
        let air = ShaCompressPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SHA_COMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sha_compress_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let bp = beta_powers_for(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let num_lookups = NUM_LOOKUPS_PER_ROW;
        let num_precomputed = NUM_LOOKUPS_PER_ROW;
        let num_gate_constraints = 1 + 2 + 3 + 6 + 1; // +1 for is_real bool gate
        let num_reducer = num_gate_constraints + num_lookups.div_ceil(BATCH_SIZE) + 3;
        let mut reducer_rng = StdRng::seed_from_u64(seed.wrapping_add(3000));
        let constraint_reducer: Vec<EF> =
            (0..num_reducer).map(|_| random_ef(&mut reducer_rng)).collect();
        let global = EF::zero();
        let reserved_poly_desc = <ShaCompressPolyAir as FullAir<
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
            beta_septix,
            num_precomputed,
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
            num_lookups,
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
                &bp,
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
impl ShaCompressPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS_PER_ROW
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_LOOKUPS_PER_ROW
    }
}
