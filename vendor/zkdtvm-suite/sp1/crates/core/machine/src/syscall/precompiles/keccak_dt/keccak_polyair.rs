//! PolyAir adaptation of KeccakPermuteChip.
//!
//! The KeccakPermute chip computes one round of the Keccak-f[1600] permutation:
//!   - Receives current state via Keccak interaction
//!   - Applies θ, ρ, π, χ, ι steps with gate constraints
//!   - Sends next-round state via Keccak interaction
//!   - XOR operations for round-constant application (XorNOperation<U2> × 2)
//!
//! ## Interaction Summary (131 total)
//!
//!   #1:       recv(Keccak, is_real)  — [shard, clk, step, a[50 compact]]
//!   #2:       send(Keccak, is_real)  — [shard, clk, step+1, output[50 compact]]
//!   #3-6:     XorN #0 (a'''[0][0] low)  — 4 XOR byte lookups
//!   #7-10:    XorN #1 (a'''[0][0] high) — 4 XOR byte lookups
//!   #11-131:  121 BitVec lookups for 1932 booleans (mult = is_real)
//!
//! ## Gate Constraints
//!
//!   - is_real boolean gate
//!   - step decomposition (when is_real)
//!   - c_prime definition: 320 constraints
//!   - a ↔ a_prime bit-packing: 100 constraints
//!   - c_prime ↔ a_prime consistency: 320 cubic constraints
//!   - a_prime_prime computation: 100 constraints
//!   - RC round constant: 4 constraints

use std::ops::Deref;

use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::keccak_cols::{PI, RHO};
use crate::{
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    operations_dt::{compact_word_to_arr, xor_n_lookup, xor_n_precompute_lc},
};

use super::{
    columns::{KeccakPermuteCols, NUM_KECCAK_PERMUTE_COLS},
    keccak_cols::R,
};

// ============================================================================
// Constants
// ============================================================================

/// Number of booleans enforced via BitVec lookups.
/// step_low_one_hot(4) + step_high_one_hot(6) + sum_low(1) + sum_high(1)
/// + c(320) + a_prime(1600) = 1932  (is_real moved to explicit assert_bool gate)
const NUM_BOOLEANS: usize = 4 + 6 + 1 + 1 + 5 * 64 + 5 * 5 * 64;

/// Number of BitVec interactions: ceil(1932 / 16) = 121.
const NUM_BITVEC: usize = NUM_BOOLEANS.div_ceil(16);

/// XorNOperation<U2> produces 4 XOR byte lookups per invocation; 2 invocations.
const NUM_XOR_INTERACTIONS: usize = 2 * 4;

/// Total lookup interactions: 2 Keccak + 8 XOR + 121 BitVec = 131.
const NUM_LOOKUPS: usize = 2 + NUM_XOR_INTERACTIONS + NUM_BITVEC;

/// Max payload size across all interactions.
/// Keccak recv/send: 3 header + 25×2×2 compact words = 103 values.
const MAX_LOOKUP_VALUES: usize = 3 + 25 * 2 * 2;

/// Precomputed linear combinations (one per lookup).
pub(crate) const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for KeccakPermuteChip.
#[derive(Clone, Copy, Default)]
pub struct KeccakPermutePolyAir;

impl KeccakPermutePolyAir {
    pub const fn new() -> Self {
        Self
    }
}

// ============================================================================
// Helpers
// ============================================================================

impl<AB: FullAirBuilder> FullAir<AB> for KeccakPermutePolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Reserve all main trace columns — every column is needed for the
        // massive gate constraints and Keccak payload construction.
        (0..NUM_KECCAK_PERMUTE_COLS).map(PairCol::Main).collect()
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &KeccakPermuteCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let keccak_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Keccak as usize));

        let keccak = &local.keccak;

        // =================================================================
        // #1: recv(Keccak, is_real) — current state
        //     payload: [shard, clk, step, a[0..24][0..1] as compact words]
        //     (air.rs L44-57)
        // =================================================================
        let mut recv_values: Vec<AB::VarMaybeExt> = Vec::with_capacity(MAX_LOOKUP_VALUES);
        recv_values.push(local.shard.clone());
        recv_values.push(local.clk.clone());
        recv_values.push(keccak.step.clone());
        for a_row in keccak.a.as_flattened() {
            for compact in a_row {
                recv_values.push(compact.0[0].clone());
                recv_values.push(compact.0[1].clone());
            }
        }
        builder.retain_precomputed(builder.lookup_denominator(keccak_kind.clone(), recv_values));

        // =================================================================
        // #2: send(Keccak, is_real) — next-round state
        //     payload: [shard, clk, step+1, a'''[0][0], a''[rest]]
        //     (air.rs L59-76)
        // =================================================================
        let mut send_values: Vec<AB::VarMaybeExt> = Vec::with_capacity(MAX_LOOKUP_VALUES);
        send_values.push(local.shard.clone());
        send_values.push(local.clk.clone());
        send_values.push(keccak.step.clone() + AB::VarMaybeExt::from(AB::F::one()));

        // a'''[0][0] — the XorN result (2 compact words = 4 limbs)
        // We need to build this from a_prime_prime_prime_0_0.value[0]
        // XorNOperation<U2> has value: [Word<T>; 1] (N-1=1 intermediate)
        // The result word is cols.a_prime_prime_prime_0_0[half].value[0]
        for half in 0..2 {
            let result_word = &keccak.a_prime_prime_prime_0_0[half].value[0];
            let byte_shift = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
            // Compact: [low16, high16] = [byte0 + byte1*256, byte2 + byte3*256]
            send_values.push(result_word[0].clone() + result_word[1].clone() * byte_shift.clone());
            send_values.push(result_word[2].clone() + result_word[3].clone() * byte_shift);
        }

        // a''[rest] — a_prime_prime[0..4][0..4] skipping [0][0]
        for compact in keccak.a_prime_prime.as_flattened()[1..].as_flattened() {
            send_values.push(compact.0[0].clone());
            send_values.push(compact.0[1].clone());
        }
        builder.retain_precomputed(builder.lookup_denominator(keccak_kind, send_values));

        // =================================================================
        // #3-6: XorN #0 — a_prime_prime_prime_0_0[0]
        //   XorNOperation<U2>(a''[0][0] low word, rc low word) → 4 XOR lookups
        //   (keccak_cols.rs L260-268)
        // =================================================================
        let a_pp_0_0_low = compact_word_to_arr::<AB>(
            &keccak.a_prime_prime[0][0][0],
            &keccak.a_prime_prime_0_0_witness[0],
        );
        let rc_low = compact_word_to_arr::<AB>(&keccak.rc[0], &keccak.rc_witness[0]);
        let result_low: [AB::VarMaybeExt; 4] = keccak.a_prime_prime_prime_0_0[0].value[0].0.clone();
        xor_n_precompute_lc::<AB>(builder, &[result_low], &[a_pp_0_0_low], &[rc_low]);

        // =================================================================
        // #7-10: XorN #1 — a_prime_prime_prime_0_0[1]
        //   XorNOperation<U2>(a''[0][0] high word, rc high word) → 4 XOR lookups
        //   (keccak_cols.rs L269-277)
        // =================================================================
        let a_pp_0_0_high = compact_word_to_arr::<AB>(
            &keccak.a_prime_prime[0][0][1],
            &keccak.a_prime_prime_0_0_witness[1],
        );
        let rc_high = compact_word_to_arr::<AB>(&keccak.rc[1], &keccak.rc_witness[1]);
        let result_high: [AB::VarMaybeExt; 4] =
            keccak.a_prime_prime_prime_0_0[1].value[0].0.clone();
        xor_n_precompute_lc::<AB>(builder, &[result_high], &[a_pp_0_0_high], &[rc_high]);

        // =================================================================
        // #11-131: 121 BitVec lookups for 1933 booleans
        //   All unconditional (fire on every row).
        //   (keccak_cols.rs L141-172)
        // =================================================================
        let mut all_booleans: Vec<AB::VarMaybeExt> = Vec::with_capacity(NUM_BOOLEANS);

        // step_low_one_hot[4]
        all_booleans.extend_from_slice(&keccak.step_low_one_hot);

        // step_high_one_hot[6]
        all_booleans.extend_from_slice(&keccak.step_high_one_hot);

        // sum(step_low_one_hot) — must be boolean
        let sum_low: AB::VarMaybeExt = keccak
            .step_low_one_hot
            .iter()
            .cloned()
            .fold(AB::VarMaybeExt::from(AB::F::zero()), |acc, x| acc + x);
        all_booleans.push(sum_low);

        // sum(step_high_one_hot) — must be boolean
        let sum_high: AB::VarMaybeExt = keccak
            .step_high_one_hot
            .iter()
            .cloned()
            .fold(AB::VarMaybeExt::from(AB::F::zero()), |acc, x| acc + x);
        all_booleans.push(sum_high);

        // c[5][64] = 320 bits
        for c_row in &keccak.c {
            all_booleans.extend_from_slice(c_row);
        }

        // a_prime[5][5][64] = 1600 bits
        for a_prime_plane in &keccak.a_prime {
            for a_prime_row in a_prime_plane {
                all_booleans.extend_from_slice(a_prime_row);
            }
        }

        debug_assert_eq!(all_booleans.len(), NUM_BOOLEANS);

        // Pack into BitVec chunks of 16
        for chunk in all_booleans.chunks(16) {
            bitvec_precompute_lc(builder, chunk.to_vec());
        }
    }

    // ========================================================================
    // Phase 2: gate constraints (reserved_poly columns only)
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &KeccakPermuteCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const KeccakPermuteCols<AB::VarMaybeExt>)
        };

        let keccak = &local.keccak;
        let is_real = local.is_real.clone();
        let one = AB::one_maybe();
        let two = AB::VarMaybeExt::from(AB::F::two());
        let four = AB::VarMaybeExt::from(AB::F::from_canonical_u32(4));

        let xor =
            |a: AB::VarMaybeExt, b: AB::VarMaybeExt| a.clone() + b.clone() - a * b * two.clone();
        let and = |a: AB::VarMaybeExt, b: AB::VarMaybeExt| a * b;
        let not = |a: AB::VarMaybeExt| one.clone() - a;

        // is_real boolean gate (replaces implicit BitVec enforcement)
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // ── keccak_cols.rs L146-161: step decomposition ──
        // when(is_real): step = sum(step_low * i) + sum(step_high * (i<<2))
        let step_sum: AB::VarMaybeExt = {
            let low_sum = keccak.step_low_one_hot.iter().enumerate().fold(
                AB::VarMaybeExt::from(AB::F::zero()),
                |acc, (i, b)| {
                    acc + b.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(i as u32))
                },
            );
            let high_sum = keccak.step_high_one_hot.iter().enumerate().fold(
                AB::VarMaybeExt::from(AB::F::zero()),
                |acc, (i, b)| {
                    acc + b.clone() *
                        AB::VarMaybeExt::from(AB::F::from_canonical_u32((i << 2) as u32))
                },
            );
            low_sum + high_sum
        };
        builder.assert_zero(is_real * (keccak.step.clone() - step_sum));

        // ── keccak_cols.rs L175-185: c_prime definition ──
        // c_prime[i][j] = xor(xor(c[i][j], c[(i+4)%5][j]), c[(i+1)%5][(j+63)%64])
        for i in 0..5 {
            for j in 0..64 {
                let expected = xor(
                    xor.clone()(keccak.c[i][j].clone(), keccak.c[(i + 4) % 5][j].clone()),
                    keccak.c[(i + 1) % 5][(j + 63) % 64].clone(),
                );
                builder.assert_zero(keccak.c_prime[i][j].clone() - expected);
            }
        }

        // ── keccak_cols.rs L187-204: a ↔ a_prime bit-packing ──
        // For each (i, j): bit-pack a_prime[j][i] XOR (c[i] XOR c_prime[i])
        //   into 16-bit chunks matching a[j][i] compact words.
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..4 {
                    // Compute the 16-bit sum from bits
                    let start = k * 16;
                    let end = (k + 1) * 16;
                    let mut bit_sum = AB::VarMaybeExt::from(AB::F::zero());
                    for bit_idx in (start..end).rev() {
                        let c_xor_c_prime =
                            xor(keccak.c[i][bit_idx].clone(), keccak.c_prime[i][bit_idx].clone());
                        let a_bit =
                            xor.clone()(c_xor_c_prime, keccak.a_prime[j][i][bit_idx].clone());
                        bit_sum = bit_sum.clone() + bit_sum + a_bit;
                    }
                    builder.assert_zero(bit_sum - keccak.a[j][i][k >> 1][k & 0x1].clone());
                }
            }
        }

        // ── keccak_cols.rs L206-215: c_prime ↔ a_prime consistency ──
        // sum(a_prime[k][i][j] for k in 0..5) - c_prime[i][j] ∈ {0, 2, 4}
        // Enforced as: diff * (diff - 2) * (diff - 4) = 0
        for i in 0..5 {
            for j in 0..64 {
                let sum: AB::VarMaybeExt = (0..5)
                    .map(|k| keccak.a_prime[k][i][j].clone())
                    .fold(AB::VarMaybeExt::from(AB::F::zero()), |acc, x| acc + x);
                let diff = sum - keccak.c_prime[i][j].clone();
                builder.assert_zero(
                    diff.clone() * (diff.clone() - two.clone()) * (diff - four.clone()),
                );
            }
        }

        // ── keccak_cols.rs L217-238: a_prime_prime computation ──
        // a''[i][j] = xor(b[j], and(not(b[(j+1)%5]), b[(j+2)%5]))
        // where b(i, j)[k] = a_prime[j][(j+3*i)%5][(k + 64 - R[j][(j+3*i)%5]) % 64]
        for i in 0..5 {
            for j in 0..5 {
                let row_b = j;
                let col_b = (j + 3 * i) % 5;
                let rot_b = R[row_b][col_b];

                for k in 0..4 {
                    let start = k * 16;
                    let end = (k + 1) * 16;
                    let mut bit_sum = AB::VarMaybeExt::from(AB::F::zero());
                    for bit_idx in (start..end).rev() {
                        // b[j][bit_idx]
                        let bj = keccak.a_prime[row_b][col_b][(bit_idx + 64 - rot_b) % 64].clone();

                        // b[(j+1)%5][bit_idx]
                        let j1 = (j + 1) % 5;
                        let row_b1 = j1;
                        let col_b1 = (j1 + 3 * i) % 5;
                        let rot_b1 = R[row_b1][col_b1];
                        let bj1 =
                            keccak.a_prime[row_b1][col_b1][(bit_idx + 64 - rot_b1) % 64].clone();

                        // b[(j+2)%5][bit_idx]
                        let j2 = (j + 2) % 5;
                        let row_b2 = j2;
                        let col_b2 = (j2 + 3 * i) % 5;
                        let rot_b2 = R[row_b2][col_b2];
                        let bj2 =
                            keccak.a_prime[row_b2][col_b2][(bit_idx + 64 - rot_b2) % 64].clone();

                        let a_pp_bit = xor.clone()(bj, and(not(bj1), bj2));
                        bit_sum = bit_sum.clone() + bit_sum + a_pp_bit;
                    }
                    builder
                        .assert_zero(bit_sum - keccak.a_prime_prime[i][j][k >> 1][k & 0x1].clone());
                }
            }
        }

        // ── keccak_cols.rs L248-258: RC round constant ──
        // rc[i>>1][i&1] = sum(one_hot[step] * RC[step]_limb_i) for step in 0..24
        for limb_idx in 0..4 {
            let mut rc_sum = AB::VarMaybeExt::from(AB::F::zero());
            for step in 0..24 {
                let one_hot = keccak.step_low_one_hot[step & 0x3].clone() *
                    keccak.step_high_one_hot[step >> 2].clone();
                let rc_limb = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    ((RC[step] >> (16 * limb_idx)) & 0xFFFFu64) as u32,
                ));
                rc_sum = rc_sum + one_hot * rc_limb;
            }
            builder.assert_zero(rc_sum - keccak.rc[limb_idx >> 1][limb_idx & 0x1].clone());
        }
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &KeccakPermuteCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const KeccakPermuteCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // #1: recv(Keccak) — current state
        builder.recv(is_real.clone());

        // #2: send(Keccak) — next-round state
        builder.send(is_real.clone());

        // #3-6: XorN #0 (4 XOR byte lookups)
        xor_n_lookup(builder, is_real.clone(), 2);

        // #7-10: XorN #1 (4 XOR byte lookups)
        xor_n_lookup(builder, is_real.clone(), 2);

        // #11-131: 121 BitVec lookups (mult = is_real — skip padding rows)
        for _ in 0..NUM_BITVEC {
            bitvec_lookup(builder, is_real.clone());
        }
    }
}

// ============================================================================
// BitVec helper
// ============================================================================

/// Compute the 121 packed u16 BitVec values for one keccak round (is_real excluded from payload),
/// then advance `state` through the full keccak round in-place.
///
/// Payload bit order matches `precompute_lc` (is_real removed from front):
///   step_low_one_hot[4] | step_high_one_hot[6] | sum_low | sum_high | c[5][64] | a_prime[5][5][64]
fn keccak_round_bitvec_values(
    state: &mut [u64; super::STATE_SIZE],
    step: usize,
) -> [u16; NUM_BITVEC] {
    // ── θ: column parity ──
    let mut c = [0u64; 5];
    for i in 0..5 {
        for j in 0..5 {
            c[j] ^= state[i * 5 + j];
        }
    }
    // d[j] = c[(j+4)%5] ^ rot(c[(j+1)%5], 1)
    let mut d = [0u64; 5];
    for j in 0..5 {
        d[j] = c[(j + 4) % 5] ^ c[(j + 1) % 5].rotate_left(1);
    }

    // ── Pack boolean payload (1932 bits) ──
    let mut out = [0u16; NUM_BITVEC];
    let mut pos = 0usize;

    macro_rules! push {
        ($b:expr) => {{
            if $b {
                out[pos >> 4] |= 1u16 << (pos & 15);
            }
            pos += 1;
        }};
    }

    // step_low_one_hot[4]
    for i in 0..4 {
        push!((step & 3) == i);
    }
    // step_high_one_hot[6]
    for i in 0..6 {
        push!((step >> 2) == i);
    }
    // sum_low = 1, sum_high = 1 (always true for real rows)
    push!(true);
    push!(true);
    // c[5][64]
    for i in 0..5 {
        for j in 0..64u32 {
            push!((c[i] >> j) & 1 != 0);
        }
    }
    // a_prime[5][5][64] = (state[i*5+j] ^ d[j]) bits
    for i in 0..5 {
        for j in 0..5 {
            let a_prime = state[i * 5 + j] ^ d[j];
            for k in 0..64u32 {
                push!((a_prime >> k) & 1 != 0);
            }
        }
    }
    debug_assert_eq!(pos, NUM_BOOLEANS);

    // ── Advance state through one keccak round ──
    // θ
    for i in 0..5 {
        for j in 0..5 {
            state[i * 5 + j] ^= d[j];
        }
    }
    // ρ+π
    let mut last = state[1];
    for i in 0..24 {
        let temp = state[PI[i]];
        state[PI[i]] = last.rotate_left(RHO[i]);
        last = temp;
    }
    // χ
    for i in 0..5 {
        let arr: [u64; 5] = std::array::from_fn(|j| state[i * 5 + j]);
        for j in 0..5 {
            state[i * 5 + j] ^= (!arr[(j + 1) % 5]) & arr[(j + 2) % 5];
        }
    }
    // ι
    state[0] ^= RC[step];

    out
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::KeccakPermuteChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for KeccakPermutePolyAir {
    fn width(&self) -> usize {
        NUM_KECCAK_PERMUTE_COLS
    }
}

impl<F: Field> MachineAir<F> for KeccakPermutePolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <KeccakPermuteChip as MachineAir<F>>::name(&KeccakPermuteChip {}) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        KeccakPermuteChip {}.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use dt_core_executor::{
            events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
            syscalls::SyscallCode,
        };
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        // [1] Base chip: XOR BLUs (from XorNOperation on the RC application)
        <KeccakPermuteChip as MachineAir<F>>::generate_dependencies(
            &KeccakPermuteChip {},
            input,
            output,
        );

        // [2] PolyAir-only: 121 BitVec BLUs per real row (mult = is_real)
        let events = input.get_precompile_events(SyscallCode::KECCAK_PERMUTE);
        if events.is_empty() {
            return;
        }

        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);
        let blu_batches = events
            .chunks(chunk_size)
            .par_bridge()
            .map(|chunk| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (_, event) in chunk {
                    let event = if let PrecompileEvent::KeccakPermute(event) = event {
                        event
                    } else {
                        unreachable!()
                    };
                    let mut state = event.pre_state;
                    for step in 0..24 {
                        for &value in keccak_round_bitvec_values(&mut state, step).iter() {
                            blu.add_bit_vec_lookup(value);
                        }
                    }
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <KeccakPermuteChip as MachineAir<F>>::included(&KeccakPermuteChip {}, shard)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 3;

    use crate::{
        programs::tests::keccak_program, syscall::precompiles::keccak_dt::KeccakPermuteChip,
    };
    use dt_core_executor::{ExecutionRecord, Executor};
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
    use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers(air: &KeccakPermutePolyAir) -> Vec<EF> {
        let beta = challenge_beta();
        let max = <KeccakPermutePolyAir as FullAir<
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
        air: &KeccakPermutePolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <KeccakPermutePolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(
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

    fn reducer() -> Vec<EF> {
        // Gate constraints: is_real_bool(1) + step(1) + c_prime(320) + a_packing(100)
        //   + consistency(320) + a_pp(100) + RC(4) = 846
        // Lookup batch: ceil(131/3) = 44
        // Cumulative sum: 3
        // Total: 846 + 44 + 3 = 893
        let num = 846 + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..num as u32).map(|i| ef(i + 1)).collect()
    }

    /// Build a real trace from the keccak test program.
    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();
            if shard.precompile_events.is_keccak_empty() {
                continue;
            }

            let mut keccak_shard = ExecutionRecord::new(shard.program.clone());
            keccak_shard.precompile_events = shard.precompile_events.clone();

            let chip = KeccakPermuteChip::new();
            return Some(
                chip.generate_trace(&keccak_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_keccak_permute_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No KeccakPermute trace found -- skipping test");
                return;
            }
        };

        let air = KeccakPermutePolyAir::new();
        let height = main.height();
        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers(&air);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];

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

        let precomputed = {
            let width = precomputed_full.width();
            RowMajorMatrix::new(precomputed_full.values[..height * width].to_vec(), width)
        };

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

        let permutation = {
            let width = permutation_full.width();
            RowMajorMatrix::new(permutation_full.values[..height * width].to_vec(), width)
        };

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
        assert!(
            first.iter().all(|x| x.is_zero()),
            "first_round non-zero at indices: {:?}",
            first
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
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
                .take(5)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::{syscalls::SyscallCode, ByteOpcode, ExecutionRecord};

        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        let (keccak_shard, num_events) = runtime
            .records
            .iter()
            .find_map(|r| {
                let shard = r.as_ref();
                let events = shard.get_precompile_events(SyscallCode::KECCAK_PERMUTE);
                if events.is_empty() {
                    return None;
                }
                let n = events.len();
                let mut s = ExecutionRecord::new(shard.program.clone());
                s.precompile_events = shard.precompile_events.clone();
                Some((s, n))
            })
            .expect("no keccak events in test fixture");

        assert!(num_events > 0, "fixture must contain keccak events");

        let mut deps = ExecutionRecord::default();
        <KeccakPermutePolyAir as MachineAir<F>>::generate_dependencies(
            &KeccakPermutePolyAir,
            &keccak_shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = num_events * 24 * NUM_BITVEC;
        assert_eq!(bitvec_total, expected, "BitVec BLU count must match send count");
    }

    fn random_keccak_permute_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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
        let air = KeccakPermutePolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::KECCAK_PERMUTE_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_keccak_permute_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers(&air);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc = <KeccakPermutePolyAir as FullAir<
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
            NUM_PRECOMPUTED,
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
            NUM_LOOKUPS,
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
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl KeccakPermutePolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
