//! PolyAir adaptation of MemoryGlobalChip (Init/Finalize).
//!
//! The MemoryGlobalChip is a **sending** side chip that initializes or finalizes
//! memory values. It sends global interactions and address-chain interactions,
//! plus byte lookups for range checks and address ordering.
//!
//! ## Interaction Summary (12 total)
//!
//! From `MemoryGlobalChip::eval()` execution order:
//!  1. send(Byte/U8Range, is_real)       — addr_word[0..1]
//!  2. send(Byte/U8Range, is_real)       — addr_word[2..3]
//!  3. send(Byte/LTU, is_real)           — field range: is_addr_lt_threshold, addr_word[3],
//!     FIELD_ADDR_MSB_THRESHOLD
//!  4. send(Byte/U8Range, lt_mult)       — prev_addr_word[0..1]
//!  5. send(Byte/U8Range, lt_mult)       — prev_addr_word[2..3]
//!  6. send(Byte/U8Range, is_real)       — value[0..1]
//!  7. send(Byte/U8Range, is_real)       — value[2..3]
//!  8. send(Global, is_real)             — memory init/finalize interaction
//!  9. send(MemoryGlobalAddr, is_real)   — addr chain send
//! 10. recv(MemoryGlobalAddr, is_real)   — prev_addr chain recv
//! 11. send(Byte/LTU, lt_mult)          — lt_cols comparison byte
//! 12. send(BitVec, is_real)            — boolean enforcement (is_real, is_addr_zero,
//!     is_addr_lt_threshold, byte_flags[0..3])

use std::ops::Deref;

use dt_core_executor::{events::ByteRecord, ExecutionRecord, Program};
use dt_stark::{
    air::{FullAir, FullAirBuilder, InteractionScope, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{MemoryChipType, MemoryGlobalChip, FIELD_ADDR_MSB_THRESHOLD, NUM_MEMORY_INIT_COLS};
use crate::{
    bytes::polyair::{
        bitvec_lookup, bitvec_precompute_lc, ltu_lookup, ltu_precompute_lc,
        u8_range_pair_precompute_lc,
    },
    operations::assert_lt_bytes_gate_constraints,
};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
const NUM_LOOKUPS: usize = 12;

/// Maximum number of values in a single lookup payload.
/// BitVec lookups pad to 16 values internally (via `bits.resize(16, zero)`),
/// which is larger than the Global interaction's 10 values.
const MAX_LOOKUP_VALUES: usize = 16;

// Column indices into MemoryInitCols for reserved_poly.
// MemoryInitCols layout (24 columns):
//  0: shard
//  1: timestamp
//  2: addr
//  3..6: addr_word[0..3]
//  7..10: prev_addr_word[0..3]
//  11..14: lt_cols.byte_flags[0..3]
//  15: lt_cols.a_comparison_byte
//  16: lt_cols.b_comparison_byte
//  17..20: value[0..3]
//  21: is_real
//  22: is_addr_zero
//  23: is_addr_lt_threshold

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for MemoryGlobalChip.
///
/// Parameterized by `kind` to distinguish Initialize vs Finalize behavior,
/// which affects the Global interaction payload and the MemoryGlobalAddr discriminant.
#[derive(Clone, Copy)]
pub struct MemoryGlobalChipPolyAir {
    pub kind: MemoryChipType,
}

impl MemoryGlobalChipPolyAir {
    pub const fn new(kind: MemoryChipType) -> Self {
        Self { kind }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for MemoryGlobalChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEMORY_INIT_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(2),  // addr
            PairCol::Main(3),  // addr_word[0]
            PairCol::Main(4),  // addr_word[1]
            PairCol::Main(5),  // addr_word[2]
            PairCol::Main(6),  // addr_word[3]
            PairCol::Main(7),  // prev_addr_word[0]
            PairCol::Main(8),  // prev_addr_word[1]
            PairCol::Main(9),  // prev_addr_word[2]
            PairCol::Main(10), // prev_addr_word[3]
            PairCol::Main(11), // byte_flags[0]
            PairCol::Main(12), // byte_flags[1]
            PairCol::Main(13), // byte_flags[2]
            PairCol::Main(14), // byte_flags[3]
            PairCol::Main(15), // a_comparison_byte
            PairCol::Main(16), // b_comparison_byte
            PairCol::Main(17), // value[0]
            PairCol::Main(18), // value[1]
            PairCol::Main(19), // value[2]
            PairCol::Main(20), // value[3]
            PairCol::Main(21), // is_real
            PairCol::Main(22), // is_addr_zero
            PairCol::Main(23), // is_addr_lt_threshold
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();

        // Access main trace columns by index via raw pointer arithmetic.
        // MemoryInitCols column indices:
        //  0: shard, 1: timestamp, 2: addr
        //  3..6: addr_word[0..3], 7..10: prev_addr_word[0..3]
        //  11..14: lt_cols.byte_flags[0..3], 15: a_comparison_byte, 16: b_comparison_byte
        //  17..20: value[0..3], 21: is_real, 22: is_addr_zero, 23: is_addr_lt_threshold
        let ptr = main.as_ptr();
        let col = |i: usize| -> AB::VarMaybeExt { unsafe { (*ptr.add(i)).clone() } };

        let zero = AB::zero_maybe();

        let shard = col(0);
        let timestamp = col(1);
        let addr = col(2);
        let addr_word = [col(3), col(4), col(5), col(6)];
        let prev_addr_word = [col(7), col(8), col(9), col(10)];
        let a_comparison_byte = col(15);
        let b_comparison_byte = col(16);
        let value = [col(17), col(18), col(19), col(20)];
        let is_real = col(21);
        let is_addr_zero = col(22);
        let is_addr_lt_threshold = col(23);

        // =====================================================================
        // #1: U8Range — addr_word[0..1]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, addr_word[0].clone(), addr_word[1].clone());

        // =====================================================================
        // #2: U8Range — addr_word[2..3]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, addr_word[2].clone(), addr_word[3].clone());

        // =====================================================================
        // #3: LTU — field range check: LTU(is_addr_lt_threshold, 0, addr_word[3],
        // FIELD_ADDR_MSB_THRESHOLD) Payload: [LTU, is_addr_lt_threshold, 0, addr_word[3],
        // FIELD_ADDR_MSB_THRESHOLD]
        // =====================================================================
        {
            let byte_kind =
                AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
            let ltu_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(
                dt_core_executor::ByteOpcode::LTU as u8,
            ));
            let threshold =
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(FIELD_ADDR_MSB_THRESHOLD));

            builder.retain_precomputed(builder.lookup_denominator(
                byte_kind,
                vec![
                    ltu_opcode,
                    is_addr_lt_threshold.clone(),
                    zero.clone(),
                    addr_word[3].clone(),
                    threshold,
                ],
            ));
        }

        // =====================================================================
        // #4: U8Range — prev_addr_word[0..1]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, prev_addr_word[0].clone(), prev_addr_word[1].clone());

        // =====================================================================
        // #5: U8Range — prev_addr_word[2..3]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, prev_addr_word[2].clone(), prev_addr_word[3].clone());

        // =====================================================================
        // #6: U8Range — value[0..1]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, value[0].clone(), value[1].clone());

        // =====================================================================
        // #7: U8Range — value[2..3]
        // =====================================================================
        u8_range_pair_precompute_lc(builder, value[2].clone(), value[3].clone());

        // =====================================================================
        // #8: Global — memory init/finalize interaction
        // Payload: [shard_or_0, timestamp_or_0, addr, value[0..3], is_send, is_recv, Memory_kind]
        // =====================================================================
        {
            let global_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
                InteractionKind::Global as usize,
            ));
            let memory_kind =
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(InteractionKind::Memory as u8));

            let (shard_val, timestamp_val, is_send, is_recv) = match self.kind {
                MemoryChipType::Initialize => {
                    (zero.clone(), zero.clone(), AB::VarMaybeExt::from(AB::F::one()), zero.clone())
                }
                MemoryChipType::Finalize => {
                    (shard, timestamp, zero.clone(), AB::VarMaybeExt::from(AB::F::one()))
                }
            };

            builder.retain_precomputed(builder.lookup_denominator(
                global_kind,
                vec![
                    shard_val,
                    timestamp_val,
                    addr.clone(),
                    value[0].clone(),
                    value[1].clone(),
                    value[2].clone(),
                    value[3].clone(),
                    is_send,
                    is_recv,
                    memory_kind,
                ],
            ));
        }

        // =====================================================================
        // #9: MemoryGlobalAddr send — [discriminant, addr]
        // =====================================================================
        {
            let addr_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
                InteractionKind::MemoryGlobalAddr as usize,
            ));
            let discriminant = match self.kind {
                MemoryChipType::Initialize => zero.clone(),
                MemoryChipType::Finalize => AB::VarMaybeExt::from(AB::F::one()),
            };

            builder.retain_precomputed(
                builder.lookup_denominator(addr_kind, vec![discriminant, addr.clone()]),
            );
        }

        // =====================================================================
        // #10: MemoryGlobalAddr recv — [discriminant, prev_addr_reconstructed]
        // =====================================================================
        {
            let addr_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
                InteractionKind::MemoryGlobalAddr as usize,
            ));
            let discriminant = match self.kind {
                MemoryChipType::Initialize => zero.clone(),
                MemoryChipType::Finalize => AB::VarMaybeExt::from(AB::F::one()),
            };

            let prev_addr_reconstructed = prev_addr_word[0].clone() +
                prev_addr_word[1].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8)) +
                prev_addr_word[2].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16)) +
                prev_addr_word[3].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 24));

            builder.retain_precomputed(
                builder.lookup_denominator(addr_kind, vec![discriminant, prev_addr_reconstructed]),
            );
        }

        // =====================================================================
        // #11: LTU — lt_cols comparison byte (from AssertLtColsBytes)
        // Payload: [LTU, 1, 0, a_comparison_byte, b_comparison_byte]
        // =====================================================================
        ltu_precompute_lc(builder, a_comparison_byte, b_comparison_byte);

        // =====================================================================
        // #12: BitVec — boolean enforcement
        // Booleans: is_real, is_addr_zero, is_addr_lt_threshold, byte_flags[0..3]
        // =====================================================================
        let byte_flags = [col(11), col(12), col(13), col(14)];
        bitvec_precompute_lc(
            builder,
            vec![
                is_real,
                is_addr_zero,
                is_addr_lt_threshold,
                byte_flags[0].clone(),
                byte_flags[1].clone(),
                byte_flags[2].clone(),
                byte_flags[3].clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local_slice: &[AB::VarMaybeExt] = local_binding.deref();

        // Read reserved columns in reserved_poly() order.
        let addr = local_slice[0].clone();
        let addr_word = [
            local_slice[1].clone(),
            local_slice[2].clone(),
            local_slice[3].clone(),
            local_slice[4].clone(),
        ];
        let prev_addr_word = [
            local_slice[5].clone(),
            local_slice[6].clone(),
            local_slice[7].clone(),
            local_slice[8].clone(),
        ];
        let byte_flags = [
            local_slice[9].clone(),
            local_slice[10].clone(),
            local_slice[11].clone(),
            local_slice[12].clone(),
        ];
        let a_comparison_byte = local_slice[13].clone();
        let b_comparison_byte = local_slice[14].clone();
        let value = [
            local_slice[15].clone(),
            local_slice[16].clone(),
            local_slice[17].clone(),
            local_slice[18].clone(),
        ];
        let is_real = local_slice[19].clone();
        let is_addr_zero = local_slice[20].clone();
        let is_addr_lt_threshold = local_slice[21].clone();

        let one = AB::one_maybe();

        // is_real boolean: required because is_real is now the BitVec mult (#12), so BitVec
        // only enforces payload bits boolean on rows where mult≠0. An explicit gate is needed
        // so that is_real itself is constrained to 0 or 1 on padding rows.
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // ── air.rs: is_addr_zero can only be set on real rows ──
        builder.when_ne(is_real.clone(), one.clone()).assert_zero(is_addr_zero.clone());

        // ── air.rs: addr byte decomposition ──
        // addr_from_word = addr_word[0] + addr_word[1]*256 + addr_word[2]*65536 +
        // addr_word[3]*16777216
        let addr_from_word = addr_word[0].clone() +
            addr_word[1].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8)) +
            addr_word[2].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16)) +
            addr_word[3].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 24));
        builder.when(is_real.clone()).assert_zero(addr_from_word - addr.clone());

        // ── air.rs: field range check ──
        // When is_real and NOT is_addr_lt_threshold: addr_word[3] must equal
        // FIELD_ADDR_MSB_THRESHOLD, lower bytes must be 0
        let threshold_msb =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(FIELD_ADDR_MSB_THRESHOLD));
        let guard = is_real.clone() * (one.clone() - is_addr_lt_threshold.clone());
        builder.when(guard.clone()).assert_zero(addr_word[3].clone() - threshold_msb);
        builder.when(guard.clone()).assert_zero(addr_word[2].clone());
        builder.when(guard.clone()).assert_zero(addr_word[1].clone());
        builder.when(guard.clone()).assert_zero(addr_word[0].clone());

        // ── air.rs: is_addr_zero constraints ──
        // When is_addr_zero = 1: addr must be 0
        builder.when(is_addr_zero.clone()).assert_zero(addr.clone());

        // When is_addr_zero = 1: prev_addr must be 0
        let prev_addr_reconstructed = prev_addr_word[0].clone() +
            prev_addr_word[1].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8)) +
            prev_addr_word[2].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16)) +
            prev_addr_word[3].clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 24));
        builder.when(is_addr_zero.clone()).assert_zero(prev_addr_reconstructed);

        // When is_addr_zero = 1: value must be 0
        for i in 0..4 {
            builder.when(is_addr_zero.clone()).assert_zero(value[i].clone());
        }

        // ── air.rs: AssertLtColsBytes gate constraints ──
        // lt_mult = is_real - is_addr_zero
        let lt_mult = is_real.clone() - is_addr_zero.clone();
        assert_lt_bytes_gate_constraints(
            builder,
            prev_addr_word.clone(),
            addr_word.clone(),
            byte_flags,
            a_comparison_byte,
            b_comparison_byte,
            lt_mult,
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local_slice: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local_slice[19].clone();
        let is_addr_zero = local_slice[20].clone();

        // lt_mult = is_real - is_addr_zero
        let lt_mult = is_real.clone() - is_addr_zero.clone();

        // #1: U8Range addr_word[0..1] — is_real
        builder.send(is_real.clone());

        // #2: U8Range addr_word[2..3] — is_real
        builder.send(is_real.clone());

        // #3: LTU BabyBear range check — is_real
        builder.send(is_real.clone());

        // #4: U8Range prev_addr_word[0..1] — lt_mult
        builder.send(lt_mult.clone());

        // #5: U8Range prev_addr_word[2..3] — lt_mult
        builder.send(lt_mult.clone());

        // #6: U8Range value[0..1] — is_real
        builder.send(is_real.clone());

        // #7: U8Range value[2..3] — is_real
        builder.send(is_real.clone());

        // #8: Global memory init/finalize — is_real
        builder.send(is_real.clone());

        // #9: MemoryGlobalAddr send addr — is_real
        builder.send(is_real.clone());

        // #10: MemoryGlobalAddr recv prev_addr — is_real
        builder.recv(is_real.clone());

        // #11: LTU lt_cols comparison — lt_mult
        ltu_lookup(builder, lt_mult);

        // #12: BitVec boolean enforcement — is_real (mirrors global chip pattern).
        // On real rows (is_real=1) BitVec enforces all 7 payload bits boolean.
        // On padding rows (is_real=0) no send; is_real itself is constrained
        // boolean by the explicit gate in eval.
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// BitVec payload helper
// =============================================================================

/// Compute the BitVec #12 payload for one real row.
///
/// Bit ordering matches `bitvec_precompute_lc`:
///   bit 0: is_real (= 1, always for real rows)
///   bit 1: is_addr_zero
///   bit 2: is_addr_lt_threshold
///   bits 3-6: byte_flags[0..3]  (mirrors AssertLtColsBytes::populate, LE index order)
fn memory_global_bitvec_value(addr: u32, prev_addr: u32) -> u16 {
    let addr_bytes = addr.to_le_bytes();
    let is_addr_zero = (addr == 0) as u16;
    let is_addr_lt_threshold = (addr_bytes[3] < FIELD_ADDR_MSB_THRESHOLD) as u16;

    // byte_flags[i] = 1 iff byte i (LE index) is the first from MSB where prev_byte < addr_byte.
    // Not populated when addr == 0 (matches generate_trace behavior).
    let mut byte_flags = [0u16; 4];
    if addr != 0 {
        let prev_bytes = prev_addr.to_le_bytes();
        for (i, (&a_byte, &b_byte)) in
            prev_bytes.iter().rev().zip(addr_bytes.iter().rev()).enumerate()
        {
            if a_byte < b_byte {
                byte_flags[3 - i] = 1; // 3-i converts MSB iteration index to LE byte index
                break;
            }
        }
    }

    1u16 // is_real
        | (is_addr_zero << 1)
        | (is_addr_lt_threshold << 2)
        | (byte_flags[0] << 3)
        | (byte_flags[1] << 4)
        | (byte_flags[2] << 5)
        | (byte_flags[3] << 6)
}

// =============================================================================
// MachineAir implementation (delegation to MemoryGlobalChip via field forwarding)
// =============================================================================

impl<F: Field> BaseAir<F> for MemoryGlobalChipPolyAir {
    fn width(&self) -> usize {
        NUM_MEMORY_INIT_COLS
    }
}

impl<F: Field> MachineAir<F> for MemoryGlobalChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        let c = MemoryGlobalChip { kind: self.kind };
        <MemoryGlobalChip as MachineAir<F>>::name(&c) + "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        // 1. Delegate base BLUs (U8Range × 4, LTU × 2).
        let c = MemoryGlobalChip { kind: self.kind };
        <MemoryGlobalChip as MachineAir<F>>::generate_dependencies(&c, input, output);

        // 2. PolyAir-only: BitVec #12, mult = is_real — emit one BLU per real event only. Padding
        //    rows have is_real=0, so no send there (mirrors global chip pattern).
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };
        memory_events.sort_by_key(|e| e.addr);

        let prev_start = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr,
        };

        for (i, event) in memory_events.iter().enumerate() {
            let prev_addr = if i == 0 { prev_start } else { memory_events[i - 1].addr };
            output.add_bit_vec_lookup(memory_global_bitvec_value(event.addr, prev_addr));
        }
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let c = MemoryGlobalChip { kind: self.kind };
        <MemoryGlobalChip as MachineAir<F>>::num_rows(&c, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        MemoryGlobalChip { kind: self.kind }.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        let c = MemoryGlobalChip { kind: self.kind };
        <MemoryGlobalChip as MachineAir<F>>::included(&c, shard)
    }

    fn commit_scope(&self) -> InteractionScope {
        let c = MemoryGlobalChip { kind: self.kind };
        <MemoryGlobalChip as MachineAir<F>>::commit_scope(&c)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        check_constraints::run_generate_dependencies, memory::MemoryGlobalChip,
        programs::tests::fibonacci_program,
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

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const BATCH_SIZE: usize = 3;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers(air: &MemoryGlobalChipPolyAir) -> Vec<EF> {
        let beta = challenge_beta();
        let required_max_beta_power = <MemoryGlobalChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(air);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn reducer() -> Vec<EF> {
        // Gate constraints: 19 base + 1 is_real boolean gate (added when BitVec mult changed
        // from one_maybe to is_real — BitVec no longer enforces is_real boolean on padding rows)
        // Lookup batch: ceil(12/3) = 4
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 20;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS as u32).map(|i| ef(i + 1)).collect()
    }

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(
        air: &MemoryGlobalChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <MemoryGlobalChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(air);
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

    /// Build the reference MemoryGlobal trace by running a program through the executor,
    /// generating derived dependencies, and calling `MemoryGlobalChip::generate_trace`.
    fn sample_trace(kind: MemoryChipType) -> RowMajorMatrix<F> {
        let program = fibonacci_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let mut shard = *runtime.records[0].clone();
        run_generate_dependencies(&mut shard);

        let chip = MemoryGlobalChip { kind };
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    fn run_constraint_check(kind: MemoryChipType) {
        let air = MemoryGlobalChipPolyAir::new(kind);
        let main = sample_trace(kind);
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

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
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
            "{:?} first_round non-zero at indices: {:?}",
            kind,
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
            "{:?} nonfirst_round non-zero at indices: {:?}",
            kind,
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
    fn test_init_constraint_check_satisfied() {
        run_constraint_check(MemoryChipType::Initialize);
    }

    #[test]
    fn test_finalize_constraint_check_satisfied() {
        run_constraint_check(MemoryChipType::Finalize);
    }

    /// BitVec #12 mult = is_real — one BLU per real event, none for padding rows.
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        for kind in [MemoryChipType::Initialize, MemoryChipType::Finalize] {
            let program = fibonacci_program();
            let mut runtime = Executor::new(program, DTCoreOpts::default());
            runtime.run().unwrap();
            let mut shard = *runtime.records[0].clone();
            run_generate_dependencies(&mut shard);

            let mut deps = ExecutionRecord::default();
            <MemoryGlobalChipPolyAir as MachineAir<F>>::generate_dependencies(
                &MemoryGlobalChipPolyAir::new(kind),
                &shard,
                &mut deps,
            );

            let bitvec_total: usize = deps
                .byte_lookups
                .iter()
                .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
                .map(|(_, m)| *m)
                .sum();

            let expected = match kind {
                MemoryChipType::Initialize => shard.global_memory_initialize_events.len(),
                MemoryChipType::Finalize => shard.global_memory_finalize_events.len(),
            };
            assert!(expected > 0, "{:?}: fixture must include memory events", kind);
            // bitvec_total counts unique (b,c) pairs weighted by multiplicity.
            // Since different events may pack to the same u16 value, we check the
            // sum of multiplicities equals the event count.
            assert_eq!(
                bitvec_total, expected,
                "{:?}: BitVec BLU count must equal event count (mult=is_real)",
                kind
            );
        }
    }

    fn random_memory_global_trace(
        log_n: usize,
        _seed: u64,
        kind: MemoryChipType,
    ) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace(kind);
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

        let last_row_start = (base_height - 1) * NUM_MEMORY_INIT_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_MEMORY_INIT_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_MEMORY_INIT_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_MEMORY_INIT_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let kind = MemoryChipType::Initialize;
        let air = MemoryGlobalChipPolyAir::new(kind);
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_memory_global_trace(log_n, seed, kind);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!(
            "perf_multi_round: log_n={}, h={}, seed={}, kind={:?}",
            log_n,
            height,
            seed,
            kind
        );

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers(&air);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc = <MemoryGlobalChipPolyAir as FullAir<
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
            &beta_powers,
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
