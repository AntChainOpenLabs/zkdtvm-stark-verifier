//! PolyAir helpers for memory access lookup interactions.
//!
//! Each memory access (read or read-write) produces 4 interactions:
//!   1. timestamp diff U16Range check
//!   2. timestamp diff BitRange(12) check
//!   3. memory send (previous value/shard/clk)
//!   4. memory recv (current value/shard/clk)

use dt_core_executor::ByteOpcode;
use dt_stark::{air::FullAirBuilder, InteractionKind, Word};
use p3_field::AbstractField;

use crate::memory::MemoryAccessCols;

// ============================================================================
// Interaction count constants
// ============================================================================

/// Memory read access: ts_u16 + ts_bit12 + mem_send(prev) + mem_recv(curr)
pub(crate) const MEMORY_READ_NUM_INTERACTIONS: usize = 4;

/// Memory read-write access: ts_u16 + ts_bit12 + mem_send(prev_value) + mem_recv(curr)
pub(crate) const MEMORY_READWRITE_NUM_INTERACTIONS: usize = 4;

// ============================================================================
// Memory Read Access (4 interactions)
// ============================================================================

/// Precompute denominators for a memory read access.
///
/// Interactions (in order):
///   1. send Byte(U16Range, diff_16bit_limb)
///   2. send Byte(BitRange, diff_12bit_limb, 12)
///   3. send Memory(prev_shard, prev_clk, addr, value[0..3])
///   4. recv Memory(shard, clk, addr, value[0..3])
pub fn memory_read_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    access: &MemoryAccessCols<AB::VarMaybeExt>,
    addr: AB::VarMaybeExt,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let mem_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));
    let u16_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U16Range as u8));
    let bit_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::BitRange as u8));
    let twelve = AB::VarMaybeExt::from(AB::F::from_canonical_u32(12));

    // ts U16Range
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind.clone(),
        vec![u16_opcode, access.diff_16bit_limb.clone(), zero.clone(), zero.clone(), zero.clone()],
    ));
    // ts BitRange(12)
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![bit_opcode, access.diff_12bit_limb.clone(), zero.clone(), twelve, zero.clone()],
    ));
    // mem_send (prev)
    let mut mem_send = vec![access.prev_shard.clone(), access.prev_clk.clone(), addr.clone()];
    mem_send.extend(access.value.0.iter().cloned());
    builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), mem_send));
    // mem_recv (curr)
    let mut mem_recv = vec![shard, clk, addr];
    mem_recv.extend(access.value.0.iter().cloned());
    builder.retain_precomputed(builder.lookup_denominator(mem_kind, mem_recv));
}

/// Declare multiplicities for a memory read access.
pub fn memory_read_lookup<AB: FullAirBuilder>(builder: &mut AB, is_real: AB::VarMaybeExt) {
    builder.send(is_real.clone()); // ts U16Range
    builder.send(is_real.clone()); // ts BitRange
    builder.send(is_real.clone()); // memory send
    builder.recv(is_real); // memory recv
}

// ============================================================================
// Memory ReadWrite Access (4 interactions)
// ============================================================================

/// Precompute denominators for a memory read-write access.
///
/// Same as `memory_read_precompute_lc` except the mem_send uses `prev_value`
/// instead of `access.value`.
pub fn memory_readwrite_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    access: &MemoryAccessCols<AB::VarMaybeExt>,
    prev_value: &Word<AB::VarMaybeExt>,
    addr: AB::VarMaybeExt,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let mem_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Memory as usize));
    let u16_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U16Range as u8));
    let bit_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::BitRange as u8));
    let twelve = AB::VarMaybeExt::from(AB::F::from_canonical_u32(12));

    // ts U16Range
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind.clone(),
        vec![u16_opcode, access.diff_16bit_limb.clone(), zero.clone(), zero.clone(), zero.clone()],
    ));
    // ts BitRange(12)
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![bit_opcode, access.diff_12bit_limb.clone(), zero.clone(), twelve, zero.clone()],
    ));
    // mem_send (prev_value)
    let mut mem_send = vec![access.prev_shard.clone(), access.prev_clk.clone(), addr.clone()];
    mem_send.extend(prev_value.0.iter().cloned());
    builder.retain_precomputed(builder.lookup_denominator(mem_kind.clone(), mem_send));
    // mem_recv (curr value)
    let mut mem_recv = vec![shard, clk, addr];
    mem_recv.extend(access.value.0.iter().cloned());
    builder.retain_precomputed(builder.lookup_denominator(mem_kind, mem_recv));
}

/// Declare multiplicities for a memory read-write access.
pub fn memory_readwrite_lookup<AB: FullAirBuilder>(builder: &mut AB, is_real: AB::VarMaybeExt) {
    builder.send(is_real.clone()); // ts U16Range
    builder.send(is_real.clone()); // ts BitRange
    builder.send(is_real.clone()); // memory send
    builder.recv(is_real); // memory recv
}

// ============================================================================
// Memory Timestamp Gate Constraints
// ============================================================================

/// Gate constraints for `eval_memory_access_timestamp`.
///
/// Reproduces:
/// - `compare_clk` boolean check
/// - shard equality when `compare_clk` is set
/// - 28-bit range decomposition: `diff_minus_one = diff_16bit_limb + diff_12bit_limb * 2^16`
///
/// NOTE: The U16Range and BitRange lookups for the timestamp limbs are already
/// handled by `memory_read_precompute_lc` / `memory_readwrite_precompute_lc`.
pub fn memory_timestamp_gate_constraints<AB: FullAirBuilder>(
    builder: &mut AB,
    access: &crate::memory::MemoryAccessCols<AB::VarMaybeExt>,
    shard: AB::VarMaybeExt,
    clk: AB::VarMaybeExt,
    is_real: AB::VarMaybeExt,
) where
    AB::VarMaybeExt: Clone,
{
    let one = AB::one_maybe();
    let limb_base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
    let compare_clk = access.compare_clk.clone();

    // compare_clk boolean
    builder
        .when(is_real.clone())
        .assert_zero(compare_clk.clone() * (one.clone() - compare_clk.clone()));

    // shard == prev_shard when compare_clk
    builder
        .when(is_real.clone())
        .when(compare_clk.clone())
        .assert_eq(shard.clone(), access.prev_shard.clone());

    // 28-bit range decomposition
    let prev_comp_value = compare_clk.clone() * access.prev_clk.clone() +
        (one.clone() - compare_clk.clone()) * access.prev_shard.clone();
    let current_comp_value = compare_clk.clone() * clk + (one.clone() - compare_clk) * shard;
    let diff_minus_one = current_comp_value - prev_comp_value - one;

    builder.when(is_real).assert_eq(
        diff_minus_one,
        access.diff_16bit_limb.clone() + access.diff_12bit_limb.clone() * limb_base,
    );
}
