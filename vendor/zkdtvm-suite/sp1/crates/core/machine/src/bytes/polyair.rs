//! PolyAir helpers for byte-level lookup interactions.
//!
//! These primitives wrap individual `ByteOpcode` lookups (MSB, LTU, U8Range,
//! U16Range, BitVec) into composable `_precompute_lc` / `_lookup` pairs.
//!
//! They are the lowest layer of the helper hierarchy — consumed by
//! operation-level helpers (MulOp, AddOp, …) and chip-level PolyAir impls.

use dt_core_executor::ByteOpcode;
use dt_stark::{air::FullAirBuilder, InteractionKind};
use p3_field::AbstractField;

// ============================================================================
// MSB (1 interaction)
// ============================================================================

/// Precompute denominator for an MSB byte lookup.
///
/// Payload: [MSB_opcode, msb_result, 0, msb_byte, 0]
pub fn msb_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    msb_result: AB::VarMaybeExt,
    msb_byte: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let msb_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::MSB as u8));

    builder.retain_precomputed(
        builder.lookup_denominator(
            byte_kind,
            vec![msb_opcode, msb_result, zero.clone(), msb_byte, zero],
        ),
    );
}

/// Declare multiplicity for an MSB lookup.
pub fn msb_lookup<AB: FullAirBuilder>(builder: &mut AB, is_real: AB::VarMaybeExt) {
    builder.send(is_real);
}

// ============================================================================
// LTU (1 interaction)
// ============================================================================

/// Precompute denominator for an LTU byte lookup.
///
/// Payload: [LTU_opcode, 1, 0, a_byte, b_byte]
pub fn ltu_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    a_byte: AB::VarMaybeExt,
    b_byte: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let one = AB::VarMaybeExt::from(AB::F::one());
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let ltu_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::LTU as u8));

    builder.retain_precomputed(
        builder.lookup_denominator(byte_kind, vec![ltu_opcode, one, zero, a_byte, b_byte]),
    );
}

/// Declare multiplicity for an LTU lookup.
pub fn ltu_lookup<AB: FullAirBuilder>(builder: &mut AB, multiplicity: AB::VarMaybeExt) {
    builder.send(multiplicity);
}

// ============================================================================
// U8Range (1 interaction per pair)
// ============================================================================

/// Precompute denominator for a single U8Range pair lookup.
///
/// Payload: [U8Range_opcode, 0, 0, byte_a, byte_b]
pub fn u8_range_pair_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    byte_a: AB::VarMaybeExt,
    byte_b: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let u8_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U8Range as u8));

    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![u8_opcode, zero.clone(), zero.clone(), byte_a, byte_b],
    ));
}

/// Precompute denominators for a slice of bytes as U8Range pairs.
///
/// The slice is processed in consecutive pairs: (slice[0], slice[1]), (slice[2], slice[3]), ...
/// Produces `slice.len() / 2` interactions.
///
/// # Panics
/// Panics if `slice.len()` is odd.
pub fn slice_u8_range_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    slice: &[AB::VarMaybeExt],
) {
    assert!(slice.len() % 2 == 0, "slice_u8_range_precompute_lc: slice length must be even");
    for pair in slice.chunks(2) {
        u8_range_pair_precompute_lc(builder, pair[0].clone(), pair[1].clone());
    }
}

/// Declare multiplicities for U8Range pair lookups.
pub fn slice_u8_range_lookup<AB: FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    num_pairs: usize,
) {
    for _ in 0..num_pairs {
        builder.send(is_real.clone());
    }
}

// ============================================================================
// U16Range (1 interaction each)
// ============================================================================

/// Precompute denominator for a single U16Range lookup.
///
/// Payload: [U16Range_opcode, value, 0, 0, 0]
pub fn u16_range_precompute_lc<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let u16_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::U16Range as u8));

    builder.retain_precomputed(
        builder.lookup_denominator(
            byte_kind,
            vec![u16_opcode, value, zero.clone(), zero.clone(), zero],
        ),
    );
}

/// Precompute denominators for a slice of U16Range lookups.
///
/// Produces `slice.len()` interactions (one per element).
pub fn slice_u16_range_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    slice: &[AB::VarMaybeExt],
) {
    for value in slice {
        u16_range_precompute_lc(builder, value.clone());
    }
}

/// Declare multiplicities for U16Range lookups.
pub fn slice_u16_range_lookup<AB: FullAirBuilder>(
    builder: &mut AB,
    is_real: AB::VarMaybeExt,
    count: usize,
) {
    for _ in 0..count {
        builder.send(is_real.clone());
    }
}

// ============================================================================
// BitVec (1 interaction)
// ============================================================================

/// Precompute denominator for a BitVec lookup.
///
/// Pads `bits` to 16 elements with zeros (the fixed BitVec width).
/// Produces 1 interaction.
pub fn bitvec_precompute_lc<AB: FullAirBuilder>(builder: &mut AB, mut bits: Vec<AB::VarMaybeExt>) {
    let zero = AB::zero_maybe();
    let bitvec_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::BitVec as usize));

    bits.resize(16, zero);
    builder.retain_precomputed(builder.lookup_denominator(bitvec_kind, bits));
}

/// Declare multiplicity for a BitVec lookup.
pub fn bitvec_lookup<AB: FullAirBuilder>(builder: &mut AB, multiplicity: AB::VarMaybeExt) {
    builder.send(multiplicity);
}

// ============================================================================
// BitRange (1 interaction)
// ============================================================================

/// BitRange lookup: 1 Byte send
pub(crate) const BIT_RANGE_NUM_INTERACTIONS: usize = 1;

/// Precompute denominator for a BitRange byte lookup.
///
/// Payload: [BitRange_opcode, value, num_bits, 0, 0]
pub fn bit_range_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    value: AB::VarMaybeExt,
    num_bits: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let bit_range_opcode =
        AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::BitRange as u8));

    // ByteChip stores BitRange as [BitRange, value, 0, bit_width, 0] (bit_width at b-field/pos 3).
    builder.retain_precomputed(builder.lookup_denominator(
        byte_kind,
        vec![bit_range_opcode, value, zero.clone(), num_bits, zero],
    ));
}

/// Declare multiplicity for a BitRange lookup.
pub fn bit_range_lookup<AB: FullAirBuilder>(builder: &mut AB, multiplicity: AB::VarMaybeExt) {
    builder.send(multiplicity);
}

// ============================================================================
// XOR (1 interaction)
// ============================================================================

/// XOR lookup: 1 Byte send
pub(crate) const XOR_NUM_INTERACTIONS: usize = 1;

/// Precompute denominator for an XOR byte lookup.
///
/// Payload: [XOR_opcode, result, 0, a_byte, b_byte]
pub fn xor_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    result: AB::VarMaybeExt,
    a_byte: AB::VarMaybeExt,
    b_byte: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let xor_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::XOR as u8));

    builder.retain_precomputed(
        builder.lookup_denominator(byte_kind, vec![xor_opcode, result, zero, a_byte, b_byte]),
    );
}

/// Declare multiplicity for an XOR lookup.
pub fn xor_lookup<AB: FullAirBuilder>(builder: &mut AB, multiplicity: AB::VarMaybeExt) {
    builder.send(multiplicity);
}

// ============================================================================
// AND (1 interaction)
// ============================================================================

/// AND lookup: 1 Byte send
pub(crate) const AND_NUM_INTERACTIONS: usize = 1;

/// Precompute denominator for an AND byte lookup.
///
/// Mirrors `send_byte(AND, result, b, c, mult)` which expands to
/// `send_byte_pair(AND, result, 0, b, c, mult)`.
///
/// Payload: [AND_opcode, result, 0, b, c]
pub fn and_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    result: AB::VarMaybeExt,
    b: AB::VarMaybeExt,
    c: AB::VarMaybeExt,
) {
    let zero = AB::zero_maybe();
    let byte_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
    let and_opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::AND as u8));

    builder.retain_precomputed(
        builder.lookup_denominator(byte_kind, vec![and_opcode, result, zero, b, c]),
    );
}

/// Declare multiplicity for an AND lookup.
pub fn and_lookup<AB: FullAirBuilder>(builder: &mut AB, multiplicity: AB::VarMaybeExt) {
    builder.send(multiplicity);
}
