use dt_core_executor::{ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_primitives::consts::WORD_SIZE;
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{ShiftRightChip, ShiftRightCols, BYTE_SIZE, LONG_WORD_SIZE, NUM_SHIFT_RIGHT_COLS};
use crate::{
    adapter::{
        register::alu_type::{
            alu_type_register_op_gate_constraints, alu_type_register_op_lookup,
            alu_type_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{
        bitvec_lookup, bitvec_precompute_lc, msb_lookup, msb_precompute_lc, slice_u8_range_lookup,
        slice_u8_range_precompute_lc,
    },
};

/// Public values index for `execution_shard`.
/// PublicValues<Word<u8>, u8>: committed_value_digest(32) + deferred_proofs_digest(8)
/// + start_pc(1) + next_pc(1) + exit_code(1) + shard(1) = 44.
const PV_EXECUTION_SHARD_IDX: usize = 44;

const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `ShiftRightCols<u8>` (NUM_SHIFT_RIGHT_COLS = 99).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1..4]   cpu_state.{clk_16_28, clk_0_16, pc}        ← precompute-only
//   [4]      mem_ops.op_a                               ← precompute-only
//   [5..9]   mem_ops.op_a_access.prev_value             ← precompute-only
//   [9..13]  mem_ops.op_a_access.access.value
//   [13..18] mem_ops.op_a_access.access.{ts fields}     ← precompute-only
//   [18]     mem_ops.op_a_zero
//   [19]     mem_ops.op_b                               ← precompute-only
//   [20..24] mem_ops.op_b_access.access.value
//   [24..29] mem_ops.op_b_access.access.{ts fields}     ← precompute-only
//   [29..33] mem_ops.op_c (Word)
//   [33..37] mem_ops.op_c_access.access.value
//   [37..42] mem_ops.op_c_access.access.{ts fields}     ← precompute-only
//   [42]     mem_ops.imm_c
//   [43..51] shift_by_n_bits
//   [51..55] shift_by_n_bytes
//   [55..63] byte_shift_result
//   [63..71] bit_shift_result
//   [71..79] shr_carry_output_carry
//   [79..87] shr_carry_output_shifted_byte
//   [87]     b_msb
//   [88..96] c_least_sig_byte
//   [96]     is_srl
//   [97]     is_sra
//   [98]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_B_VALUE: usize = 20;
const COL_OP_C: usize = 29;
const COL_OP_C_ACCESS_VALUE: usize = 33;
const COL_IMM_C: usize = 42;
const COL_SHIFT_BY_N_BITS: usize = 43;
const COL_SHIFT_BY_N_BYTES: usize = 51;
const COL_BYTE_SHIFT_RESULT: usize = 55;
const COL_BIT_SHIFT_RESULT: usize = 63;
const COL_SHR_CARRY_OUTPUT_CARRY: usize = 71;
const COL_SHR_CARRY_OUTPUT_SHIFTED_BYTE: usize = 79;
const COL_B_MSB: usize = 87;
const COL_C_LEAST_SIG_BYTE: usize = 88;
const COL_IS_SRL: usize = 96;
const COL_IS_SRA: usize = 97;
const COL_IS_REAL: usize = 98;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 75).
//
// Only fields read by `eval` or `lookup` are retained.
//
//   [0]      is_real
//   [1]      cpu_state.shard
//   [2]      is_sra
//   [3]      is_srl
//   [4]      op_a_zero
//   [5]      imm_c
//   [6]      b_msb
//   [7..11]  op_a_access.access.value
//   [11..15] op_b_access.access.value
//   [15..19] op_c_access.access.value
//   [19..23] op_c (Word)
//   [23..31] c_least_sig_byte
//   [31..39] shift_by_n_bits
//   [39..43] shift_by_n_bytes
//   [43..51] byte_shift_result
//   [51..59] bit_shift_result
//   [59..67] shr_carry_output_carry
//   [67..75] shr_carry_output_shifted_byte
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_CPU_SHARD: usize = 1;
const RES_IS_SRA: usize = 2;
const RES_IS_SRL: usize = 3;
const RES_OP_A_ZERO: usize = 4;
const RES_IMM_C: usize = 5;
const RES_B_MSB: usize = 6;
const RES_OP_A_VALUE: usize = 7;
const RES_OP_B_VALUE: usize = 11;
const RES_OP_C_ACCESS_VALUE: usize = 15;
const RES_OP_C: usize = 19;
const RES_C_LEAST_SIG_BYTE: usize = 23;
const RES_SHIFT_BY_N_BITS: usize = 31;
const RES_SHIFT_BY_N_BYTES: usize = 39;
const RES_BYTE_SHIFT_RESULT: usize = 43;
const RES_BIT_SHIFT_RESULT: usize = 51;
const RES_SHR_CARRY_OUTPUT_CARRY: usize = 59;
const RES_SHR_CARRY_OUTPUT_SHIFTED_BYTE: usize = 67;
const RES_NUM_COLS: usize = 75;

#[derive(Default, Clone, Copy)]
pub struct ShiftRightPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for ShiftRightPolyAir {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_IS_SRA));
        cols.push(PairCol::Main(COL_IS_SRL));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_IMM_C));
        cols.push(PairCol::Main(COL_B_MSB));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_ACCESS_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C + i));
        }
        for i in 0..8 {
            cols.push(PairCol::Main(COL_C_LEAST_SIG_BYTE + i));
        }
        for i in 0..8 {
            cols.push(PairCol::Main(COL_SHIFT_BY_N_BITS + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_SHIFT_BY_N_BYTES + i));
        }
        for i in 0..LONG_WORD_SIZE {
            cols.push(PairCol::Main(COL_BYTE_SHIFT_RESULT + i));
        }
        for i in 0..LONG_WORD_SIZE {
            cols.push(PairCol::Main(COL_BIT_SHIFT_RESULT + i));
        }
        for i in 0..LONG_WORD_SIZE {
            cols.push(PairCol::Main(COL_SHR_CARRY_OUTPUT_CARRY + i));
        }
        for i in 0..LONG_WORD_SIZE {
            cols.push(PairCol::Main(COL_SHR_CARRY_OUTPUT_SHIFTED_BYTE + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShiftRightCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        // =====================================================================
        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        // =====================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc.clone(),
            next_pc,
        );

        // =====================================================================
        // #5: MSB send_byte(MSB, b_msb, b[3])
        // =====================================================================
        let b_word_3 = local.mem_ops.op_b_access.access.value.0[WORD_SIZE - 1].clone();
        msb_precompute_lc(builder, local.b_msb.clone(), b_word_3);

        // =====================================================================
        // #6-13: ShrCarry send_byte_pair × 8 (reversed order: 7,6,...,0)
        // No helper exists — keep manual
        // =====================================================================
        let byte_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
        let shr_carry_opcode =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::ShrCarry as u8));
        let mut num_bits_to_shift = local.c_least_sig_byte[0].clone();
        for i in 1..3 {
            num_bits_to_shift = num_bits_to_shift +
                local.c_least_sig_byte[i].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << i));
        }
        for i in (0..LONG_WORD_SIZE).rev() {
            builder.retain_precomputed(builder.lookup_denominator(
                byte_kind.clone(),
                vec![
                    shr_carry_opcode.clone(),
                    local.shr_carry_output_shifted_byte[i].clone(),
                    local.shr_carry_output_carry[i].clone(),
                    local.byte_shift_result[i].clone(),
                    num_bits_to_shift.clone(),
                ],
            ));
        }

        // =====================================================================
        // #14-29: slice_range_check_u8 on 4 long words (8 bytes each → 4 pairs = 16)
        // Order: byte_shift_result, bit_shift_result, shr_carry_output_carry,
        //        shr_carry_output_shifted_byte
        // =====================================================================
        slice_u8_range_precompute_lc(builder, &local.byte_shift_result);
        slice_u8_range_precompute_lc(builder, &local.bit_shift_result);
        slice_u8_range_precompute_lc(builder, &local.shr_carry_output_carry);
        slice_u8_range_precompute_lc(builder, &local.shr_carry_output_shifted_byte);

        // =====================================================================
        // #30-42: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        // =====================================================================
        {
            let opcode = local.is_sra.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::SRA as u32)) +
                local.is_srl.clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(Opcode::SRL as u32));
            alu_type_register_op_precompute_lc(
                builder,
                pc,
                opcode,
                local.mem_ops.op_a.clone(),
                local.mem_ops.op_b.clone(),
                [
                    local.mem_ops.op_c[0].clone(),
                    local.mem_ops.op_c[1].clone(),
                    local.mem_ops.op_c[2].clone(),
                    local.mem_ops.op_c[3].clone(),
                ],
                local.mem_ops.op_a_zero.clone(),
                local.mem_ops.imm_c.clone(),
                &local.mem_ops.op_b_access.access,
                &local.mem_ops.op_c_access.access,
                &local.mem_ops.op_a_access.access,
                &local.mem_ops.op_a_access.prev_value,
                shard,
                clk,
            );
        }

        // =====================================================================
        // #43: BitVec [c_least_sig_byte[0..8], shift_by_n_bits[0..8]] (16 bits)
        // All assert_bool in original AIR are UNCONDITIONAL (mod.rs:477-490)
        // =====================================================================
        let mut bits_1: Vec<AB::VarMaybeExt> = Vec::with_capacity(16);
        for i in 0..8 {
            bits_1.push(local.c_least_sig_byte[i].clone());
        }
        for i in 0..8 {
            bits_1.push(local.shift_by_n_bits[i].clone());
        }
        bitvec_precompute_lc(builder, bits_1);

        // =====================================================================
        // #44: BitVec [shift_by_n_bytes[0..4], is_srl, is_sra, b_msb] (7 bits)
        // =====================================================================
        // is_real is dropped from the payload because the BitVec mult is now
        // conditioned on it (see `lookup`); BitVec only enforces booleanness
        // when mult ≠ 0, so is_real's booleanness is instead asserted as an
        // explicit gate in `eval`.
        bitvec_precompute_lc(
            builder,
            vec![
                local.shift_by_n_bytes[0].clone(),
                local.shift_by_n_bytes[1].clone(),
                local.shift_by_n_bytes[2].clone(),
                local.shift_by_n_bytes[3].clone(),
                local.is_srl.clone(),
                local.is_sra.clone(),
                local.b_msb.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let is_sra = local[RES_IS_SRA].clone();
        let is_srl = local[RES_IS_SRL].clone();

        // Replaces the implicit boolean enforcement BitVec #44 used to provide on is_real.
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // CPUState: shard == execution_shard when is_real
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // is_real = is_sra + is_srl
        builder.assert_zero(is_real.clone() - is_sra.clone() - is_srl);

        // ALUType adapter gate constraints (op_a_zero, imm_c padding, op_c consistency)
        let a_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_A_VALUE + i].clone());
        alu_type_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            a_word.clone(),
            local[RES_IMM_C].clone(),
            core::array::from_fn(|i| local[RES_OP_C_ACCESS_VALUE + i].clone()),
            core::array::from_fn(|i| local[RES_OP_C + i].clone()),
            is_real.clone(),
        );

        // c_least_sig_byte decomposition = c_word[0]
        let mut recon_c0 = AB::zero_maybe();
        for i in 0..BYTE_SIZE {
            recon_c0 = recon_c0 +
                local[RES_C_LEAST_SIG_BYTE + i].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << i));
        }
        builder.assert_zero(recon_c0 - local[RES_OP_C_ACCESS_VALUE].clone());

        // num_bits_to_shift = c_bits[0..3]
        let mut num_bits = local[RES_C_LEAST_SIG_BYTE].clone();
        for i in 1..3 {
            num_bits = num_bits +
                local[RES_C_LEAST_SIG_BYTE + i].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << i));
        }
        for i in 0..BYTE_SIZE {
            builder.assert_zero(
                local[RES_SHIFT_BY_N_BITS + i].clone() *
                    (num_bits.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_usize(i))),
            );
        }
        let mut sum_bit_flags = AB::zero_maybe();
        for i in 0..BYTE_SIZE {
            sum_bit_flags = sum_bit_flags + local[RES_SHIFT_BY_N_BITS + i].clone();
        }
        builder.assert_zero(AB::one_maybe() - sum_bit_flags);

        // num_bytes_to_shift = c_bits[3..5)
        let num_bytes = local[RES_C_LEAST_SIG_BYTE + 3].clone() +
            local[RES_C_LEAST_SIG_BYTE + 4].clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(2));
        for i in 0..WORD_SIZE {
            builder.assert_zero(
                local[RES_SHIFT_BY_N_BYTES + i].clone() *
                    (num_bytes.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_usize(i))),
            );
        }
        let mut sum_byte_flags = AB::zero_maybe();
        for i in 0..WORD_SIZE {
            sum_byte_flags = sum_byte_flags + local[RES_SHIFT_BY_N_BYTES + i].clone();
        }
        builder.assert_zero(AB::one_maybe() - sum_byte_flags);

        // Byte shift: sign-extend b to 8 bytes, then byte-shift
        let leading = is_sra *
            local[RES_B_MSB].clone() *
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(0xff));
        let mut sign_ext_b: Vec<AB::VarMaybeExt> = Vec::with_capacity(LONG_WORD_SIZE);
        for i in 0..WORD_SIZE {
            sign_ext_b.push(local[RES_OP_B_VALUE + i].clone());
        }
        for _ in 0..WORD_SIZE {
            sign_ext_b.push(leading.clone());
        }
        for n in 0..WORD_SIZE {
            for i in 0..(LONG_WORD_SIZE - n) {
                builder.assert_zero(
                    local[RES_SHIFT_BY_N_BYTES + n].clone() *
                        (local[RES_BYTE_SHIFT_RESULT + i].clone() - sign_ext_b[i + n].clone()),
                );
            }
        }

        // Bit shift using ShrCarry results
        let mut carry_mul = AB::zero_maybe();
        for i in 0..BYTE_SIZE {
            carry_mul = carry_mul +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << (8 - i))) *
                    local[RES_SHIFT_BY_N_BITS + i].clone();
        }
        for i in (0..LONG_WORD_SIZE).rev() {
            let mut v = local[RES_SHR_CARRY_OUTPUT_SHIFTED_BYTE + i].clone();
            if i + 1 < LONG_WORD_SIZE {
                v = v + local[RES_SHR_CARRY_OUTPUT_CARRY + i + 1].clone() * carry_mul.clone();
            }
            builder.assert_zero(local[RES_BIT_SHIFT_RESULT + i].clone() - v);
        }

        // a[i] = bit_shift_result[i] when op_a is non-zero
        let op_a_not_0 = AB::one_maybe() - local[RES_OP_A_ZERO].clone();
        for i in 0..WORD_SIZE {
            builder.assert_zero(
                op_a_not_0.clone() * (a_word[i].clone() - local[RES_BIT_SHIFT_RESULT + i].clone()),
            );
        }

        // (op_a_zero, imm_c gate constraints handled by alu_type_register_op_gate_constraints
        // above)
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let imm_c = local[RES_IMM_C].clone();

        // #1-4: CPUState (mult = is_real)
        cpu_state_lookup(builder, is_real.clone());

        // #5: MSB (mult = is_real)
        msb_lookup(builder, is_real.clone());

        // #6-13: ShrCarry × 8 (mult = is_real)
        for _ in 0..LONG_WORD_SIZE {
            builder.send(is_real.clone());
        }

        // #14-29: U8Range × 16 (4 long words × 4 pairs) (mult = is_real)
        slice_u8_range_lookup(builder, is_real.clone(), 4); // byte_shift_result
        slice_u8_range_lookup(builder, is_real.clone(), 4); // bit_shift_result
        slice_u8_range_lookup(builder, is_real.clone(), 4); // shr_carry_output_carry
        slice_u8_range_lookup(builder, is_real.clone(), 4); // shr_carry_output_shifted_byte

        // #30-42: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        alu_type_register_op_lookup(builder, is_real.clone(), imm_c);

        // #43: BitVec [c_least_sig_byte + shift_by_n_bits]
        // Mult = is_real, matching the conditioning pattern of every other
        // lookup in the chip (MSB, ShrCarry, U8Range, etc. are all conditioned
        // on is_real). On real rows, BitVec enforces booleanness. On padding
        // rows is_real=0 ⇒ no send, so no padding-row BLU emission needed.
        // The padding template (mod.rs:204-210) sets the relevant columns to
        // boolean values structurally, preserving soundness.
        bitvec_lookup(builder, is_real.clone());

        // #44: BitVec [shift_by_n_bytes + is_srl + is_sra + b_msb] — same conditioning.
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to ShiftRightChip)
// =============================================================================

impl<F: Field> BaseAir<F> for ShiftRightPolyAir {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_COLS
    }
}

impl<F: Field> MachineAir<F> for ShiftRightPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShiftRightPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShiftRightChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.shift_right_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .shift_right_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse base chip path: MSB, ShrCarry × 8, U8Range × 4
                    // are all emitted by ShiftRightChip::event_to_row.
                    let mut row = [F::zero(); NUM_SHIFT_RIGHT_COLS];
                    let cols: &mut ShiftRightCols<F> = row.as_mut_slice().borrow_mut();
                    ShiftRightChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec #43 + #44 per real row.
                    // BitVec mults = is_real, so padding rows are skipped here
                    // (matching the base chip's generate_dependencies pattern).
                    let shamt = (event.c & 0x1F) as usize;
                    let bit_shift = shamt % 8;
                    let byte_shift = shamt / 8;

                    // #43: bits 0..8 ← c_least_sig_byte[i] = (event.c >> i) & 1
                    //      bits 8..16 ← shift_by_n_bits[i] = (bit_shift == i)
                    let c_low = (event.c & 0xFF) as u16;
                    let value_43 = c_low | ((1u16 << bit_shift) << 8);
                    blu.add_bit_vec_lookup(value_43);

                    // #44 (is_real dropped): bits 0..4 ← shift_by_n_bytes[i]
                    //                        bit 4     ← is_srl
                    //                        bit 5     ← is_sra
                    //                        bit 6     ← b_msb
                    let is_srl = (event.opcode == Opcode::SRL) as u16;
                    let is_sra = (event.opcode == Opcode::SRA) as u16;
                    let b_msb = ((event.b >> 31) & 1) as u16;
                    let value_44 =
                        (1u16 << byte_shift) | (is_srl << 4) | (is_sra << 5) | (b_msb << 6);
                    blu.add_bit_vec_lookup(value_44);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShiftRightChip as MachineAir<F>>::included(&ShiftRightChip, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        ShiftRightChip.padding_row()
    }

    fn local_only(&self) -> bool {
        <ShiftRightChip as MachineAir<F>>::local_only(&ShiftRightChip)
    }
}

#[cfg(test)]
mod tests {
    use std;

    use super::*;

    /// Total interactions:
    /// - 4 CPUState (state recv, send, clk U16, clk BitRange)
    /// - 1 MSB send_byte
    /// - 8 ShrCarry send_byte_pair (LONG_WORD_SIZE=8, reversed order)
    /// - 16 slice_range_check_u8 (4 long words × 8 bytes → 4 pairs each = 16)
    /// - 1 Program send
    /// - 12 Memory (3 accesses × 4)
    /// - 2 BitVec boolean constraints
    const NUM_LOOKUPS: usize = 44;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

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

    use super::super::ShiftRightChip;
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }
    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }
    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let n = <ShiftRightPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::required_max_beta_power(&ShiftRightPolyAir);
        (0..=n).map(|i| beta.exp_u64(i as u64)).collect()
    }
    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }
    fn reducer() -> Vec<EF> {
        // Gate constraints: 66 (added is_real boolean gate)
        // Lookup batch: ceil(44/3) = 15
        // Cumulative sum: 3
        // Total: 66 + 15 + 3 = 84
        const NUM_GATE_CONSTRAINTS: usize = 66;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS as u32).map(|i| ef(i + 1)).collect()
    }
    fn trim_rows<T: Clone + Send + Sync>(m: &RowMajorMatrix<T>, n: usize) -> RowMajorMatrix<T> {
        let w = m.width();
        RowMajorMatrix::new(m.values[..n * w].to_vec(), w)
    }
    fn reserved_poly_matrix(
        air: &ShiftRightPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let rp =
            <ShiftRightPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let mut vals = Vec::new();
        for r in 0..main.height() {
            let rb = main.row_slice(r);
            use std::ops::Deref;
            let row: &[F] = rb.deref();
            vals.extend(collect_reserved_poly(row, &[], &rp).into_iter().map(EF::from));
        }
        RowMajorMatrix::new(vals, rp.len())
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = ShiftRightChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_sr_first_and_nonfirst_round_evaluation_satisfied() {
        let air = ShiftRightPolyAir;
        let main = sample_trace();
        let height = main.height();
        assert!(height >= 2);

        let alpha = ef(123);
        let beta = challenge_beta();
        let bp = beta_powers();
        let bs = beta_septix(beta);
        let public = make_public_values(1);

        let pre = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
            NUM_PRECOMPUTED,
        );
        let (perm, ls) = generate_permutation_trace_(
            &air,
            None,
            &main,
            &pre,
            alpha,
            &bp,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );
        let pre_t = trim_rows(&pre, height);
        let perm_t = trim_rows(&perm, height);
        let res = reserved_poly_matrix(&air, &main);
        let red = reducer();
        let g = EF::zero();

        let first = first_round_evaluation(
            &air,
            &public,
            None,
            &main,
            &pre_t,
            &perm_t,
            alpha,
            &bp,
            bs,
            g,
            F::one(),
            F::one(),
            ls,
            BATCH_SIZE,
            &red,
        );
        assert!(first.iter().all(|x| x.is_zero()), "first_round failed: {:?}", first);

        let nonfirst = nonfirst_round_evaluation(
            &air,
            &public,
            &res,
            &pre_t,
            &perm_t,
            alpha,
            &bp,
            bs,
            g,
            EF::one(),
            EF::one(),
            ls,
            BATCH_SIZE,
            &red,
        );
        assert!(nonfirst.iter().all(|x| x.is_zero()), "nonfirst_round failed: {:?}", nonfirst);
    }

    fn random_shift_right_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace();
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

        let last_row_start = (base_height - 1) * NUM_SHIFT_RIGHT_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_SHIFT_RIGHT_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_SHIFT_RIGHT_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_SHIFT_RIGHT_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = ShiftRightPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_shift_right_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = reducer();
        let global = EF::zero();
        let reserved_poly_desc =
            <ShiftRightPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    use dt_core_executor::{Instruction, Opcode, Program};

    fn simple_sr_program() -> Program {
        // Mix of SRL/SRA with several shift amounts; vary b's MSB to cover
        // sign-extension paths and exercise b_msb=0/1.
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 0x80ff_ff00, false, true), // MSB=1
            Instruction::new(Opcode::ADD, 2, 0, 0x0000_00ff, false, true), // MSB=0
            Instruction::new(Opcode::ADD, 3, 0, 3, false, true),
            Instruction::new(Opcode::SRL, 4, 1, 2, false, true),
            Instruction::new(Opcode::SRL, 5, 1, 9, false, true),
            Instruction::new(Opcode::SRA, 6, 1, 5, false, true),
            Instruction::new(Opcode::SRA, 7, 2, 17, false, true),
            Instruction::new(Opcode::SRL, 8, 1, 3, false, false),
            Instruction::new(Opcode::SRL, 0, 1, 4, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    /// SR emits 2 BitVec lookups per real event (#43 and #44). Both mults =
    /// is_real, so total emission equals 2 × event count. Padding rows are
    /// not emitted (is_real=0 on padding ⇒ no send).
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_sr_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <ShiftRightPolyAir as MachineAir<F>>::generate_dependencies(
            &ShiftRightPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = 2 * shard.shift_right_events.len();
        assert!(expected > 0, "fixture must include SR events");
        assert_eq!(
            bitvec_total, expected,
            "BitVec BLU count must equal 2 × event count (one per BitVec lookup)",
        );
    }
}
