use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{ShiftLeft, ShiftLeftCols, NUM_SHIFT_LEFT_COLS};
use crate::{
    adapter::{
        register::alu_type::{alu_type_register_op_gate_constraints, alu_type_register_op_lookup},
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{
        bitvec_lookup, bitvec_precompute_lc, slice_u8_range_lookup, slice_u8_range_precompute_lc,
    },
};

/// Public values index for `execution_shard`.
const PV_EXECUTION_SHARD_IDX: usize = 44;

const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `ShiftLeftCols<u8>` (NUM_SHIFT_LEFT_COLS = 73).
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
//   [43..51] c_least_sig_byte
//   [51..59] shift_by_n_bits
//   [59]     bit_shift_multiplier
//   [60..64] bit_shift_result
//   [64..68] bit_shift_result_carry
//   [68..72] shift_by_n_bytes
//   [72]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_B_VALUE: usize = 20;
const COL_OP_C: usize = 29;
const COL_OP_C_ACCESS_VALUE: usize = 33;
const COL_IMM_C: usize = 42;
const COL_C_LEAST_SIG_BYTE: usize = 43;
const COL_SHIFT_BY_N_BITS: usize = 51;
const COL_BIT_SHIFT_MULTIPLIER: usize = 59;
const COL_BIT_SHIFT_RESULT: usize = 60;
const COL_BIT_SHIFT_RESULT_CARRY: usize = 64;
const COL_SHIFT_BY_N_BYTES: usize = 68;
const COL_IS_REAL: usize = 72;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 49).
//
// Only fields read by `eval` or `lookup` are retained.
//
//   [0]      is_real
//   [1]      cpu_state.shard
//   [2]      op_a_zero
//   [3]      imm_c
//   [4..8]   op_a_access.access.value
//   [8..12]  op_b_access.access.value
//   [12..16] op_c_access.access.value
//   [16..20] op_c (Word)
//   [20..28] c_least_sig_byte
//   [28..36] shift_by_n_bits
//   [36]     bit_shift_multiplier
//   [37..41] bit_shift_result
//   [41..45] bit_shift_result_carry
//   [45..49] shift_by_n_bytes
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_CPU_SHARD: usize = 1;
const RES_OP_A_ZERO: usize = 2;
const RES_IMM_C: usize = 3;
const RES_OP_A_VALUE: usize = 4;
const RES_OP_B_VALUE: usize = 8;
const RES_OP_C_ACCESS_VALUE: usize = 12;
const RES_OP_C: usize = 16;
const RES_C_LEAST_SIG_BYTE: usize = 20;
const RES_SHIFT_BY_N_BITS: usize = 28;
const RES_BIT_SHIFT_MULTIPLIER: usize = 36;
const RES_BIT_SHIFT_RESULT: usize = 37;
const RES_BIT_SHIFT_RESULT_CARRY: usize = 41;
const RES_SHIFT_BY_N_BYTES: usize = 45;
const RES_NUM_COLS: usize = 49;

#[derive(Default, Clone, Copy)]
pub struct ShiftLeftPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for ShiftLeftPolyAir {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_IMM_C));
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
        cols.push(PairCol::Main(COL_BIT_SHIFT_MULTIPLIER));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_BIT_SHIFT_RESULT + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_BIT_SHIFT_RESULT_CARRY + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_SHIFT_BY_N_BYTES + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShiftLeftCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

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
        // #5-6: U8Range on bit_shift_result (2 pairs)
        // =====================================================================
        slice_u8_range_precompute_lc(builder, &local.bit_shift_result);

        // =====================================================================
        // #7-8: U8Range on bit_shift_result_carry (2 pairs)
        // =====================================================================
        slice_u8_range_precompute_lc(builder, &local.bit_shift_result_carry);

        // =====================================================================
        // #9-21: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        // =====================================================================
        {
            let opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::SLL as u8));
            crate::adapter::register::alu_type::alu_type_register_op_precompute_lc(
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
        // #22: BitVec [c_least_sig_byte[0..8], shift_by_n_bits[0..8]] (16 bits)
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
        // #23: BitVec [shift_by_n_bytes[0..4]] (4 bits)
        // =====================================================================
        // is_real is dropped from the payload because the BitVec mult is now
        // conditioned on it (see `lookup`); BitVec only enforces booleanness
        // when mult ≠ 0, so is_real's booleanness is instead asserted as an
        // explicit gate in `eval`.
        let mut bits_2 = Vec::with_capacity(4);
        for i in 0..4 {
            bits_2.push(local.shift_by_n_bytes[i].clone());
        }
        bitvec_precompute_lc(builder, bits_2);
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        // Replaces the implicit boolean enforcement BitVec #23 used to provide on is_real.
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

        // Reconstruct c[0] from bit decomposition
        let mut reconstructed_c0 = AB::zero_maybe();
        for i in 0..8 {
            reconstructed_c0 = reconstructed_c0 +
                local[RES_C_LEAST_SIG_BYTE + i].clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << i));
        }
        builder.assert_zero(reconstructed_c0 - local[RES_OP_C_ACCESS_VALUE].clone());

        // bit_shift_amount = c_bits[0..3]
        let bit_shift_amount = local[RES_C_LEAST_SIG_BYTE].clone() +
            local[RES_C_LEAST_SIG_BYTE + 1].clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(2)) +
            local[RES_C_LEAST_SIG_BYTE + 2].clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(4));

        // shift_by_n_bits one-hot constraints
        let mut sum_bit_flags = AB::zero_maybe();
        for i in 0..8 {
            let flag = local[RES_SHIFT_BY_N_BITS + i].clone();
            builder.assert_zero(
                flag.clone() *
                    (bit_shift_amount.clone() -
                        AB::VarMaybeExt::from(AB::F::from_canonical_usize(i))),
            );
            builder.assert_zero(
                flag.clone() *
                    (local[RES_BIT_SHIFT_MULTIPLIER].clone() -
                        AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << i))),
            );
            sum_bit_flags = sum_bit_flags + flag;
        }
        builder.assert_zero(AB::one_maybe() - sum_bit_flags);

        // Bit shift: b[i] * multiplier + carry[i-1] - carry[i]*256 = result[i]
        let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
        let multiplier = local[RES_BIT_SHIFT_MULTIPLIER].clone();
        for i in 0..4 {
            let mut v = local[RES_OP_B_VALUE + i].clone() * multiplier.clone() -
                local[RES_BIT_SHIFT_RESULT_CARRY + i].clone() * base.clone();
            if i > 0 {
                v = v + local[RES_BIT_SHIFT_RESULT_CARRY + i - 1].clone();
            }
            builder.assert_zero(local[RES_BIT_SHIFT_RESULT + i].clone() - v);
        }

        // byte_shift_amount = c_bits[3..5)
        let byte_shift_amount = local[RES_C_LEAST_SIG_BYTE + 3].clone() +
            local[RES_C_LEAST_SIG_BYTE + 4].clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(2));
        let mut sum_byte_flags = AB::zero_maybe();
        for i in 0..4 {
            let flag = local[RES_SHIFT_BY_N_BYTES + i].clone();
            builder.assert_zero(
                flag.clone() *
                    (byte_shift_amount.clone() -
                        AB::VarMaybeExt::from(AB::F::from_canonical_usize(i))),
            );
            sum_byte_flags = sum_byte_flags + flag;
        }
        builder.assert_zero(AB::one_maybe() - sum_byte_flags);

        // Byte shift result constraints: when performing calc and shift_by_n_bytes[n]:
        //   a[i] = 0 for i < n, a[i] = bit_shift_result[i-n] for i >= n
        let perform_calc = is_real - local[RES_OP_A_ZERO].clone();
        for n in 0..4 {
            for i in 0..4 {
                let guard = perform_calc.clone() * local[RES_SHIFT_BY_N_BYTES + n].clone();
                if i < n {
                    builder.when(guard).assert_zero(a_word[i].clone());
                } else {
                    builder.assert_zero(
                        guard * (a_word[i].clone() - local[RES_BIT_SHIFT_RESULT + i - n].clone()),
                    );
                }
            }
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

        // #5-6: U8Range on bit_shift_result (mult = is_real)
        slice_u8_range_lookup(builder, is_real.clone(), 2);

        // #7-8: U8Range on bit_shift_result_carry (mult = is_real)
        slice_u8_range_lookup(builder, is_real.clone(), 2);

        // #9-21: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        alu_type_register_op_lookup(builder, is_real.clone(), imm_c);

        // #22: BitVec [c_least_sig_byte + shift_by_n_bits]
        // Mult = is_real, matching the conditioning pattern of every other
        // lookup in the chip (CPUState, U8Range, ALUType, etc. are all
        // conditioned on is_real). On real rows, BitVec enforces booleanness
        // of the payload bits. On padding rows is_real=0 ⇒ no send, so no
        // padding-row BLU emission is needed in generate_dependencies. The
        // padding template (mod.rs:133-140) sets the relevant columns to
        // boolean values structurally, so soundness is preserved.
        bitvec_lookup(builder, is_real.clone());

        // #23: BitVec [shift_by_n_bytes] — same conditioning rationale as #22.
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to ShiftLeft)
// =============================================================================

impl<F: Field> BaseAir<F> for ShiftLeftPolyAir {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }
}

impl<F: Field> MachineAir<F> for ShiftLeftPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "ShiftLeftPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShiftLeft.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.shift_left_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .shift_left_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse base chip path: all BLU that ShiftLeft already emits.
                    let mut row = [F::zero(); NUM_SHIFT_LEFT_COLS];
                    let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
                    ShiftLeft.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec #22 + #23 per real row.
                    // Mirrors event_to_row's bit decomposition of event.c.
                    // BitVec mult = is_real, so padding rows are skipped here
                    // (matching the base chip's generate_dependencies pattern,
                    // which also only iterates real events).
                    let shamt = (event.c & 0x1F) as usize;
                    let bit_shift = shamt % 8;
                    let byte_shift = shamt / 8;

                    // #22: bits 0..8 ← c_least_sig_byte[i] = (event.c >> i) & 1
                    //      bits 8..16 ← shift_by_n_bits[i] = (bit_shift == i)
                    let c_low = (event.c & 0xFF) as u16;
                    let value_22 = c_low | ((1u16 << bit_shift) << 8);
                    blu.add_bit_vec_lookup(value_22);

                    // #23: bits 0..4 ← shift_by_n_bytes[i] = (byte_shift == i)
                    let value_23 = 1u16 << byte_shift;
                    blu.add_bit_vec_lookup(value_23);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShiftLeft as MachineAir<F>>::included(&ShiftLeft, shard)
    }

    fn padding_row(&self) -> Vec<F> {
        ShiftLeft.padding_row()
    }

    fn local_only(&self) -> bool {
        <ShiftLeft as MachineAir<F>>::local_only(&ShiftLeft)
    }
}

#[cfg(test)]
mod tests {
    use std;

    use super::*;

    /// Total number of lookup interactions:
    /// - 4 CPUState (state recv, state send, clk U16Range, clk BitRange)
    /// - 2 U8Range (bit_shift_result: 4 bytes → 2 pairs)
    /// - 2 U8Range (bit_shift_result_carry: 4 bytes → 2 pairs)
    /// - 1 Program send (manual, ALUType payload)
    /// - 4 Memory read (op_b)
    /// - 4 Memory read (op_c)
    /// - 4 Memory readwrite (op_a)
    /// - 2 BitVec (c_least_sig_byte+shift_by_n_bits; is_real+shift_by_n_bytes)
    const NUM_LOOKUPS: usize = 23;
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

    use super::super::ShiftLeft;

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
        let n = <ShiftLeftPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&ShiftLeftPolyAir);
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
        // Gate constraints: is_real_bool(1) + cpu_state(1) + alu_type(10) + sll_specific(43) = 55
        // Lookup batch: ceil(23/3) = 8
        // Cumulative sum: 3
        // Total: 55 + 8 + 3 = 66
        const NUM_GATE_CONSTRAINTS: usize = 55;
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
        air: &ShiftLeftPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <ShiftLeftPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let row_binding = main.row_slice(row_idx);
            use std::ops::Deref;
            let row: &[F] = row_binding.deref();
            let reserved = collect_reserved_poly(row, &[], &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = ShiftLeft;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_sll_first_and_nonfirst_round_evaluation_satisfied() {
        let air = ShiftLeftPolyAir;
        let main = sample_trace();
        let height = main.height();
        assert!(height >= 2);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);

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
        assert!(first.iter().all(|x| x.is_zero()), "first_round failed: {:?}", first);

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
        assert!(nonfirst.iter().all(|x| x.is_zero()), "nonfirst_round failed: {:?}", nonfirst);
    }

    fn random_shift_left_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_SHIFT_LEFT_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_SHIFT_LEFT_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_SHIFT_LEFT_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_SHIFT_LEFT_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = ShiftLeftPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_shift_left_trace(log_n, seed);
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
            <ShiftLeftPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    fn simple_sll_program() -> Program {
        // Mix of immediate / register / x0-writeback SLLs covering several
        // shift amounts so bit_shift and byte_shift both vary.
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 0x0000_00ff, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 7, false, true),
            Instruction::new(Opcode::ADD, 3, 0, 12, false, true),
            Instruction::new(Opcode::SLL, 4, 1, 3, false, true),
            Instruction::new(Opcode::SLL, 5, 1, 2, false, false),
            Instruction::new(Opcode::SLL, 6, 1, 3, false, false),
            Instruction::new(Opcode::SLL, 7, 1, 12, false, true),
            Instruction::new(Opcode::SLL, 0, 1, 5, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    /// SLL emits 2 BitVec lookups per real event (one for #22, one for #23).
    /// Both BitVec mults = is_real, so total emission equals 2 × event count.
    /// Padding rows are not emitted (is_real=0 on padding ⇒ no send), matching
    /// the base chip's generate_dependencies pattern.
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_sll_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <ShiftLeftPolyAir as MachineAir<F>>::generate_dependencies(
            &ShiftLeftPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = 2 * shard.shift_left_events.len();
        assert!(expected > 0, "fixture must include SLL events");
        assert_eq!(
            bitvec_total, expected,
            "BitVec BLU count must equal 2 × event count (one per BitVec lookup)",
        );
    }
}
