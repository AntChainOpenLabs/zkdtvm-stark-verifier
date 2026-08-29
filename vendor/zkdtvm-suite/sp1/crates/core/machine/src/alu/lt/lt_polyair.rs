//! PolyAir-optimized FullAir implementation for LtChip.
//!
//! This module adapts the original `Air<AB>` LT chip to PolyAir's `FullAir`
//! four-phase model while preserving the original lookup interaction order.

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{LtChip, LtCols, NUM_LT_COLS};
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
        u8_range_pair_precompute_lc,
    },
};

/// Largest lookup payload is BitVec with 16 values.
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `LtCols<u8>` (NUM_LT_COLS = 55).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1..4]   cpu_state.{clk_16_28, clk_0_16, pc}   ← precompute-only
//   [4]      mem_ops.op_a                          ← precompute-only
//   [5..9]   mem_ops.op_a_access.prev_value        ← precompute-only
//   [9..13]  mem_ops.op_a_access.access.value
//   [13..18] mem_ops.op_a_access.access.{ts fields}← precompute-only
//   [18]     mem_ops.op_a_zero
//   [19]     mem_ops.op_b                          ← precompute-only
//   [20..24] mem_ops.op_b_access.access.value
//   [24..29] mem_ops.op_b_access.access.{ts fields}← precompute-only
//   [29..33] mem_ops.op_c
//   [33..37] mem_ops.op_c_access.access.value
//   [37..42] mem_ops.op_c_access.access.{ts fields}← precompute-only
//   [42]     mem_ops.imm_c
//   [43]     is_slt
//   [44]     is_sltu
//   [45..49] lt_operation.result.byte_flags
//   [49..51] lt_operation.result.comparison_bytes
//   [51]     lt_operation.result.not_eq_inv
//   [52]     lt_operation.result.result
//   [53]     lt_operation.b_msb
//   [54]     lt_operation.c_msb
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_B_VALUE: usize = 20;
const COL_OP_C: usize = 29;
const COL_OP_C_ACCESS_VALUE: usize = 33;
const COL_IMM_C: usize = 42;
const COL_IS_SLT: usize = 43;
const COL_IS_SLTU: usize = 44;
const COL_LT_BYTE_FLAGS: usize = 45;
const COL_LT_COMPARISON_BYTES: usize = 49;
const COL_LT_NOT_EQ_INV: usize = 51;
const COL_LT_RESULT: usize = 52;
const COL_LT_B_MSB: usize = 53;
const COL_LT_C_MSB: usize = 54;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 31).
//
// Only fields read by `eval` or `lookup` are retained. `is_real` is derived
// at evaluation time as `is_slt + is_sltu`.
//
//   [0]      cpu_state.shard
//   [1]      is_slt
//   [2]      is_sltu
//   [3]      imm_c
//   [4]      op_a_zero
//   [5..9]   op_a_access.access.value (Word)
//   [9..13]  op_b_access.access.value (Word)
//   [13..17] op_c (Word)
//   [17..21] op_c_access.access.value (Word)
//   [21..25] lt_operation.result.byte_flags
//   [25..27] lt_operation.result.comparison_bytes
//   [27]     lt_operation.result.not_eq_inv
//   [28]     lt_operation.result.result
//   [29]     lt_operation.b_msb
//   [30]     lt_operation.c_msb
// ============================================================================

const RES_CPU_SHARD: usize = 0;
const RES_IS_SLT: usize = 1;
const RES_IS_SLTU: usize = 2;
const RES_IMM_C: usize = 3;
const RES_OP_A_ZERO: usize = 4;
const RES_OP_A_VALUE: usize = 5;
const RES_OP_B_VALUE: usize = 9;
const RES_OP_C: usize = 13;
const RES_OP_C_ACCESS_VALUE: usize = 17;
const RES_LT_BYTE_FLAGS: usize = 21;
const RES_LT_COMPARISON_BYTES: usize = 25;
const RES_LT_NOT_EQ_INV: usize = 27;
const RES_LT_RESULT: usize = 28;
const RES_LT_B_MSB: usize = 29;
const RES_LT_C_MSB: usize = 30;
const RES_NUM_COLS: usize = 31;

#[derive(Default, Clone, Copy)]
pub struct LtChipPolyAir;

/// Compute the BitVec payload bits for a single LT event, mirroring the
/// recurrence in `LtOperationUnsigned::populate` (`operations/lt.rs:272-309`)
/// composed with the signed→unsigned XOR rewrite from
/// `LtOperationSigned::populate` (`operations/lt.rs:152-192`).
///
/// Bit layout matches the order in `precompute_lc`:
///   bit 0..1: is_slt, is_sltu
///   bit 2..5: byte_flags[0..3] (one-hot over the first differing byte, MSB-first)
///   bit 6:    sum_flags = byte_flags[0] + byte_flags[1] + byte_flags[2] + byte_flags[3]
///   bit 7:    result = (b_eff < c_eff) as bit
#[inline]
fn lt_bitvec_value(b: u32, c: u32, opcode: Opcode) -> u16 {
    let is_slt_bit: u16 = (opcode == Opcode::SLT) as u16;
    let is_sltu_bit: u16 = (opcode == Opcode::SLTU) as u16;
    let is_signed = opcode == Opcode::SLT;

    let (b_eff, c_eff) = if is_signed { (b ^ (1u32 << 31), c ^ (1u32 << 31)) } else { (b, c) };
    let bb = b_eff.to_le_bytes();
    let cc = c_eff.to_le_bytes();

    let mut flags = [0u16; 4];
    for i in (0..4).rev() {
        if bb[i] != cc[i] {
            flags[i] = 1;
            break;
        }
    }
    let sum_flags: u16 = flags[0] + flags[1] + flags[2] + flags[3];
    let result: u16 = (b_eff < c_eff) as u16;

    is_slt_bit |
        (is_sltu_bit << 1) |
        (flags[0] << 2) |
        (flags[1] << 3) |
        (flags[2] << 4) |
        (flags[3] << 5) |
        (sum_flags << 6) |
        (result << 7)
}

impl<AB: FullAirBuilder> FullAir<AB> for LtChipPolyAir {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_IS_SLT));
        cols.push(PairCol::Main(COL_IS_SLTU));
        cols.push(PairCol::Main(COL_IMM_C));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_ACCESS_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_LT_BYTE_FLAGS + i));
        }
        for i in 0..2 {
            cols.push(PairCol::Main(COL_LT_COMPARISON_BYTES + i));
        }
        cols.push(PairCol::Main(COL_LT_NOT_EQ_INV));
        cols.push(PairCol::Main(COL_LT_RESULT));
        cols.push(PairCol::Main(COL_LT_B_MSB));
        cols.push(PairCol::Main(COL_LT_C_MSB));
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let zero = AB::zero_maybe();

        // SAFETY: LtCols is #[repr(C)] with only T-typed fields.
        let local: &LtCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        let is_slt = local.is_slt.clone();
        let is_sltu = local.is_sltu.clone();
        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        // =====================================================================
        // #1-2: MSB(b), MSB(c) — LtOperationSigned
        // =====================================================================
        msb_precompute_lc(
            builder,
            local.lt_operation.b_msb.clone(),
            local.mem_ops.op_b_access.access.value[3].clone(),
        );
        msb_precompute_lc(
            builder,
            local.lt_operation.c_msb.clone(),
            local.mem_ops.op_c_access.access.value[3].clone(),
        );

        // =====================================================================
        // #3: U8Range(diff) — LtOperationUnsigned
        // =====================================================================
        let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
        let diff = local.lt_operation.result.comparison_bytes[0].clone() -
            local.lt_operation.result.comparison_bytes[1].clone() +
            local.lt_operation.result.result.clone() * base;
        u8_range_pair_precompute_lc(builder, diff, zero.clone());

        // =====================================================================
        // #4-7: CPUState (recv_state, send_state, U16Range, BitRange)
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
        // #8-20: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        // =====================================================================
        {
            let opcode = is_slt *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::SLT as u8)) +
                is_sltu * AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::SLTU as u8));
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
        // #21: BitVec (8 bools: is_slt, is_sltu, byte_flags[0..3],
        //       sum_flags, result)
        // =====================================================================
        // `is_real = is_slt + is_sltu` is dropped from the payload because the
        // BitVec mult is now conditioned on it (see `lookup`). Booleanness of
        // `is_slt`, `is_sltu`, and their mutual exclusion are restated as
        // explicit gates in `eval` — required to keep mult well-defined on
        // padding rows where BitVec doesn't enforce.
        let sum_flags_expr = local.lt_operation.result.byte_flags[0].clone() +
            local.lt_operation.result.byte_flags[1].clone() +
            local.lt_operation.result.byte_flags[2].clone() +
            local.lt_operation.result.byte_flags[3].clone();
        bitvec_precompute_lc(
            builder,
            vec![
                local.is_slt.clone(),
                local.is_sltu.clone(),
                local.lt_operation.result.byte_flags[0].clone(),
                local.lt_operation.result.byte_flags[1].clone(),
                local.lt_operation.result.byte_flags[2].clone(),
                local.lt_operation.result.byte_flags[3].clone(),
                sum_flags_expr,
                local.lt_operation.result.result.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let one = AB::one_maybe();
        let zero = AB::zero_maybe();
        let is_slt = local[RES_IS_SLT].clone();
        let is_sltu = local[RES_IS_SLTU].clone();
        let imm_c = local[RES_IMM_C].clone();
        let is_real = is_slt.clone() + is_sltu.clone();

        // Boolean enforcement for byte_flags[0..3], sum_flags, result is provided
        // by BitVec #21 on real rows (mult = is_real). On padding (is_real=0)
        // those bits are unconstrained — safe because all their downstream uses
        // are gated by is_real and padding has all-zero mem_ops.
        //
        // However, is_slt and is_sltu must be globally boolean because
        // `msb_lookup(builder, is_slt)` would otherwise imbalance on padding.
        // Three explicit gates restate what BitVec used to provide globally:
        builder.assert_zero(is_slt.clone() * (one.clone() - is_slt.clone()));
        builder.assert_zero(is_sltu.clone() * (one.clone() - is_sltu.clone()));
        // At most one selector active ⇒ is_real = is_slt + is_sltu ∈ {0, 1}.
        builder.assert_zero(is_slt.clone() * is_sltu.clone());

        // CPUState: shard == execution_shard when is_real
        let pv = builder.public();
        const PV_EXECUTION_SHARD_IDX: usize = 44;
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // ALUType adapter gate constraints (op_a_zero, imm_c padding, op_c consistency)
        alu_type_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            [
                local[RES_OP_A_VALUE].clone(),
                local[RES_OP_A_VALUE + 1].clone(),
                local[RES_OP_A_VALUE + 2].clone(),
                local[RES_OP_A_VALUE + 3].clone(),
            ],
            imm_c.clone(),
            [
                local[RES_OP_C_ACCESS_VALUE].clone(),
                local[RES_OP_C_ACCESS_VALUE + 1].clone(),
                local[RES_OP_C_ACCESS_VALUE + 2].clone(),
                local[RES_OP_C_ACCESS_VALUE + 3].clone(),
            ],
            [
                local[RES_OP_C].clone(),
                local[RES_OP_C + 1].clone(),
                local[RES_OP_C + 2].clone(),
                local[RES_OP_C + 3].clone(),
            ],
            is_real.clone(),
        );

        let b_msb = local[RES_LT_B_MSB].clone();
        let c_msb = local[RES_LT_C_MSB].clone();

        // LtOperationSigned: when unsigned mode, MSB flags must be zero.
        builder.when(one.clone() - is_slt.clone()).assert_zero(b_msb.clone());
        builder.when(one.clone() - is_slt.clone()).assert_zero(c_msb.clone());

        // LT signed->unsigned rewrite of top byte.
        let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
        let sign_bit = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 7));
        let mut b_cmp = [
            local[RES_OP_B_VALUE].clone(),
            local[RES_OP_B_VALUE + 1].clone(),
            local[RES_OP_B_VALUE + 2].clone(),
            local[RES_OP_B_VALUE + 3].clone(),
        ];
        let mut c_cmp = [
            local[RES_OP_C_ACCESS_VALUE].clone(),
            local[RES_OP_C_ACCESS_VALUE + 1].clone(),
            local[RES_OP_C_ACCESS_VALUE + 2].clone(),
            local[RES_OP_C_ACCESS_VALUE + 3].clone(),
        ];
        b_cmp[3] = b_cmp[3].clone() + is_slt.clone() * sign_bit.clone() - base.clone() * b_msb;
        c_cmp[3] = c_cmp[3].clone() + is_slt.clone() * sign_bit - base.clone() * c_msb;

        // LtOperationUnsigned core constraints.
        let flags = [
            local[RES_LT_BYTE_FLAGS].clone(),
            local[RES_LT_BYTE_FLAGS + 1].clone(),
            local[RES_LT_BYTE_FLAGS + 2].clone(),
            local[RES_LT_BYTE_FLAGS + 3].clone(),
        ];
        let sum_flags = flags[0].clone() + flags[1].clone() + flags[2].clone() + flags[3].clone();
        let is_comp_eq = one.clone() - sum_flags;

        let mut is_inequality_visited = zero.clone();
        let mut b_comparison_limb = zero.clone();
        let mut c_comparison_limb = zero.clone();
        for i in (0..4).rev() {
            let flag = flags[i].clone();
            is_inequality_visited = is_inequality_visited + flag.clone();
            builder
                .when(is_real.clone() - is_inequality_visited.clone())
                .assert_eq(b_cmp[i].clone(), c_cmp[i].clone());
            b_comparison_limb = b_comparison_limb + b_cmp[i].clone() * flag.clone();
            c_comparison_limb = c_comparison_limb + c_cmp[i].clone() * flag;
        }

        let b_comp = local[RES_LT_COMPARISON_BYTES].clone();
        let c_comp = local[RES_LT_COMPARISON_BYTES + 1].clone();
        builder.assert_eq(b_comparison_limb, b_comp.clone());
        builder.assert_eq(c_comparison_limb, c_comp.clone());

        builder.when(one.clone() - is_comp_eq).assert_eq(
            local[RES_LT_NOT_EQ_INV].clone() * (b_comp.clone() - c_comp),
            is_real.clone(),
        );

        let lt_result = local[RES_LT_RESULT].clone();
        let expected_lt_word = [lt_result, AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe()];

        // Final ALU correctness: op_a must equal the LT result word.
        // Gate on perform_calc (not is_real) to skip when rd=x0, where adapter forces op_a=0.
        let perform_calc = is_real.clone() - local[RES_OP_A_ZERO].clone();
        for (i, expected) in expected_lt_word.into_iter().enumerate() {
            builder
                .when(perform_calc.clone())
                .assert_eq(local[RES_OP_A_VALUE + i].clone(), expected);
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_slt = local[RES_IS_SLT].clone();
        let is_sltu = local[RES_IS_SLTU].clone();
        let imm_c = local[RES_IMM_C].clone();
        let is_real = is_slt.clone() + is_sltu;

        // Order matches precompute_lc.

        // #1-2: MSB(b), MSB(c)
        msb_lookup(builder, is_slt.clone());
        msb_lookup(builder, is_slt);

        // #3: U8Range(diff)
        slice_u8_range_lookup(builder, is_real.clone(), 1);

        // #4-7: CPUState
        cpu_state_lookup(builder, is_real.clone());

        // #8-20: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        alu_type_register_op_lookup(builder, is_real.clone(), imm_c);

        // #21: BitVec — emit only on real rows.
        // is_real ∈ {0, 1} by the three explicit gates added in `eval`
        // (is_slt boolean, is_sltu boolean, is_slt * is_sltu = 0).
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to LtChip)
// =============================================================================

impl<F: Field> BaseAir<F> for LtChipPolyAir {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }
}

impl<F: Field> MachineAir<F> for LtChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "LtPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        LtChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.lt_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .lt_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse LtChip path: cpu_state, mem_ops, lt_operation BLU
                    // (MSB + U8Range(diff)). lt_operation.populate runs even when
                    // op_a_0=true (synthesizes the comparison result), so all
                    // BLU it emits are well-defined regardless.
                    let mut row = [F::zero(); NUM_LT_COLS];
                    let cols: &mut LtCols<F> = row.as_mut_slice().borrow_mut();
                    LtChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec on every real row
                    // (mult = is_real, populate always runs for lt_events).
                    let value = lt_bitvec_value(event.b, event.c, event.opcode);
                    blu.add_bit_vec_lookup(value);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <LtChip as MachineAir<F>>::included(&LtChip, shard)
    }

    fn local_only(&self) -> bool {
        <LtChip as MachineAir<F>>::local_only(&LtChip)
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;
    use std;

    use dt_core_executor::{ExecutionRecord, Executor, Instruction, Program};
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

    use super::{super::LtChip, *};

    /// Total number of lookup interactions:
    /// - LtOperationSigned::eval(): 3 (2 MSB + 1 U8Range)
    /// - CPUState::eval(): 4 (State recv/send + clk U16Range/BitRange)
    /// - ALUTypeRegisterOp::eval(): 13 (1 Program + 3 Memory * 4)
    /// - BitVec boolean: 1 (is_slt, is_sltu, byte_flags[0..3], result = 7 bools)
    const NUM_LOOKUPS: usize = 21;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_lt_column_layout() {
        assert_eq!(
            NUM_LT_COLS,
            size_of::<LtCols<u8>>(),
            "LtCols layout changed! NUM_LT_COLS ({}) != size_of::<LtCols<u8>>() ({})",
            NUM_LT_COLS,
            size_of::<LtCols<u8>>(),
        );
    }

    const PV_EXECUTION_SHARD_IDX: usize = 44;

    /// Build public values with execution_shard set at the expected index.
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
        let required_max_beta_power = <LtChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&LtChipPolyAir);
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
        // Gate constraints: cpu_state(1) + alu_type(10) + lt_specific(13)
        //                   + is_slt_bool(1) + is_sltu_bool(1) + selector_one_hot(1) = 27
        // Lookup batch: ceil(21/3) = 7
        // Cumulative sum: 3
        // Total: 27 + 7 + 3 = 37
        const NUM_GATE_CONSTRAINTS: usize = 27;
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

    fn reserved_poly_matrix(air: &LtChipPolyAir, main: &RowMajorMatrix<F>) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <LtChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    fn simple_lt_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 10, false, true),
            Instruction::new(Opcode::ADD, 3, 0, 0xffff_ffff, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0x8000_0000, false, true),
            Instruction::new(Opcode::SLT, 10, 1, 2, false, false),
            Instruction::new(Opcode::SLT, 11, 3, 1, false, false),
            Instruction::new(Opcode::SLTU, 12, 1, 2, false, false),
            Instruction::new(Opcode::SLTU, 13, 4, 1, false, false),
            Instruction::new(Opcode::SLT, 14, 1, 6, false, true),
            Instruction::new(Opcode::SLT, 15, 3, 0, false, true),
            Instruction::new(Opcode::SLTU, 16, 1, 6, false, true),
            Instruction::new(Opcode::SLTU, 17, 4, 0x7fff_ffff, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        // let program = simple_lt_program();
        use crate::programs::tests::keccak_program;
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = LtChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = LtChipPolyAir;
        let main = sample_trace();
        let height = main.height();
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
        assert!(first.iter().all(|x| x.is_zero()), "first_round_evaluation failed: {:?}", first);

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
            "nonfirst_round_evaluation failed: {:?}",
            nonfirst
        );
    }

    fn random_lt_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_LT_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_LT_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_LT_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_LT_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = LtChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_lt_trace(log_n, seed);
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
            <LtChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    /// Every lt event emits one BitVec — `lt_operation.populate` runs
    /// unconditionally (op_a_0=true rows synthesize the comparison result), so
    /// the BitVec mult conditioning is just `is_real`.
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_lt_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <LtChipPolyAir as MachineAir<F>>::generate_dependencies(&LtChipPolyAir, &shard, &mut deps);

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = shard.lt_events.len();
        assert!(expected > 0, "fixture must include lt events");
        assert_eq!(bitvec_total, expected, "BitVec BLU emit count must equal lookup send count");
    }

    /// Even when every LT writes to x0, BitVec still emits — unlike ADD/SUB
    /// where AddOperation::populate is skipped, LtOperation::populate runs
    /// regardless of op_a_0.
    fn only_x0_lt_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 10, false, true),
            Instruction::new(Opcode::SLT, 0, 1, 2, false, false),
            Instruction::new(Opcode::SLTU, 0, 1, 2, false, false),
            Instruction::new(Opcode::SLT, 0, 2, 1, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_emitted_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_lt_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(
            shard.lt_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all lt events must be op_a_0=true",
        );
        let expected = shard.lt_events.len();
        assert!(expected > 0, "fixture must yield lt events");

        let mut deps = ExecutionRecord::default();
        <LtChipPolyAir as MachineAir<F>>::generate_dependencies(&LtChipPolyAir, &shard, &mut deps);

        let bitvec_count: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();
        assert_eq!(bitvec_count, expected, "op_a_0=true rows must still emit BitVec for lt chip");
    }

    /// Sanity: helper value matches the recurrence for a known case.
    /// SLTU 5 < 10 → byte_flags = [1,0,0,0] (LSB differs first MSB-first scan),
    /// sum_flags = 1, result = 1, is_sltu = 1, is_slt = 0.
    #[test]
    fn lt_bitvec_value_sltu_lt() {
        let v = lt_bitvec_value(5, 10, Opcode::SLTU);
        // bit 0: is_slt = 0
        // bit 1: is_sltu = 1
        // bits 2..5: byte_flags one-hot at position 0 (MSB-first finds byte 0 differs)
        // bit 6: sum_flags = 1
        // bit 7: result = 1
        let expected: u16 = (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7);
        assert_eq!(v, expected, "got 0b{:08b}, expected 0b{:08b}", v, expected);
    }
}
