use dt_core_executor::{ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{BitwiseChip, BitwiseCols, NUM_BITWISE_COLS};
use crate::{
    adapter::{
        register::alu_type::{
            alu_type_register_op_gate_constraints, alu_type_register_op_lookup,
            alu_type_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
};

/// Public values index for `execution_shard`.
/// PublicValues<Word<u8>, u8>: committed_value_digest(32) + deferred_proofs_digest(8)
/// + start_pc(1) + next_pc(1) + exit_code(1) + shard(1) = 44.
const PV_EXECUTION_SHARD_IDX: usize = 44;

const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `BitwiseCols<u8>` (NUM_BITWISE_COLS = 47).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1]      cpu_state.clk_16_28        ← precompute-only
//   [2]      cpu_state.clk_0_16         ← precompute-only
//   [3]      cpu_state.pc               ← precompute-only
//   [4]      mem_ops.op_a               ← precompute-only
//   [5..9]   mem_ops.op_a_access.prev_value   ← precompute-only
//   [9..13]  mem_ops.op_a_access.access.value
//   [13..18] mem_ops.op_a_access.access.{prev_shard,prev_clk,compare_clk,
//                                        diff_16bit_limb,diff_12bit_limb}
//                                                   ← precompute-only
//   [18]     mem_ops.op_a_zero
//   [19]     mem_ops.op_b               ← precompute-only
//   [20..29] mem_ops.op_b_access        ← precompute-only (entirely)
//   [29..33] mem_ops.op_c
//   [33..37] mem_ops.op_c_access.access.value
//   [37..42] mem_ops.op_c_access.access.{...}      ← precompute-only
//   [42]     mem_ops.imm_c
//   [43]     is_xor
//   [44]     is_or
//   [45]     is_and
//   [46]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_C: usize = 29;
const COL_OP_C_ACCESS_VALUE: usize = 33;
const COL_IMM_C: usize = 42;
const COL_IS_XOR: usize = 43;
const COL_IS_OR: usize = 44;
const COL_IS_AND: usize = 45;
const COL_IS_REAL: usize = 46;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 19).
//
// Order columns are emitted by `reserved_poly()`. Only fields read by `eval`
// or `lookup` are retained; everything else (timestamps, prev_values,
// op_b_access, clk/pc, op_a/op_b scalars) is consumed as β-evaluations or
// retained-precomputed lookup denominators during `precompute_lc`.
//
//   [0]      is_real
//   [1]      cpu_state.shard
//   [2]      is_xor
//   [3]      is_or
//   [4]      is_and
//   [5]      op_a_zero
//   [6]      imm_c
//   [7..11]  op_a_access.access.value (Word)
//   [11..15] op_c (Word)
//   [15..19] op_c_access.access.value (Word)
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_CPU_SHARD: usize = 1;
const RES_IS_XOR: usize = 2;
const RES_IS_OR: usize = 3;
const RES_IS_AND: usize = 4;
const RES_OP_A_ZERO: usize = 5;
const RES_IMM_C: usize = 6;
const RES_OP_A_VALUE: usize = 7;
const RES_OP_C: usize = 11;
const RES_OP_C_ACCESS_VALUE: usize = 15;
const RES_NUM_COLS: usize = 19;

#[derive(Default, Clone, Copy)]
pub struct BitwiseChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for BitwiseChipPolyAir {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_IS_XOR));
        cols.push(PairCol::Main(COL_IS_OR));
        cols.push(PairCol::Main(COL_IS_AND));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_IMM_C));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_C_ACCESS_VALUE + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let zero = AB::zero_maybe();

        // SAFETY: BitwiseCols is #[repr(C)] with only T-typed fields.
        let local: &BitwiseCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        // --- Derived values ---
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
        // #5-8: Bitwise byte operations (manual — chip-specific byte_opcode)
        // =====================================================================
        // byte_opcode = is_xor * XOR + is_or * OR + is_and * AND
        let byte_opcode = local.is_xor.clone() *
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::XOR as u8)) +
            local.is_or.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::OR as u8)) +
            local.is_and.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(ByteOpcode::AND as u8));
        let byte_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
        let a_val = local.mem_ops.op_a_value();
        let b_val = local.mem_ops.op_b_value();
        let c_val = local.mem_ops.op_c_value();
        for i in 0..4 {
            builder.retain_precomputed(builder.lookup_denominator(
                byte_kind.clone(),
                vec![
                    byte_opcode.clone(),
                    a_val[i].clone(),
                    zero.clone(),
                    b_val[i].clone(),
                    c_val[i].clone(),
                ],
            ));
        }

        // =====================================================================
        // #9-21: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        // =====================================================================
        {
            let opcode = local.is_xor.clone() *
                AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::XOR as u8)) +
                local.is_or.clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::OR as u8)) +
                local.is_and.clone() *
                    AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::AND as u8));
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
        // #22: BitVec (3 selectors + op_a_zero + imm_c)
        // =====================================================================
        // is_real is dropped from the payload because the BitVec mult is now
        // conditioned on it (see `lookup`); BitVec only enforces booleanness
        // when mult ≠ 0, so is_real's booleanness is instead asserted as an
        // explicit gate in `eval`.
        bitvec_precompute_lc(
            builder,
            vec![
                local.is_xor.clone(),
                local.is_or.clone(),
                local.is_and.clone(),
                local.mem_ops.op_a_zero.clone(),
                local.mem_ops.imm_c.clone(),
            ],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();

        // Replaces the implicit boolean enforcement BitVec used to provide on is_real.
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
        alu_type_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            [
                local[RES_OP_A_VALUE].clone(),
                local[RES_OP_A_VALUE + 1].clone(),
                local[RES_OP_A_VALUE + 2].clone(),
                local[RES_OP_A_VALUE + 3].clone(),
            ],
            local[RES_IMM_C].clone(),
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

        // Chip-specific: exactly one selector active when is_real
        builder.assert_zero(
            local[RES_IS_XOR].clone() + local[RES_IS_OR].clone() + local[RES_IS_AND].clone() -
                is_real.clone(),
        );
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let imm_c = local[RES_IMM_C].clone();
        let perform_calc = is_real.clone() - local[RES_OP_A_ZERO].clone();

        // #1-4: CPUState
        cpu_state_lookup(builder, is_real.clone());

        // #5-8: Bitwise byte ops (mult = is_real - op_a_zero)
        for _ in 0..4 {
            builder.send(perform_calc.clone());
        }

        // #9-21: ALUType adapter (1 program + 4 op_b read + 4 op_c read + 4 op_a readwrite)
        alu_type_register_op_lookup(builder, is_real.clone(), imm_c);

        // #22: BitVec — mult = is_real, matching the conditioning of every
        // other lookup in the chip (CPUState, byte ops, ALUType are all
        // conditioned on is_real or a sub-expression of it). Booleanness is
        // enforced on real rows; on padding rows is_real=0 ⇒ no send, and the
        // padding template (PaddingRow::Zero) makes payload bits trivially
        // boolean (all zero).
        bitvec_lookup(builder, is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to BitwiseChip)
// =============================================================================

impl<F: Field> BaseAir<F> for BitwiseChipPolyAir {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}

impl<F: Field> MachineAir<F> for BitwiseChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "BitwisePolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        BitwiseChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .bitwise_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse BitwiseChip path: per-byte XOR/OR/AND BLU
                    // (conditioned on !op_a_0 inside event_to_row).
                    let mut row = [F::zero(); NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                    BitwiseChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec per real row.
                    // BitVec mult = is_real, so padding rows are skipped here
                    // (matching the base chip's generate_dependencies pattern).
                    // Payload bits (in precompute_lc order):
                    //   bit 0: is_xor
                    //   bit 1: is_or
                    //   bit 2: is_and
                    //   bit 3: op_a_zero
                    //   bit 4: imm_c
                    let is_xor = (event.opcode == Opcode::XOR) as u16;
                    let is_or = (event.opcode == Opcode::OR) as u16;
                    let is_and = (event.opcode == Opcode::AND) as u16;
                    let op_a_zero = event.op_a_0 as u16;
                    let imm_c = record.c.is_none() as u16;
                    let value: u16 =
                        is_xor | (is_or << 1) | (is_and << 2) | (op_a_zero << 3) | (imm_c << 4);
                    blu.add_bit_vec_lookup(value);
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <BitwiseChip as MachineAir<F>>::included(&BitwiseChip, shard)
    }

    fn local_only(&self) -> bool {
        <BitwiseChip as MachineAir<F>>::local_only(&BitwiseChip)
    }
}

#[cfg(test)]
mod tests {
    use std;

    use core::mem::size_of;

    use super::super::BitwiseChip;
    use dt_core_executor::{ExecutionRecord, Executor, Instruction, Opcode, Program};
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

    const NUM_LOOKUPS: usize = 22;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    // Column offsets used for test trace construction
    const COL_IS_REAL: usize = 46;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    fn validate_column_layout() {
        assert_eq!(
            NUM_BITWISE_COLS,
            size_of::<BitwiseCols<u8>>(),
            "BitwiseCols layout changed! NUM_BITWISE_COLS ({}) != size_of::<BitwiseCols<u8>>() ({})",
            NUM_BITWISE_COLS,
            size_of::<BitwiseCols<u8>>(),
        );
        assert_eq!(COL_IS_REAL, NUM_BITWISE_COLS - 1);
    }

    #[test]
    fn test_column_layout_valid() {
        validate_column_layout();
    }

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let required_max_beta_power = <BitwiseChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&BitwiseChipPolyAir);
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
        // Gate constraints: is_real_bool(1) + cpu_state(1) + alu_type(10) + opcode_one_hot(1) = 13
        // Lookup batch: ceil(22/3) = 8
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 13;
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
        air: &BitwiseChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <BitwiseChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    fn simple_bitwise_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 1, 0, 0xaa55_aa55, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 0x0f0f_0f0f, false, true),
            Instruction::new(Opcode::ADD, 3, 0, 0x1234_0000, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0x0000_ff00, false, true),
            Instruction::new(Opcode::ADD, 5, 0, 0xffff_0000, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 0x0f0f_f0f0, false, true),
            Instruction::new(Opcode::ADD, 7, 0, 0x1111_2222, false, true),
            Instruction::new(Opcode::ADD, 8, 0, 0x3333_4444, false, true),
            Instruction::new(Opcode::XOR, 9, 1, 2, false, false),
            Instruction::new(Opcode::XOR, 10, 1, 0x00ff_00ff, false, true),
            Instruction::new(Opcode::OR, 11, 3, 4, false, false),
            Instruction::new(Opcode::OR, 12, 3, 0x0000_00ff, false, true),
            Instruction::new(Opcode::AND, 13, 5, 6, false, false),
            Instruction::new(Opcode::AND, 14, 7, 0xffff_00ff, false, true),
            Instruction::new(Opcode::XOR, 0, 1, 2, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    fn sample_trace() -> RowMajorMatrix<F> {
        let program = simple_bitwise_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = BitwiseChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    fn random_bitwise_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_BITWISE_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_BITWISE_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_BITWISE_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_BITWISE_COLS)
    }

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = BitwiseChipPolyAir;
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

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = BitwiseChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_bitwise_trace(log_n, seed);
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
        let reserved_poly_desc = <BitwiseChipPolyAir as FullAir<
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

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    /// Bitwise emits 1 BitVec lookup per real event. Mult = is_real, so total
    /// emission equals event count. Padding rows are not emitted (is_real=0
    /// on padding ⇒ no send).
    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_bitwise_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <BitwiseChipPolyAir as MachineAir<F>>::generate_dependencies(
            &BitwiseChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected = shard.bitwise_events.len();
        assert!(expected > 0, "fixture must include bitwise events");
        assert_eq!(bitvec_total, expected, "BitVec BLU count must equal event count");
    }
}
