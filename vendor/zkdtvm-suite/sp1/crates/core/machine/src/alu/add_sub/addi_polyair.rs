//! PolyAir-optimized FullAir implementation for AddiChip.
//!
//! This module provides a `FullAir` implementation that maps the real SP1 `AddiCols`
//! layout to PolyAir's four-phase constraint model. It coexists with the original
//! `Air<AB>` implementation in `addi.rs` without modifying it.
//!
//! Key differences from AddChip/SubChip:
//! - Uses `AddiRegisterOp` adapter (2 Memory accesses: op_b + op_a, no op_c)
//! - `op_b` memory access is conditional: multiplicity = `is_reg_b` (0 when is_imm_b=1)
//! - `is_reg_b` must be reserved (used in lookup multiplicity)
//! - `op_c` is always an immediate (`op_c_imm: Word<T>`), no memory access
//! - Total: 15 lookup interactions (vs 19 for AddChip/SubChip)

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    add_polyair::add_carry_bits,
    addi::{AddiChip, AddiCols, NUM_ADDI_COLS},
};
use crate::{
    adapter::{
        register::addi_type::{
            addi_register_op_gate_constraints, addi_register_op_lookup,
            addi_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    operations::{add_op_lookup, add_op_precompute_lc},
};

// =============================================================================
// Column offset constants
// =============================================================================
// AddiCols layout:
//   add_operation: AddOperation<T>    = 4 cols  (offset  0- 3)
//   effective_b:   Word<T>            = 4 cols  (offset  4- 7)
//   memory_operations: AddiRegisterOp = 31 cols (offset  8-38)
//     op_a:            T              = 1 col   (offset  8)
//     op_a_access: MemoryReadWriteCols= 13 cols (offset  9-21)
//       prev_value:    Word<T>        = 4 cols  (offset  9-12)
//       access: MemoryAccessCols       = 9 cols  (offset 13-21)
//         value:       Word<T>        = 4 cols  (offset 13-16)
//     op_a_zero:       T              = 1 col   (offset 22)
//     op_b:            T              = 1 col   (offset 23)
//     op_b_access: MemoryReadCols     = 9 cols  (offset 24-32)
//       access: MemoryAccessCols      = 9 cols  (offset 24-32)
//         value:       Word<T>        = 4 cols  (offset 24-27)
//     op_c_imm:    Word<T>            = 4 cols  (offset 33-36)
//     is_imm_b:        T              = 1 col   (offset 37)
//     is_reg_b:        T              = 1 col   (offset 38)
//   cpu_state: CPUState<T>            = 4 cols  (offset 39-42)
//   is_real: T                        = 1 col   (offset 43)
//   TOTAL: 44 cols

/// Column offset of `is_real` (last column in AddiCols).
const COL_IS_REAL: usize = NUM_ADDI_COLS - 1;

/// Column offsets of `add_operation.value[0..3]` within AddiCols.
const COL_ADD_VALUE_0: usize = 0;
const COL_ADD_VALUE_1: usize = 1;
const COL_ADD_VALUE_2: usize = 2;
const COL_ADD_VALUE_3: usize = 3;

/// Column offsets of `effective_b[0..3]` within AddiCols.
/// Layout: add_operation(4) = 4.
const COL_EFFECTIVE_B_0: usize = 4;
const COL_EFFECTIVE_B_1: usize = 5;
const COL_EFFECTIVE_B_2: usize = 6;
const COL_EFFECTIVE_B_3: usize = 7;

/// Column offsets of `op_a_access.access.value[0..3]` within AddiCols.
/// Layout: add_operation(4) + effective_b(4) + op_a(1) + prev_value(4) = 13.
const COL_OP_A_VALUE_0: usize = 13;
const COL_OP_A_VALUE_1: usize = 14;
const COL_OP_A_VALUE_2: usize = 15;
const COL_OP_A_VALUE_3: usize = 16;

/// Column offset of `op_a_zero` within AddiCols.
const COL_OP_A_ZERO: usize = 22;

/// Column offset of `op_b` scalar within AddiCols.
const COL_OP_B: usize = 23;

/// Column offsets of `op_b_access.access.value[0..3]` (register-read b operand bytes).
const COL_OP_B_VAL_0: usize = 24;
const COL_OP_B_VAL_1: usize = 25;
const COL_OP_B_VAL_2: usize = 26;
const COL_OP_B_VAL_3: usize = 27;

/// Column offset of `is_imm_b` within AddiCols.
const COL_IS_IMM_B: usize = 37;

/// Column offset of `is_reg_b` within AddiCols.
const COL_IS_REG_B: usize = 38;

/// Column offset of `cpu_state.shard` within AddiCols.
/// Layout: add_operation(4) + effective_b(4) + memory_operations(31) = 39.
const COL_SHARD: usize = 39;

/// Public values index for `execution_shard`.
const PV_EXECUTION_SHARD_IDX: usize = 44;

/// Maximum number of values in any single lookup interaction.
/// BitVec interaction has 16 values (bit0..bit15), which is the largest.
const MAX_LOOKUP_VALUES: usize = 16;

// =============================================================================
// AddiChipPolyAir wrapper type
// =============================================================================

/// PolyAir-optimized wrapper for the SP1 AddiChip.
///
/// This type implements `FullAir` using the real `AddiCols` column layout.
/// Key distinction from AddChip/SubChip: uses `AddiRegisterOp` (2 Memory accesses)
/// and has conditional op_b memory access controlled by `is_reg_b`.
#[derive(Default, Clone, Copy)]
pub struct AddiChipPolyAir;

// =============================================================================
// FullAir implementation
// =============================================================================

impl<AB: FullAirBuilder> FullAir<AB> for AddiChipPolyAir {
    fn width(&self) -> usize {
        NUM_ADDI_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(COL_IS_REAL),      // [0]  is_real: main multiplicity
            PairCol::Main(COL_OP_A_ZERO),    // [1]  op_a_zero: used in result byte multiplicity
            PairCol::Main(COL_IS_REG_B),     // [2]  is_reg_b: multiplicity for op_b memory access
            PairCol::Main(COL_SHARD),        // [3]  shard: for shard == execution_shard gate
            PairCol::Main(COL_OP_A_VALUE_0), // [4]  op_a_value[0]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_1), // [5]  op_a_value[1]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_2), // [6]  op_a_value[2]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_3), // [7]  op_a_value[3]: rd writeback
            PairCol::Main(COL_IS_IMM_B),     /* [8]  is_imm_b: for is_reg_b relationship
                                              * constraint */
            PairCol::Main(COL_EFFECTIVE_B_0), // [9]  effective_b[0]: select gate
            PairCol::Main(COL_EFFECTIVE_B_1), // [10] effective_b[1]: select gate
            PairCol::Main(COL_EFFECTIVE_B_2), // [11] effective_b[2]: select gate
            PairCol::Main(COL_EFFECTIVE_B_3), // [12] effective_b[3]: select gate
            PairCol::Main(COL_OP_B_VAL_0),    // [13] reg_b[0]: select gate
            PairCol::Main(COL_OP_B_VAL_1),    // [14] reg_b[1]: select gate
            PairCol::Main(COL_OP_B_VAL_2),    // [15] reg_b[2]: select gate
            PairCol::Main(COL_OP_B_VAL_3),    // [16] reg_b[3]: select gate
            PairCol::Main(COL_OP_B),          // [17] op_b: imm_b[0] for select gate
            PairCol::Main(COL_ADD_VALUE_0),   // [18] add_value[0]: assert_word_eq
            PairCol::Main(COL_ADD_VALUE_1),   // [19] add_value[1]: assert_word_eq
            PairCol::Main(COL_ADD_VALUE_2),   // [20] add_value[2]: assert_word_eq
            PairCol::Main(COL_ADD_VALUE_3),   // [21] add_value[3]: assert_word_eq
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();

        // SAFETY: AddiCols is #[repr(C)] with only T-typed fields. The main trace
        // slice has exactly NUM_ADDI_COLS elements.
        let local: &AddiCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        // --- Derived values ---
        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();

        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));
        let opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::ADD as u8));

        let op_a = local.memory_operations.op_a.clone();
        let op_b = local.memory_operations.op_b.clone();
        let op_a_zero = local.memory_operations.op_a_zero.clone();
        let is_imm_b = local.memory_operations.is_imm_b.clone();
        let op_c_imm = &local.memory_operations.op_c_imm;
        let add_value = &local.add_operation.value;

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
        // #5-6: AddOperation U8Range (2 pairs for 4 result bytes)
        // =====================================================================
        add_op_precompute_lc(builder, &local.add_operation.value);

        // =====================================================================
        // #7-15: AddiRegisterOp (1 program + 4 op_b read + 4 op_a readwrite)
        // =====================================================================
        addi_register_op_precompute_lc(
            builder,
            pc,
            opcode,
            op_a,
            op_b,
            [op_c_imm[0].clone(), op_c_imm[1].clone(), op_c_imm[2].clone(), op_c_imm[3].clone()],
            op_a_zero,
            is_imm_b.clone(),
            &local.memory_operations.op_b_access.access,
            &local.memory_operations.op_a_access.access,
            &local.memory_operations.op_a_access.prev_value,
            shard,
            clk,
        );

        // =====================================================================
        // #16: BitVec — carry chain (using effective_b column) + is_real boolean
        // =====================================================================
        // Carry bits: carry[i] = (effective_b[i] + c[i] + carry[i-1] - result[i]) / 256.
        // BitVec enforces they are boolean. effective_b is now a trace column
        // (constrained against reg_b/imm_b in `eval`), so it stays degree 1 here.
        let effective_b = &local.effective_b;
        let base_inverse: <AB as FullAirBuilder>::VarMaybeExt =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(256).inverse());

        let mut carry = [AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe()];
        carry[0] = (effective_b[0].clone() + op_c_imm[0].clone() - add_value[0].clone()) *
            base_inverse.clone();
        for i in 1..4 {
            carry[i] = (effective_b[i].clone() + op_c_imm[i].clone() + carry[i - 1].clone() -
                add_value[i].clone()) *
                base_inverse.clone();
        }

        bitvec_precompute_lc(
            builder,
            vec![carry[0].clone(), carry[1].clone(), carry[2].clone(), carry[3].clone()],
        );
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[0].clone();
        let op_a_zero = local[1].clone();
        let is_reg_b = local[2].clone();
        let shard = local[3].clone();
        let op_a_value = [local[4].clone(), local[5].clone(), local[6].clone(), local[7].clone()];
        let is_imm_b = local[8].clone();
        let effective_b =
            [local[9].clone(), local[10].clone(), local[11].clone(), local[12].clone()];
        let reg_b = [local[13].clone(), local[14].clone(), local[15].clone(), local[16].clone()];
        let op_b = local[17].clone();
        let add_value =
            [local[18].clone(), local[19].clone(), local[20].clone(), local[21].clone()];

        // Replaces the implicit boolean enforcement BitVec used to provide on is_real.
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // CPUState: shard == execution_shard when is_real
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(builder, shard, execution_shard, is_real.clone());

        // AddiRegisterOp gate constraints:
        // op_a_zero, is_imm_b/is_reg_b booleans, linkage, padding
        addi_register_op_gate_constraints(
            builder,
            op_a_zero,
            op_a_value.clone(),
            is_imm_b.clone(),
            is_reg_b,
            is_real,
        );

        // effective_b select gate (unconditional; padding rows are all zero):
        //   effective_b[i] = reg_b[i] + is_imm_b * (imm_b[i] - reg_b[i])
        // where imm_b[0] = op_b and imm_b[1..3] = 0.
        // For i = 0:  effective_b[0] - reg_b[0] - is_imm_b * (op_b - reg_b[0]) = 0
        // For i > 0:  effective_b[i] - reg_b[i] + is_imm_b * reg_b[i] = 0
        builder.assert_zero(
            effective_b[0].clone() -
                reg_b[0].clone() -
                is_imm_b.clone() * (op_b - reg_b[0].clone()),
        );
        for i in 1..dt_primitives::consts::WORD_SIZE {
            builder.assert_zero(
                effective_b[i].clone() - reg_b[i].clone() + is_imm_b.clone() * reg_b[i].clone(),
            );
        }

        // assert_word_eq(op_a_value, add_operation.value):
        // unconditional — padding rows are all zero and op_a_zero rows force both sides to zero.
        for i in 0..dt_primitives::consts::WORD_SIZE {
            builder.assert_zero(op_a_value[i].clone() - add_value[i].clone());
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[0].clone();
        let op_a_zero = local[1].clone();
        let is_reg_b = local[2].clone();
        let result_mult = is_real.clone() - op_a_zero;

        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        cpu_state_lookup(builder, is_real.clone());

        // #5-6: AddOperation U8Range (mult = is_real - op_a_zero)
        add_op_lookup(builder, result_mult.clone());

        // #7-15: AddiRegisterOp (1 program + 4 op_b read + 4 op_a readwrite)
        addi_register_op_lookup(builder, is_real, is_reg_b);

        // #16: BitVec — only emit on real, non-x0 rows (4 carry bits).
        // addi_register_op's `op_a_zero => is_real = 1` guarantees mult >= 0.
        bitvec_lookup(builder, result_mult);
    }
}

// =============================================================================
// MachineAir implementation (delegation to AddiChip)
// =============================================================================

impl<F: Field> BaseAir<F> for AddiChipPolyAir {
    fn width(&self) -> usize {
        NUM_ADDI_COLS
    }
}

impl<F: Field> MachineAir<F> for AddiChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "AddiPolyAir".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <AddiChip as MachineAir<F>>::num_rows(&AddiChip, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        AddiChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.addi_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .addi_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse AddiChip path: CPUState / memory / U8Range BLU events.
                    let mut row = [F::zero(); NUM_ADDI_COLS];
                    let cols: &mut AddiCols<F> = row.as_mut_slice().borrow_mut();
                    AddiChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec on real, non-x0 rows.
                    // For ADDI, the carry chain uses `effective_b + op_c_imm`, which
                    // are the byte decompositions of `event.b` and `event.c` —
                    // structurally identical to ADD.
                    if !event.op_a_0 {
                        let (c0, c1, c2, c3) = add_carry_bits(event.b, event.c);
                        let value: u16 = (c0 as u16) |
                            ((c1 as u16) << 1) |
                            ((c2 as u16) << 2) |
                            ((c3 as u16) << 3);
                        blu.add_bit_vec_lookup(value);
                    }
                }
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <AddiChip as MachineAir<F>>::included(&AddiChip, shard)
    }

    fn local_only(&self) -> bool {
        <AddiChip as MachineAir<F>>::local_only(&AddiChip)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use std;

    use core::mem::size_of;

    use super::{
        super::addi::{AddiChip, AddiCols},
        *,
    };

    /// Total number of lookup interactions:
    /// - 1 State recv (receive_state)
    /// - 1 State send (send_state)
    /// - 2 Byte (clk range checks: U16Range + BitRange)
    /// - 2 Byte (result value U8Range: 4 bytes in 2 pairs)
    /// - 1 Program send
    /// - 4 eval_memory_access(op_b): ts_u16, ts_bit, mem_send, mem_recv
    /// - 4 eval_memory_access(op_a): ts_u16, ts_bit, mem_send, mem_recv
    /// - 1 BitVec (is_real + 4 carry bits boolean constraint)
    const NUM_LOOKUPS: usize = 16;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    use crate::alu::add_sub::tests::simple_add_sub_program;
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

    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    // =============================================================================
    // Compile-time layout validation
    // =============================================================================

    fn validate_column_layout() {
        assert_eq!(
            NUM_ADDI_COLS,
            size_of::<AddiCols<u8>>(),
            "AddiCols layout changed! NUM_ADDI_COLS ({}) != size_of::<AddiCols<u8>>() ({})",
            NUM_ADDI_COLS,
            size_of::<AddiCols<u8>>(),
        );
        assert_eq!(COL_IS_REAL, NUM_ADDI_COLS - 1);
    }

    #[test]
    fn test_column_layout_valid() {
        validate_column_layout();
        std::println!("NUM_ADDI_COLS = {}", NUM_ADDI_COLS);
    }

    // =========================================================================
    // Test Helper Functions
    // =========================================================================

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn challenge_beta() -> EF {
        EF::two_adic_generator(4) + ef(7)
    }

    fn beta_powers() -> Vec<EF> {
        let beta = challenge_beta();
        let required_max_beta_power = <AddiChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&AddiChipPolyAir);
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
        // Gate constraints: is_real_bool(1) + cpu_state(1) + addi_register_op(8) + effective_b(4) +
        // writeback(4) + sign_ext(1) = 19 Lookup batch: ceil(16/3) = 6
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 19;
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

    fn reserved_poly_matrix(air: &AddiChipPolyAir, main: &RowMajorMatrix<F>) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <AddiChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);
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

    /// Generate a sample trace with valid ADDI operations.
    fn sample_trace() -> RowMajorMatrix<F> {
        let program = simple_add_sub_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        let shard = runtime.records[0].clone();
        let chip = AddiChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    // =========================================================================
    // Constraint satisfaction tests
    // =========================================================================

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = AddiChipPolyAir;
        let main = sample_trace();
        let height = main.height();
        std::println!("trace height = {}, width = {}", height, main.width());
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

    /// Generate a random ADDI trace for performance testing.
    fn random_addi_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_ADDI_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_ADDI_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_ADDI_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }

        RowMajorMatrix::new(values, NUM_ADDI_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_first_and_nonfirst_round_evaluation_random_trace() {
        let air = AddiChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(15);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_addi_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf config: log_n={}, h={}, seed={}", log_n, height, seed);

        let alpha = ef(123);
        let beta = challenge_beta();
        let beta_powers = beta_powers();
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = reducer();
        let global = EF::zero();

        let t0 = std::time::Instant::now();
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
        std::println!("precompute_linear_combination: {:?}", t0.elapsed());

        let t1 = std::time::Instant::now();
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
        std::println!("generate_permutation_trace_: {:?}", t1.elapsed());

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main);

        let t2 = std::time::Instant::now();
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
        std::println!("first_round_evaluation: {:?}", t2.elapsed());
        assert!(first.iter().all(|x| x.is_zero()), "first_round: {:?}", first);

        let t3 = std::time::Instant::now();
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
        std::println!("nonfirst_round_evaluation: {:?}", t3.elapsed());
        assert!(nonfirst.iter().all(|x| x.is_zero()), "nonfirst_round: {:?}", nonfirst);
    }

    /// Multi-round sumcheck benchmark for AddiChip PolyAir.
    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = AddiChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_addi_trace(log_n, seed);
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
            <AddiChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    /// Program where every ADDI writes to x0 (op_a_0=true everywhere).
    fn only_x0_addis_program() -> dt_core_executor::Program {
        use dt_core_executor::{Instruction, Opcode, Program};
        let instructions = vec![
            Instruction::new(Opcode::ADD, 0, 1, 10, false, true),
            Instruction::new(Opcode::ADD, 0, 4, 255, false, true),
            Instruction::new(Opcode::ADD, 0, 6, 1000, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_skipped_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_addis_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(
            shard.addi_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all events must be op_a_0=true",
        );
        assert!(!shard.addi_events.is_empty(), "fixture must yield addi events");

        let mut deps = ExecutionRecord::default();
        <AddiChipPolyAir as MachineAir<F>>::generate_dependencies(
            &AddiChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_count: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();
        assert_eq!(bitvec_count, 0, "op_a_0=true rows must not emit BitVec");
    }

    #[test]
    fn bitvec_mult_matches_send_count() {
        use dt_core_executor::ByteOpcode;

        let program = simple_add_sub_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let mut deps = ExecutionRecord::default();
        <AddiChipPolyAir as MachineAir<F>>::generate_dependencies(
            &AddiChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected: usize = shard.addi_events.iter().filter(|(_, e)| !e.op_a_0).count();
        assert!(expected > 0, "test fixture must include non-x0 ADDIs");
        assert_eq!(bitvec_total, expected, "BitVec BLU emit count must equal lookup send count");
    }
}
