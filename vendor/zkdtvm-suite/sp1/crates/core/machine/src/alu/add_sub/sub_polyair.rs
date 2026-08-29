//! PolyAir-optimized FullAir implementation for SubChip.
//!
//! Uses co-located helpers for sub-operation constraints:
//! - `cpu_state_*` (4 interactions): state recv/send + clk range checks
//! - `add_op_*` (2 interactions): result U8Range checks
//! - `rtype_register_op_*` (13 interactions): program + 3×4 memory access
//! - `bitvec_*` (1 interaction): is_real + 4 borrow bits boolean
//!
//! SubChip and AddChip share identical column layouts (SubCols ≡ AddCols structurally).
//! The only difference is the opcode constant (SUB vs ADD) and borrow vs carry.

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::sub::{SubChip, SubCols, NUM_SUB_COLS};
use crate::{
    adapter::{
        register::r_type::{
            rtype_register_op_gate_constraints, rtype_register_op_lookup,
            rtype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    bytes::polyair::{bitvec_lookup, bitvec_precompute_lc},
    operations::{add_op_lookup, add_op_precompute_lc},
};

// =============================================================================
// Column offset constants (only for reserved_poly)
// =============================================================================

/// Column offset of `is_real` (last column in SubCols).
const COL_IS_REAL: usize = NUM_SUB_COLS - 1;

/// Column offsets of `sub_operation.value[0..3]` within SubCols.
const COL_SUB_VALUE_0: usize = 0;
const COL_SUB_VALUE_1: usize = 1;
const COL_SUB_VALUE_2: usize = 2;
const COL_SUB_VALUE_3: usize = 3;

/// Column offsets of `op_a_access.access.value[0..3]` within SubCols.
/// Layout: sub_operation(4) + op_a(1) + prev_value(4) = 9.
const COL_OP_A_VALUE_0: usize = 9;
const COL_OP_A_VALUE_1: usize = 10;
const COL_OP_A_VALUE_2: usize = 11;
const COL_OP_A_VALUE_3: usize = 12;

/// Column offset of `op_a_zero` within SubCols.
/// Layout: sub_operation(4) + op_a(1) + op_a_access(13) = 18.
const COL_OP_A_ZERO: usize = 18;

/// Column offset of `cpu_state.shard` within SubCols.
/// Layout: sub_operation(4) + memory_operations(35) = 39.
const COL_SHARD: usize = 39;

/// Public values index for `execution_shard`.
const PV_EXECUTION_SHARD_IDX: usize = 44;

/// Maximum number of values in any single lookup interaction.
/// BitVec interaction has 16 values (bit0..bit15), which is the largest.
const MAX_LOOKUP_VALUES: usize = 16;

// =============================================================================
// SubChipPolyAir wrapper type
// =============================================================================

/// PolyAir-optimized wrapper for the SP1 SubChip.
///
/// This type implements `FullAir` using the real `SubCols` column layout,
/// mapping SP1's constraint and interaction patterns to PolyAir's four-phase model.
/// The column layout is structurally identical to `AddCols`; only the opcode differs.
#[derive(Default, Clone, Copy)]
pub struct SubChipPolyAir;

/// Compute the 4 borrow-chain bits of `b - c`, mirroring `SubOperation::eval`'s
/// recurrence: `carry[i] = (b[i] + 255 - c[i] - value[i] + carry[i-1]) * 256^-1`
/// with `carry[-1] = 1` and `value = wrapping_sub(b, c)`.
#[inline]
fn sub_borrow_bits(b: u32, c: u32) -> (u8, u8, u8, u8) {
    let bb = b.to_le_bytes();
    let cc = c.to_le_bytes();
    let value = b.wrapping_sub(c).to_le_bytes();

    let mut carry = [0u8; 4];
    let mut prev: i32 = 1;
    for i in 0..4 {
        let num = bb[i] as i32 + 255 - cc[i] as i32 - value[i] as i32 + prev;
        debug_assert!(num >= 0 && num % 256 == 0, "sub borrow numerator must be 0 or 256");
        carry[i] = (num / 256) as u8;
        debug_assert!(carry[i] < 2, "borrow bit must be boolean");
        prev = carry[i] as i32;
    }
    (carry[0], carry[1], carry[2], carry[3])
}

// =============================================================================
// FullAir implementation
// =============================================================================

impl<AB: FullAirBuilder> FullAir<AB> for SubChipPolyAir {
    fn width(&self) -> usize {
        NUM_SUB_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(COL_IS_REAL),      // [0] is_real: main multiplicity
            PairCol::Main(COL_OP_A_ZERO),    // [1] op_a_zero: used in conditional constraints
            PairCol::Main(COL_SHARD),        // [2] shard: for shard == execution_shard gate
            PairCol::Main(COL_OP_A_VALUE_0), // [3] op_a_value[0]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_1), // [4] op_a_value[1]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_2), // [5] op_a_value[2]: rd writeback
            PairCol::Main(COL_OP_A_VALUE_3), // [6] op_a_value[3]: rd writeback
            PairCol::Main(COL_SUB_VALUE_0),  // [7] sub_value[0]: SubOperation result
            PairCol::Main(COL_SUB_VALUE_1),  // [8] sub_value[1]: SubOperation result
            PairCol::Main(COL_SUB_VALUE_2),  // [9] sub_value[2]: SubOperation result
            PairCol::Main(COL_SUB_VALUE_3),  // [10] sub_value[3]: SubOperation result
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();

        // SAFETY: SubCols is #[repr(C)] with only T-typed fields. The main trace
        // slice has exactly NUM_SUB_COLS elements.
        let local: &SubCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        // --- Derived values ---
        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc = local.cpu_state.pc.clone();

        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc = pc.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));
        let opcode = AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::SUB as u8));

        let op_a = local.memory_operations.op_a.clone();
        let op_b = local.memory_operations.op_b.clone();
        let op_c = local.memory_operations.op_c.clone();
        let op_a_zero = local.memory_operations.op_a_zero.clone();

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
        // #5-6: SubOperation U8Range (2 pairs for 4 result bytes)
        // =====================================================================
        add_op_precompute_lc(builder, &local.add_operation.value);

        // =====================================================================
        // #7-19: RTypeRegisterOp (1 program + 3x4 memory)
        // =====================================================================
        rtype_register_op_precompute_lc(
            builder,
            pc,
            opcode,
            op_a,
            op_b,
            op_c,
            op_a_zero,
            &local.memory_operations.op_b_access.access,
            &local.memory_operations.op_c_access.access,
            &local.memory_operations.op_a_access.access,
            &local.memory_operations.op_a_access.prev_value,
            shard,
            clk,
        );

        // =====================================================================
        // #20: BitVec — derived borrow chain + is_real boolean
        // =====================================================================
        // Borrow bits computed algebraically from the main trace; BitVec enforces
        // they are boolean. This is the "derived carry pattern" — no gate
        // constraints needed because the borrow is uniquely determined.
        //
        // SUB: result = b - c (wrapping). Borrow chain:
        //   carry[0] = (b[0] + 256 - c[0] - value[0]) / 256
        //   carry[i] = (b[i] + 255 - c[i] - value[i] + carry[i-1]) / 256
        let sub_value = &local.add_operation.value;
        let op_b_val = &local.memory_operations.op_b_access.access.value;
        let op_c_val = &local.memory_operations.op_c_access.access.value;
        let base_inverse: AB::VarMaybeExt =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(256).inverse());
        let base_val = AB::VarMaybeExt::from(AB::F::from_canonical_u32(256));
        let base_minus_one = AB::VarMaybeExt::from(AB::F::from_canonical_u32(255));

        let mut carry = [AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe()];
        carry[0] = (op_b_val[0].clone() + base_val - op_c_val[0].clone() - sub_value[0].clone()) *
            base_inverse.clone();
        for i in 1..4 {
            carry[i] = (op_b_val[i].clone() + base_minus_one.clone() -
                op_c_val[i].clone() -
                sub_value[i].clone() +
                carry[i - 1].clone()) *
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
        let shard = local[2].clone();
        let op_a_value = [local[3].clone(), local[4].clone(), local[5].clone(), local[6].clone()];
        let sub_value = [local[7].clone(), local[8].clone(), local[9].clone(), local[10].clone()];

        // Replaces the implicit boolean enforcement BitVec used to provide on is_real.
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // CPUState: shard == execution_shard when is_real
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(builder, shard, execution_shard, is_real.clone());

        // RType: op_a_zero only on real rows + x0 register must be zero
        rtype_register_op_gate_constraints(builder, op_a_zero, op_a_value, is_real);

        // SubChip constrains the rd writeback word to equal the computed subtraction result.
        for i in 0..4 {
            builder.assert_zero(local[0].clone() * (local[3 + i].clone() - sub_value[i].clone()));
        }

        // Note: Borrow-based subtraction constraints use the derived carry pattern —
        // borrows are computed algebraically in precompute_lc() and enforced
        // boolean via BitVec lookup. No gate constraints needed here.
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[0].clone();
        let op_a_zero = local[1].clone();
        let result_mult = is_real.clone() - op_a_zero;

        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        cpu_state_lookup(builder, is_real.clone());

        // #5-6: SubOperation U8Range (mult = is_real - op_a_zero)
        add_op_lookup(builder, result_mult.clone());

        // #7-19: RTypeRegisterOp (1 program + 3x4 memory)
        rtype_register_op_lookup(builder, is_real);

        // #20: BitVec — only emit on real, non-x0 rows (4 borrow bits).
        // r_type's `op_a_zero => is_real = 1` guarantees mult >= 0.
        bitvec_lookup(builder, result_mult);
    }
}

// =============================================================================
// MachineAir implementation (delegation to SubChip)
// =============================================================================

impl<F: Field> BaseAir<F> for SubChipPolyAir {
    fn width(&self) -> usize {
        NUM_SUB_COLS
    }
}

impl<F: Field> MachineAir<F> for SubChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "SubPolyAir".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <SubChip as MachineAir<F>>::num_rows(&SubChip, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        SubChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.sub_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .sub_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse SubChip path: CPUState / memory / U8Range BLU events.
                    let mut row = [F::zero(); NUM_SUB_COLS];
                    let cols: &mut SubCols<F> = row.as_mut_slice().borrow_mut();
                    SubChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec on real, non-x0 rows.
                    if !event.op_a_0 {
                        let (c0, c1, c2, c3) = sub_borrow_bits(event.b, event.c);
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
        <SubChip as MachineAir<F>>::included(&SubChip, shard)
    }

    fn local_only(&self) -> bool {
        <SubChip as MachineAir<F>>::local_only(&SubChip)
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
        super::sub::{SubChip, SubCols},
        *,
    };

    /// Total number of lookup interactions (identical to AddChip):
    /// - 1 State recv (receive_state)
    /// - 1 State send (send_state)
    /// - 2 Byte (clk range checks: U16Range + BitRange)
    /// - 2 Byte (result value U8Range: 4 bytes in 2 pairs)
    /// - 1 Program send
    /// - 4 eval_memory_access(op_b): ts_u16, ts_bit, mem_send, mem_recv
    /// - 4 eval_memory_access(op_c): ts_u16, ts_bit, mem_send, mem_recv
    /// - 4 eval_memory_access(op_a): ts_u16, ts_bit, mem_send, mem_recv
    /// - 1 BitVec (is_real + 4 borrow bits boolean constraint)
    const NUM_LOOKUPS: usize = 20;
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
            NUM_SUB_COLS,
            size_of::<SubCols<u8>>(),
            "SubCols layout changed! NUM_SUB_COLS ({}) != size_of::<SubCols<u8>>() ({})",
            NUM_SUB_COLS,
            size_of::<SubCols<u8>>(),
        );
        assert_eq!(COL_IS_REAL, NUM_SUB_COLS - 1);
    }

    #[test]
    fn test_column_layout_valid() {
        validate_column_layout();
        std::println!("NUM_SUB_COLS = {}", NUM_SUB_COLS);
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
        let required_max_beta_power = <SubChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&SubChipPolyAir);
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
        // Gate constraints: is_real_bool(1) + cpu_state(1) + rtype_register_op(5) +
        // sub_writeback(4) = 11 Lookup batch: ceil(20/3) = 7
        // Cumulative sum: 3
        // Total: 11 + 7 + 3 = 21
        const NUM_GATE_CONSTRAINTS: usize = 11;
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

    fn reserved_poly_matrix(air: &SubChipPolyAir, main: &RowMajorMatrix<F>) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <SubChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);
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

    /// Generate a sample trace with valid SUB operations.
    fn sample_trace() -> RowMajorMatrix<F> {
        let program = simple_add_sub_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        let shard = runtime.records[0].clone();
        let chip = SubChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    // =========================================================================
    // Constraint satisfaction tests
    // =========================================================================

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = SubChipPolyAir;
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

    /// Generate a random SUB trace for performance testing.
    fn random_sub_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_SUB_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_SUB_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_SUB_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }

        RowMajorMatrix::new(values, NUM_SUB_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_first_and_nonfirst_round_evaluation_random_trace() {
        let air = SubChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(15);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sub_trace(log_n, seed);
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

    /// Multi-round sumcheck benchmark for SubChip PolyAir.
    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = SubChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sub_trace(log_n, seed);
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
            <SubChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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

            std::println!("  round {} (nonfirst): {:?}", round, t_round.elapsed());

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

    // =========================================================================
    // generate_dependencies tests
    // =========================================================================

    /// Program where every SUB writes to x0 (op_a_0=true everywhere).
    /// Used to verify BitVec lookups are skipped on op_a_0=true rows.
    fn only_x0_subs_program() -> dt_core_executor::Program {
        use dt_core_executor::{Instruction, Opcode, Program};
        let instructions = vec![
            Instruction::new(Opcode::SUB, 0, 1, 2, false, false),
            Instruction::new(Opcode::SUB, 0, 4, 5, false, false),
            Instruction::new(Opcode::SUB, 0, 6, 7, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_skipped_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_subs_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        assert!(
            shard.sub_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all events must be op_a_0=true",
        );
        assert!(!shard.sub_events.is_empty(), "fixture must yield sub events");

        let mut deps = ExecutionRecord::default();
        <SubChipPolyAir as MachineAir<F>>::generate_dependencies(
            &SubChipPolyAir,
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
        <SubChipPolyAir as MachineAir<F>>::generate_dependencies(
            &SubChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected: usize = shard.sub_events.iter().filter(|(_, e)| !e.op_a_0).count();
        assert!(expected > 0, "test fixture must include non-x0 SUBs");
        assert_eq!(bitvec_total, expected, "BitVec BLU emit count must equal lookup send count");
    }
}
