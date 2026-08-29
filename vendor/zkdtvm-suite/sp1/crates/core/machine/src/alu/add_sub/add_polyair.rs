//! PolyAir-optimized FullAir implementation for AddChip.
//!
//! Uses co-located helpers for sub-operation constraints:
//! - `cpu_state_*` (4 interactions): state recv/send + clk range checks
//! - `add_op_*` (2 interactions): result U8Range checks
//! - `rtype_register_op_*` (13 interactions): program + 3×4 memory access
//! - `bitvec_*` (1 interaction): is_real + 4 carry bits boolean

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::add::{AddChip, AddCols, NUM_ADD_COLS};
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
// Column offset constants (only for reserved_poly and test helpers)
// =============================================================================

/// Column offset of `is_real` (last column in AddCols).
const COL_IS_REAL: usize = NUM_ADD_COLS - 1;

/// Column offsets of `add_operation.value[0..3]` within AddCols.
const COL_ADD_VALUE_0: usize = 0;
const COL_ADD_VALUE_1: usize = 1;
const COL_ADD_VALUE_2: usize = 2;
const COL_ADD_VALUE_3: usize = 3;

/// Column offsets of `op_a_access.access.value[0..3]` within AddCols.
/// Layout: add_operation(4) + op_a(1) + prev_value(4) = 9.
const COL_OP_A_VALUE_0: usize = 9;
const COL_OP_A_VALUE_1: usize = 10;
const COL_OP_A_VALUE_2: usize = 11;
const COL_OP_A_VALUE_3: usize = 12;

/// Column offset of `op_a_zero` within AddCols.
/// Layout: add_operation(4) + op_a(1) + op_a_access(13) = 18.
const COL_OP_A_ZERO: usize = 18;

/// Column offset of `cpu_state.shard` within AddCols.
/// Layout: add_operation(4) + memory_operations(35) = 39.
const COL_SHARD: usize = 39;

/// Public values index for `execution_shard`.
/// PublicValues<Word<u8>, u8>: committed_value_digest(32) + deferred_proofs_digest(8)
/// + start_pc(1) + next_pc(1) + exit_code(1) + shard(1) = 44.
const PV_EXECUTION_SHARD_IDX: usize = 44;

/// Maximum number of values in any single lookup interaction.
/// BitVec interaction has 16 values (bit0..bit15), which is the largest.
const MAX_LOOKUP_VALUES: usize = 16;

// =============================================================================
// AddChipPolyAir wrapper type
// =============================================================================

/// PolyAir-optimized wrapper for the SP1 AddChip.
///
/// This type implements `FullAir` using the real `AddCols` column layout,
/// mapping SP1's constraint and interaction patterns to PolyAir's four-phase model.
#[derive(Default, Clone, Copy)]
pub struct AddChipPolyAir;

/// Compute the 4 carry bits of `b + c`, mirroring `AddOperation::eval`'s
/// recurrence: `carry[i] = (b[i] + c[i] + carry[i-1] - value[i]) * 256^-1`
/// where `value = wrapping_add(b, c)`.
#[inline]
pub(super) fn add_carry_bits(b: u32, c: u32) -> (u8, u8, u8, u8) {
    let bb = b.to_le_bytes();
    let cc = c.to_le_bytes();
    let value = b.wrapping_add(c).to_le_bytes();

    let mut carry = [0u8; 4];
    let mut prev: u16 = 0;
    for i in 0..4 {
        let sum = bb[i] as u16 + cc[i] as u16 + prev;
        debug_assert!(sum >= value[i] as u16);
        carry[i] = ((sum - value[i] as u16) / 256) as u8;
        debug_assert!(carry[i] < 2, "carry must be boolean");
        prev = carry[i] as u16;
    }
    (carry[0], carry[1], carry[2], carry[3])
}

// =============================================================================
// FullAir implementation
// =============================================================================

impl<AB: FullAirBuilder> FullAir<AB> for AddChipPolyAir {
    fn width(&self) -> usize {
        NUM_ADD_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        // Program send has 14 values, so we need beta^15 (beta^0 unused, beta^1..beta^14 for
        // values)
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
            PairCol::Main(COL_ADD_VALUE_0),  // [7] add_value[0]: AddOperation result
            PairCol::Main(COL_ADD_VALUE_1),  // [8] add_value[1]: AddOperation result
            PairCol::Main(COL_ADD_VALUE_2),  // [9] add_value[2]: AddOperation result
            PairCol::Main(COL_ADD_VALUE_3),  // [10] add_value[3]: AddOperation result
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();

        // SAFETY: All nested structs in AddCols are #[repr(C)] and contain only
        // fields of type T, [T; N], or nested #[repr(C)] structs. The main trace
        // slice has exactly NUM_ADD_COLS elements.
        let local: &AddCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

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
        // #5-6: AddOperation U8Range (2 pairs for 4 result bytes)
        // =====================================================================
        add_op_precompute_lc(builder, &local.add_operation.value);

        // =====================================================================
        // #7-19: RTypeRegisterOp (1 program + 3×4 memory)
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
        // #20: BitVec — derived carry chain + is_real boolean
        // =====================================================================
        // Carry bits computed algebraically from the main trace; BitVec enforces
        // they are boolean. This is the "derived carry pattern" — no gate
        // constraints needed because the carry is uniquely determined.
        let add_value = &local.add_operation.value;
        let op_b_val = &local.memory_operations.op_b_access.access.value;
        let op_c_val = &local.memory_operations.op_c_access.access.value;
        let base_inverse: AB::VarMaybeExt =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(256).inverse());

        let mut carry = [AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe(), AB::zero_maybe()];
        carry[0] = (op_b_val[0].clone() + op_c_val[0].clone() - add_value[0].clone()) *
            base_inverse.clone();
        for i in 1..4 {
            carry[i] = (op_b_val[i].clone() + op_c_val[i].clone() + carry[i - 1].clone() -
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
        let shard = local[2].clone();
        let op_a_value = [local[3].clone(), local[4].clone(), local[5].clone(), local[6].clone()];
        let add_value = [local[7].clone(), local[8].clone(), local[9].clone(), local[10].clone()];

        // Replaces the implicit boolean enforcement BitVec used to provide on is_real.
        let one = AB::one_maybe();
        builder.assert_zero(is_real.clone() * (one - is_real.clone()));

        // CPUState: shard == execution_shard when is_real
        let pv = builder.public();
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(builder, shard, execution_shard, is_real.clone());

        // RType: op_a_zero only on real rows + x0 register must be zero
        rtype_register_op_gate_constraints(builder, op_a_zero, op_a_value, is_real);

        // AddChip constrains the rd writeback word to equal the computed add result.
        for i in 0..4 {
            builder.assert_zero(local[0].clone() * (local[3 + i].clone() - add_value[i].clone()));
        }

        // Note: Carry-based addition constraints use the derived carry pattern —
        // carries are computed algebraically in precompute_lc() and enforced
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

        // #5-6: AddOperation U8Range (mult = is_real - op_a_zero)
        add_op_lookup(builder, result_mult.clone());

        // #7-19: RTypeRegisterOp (1 program + 3×4 memory)
        rtype_register_op_lookup(builder, is_real);

        // #20: BitVec — only emit on real, non-x0 rows (4 carry bits).
        // r_type's `op_a_zero => is_real = 1` guarantees mult >= 0.
        bitvec_lookup(builder, result_mult);
    }
}

// =============================================================================
// MachineAir implementation (delegation to AddChip)
// =============================================================================

impl<F: Field> BaseAir<F> for AddChipPolyAir {
    fn width(&self) -> usize {
        NUM_ADD_COLS
    }
}

impl<F: Field> MachineAir<F> for AddChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "AddPolyAir".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <AddChip as MachineAir<F>>::num_rows(&AddChip, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        AddChip.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        use core::borrow::BorrowMut;
        use dt_core_executor::events::{ByteLookupEvent, ByteRecord};
        use hashbrown::HashMap;
        use itertools::Itertools;
        use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};

        let chunk_size = std::cmp::max(input.add_events.len() / num_cpus::get(), 1);
        let shard = input.execution_shard();

        let blu_batches = input
            .add_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                for (record, event) in events {
                    // [1] Reuse AddChip path: CPUState / memory / U8Range BLU events.
                    let mut row = [F::zero(); NUM_ADD_COLS];
                    let cols: &mut AddCols<F> = row.as_mut_slice().borrow_mut();
                    AddChip.event_to_row(record, event, cols, &mut blu, shard);

                    // [2] PolyAir-only: emit BitVec on real, non-x0 rows.
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
        <AddChip as MachineAir<F>>::included(&AddChip, shard)
    }

    fn local_only(&self) -> bool {
        <AddChip as MachineAir<F>>::local_only(&AddChip)
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
        super::add::{AddChip, AddCols},
        *,
    };

    /// Total number of lookup interactions:
    /// - 1 State recv (receive_state)
    /// - 1 State send (send_state)
    /// - 2 Byte (clk range checks: U16Range + BitRange)
    /// - 2 Byte (result value U8Range: 4 bytes in 2 pairs)
    /// - 1 Program send
    /// - 4 eval_memory_access(op_b): ts_u16, ts_bit, mem_send, mem_recv
    /// - 4 eval_memory_access(op_c): ts_u16, ts_bit, mem_send, mem_recv
    /// - 4 eval_memory_access(op_a): ts_u16, ts_bit, mem_send, mem_recv
    /// - 1 BitVec (is_real + 4 carry bits boolean constraint)
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
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn make_public_values(execution_shard: u32) -> Vec<F> {
        let mut pv = vec![F::zero(); PV_EXECUTION_SHARD_IDX + 1];
        pv[PV_EXECUTION_SHARD_IDX] = F::from_canonical_u32(execution_shard);
        pv
    }

    // =========================================================================
    // Test Helper Functions
    // =========================================================================

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    /// BabyBear modulus: 15 * 2^27 + 1 = 2013265921
    const BABYBEAR_MODULUS: u32 = 2013265921;

    /// Generate a random BabyBear field element using the provided RNG.
    fn random_f(rng: &mut StdRng) -> F {
        let value = rng.gen_range(0..BABYBEAR_MODULUS);
        F::from_canonical_u32(value)
    }

    /// Generate a random extension field element (4 base field elements).
    fn random_ef(rng: &mut StdRng) -> EF {
        let values: [F; 4] = [random_f(rng), random_f(rng), random_f(rng), random_f(rng)];
        EF::from_base_slice(&values)
    }

    /// Generate a random challenge beta using the provided seed.
    /// Each seed produces a deterministic but different random value.
    fn challenge_beta_with_seed(seed: u64) -> EF {
        let mut rng = StdRng::seed_from_u64(seed);
        random_ef(&mut rng)
    }

    fn beta_powers(beta: EF) -> Vec<EF> {
        let required_max_beta_power = <AddChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&AddChipPolyAir);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    /// Generate random reducer coefficients using the provided seed.
    fn random_reducer(seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
        // Gate constraints: is_real_bool(1) + cpu_state(1) + rtype_register_op(5) +
        // add_writeback(4) = 11 Lookup batch: ceil(20/3) = 7
        // Cumulative sum: 3
        // Total: 11 + 7 + 3 = 21
        const NUM_GATE_CONSTRAINTS: usize = 11;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS).map(|_| random_ef(&mut rng)).collect()
    }

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(air: &AddChipPolyAir, main: &RowMajorMatrix<F>) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <AddChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    /// Generate a sample trace with valid ADD operations.
    fn sample_trace() -> RowMajorMatrix<F> {
        let program = simple_add_sub_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        let shard = runtime.records[0].clone();
        let chip = AddChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    // =========================================================================
    // Constraint satisfaction tests
    // =========================================================================

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = AddChipPolyAir;
        let main = sample_trace();
        let height = main.height();
        std::println!("trace height = {}, width = {}", height, main.width());
        assert!(height >= 2);

        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let beta_powers = beta_powers(beta);
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

        let constraint_reducer = random_reducer(reducer_seed);
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

    /// Generate a random ADD trace for performance testing.
    fn random_add_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_ADD_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_ADD_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_ADD_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }

        RowMajorMatrix::new(values, NUM_ADD_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_first_and_nonfirst_round_evaluation_random_trace() {
        let air = AddChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(15);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_add_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf config: log_n={}, h={}, seed={}", log_n, height, seed);

        // Use random challenges derived from the test seed
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers(beta);
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
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

    /// Multi-round sumcheck benchmark for PolyAir.
    ///
    /// Runs a complete `log_n`-round sumcheck:
    ///   Round 0: first_round_evaluation (base-field trace)
    ///   Rounds 1..log_n-1: bound_var_* folding + nonfirst_round_evaluation
    ///
    /// This measures the **total** sumcheck proving time, where PolyAir's
    /// precompute optimization should show cumulative benefits.
    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = AddChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_add_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        // Use random challenges derived from the test seed
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers(beta);
        let beta_septix = beta_septix(beta);
        let public = make_public_values(1);
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
        let global = EF::zero();
        let reserved_poly_desc =
            <AddChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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
            // if round <= 3 || round == log_n - 1 {
            std::println!("  round {} (nonfirst): {:?}", round, round_elapsed);
            // }

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

    /// Program where every ADD writes to x0 (op_a_0=true everywhere).
    /// Used to verify BitVec lookups are skipped on op_a_0=true rows.
    fn only_x0_adds_program() -> dt_core_executor::Program {
        use dt_core_executor::{Instruction, Opcode, Program};
        let instructions = vec![
            Instruction::new(Opcode::ADD, 0, 1, 2, false, false),
            Instruction::new(Opcode::ADD, 0, 4, 5, false, false),
            Instruction::new(Opcode::ADD, 0, 6, 7, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    #[test]
    fn bitvec_skipped_when_op_a_zero() {
        use dt_core_executor::ByteOpcode;

        let program = only_x0_adds_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        // Fixture invariants: every recorded add event must be op_a_0=true.
        assert!(
            shard.add_events.iter().all(|(_, e)| e.op_a_0),
            "fixture invariant: all events must be op_a_0=true",
        );
        assert!(!shard.add_events.is_empty(), "fixture must yield add events");

        let mut deps = ExecutionRecord::default();
        <AddChipPolyAir as MachineAir<F>>::generate_dependencies(
            &AddChipPolyAir,
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
        <AddChipPolyAir as MachineAir<F>>::generate_dependencies(
            &AddChipPolyAir,
            &shard,
            &mut deps,
        );

        let bitvec_total: usize = deps
            .byte_lookups
            .iter()
            .filter(|(ev, _)| ev.opcode == ByteOpcode::BitVec)
            .map(|(_, m)| *m)
            .sum();

        let expected: usize = shard.add_events.iter().filter(|(_, e)| !e.op_a_0).count();
        std::println!("bitvec_total: {} epected: {}", bitvec_total, expected);
        assert!(expected > 0, "test fixture must include non-x0 ADDs");
        assert_eq!(bitvec_total, expected, "BitVec BLU emit count must equal lookup send count",);
    }
}
