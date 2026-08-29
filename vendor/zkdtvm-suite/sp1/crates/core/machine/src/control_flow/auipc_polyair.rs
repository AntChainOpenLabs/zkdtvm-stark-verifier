//! PolyAir `FullAir` adapter for `AuipcChip` (`auipc.rs`).
//!
//! Mirrors the `Air` implementation using the four-phase PolyAir model
//! (`reserved_poly` / `precompute_lc` / `eval` / `lookup`).

use dt_core_executor::{ExecutionRecord, Opcode, Program, DEFAULT_PC_INC};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::auipc::{AuipcChip, AuipcColumns, NUM_AUIPC_COLS};
use crate::{
    adapter::{
        register::j_type::{
            jtype_register_op_gate_constraints, jtype_register_op_lookup,
            jtype_register_op_precompute_lc,
        },
        state::{cpu_state_gate_constraints, cpu_state_lookup, cpu_state_precompute_lc},
    },
    operations::{
        add::{add_op_gate_constraints, add_op_lookup, add_op_precompute_lc},
        baby_bear_word::{
            baby_bear_range_check_gate_constraints, baby_bear_range_check_lookup,
            baby_bear_range_check_precompute_lc,
        },
    },
};

/// BitVec has 16 elements (largest payload).
const MAX_LOOKUP_VALUES: usize = 16;

// ============================================================================
// Main column offsets within `AuipcColumns<u8>` (NUM_AUIPC_COLS = 40).
//
// Layout (#[repr(C)]):
//   [0]      cpu_state.shard
//   [1..3]   cpu_state.{clk_16_28, clk_0_16}            ← precompute-only
//   [3]      cpu_state.pc
//   [4]      mem_ops.op_a                               ← precompute-only
//   [5..9]   mem_ops.op_a_access.prev_value             ← precompute-only
//   [9..13]  mem_ops.op_a_access.access.value
//   [13..18] mem_ops.op_a_access.access.{ts fields}     ← precompute-only
//   [18]     mem_ops.op_a_zero
//   [19..23] mem_ops.op_b_imm
//   [23..27] mem_ops.op_c_imm                           ← precompute-only
//   [27..31] add_op.value
//   [31..35] pc (Word)
//   [35]     pc_range_checker.most_sig_byte_lt_120
//   [36]     is_auipc
//   [37]     is_unimp
//   [38]     is_ebreak
//   [39]     is_real
// ============================================================================

const COL_CPU_SHARD: usize = 0;
const COL_CPU_PC: usize = 3;
const COL_OP_A_VALUE: usize = 9;
const COL_OP_A_ZERO: usize = 18;
const COL_OP_B_IMM: usize = 19;
const COL_ADD_OP_VALUE: usize = 27;
const COL_PC_WORD: usize = 31;
const COL_PC_RANGE_CHECKER: usize = 35;
const COL_IS_AUIPC: usize = 36;
const COL_IS_UNIMP: usize = 37;
const COL_IS_EBREAK: usize = 38;
const COL_IS_REAL: usize = 39;

// ============================================================================
// Reserved-poly slice layout (RES_NUM_COLS = 24).
//
// Only fields read by `eval` or `lookup` are retained.
//
//   [0]      is_real
//   [1]      is_auipc
//   [2]      is_unimp
//   [3]      is_ebreak
//   [4]      cpu_state.shard
//   [5]      cpu_state.pc
//   [6]      op_a_zero
//   [7]      pc_range_checker.most_sig_byte_lt_120
//   [8..12]  op_a_access.access.value
//   [12..16] op_b_imm
//   [16..20] add_op.value
//   [20..24] pc (Word)
// ============================================================================

const RES_IS_REAL: usize = 0;
const RES_IS_AUIPC: usize = 1;
const RES_IS_UNIMP: usize = 2;
const RES_IS_EBREAK: usize = 3;
const RES_CPU_SHARD: usize = 4;
const RES_CPU_PC: usize = 5;
const RES_OP_A_ZERO: usize = 6;
const RES_PC_RANGE_CHECKER: usize = 7;
const RES_OP_A_VALUE: usize = 8;
const RES_OP_B_IMM: usize = 12;
const RES_ADD_OP_VALUE: usize = 16;
const RES_PC_WORD: usize = 20;
const RES_NUM_COLS: usize = 24;

#[derive(Default, Clone, Copy)]
pub struct AuipcChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for AuipcChipPolyAir {
    fn width(&self) -> usize {
        NUM_AUIPC_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        let mut cols = Vec::with_capacity(RES_NUM_COLS);
        cols.push(PairCol::Main(COL_IS_REAL));
        cols.push(PairCol::Main(COL_IS_AUIPC));
        cols.push(PairCol::Main(COL_IS_UNIMP));
        cols.push(PairCol::Main(COL_IS_EBREAK));
        cols.push(PairCol::Main(COL_CPU_SHARD));
        cols.push(PairCol::Main(COL_CPU_PC));
        cols.push(PairCol::Main(COL_OP_A_ZERO));
        cols.push(PairCol::Main(COL_PC_RANGE_CHECKER));
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_A_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_OP_B_IMM + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_ADD_OP_VALUE + i));
        }
        for i in 0..4 {
            cols.push(PairCol::Main(COL_PC_WORD + i));
        }
        debug_assert_eq!(cols.len(), RES_NUM_COLS);
        cols
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &AuipcColumns<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.cpu_state.shard.clone();
        let clk_0_16 = local.cpu_state.clk_0_16.clone();
        let clk_16_28 = local.cpu_state.clk_16_28.clone();
        let pc_scalar = local.cpu_state.pc.clone();
        let clk = clk_0_16.clone() +
            clk_16_28.clone() * AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
        let next_pc =
            pc_scalar.clone() + AB::VarMaybeExt::from(AB::F::from_canonical_u32(DEFAULT_PC_INC));

        let opcode_expr = AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::AUIPC as u8)) *
            local.is_auipc.clone() +
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::UNIMP as u8)) *
                local.is_unimp.clone() +
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(Opcode::EBREAK as u8)) *
                local.is_ebreak.clone();

        let op_b_imm = &local.mem_ops.op_b_imm;
        let op_c_imm = &local.mem_ops.op_c_imm;
        let add_value = &local.add_op.value;
        let pc_word = &local.pc;

        // =====================================================================
        // #1-4: CPUState (recv_state, send_state, U16Range, BitRange)
        // =====================================================================
        cpu_state_precompute_lc(
            builder,
            shard.clone(),
            clk.clone(),
            clk_0_16,
            clk_16_28,
            pc_scalar.clone(),
            next_pc,
        );

        // =====================================================================
        // #5-9: JTypeRegisterOp (1 program + 4 op_a readwrite)
        // =====================================================================
        jtype_register_op_precompute_lc(
            builder,
            pc_scalar,
            opcode_expr,
            local.mem_ops.op_a.clone(),
            [op_b_imm[0].clone(), op_b_imm[1].clone(), op_b_imm[2].clone(), op_b_imm[3].clone()],
            [op_c_imm[0].clone(), op_c_imm[1].clone(), op_c_imm[2].clone(), op_c_imm[3].clone()],
            local.mem_ops.op_a_zero.clone(),
            &local.mem_ops.op_a_access.access,
            &local.mem_ops.op_a_access.prev_value,
            shard,
            clk,
        );

        // =====================================================================
        // #10: BabyBearWordRangeChecker LTU (pc word)
        // =====================================================================
        baby_bear_range_check_precompute_lc(
            builder,
            pc_word[3].clone(),
            local.pc_range_checker.most_sig_byte_lt_120.clone(),
        );

        // =====================================================================
        // #11-12: AddOperation U8Range (2 pairs for 4 result bytes)
        // =====================================================================
        add_op_precompute_lc(builder, add_value);

        // BitVec removed: of the 4 originally-boolean columns
        //   {is_real, is_auipc, is_unimp, is_ebreak},
        // only is_auipc needs explicit enforcement. is_unimp and is_ebreak are
        // already forced to 0 by unconditional `assert_zero` gates in `eval`
        // (auipc.rs L113-114), and is_real is constrained to equal the sum
        // (= is_auipc + 0 + 0 = is_auipc), so its booleanness follows. A
        // single boolean gate in `eval` replaces the 4-bit BitVec lookup.
    }

    /// Gate constraints, ordered to match the original `Air<AB>::eval()` in `auipc.rs`.
    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_auipc = local[RES_IS_AUIPC].clone();
        let is_unimp = local[RES_IS_UNIMP].clone();
        let is_ebreak = local[RES_IS_EBREAK].clone();
        let is_real = local[RES_IS_REAL].clone();
        let a_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_A_VALUE + i].clone());
        let b_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_OP_B_IMM + i].clone());
        let add_value: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_ADD_OP_VALUE + i].clone());
        let pc_word: [AB::VarMaybeExt; 4] =
            core::array::from_fn(|i| local[RES_PC_WORD + i].clone());

        // ── auipc.rs L85-88: assert_bool × 3 ──────────────────────────
        // Only is_auipc needs an explicit boolean gate. is_unimp and is_ebreak
        // are forced to 0 below (so they're trivially boolean), and is_real
        // is constrained = is_auipc + is_unimp + is_ebreak = is_auipc, so its
        // booleanness follows from is_auipc's.
        let one = AB::one_maybe();
        builder.assert_zero(is_auipc.clone() * (one - is_auipc.clone()));

        // ── auipc.rs L89-91: is_real = is_auipc + is_unimp + is_ebreak ─
        let sum_sel = is_auipc.clone() + is_unimp.clone() + is_ebreak.clone();
        builder.assert_zero(is_real.clone() - sum_sel);

        // ── auipc.rs L93-100: CPUState::eval() ─────────────────────────
        let pv = builder.public();
        const PV_EXECUTION_SHARD_IDX: usize = 44;
        let execution_shard: AB::VarMaybeExt = pv[PV_EXECUTION_SHARD_IDX].clone().into();
        cpu_state_gate_constraints(
            builder,
            local[RES_CPU_SHARD].clone(),
            execution_shard,
            is_real.clone(),
        );

        // ── auipc.rs L102-111: JTypeRegisterOp::eval() ─────────────────
        jtype_register_op_gate_constraints(
            builder,
            local[RES_OP_A_ZERO].clone(),
            a_word.clone(),
            is_real.clone(),
        );

        // ── auipc.rs L113-114: assert_zero(is_unimp), assert_zero(is_ebreak)
        builder.assert_zero(is_unimp);
        builder.assert_zero(is_ebreak);

        // ── auipc.rs L120-125: BabyBearWordRangeChecker(pc, is_auipc) ──
        baby_bear_range_check_gate_constraints(
            builder,
            pc_word.clone(),
            local[RES_PC_RANGE_CHECKER].clone(),
            is_auipc.clone(),
        );

        // ── auipc.rs L126: assert_eq(pc.reduce, cpu_state.pc) ──────────
        let base_w = |i: u32| AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << (8 * i)));
        let pc_reduced = pc_word[0].clone() * base_w(0) +
            pc_word[1].clone() * base_w(1) +
            pc_word[2].clone() * base_w(2) +
            pc_word[3].clone() * base_w(3);
        builder.assert_zero(pc_reduced - local[RES_CPU_PC].clone());

        // ── auipc.rs L129-131: when(is_real - op_a_zero) a_word == add_op.value
        let perform = is_real.clone() - local[RES_OP_A_ZERO].clone();
        for i in 0..4 {
            builder.when(perform.clone()).assert_zero(a_word[i].clone() - add_value[i].clone());
        }

        // ── auipc.rs L134-139: AddOperation::eval(b_word, pc, add_op, is_auipc)
        add_op_gate_constraints(builder, b_word, pc_word, add_value, is_auipc);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[RES_IS_REAL].clone();
        let is_auipc = local[RES_IS_AUIPC].clone();

        // #1-4: CPUState
        cpu_state_lookup(builder, is_real.clone());
        // #5-9: JTypeRegisterOp
        jtype_register_op_lookup(builder, is_real.clone());
        // #10: BabyBearWordRangeChecker LTU (pc, mult=is_auipc)
        baby_bear_range_check_lookup(builder, is_auipc.clone());
        // #11-12: AddOp U8Range (mult=is_auipc)
        add_op_lookup(builder, is_auipc);
        // BitVec #13 removed — replaced with an explicit assert_zero gate
        // in `eval` (is_auipc booleanness implies all four bits boolean).
    }
}

// =============================================================================
// MachineAir implementation (delegation to AuipcChip)
// =============================================================================

impl<F: Field> BaseAir<F> for AuipcChipPolyAir {
    fn width(&self) -> usize {
        NUM_AUIPC_COLS
    }
}

impl<F: Field> MachineAir<F> for AuipcChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "AuipcPolyAir".to_string()
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        AuipcChip.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <AuipcChip as MachineAir<F>>::included(&AuipcChip, shard)
    }

    fn local_only(&self) -> bool {
        <AuipcChip as MachineAir<F>>::local_only(&AuipcChip)
    }
}

#[cfg(test)]
mod tests {
    use std;

    use super::*;

    /// Lookup interaction count: CPUState (4) + JTypeRegisterOp (5) + BabyBear LTU (1) + Add U8×2
    /// (2) + BitVec boolean (1).
    const NUM_LOOKUPS: usize = 12;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    use crate::control_flow::AuipcChip;
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
        let n = <AuipcChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&AuipcChipPolyAir);
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
        // Gate constraints: 22
        // Lookup batch: ceil(13/3) = 5
        // Cumulative sum: 3
        // +1 vs. pre-refactor: is_auipc boolean gate replaces BitVec #13.
        const NUM_GATE_CONSTRAINTS: usize = 23;
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
        air: &AuipcChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <AuipcChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

        let chip = AuipcChip;
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_auipc_first_and_nonfirst_round_evaluation_satisfied() {
        let air = AuipcChipPolyAir;
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

    fn random_auipc_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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

        let last_row_start = (base_height - 1) * NUM_AUIPC_COLS;
        let last_row = &base.values[last_row_start..last_row_start + NUM_AUIPC_COLS];
        let mut values = Vec::with_capacity(target_height * NUM_AUIPC_COLS);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, NUM_AUIPC_COLS)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = AuipcChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_auipc_trace(log_n, seed);
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
            <AuipcChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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
