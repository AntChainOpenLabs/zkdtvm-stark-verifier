//! PolyAir helpers for program lookup interactions.
//!
//! A program lookup sends the instruction fields (pc, opcode, operands) to
//! the ProgramChip for verification against the preprocessed instruction table.

use dt_stark::{air::FullAirBuilder, InteractionKind};
use p3_field::AbstractField;

// ============================================================================
// Interaction count constants
// ============================================================================

/// Program lookup: 1 send_program
pub(crate) const PROGRAM_NUM_INTERACTIONS: usize = 1;

// ============================================================================
// Program Lookup (1 interaction)
// ============================================================================

/// Precompute denominator for a program lookup.
///
/// The send_program payload matches `InstructionCols::into_iter()`:
/// ```text
/// [pc, opcode, opcode, op_a, op_b[0..3], op_c[0..3], op_a_zero, imm_b, imm_c]
/// ```
/// (15 values matching the Program interaction format).
///
/// Different adapter types fill `op_b`/`op_c`/`imm_b`/`imm_c` differently:
/// - R-Type: op_b = [scalar, 0, 0, 0], op_c = [scalar, 0, 0, 0], imm_b = 0, imm_c = 0
/// - ALU-Type: op_b = [scalar, 0, 0, 0], op_c = Word, imm_b = 0, imm_c = col
/// - I-Type: op_b = [scalar, 0, 0, 0], op_c = Word, imm_b = 0, imm_c = 1
/// - ADDI-Type: op_b = [scalar, 0, 0, 0], op_c = Word, imm_b = col, imm_c = 1
/// - B-Type: op_b = [scalar, 0, 0, 0], op_c = Word, imm_b = 0, imm_c = 1
/// - J-Type: op_b = Word, op_c = Word, imm_b = 1, imm_c = 1
pub fn program_precompute_lc<AB: FullAirBuilder>(
    builder: &mut AB,
    pc: AB::VarMaybeExt,
    opcode: AB::VarMaybeExt,
    op_a: AB::VarMaybeExt,
    op_b: [AB::VarMaybeExt; 4],
    op_c: [AB::VarMaybeExt; 4],
    op_a_zero: AB::VarMaybeExt,
    imm_b: AB::VarMaybeExt,
    imm_c: AB::VarMaybeExt,
) {
    let program_kind =
        AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Program as usize));

    let [ob0, ob1, ob2, ob3] = op_b;
    let [oc0, oc1, oc2, oc3] = op_c;

    builder.retain_precomputed(builder.lookup_denominator(
        program_kind,
        vec![
            pc,
            opcode.clone(),
            opcode,
            op_a,
            ob0,
            ob1,
            ob2,
            ob3,
            oc0,
            oc1,
            oc2,
            oc3,
            op_a_zero,
            imm_b,
            imm_c,
        ],
    ));
}

/// Declare multiplicity for a program lookup.
pub fn program_lookup<AB: FullAirBuilder>(builder: &mut AB, is_real: AB::VarMaybeExt) {
    builder.send(is_real);
}

// ============================================================================
// ProgramChipPolyAir — FullAir adaptation of ProgramChip (recv side)
// ============================================================================

use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{FullAir, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::Field;
use p3_matrix::Matrix;

use super::{ProgramChip, ProgramPreprocessedCols, NUM_PROGRAM_MULT_COLS};

/// Number of lookup interactions for ProgramChipPolyAir.
/// 1 receive_program interaction.
const NUM_LOOKUPS: usize = 1;

/// Maximum number of values in a single lookup payload (program has 15).
const MAX_LOOKUP_VALUES: usize = 15;

/// PolyAir wrapper for ProgramChip.
///
/// The ProgramChip is the **receiving** side of all program lookup interactions.
/// It has a preprocessed trace containing the instruction table (pc + instruction fields),
/// and a main trace with a single multiplicity column.
/// There are no gate constraints — correctness is enforced entirely by the
/// permutation argument (LogUp).
#[derive(Default, Clone, Copy)]
pub struct ProgramChipPolyAir;

impl<AB: FullAirBuilder> FullAir<AB> for ProgramChipPolyAir {
    fn width(&self) -> usize {
        NUM_PROGRAM_MULT_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only the multiplicity column from main trace is needed in lookup().
        vec![PairCol::Main(0)]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        // SAFETY: ProgramPreprocessedCols is #[repr(C)] with only T-typed fields.
        let local: &ProgramPreprocessedCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(prep.as_ptr()) };

        let program_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Program as usize));

        // Build the receive_program payload:
        // [pc, opcode, opcode, op_a, op_b[0..3], op_c[0..3], op_a_0, imm_b, imm_c]
        let instruction = &local.instruction;
        builder.retain_precomputed(builder.lookup_denominator(
            program_kind,
            vec![
                local.pc.clone(),
                instruction.opcode.clone(),
                instruction.opcode.clone(),
                instruction.op_a.clone(),
                instruction.op_b[0].clone(),
                instruction.op_b[1].clone(),
                instruction.op_b[2].clone(),
                instruction.op_b[3].clone(),
                instruction.op_c[0].clone(),
                instruction.op_c[1].clone(),
                instruction.op_c[2].clone(),
                instruction.op_c[3].clone(),
                instruction.op_a_0.clone(),
                instruction.imm_b.clone(),
                instruction.imm_c.clone(),
            ],
        ));
    }

    fn eval(&self, _builder: &mut AB) {
        // No gate constraints — the ProgramChip is a pure lookup table.
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = core::ops::Deref::deref(&local_binding);

        // 1 recv call matching precompute_lc order.
        // local[0] = multiplicity
        builder.recv(local[0].clone());
    }
}

// =============================================================================
// MachineAir implementation (delegation to ProgramChip)
// =============================================================================

impl<F: Field> BaseAir<F> for ProgramChipPolyAir {
    fn width(&self) -> usize {
        NUM_PROGRAM_MULT_COLS
    }
}

impl<F: Field> MachineAir<F> for ProgramChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <ProgramChip as MachineAir<F>>::name(&ProgramChip) + "PolyAir"
    }

    fn preprocessed_width(&self) -> usize {
        <ProgramChip as MachineAir<F>>::preprocessed_width(&ProgramChip)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        ProgramChip.generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <ProgramChip as MachineAir<F>>::generate_dependencies(&ProgramChip, input, output)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ProgramChip.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ProgramChip as MachineAir<F>>::included(&ProgramChip, shard)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dt_core_executor::ExecutionRecord;
    use dt_stark::air::{
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
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::program::ProgramChip;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const BATCH_SIZE: usize = 3;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn beta_powers(beta: EF) -> Vec<EF> {
        let required_max_beta_power = <ProgramChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&ProgramChipPolyAir);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    /// BabyBear modulus: 15 * 2^27 + 1 = 2013265921
    const BABYBEAR_MODULUS: u32 = 2013265921;

    fn random_f(rng: &mut StdRng) -> F {
        F::from_canonical_u32(rng.gen_range(0..BABYBEAR_MODULUS))
    }

    fn random_ef(rng: &mut StdRng) -> EF {
        EF::from_base_slice(&[random_f(rng), random_f(rng), random_f(rng), random_f(rng)])
    }

    fn challenge_beta_with_seed(seed: u64) -> EF {
        let mut rng = StdRng::seed_from_u64(seed);
        random_ef(&mut rng)
    }

    fn random_reducer(seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
        const NUM_GATE_CONSTRAINTS: usize = 0;
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

    fn reserved_poly_matrix(
        air: &ProgramChipPolyAir,
        main: &RowMajorMatrix<F>,
        preprocessed: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <ProgramChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            let main_row: &[F] = core::ops::Deref::deref(&main_binding);
            let prep_binding = preprocessed.row_slice(row_idx);
            let prep_row: &[F] = core::ops::Deref::deref(&prep_binding);
            let reserved = collect_reserved_poly(main_row, prep_row, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_traces() -> (RowMajorMatrix<F>, RowMajorMatrix<F>) {
        use crate::programs::tests::fibonacci_program;

        let program = Arc::new(fibonacci_program());
        let chip = ProgramChip::new();

        let preprocessed = chip
            .generate_preprocessed_trace(&program)
            .expect("program preprocessed trace should exist")
            .decompress();

        let shard = ExecutionRecord { program, ..Default::default() };
        let main = chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress();

        (preprocessed, main)
    }

    #[test]
    fn test_first_and_nonfirst_round_evaluation_satisfied() {
        let air = ProgramChipPolyAir;
        let (preprocessed, main) = sample_traces();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let beta_powers = beta_powers(beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];

        let precomputed_full = precompute_linear_combination(
            &air,
            Some(&preprocessed),
            &main,
            &public,
            alpha,
            &beta_powers,
            beta_septix,
            NUM_PRECOMPUTED,
        );
        let (permutation_full, local_sum) = generate_permutation_trace_(
            &air,
            Some(&preprocessed),
            &main,
            &precomputed_full,
            alpha,
            &beta_powers,
            BATCH_SIZE,
            NUM_LOOKUPS,
        );

        let precomputed = trim_rows(&precomputed_full, height);
        let permutation = trim_rows(&permutation_full, height);
        let reserved = reserved_poly_matrix(&air, &main, &preprocessed);
        let constraint_reducer = random_reducer(reducer_seed);
        let global = EF::zero();

        let first = first_round_evaluation(
            &air,
            &public,
            Some(&preprocessed),
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
        assert!(
            first.iter().all(|x| x.is_zero()),
            "first_round non-zero at indices: {:?}",
            first
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
                .collect::<Vec<_>>()
        );

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
            "nonfirst_round non-zero at indices: {:?}",
            nonfirst
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
                .collect::<Vec<_>>()
        );
    }

    fn random_program_trace(log_n: usize, _seed: u64) -> (RowMajorMatrix<F>, RowMajorMatrix<F>) {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let (base_prep, base_main) = sample_traces();
        let base_height = base_main.height();
        assert!(base_height >= 1, "sample trace must contain at least one row");
        assert!(
            target_height >= base_height,
            "target height {} smaller than sample trace height {}",
            target_height,
            base_height
        );

        let main = if target_height == base_height {
            base_main
        } else {
            let last_row_start = (base_height - 1) * NUM_PROGRAM_MULT_COLS;
            let last_row =
                &base_main.values[last_row_start..last_row_start + NUM_PROGRAM_MULT_COLS];
            let mut values = Vec::with_capacity(target_height * NUM_PROGRAM_MULT_COLS);
            values.extend_from_slice(&base_main.values);
            for _ in base_height..target_height {
                values.extend_from_slice(last_row);
            }
            RowMajorMatrix::new(values, NUM_PROGRAM_MULT_COLS)
        };

        let prep_width = base_prep.width();
        let prep_height = base_prep.height();
        let preprocessed = if target_height == prep_height {
            base_prep
        } else {
            let last_row_start = (prep_height - 1) * prep_width;
            let last_row = &base_prep.values[last_row_start..last_row_start + prep_width];
            let mut values = Vec::with_capacity(target_height * prep_width);
            values.extend_from_slice(&base_prep.values);
            for _ in prep_height..target_height {
                values.extend_from_slice(last_row);
            }
            RowMajorMatrix::new(values, prep_width)
        };

        (preprocessed, main)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = ProgramChipPolyAir;
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let (preprocessed, main) = random_program_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng_alpha = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng_alpha);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers(beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
        let global = EF::zero();
        let reserved_poly_desc = <ProgramChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(&air);

        // --- Precompute phase ---
        let t_precompute = std::time::Instant::now();
        let precomputed_full = precompute_linear_combination(
            &air,
            Some(&preprocessed),
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
            Some(&preprocessed),
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
            Some(&preprocessed),
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
        let mut reserved =
            bound_var_main_prep(&main, Some(&preprocessed), &reserved_poly_desc, ef(42));
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
