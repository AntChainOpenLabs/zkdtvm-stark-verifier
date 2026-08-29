//! PolyAir adaptation of the ByteChip.
//!
//! The ByteChip is the **receiving** side of all byte lookup interactions.
//! It has a preprocessed trace (65536 rows) containing all byte operation results,
//! and a main trace of just 11 multiplicity columns (one per ByteOpcode).
//! There are no gate constraints — correctness is enforced entirely by the
//! permutation argument (LogUp).

use dt_core_executor::{ByteOpcode, ExecutionRecord, Program};
use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    columns::{BytePreprocessedCols, NUM_BYTE_MULT_COLS},
    ByteChip, NUM_BYTE_OPS,
};

const MAX_LOOKUP_VALUES: usize = 16; // BitVec has 16 values

#[derive(Default, Clone, Copy)]
pub struct ByteChipPolyAir;

#[allow(deprecated)]
impl<AB: FullAirBuilder> FullAir<AB> for ByteChipPolyAir {
    fn width(&self) -> usize {
        NUM_BYTE_MULT_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<AB::F>> {
        Some(ByteChip::<AB::F>::trace())
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only multiplicity columns needed in lookup().
        (0..NUM_BYTE_MULT_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        // SAFETY: BytePreprocessedCols is #[repr(C)] with only T-typed fields.
        let local: &BytePreprocessedCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(prep.as_ptr()) };

        let zero = AB::zero_maybe();
        let byte_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Byte as usize));
        let bitvec_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::BitVec as usize));

        // Helper closure for Byte-kind interactions.
        // Payload: [opcode, a1, a2, b, c]
        let mut byte_denom = |opcode: ByteOpcode,
                              a1: AB::VarMaybeExt,
                              a2: AB::VarMaybeExt,
                              b: AB::VarMaybeExt,
                              c: AB::VarMaybeExt| {
            let op = AB::VarMaybeExt::from(AB::F::from_canonical_u8(opcode as u8));
            builder.retain_precomputed(
                builder.lookup_denominator(byte_kind.clone(), vec![op, a1, a2, b, c]),
            );
        };

        // #1: AND — [AND, and, 0, b, c]
        byte_denom(
            ByteOpcode::AND,
            local.and.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #2: OR — [OR, or, 0, b, c]
        byte_denom(
            ByteOpcode::OR,
            local.or.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #3: XOR — [XOR, xor, 0, b, c]
        byte_denom(
            ByteOpcode::XOR,
            local.xor.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #4: SLL — [SLL, sll, 0, b, c]
        byte_denom(
            ByteOpcode::SLL,
            local.sll.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #5: U8Range — [U8Range, 0, 0, b, c]
        byte_denom(
            ByteOpcode::U8Range,
            zero.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #6: ShrCarry — [ShrCarry, shr, shr_carry, b, c]
        byte_denom(
            ByteOpcode::ShrCarry,
            local.shr.clone(),
            local.shr_carry.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #7: LTU — [LTU, ltu, 0, b, c]
        byte_denom(
            ByteOpcode::LTU,
            local.ltu.clone(),
            zero.clone(),
            local.b.clone(),
            local.c.clone(),
        );

        // #8: MSB — [MSB, msb, 0, b, 0]
        byte_denom(ByteOpcode::MSB, local.msb.clone(), zero.clone(), local.b.clone(), zero.clone());

        // #9: BitRange — [BitRange, bit_range[0], 0, bit_range[1], 0]
        byte_denom(
            ByteOpcode::BitRange,
            local.bit_range[0].clone(),
            zero.clone(),
            local.bit_range[1].clone(),
            zero.clone(),
        );

        // #10: U16Range — [U16Range, value_u16, 0, 0, 0]
        byte_denom(ByteOpcode::U16Range, local.value_u16.clone(), zero.clone(), zero.clone(), zero);

        // #11: BitVec — 16 bits with InteractionKind::BitVec
        let bits: Vec<AB::VarMaybeExt> = local.bit_vec.iter().cloned().collect();
        builder.retain_precomputed(builder.lookup_denominator(bitvec_kind, bits));
    }

    fn eval(&self, _builder: &mut AB) {
        // No gate constraints — the ByteChip is a pure lookup table.
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        use std::ops::Deref;
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        // 11 recv calls, one per ByteOpcode, matching precompute_lc order.
        for i in 0..NUM_BYTE_OPS {
            builder.recv(local[i].clone());
        }
    }
}

// =============================================================================
// MachineAir implementation (delegation to ByteChip<F>)
// =============================================================================

impl<F: Field> BaseAir<F> for ByteChipPolyAir {
    fn width(&self) -> usize {
        NUM_BYTE_MULT_COLS
    }
}

impl<F: Field> MachineAir<F> for ByteChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "BytePolyAir".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        ByteChip::<F>::default().preprocessed_width()
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<CompressedMatrix<F>> {
        ByteChip::<F>::default().generate_preprocessed_trace(program)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        ByteChip::<F>::default().generate_dependencies(input, output)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ByteChip::<F>::default().generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        ByteChip::<F>::default().included(shard)
    }
}

#[cfg(test)]
mod tests {
    use core::{borrow::BorrowMut, mem::size_of};

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
        FullAir,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

    use super::*;

    const NUM_LOOKUPS: usize = NUM_BYTE_OPS;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;
    const BATCH_SIZE: usize = 3;

    use crate::bytes::columns::ByteMultCols;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_column_layout_valid() {
        assert_eq!(
            NUM_BYTE_MULT_COLS,
            size_of::<ByteMultCols<u8>>(),
            "ByteMultCols layout mismatch"
        );
        assert_eq!(NUM_BYTE_OPS, 11);
        assert_eq!(NUM_BYTE_MULT_COLS, NUM_BYTE_OPS);
    }

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    fn beta_powers(beta: EF) -> Vec<EF> {
        let required_max_beta_power = <ByteChipPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(&ByteChipPolyAir);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

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
        air: &ByteChipPolyAir,
        main: &RowMajorMatrix<F>,
        preprocessed: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <ByteChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            use std::ops::Deref;
            let main_row: &[F] = main_binding.deref();
            let prep_binding = preprocessed.row_slice(row_idx);
            let prep_row: &[F] = prep_binding.deref();
            let reserved = collect_reserved_poly(main_row, prep_row, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    /// Sample multiplicity trace generated from a real program with the ByteChip.
    fn sample_trace() -> RowMajorMatrix<F> {
        use crate::programs::tests::keccak_program;
        use dt_core_executor::{ExecutionRecord, Executor};
        use dt_stark::{air::MachineAir, DTCoreOpts};
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip = ByteChip::<F>::default();
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    #[test]
    fn test_first_and_nonfirst_round_constraintcheck_satisfied() {
        let air = ByteChipPolyAir;
        let main = sample_trace();
        let preprocessed = ByteChip::<F>::trace();
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

    /// ByteChip has a fixed trace size of 65536 rows (2^16).
    const BYTE_LOG_N: usize = 16;

    fn random_byte_trace(_log_n: usize, seed: u64) -> (RowMajorMatrix<F>, RowMajorMatrix<F>) {
        let num_rows = 1usize << BYTE_LOG_N;
        let mut values = vec![F::zero(); num_rows * NUM_BYTE_MULT_COLS];
        let mut rng = StdRng::seed_from_u64(seed);

        for row_idx in 0..num_rows {
            let row_start = row_idx * NUM_BYTE_MULT_COLS;
            let row = &mut values[row_start..row_start + NUM_BYTE_MULT_COLS];
            let cols: &mut ByteMultCols<F> = row.borrow_mut();

            for i in 0..NUM_BYTE_OPS {
                cols.multiplicities[i] = F::from_canonical_u32(rng.next_u32() % 256);
            }
        }

        let main = RowMajorMatrix::new(values, NUM_BYTE_MULT_COLS);
        let preprocessed = ByteChip::<F>::trace();
        (preprocessed, main)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = ByteChipPolyAir;
        let log_n = BYTE_LOG_N;
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let (preprocessed, main) = random_byte_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!(
            "perf_multi_round: log_n={}, h={}, seed={} (ByteChip fixed trace size)",
            log_n,
            height,
            seed
        );

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers(beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
        let global = EF::zero();
        let reserved_poly_desc =
            <ByteChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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
