//! PolyAir adaptation of ShaExtendControllerChip.
//!
//! The ShaExtendControllerChip bridges SHA-256 extend syscall invocations between
//! the syscall table and the ShaExtend permutation chip. It is a pure lookup bridge
//! with minimal gate constraints.
//!
//! ## Interaction Summary (3 total)
//!
//!  1. send(ShaExtend, is_real)  — [shard, clk, w_ptr, 16]
//!  2. recv(ShaExtend, is_real)  — [shard, clk, w_ptr, 64]
//!  3. send(Global, is_real)     — [shard, clk, syscall_id, w_ptr, 0, 0, 0, 0, 1, Syscall_kind]
//!
//! Gate constraints:
//!  - `is_real * (1 - is_real) = 0` (boolean enforcement for single bit)

use std::ops::Deref;

use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use dt_core_executor::syscalls::SyscallCode;

use super::controller::NUM_SHA_EXTEND_CONTROLLER_COLS;

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
const NUM_LOOKUPS: usize = 3;

/// Maximum number of values in a single lookup payload.
/// Global interaction has 10 values — the largest payload.
const MAX_LOOKUP_VALUES: usize = 10;

// ShaExtendControllerCols column indices:
//  0: shard
//  1: clk
//  2: w_ptr
//  3: is_real

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for ShaExtendControllerChip.
///
/// Pure lookup bridge: sends to ShaExtend chip, receives from ShaExtend chip,
/// and sends a Global interaction for the syscall table.
#[derive(Clone, Copy, Default)]
pub struct ShaExtendControllerPolyAir;

impl ShaExtendControllerPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ShaExtendControllerPolyAir {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROLLER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only is_real is needed in eval()/lookup() for multiplicities and boolean check.
        vec![PairCol::Main(3)] // is_real
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let ptr = main.as_ptr();
        let col = |i: usize| -> AB::VarMaybeExt { unsafe { (*ptr.add(i)).clone() } };

        let zero = AB::zero_maybe();

        let shard = col(0);
        let clk = col(1);
        let w_ptr = col(2);

        let sha_extend_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::ShaExtend as usize));
        let global_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));
        let syscall_interaction_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(InteractionKind::Syscall as u8));
        let syscall_id =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(SyscallCode::SHA_EXTEND.syscall_id()));

        // =====================================================================
        // #1: send(ShaExtend, is_real) — [shard, clk, w_ptr, 16]
        // =====================================================================
        builder.retain_precomputed(builder.lookup_denominator(
            sha_extend_kind.clone(),
            vec![
                shard.clone(),
                clk.clone(),
                w_ptr.clone(),
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(16)),
            ],
        ));

        // =====================================================================
        // #2: recv(ShaExtend, is_real) — [shard, clk, w_ptr, 64]
        // =====================================================================
        builder.retain_precomputed(builder.lookup_denominator(
            sha_extend_kind,
            vec![
                shard.clone(),
                clk.clone(),
                w_ptr.clone(),
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(64)),
            ],
        ));

        // =====================================================================
        // #3: send(Global, is_real) — syscall interaction to global table
        // [shard, clk, syscall_id, w_ptr, 0, 0, 0, 0, 1, Syscall_kind]
        // =====================================================================
        builder.retain_precomputed(builder.lookup_denominator(
            global_kind,
            vec![
                shard,
                clk,
                syscall_id,
                w_ptr,
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                AB::VarMaybeExt::from(AB::F::one()),
                syscall_interaction_kind,
            ],
        ));
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[0].clone();

        // assert_bool(is_real): is_real * (1 - is_real) = 0
        builder.assert_zero(is_real.clone() * (AB::one_maybe() - is_real));
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &[AB::VarMaybeExt] = local_binding.deref();

        let is_real = local[0].clone(); // reserved_poly[0] = Main(3) = is_real

        // #1: send(ShaExtend) — send to ShaExtend chip (16 rounds)
        builder.send(is_real.clone());

        // #2: recv(ShaExtend) — receive from ShaExtend chip (64 rounds)
        builder.recv(is_real.clone());

        // #3: send(Global) — syscall interaction to global table
        builder.send(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::controller::ShaExtendControllerChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{InteractionScope, MachineAir},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for ShaExtendControllerPolyAir {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROLLER_COLS
    }
}

impl<F: Field> MachineAir<F> for ShaExtendControllerPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <ShaExtendControllerChip as MachineAir<F>>::name(&ShaExtendControllerChip {}) + "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <ShaExtendControllerChip as MachineAir<F>>::generate_dependencies(
            &ShaExtendControllerChip {},
            input,
            output,
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <ShaExtendControllerChip as MachineAir<F>>::num_rows(&ShaExtendControllerChip {}, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShaExtendControllerChip {}.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShaExtendControllerChip as MachineAir<F>>::included(&ShaExtendControllerChip {}, shard)
        /*
        // for check_polyair_lookups test
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_sha_extend_empty()
        }
        */
    }

    fn commit_scope(&self) -> InteractionScope {
        <ShaExtendControllerChip as MachineAir<F>>::commit_scope(&ShaExtendControllerChip {})
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        programs::tests::sha_extend_program,
        syscall::precompiles::sha256::extend_dt::controller::ShaExtendControllerChip,
    };
    use dt_core_executor::{syscalls::SyscallCode, ExecutionRecord, Executor};
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

    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const BATCH_SIZE: usize = 3;
    const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

    fn ef(x: u32) -> EF {
        EF::from(F::from_canonical_u32(x))
    }

    /// BabyBear modulus: 15 * 2^27 + 1
    const BABYBEAR_MODULUS: u32 = 2013265921;

    fn random_f(rng: &mut StdRng) -> F {
        let value = rng.gen_range(0..BABYBEAR_MODULUS);
        F::from_canonical_u32(value)
    }

    fn random_ef(rng: &mut StdRng) -> EF {
        let values: [F; 4] = [random_f(rng), random_f(rng), random_f(rng), random_f(rng)];
        EF::from_base_slice(&values)
    }

    fn challenge_beta_with_seed(seed: u64) -> EF {
        let mut rng = StdRng::seed_from_u64(seed);
        random_ef(&mut rng)
    }

    fn beta_powers(air: &ShaExtendControllerPolyAir, beta: EF) -> Vec<EF> {
        let required_max_beta_power = <ShaExtendControllerPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(air);
        (0..=required_max_beta_power).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    fn random_reducer(seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
        // Gate constraints: 1
        // Lookup batch: ceil(3/3) = 1
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 1;
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
        air: &ShaExtendControllerPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <ShaExtendControllerPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(air);
        let empty_prep: Vec<F> = vec![];
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            let main_row: &[F] = core::ops::Deref::deref(&main_binding);
            let reserved = collect_reserved_poly(main_row, &empty_prep, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = sha_extend_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty() {
                continue;
            }

            let mut sub_shard = ExecutionRecord::new(shard.program.clone());
            sub_shard.precompile_events = shard.precompile_events.clone();

            let chip = ShaExtendControllerChip::new();
            return Some(
                chip.generate_trace(&sub_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_sha_extend_controller_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No ShaExtendController trace found — skipping test");
                return;
            }
        };

        let air = ShaExtendControllerPolyAir::new();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let beta_powers = beta_powers(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];

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
        assert!(
            first.iter().all(|x| x.is_zero()),
            "ShaExtendController first_round non-zero at indices: {:?}",
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
            "ShaExtendController nonfirst_round non-zero at indices: {:?}",
            nonfirst
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
                .collect::<Vec<_>>()
        );
    }

    fn random_sha_extend_controller_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace().expect("sample trace should exist");
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
        let width = base.width();
        let last_row_start = (base_height - 1) * width;
        let last_row = &base.values[last_row_start..last_row_start + width];
        let mut values = Vec::with_capacity(target_height * width);
        values.extend_from_slice(&base.values);
        for _ in base_height..target_height {
            values.extend_from_slice(last_row);
        }
        RowMajorMatrix::new(values, width)
    }

    #[test]
    #[ignore = "performance test; run manually"]
    fn perf_multi_round_sumcheck() {
        let air = ShaExtendControllerPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SHA_EXTEND_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sha_extend_controller_trace(log_n, seed);
        let height = main.height();
        assert_eq!(height, 1 << log_n);
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let bp = beta_powers(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
        let global = EF::zero();
        let reserved_poly_desc = <ShaExtendControllerPolyAir as FullAir<
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
            &bp,
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
            &bp,
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
            &bp,
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
                &bp,
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
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl ShaExtendControllerPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_LOOKUPS
    }
}
