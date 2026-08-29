//! PolyAir adaptation of SyscallChip.
//!
//! The SyscallChip bridges syscall invocations between local and global scopes.
//! It is parameterized by `SyscallShardKind` (Core / Precompile), which determines
//! the direction of the Syscall interaction (recv vs send) and the Global interaction
//! payload (is_send/is_receive flags).
//!
//! ## Interaction Summary (2 total)
//!
//! ### Core variant:
//!  1. recv(Syscall, is_real)  — [shard, clk, syscall_id, arg1, arg2]
//!  2. send(Global, is_real)   — [shard, clk, syscall_id, arg1, arg2, 0, 0, 1, 0, Syscall_kind]
//!
//! ### Precompile variant:
//!  1. send(Syscall, is_real)  — [shard, clk, syscall_id, arg1, arg2]
//!  2. send(Global, is_real)   — [shard, clk, syscall_id, arg1, arg2, 0, 0, 0, 1, Syscall_kind]
//!
//! Gate constraints:
//!  - `is_real * (1 - is_real) = 0` (boolean enforcement, replaces BitVec for single-bit case)

use std::ops::Deref;

use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{FullAir, FullAirBuilder, InteractionScope, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
    InteractionKind,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::chip::{SyscallChip, SyscallShardKind, NUM_SYSCALL_COLS};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
const NUM_LOOKUPS: usize = 2;

/// Maximum number of values in a single lookup payload.
/// Global interaction has 10 values — the largest payload.
const MAX_LOOKUP_VALUES: usize = 10;

// SyscallCols column indices:
//  0: shard
//  1: clk
//  2: syscall_id
//  3: arg1
//  4: arg2
//  5: is_real

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for SyscallChip.
///
/// Parameterized by `SyscallShardKind` to distinguish Core (recv Syscall)
/// from Precompile (send Syscall) behavior.
///
/// Gate constraints: assert_bool(is_real) — is_real * (1 - is_real) = 0.
/// Correctness is enforced by the permutation argument (LogUp) and BitVec.
#[derive(Clone, Copy)]
pub struct SyscallChipPolyAir {
    pub shard_kind: SyscallShardKind,
}

impl SyscallChipPolyAir {
    pub const fn new(shard_kind: SyscallShardKind) -> Self {
        Self { shard_kind }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for SyscallChipPolyAir {
    fn width(&self) -> usize {
        NUM_SYSCALL_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Only is_real is needed in lookup() for multiplicities.
        vec![PairCol::Main(5)] // is_real
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let ptr = main.as_ptr();
        let col = |i: usize| -> AB::VarMaybeExt { unsafe { (*ptr.add(i)).clone() } };

        let zero = AB::zero_maybe();
        let one = AB::VarMaybeExt::from(AB::F::one());

        let shard = col(0);
        let clk = col(1);
        let syscall_id = col(2);
        let arg1 = col(3);
        let arg2 = col(4);
        let _is_real = col(5);

        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let global_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));
        let syscall_interaction_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(InteractionKind::Syscall as u8));

        // =====================================================================
        // #1: Syscall interaction (recv for Core, send for Precompile)
        // Payload: [shard, clk, syscall_id, arg1, arg2]
        // =====================================================================
        builder.retain_precomputed(builder.lookup_denominator(
            syscall_kind,
            vec![shard.clone(), clk.clone(), syscall_id.clone(), arg1.clone(), arg2.clone()],
        ));

        // =====================================================================
        // #2: send(Global) — syscall interaction to global table
        // Core:       [shard, clk, syscall_id, arg1, arg2, 0, 0, 1, 0, Syscall_kind]
        // Precompile: [shard, clk, syscall_id, arg1, arg2, 0, 0, 0, 1, Syscall_kind]
        // =====================================================================
        let (is_send, is_receive) = match self.shard_kind {
            SyscallShardKind::Core => (one, zero.clone()),
            SyscallShardKind::Precompile => (zero.clone(), one),
        };

        builder.retain_precomputed(builder.lookup_denominator(
            global_kind,
            vec![
                shard,
                clk,
                syscall_id,
                arg1,
                arg2,
                zero.clone(), // padding 0
                zero,         // padding 0
                is_send,
                is_receive,
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

        let is_real = local[0].clone(); // reserved_poly[0] = Main(5) = is_real

        // #1: Syscall interaction (recv for Core, send for Precompile)
        match self.shard_kind {
            SyscallShardKind::Core => builder.recv(is_real.clone()),
            SyscallShardKind::Precompile => builder.send(is_real.clone()),
        }

        // #2: send(Global) — syscall interaction to global table
        builder.send(is_real);
    }
}

// =============================================================================
// MachineAir implementation (delegation to SyscallChip via field forwarding)
// =============================================================================

impl<F: Field> BaseAir<F> for SyscallChipPolyAir {
    fn width(&self) -> usize {
        NUM_SYSCALL_COLS
    }
}

impl<F: Field> MachineAir<F> for SyscallChipPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        let c = SyscallChip::new(self.shard_kind);
        <SyscallChip as MachineAir<F>>::name(&c) + "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let c = SyscallChip::new(self.shard_kind);
        <SyscallChip as MachineAir<F>>::generate_dependencies(&c, input, output)
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let c = SyscallChip::new(self.shard_kind);
        <SyscallChip as MachineAir<F>>::num_rows(&c, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        SyscallChip::new(self.shard_kind).generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        let c = SyscallChip::new(self.shard_kind);
        <SyscallChip as MachineAir<F>>::included(&c, shard)
    }

    fn commit_scope(&self) -> InteractionScope {
        let c = SyscallChip::new(self.shard_kind);
        <SyscallChip as MachineAir<F>>::commit_scope(&c)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        check_constraints::run_generate_dependencies, programs::tests::keccak_program,
        syscall::chip::SyscallChip,
    };
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

    fn beta_powers(air: &SyscallChipPolyAir, beta: EF) -> Vec<EF> {
        let required_max_beta_power = <SyscallChipPolyAir as FullAir<
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
        // Gate constraints: cpu_state(1) = 1
        // Lookup batch: ceil(2/3) = 1
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
        air: &SyscallChipPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <SyscallChipPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

    /// Build the reference Syscall trace by running a program through the executor,
    /// generating derived dependencies, and calling `SyscallChip::generate_trace`.
    fn sample_trace(shard_kind: SyscallShardKind) -> RowMajorMatrix<F> {
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();
        let mut shard = *runtime.records[0].clone();
        run_generate_dependencies(&mut shard);

        let chip = SyscallChip::new(shard_kind);
        chip.generate_trace(&shard, &mut ExecutionRecord::default()).decompress()
    }

    fn run_constraint_check(shard_kind: SyscallShardKind) {
        let air = SyscallChipPolyAir::new(shard_kind);
        let main = sample_trace(shard_kind);
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
            "{:?} first_round non-zero at indices: {:?}",
            shard_kind,
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
            "{:?} nonfirst_round non-zero at indices: {:?}",
            shard_kind,
            nonfirst
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_core_constraint_check_satisfied() {
        run_constraint_check(SyscallShardKind::Core);
    }

    #[test]
    fn test_precompile_constraint_check_satisfied() {
        run_constraint_check(SyscallShardKind::Precompile);
    }

    fn random_syscall_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
        assert!(log_n < usize::BITS as usize);
        let target_height = 1usize << log_n;
        let base = sample_trace(SyscallShardKind::Core);
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
        let air = SyscallChipPolyAir::new(SyscallShardKind::Core);
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_syscall_trace(log_n, seed);
        let height = main.height();
        assert!(height >= 2);
        std::println!("perf_multi_round: log_n={}, h={}, seed={}", log_n, height, seed);

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000));
        let alpha = random_ef(&mut rng);
        let beta = challenge_beta_with_seed(seed.wrapping_add(2000));
        let beta_powers = beta_powers(&air, beta);
        let beta_septix = beta_septix(beta);
        let public: Vec<F> = vec![];
        let constraint_reducer = random_reducer(seed.wrapping_add(3000));
        let global = EF::zero();
        let reserved_poly_desc = <SyscallChipPolyAir as FullAir<
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
}
