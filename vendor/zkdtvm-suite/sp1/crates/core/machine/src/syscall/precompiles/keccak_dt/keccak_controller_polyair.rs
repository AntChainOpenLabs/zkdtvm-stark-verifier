//! PolyAir adaptation of KeccakControllerChip.
//!
//! The KeccakController bridges syscall invocations to the KeccakPermute chip:
//!   - Sends initial 1600-bit state (50 CompactWords) to Keccak permutation
//!   - Receives final state after 24 rounds
//!   - Writes 50 words of final state back to memory
//!   - Registers the syscall with the global interaction table
//!
//! ## Interaction Summary (203 total)
//!
//!   #1:      send(Keccak, is_real)  — [shard, clk, 0, prev_state_compact[0..49]]
//!   #2:      recv(Keccak, is_real)  — [shard, clk, 24, new_state_compact[0..49]]
//!   #3-202:  50x memory_readwrite  — 4 interactions each (ts_u16, ts_bit12, mem_send, mem_recv)
//!   #203:    send(Global, is_real)  — [shard, clk, syscall_id, state_ptr, 0,0,0,0, 1, Syscall]
//!
//! ## Gate Constraints
//!
//!   - is_real boolean                         (air.rs L157)
//!   - 50x memory timestamp constraints        (air.rs L190-198)
//!     - compare_clk boolean                   when is_real
//!     - shard == prev_shard                   when is_real and compare_clk
//!     - diff_minus_one = limb_16 + limb_12 * 2^16    when is_real

use std::ops::Deref;

use dt_core_executor::syscalls::SyscallCode;
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::memory::polyair::{memory_readwrite_lookup, memory_readwrite_precompute_lc};

use super::{
    controller::{KeccakControllerCols, NUM_KECCAK_CONTROLLER_COLS},
    STATE_NUM_WORDS,
};

// ============================================================================
// Constants
// ============================================================================

/// Total lookup interactions: 2 Keccak + 50x4 memory_readwrite + 1 Global = 203.
const NUM_LOOKUPS: usize = 2 + STATE_NUM_WORDS * 4 + 1;

/// Max payload size across all interactions.
/// Keccak send/recv: 3 header + 50x2 compact words = 103 values.
const MAX_LOOKUP_VALUES: usize = 3 + STATE_NUM_WORDS * 2;

/// Precomputed linear combinations (one per lookup).
const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for KeccakControllerChip.
#[derive(Clone, Copy, Default)]
pub struct KeccakControllerPolyAir;

impl KeccakControllerPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for KeccakControllerPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_KECCAK_CONTROLLER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Reserve all main trace columns — needed for memory timestamp gate
        // constraints and the Keccak/Global payloads.
        (0..NUM_KECCAK_CONTROLLER_COLS).map(PairCol::Main).collect()
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &KeccakControllerCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let zero = AB::zero_maybe();
        let multiplier_256 = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let state_ptr = local.state_ptr.clone();

        let keccak_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Keccak as usize));
        let global_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));

        // =================================================================
        // #1: send(Keccak, is_real) — initial state before permutation
        //     payload: [shard, clk, 0, compact(prev_value[0..49])]
        //     (air.rs L164-175)
        // =================================================================
        let mut send_values =
            vec![shard.clone(), clk.clone(), AB::VarMaybeExt::from(AB::F::from_canonical_u32(0))];
        for i in 0..STATE_NUM_WORDS {
            let pv = &local.state_access[i].prev_value;
            send_values.push(pv[0].clone() + pv[1].clone() * multiplier_256.clone());
            send_values.push(pv[2].clone() + pv[3].clone() * multiplier_256.clone());
        }
        builder.retain_precomputed(builder.lookup_denominator(keccak_kind.clone(), send_values));

        // =================================================================
        // #2: recv(Keccak, is_real) — final state after 24 rounds
        //     payload: [shard, clk, 24, compact(value[0..49])]
        //     (air.rs L176-187)
        // =================================================================
        let mut recv_values =
            vec![shard.clone(), clk.clone(), AB::VarMaybeExt::from(AB::F::from_canonical_u32(24))];
        for i in 0..STATE_NUM_WORDS {
            let val = &local.state_access[i].access.value;
            recv_values.push(val[0].clone() + val[1].clone() * multiplier_256.clone());
            recv_values.push(val[2].clone() + val[3].clone() * multiplier_256.clone());
        }
        builder.retain_precomputed(builder.lookup_denominator(keccak_kind, recv_values));

        // =================================================================
        // #3-202: 50x memory_readwrite (4 interactions each)
        //     (air.rs L189-198)
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..STATE_NUM_WORDS {
            let addr = state_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            memory_readwrite_precompute_lc(
                builder,
                &local.state_access[i].access,
                &local.state_access[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // #203: send(Global, is_real) — syscall registration
        //     payload: [shard, clk, syscall_id, state_ptr, 0,0,0,0, 1, Syscall]
        //     (air.rs L201-219)
        // =================================================================
        let syscall_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
            SyscallCode::KECCAK_PERMUTE.syscall_id(),
        ));
        let syscall_interaction_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(InteractionKind::Syscall as u8));

        builder.retain_precomputed(builder.lookup_denominator(
            global_kind,
            vec![
                shard,
                clk,
                syscall_id,
                state_ptr,
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
                AB::VarMaybeExt::from(AB::F::one()),
                syscall_interaction_kind,
            ],
        ));
    }

    // ========================================================================
    // Phase 2: gate constraints (reserved_poly columns only)
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &KeccakControllerCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const KeccakControllerCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();
        let one = AB::one_maybe();

        // -- air.rs L157: assert_bool(is_real) --
        builder.assert_zero(is_real.clone() * (one.clone() - is_real.clone()));

        // -- air.rs L190-198: 50x memory timestamp constraints --
        let write_clk = local.clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        let limb_base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));

        for access in &local.state_access {
            let mem = &access.access;
            let compare_clk = mem.compare_clk.clone();

            // compare_clk boolean (when is_real)
            builder
                .when(is_real.clone())
                .assert_zero(compare_clk.clone() * (one.clone() - compare_clk.clone()));

            // shard == prev_shard (when is_real and compare_clk)
            builder
                .when(is_real.clone())
                .when(compare_clk.clone())
                .assert_eq(local.shard.clone(), mem.prev_shard.clone());

            // 28-bit range decomposition: diff_minus_one = limb_16 + limb_12 * 2^16
            let prev_comp_value = compare_clk.clone() * mem.prev_clk.clone() +
                (one.clone() - compare_clk.clone()) * mem.prev_shard.clone();
            let current_comp_value = compare_clk.clone() * write_clk.clone() +
                (one.clone() - compare_clk) * local.shard.clone();
            let diff_minus_one = current_comp_value - prev_comp_value - one.clone();

            builder.when(is_real.clone()).assert_eq(
                diff_minus_one,
                mem.diff_16bit_limb.clone() + mem.diff_12bit_limb.clone() * limb_base.clone(),
            );
        }
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &KeccakControllerCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const KeccakControllerCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // #1: send(Keccak) — initial state
        builder.send(is_real.clone());

        // #2: recv(Keccak) — final state after 24 rounds
        builder.recv(is_real.clone());

        // #3-202: 50x memory_readwrite (4 interactions each)
        for _ in 0..STATE_NUM_WORDS {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #203: send(Global) — syscall registration
        builder.send(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::controller::KeccakControllerChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{InteractionScope, MachineAir},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for KeccakControllerPolyAir {
    fn width(&self) -> usize {
        NUM_KECCAK_CONTROLLER_COLS
    }
}

impl<F: Field> MachineAir<F> for KeccakControllerPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <KeccakControllerChip as MachineAir<F>>::name(&KeccakControllerChip {}) + "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <KeccakControllerChip as MachineAir<F>>::generate_dependencies(
            &KeccakControllerChip {},
            input,
            output,
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <KeccakControllerChip as MachineAir<F>>::num_rows(&KeccakControllerChip {}, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        KeccakControllerChip {}.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <KeccakControllerChip as MachineAir<F>>::included(&KeccakControllerChip {}, shard)
        /*
        // for check_polyair_lookups test
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_keccak_empty()
        }
        */
    }

    fn commit_scope(&self) -> InteractionScope {
        <KeccakControllerChip as MachineAir<F>>::commit_scope(&KeccakControllerChip {})
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 3;

    use crate::{
        programs::tests::keccak_program,
        syscall::precompiles::keccak_dt::controller::KeccakControllerChip,
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
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

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

    fn beta_powers(air: &KeccakControllerPolyAir, beta: EF) -> Vec<EF> {
        let max = <KeccakControllerPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::required_max_beta_power(air);
        (0..=max).map(|i| beta.exp_u64(i as u64)).collect()
    }

    fn beta_septix(beta: EF) -> EF {
        dt_stark::septic_curve_params::compute_beta_septix::<
            F,
            EF,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(beta)
    }

    // Gate constraints: 1 (is_real bool) + 50 * 3 (memory timestamp) = 151
    // Lookup batch constraints: ceil(203 / 3) = 68
    // Cumulative sum constraints: 3 (first_row + transition + last_row)
    // Total: 151 + 68 + 3 = 222
    const NUM_GATE_CONSTRAINTS: usize = 1 + STATE_NUM_WORDS * 3;
    const NUM_REDUCER_CONSTRAINTS: usize =
        NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;

    fn random_reducer(seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
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
        air: &KeccakControllerPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <KeccakControllerPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(air);
        let empty_prep: Vec<F> = vec![];
        let mut values = Vec::new();
        for row_idx in 0..main.height() {
            let main_binding = main.row_slice(row_idx);
            let main_row: &[F] = Deref::deref(&main_binding);
            let reserved = collect_reserved_poly(main_row, &empty_prep, &reserved_poly);
            values.extend(reserved.into_iter().map(EF::from));
        }
        RowMajorMatrix::new(values, reserved_poly.len())
    }

    /// Build a real trace from the keccak test program.
    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = keccak_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.precompile_events.is_keccak_empty() {
                continue;
            }

            // Build a minimal record containing only keccak precompile events.
            // We intentionally skip run_generate_dependencies and the `included`
            // check: run_generate_dependencies iterates all RiscvAir chips and
            // appends global_memory events that would violate the `included`
            // predicate (which requires an empty global_memory_*_events).
            // Since we only need the KeccakController trace, we directly invoke
            // generate_trace on a clean shard with just the precompile events.
            let mut keccak_shard = ExecutionRecord::new(shard.program.clone());
            keccak_shard.precompile_events = shard.precompile_events.clone();

            let chip = KeccakControllerChip::new();
            return Some(
                chip.generate_trace(&keccak_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_keccak_controller_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No KeccakController trace found -- skipping test");
                return;
            }
        };

        let air = KeccakControllerPolyAir::new();
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

    fn random_keccak_controller_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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
        let air = KeccakControllerPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::KECCAK_PERMUTE_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_keccak_controller_trace(log_n, seed);
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
        let reserved_poly_desc = <KeccakControllerPolyAir as FullAir<
            PrecomputeRowBuilder<'_, F, F, EF>,
        >>::reserved_poly(&air);

        // Precompute
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

        // Permutation
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

        // Round 0
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

        // Rounds 1..log_n-1
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
}

// PolyAir local-scope interaction counts (used by the check_polyair_lookups binary).
impl KeccakControllerPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
