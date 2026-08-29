//! PolyAir adaptation of ShaCompressControllerChip.
//!
//! The ShaCompressControllerChip bridges SHA-256 compress syscall invocations between
//! the syscall table and the ShaCompress permutation chip. It handles:
//!   - ShaCompress send/recv interactions (initial/final hash state)
//!   - 8 memory write accesses (writing final h[0..7] values)
//!   - 8 AddN<U2> operations (h_init + h_finalize = h_access.value)
//!   - Global syscall interaction
//!
//! ## Interaction Summary (35 total)
//!
//!  1. send(ShaCompress, is_real)  — [shard, clk, w_ptr, 0, h_init[0..7].compact()]
//!  2. recv(ShaCompress, is_real)  — [shard, clk, w_ptr, 64, h_final[0..7].compact()]
//!  3-34. 8× memory_readwrite (4 interactions each = 32)
//!  35. send(Global, is_real)      — [shard, clk, syscall_id, w_ptr, h_ptr, 0, 0, 0, 1,
//!      Syscall_kind]
//!
//! ## Gate Constraints
//!
//!  - assert_bool(is_real)
//!  - 8× AddN<U2> carry chain: carry = (sum + carry_prev - result) / 2^16, assert_bool(carry)
//!  - 8× memory_timestamp_gate_constraints for the final hash writes

use std::ops::Deref;

use dt_core_executor::syscalls::SyscallCode;
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    memory::polyair::{
        memory_readwrite_lookup, memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
    },
    operations_dt::{add_n_without_result_gate_constraints, CompactWord},
};

use super::controller::{ShaCompressControllerCols, NUM_SHA_COMPRESS_CONTROLLER_COLS};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions:
/// 2 ShaCompress + 8×4 memory_readwrite + 1 Global = 35
const NUM_LOOKUPS: usize = 35;

/// Maximum number of values in a single lookup payload.
/// ShaCompress interaction has 20 values (4 header + 8×2 compact words).
const MAX_LOOKUP_VALUES: usize = 20;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for ShaCompressControllerChip.
#[derive(Clone, Copy, Default)]
pub struct ShaCompressControllerPolyAir;

impl ShaCompressControllerPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for ShaCompressControllerPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROLLER_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Reserve all columns — gate constraints need h_access, h_finalize, is_real.
        (0..NUM_SHA_COMPRESS_CONTROLLER_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShaCompressControllerCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let zero = AB::zero_maybe();

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let w_ptr = local.w_ptr.clone();
        let h_ptr = local.h_ptr.clone();

        let sha_compress_kind = AB::VarMaybeExt::from(AB::F::from_canonical_usize(
            InteractionKind::ShaCompress as usize,
        ));
        let global_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Global as usize));
        let syscall_interaction_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_u8(InteractionKind::Syscall as u8));
        let syscall_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
            SyscallCode::SHA_COMPRESS.syscall_id(),
        ));

        // Build h_initialize (prev_value as CompactWord) and h_finalize
        let h_initialize: [CompactWord<AB::VarMaybeExt>; 8] = std::array::from_fn(|i| {
            let pv = &local.h_access[i].prev_value;
            let multiplier = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
            CompactWord([
                pv[0].clone() + pv[1].clone() * multiplier.clone(),
                pv[2].clone() + pv[3].clone() * multiplier.clone(),
            ])
        });

        // =====================================================================
        // #1: send(ShaCompress, is_real) — [shard, clk, w_ptr, 0, h_init[0..7].compact()]
        // =====================================================================
        let mut send_values = vec![
            shard.clone(),
            clk.clone(),
            w_ptr.clone(),
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(0)),
        ];
        for h_init in &h_initialize {
            send_values.push(h_init.0[0].clone());
            send_values.push(h_init.0[1].clone());
        }
        builder
            .retain_precomputed(builder.lookup_denominator(sha_compress_kind.clone(), send_values));

        // =====================================================================
        // #2: recv(ShaCompress, is_real) — [shard, clk, w_ptr, 64, h_final[0..7].compact()]
        // =====================================================================
        let mut recv_values = vec![
            shard.clone(),
            clk.clone(),
            w_ptr.clone(),
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(64)),
        ];
        for h_fin in &local.h_finalize {
            recv_values.push(h_fin.0[0].clone());
            recv_values.push(h_fin.0[1].clone());
        }
        builder.retain_precomputed(builder.lookup_denominator(sha_compress_kind, recv_values));

        // =====================================================================
        // #3-34: 8× memory_readwrite (4 interactions each)
        // =====================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..8 {
            let addr = h_ptr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32(
                    (i * core::mem::size_of::<u32>()) as u32,
                ));
            let prev_value = &local.h_access[i].prev_value;
            memory_readwrite_precompute_lc(
                builder,
                &local.h_access[i].access,
                prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =====================================================================
        // #35: send(Global, is_real) — syscall interaction to global table
        // [shard, clk, syscall_id, w_ptr, h_ptr, 0, 0, 0, 1, Syscall_kind]
        // =====================================================================
        builder.retain_precomputed(builder.lookup_denominator(
            global_kind,
            vec![
                shard,
                clk,
                syscall_id,
                w_ptr,
                h_ptr,
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
        let local: &ShaCompressControllerCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const ShaCompressControllerCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // ── controller.rs L157: assert_bool(is_real) ──
        builder.assert_zero(is_real.clone() * (AB::one_maybe() - is_real.clone()));

        // ── controller.rs L175-183: 8× AddNOperationWithoutResult<U2>::eval ──
        let multiplier_256 = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
        let write_clk = local.clk.clone() + AB::VarMaybeExt::from(AB::F::one());

        for i in 0..8 {
            // h_initialize[i] = prev_value as CompactWord
            let pv = &local.h_access[i].prev_value;
            let h_init = CompactWord([
                pv[0].clone() + pv[1].clone() * multiplier_256.clone(),
                pv[2].clone() + pv[3].clone() * multiplier_256.clone(),
            ]);

            // h_finalize[i]
            let h_fin = &local.h_finalize[i];

            // result = h_access[i].access.value as CompactWord
            let val = &local.h_access[i].access.value;
            let result = CompactWord([
                val[0].clone() + val[1].clone() * multiplier_256.clone(),
                val[2].clone() + val[3].clone() * multiplier_256.clone(),
            ]);

            add_n_without_result_gate_constraints(
                builder,
                &[h_init, h_fin.clone()],
                result,
                is_real.clone(),
            );

            memory_timestamp_gate_constraints(
                builder,
                &local.h_access[i].access,
                local.shard.clone(),
                write_clk.clone(),
                is_real.clone(),
            );
        }
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &ShaCompressControllerCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const ShaCompressControllerCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // #1: send(ShaCompress) — initial hash state
        builder.send(is_real.clone());

        // #2: recv(ShaCompress) — final hash state
        builder.recv(is_real.clone());

        // #3-34: 8× memory_readwrite (4 interactions each)
        for _ in 0..8 {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #35: send(Global) — syscall interaction to global table
        builder.send(is_real);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::controller::ShaCompressControllerChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{
    air::{InteractionScope, MachineAir},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for ShaCompressControllerPolyAir {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROLLER_COLS
    }
}

impl<F: Field> MachineAir<F> for ShaCompressControllerPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <ShaCompressControllerChip as MachineAir<F>>::name(&ShaCompressControllerChip {}) +
            "PolyAir"
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <ShaCompressControllerChip as MachineAir<F>>::generate_dependencies(
            &ShaCompressControllerChip {},
            input,
            output,
        )
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        <ShaCompressControllerChip as MachineAir<F>>::num_rows(&ShaCompressControllerChip {}, input)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShaCompressControllerChip {}.generate_trace(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShaCompressControllerChip as MachineAir<F>>::included(&ShaCompressControllerChip {}, shard)
        /*
        // for check_polyair_lookups test
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.precompile_events.is_sha_compress_empty()
        }
        */
    }

    fn commit_scope(&self) -> InteractionScope {
        <ShaCompressControllerChip as MachineAir<F>>::commit_scope(&ShaCompressControllerChip {})
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        programs::tests::sha_compress_program,
        syscall::precompiles::sha256::compress_dt::controller::ShaCompressControllerChip,
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

    fn beta_powers(air: &ShaCompressControllerPolyAir, beta: EF) -> Vec<EF> {
        let required_max_beta_power = <ShaCompressControllerPolyAir as FullAir<
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
        // Gate constraints: 41
        // Lookup batch: ceil(35/3) = 12
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 41;
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
        air: &ShaCompressControllerPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <ShaCompressControllerPolyAir as FullAir<
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
        let program = sha_compress_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();

            if shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty() {
                continue;
            }

            let mut sub_shard = ExecutionRecord::new(shard.program.clone());
            sub_shard.precompile_events = shard.precompile_events.clone();

            let chip = ShaCompressControllerChip::new();
            return Some(
                chip.generate_trace(&sub_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_sha_compress_controller_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No ShaCompressController trace found — skipping test");
                return;
            }
        };

        let air = ShaCompressControllerPolyAir::new();
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
            "ShaCompressController first_round non-zero at indices: {:?}",
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
            "ShaCompressController nonfirst_round non-zero at indices: {:?}",
            nonfirst
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, _)| i)
                .take(5)
                .collect::<Vec<_>>()
        );
    }

    fn random_sha_compress_controller_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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
        let air = ShaCompressControllerPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SHA_COMPRESS_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sha_compress_controller_trace(log_n, seed);
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
        let reserved_poly_desc = <ShaCompressControllerPolyAir as FullAir<
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
impl ShaCompressControllerPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_LOOKUPS
    }
}
