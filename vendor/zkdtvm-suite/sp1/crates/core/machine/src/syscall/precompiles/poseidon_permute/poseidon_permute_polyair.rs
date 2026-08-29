//! PolyAir adaptation of Poseidon2PermuteChip.
//!
//! Bridges `Poseidon2MemCols` constraints to PolyAir's `FullAir` four-phase model.
//!
//! ## Interaction Summary (289 total)
//!
//!   #1   ..#96:   state_mem_read — 24 memory readwrite accesses × 4 interactions each
//!   #97  ..#192:  state_mem_write — 24 memory readwrite accesses × 4 interactions each
//!   #193 ..#240:  read U8Range — 24 words × 2 pairs each = 48
//!   #241 ..#288:  write U8Range — 24 words × 2 pairs each = 48
//!   #289:         recv(Syscall)
//!
//! ## Gate Constraints
//!
//!   Outer constraints:
//!   - 24 × assert_word_eq(read.value, read.prev_value) — read-only enforcement
//!   - 24 × assert_eq(input_state[i], read_val_reduced) — input linkage
//!   - 24 × assert_eq(output_state[i], write_val_reduced) — output linkage
//!   - 48 × memory_timestamp_gate_constraints — timestamp verification
//!   - 1  × is_real boolean
//!
//!   Inner Poseidon2 constraints (426 total):
//!   - 384 full round constraints (8 rounds × 48)
//!   - 42 partial round constraints (21 rounds × 2)
//!
//! ## Boolean handling (≤3 → direct gate constraint)
//!   - is_real: 1 boolean → direct gate constraint

use std::ops::Deref;

use dt_core_executor::syscalls::SyscallCode;
use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use super::{
    columns::{Poseidon2MemCols, NUM_POSEIDON2_MEM_COLS},
    poseidon2_inner::{
        constants::RoundConstants, poseidon2_polyair::poseidon2_inner_gate_constraints,
    },
    STATE_NUM_WORDS,
};
use crate::{
    bytes::polyair::{slice_u8_range_lookup, slice_u8_range_precompute_lc},
    memory::polyair::{
        memory_readwrite_lookup, memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
    },
};

// ============================================================================
// Constants
// ============================================================================

/// Total number of lookup interactions.
///
/// = STATE_NUM_WORDS × 4 (read memory)
/// + STATE_NUM_WORDS × 4 (write memory)
/// + STATE_NUM_WORDS × 2 (read U8Range)
/// + STATE_NUM_WORDS × 2 (write U8Range)
/// + 1 (recv Syscall)
const NUM_LOOKUPS: usize =
    STATE_NUM_WORDS * 4 + STATE_NUM_WORDS * 4 + STATE_NUM_WORDS * 2 + STATE_NUM_WORDS * 2 + 1;

/// Precomputed linear combinations: one per lookup interaction.
const NUM_PRECOMPUTED: usize = NUM_LOOKUPS;

/// Maximum number of values in a single lookup payload.
/// Memory interaction has 7 values (shard, clk, addr, value[0..3]) — the largest.
const MAX_LOOKUP_VALUES: usize = 7;

// ============================================================================
// PolyAir wrapper
// ============================================================================

/// PolyAir wrapper for Poseidon2PermuteChip.
pub struct Poseidon2PermutePolyAir<F: p3_field::Field> {
    pub constants: RoundConstants<F>,
}

impl<F: p3_field::Field> Poseidon2PermutePolyAir<F> {
    pub fn new(constants: RoundConstants<F>) -> Self {
        Self { constants }
    }
}

impl<F: p3_field::Field> Default for Poseidon2PermutePolyAir<F> {
    fn default() -> Self {
        Self { constants: RoundConstants::default() }
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for Poseidon2PermutePolyAir<AB::F>
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_POSEIDON2_MEM_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        // Reserve all main trace columns for gate constraints.
        (0..NUM_POSEIDON2_MEM_COLS).map(PairCol::Main).collect()
    }

    // ========================================================================
    // Phase 1: precompute_lc — build lookup denominators
    // ========================================================================

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Poseidon2MemCols<AB::VarMaybeExt> =
            unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let state_addr = local.state_addr.clone();

        // =================================================================
        // #1..#96: state_mem_read — 24 memory readwrite accesses
        // Memory reads at (shard, clk, state_addr + i*4).
        // Although these are "reads" (value == prev_value enforced by gate),
        // the column type is MemoryReadWriteCols, so we use readwrite helper.
        // =================================================================
        for i in 0..STATE_NUM_WORDS {
            let addr = state_addr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32((i as u32) * 4));
            memory_readwrite_precompute_lc(
                builder,
                &local.state_mem_read[i].access,
                &local.state_mem_read[i].prev_value,
                addr,
                shard.clone(),
                clk.clone(),
            );
        }

        // =================================================================
        // #97..#192: state_mem_write — 24 memory readwrite accesses
        // Memory writes at (shard, clk+1, state_addr + i*4).
        // =================================================================
        let write_clk = clk.clone() + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..STATE_NUM_WORDS {
            let addr = state_addr.clone() +
                AB::VarMaybeExt::from(AB::F::from_canonical_u32((i as u32) * 4));
            memory_readwrite_precompute_lc(
                builder,
                &local.state_mem_write[i].access,
                &local.state_mem_write[i].prev_value,
                addr,
                shard.clone(),
                write_clk.clone(),
            );
        }

        // =================================================================
        // #193..#240: read U8Range — 24 words × 2 pairs = 48 interactions
        // slice_range_check_u8 on each read word's value bytes.
        // =================================================================
        for i in 0..STATE_NUM_WORDS {
            slice_u8_range_precompute_lc(builder, &local.state_mem_read[i].access.value.0);
        }

        // =================================================================
        // #241..#288: write U8Range — 24 words × 2 pairs = 48 interactions
        // slice_range_check_u8 on each write word's value bytes.
        // =================================================================
        for i in 0..STATE_NUM_WORDS {
            slice_u8_range_precompute_lc(builder, &local.state_mem_write[i].access.value.0);
        }

        // =================================================================
        // #289: recv(Syscall)
        // =================================================================
        let syscall_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::Syscall as usize));
        let syscall_id = AB::VarMaybeExt::from(AB::F::from_canonical_u32(
            SyscallCode::POSEIDON2_PERMUTE.syscall_id(),
        ));
        let zero = AB::zero_maybe();
        builder.retain_precomputed(
            builder
                .lookup_denominator(syscall_kind, vec![shard, clk, syscall_id, state_addr, zero]),
        );
    }

    // ========================================================================
    // Phase 2: eval — gate constraints (reserved_poly columns only)
    // ========================================================================

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &Poseidon2MemCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const Poseidon2MemCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();
        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let one = AB::one_maybe();

        let input_state = local.poseidon2_cols.inputs.clone();
        let output_state = local.poseidon2_cols.output_state().clone();

        // ── air.rs L38-53: Memory read constraints ──
        for i in 0..STATE_NUM_WORDS {
            // assert_word_eq(read.value, read.prev_value) — read-only enforcement
            for byte_idx in 0..4 {
                builder.when(is_real.clone()).assert_zero(
                    local.state_mem_read[i].access.value.0[byte_idx].clone() -
                        local.state_mem_read[i].prev_value.0[byte_idx].clone(),
                );
            }

            // assert_eq(input_state[i], read_val.reduce())
            // reduce() = value[0] + value[1]*256 + value[2]*65536 + value[3]*16777216
            let read_val = word_reduce::<AB>(&local.state_mem_read[i].access.value.0);
            builder.when(is_real.clone()).assert_zero(input_state[i].clone() - read_val);
        }

        // ── air.rs L55-58: Memory write constraints ──
        for i in 0..STATE_NUM_WORDS {
            // assert_eq(output_state[i], write_val.reduce())
            let write_val = word_reduce::<AB>(&local.state_mem_write[i].access.value.0);
            builder.when(is_real.clone()).assert_zero(output_state[i].clone() - write_val);
        }

        // ── air.rs L37-60: Memory timestamp gate constraints ──
        // Read memory accesses at (shard, clk)
        for i in 0..STATE_NUM_WORDS {
            memory_timestamp_gate_constraints(
                builder,
                &local.state_mem_read[i].access,
                shard.clone(),
                clk.clone(),
                is_real.clone(),
            );
        }
        // Write memory accesses at (shard, clk+1)
        let write_clk = clk + AB::VarMaybeExt::from(AB::F::one());
        for i in 0..STATE_NUM_WORDS {
            memory_timestamp_gate_constraints(
                builder,
                &local.state_mem_write[i].access,
                shard.clone(),
                write_clk.clone(),
                is_real.clone(),
            );
        }

        // ── air.rs L80: assert_bool(is_real) ──
        builder.assert_zero(is_real.clone() * (one - is_real));

        // ── air.rs L82-90: Inner Poseidon2 gate constraints ──
        poseidon2_inner_gate_constraints(builder, &local.poseidon2_cols, &self.constants);
    }

    // ========================================================================
    // Phase 3: lookup — declare send/recv multiplicities
    // ========================================================================

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &Poseidon2MemCols<AB::VarMaybeExt> = unsafe {
            &*(local_binding.deref().as_ptr() as *const Poseidon2MemCols<AB::VarMaybeExt>)
        };

        let is_real = local.is_real.clone();

        // #1..#96: state_mem_read (24 × 4 interactions)
        for _ in 0..STATE_NUM_WORDS {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #97..#192: state_mem_write (24 × 4 interactions)
        for _ in 0..STATE_NUM_WORDS {
            memory_readwrite_lookup(builder, is_real.clone());
        }

        // #193..#240: read U8Range (24 × 2 interactions)
        slice_u8_range_lookup(builder, is_real.clone(), STATE_NUM_WORDS * 2);

        // #241..#288: write U8Range (24 × 2 interactions)
        slice_u8_range_lookup(builder, is_real.clone(), STATE_NUM_WORDS * 2);

        // #289: recv(Syscall)
        builder.recv(is_real);
    }
}

// ============================================================================
// Helper: word reduce for VarMaybeExt
// ============================================================================

/// Compute `value[0] + value[1]*256 + value[2]*65536 + value[3]*16777216`
/// using `VarMaybeExt` arithmetic (no `AbstractField::from_canonical_u32`).
fn word_reduce<AB: FullAirBuilder>(value: &[AB::VarMaybeExt; 4]) -> AB::VarMaybeExt
where
    AB::VarMaybeExt: Clone,
{
    let b1 = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 8));
    let b2 = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 16));
    let b3 = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1 << 24));
    value[0].clone() + value[1].clone() * b1 + value[2].clone() * b2 + value[3].clone() * b3
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use super::Poseidon2PermuteChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;

impl<F: p3_field::Field> BaseAir<F> for Poseidon2PermutePolyAir<F> {
    fn width(&self) -> usize {
        NUM_POSEIDON2_MEM_COLS
    }
}

impl<F: p3_field::Field> MachineAir<F> for Poseidon2PermutePolyAir<F> {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        Poseidon2PermuteChip::<F>::default().name() + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        Poseidon2PermuteChip::<F>::default().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        Poseidon2PermuteChip::<F>::default().generate_dependencies(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        Poseidon2PermuteChip::<F>::default().included(shard)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "koalabear")]
mod tests {
    use super::*;

    const BATCH_SIZE: usize = 3;

    use crate::syscall::precompiles::poseidon_permute::Poseidon2PermuteChip;
    use dt_core_executor::{
        syscalls::SyscallCode, ExecutionRecord, Executor, Instruction, Opcode, Program,
    };

    /// Build a test program that invokes POSEIDON2_PERMUTE.
    /// Inlined from `permute_tests::poseidon2_permute_program()`.
    fn poseidon2_permute_program() -> Program {
        let state_ptr = 100u32;
        let mut instructions = vec![Instruction::new(Opcode::ADD, 29, 0, 1, false, true)];
        for i in 0..24u32 {
            instructions.extend(vec![
                Instruction::new(Opcode::ADD, 30, 0, state_ptr + i * 4, false, true),
                Instruction::new(Opcode::SW, 29, 30, 0, false, true),
            ]);
        }
        instructions.extend(vec![
            Instruction::new(Opcode::ADD, 5, 0, SyscallCode::POSEIDON2_PERMUTE as u32, false, true),
            Instruction::new(Opcode::ADD, 10, 0, state_ptr, false, true),
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false),
        ]);
        Program::new(instructions, 0, 0)
    }
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
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, Field, TwoAdicField,
    };
    use p3_koala_bear::KoalaBear;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type F = KoalaBear;
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

    fn beta_powers(air: &Poseidon2PermutePolyAir<F>, beta: EF) -> Vec<EF> {
        let max = <Poseidon2PermutePolyAir<F> as FullAir<
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

    fn trim_rows<T: Clone + Send + Sync>(
        matrix: &RowMajorMatrix<T>,
        num_rows: usize,
    ) -> RowMajorMatrix<T> {
        let width = matrix.width();
        RowMajorMatrix::new(matrix.values[..num_rows * width].to_vec(), width)
    }

    fn reserved_poly_matrix(
        air: &Poseidon2PermutePolyAir<F>,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly = <Poseidon2PermutePolyAir<F> as FullAir<
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

    fn sample_trace() -> Option<RowMajorMatrix<F>> {
        let program = poseidon2_permute_program();
        let mut runtime = Executor::new(program, DTCoreOpts::default());
        runtime.run().unwrap();

        for record in &runtime.records {
            let shard = record.as_ref();
            if shard.get_precompile_events(SyscallCode::POSEIDON2_PERMUTE).is_empty() {
                continue;
            }

            let mut poseidon_shard = ExecutionRecord::new(shard.program.clone());
            poseidon_shard.precompile_events = shard.precompile_events.clone();

            let chip = Poseidon2PermuteChip::<F>::default();
            return Some(
                chip.generate_trace(&poseidon_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    fn random_reducer(seed: u64) -> Vec<EF> {
        let mut rng = StdRng::seed_from_u64(seed);
        // Gate constraints: 551
        // Lookup batch: ceil(289/3) = 97
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 551;
        const NUM_REDUCER_CONSTRAINTS: usize =
            NUM_GATE_CONSTRAINTS + NUM_LOOKUPS.div_ceil(BATCH_SIZE) + 3;
        (0..NUM_REDUCER_CONSTRAINTS).map(|_| random_ef(&mut rng)).collect()
    }

    #[test]
    fn test_poseidon2_permute_polyair_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => {
                eprintln!("No Poseidon2Permute trace found -- skipping test");
                return;
            }
        };

        let air = Poseidon2PermutePolyAir::<F>::default();
        let height = main.height();
        // Use random challenges with fixed seeds for reproducibility
        let alpha_seed = 123u64;
        let beta_seed = 456u64;
        let reducer_seed = 789u64;

        let mut alpha_rng = StdRng::seed_from_u64(alpha_seed);
        let alpha = random_ef(&mut alpha_rng);
        let beta = challenge_beta_with_seed(beta_seed);
        let bp = beta_powers(&air, beta);
        let bs = beta_septix(beta);
        let public: Vec<F> = vec![];

        let precomputed_full = precompute_linear_combination(
            &air,
            None,
            &main,
            &public,
            alpha,
            &bp,
            bs,
            NUM_PRECOMPUTED,
        );
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
            &bp,
            bs,
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
                .take(10)
                .collect::<Vec<_>>()
        );

        let nonfirst = nonfirst_round_evaluation(
            &air,
            &public,
            &reserved,
            &precomputed,
            &permutation,
            alpha,
            &bp,
            bs,
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
                .take(10)
                .collect::<Vec<_>>()
        );
    }

    fn random_poseidon_permute_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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
        let air = Poseidon2PermutePolyAir::<F>::default();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::POSEIDON2_PERMUTE_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_poseidon_permute_trace(log_n, seed);
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
        let reserved_poly_desc = <Poseidon2PermutePolyAir<F> as FullAir<
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
impl<F: p3_field::Field> Poseidon2PermutePolyAir<F> {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_PRECOMPUTED
    }
}
