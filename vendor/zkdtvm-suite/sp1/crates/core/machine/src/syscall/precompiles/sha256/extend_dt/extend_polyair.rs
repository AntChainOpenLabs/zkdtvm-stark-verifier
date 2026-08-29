//! PolyAir adaptation of ShaExtendChip.

use std::ops::Deref;

use dt_stark::{
    air::{FullAir, FullAirBuilder, PairCol},
    InteractionKind, Word,
};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    memory::{
        polyair::{
            memory_read_lookup, memory_read_precompute_lc, memory_readwrite_lookup,
            memory_readwrite_precompute_lc, memory_timestamp_gate_constraints,
        },
        MemoryCols,
    },
    operations_dt::{
        add_n_without_result_gate_constraints, add_n_without_result_lookup,
        add_n_without_result_precompute_lc, fixed_rotate_right_gate_constraints,
        fixed_rotate_right_lookup, fixed_rotate_right_precompute_lc,
        fixed_shift_right_gate_constraints, fixed_shift_right_lookup,
        fixed_shift_right_precompute_lc, xor_n_lookup, xor_n_precompute_lc, CompactWord,
        CompactWordToWordWitness,
    },
    syscall::precompiles::sha256::extend_dt::{ShaExtendCols, NUM_SHA_EXTEND_COLS},
};

const NUM_LOOKUPS: usize = 61;
const MAX_LOOKUP_VALUES: usize = 7;

#[derive(Clone, Copy, Default)]
pub struct ShaExtendPolyAir;

impl ShaExtendPolyAir {
    pub const fn new() -> Self {
        Self
    }
}

fn compact_word<AB: FullAirBuilder>(word: &Word<AB::VarMaybeExt>) -> CompactWord<AB::VarMaybeExt> {
    let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
    CompactWord([
        word[0].clone() + word[1].clone() * base.clone(),
        word[2].clone() + word[3].clone() * base,
    ])
}

fn word_from_compact<AB: FullAirBuilder>(
    compact: CompactWord<AB::VarMaybeExt>,
    witness: CompactWordToWordWitness<AB::VarMaybeExt>,
) -> [AB::VarMaybeExt; 4] {
    let base = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << 8));
    let low0 = compact[0].clone() - witness[0].clone() * base.clone();
    let low1 = compact[1].clone() - witness[1].clone() * base;
    [low0, witness[0].clone(), low1, witness[1].clone()]
}

fn rotate_result_compact<AB: FullAirBuilder>(
    cols: &crate::operations_dt::FixedRotateRightOperation<AB::VarMaybeExt>,
    input: CompactWord<AB::VarMaybeExt>,
    rotation: usize,
) -> CompactWord<AB::VarMaybeExt> {
    let num_bytes_to_rotate = rotation / 16;
    let num_bits_to_rotate = rotation % 16;
    let split = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << num_bits_to_rotate));
    let glue = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << (16 - num_bits_to_rotate)));
    let lower_bits: [AB::VarMaybeExt; 2] = std::array::from_fn(|i| {
        input[(num_bytes_to_rotate + i) % 2].clone() - cols.higher_bits[i].clone() * split.clone()
    });
    CompactWord([
        cols.higher_bits[0].clone() + lower_bits[1].clone() * glue.clone(),
        cols.higher_bits[1].clone() + lower_bits[0].clone() * glue,
    ])
}

fn shift_result_compact<AB: FullAirBuilder>(
    cols: &crate::operations_dt::FixedShiftRightOperation<AB::VarMaybeExt>,
    input: CompactWord<AB::VarMaybeExt>,
    rotation: usize,
) -> CompactWord<AB::VarMaybeExt> {
    let num_bytes_to_rotate = rotation / 16;
    let num_bits_to_rotate = rotation % 16;
    let split = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << num_bits_to_rotate));
    let glue = AB::VarMaybeExt::from(AB::F::from_canonical_u32(1u32 << (16 - num_bits_to_rotate)));
    let lower_bits_1 = if num_bytes_to_rotate + 1 < 2 {
        input[num_bytes_to_rotate + 1].clone() - cols.higher_bits[1].clone() * split
    } else {
        AB::zero_maybe()
    };
    CompactWord([cols.higher_bits[0].clone() + lower_bits_1 * glue, cols.higher_bits[1].clone()])
}

impl<AB: FullAirBuilder> FullAir<AB> for ShaExtendPolyAir
where
    AB::VarMaybeExt: Clone,
{
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        MAX_LOOKUP_VALUES + 1
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        (0..NUM_SHA_EXTEND_COLS).map(PairCol::Main).collect()
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &ShaExtendCols<AB::VarMaybeExt> = unsafe { core::mem::transmute(main.as_ptr()) };

        let shard = local.shard.clone();
        let clk = local.clk.clone();
        let w_ptr = local.w_ptr.clone();
        let i = local.i.clone();
        let i_start = AB::VarMaybeExt::from(AB::F::from_canonical_u32(16));
        let num_bytes_in_word =
            AB::VarMaybeExt::from(AB::F::from_canonical_u32(core::mem::size_of::<u32>() as u32));

        let sha_extend_kind =
            AB::VarMaybeExt::from(AB::F::from_canonical_usize(InteractionKind::ShaExtend as usize));

        builder.retain_precomputed(builder.lookup_denominator(
            sha_extend_kind.clone(),
            vec![shard.clone(), clk.clone(), w_ptr.clone(), i.clone()],
        ));
        builder.retain_precomputed(builder.lookup_denominator(
            sha_extend_kind,
            vec![
                shard.clone(),
                clk.clone(),
                w_ptr.clone(),
                i.clone() + AB::VarMaybeExt::from(AB::F::one()),
            ],
        ));

        let access_clk = clk + (i.clone() - i_start);
        let w_i_minus_15_addr = w_ptr.clone() +
            (i.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_u32(15))) *
                num_bytes_in_word.clone();
        let w_i_minus_2_addr = w_ptr.clone() +
            (i.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_u32(2))) *
                num_bytes_in_word.clone();
        let w_i_minus_16_addr = w_ptr.clone() +
            (i.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_u32(16))) *
                num_bytes_in_word.clone();
        let w_i_minus_7_addr = w_ptr.clone() +
            (i.clone() - AB::VarMaybeExt::from(AB::F::from_canonical_u32(7))) *
                num_bytes_in_word.clone();
        let w_i_addr = w_ptr + i * num_bytes_in_word;

        memory_read_precompute_lc(
            builder,
            &local.w_i_minus_15.access,
            w_i_minus_15_addr,
            shard.clone(),
            access_clk.clone(),
        );
        memory_read_precompute_lc(
            builder,
            &local.w_i_minus_2.access,
            w_i_minus_2_addr,
            shard.clone(),
            access_clk.clone(),
        );
        memory_read_precompute_lc(
            builder,
            &local.w_i_minus_16.access,
            w_i_minus_16_addr,
            shard.clone(),
            access_clk.clone(),
        );
        memory_read_precompute_lc(
            builder,
            &local.w_i_minus_7.access,
            w_i_minus_7_addr,
            shard.clone(),
            access_clk.clone(),
        );
        memory_readwrite_precompute_lc(
            builder,
            &local.w_i.access,
            &local.w_i.prev_value,
            w_i_addr,
            shard,
            access_clk,
        );

        let w_i_minus_15 = compact_word::<AB>(local.w_i_minus_15.value());
        let w_i_minus_2 = compact_word::<AB>(local.w_i_minus_2.value());
        let w_i_minus_16 = compact_word::<AB>(local.w_i_minus_16.value());
        let w_i_minus_7 = compact_word::<AB>(local.w_i_minus_7.value());
        let w_i = compact_word::<AB>(local.w_i.value());

        fixed_rotate_right_precompute_lc(
            builder,
            &local.w_i_minus_15_rr_7,
            w_i_minus_15.clone(),
            7,
        );
        fixed_rotate_right_precompute_lc(
            builder,
            &local.w_i_minus_15_rr_18,
            w_i_minus_15.clone(),
            18,
        );
        fixed_shift_right_precompute_lc(builder, &local.w_i_minus_15_rs_3, w_i_minus_15.clone(), 3);

        let s0_inputs = [
            word_from_compact::<AB>(
                rotate_result_compact::<AB>(&local.w_i_minus_15_rr_7, w_i_minus_15.clone(), 7),
                local.w_i_minus_15_rr_7_witness.clone(),
            ),
            word_from_compact::<AB>(
                rotate_result_compact::<AB>(&local.w_i_minus_15_rr_18, w_i_minus_15.clone(), 18),
                local.w_i_minus_15_rr_18_witness.clone(),
            ),
            word_from_compact::<AB>(
                shift_result_compact::<AB>(&local.w_i_minus_15_rs_3, w_i_minus_15, 3),
                local.w_i_minus_15_rs_3_witness.clone(),
            ),
        ];
        let s0_results = [local.s0.value[0].0.clone(), local.s0.value[1].0.clone()];
        let s0_acc = [s0_inputs[0].clone(), s0_results[0].clone()];
        let s0_rhs = [s0_inputs[1].clone(), s0_inputs[2].clone()];
        xor_n_precompute_lc(builder, &s0_results, &s0_acc, &s0_rhs);

        fixed_rotate_right_precompute_lc(
            builder,
            &local.w_i_minus_2_rr_17,
            w_i_minus_2.clone(),
            17,
        );
        fixed_rotate_right_precompute_lc(
            builder,
            &local.w_i_minus_2_rr_19,
            w_i_minus_2.clone(),
            19,
        );
        fixed_shift_right_precompute_lc(builder, &local.w_i_minus_2_rs_10, w_i_minus_2.clone(), 10);

        let s1_inputs = [
            word_from_compact::<AB>(
                rotate_result_compact::<AB>(&local.w_i_minus_2_rr_17, w_i_minus_2.clone(), 17),
                local.w_i_minus_2_rr_17_witness.clone(),
            ),
            word_from_compact::<AB>(
                rotate_result_compact::<AB>(&local.w_i_minus_2_rr_19, w_i_minus_2.clone(), 19),
                local.w_i_minus_2_rr_19_witness.clone(),
            ),
            word_from_compact::<AB>(
                shift_result_compact::<AB>(&local.w_i_minus_2_rs_10, w_i_minus_2, 10),
                local.w_i_minus_2_rs_10_witness.clone(),
            ),
        ];
        let s1_results = [local.s1.value[0].0.clone(), local.s1.value[1].0.clone()];
        let s1_acc = [s1_inputs[0].clone(), s1_results[0].clone()];
        let s1_rhs = [s1_inputs[1].clone(), s1_inputs[2].clone()];
        xor_n_precompute_lc(builder, &s1_results, &s1_acc, &s1_rhs);

        let s0 = compact_word::<AB>(&local.s0.value[1]);
        let s1 = compact_word::<AB>(&local.s1.value[1]);
        let add_inputs = [w_i_minus_16, s0, w_i_minus_7, s1];
        add_n_without_result_precompute_lc(builder, &add_inputs, w_i);
    }

    fn eval(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &ShaExtendCols<AB::VarMaybeExt> =
            unsafe { &*(local_binding.deref().as_ptr() as *const ShaExtendCols<AB::VarMaybeExt>) };

        let is_real = local.is_real.clone();
        builder.assert_zero(is_real.clone() * (AB::one_maybe() - is_real.clone()));

        let i_start = AB::VarMaybeExt::from(AB::F::from_canonical_u32(16));
        let access_clk = local.clk.clone() + (local.i.clone() - i_start);

        memory_timestamp_gate_constraints(
            builder,
            &local.w_i_minus_15.access,
            local.shard.clone(),
            access_clk.clone(),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.w_i_minus_2.access,
            local.shard.clone(),
            access_clk.clone(),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.w_i_minus_16.access,
            local.shard.clone(),
            access_clk.clone(),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.w_i_minus_7.access,
            local.shard.clone(),
            access_clk.clone(),
            is_real.clone(),
        );
        memory_timestamp_gate_constraints(
            builder,
            &local.w_i.access,
            local.shard.clone(),
            access_clk,
            is_real.clone(),
        );

        let w_i_minus_15 = compact_word::<AB>(local.w_i_minus_15.value());
        let w_i_minus_2 = compact_word::<AB>(local.w_i_minus_2.value());
        let w_i_minus_16 = compact_word::<AB>(local.w_i_minus_16.value());
        let w_i_minus_7 = compact_word::<AB>(local.w_i_minus_7.value());
        let w_i = compact_word::<AB>(local.w_i.value());

        fixed_rotate_right_gate_constraints(
            builder,
            &local.w_i_minus_15_rr_7,
            w_i_minus_15.clone(),
            7,
        );
        fixed_rotate_right_gate_constraints(
            builder,
            &local.w_i_minus_15_rr_18,
            w_i_minus_15.clone(),
            18,
        );
        fixed_shift_right_gate_constraints(builder, &local.w_i_minus_15_rs_3, w_i_minus_15, 3);
        fixed_rotate_right_gate_constraints(
            builder,
            &local.w_i_minus_2_rr_17,
            w_i_minus_2.clone(),
            17,
        );
        fixed_rotate_right_gate_constraints(
            builder,
            &local.w_i_minus_2_rr_19,
            w_i_minus_2.clone(),
            19,
        );
        fixed_shift_right_gate_constraints(builder, &local.w_i_minus_2_rs_10, w_i_minus_2, 10);

        let s0 = compact_word::<AB>(&local.s0.value[1]);
        let s1 = compact_word::<AB>(&local.s1.value[1]);
        let add_inputs = [w_i_minus_16, s0, w_i_minus_7, s1];
        add_n_without_result_gate_constraints(builder, &add_inputs, w_i, is_real);
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved_poly = builder.reserved_poly();
        let local_binding = reserved_poly.row_slice(0);
        let local: &ShaExtendCols<AB::VarMaybeExt> =
            unsafe { &*(local_binding.deref().as_ptr() as *const ShaExtendCols<AB::VarMaybeExt>) };

        let is_real = local.is_real.clone();

        builder.recv(is_real.clone());
        builder.send(is_real.clone());

        memory_read_lookup(builder, is_real.clone());
        memory_read_lookup(builder, is_real.clone());
        memory_read_lookup(builder, is_real.clone());
        memory_read_lookup(builder, is_real.clone());
        memory_readwrite_lookup(builder, is_real.clone());

        fixed_rotate_right_lookup(builder, is_real.clone(), 7);
        fixed_rotate_right_lookup(builder, is_real.clone(), 18);
        fixed_shift_right_lookup(builder, is_real.clone(), 3);
        xor_n_lookup(builder, is_real.clone(), 3);

        fixed_rotate_right_lookup(builder, is_real.clone(), 17);
        fixed_rotate_right_lookup(builder, is_real.clone(), 19);
        fixed_shift_right_lookup(builder, is_real.clone(), 10);
        xor_n_lookup(builder, is_real.clone(), 3);

        add_n_without_result_lookup(builder, is_real, 4);
    }
}

// ============================================================================
// MachineAir delegation
// ============================================================================

use crate::syscall::precompiles::sha256::ShaExtendChip;
use dt_core_executor::{ExecutionRecord, Program};
use dt_stark::{air::MachineAir, sumcheck::trace::CompressedMatrix};
use p3_air::BaseAir;
use p3_field::Field;

impl<F: Field> BaseAir<F> for ShaExtendPolyAir {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_COLS
    }
}

impl<F: Field> MachineAir<F> for ShaExtendPolyAir {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        <ShaExtendChip as MachineAir<F>>::name(&ShaExtendChip::new()) + "PolyAir"
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        ShaExtendChip::new().generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        <ShaExtendChip as MachineAir<F>>::generate_dependencies(
            &ShaExtendChip::new(),
            input,
            output,
        )
    }

    fn included(&self, shard: &Self::Record) -> bool {
        <ShaExtendChip as MachineAir<F>>::included(&ShaExtendChip::new(), shard)
    }
}

#[cfg(test)]
mod tests {
    use crate::{programs::tests::sha_extend_program, syscall::precompiles::sha256::ShaExtendChip};
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

    fn beta_powers(air: &ShaExtendPolyAir, beta: EF) -> Vec<EF> {
        let required_max_beta_power = <ShaExtendPolyAir as FullAir<
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
        // Gate constraints: 18
        // Lookup batch: ceil(61/3) = 21
        // Cumulative sum: 3
        const NUM_GATE_CONSTRAINTS: usize = 18;
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
        air: &ShaExtendPolyAir,
        main: &RowMajorMatrix<F>,
    ) -> RowMajorMatrix<EF> {
        let reserved_poly =
            <ShaExtendPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(air);
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

            let chip = ShaExtendChip::new();
            return Some(
                chip.generate_trace(&sub_shard, &mut ExecutionRecord::default()).decompress(),
            );
        }
        None
    }

    #[test]
    fn test_sha_extend_constraint_check() {
        let main = match sample_trace() {
            Some(trace) => trace,
            None => return,
        };

        let air = ShaExtendPolyAir::new();
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
        assert!(first.iter().all(|x| x.is_zero()), "ShaExtend first_round failed: {:?}", first);

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
            "ShaExtend nonfirst_round failed: {:?}",
            nonfirst
        );
    }

    fn random_sha_extend_trace(log_n: usize, _seed: u64) -> RowMajorMatrix<F> {
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
        let air = ShaExtendPolyAir::new();
        let log_n = std::env::var("POLYAIR_PERF_LOG_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::syscall::precompiles::perf_test_defaults::SHA_EXTEND_LOG_N);
        let seed = std::env::var("POLYAIR_PERF_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let main = random_sha_extend_trace(log_n, seed);
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
        let reserved_poly_desc =
            <ShaExtendPolyAir as FullAir<PrecomputeRowBuilder<'_, F, F, EF>>>::reserved_poly(&air);

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
impl ShaExtendPolyAir {
    pub const fn num_lookups(&self) -> usize {
        NUM_LOOKUPS
    }
    pub const fn num_precomputed(&self) -> usize {
        NUM_LOOKUPS
    }
}
