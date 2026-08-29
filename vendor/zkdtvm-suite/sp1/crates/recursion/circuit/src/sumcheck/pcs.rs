use crate::challenger::{CanObserveVariable, CanSampleBitsVariable, FieldChallengerVariable};

use crate::{
    hash::FieldHasherVariable,
    sumcheck::{
        types::{
            BasefoldProofVariable, PrunedFriQueryProofVariable, WhirPrunedIoppRoundVariable,
            WhirRoundQueryProofVariable,
        },
        utils::Utils,
        SCBabyBearFriConfig, SCBabyBearFriConfigVariable,
    },
    BatchOpeningVariable, CircuitConfig, FriCommitPhaseProofStepVariable, FriQueryProofVariable,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, DslIr, Ext, Felt},
};
use itertools::{izip, Itertools};
use p3_commit::ExtensionMmcs;
use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};
use p3_fri::FriConfig;
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
    iter::{once, zip},
    marker::PhantomData,
};

pub type EF<C> = Ext<<C as Config>::F, <C as Config>::EF>;
pub type F<C> = Felt<<C as Config>::F>;
pub type SCFriMmcs<C> =
    ExtensionMmcs<dt_stark::Val<C>, dt_stark::Challenge<C>, <C as SCBabyBearFriConfig>::ValMmcs>;

fn compute_uniform_log_foldings_circuit(active_rounds: usize, num_groups: usize) -> Vec<usize> {
    if num_groups == 0 || active_rounds == 0 {
        return Vec::new();
    }
    let base = active_rounds / num_groups;
    let remainder = active_rounds % num_groups;
    (0..num_groups).map(|i| if i < remainder { base + 1 } else { base }).collect()
}

fn basefold_commit_schedule(
    num_vars: usize,
    k: usize,
    log_foldings: &[usize],
) -> Vec<(usize, usize)> {
    let mut remaining = num_vars.saturating_sub(k);
    let mut start_round = num_vars;
    let mut schedule = Vec::new();

    for &requested in log_foldings {
        if remaining == 0 {
            break;
        }
        if requested == 0 {
            continue;
        }
        let log_folding = requested.min(remaining);
        schedule.push((start_round, log_folding));
        start_round -= log_folding;
        remaining -= log_folding;
    }

    while remaining > 0 {
        schedule.push((start_round, 1));
        start_round -= 1;
        remaining -= 1;
    }
    schedule
}

fn whir_reduced_rate_commit_schedule(
    num_vars: usize,
    k: usize,
    initial_log_blowup: usize,
    log_foldings: &[usize],
) -> Vec<(usize, usize, usize, usize)> {
    let initial_codeword_log = num_vars + initial_log_blowup;
    basefold_commit_schedule(num_vars, k, log_foldings)
        .into_iter()
        .enumerate()
        .map(|(round_idx, (start_round, log_folding))| {
            let codeword_log = initial_codeword_log - round_idx;
            assert!(codeword_log >= start_round);
            (start_round, log_folding, codeword_log, codeword_log - start_round)
        })
        .collect()
}

fn basefold_log_foldings_from_query_shape<C, H>(
    num_vars: usize,
    k: usize,
    iopp_oracles_len: usize,
    iopp_queries: &[FriQueryProofVariable<C, H>],
    iopp_pruned: Option<&PrunedFriQueryProofVariable<C, H>>,
) -> Vec<usize>
where
    C: CircuitConfig,
    H: FieldHasherVariable<C>,
{
    let committed_groups = iopp_oracles_len.saturating_sub(usize::from(k == 0));
    if iopp_queries.is_empty() {
        if let Some(pruned) = iopp_pruned {
            if !pruned.round_opened_values.is_empty() {
                let mut total_log_folding = 0usize;
                let log_foldings = (0..committed_groups)
                    .map(|round| {
                        let row_width = pruned
                            .round_opened_values
                            .get(round)
                            .and_then(|rows| rows.first())
                            .and_then(|slot| slot.first())
                            .map(|row| if row.is_empty() { 2 } else { row.len() })
                            .unwrap_or(2);
                        assert!(row_width.is_power_of_two());
                        let log_folding = log2_strict_usize(row_width);
                        total_log_folding += log_folding;
                        log_folding
                    })
                    .collect::<Vec<_>>();
                assert_eq!(total_log_folding, num_vars.saturating_sub(k));
                return log_foldings;
            }
        }
        return vec![1; committed_groups];
    }

    let openings = &iopp_queries[0].commit_phase_openings;
    assert!(openings.len() >= committed_groups);

    let mut total_log_folding = 0usize;
    let log_foldings = openings
        .iter()
        .take(committed_groups)
        .map(|opening| {
            let row_width =
                if opening.leaf_values.is_empty() { 2 } else { opening.leaf_values.len() };
            assert!(row_width.is_power_of_two());
            let log_folding = log2_strict_usize(row_width);
            total_log_folding += log_folding;
            log_folding
        })
        .collect::<Vec<_>>();

    assert_eq!(total_log_folding, num_vars.saturating_sub(k));
    log_foldings
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StackedSource {
    matrix_idx: usize,
    base_col: usize,
    stacked_col: usize,
    slot: usize,
    selector_bits: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StackedBatchLayout {
    log_height: usize,
    width: usize,
    sources: Vec<StackedSource>,
}

#[derive(Clone, Copy)]
struct SlotPattern {
    depth: usize,
    slot: usize,
}

#[derive(Clone)]
struct StackedColumnAllocator {
    allocations: Vec<SlotPattern>,
}

impl SlotPattern {
    fn overlaps(self, other: Self) -> bool {
        if self.depth <= other.depth {
            let mask = (1usize << self.depth) - 1;
            (other.slot & mask) == self.slot
        } else {
            let mask = (1usize << other.depth) - 1;
            (self.slot & mask) == other.slot
        }
    }
}

impl StackedColumnAllocator {
    fn new() -> Self {
        Self { allocations: Vec::new() }
    }
}

fn can_place_pattern(
    allocators: &[StackedColumnAllocator],
    col: usize,
    pattern: SlotPattern,
) -> bool {
    allocators.get(col).map_or(true, |allocator| {
        allocator.allocations.iter().all(|&existing| !existing.overlaps(pattern))
    })
}

fn reserve_pattern(allocators: &mut Vec<StackedColumnAllocator>, col: usize, pattern: SlotPattern) {
    while allocators.len() <= col {
        allocators.push(StackedColumnAllocator::new());
    }
    allocators[col].allocations.push(pattern);
}

fn can_place_column_group(
    allocators: &[StackedColumnAllocator],
    start_col: usize,
    width: usize,
    pattern: SlotPattern,
) -> bool {
    (0..width).all(|offset| can_place_pattern(allocators, start_col + offset, pattern))
}

fn reserve_column_group(
    allocators: &mut Vec<StackedColumnAllocator>,
    start_col: usize,
    width: usize,
    pattern: SlotPattern,
) {
    for col in start_col..start_col + width {
        reserve_pattern(allocators, col, pattern);
    }
}

fn place_column(
    allocators: &mut Vec<StackedColumnAllocator>,
    selector_bits: usize,
) -> Option<(usize, usize)> {
    let num_slots = 1usize.checked_shl(selector_bits as u32)?;
    let max_col = allocators.len();
    for col in 0..=max_col {
        for slot in 0..num_slots {
            let pattern = SlotPattern { depth: selector_bits, slot };
            if can_place_pattern(allocators, col, pattern) {
                reserve_pattern(allocators, col, pattern);
                return Some((col, slot));
            }
        }
    }
    None
}

fn place_column_group(
    allocators: &mut Vec<StackedColumnAllocator>,
    width: usize,
    selector_bits: usize,
    column_alignment: usize,
) -> Option<(usize, usize)> {
    let num_slots = 1usize.checked_shl(selector_bits as u32)?;
    let max_start = allocators.len() + column_alignment.saturating_sub(1);
    for start_col in 0..=max_start {
        if start_col % column_alignment != 0 {
            continue;
        }
        for slot in 0..num_slots {
            let pattern = SlotPattern { depth: selector_bits, slot };
            if can_place_column_group(allocators, start_col, width, pattern) {
                reserve_column_group(allocators, start_col, width, pattern);
                return Some((start_col, slot));
            }
        }
    }
    None
}

impl StackedBatchLayout {
    fn from_dimensions(
        dimensions: &[Dimensions],
        log_height: usize,
        column_alignment: usize,
    ) -> Option<Self> {
        let column_alignment = column_alignment.max(1);
        let mut items = Vec::new();
        for (matrix_idx, dim) in dimensions.iter().enumerate() {
            if dim.width == 0 {
                continue;
            }
            if dim.height == 0 || !dim.height.is_power_of_two() {
                return None;
            }
            let log_matrix_height = log2_strict_usize(dim.height);
            if log_matrix_height > log_height {
                return None;
            }
            let selector_bits = log_height - log_matrix_height;
            let mut base_col = 0;
            while base_col < dim.width {
                let remaining = dim.width - base_col;
                let group_width = if column_alignment > 1 && remaining >= column_alignment {
                    column_alignment
                } else {
                    1
                };
                items.push((selector_bits, matrix_idx, base_col, group_width));
                base_col += group_width;
            }
        }
        items.sort_by_key(|&(selector_bits, matrix_idx, base_col, _)| {
            (selector_bits, matrix_idx, base_col)
        });

        let mut allocators = Vec::new();
        let mut sources = Vec::new();

        for (selector_bits, matrix_idx, base_col, group_width) in items {
            if group_width == 1 {
                let (stacked_col, slot) = place_column(&mut allocators, selector_bits)?;
                sources.push(StackedSource {
                    matrix_idx,
                    base_col,
                    stacked_col,
                    slot,
                    selector_bits,
                });
            } else {
                let (start_col, slot) = place_column_group(
                    &mut allocators,
                    group_width,
                    selector_bits,
                    column_alignment,
                )?;
                for offset in 0..group_width {
                    sources.push(StackedSource {
                        matrix_idx,
                        base_col: base_col + offset,
                        stacked_col: start_col + offset,
                        slot,
                        selector_bits,
                    });
                }
            }
        }
        if sources.is_empty() {
            return None;
        }
        Some(Self { log_height, width: allocators.len(), sources })
    }
}

#[derive(Clone)]
struct StackedBatchCoefficients<C: CircuitConfig> {
    column_coeffs: Vec<EF<C>>,
    chunk_coeffs: Vec<EF<C>>,
}

pub struct PcsVerifyTools<C, SC> {
    _marker: PhantomData<(C, SC)>,
}

impl<C: CircuitConfig<F = SC::Val>, SC: SCBabyBearFriConfigVariable<C>> PcsVerifyTools<C, SC>
where
    Builder<C>: CircuitV2Builder<C>,
{
    fn selector_eq(builder: &mut Builder<C>, point: &[EF<C>], source: &StackedSource) -> EF<C> {
        let one: EF<C> = builder.constant(C::EF::one());
        if source.selector_bits == 0 {
            return one;
        }
        let start = point.len() - source.selector_bits;
        let mut acc = one;
        for i in 0..source.selector_bits {
            let bit = (source.slot >> (source.selector_bits - 1 - i)) & 1;
            let factor =
                if bit == 0 { builder.eval(one - point[start + i]) } else { point[start + i] };
            acc = builder.eval(acc * factor);
        }
        acc
    }

    fn batch_uses_flattened_ext_dims(
        dimensions: &[Dimensions],
        opened_values: &[Vec<EF<C>>],
    ) -> bool {
        dimensions
            .iter()
            .zip(opened_values.iter())
            .any(|(dim, values)| values.len() != dim.width && values.len() * C::EF::D == dim.width)
    }

    fn stacked_batch_coefficients(
        builder: &mut Builder<C>,
        layout: &StackedBatchLayout,
        uses_flattened_ext: bool,
        alpha: EF<C>,
        alpha_powers: &mut EF<C>,
    ) -> StackedBatchCoefficients<C> {
        if uses_flattened_ext {
            let num_chunks = layout.width.div_ceil(C::EF::D);
            let mut chunk_coeffs = Vec::with_capacity(num_chunks);
            for _ in 0..num_chunks {
                chunk_coeffs.push(*alpha_powers);
                *alpha_powers = builder.eval(*alpha_powers * alpha);
            }

            let column_coeffs = (0..layout.width)
                .map(|col| {
                    let chunk = col / C::EF::D;
                    let limb = col % C::EF::D;
                    let basis: EF<C> = builder.constant(C::EF::from_base_fn(|i| {
                        if i == limb {
                            C::F::one()
                        } else {
                            C::F::zero()
                        }
                    }));
                    builder.eval(chunk_coeffs[chunk] * basis)
                })
                .collect();
            StackedBatchCoefficients { column_coeffs, chunk_coeffs }
        } else {
            let mut column_coeffs = Vec::with_capacity(layout.width);
            for _ in 0..layout.width {
                column_coeffs.push(*alpha_powers);
                *alpha_powers = builder.eval(*alpha_powers * alpha);
            }
            StackedBatchCoefficients { chunk_coeffs: column_coeffs.clone(), column_coeffs }
        }
    }

    fn stacked_batch_claim_from_dims(
        builder: &mut Builder<C>,
        dimensions: &[Dimensions],
        opened_values: &[Vec<EF<C>>],
        layout: &StackedBatchLayout,
        coeffs: &StackedBatchCoefficients<C>,
        full_opening_point: &[EF<C>],
        uses_flattened_ext: bool,
    ) -> EF<C> {
        let mut claim: EF<C> = builder.constant(C::EF::zero());
        let mut selector_eqs = HashMap::new();

        for source in &layout.sources {
            let selector = *selector_eqs
                .entry((source.selector_bits, source.slot))
                .or_insert_with(|| Self::selector_eq(builder, full_opening_point, source));
            let dim = dimensions.get(source.matrix_idx).expect("missing stacked matrix dims");
            let values =
                opened_values.get(source.matrix_idx).expect("missing stacked opened values");

            if values.len() == dim.width {
                let value = values.get(source.base_col).expect("missing stacked opened value");
                let term: EF<C> =
                    builder.eval(selector * coeffs.column_coeffs[source.stacked_col] * *value);
                claim = builder.eval(claim + term);
            } else if values.len() * C::EF::D == dim.width && uses_flattened_ext {
                if source.base_col % C::EF::D != 0 {
                    continue;
                }
                assert_eq!(source.stacked_col % C::EF::D, 0);
                let ext_col = source.base_col / C::EF::D;
                let chunk = source.stacked_col / C::EF::D;
                let value = values.get(ext_col).expect("missing stacked opened ext value");
                let term: EF<C> = builder.eval(selector * coeffs.chunk_coeffs[chunk] * *value);
                claim = builder.eval(claim + term);
            } else {
                panic!("invalid opened value width for stacked PCS");
            }
        }
        claim
    }

    /// Circuit equivalent of `reduction_target_for_batch`: T = Σ λ^i · original_claim_i.
    /// Returns (T, next_lambda_power).
    fn reduction_target_for_batch_circuit(
        builder: &mut Builder<C>,
        layout: &StackedBatchLayout,
        dimensions: &[Dimensions],
        opened_values: &[Vec<EF<C>>],
        lambda: EF<C>,
        lambda_power_start: EF<C>,
        uses_flattened_ext: bool,
    ) -> (EF<C>, EF<C>) {
        let mut t: EF<C> = builder.constant(C::EF::zero());
        let mut current_power = lambda_power_start;
        let mut lambdas_consumed = 0usize;
        let mut prev_logical: Option<(usize, usize, usize)> = None;

        for source in &layout.sources {
            let dim = &dimensions[source.matrix_idx];
            let values = &opened_values[source.matrix_idx];

            if uses_flattened_ext && values.len() * C::EF::D == dim.width {
                let logical_key = (source.matrix_idx, source.slot, source.base_col / C::EF::D);
                let is_new = prev_logical.map_or(true, |prev| prev != logical_key);
                if is_new {
                    if lambdas_consumed > 0 || prev_logical.is_some() {
                        current_power = builder.eval(current_power * lambda);
                    }
                    prev_logical = Some(logical_key);
                    lambdas_consumed += 1;
                }
                if source.base_col % C::EF::D != 0 {
                    continue;
                }
                let ext_col = source.base_col / C::EF::D;
                let value = values[ext_col];
                t = builder.eval(t + current_power * value);
            } else {
                if lambdas_consumed > 0 {
                    current_power = builder.eval(current_power * lambda);
                }
                let value = values[source.base_col];
                t = builder.eval(t + current_power * value);
                lambdas_consumed += 1;
            }
        }

        let next_power =
            if lambdas_consumed > 0 { builder.eval(current_power * lambda) } else { current_power };
        (t, next_power)
    }

    /// Circuit equivalent of `compute_q_at_point_for_batch`: Q_c(u) for each stacked column.
    /// Returns (q_values, next_lambda_power).
    fn compute_q_at_point_circuit(
        builder: &mut Builder<C>,
        layout: &StackedBatchLayout,
        full_opening_point: &[EF<C>],
        u: &[EF<C>],
        lambda: EF<C>,
        lambda_power_start: EF<C>,
        uses_flattened_ext: bool,
    ) -> (Vec<EF<C>>, EF<C>) {
        let l = layout.log_height;
        let mut q_values: Vec<EF<C>> =
            (0..layout.width).map(|_| builder.constant(C::EF::zero())).collect();

        let mut current_power = lambda_power_start;
        let mut lambdas_consumed = 0usize;
        let mut prev_logical: Option<(usize, usize, usize)> = None;

        let mut selector_cache: HashMap<(usize, usize), EF<C>> = HashMap::new();
        let mut eq_prefix_cache: HashMap<usize, EF<C>> = HashMap::new();

        for source in &layout.sources {
            let b = source.selector_bits;

            let selector_val = *selector_cache.entry((b, source.slot)).or_insert_with(|| {
                if b == 0 {
                    builder.constant(C::EF::one())
                } else {
                    let start = l - b;
                    let one: EF<C> = builder.constant(C::EF::one());
                    let mut acc = one;
                    for i in 0..b {
                        let bit = (source.slot >> (b - 1 - i)) & 1;
                        let factor =
                            if bit == 0 { builder.eval(one - u[start + i]) } else { u[start + i] };
                        acc = builder.eval(acc * factor);
                    }
                    acc
                }
            });

            let eq_prefix_val = *eq_prefix_cache.entry(b).or_insert_with(|| {
                let prefix_len = l - b;
                if prefix_len == 0 {
                    builder.constant(C::EF::one())
                } else {
                    Utils::<C, SC>::compute_eq(
                        builder,
                        &full_opening_point[0..prefix_len],
                        &u[0..prefix_len],
                    )
                }
            });

            if uses_flattened_ext {
                let logical_key = (source.matrix_idx, source.slot, source.base_col / C::EF::D);
                let is_new = prev_logical.map_or(true, |prev| prev != logical_key);
                if is_new {
                    if lambdas_consumed > 0 || prev_logical.is_some() {
                        current_power = builder.eval(current_power * lambda);
                    }
                    prev_logical = Some(logical_key);
                    lambdas_consumed += 1;
                }
                let limb = source.base_col % C::EF::D;
                let basis: EF<C> = builder.constant(C::EF::from_base_fn(|i| {
                    if i == limb {
                        C::F::one()
                    } else {
                        C::F::zero()
                    }
                }));
                let coeff: EF<C> =
                    builder.eval(current_power * basis * selector_val * eq_prefix_val);
                q_values[source.stacked_col] = builder.eval(q_values[source.stacked_col] + coeff);
            } else {
                if lambdas_consumed > 0 {
                    current_power = builder.eval(current_power * lambda);
                }
                let coeff: EF<C> = builder.eval(current_power * selector_val * eq_prefix_val);
                q_values[source.stacked_col] = builder.eval(q_values[source.stacked_col] + coeff);
                lambdas_consumed += 1;
            }
        }

        let next_power =
            if lambdas_consumed > 0 { builder.eval(current_power * lambda) } else { current_power };
        (q_values, next_power)
    }

    pub fn verify_basefold_pcs(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commitment_batch: Vec<<SC as FieldHasherVariable<C>>::DigestVariable>,
        matrices_size_batch: &Vec<Vec<Dimensions>>,
        opening_point: &[EF<C>],
        opened_values_batch: &Vec<Vec<Vec<EF<C>>>>,
        proof: &BasefoldProofVariable<C, SC>,
        challenger: &mut SC::FriChallengerVariable,
    ) {
        assert!(matrices_size_batch.len() == opened_values_batch.len());
        assert!(commitment_batch.len() == matrices_size_batch.len());
        for (matrices_size, opened_values) in
            matrices_size_batch.iter().zip(opened_values_batch.iter())
        {
            assert!(matrices_size.len() == opened_values.len());
            for (dim, vals) in matrices_size.iter().zip(opened_values.iter()) {
                assert!(vals.len() == dim.width || vals.len() * C::EF::D == dim.width);
            }
        }

        let BasefoldProofVariable {
            stack_log_height: proof_stack_log_height,
            sumcheck_transcript,
            iopp_oracles,
            ood_values,
            iopp_queries,
            round_iopp,
            query_openings,
            input_pruned,
            grinding_batching_witness,
            grinding_query_witness,
            final_poly,
            iopp_pruned,
            stacking_reduction,
        } = proof;

        assert!(grinding_batching_witness.len() == 2);

        let stack_log_height = if round_iopp.is_some() {
            proof_stack_log_height.expect("WHIR proof must carry stack_log_height")
        } else {
            assert!(grinding_query_witness.len() == 2);
            sumcheck_transcript.uni_polys.len()
        };
        assert!(opening_point.len() <= stack_log_height);

        // Legacy non-stacking (Jagged) proofs carry `stack_log_height = None`.
        // They group matrices by height and merge height groups with
        // `merge_beta` instead of pre-stacking into a single matrix, so they
        // need a different verifier (and a different Fiat-Shamir transcript)
        // than the stacked paths. Dispatch BEFORE any stacked-path challenger
        // interaction so the transcript matches the native non-stacking verifier.
        if round_iopp.is_none() && proof_stack_log_height.is_none() {
            Self::verify_basefold_pcs_nonstacking(
                builder,
                config,
                commitment_batch,
                matrices_size_batch,
                opening_point,
                opened_values_batch,
                proof,
                challenger,
            );
            return;
        }

        let mut full_opening_point = opening_point.to_vec();
        while full_opening_point.len() < stack_log_height {
            full_opening_point.push(challenger.sample_ext(builder));
        }

        let layouts = matrices_size_batch
            .iter()
            .map(|dims| {
                StackedBatchLayout::from_dimensions(dims, stack_log_height, C::EF::D)
                    .expect("invalid stacked PCS layout")
            })
            .collect::<Vec<_>>();

        // --- Phase 1: Stacking reduction sumcheck ---
        // 1. Absorb opened_values into transcript
        for batch_values in opened_values_batch.iter() {
            for mat_values in batch_values.iter() {
                for v in mat_values.iter() {
                    let v_felts = C::ext2felt(builder, *v);
                    challenger.observe_slice(builder, v_felts);
                }
            }
        }

        // 2. Sample λ
        let lambda = challenger.sample_ext(builder);

        let flattened_flags: Vec<bool> = matrices_size_batch
            .iter()
            .zip(opened_values_batch.iter())
            .map(|(dims, values)| Self::batch_uses_flattened_ext_dims(dims, values))
            .collect();

        // 3. Compute T = Σ λ^i · original_claim_i
        let mut target: EF<C> = builder.constant(C::EF::zero());
        let mut lambda_power: EF<C> = builder.constant(C::EF::one());
        for ((dims, values), (layout, &uses_flat)) in matrices_size_batch
            .iter()
            .zip(opened_values_batch.iter())
            .zip(layouts.iter().zip(flattened_flags.iter()))
        {
            let (t_batch, next_power) = Self::reduction_target_for_batch_circuit(
                builder,
                layout,
                dims,
                values,
                lambda,
                lambda_power,
                uses_flat,
            );
            target = builder.eval(target + t_batch);
            lambda_power = next_power;
        }

        // 4. Verify reduction sumcheck
        let reduction = stacking_reduction.as_ref().expect("stacking_reduction proof required");
        assert_eq!(
            reduction.sumcheck.uni_polys.len(),
            stack_log_height,
            "reduction sumcheck round count mismatch"
        );
        let mut reduction_claim = target;
        let mut u: Vec<EF<C>> = Vec::with_capacity(stack_log_height);
        for uni_poly in &reduction.sumcheck.uni_polys {
            let f0 = uni_poly.eval_at_zero();
            let f1 = uni_poly.eval_at_one_horner(builder);
            let sum: EF<C> = builder.eval(f0 + f1);
            builder.assert_ext_eq(sum, reduction_claim);
            uni_poly.observe_into(builder, challenger);
            let r_j = challenger.sample_ext(builder);
            reduction_claim = uni_poly.evaluate_horner(builder, &r_j);
            u.push(r_j);
        }
        // Reverse u from LSB-first (binding order) to MSB-first (EqPolynomial convention)
        u.reverse();

        // 5. Compute Q_c(u) coefficients for each batch
        let mut coeffs_by_batch = Vec::with_capacity(matrices_size_batch.len());
        let mut current_claim = reduction_claim;
        lambda_power = builder.constant(C::EF::one());
        for (layout, &uses_flat) in layouts.iter().zip(flattened_flags.iter()) {
            let (q_values, next_power) = Self::compute_q_at_point_circuit(
                builder,
                layout,
                &full_opening_point,
                &u,
                lambda,
                lambda_power,
                uses_flat,
            );
            lambda_power = next_power;
            let chunk_coeffs = q_values.clone();
            coeffs_by_batch
                .push(StackedBatchCoefficients { column_coeffs: q_values, chunk_coeffs });
        }

        // --- Verify batching proof of work ---
        challenger.observe(builder, grinding_batching_witness[0]);
        challenger.check_witness(
            builder,
            config.grinding_bits_batching,
            grinding_batching_witness[1],
        );

        let k = config.log_final_poly_len.min(stack_log_height);
        if let Some(round_iopp) = round_iopp.as_ref() {
            Self::verify_whir_round_pcs(
                builder,
                config,
                commitment_batch,
                &layouts,
                &coeffs_by_batch,
                query_openings,
                input_pruned.as_ref(),
                iopp_oracles,
                ood_values,
                round_iopp,
                final_poly,
                sumcheck_transcript,
                stack_log_height,
                k,
                current_claim,
                &u,
                challenger,
            );
            return;
        }

        // --- Phase 2: Verify the stacked sumcheck rounds. ---
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        assert_eq!(final_poly.len(), if k == 0 { 0 } else { 1usize << k });
        let cross_round_log_foldings = basefold_log_foldings_from_query_shape(
            stack_log_height,
            k,
            iopp_oracles.len(),
            iopp_queries,
            iopp_pruned.as_ref(),
        );
        let default_commit_schedule = basefold_commit_schedule(stack_log_height, k, &[]);
        let cross_round_commit_schedule =
            basefold_commit_schedule(stack_log_height, k, &cross_round_log_foldings);
        let cross_round_expected_oracles = cross_round_commit_schedule.len() + usize::from(k == 0);
        let commit_schedule = if !cross_round_log_foldings.is_empty() &&
            iopp_oracles.len() == cross_round_expected_oracles
        {
            cross_round_commit_schedule
        } else {
            default_commit_schedule
        };
        assert_eq!(iopp_oracles.len(), commit_schedule.len() + usize::from(k == 0));
        let iopp_log_foldings =
            commit_schedule.iter().map(|(_, log_folding)| *log_folding).collect::<Vec<_>>();

        let mut folding_challenges: Vec<EF<C>> = Vec::with_capacity(stack_log_height);
        let mut oracle_idx = 0usize;
        let mut schedule_idx = 0usize;

        for round in (0..=stack_log_height).rev() {
            if schedule_idx < commit_schedule.len() && commit_schedule[schedule_idx].0 == round {
                challenger.observe(builder, iopp_oracles[oracle_idx].clone());
                oracle_idx += 1;
                schedule_idx += 1;
            } else if round == 0 && k == 0 {
                challenger.observe(builder, iopp_oracles[oracle_idx].clone());
                oracle_idx += 1;
            } else if round == k && k > 0 {
                for coeff in final_poly.iter() {
                    let coeff_felts = C::ext2felt(builder, *coeff);
                    challenger.observe_slice(builder, coeff_felts);
                }
            }
            if round == 0 {
                break;
            }

            let uni_poly = poly_iter.next().expect("not enough sumcheck unipolys");
            let eval_at_one = uni_poly.eval_at_one_horner(builder);
            builder.assert_ext_eq(uni_poly.eval_at_zero() + eval_at_one, current_claim);

            uni_poly.observe_into(builder, challenger);
            let r_fold = challenger.sample_ext(builder);
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate_horner(builder, &r_fold);
        }

        // --- Phase 3: Reconstruct combined EQ sum ---
        let fc_rev: Vec<EF<C>> = folding_challenges.iter().rev().cloned().collect();

        let combined_eq_sum = Utils::<C, SC>::compute_eq(builder, &u, &fc_rev);

        // --- Phase 4: Final codeword commitment check ---
        let combined_f_r: EF<C> = builder.eval(current_claim / combined_eq_sum);
        let final_codeword = if k > 0 {
            Some(Self::encode_final_poly_to_codeword(builder, config, final_poly))
        } else {
            None
        };

        // --- Phase 5: Query proof of work ---
        challenger.observe(builder, grinding_query_witness[0]);
        challenger.check_witness(builder, config.grinding_bits_query, grinding_query_witness[1]);

        // --- Phase 6: IOPP query verification ---
        let query_points: Vec<Vec<C::Bit>> = (0..config.num_queries)
            .map(|_| challenger.sample_bits(builder, stack_log_height + config.log_blowup))
            .collect();

        let stacked_height = (1usize << stack_log_height) << config.log_blowup;
        let stacked_heights = vec![stacked_height];
        let leaf_sum_key = stack_log_height + config.log_blowup;
        let empty_merge_betas: Vec<EF<C>> = Vec::new();
        let empty_branch_codewords: BTreeMap<usize, Vec<EF<C>>> = BTreeMap::new();

        // [SS] dispatch: pruned vs standard IOPP path
        if let Some(pruned_proof) = iopp_pruned.as_ref() {
            // === Pruned path (size-saving): ===
            // per_query is empty; data comes from input_pruned.
            let inp = input_pruned.as_ref().expect(
                "verify_basefold_pcs (pruned): input_pruned must be Some when iopp_pruned is Some",
            );
            let n_queries = query_points.len();

            // Step A: per-batch BFS merkle verification via verify_batch_pruned.
            // For each batch, compute leaf digests from the unique-slot
            // opened values, then verify all unique queries against the
            // batch commitment in one BFS-merged proof.
            for batch_idx in 0..layouts.len() {
                let commitment = &commitment_batch[batch_idx];

                // Compute leaf digest for each unique slot by replicating
                // the tallest-first hash logic from verify_batch.
                let unique_opened = &inp.round_opened_values[batch_idx];
                let num_unique = unique_opened.len();
                let mut leaf_digests: Vec<SC::DigestVariable> = Vec::with_capacity(num_unique);

                for slot in 0..num_unique {
                    let opened_values = &unique_opened[slot]; // [mat_idx][values]
                    let leaf_digest =
                        Self::compute_leaf_digest(builder, &stacked_heights, opened_values);
                    leaf_digests.push(leaf_digest);
                }

                // Matrix injection is unnecessary after stacking because the
                // input MMCS has a single stacked matrix. Avoid cloning all
                // opened rows just to pass an unused injection payload.
                let opened_per_query_refs: &[Vec<Vec<Felt<C::F>>>] =
                    if stacked_heights.len() <= 1 { &[] } else { unique_opened.as_slice() };

                Self::verify_batch_pruned(
                    builder,
                    *commitment,
                    leaf_sum_key,
                    leaf_digests,
                    &inp.round_pruned[batch_idx],
                    &stacked_heights,
                    &opened_per_query_refs,
                );
            }

            // Step B: per-query ro computation.
            // For each query, index into round_opened_values via q2u to
            // get per-batch per-matrix opened values, then compute the
            // linear combination (ro) per height group.
            let mut ros: Vec<BTreeMap<usize, EF<C>>> = Vec::with_capacity(n_queries);

            for query_idx in 0..n_queries {
                let mut ro: BTreeMap<usize, EF<C>> = BTreeMap::new();
                let mut reduce_sum: EF<C> = builder.constant(C::EF::zero());
                for (batch_idx, coeffs) in coeffs_by_batch.iter().enumerate() {
                    let unique_slot = inp.query_to_unique_slot[batch_idx][query_idx];
                    let opened_vals = &inp.round_opened_values[batch_idx][unique_slot][0];
                    let val = Utils::<C, SC>::compute_dotproduct_mix(
                        builder,
                        &coeffs.column_coeffs,
                        opened_vals,
                    );
                    reduce_sum = builder.eval(reduce_sum + val);
                }
                ro.insert(leaf_sum_key, reduce_sum);
                ros.push(ro);
            }

            // Step C: batched IOPP verification across all queries (unchanged).
            Self::verify_iopp_query_p3_pruned(
                builder,
                config,
                iopp_oracles.as_slice(),
                &query_points,
                ros,
                pruned_proof,
                &iopp_log_foldings,
                &folding_challenges,
                &empty_merge_betas,
                &u,
                &combined_f_r,
                final_codeword.as_deref(),
                &empty_branch_codewords,
            );
        } else {
            // === Standard path: per-query independent merkle proofs ===
            // [F-018] Pin the query counts so the zip cannot be silently
            // truncated (defensive; recursion shape fixed by dummy proof).
            assert_eq!(iopp_queries.len(), config.num_queries);
            assert_eq!(query_openings.len(), config.num_queries);
            iopp_queries.iter().zip(query_openings.iter()).enumerate().for_each(
                |(i, (query, leaf_opening))| {
                    for (batch_idx, opening) in leaf_opening.iter().enumerate() {
                        Self::verify_batch(
                            builder,
                            commitment_batch[batch_idx],
                            &stacked_heights,
                            &query_points[i],
                            opening.opened_values.clone(),
                            opening.opening_proof.clone(),
                        );
                    }

                    let mut leaf_sum: EF<C> = builder.constant(C::EF::zero());
                    for (coeffs, opening) in coeffs_by_batch.iter().zip(leaf_opening.iter()) {
                        let opened_vals: Vec<F<C>> =
                            opening.opened_values[0].clone().into_iter().flatten().collect();
                        let val = Utils::<C, SC>::compute_dotproduct_mix(
                            builder,
                            &coeffs.column_coeffs,
                            &opened_vals,
                        );
                        leaf_sum = builder.eval(leaf_sum + val);
                    }

                    let mut ro = BTreeMap::new();
                    ro.insert(leaf_sum_key, leaf_sum);

                    Self::verify_iopp_query_basefold(
                        builder,
                        config,
                        iopp_oracles.as_slice(),
                        &query_points[i],
                        ro,
                        query,
                        &iopp_log_foldings,
                        &folding_challenges,
                        &empty_merge_betas,
                        &u,
                        &combined_f_r,
                        final_codeword.as_deref(),
                        &empty_branch_codewords,
                    );
                },
            );
        }
    }

    /// Verify a legacy non-stacking (Jagged) Basefold proof in-circuit.
    ///
    /// Mirrors the native `WhirPcs::verify` (non-stacking path): matrices are
    /// grouped by log-height, height groups are merged with `merge_beta` during
    /// the sumcheck, and each query reconstructs a per-height `ro` map fed into
    /// the multi-height-capable IOPP query verifier. Non-stacking always runs
    /// with FRI early-stop disabled (`k = 0`), so there is no final polynomial.
    #[allow(clippy::too_many_arguments)]
    fn verify_basefold_pcs_nonstacking(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commitment_batch: Vec<<SC as FieldHasherVariable<C>>::DigestVariable>,
        matrices_size_batch: &Vec<Vec<Dimensions>>,
        opening_point: &[EF<C>],
        opened_values_batch: &Vec<Vec<Vec<EF<C>>>>,
        proof: &BasefoldProofVariable<C, SC>,
        challenger: &mut SC::FriChallengerVariable,
    ) {
        let num_vars = opening_point.len();
        let one: EF<C> = builder.constant(C::EF::one());

        // --- Group matrices by log-height (flattened global index). ---
        // `groups[log_height] = Vec<(batch_idx, mat_idx, num_coeffs)>` in the
        // same flattened order native uses to draw alpha powers.
        let mut flat_idx = 0usize;
        let mut groups: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
        for (batch_idx, (batch_dims, batch_vals)) in
            matrices_size_batch.iter().zip(opened_values_batch.iter()).enumerate()
        {
            for (mat_idx, (dim, vals)) in batch_dims.iter().zip(batch_vals.iter()).enumerate() {
                assert!(vals.len() == dim.width || vals.len() * C::EF::D == dim.width);
                let log_height = log2_strict_usize(dim.height);
                groups.entry(log_height).or_default().push((batch_idx, mat_idx, vals.len()));
                flat_idx += 1;
            }
        }
        let _ = flat_idx;
        let log_max_height = *groups.keys().max().expect("at least one matrix");
        let min_log_height = *groups.keys().min().expect("at least one matrix");
        assert_eq!(log_max_height, num_vars);

        // --- Phase 1: per-height coefficients + claimed sums (descending). ---
        let alpha = challenger.sample_ext(builder);
        let mut alpha_powers: EF<C> = one;
        // coeffs_by_height[log_height] = Vec<(batch_idx, mat_idx, coeffs)>.
        let mut coeffs_by_height: BTreeMap<usize, Vec<(usize, usize, Vec<EF<C>>)>> =
            BTreeMap::new();
        let mut claims_by_height: BTreeMap<usize, EF<C>> = BTreeMap::new();
        for (&log_height, members) in groups.iter().rev() {
            let mut height_coeffs = Vec::with_capacity(members.len());
            let mut claim: EF<C> = builder.constant(C::EF::zero());
            for &(batch_idx, mat_idx, num_coeffs) in members {
                let vals = &opened_values_batch[batch_idx][mat_idx];
                let mut coeffs = Vec::with_capacity(num_coeffs);
                for v in vals.iter() {
                    let c = alpha_powers;
                    alpha_powers = builder.eval(alpha_powers * alpha);
                    let term: EF<C> = builder.eval(c * *v);
                    claim = builder.eval(claim + term);
                    coeffs.push(c);
                }
                height_coeffs.push((batch_idx, mat_idx, coeffs));
            }
            coeffs_by_height.insert(log_height, height_coeffs);
            claims_by_height.insert(log_height, claim);
        }

        // --- Verify batching proof of work. ---
        assert!(proof.grinding_batching_witness.len() == 2);
        challenger.observe(builder, proof.grinding_batching_witness[0]);
        challenger.check_witness(
            builder,
            config.grinding_bits_batching,
            proof.grinding_batching_witness[1],
        );

        // Non-stacking always runs with early-stop disabled.
        let k = 0usize;
        let commit_schedule = basefold_commit_schedule(num_vars, k, &[]);
        let commit_start_rounds: std::collections::BTreeSet<usize> =
            commit_schedule.iter().map(|(start, _)| *start).collect();
        let expected_oracles = commit_schedule.len() + usize::from(k == 0);
        assert_eq!(proof.iopp_oracles.len(), expected_oracles);
        assert_eq!(proof.final_poly.len(), 0);

        // --- Phase 2: sumcheck rounds (fold + merge). ---
        let mut poly_iter = proof.sumcheck_transcript.uni_polys.iter();
        let mut current_claim = *claims_by_height.get(&num_vars).expect("missing top-height claim");

        challenger.observe(builder, proof.iopp_oracles[0].clone());

        let mut folding_challenges: Vec<EF<C>> = Vec::with_capacity(num_vars);
        let mut merge_betas: Vec<EF<C>> = Vec::new();
        let mut oracle_idx = 1usize;

        for round in (0..=num_vars).rev() {
            let should_observe_next_oracle = (round < num_vars &&
                commit_start_rounds.contains(&round)) ||
                (round == 0 && k == 0);
            if should_observe_next_oracle {
                if oracle_idx < proof.iopp_oracles.len() {
                    challenger.observe(builder, proof.iopp_oracles[oracle_idx].clone());
                    oracle_idx += 1;
                }
            }
            if round == 0 {
                break;
            }

            let uni_poly = poly_iter.next().expect("not enough sumcheck unipolys");
            let eval_at_one = uni_poly.eval_at_one_horner(builder);
            builder.assert_ext_eq(uni_poly.eval_at_zero() + eval_at_one, current_claim);
            uni_poly.observe_into(builder, challenger);
            let r_fold = challenger.sample_ext(builder);
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate_horner(builder, &r_fold);

            // WHIR merge when a height group ends at this boundary.
            if let Some(&branch_claim) = claims_by_height.get(&(round - 1)) {
                let merge_beta = challenger.sample_ext(builder);
                merge_betas.push(merge_beta);
                current_claim = builder.eval(current_claim + merge_beta * branch_claim);
            }
        }

        // --- Phase 3: combined EQ sum over the min-height suffix. ---
        let fc_rev: Vec<EF<C>> = folding_challenges.iter().rev().cloned().collect();
        let combined_eq_sum = Utils::<C, SC>::compute_eq(
            builder,
            &opening_point[..min_log_height],
            &fc_rev[..min_log_height],
        );

        // --- Phase 4: combined final value. ---
        // The IOPP query path folds each query down to a single value and
        // asserts it equals `combined_f_r` (the k == 0 constant codeword),
        // which is the final consistency check (mirrors the stacked path).
        let combined_f_r: EF<C> = builder.eval(current_claim / combined_eq_sum);

        // --- Phase 5: query proof of work. ---
        assert!(proof.grinding_query_witness.len() == 2);
        challenger.observe(builder, proof.grinding_query_witness[0]);
        challenger.check_witness(
            builder,
            config.grinding_bits_query,
            proof.grinding_query_witness[1],
        );

        // --- Phase 6: IOPP query verification. ---
        let query_points: Vec<Vec<C::Bit>> = (0..config.num_queries)
            .map(|_| challenger.sample_bits(builder, num_vars + config.log_blowup))
            .collect();

        // Per-batch max log-height (for index right-shift) and codeword heights.
        let batch_max_log_height: Vec<usize> = matrices_size_batch
            .iter()
            .map(|dims| dims.iter().map(|d| log2_strict_usize(d.height)).max().unwrap_or(0))
            .collect();

        let commit_log_foldings: Vec<usize> =
            commit_schedule.iter().map(|(_, log_folding)| *log_folding).collect();
        let final_codeword: Option<&[EF<C>]> = None;
        let empty_branch_codewords: BTreeMap<usize, Vec<EF<C>>> = BTreeMap::new();

        if let Some(pruned) = proof.input_pruned.as_ref() {
            // Path-pruned input openings: per-batch BFS merkle verify, then
            // reconstruct each query's per-height `ro` from the unique slots.
            let n_queries = query_points.len();
            assert_eq!(pruned.round_pruned.len(), matrices_size_batch.len());
            assert_eq!(pruned.round_opened_values.len(), matrices_size_batch.len());
            assert_eq!(pruned.query_to_unique_slot.len(), matrices_size_batch.len());

            for (batch_idx, batch_dims) in matrices_size_batch.iter().enumerate() {
                let codeword_heights: Vec<usize> =
                    batch_dims.iter().map(|d| d.height << config.log_blowup).collect();
                let unique_opened = &pruned.round_opened_values[batch_idx];
                let q2u = &pruned.query_to_unique_slot[batch_idx];
                assert_eq!(q2u.len(), n_queries);
                let mut leaf_digests: Vec<SC::DigestVariable> =
                    Vec::with_capacity(unique_opened.len());
                for slot_vals in unique_opened.iter() {
                    leaf_digests.push(Self::compute_leaf_digest(
                        builder,
                        &codeword_heights,
                        slot_vals,
                    ));
                }
                Self::verify_batch_pruned(
                    builder,
                    commitment_batch[batch_idx],
                    batch_max_log_height[batch_idx] + config.log_blowup,
                    leaf_digests,
                    &pruned.round_pruned[batch_idx],
                    &codeword_heights,
                    unique_opened.as_slice(),
                );
            }

            let mut ros: Vec<BTreeMap<usize, EF<C>>> = Vec::with_capacity(n_queries);
            for q in 0..n_queries {
                let mut ro: BTreeMap<usize, EF<C>> = BTreeMap::new();
                for (&log_height, members) in coeffs_by_height.iter() {
                    let mut sum: EF<C> = builder.constant(C::EF::zero());
                    for (batch_idx, mat_idx, coeffs) in members {
                        let slot = pruned.query_to_unique_slot[*batch_idx][q];
                        let opened = &pruned.round_opened_values[*batch_idx][slot][*mat_idx];
                        let val = Utils::<C, SC>::compute_dotproduct_mix(builder, coeffs, opened);
                        sum = builder.eval(sum + val);
                    }
                    ro.insert(log_height + config.log_blowup, sum);
                }
                ros.push(ro);
            }

            let pruned_iopp = proof
                .iopp_pruned
                .as_ref()
                .expect("non-stacking pruned input requires pruned IOPP proof");
            Self::verify_iopp_query_p3_pruned(
                builder,
                config,
                proof.iopp_oracles.as_slice(),
                &query_points,
                ros,
                pruned_iopp,
                &commit_log_foldings,
                &folding_challenges,
                &merge_betas,
                opening_point,
                &combined_f_r,
                final_codeword,
                &empty_branch_codewords,
            );
        } else {
            // Standard per-query input openings: independent merkle proofs.
            // [F-018] Pin the query counts so the zip below cannot be silently
            // truncated by a short proof (defensive; the recursion shape is
            // already fixed by the dummy-proof shape).
            assert_eq!(proof.iopp_queries.len(), config.num_queries);
            assert_eq!(proof.query_openings.len(), config.num_queries);
            for (q, (query, leaf_opening)) in
                proof.iopp_queries.iter().zip(proof.query_openings.iter()).enumerate()
            {
                // Per-batch merkle verification with that batch's codeword heights
                // and a right-shifted query index.
                for (batch_idx, opening) in leaf_opening.iter().enumerate() {
                    let codeword_heights: Vec<usize> = matrices_size_batch[batch_idx]
                        .iter()
                        .map(|d| d.height << config.log_blowup)
                        .collect();
                    let shift = num_vars - batch_max_log_height[batch_idx];
                    let index_bits = &query_points[q][shift..];
                    Self::verify_batch(
                        builder,
                        commitment_batch[batch_idx],
                        &codeword_heights,
                        index_bits,
                        opening.opened_values.clone(),
                        opening.opening_proof.clone(),
                    );
                }

                // Build per-height `ro` from this query's opened values.
                let mut ro: BTreeMap<usize, EF<C>> = BTreeMap::new();
                for (&log_height, members) in coeffs_by_height.iter() {
                    let mut sum: EF<C> = builder.constant(C::EF::zero());
                    for (batch_idx, mat_idx, coeffs) in members {
                        let opened: Vec<F<C>> = leaf_opening[*batch_idx].opened_values[*mat_idx]
                            .clone()
                            .into_iter()
                            .flatten()
                            .collect();
                        let val = Utils::<C, SC>::compute_dotproduct_mix(builder, coeffs, &opened);
                        sum = builder.eval(sum + val);
                    }
                    ro.insert(log_height + config.log_blowup, sum);
                }

                Self::verify_iopp_query_basefold(
                    builder,
                    config,
                    proof.iopp_oracles.as_slice(),
                    &query_points[q],
                    ro,
                    query,
                    &commit_log_foldings,
                    &folding_challenges,
                    &merge_betas,
                    opening_point,
                    &combined_f_r,
                    final_codeword,
                    &empty_branch_codewords,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_whir_round_pcs(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commitment_batch: Vec<<SC as FieldHasherVariable<C>>::DigestVariable>,
        layouts: &[StackedBatchLayout],
        coeffs_by_batch: &[StackedBatchCoefficients<C>],
        query_openings: &[Vec<BatchOpeningVariable<C, SC>>],
        input_pruned: Option<&crate::sumcheck::types::InputPrunedVariable<C, SC>>,
        iopp_oracles: &[SC::DigestVariable],
        ood_values: &[EF<C>],
        round_iopp: &WhirRoundQueryProofVariable<C, SC>,
        final_poly: &[EF<C>],
        sumcheck_transcript: &crate::sumcheck::types::SumcheckInstanceProofVariable<C>,
        stack_log_height: usize,
        k: usize,
        mut current_claim: EF<C>,
        full_opening_point: &[EF<C>],
        challenger: &mut SC::FriChallengerVariable,
    ) {
        let pruned_iopp = round_iopp.pruned.as_ref();
        let effective_log_foldings = if let Some(n) = config.num_committed_groups {
            let active = stack_log_height.saturating_sub(k);
            compute_uniform_log_foldings_circuit(active, n)
        } else {
            config.cross_round_log_foldings.clone()
        };
        let round_schedule = whir_reduced_rate_commit_schedule(
            stack_log_height,
            k,
            config.log_blowup,
            &effective_log_foldings,
        );
        let committed_groups = round_schedule.len();
        assert_eq!(iopp_oracles.len(), committed_groups);
        if let Some(pruned) = pruned_iopp {
            assert!(round_iopp.rounds.is_empty());
            assert_eq!(pruned.rounds.len(), committed_groups);
            assert!(query_openings.is_empty());
            assert!(input_pruned.is_some());
        } else {
            assert_eq!(round_iopp.rounds.len(), committed_groups);
            assert!(input_pruned.is_none());
        }
        assert_eq!(round_iopp.query_witnesses.len(), committed_groups);
        assert_eq!(round_iopp.folding_witnesses.len(), committed_groups);
        assert_eq!(ood_values.len(), committed_groups.saturating_sub(1));

        let consumed_sumcheck_rounds =
            round_schedule.iter().map(|(_, log_folding, _, _)| *log_folding).sum::<usize>();
        assert_eq!(sumcheck_transcript.uni_polys.len(), consumed_sumcheck_rounds);
        let final_log_height = stack_log_height - consumed_sumcheck_rounds;
        assert_eq!(final_poly.len(), 1usize << final_log_height);

        let one: EF<C> = builder.constant(C::EF::one());
        let mut weight_terms = vec![(one, full_opening_point.to_vec())];
        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut consumed_rounds = 0usize;

        challenger.observe(builder, iopp_oracles[0].clone());

        for (round_idx, &(start_round, log_folding, current_codeword_log, round_log_blowup)) in
            round_schedule.iter().enumerate()
        {
            assert_eq!(start_round, stack_log_height - consumed_rounds);
            let log_row_height = current_codeword_log - log_folding;
            let remaining_dim = start_round - log_folding;
            let mut group_challenges = Vec::with_capacity(log_folding);
            let folding_witness = &round_iopp.folding_witnesses[round_idx];

            for fold_idx in 0..log_folding {
                let uni_poly = poly_iter.next().expect("not enough WHIR sumcheck rounds");
                let eval_at_one = uni_poly.eval_at_one_horner(builder);
                builder.assert_ext_eq(uni_poly.eval_at_zero() + eval_at_one, current_claim);

                uni_poly.observe_into(builder, challenger);
                let r_fold = challenger.sample_ext(builder);
                current_claim = uni_poly.evaluate_horner(builder, &r_fold);
                group_challenges.push(r_fold);
                Self::fold_whir_symbolic_weight_terms(builder, &mut weight_terms, r_fold);

                if config.grinding_bits_folding > 0 {
                    assert_eq!(folding_witness.len(), 2 * log_folding);
                    let witness_offset = 2 * fold_idx;
                    challenger.observe(builder, folding_witness[witness_offset]);
                    challenger.check_witness(
                        builder,
                        config.grinding_bits_folding,
                        folding_witness[witness_offset + 1],
                    );
                }
            }

            let is_last_round = round_idx + 1 == committed_groups;
            let ood_point = if is_last_round {
                for coeff in final_poly.iter() {
                    let coeff_felts = C::ext2felt(builder, *coeff);
                    challenger.observe_slice(builder, coeff_felts);
                }
                None
            } else {
                challenger.observe(builder, iopp_oracles[round_idx + 1].clone());
                let z0 = challenger.sample_ext(builder);
                let y0 = ood_values[round_idx];
                let y0_felts = C::ext2felt(builder, y0);
                challenger.observe_slice(builder, y0_felts);
                Some((Self::whir_pow2_ext_point(builder, z0, remaining_dim), y0))
            };

            let query_witness = &round_iopp.query_witnesses[round_idx];
            assert_eq!(query_witness.len(), 2);
            challenger.observe(builder, query_witness[0]);
            challenger.check_witness(builder, config.grinding_bits_query, query_witness[1]);

            let num_queries = pruned_iopp
                .map(|pruned| pruned.rounds[round_idx].query_to_unique_slot.len())
                .unwrap_or_else(|| round_iopp.rounds[round_idx].query_proofs.len());
            let query_points: Vec<Vec<C::Bit>> = (0..num_queries)
                .map(|_| challenger.sample_bits(builder, current_codeword_log))
                .collect();

            let input_leaf_sums = if round_idx == 0 {
                if let Some(pruned) = input_pruned {
                    Self::verify_whir_first_round_inputs_pruned(
                        builder,
                        commitment_batch.as_slice(),
                        layouts,
                        coeffs_by_batch,
                        pruned,
                        &query_points,
                        stack_log_height,
                        config.log_blowup,
                    )
                } else {
                    Self::verify_whir_first_round_inputs(
                        builder,
                        commitment_batch.as_slice(),
                        layouts,
                        coeffs_by_batch,
                        query_openings,
                        &query_points,
                        stack_log_height,
                        config.log_blowup,
                    )
                }
            } else {
                Vec::new()
            };

            let gamma = challenger.sample_ext(builder);
            if let Some((point, value)) = ood_point.as_ref() {
                current_claim = builder.eval(current_claim + gamma * *value);
                weight_terms.push((gamma, point.clone()));
            }

            let mut gamma_power = builder.eval(gamma * gamma);
            let opened_rows_by_query = if let Some(pruned) = pruned_iopp {
                Self::verify_whir_iopp_round_pruned(
                    builder,
                    iopp_oracles[round_idx].clone(),
                    &query_points,
                    current_codeword_log,
                    log_folding,
                    &pruned.rounds[round_idx],
                )
            } else {
                let round_queries = &round_iopp.rounds[round_idx].query_proofs;
                query_points
                    .iter()
                    .zip(round_queries.iter())
                    .map(|(query_bits, query_proof)| {
                        Self::verify_whir_iopp_step_full(
                            builder,
                            iopp_oracles[round_idx].clone(),
                            query_bits,
                            current_codeword_log,
                            log_folding,
                            &query_proof.current_opening,
                        )
                    })
                    .collect::<Vec<_>>()
            };

            for (query_idx, (query_bits, opened_row)) in
                query_points.iter().zip(opened_rows_by_query.into_iter()).enumerate()
            {
                if round_idx == 0 {
                    let local_value =
                        Self::select_ext_by_bits(builder, &opened_row, &query_bits[..log_folding]);
                    builder.assert_ext_eq(local_value, input_leaf_sums[query_idx]);
                }

                let row_index_bits = &query_bits[log_folding..];
                let yi = Self::fold_whir_opened_iopp_row(
                    builder,
                    opened_row,
                    row_index_bits,
                    log_folding,
                    log_row_height,
                    &group_challenges,
                );
                current_claim = builder.eval(current_claim + gamma_power * yi);

                let query_point = Self::whir_codeword_query_point(
                    builder,
                    row_index_bits,
                    remaining_dim,
                    round_log_blowup,
                );
                weight_terms.push((gamma_power, query_point));
                gamma_power = builder.eval(gamma_power * gamma);
            }

            consumed_rounds += log_folding;
        }

        assert!(poly_iter.next().is_none());
        let final_acc = Self::whir_symbolic_final_accumulator(builder, final_poly, &weight_terms);
        builder.assert_ext_eq(final_acc, current_claim);
    }

    #[allow(clippy::too_many_arguments, clippy::ptr_arg)]
    pub fn verify_query_p3_batch(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commitments: &Vec<SC::DigestVariable>,
        commit_phase_commits: &[SC::DigestVariable],
        challenge_point: &[C::Bit],
        matrices_size: &Vec<Vec<Dimensions>>,
        query_proof: &FriQueryProofVariable<C, SC>,
        leaf_openings: &Vec<BatchOpeningVariable<C, SC>>,
        #[allow(clippy::type_complexity)] coefficients_by_height: &BTreeMap<
            usize,
            Vec<((usize, usize), Vec<EF<C>>)>,
        >,
        folding_rs: &[EF<C>],
        merging_rs: &[EF<C>],
        codeword: &EF<C>,
    ) {
        // Step 1: Verify leaf Merkle openings for each batch
        for i in 0..matrices_size.len() {
            let matrices_size_i = &matrices_size[i];
            let commitment = &commitments[i];
            let leaf_opening_i = leaf_openings[i].clone();

            let max_log_height = matrices_size_i
                .iter()
                .map(|shape| log2_strict_usize(shape.height))
                .max()
                .unwrap_or(0);

            let batch_heights =
                matrices_size_i.iter().map(|shape| shape.height << config.log_blowup).collect_vec();

            Self::verify_batch(
                builder,
                *commitment,
                &batch_heights,
                &challenge_point[(folding_rs.len() - max_log_height)..],
                leaf_opening_i.opened_values,
                leaf_opening_i.opening_proof,
            );
        }

        // Step 2: Compute the linear combination of opened leaves per height group
        let mut ro: BTreeMap<usize, EF<C>> = BTreeMap::new();
        coefficients_by_height.iter().for_each(|(&log_height, entries)| {
            let mut reduce_sum: EF<C> = builder.constant(C::EF::zero());
            entries.iter().for_each(|((batch_idx, mat_idx), coeffs)| {
                let opened_vals: Vec<F<C>> = leaf_openings[*batch_idx].opened_values[*mat_idx]
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect();
                let val = Utils::<C, SC>::compute_dotproduct_mix(builder, coeffs, &opened_vals);
                reduce_sum = builder.eval(reduce_sum + val);
            });
            ro.insert(log_height + config.log_blowup, reduce_sum);
        });

        // Step 3: Verify IOPP folding
        Self::verify_iopp_query_p3(
            builder,
            config,
            commit_phase_commits,
            challenge_point,
            ro,
            query_proof,
            folding_rs,
            merging_rs,
            codeword,
        );
    }

    /// [SS] Compute the leaf-layer digest for a single query's opened values,
    /// mirroring the tallest-first hash logic used by `verify_batch`.
    ///
    /// `heights[i]` is the (blown-up) height for matrix i in this batch.
    /// `opened_values[i]` is the per-row leaf data for matrix i at the query point.
    ///
    /// Only the tallest-height-group matrices contribute to the leaf digest
    /// (shorter matrices are injected at higher layers by the Plonky3 merkle tree).
    fn compute_leaf_digest(
        builder: &mut Builder<C>,
        heights: &[usize],
        opened_values: &[Vec<Felt<C::F>>],
    ) -> SC::DigestVariable {
        if heights.len() == 1 {
            return SC::hash(builder, opened_values[0].as_slice());
        }

        let mut heights_tallest_first =
            heights.iter().enumerate().sorted_by_key(|(_, height)| Reverse(*height)).peekable();

        let curr_height_padded = heights_tallest_first.peek().unwrap().1.next_power_of_two();

        let ext_slice = heights_tallest_first
            .peeking_take_while(|(_, height)| height.next_power_of_two() == curr_height_padded)
            .flat_map(|(i, _)| opened_values[i].as_slice());
        let felt_slice: Vec<Felt<C::F>> = ext_slice.cloned().collect::<Vec<_>>();
        SC::hash(builder, &felt_slice[..])
    }

    fn verify_batch(
        builder: &mut Builder<C>,
        commit: SC::DigestVariable,
        heights: &[usize],
        index_bits: &[C::Bit],
        opened_values: Vec<Vec<Vec<Felt<C::F>>>>,
        proof: Vec<SC::DigestVariable>,
    ) {
        let mut heights_tallest_first =
            heights.iter().enumerate().sorted_by_key(|(_, height)| Reverse(*height)).peekable();

        let mut curr_height_padded = heights_tallest_first.peek().unwrap().1.next_power_of_two();

        let ext_slice = heights_tallest_first
            .peeking_take_while(|(_, height)| height.next_power_of_two() == curr_height_padded)
            .flat_map(|(i, _)| opened_values[i].as_slice());
        let felt_slice: Vec<Felt<C::F>> = ext_slice.flatten().cloned().collect::<Vec<_>>();
        let mut root: SC::DigestVariable = SC::hash(builder, &felt_slice[..]);

        zip(index_bits.iter(), proof).for_each(|(&bit, sibling): (&C::Bit, SC::DigestVariable)| {
            let compress_args = SC::select_chain_digest(builder, bit, [root, sibling]);

            root = SC::compress(builder, compress_args);
            curr_height_padded >>= 1;

            let next_height = heights_tallest_first
                .peek()
                .map(|(_, height)| *height)
                .filter(|h| h.next_power_of_two() == curr_height_padded);

            if let Some(next_height) = next_height {
                let ext_slice = heights_tallest_first
                    .peeking_take_while(|(_, height)| *height == next_height)
                    .flat_map(|(i, _)| opened_values[i].as_slice());
                let felt_slice: Vec<Felt<C::F>> = ext_slice.flatten().cloned().collect::<Vec<_>>();
                let next_height_openings_digest = SC::hash(builder, &felt_slice);
                root = SC::compress(builder, [root, next_height_openings_digest]);
            }
        });

        SC::assert_digest_eq(builder, root, commit);
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_iopp_query_p3(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commit_phase_commits: &[SC::DigestVariable],
        challenge_point: &[C::Bit],
        ro: BTreeMap<usize, EF<C>>,
        query_proof: &FriQueryProofVariable<C, SC>,
        folding_rs: &[EF<C>],
        merging_rs: &[EF<C>],
        codeword: &EF<C>,
    ) {
        let num_vars = folding_rs.len();
        let mut ro_iter = ro.iter().rev().peekable();

        let one: EF<C> = builder.constant(C::EF::one());
        let mut folded_eval: EF<C> = builder.constant(C::EF::zero());
        let log_max_height = num_vars + config.log_blowup;
        let mut idx_rs_merge = 0;
        let padded_rs_merge: Vec<_> = once(one).chain(merging_rs.iter().cloned()).collect();
        for (round_i, (&_beta, comm, opening)) in
            izip!(folding_rs, commit_phase_commits, &query_proof.commit_phase_openings).enumerate()
        {
            let log_folded_height = log_max_height - round_i - 1;
            if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| **lh == log_folded_height + 1) {
                // ro merge: folded_eval = folded_eval + merge_r * (ro - folded_eval)
                let diff: EF<C> = builder.uninit();
                builder.push_op(DslIr::SubE(diff, *ro, folded_eval));
                let prod: EF<C> = builder.uninit();
                builder.push_op(DslIr::MulE(prod, padded_rs_merge[idx_rs_merge], diff));
                let new_folded: EF<C> = builder.uninit();
                builder.push_op(DslIr::AddE(new_folded, folded_eval, prod));
                folded_eval = new_folded;
                idx_rs_merge += 1;
            }
            let index_sibling = challenge_point[round_i];
            let index_pair = &challenge_point[(round_i + 1)..];

            let evals_ext = C::select_chain_ef(
                builder,
                index_sibling,
                once(folded_eval),
                once(opening.sibling_value),
            );
            let evals_felt = vec![
                C::ext2felt(builder, evals_ext[0]).to_vec(),
                C::ext2felt(builder, evals_ext[1]).to_vec(),
            ];
            let heights = &[1 << log_folded_height];
            Self::verify_batch(
                builder,
                *comm,
                heights,
                index_pair,
                [evals_felt.clone()].to_vec(),
                opening.opening_proof.clone(),
            );

            let generator: EF<C> =
                builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
            let g1: EF<C> = C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
            // Interpolation using raw DslIr instructions
            let g2: EF<C> = builder.uninit();
            builder.push_op(DslIr::NegE(g2, g1));
            let num: EF<C> = builder.uninit();
            builder.push_op(DslIr::SubE(num, evals_ext[1], evals_ext[0]));
            let den: EF<C> = builder.uninit();
            builder.push_op(DslIr::SubE(den, g2, g1));
            let k: EF<C> = builder.uninit();
            builder.push_op(DslIr::DivE(k, num, den));
            let k_g1: EF<C> = builder.uninit();
            builder.push_op(DslIr::MulE(k_g1, k, g1));
            let b: EF<C> = builder.uninit();
            builder.push_op(DslIr::SubE(b, evals_ext[0], k_g1));
            let k_fr: EF<C> = builder.uninit();
            builder.push_op(DslIr::MulE(k_fr, k, folding_rs[round_i]));
            let new_folded: EF<C> = builder.uninit();
            builder.push_op(DslIr::AddE(new_folded, b, k_fr));
            folded_eval = new_folded;
        }

        builder.assert_ext_eq(*codeword, folded_eval);
    }

    // [B-Stage 2] Pruned batch verification: BFS layer-walk gadget
    //
    // Verifies that N queries (N == leaf_digests.len()) all open against
    // the same merkle commitment, sharing one path-pruned authentication
    // proof. Statically unrolls the BFS schedule (computed natively in
    // ) and uses prover-supplied LSB hints to drive
    // the left/right ordering of each compress.
    //
    // Soundness (Stage 2 limitation): the BFS schedule itself is trusted.
    // A malicious prover could supply a mismatching schedule and walk an
    // arbitrary commit; Stage 3 (R1 fix) will constrain mask/pos against
    //  LSBs. Since  here are still
    // const-eval hints (see ), wiring the LSB
    // constraint requires reading  as bits-of-query-point
    // instead. Tracked separately.
    /// BFS-merged merkle proof verification, mirroring native
    /// `verify_batch_pruned` in `pruned.rs`.
    ///
    /// `heights[i]` is the (blown-up) height for matrix i in this batch.
    /// `opened_values_per_query[q][i]` is the opened values for query q,
    /// matrix i. These are needed for **matrix injection**: shorter
    /// matrices are injected at higher layers via hash-then-compress.
    ///
    /// When `heights` is empty (e.g. IOPP round where injection is not
    /// needed), the injection step is skipped entirely.
    fn verify_batch_pruned(
        builder: &mut Builder<C>,
        commit: SC::DigestVariable,
        log_max_height: usize,
        leaf_digests: Vec<SC::DigestVariable>,
        pruned: &crate::sumcheck::types::PrunedBatchProofVariable<C, SC>,
        heights: &[usize],
        opened_values_per_query: &[Vec<Vec<Felt<C::F>>>],
    ) {
        let schedule = pruned
            .native_schedule
            .as_ref()
            .expect("verify_batch_pruned: native_schedule must be Some (set in witness read)");
        assert_eq!(
            schedule.layer_count, log_max_height,
            "verify_batch_pruned: schedule.layer_count != log_max_height"
        );
        assert_eq!(
            schedule.layer_active_size[0] as usize,
            leaf_digests.len(),
            "verify_batch_pruned: schedule.layer_active_size[0] != n_queries"
        );

        // Pre-compute which heights (padded) correspond to which layer,
        // so we know when to inject shorter-matrix digests.
        // inject_at_padded[layer] = Some(actual_height) if matrices with
        // that padded height need injection after layer's compress.
        let max_height_padded = if heights.is_empty() {
            1usize << log_max_height
        } else {
            heights.iter().copied().map(|h| h.next_power_of_two()).max().unwrap_or(0)
        };

        // Build sorted-descending list of distinct padded heights (skip
        // the tallest, which is the leaf layer already handled).
        let mut inject_schedule: Vec<(usize, usize)> = Vec::new(); // (layer_after_which, actual_height)
        if !heights.is_empty() {
            let mut sorted_heights: Vec<(usize, usize)> = heights
                .iter()
                .enumerate()
                .sorted_by_key(|(_, &h)| Reverse(h))
                .map(|(i, &h)| (i, h))
                .collect();
            // Group by padded height; skip tallest group (already in leaf digest).
            let mut pad = max_height_padded;
            // Skip tallest group
            while !sorted_heights.is_empty() {
                let front_pad = sorted_heights[0].1.next_power_of_two();
                if front_pad != pad {
                    break;
                }
                sorted_heights.remove(0);
            }
            // Remaining groups: each distinct padded height triggers injection
            while !sorted_heights.is_empty() {
                let target_pad = sorted_heights[0].1.next_power_of_two();
                // At which layer? The layer is log2(max_height_padded) - log2(target_pad).
                // But injection happens *after* the compress at that layer.
                // In the native code, injection happens when curr_padded >>= 1
                // equals target_pad, i.e. after layer = log2(max_height_padded/target_pad) - 1.
                let layer_idx = log2_strict_usize(max_height_padded / target_pad) - 1;
                let actual_height = sorted_heights[0].1;
                inject_schedule.push((layer_idx, actual_height));
                // Remove all heights in this padded group
                while !sorted_heights.is_empty() &&
                    sorted_heights[0].1.next_power_of_two() == target_pad
                {
                    sorted_heights.remove(0);
                }
            }
        }

        // Track source-query index through BFS layers for injection.
        // source_q[i] = which original query index active[i] came from.
        let n_queries = leaf_digests.len();
        let mut source_q: Vec<usize> = (0..n_queries).collect();

        let mut active = leaf_digests;
        let mut sibling_cursor: usize = 0;

        for layer in 0..log_max_height {
            let steps = &schedule.per_layer_steps[layer];
            let next_size = schedule.layer_active_size[layer + 1] as usize;
            let mut next_active: Vec<SC::DigestVariable> = Vec::with_capacity(next_size);
            let mut next_source_q: Vec<usize> = Vec::with_capacity(next_size);
            let mut active_cursor: usize = 0;

            for (step_idx, step) in steps.iter().enumerate() {
                let (merged, src_q) = match step {
                    p3_merkle_tree::BfsStep::PairMerge => {
                        let m = SC::compress(
                            builder,
                            [active[active_cursor], active[active_cursor + 1]],
                        );
                        let sq = source_q[active_cursor];
                        active_cursor += 2;
                        (m, sq)
                    }
                    p3_merkle_tree::BfsStep::ConsumeSibling => {
                        let active_is_right = pruned.per_layer_sibling_pos[layer][step_idx];
                        let chain = if active_is_right {
                            [pruned.siblings[sibling_cursor], active[active_cursor]]
                        } else {
                            [active[active_cursor], pruned.siblings[sibling_cursor]]
                        };
                        let m = SC::compress(builder, chain);
                        let sq = source_q[active_cursor];
                        active_cursor += 1;
                        sibling_cursor += 1;
                        (m, sq)
                    }
                };

                // Matrix injection: check if shorter matrices need to
                // be injected at this layer. Mirrors native verify_batch_pruned.
                let mut parent = merged;
                for &(inject_layer, actual_height) in inject_schedule.iter() {
                    if inject_layer == layer {
                        // Hash the opened values of matrices at this height
                        // for the source query, then compress with parent.
                        let opened = &opened_values_per_query[src_q];
                        let inj_slice: Vec<Felt<C::F>> = heights
                            .iter()
                            .enumerate()
                            .sorted_by_key(|(_, &h)| Reverse(h))
                            .skip_while(|(_, &h)| {
                                h.next_power_of_two() > actual_height.next_power_of_two()
                            })
                            .take_while(|(_, &h)| h == actual_height)
                            .flat_map(|(i, _)| opened[i].iter().cloned())
                            .collect();
                        if !inj_slice.is_empty() {
                            let inj_digest = SC::hash(builder, &inj_slice);
                            parent = SC::compress(builder, [parent, inj_digest]);
                        }
                    }
                }

                next_active.push(parent);
                next_source_q.push(src_q);
            }

            assert_eq!(
                next_active.len(),
                next_size,
                "verify_batch_pruned: layer {} produced {} digests, expected {}",
                layer,
                next_active.len(),
                next_size,
            );
            active = next_active;
            source_q = next_source_q;
        }

        assert_eq!(active.len(), 1, "verify_batch_pruned: BFS must collapse to a single root");
        assert_eq!(
            sibling_cursor,
            pruned.siblings.len(),
            "verify_batch_pruned: sibling_cursor consumed {} siblings, proof has {}",
            sibling_cursor,
            pruned.siblings.len(),
        );

        SC::assert_digest_eq(builder, active[0], commit);
    }

    // [C-Stage 2] Pruned IOPP query verification (batched).
    //
    // Replaces the standard per-query Phase 6 IOPP loop with a single
    // batched call: instead of N independent merkle openings per round,
    // the prover ships ONE PrunedBatchProof per round that authenticates
    // all N queries against the same commit-phase commit. The per-round
    // sibling values (one per query) are still per-query.
    //
    // Layout (per query i, per round r):
    //   * folded_eval_i  -- per-query rolling fold state
    //   * sibling_value  -- pruned.sibling_values[i][r]
    //   * leaf_digest_i  -- hash([folded_eval_i, sibling_value] selected by query bit)
    //
    // Per round r:
    //   1. for i in 0..N: build evals_ext, leaf_digest_i
    //   2. verify_batch_pruned(commit_phase_commits[r], log_folded+1, leaf_digests,
    //      &round_pruned[r])
    //   3. for i in 0..N: interpolate + advance per-query folded_eval_i
    //
    // Soundness: same as verify_iopp_query_basefold for the fold arithmetic
    // (every query is independently verified). The merkle batching only
    // changes how the prover authenticates "all openings are consistent
    // with the same commit", which verify_batch_pruned enforces via the
    // BFS layer-walk gadget.
    #[allow(clippy::too_many_arguments)]
    fn verify_iopp_query_p3_pruned(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commit_phase_commits: &[SC::DigestVariable],
        query_points: &[Vec<C::Bit>],
        ros: Vec<BTreeMap<usize, EF<C>>>,
        pruned: &crate::sumcheck::types::PrunedFriQueryProofVariable<C, SC>,
        commit_log_foldings: &[usize],
        folding_rs: &[EF<C>],
        merge_betas: &[EF<C>],
        opening_point: &[EF<C>],
        codeword: &EF<C>,
        final_codeword: Option<&[EF<C>]>,
        final_branch_codewords: &BTreeMap<usize, Vec<EF<C>>>,
    ) {
        let n_queries = query_points.len();
        let num_vars = folding_rs.len();
        let log_max_height = num_vars + config.log_blowup;
        let mut active_log_foldings = Vec::new();
        let mut scheduled_rounds = 0usize;
        if commit_log_foldings.is_empty() {
            active_log_foldings.resize(commit_phase_commits.len().min(num_vars), 1);
        } else {
            for &log_folding in commit_log_foldings {
                if scheduled_rounds >= num_vars {
                    break;
                }
                assert!(log_folding > 0);
                assert!(scheduled_rounds + log_folding <= num_vars);
                active_log_foldings.push(log_folding);
                scheduled_rounds += log_folding;
            }
        }
        let committed_groups = active_log_foldings.len();

        assert_eq!(ros.len(), n_queries, "verify_iopp_query_p3_pruned: ros.len() != n_queries",);
        assert!(pruned.round_pruned_proofs.len() >= committed_groups);
        assert_eq!(
            pruned.sibling_values.len(),
            n_queries,
            "verify_iopp_query_p3_pruned: sibling_values outer len != n_queries",
        );
        assert!(pruned.query_to_unique_slot.len() >= committed_groups);
        if !pruned.round_opened_values.is_empty() {
            assert!(pruned.round_opened_values.len() >= committed_groups);
        }

        let one: EF<C> = builder.constant(C::EF::one());

        // Per-query rolling state. Initialized to zero; first ro merge
        // (round 0 typically) replaces it with the leaf_sum, mirroring
        // verify_iopp_query_basefold.
        let mut folded_evals: Vec<EF<C>> =
            (0..n_queries).map(|_| builder.constant(C::EF::zero())).collect();
        let mut eq_factors: Vec<EF<C>> = (0..n_queries).map(|_| one).collect();
        let mut merge_indices: Vec<usize> = vec![0; n_queries];

        // Wrap each ro in a peekable iterator so we can drain "next height
        // group" entries per query, as standard verify_iopp_query_basefold does.
        let mut ro_iters: Vec<_> = ros.iter().map(|ro| ro.iter().rev().peekable()).collect();
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());
        let mut consumed_rounds = 0usize;

        for (opening_idx, &log_folding) in active_log_foldings.iter().enumerate() {
            let current_codeword_log = log_max_height - consumed_rounds;
            let log_row_height = current_codeword_log - log_folding;
            let row_width = 1usize << log_folding;
            let first_log_folded_height = current_codeword_log - 1;
            let round_pruned = &pruned.round_pruned_proofs[opening_idx];
            let q_to_slot = &pruned.query_to_unique_slot[opening_idx];
            assert_eq!(q_to_slot.len(), n_queries);
            let opened_rows_for_round =
                pruned.round_opened_values.get(opening_idx).filter(|rows| !rows.is_empty());

            let mut rows_per_query: Vec<Vec<EF<C>>> = Vec::with_capacity(n_queries);
            let mut leaf_digests: Vec<SC::DigestVariable> = Vec::new();

            for q in 0..n_queries {
                // Drain ro at this height for query q.
                if let Some((_, &leaf_sum)) =
                    ro_iters[q].next_if(|(lh, _)| **lh == first_log_folded_height + 1)
                {
                    if merge_indices[q] == 0 {
                        folded_evals[q] = leaf_sum;
                    } else {
                        folded_evals[q] = builder.eval(
                            eq_factors[q] * folded_evals[q] +
                                merge_betas[merge_indices[q] - 1] * leaf_sum,
                        );
                        eq_factors[q] = one;
                    }
                    merge_indices[q] += 1;
                }

                if let Some(opened_rows) = opened_rows_for_round {
                    let slot = q_to_slot[q];
                    assert!(slot < opened_rows.len());
                    assert_eq!(opened_rows[slot].len(), 1);
                    assert_eq!(opened_rows[slot][0].len(), row_width);
                    rows_per_query.push(opened_rows[slot][0].clone());
                } else {
                    assert_eq!(log_folding, 1);
                    let index_sibling = query_points[q][consumed_rounds];
                    let evals_ext = C::select_chain_ef(
                        builder,
                        index_sibling,
                        once(folded_evals[q]),
                        once(pruned.sibling_values[q][opening_idx]),
                    );
                    rows_per_query.push(vec![evals_ext[0], evals_ext[1]]);
                    let evals_felt = vec![
                        C::ext2felt(builder, evals_ext[0]).to_vec(),
                        C::ext2felt(builder, evals_ext[1]).to_vec(),
                    ];
                    let felt_slice: Vec<Felt<C::F>> =
                        evals_felt.iter().flatten().cloned().collect();
                    leaf_digests.push(SC::hash(builder, &felt_slice[..]));
                }
            }

            let m_unique = round_pruned.sorted_indices.len();
            let merged_leaves = if let Some(opened_rows) = opened_rows_for_round {
                assert_eq!(opened_rows.len(), m_unique);
                opened_rows
                    .iter()
                    .map(|slot| {
                        assert_eq!(slot.len(), 1);
                        let felt_slice: Vec<Felt<C::F>> = slot[0]
                            .iter()
                            .flat_map(|value| C::ext2felt(builder, *value).to_vec())
                            .collect();
                        SC::hash(builder, &felt_slice[..])
                    })
                    .collect::<Vec<_>>()
            } else {
                let mut slot_leaves: Vec<Option<SC::DigestVariable>> = vec![None; m_unique];
                for q in 0..n_queries {
                    let k = q_to_slot[q];
                    assert!(k < m_unique);
                    match slot_leaves[k] {
                        None => slot_leaves[k] = Some(leaf_digests[q]),
                        Some(first) => {
                            SC::assert_digest_eq(builder, leaf_digests[q], first);
                        }
                    }
                }
                slot_leaves
                    .into_iter()
                    .enumerate()
                    .map(|(k, opt)| {
                        opt.unwrap_or_else(|| {
                            panic!(
                                "verify_iopp_query_p3_pruned: slot {} not covered by any query (round_i={}, m_unique={})",
                                k, opening_idx, m_unique,
                            )
                        })
                    })
                    .collect()
            };

            Self::verify_batch_pruned(
                builder,
                commit_phase_commits[opening_idx],
                log_row_height,
                merged_leaves,
                round_pruned,
                &[],
                &[],
            );

            for q in 0..n_queries {
                let row_index_bits = &query_points[q][(consumed_rounds + log_folding)..];
                let mut opened_row = rows_per_query[q].clone();
                for local_round in 0..log_folding {
                    let round_i = consumed_rounds + local_round;
                    let log_folded_height = log_max_height - round_i - 1;
                    if local_round > 0 {
                        assert!(!ro_iters[q]
                            .peek()
                            .is_some_and(|(lh, _)| **lh == log_folded_height + 1));
                    }

                    let index_sibling = query_points[q][round_i];
                    let local_pair_bits =
                        &query_points[q][(round_i + 1)..(consumed_rounds + log_folding)];
                    let even_values: Vec<_> =
                        opened_row.chunks_exact(2).map(|pair| pair[0]).collect();
                    let odd_values: Vec<_> =
                        opened_row.chunks_exact(2).map(|pair| pair[1]).collect();
                    let evals_ext = [
                        Self::select_ext_by_bits(builder, &even_values, local_pair_bits),
                        Self::select_ext_by_bits(builder, &odd_values, local_pair_bits),
                    ];
                    let selected = C::select_chain_ef(
                        builder,
                        index_sibling,
                        once(evals_ext[0]),
                        once(evals_ext[1]),
                    );
                    builder.assert_ext_eq(selected[0], folded_evals[q]);

                    let index_pair = &query_points[q][(round_i + 1)..];
                    let generator: EF<C> =
                        builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
                    let g1: EF<C> =
                        C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
                    let g2: EF<C> = builder.eval(-g1);

                    let k = (evals_ext[1] - evals_ext[0]) / (g2 - g1);
                    let b = evals_ext[0] - k * g1;
                    folded_evals[q] = builder.eval(b + k * folding_rs[round_i]);

                    if local_round + 1 < log_folding {
                        opened_row = Self::fold_opened_row_block(
                            builder,
                            &opened_row,
                            row_index_bits,
                            log_folding - local_round,
                            log_row_height,
                            folding_rs[round_i],
                        );
                    }

                    let var_idx = num_vars - 1 - round_i;
                    let p_i = opening_point[var_idx];
                    let fc_i = folding_rs[round_i];
                    eq_factors[q] =
                        builder.eval(eq_factors[q] * (p_i * fc_i + (one - p_i) * (one - fc_i)));
                }
            }
            consumed_rounds += log_folding;
        }

        for round_i in consumed_rounds..num_vars {
            let log_folded_height = log_max_height - round_i - 1;

            if let Some(branch_codeword) = final_branch_codewords.get(&(log_folded_height + 1)) {
                let codeword_values = virtual_codeword
                    .as_mut()
                    .expect("missing Basefold final codeword for early-stop pruned branch merge");
                assert_eq!(codeword_values.len(), branch_codeword.len());
                let merge_idx = merge_indices[0];
                assert!(merge_idx > 0, "invalid Basefold branch merge before first height group");
                let merge_beta = merge_betas[merge_idx - 1];
                let eq_factor = eq_factors[0];
                for (value, branch_value) in codeword_values.iter_mut().zip(branch_codeword.iter())
                {
                    *value = builder.eval(eq_factor * *value + merge_beta * *branch_value);
                }
            }

            let codeword_values = virtual_codeword
                .as_ref()
                .expect("missing Basefold final codeword for early-stop pruned query");
            let even_values: Vec<_> = codeword_values.chunks_exact(2).map(|pair| pair[0]).collect();
            let odd_values: Vec<_> = codeword_values.chunks_exact(2).map(|pair| pair[1]).collect();

            for q in 0..n_queries {
                if let Some((_, &leaf_sum)) =
                    ro_iters[q].next_if(|(lh, _)| **lh == log_folded_height + 1)
                {
                    if merge_indices[q] == 0 {
                        folded_evals[q] = leaf_sum;
                    } else {
                        folded_evals[q] = builder.eval(
                            eq_factors[q] * folded_evals[q] +
                                merge_betas[merge_indices[q] - 1] * leaf_sum,
                        );
                        eq_factors[q] = one;
                    }
                    merge_indices[q] += 1;
                }

                let index_sibling = query_points[q][round_i];
                let index_pair = &query_points[q][(round_i + 1)..];
                let evals_ext = [
                    Self::select_ext_by_bits(builder, &even_values, index_pair),
                    Self::select_ext_by_bits(builder, &odd_values, index_pair),
                ];
                if round_i == consumed_rounds {
                    let selected = C::select_chain_ef(
                        builder,
                        index_sibling,
                        once(evals_ext[0]),
                        once(evals_ext[1]),
                    );
                    builder.assert_ext_eq(selected[0], folded_evals[q]);
                }

                let generator: EF<C> =
                    builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
                let g1: EF<C> = C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
                let g2: EF<C> = builder.eval(-g1);

                let k = (evals_ext[1] - evals_ext[0]) / (g2 - g1);
                let b = evals_ext[0] - k * g1;
                folded_evals[q] = builder.eval(b + k * folding_rs[round_i]);

                let var_idx = num_vars - 1 - round_i;
                let p_i = opening_point[var_idx];
                let fc_i = folding_rs[round_i];
                eq_factors[q] =
                    builder.eval(eq_factors[q] * (p_i * fc_i + (one - p_i) * (one - fc_i)));
            }

            if let Some(ref mut codeword_values) = virtual_codeword {
                *codeword_values =
                    Self::fold_codeword(builder, codeword_values, folding_rs[round_i]);
            }
        }

        // --- Final: every per-query end state must equal combined codeword ---
        for q in 0..n_queries {
            builder.assert_ext_eq(*codeword, folded_evals[q]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_query_p3_batch_basefold(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commitments: &Vec<SC::DigestVariable>,
        commit_phase_commits: &[SC::DigestVariable],
        challenge_point: &[C::Bit],
        matrices_size: &Vec<Vec<Dimensions>>,
        query_proof: &FriQueryProofVariable<C, SC>,
        leaf_openings: &Vec<BatchOpeningVariable<C, SC>>,
        coefficients_by_height: &BTreeMap<usize, Vec<((usize, usize), Vec<EF<C>>)>>,
        folding_rs: &[EF<C>],
        merge_betas: &[EF<C>],
        opening_point: &[EF<C>],
        codeword: &EF<C>,
        final_codeword: Option<&[EF<C>]>,
        final_branch_codewords: &BTreeMap<usize, Vec<EF<C>>>,
    ) {
        // Step 1: Verify leaf Merkle openings for each batch
        for i in 0..matrices_size.len() {
            let matrices_size_i = &matrices_size[i];
            let commitment = &commitments[i];
            let leaf_opening_i = leaf_openings[i].clone();

            let max_log_height = matrices_size_i
                .iter()
                .map(|shape| log2_strict_usize(shape.height))
                .max()
                .unwrap_or(0);

            let batch_heights =
                matrices_size_i.iter().map(|shape| shape.height << config.log_blowup).collect_vec();

            Self::verify_batch(
                builder,
                *commitment,
                &batch_heights,
                &challenge_point[(folding_rs.len() - max_log_height)..],
                leaf_opening_i.opened_values,
                leaf_opening_i.opening_proof,
            );
        }

        // Step 2: Compute the linear combination of opened leaves per height group
        let mut ro: BTreeMap<usize, EF<C>> = BTreeMap::new();
        coefficients_by_height.iter().for_each(|(&log_height, entries)| {
            let mut reduce_sum: EF<C> = builder.constant(C::EF::zero());
            entries.iter().for_each(|((batch_idx, mat_idx), coeffs)| {
                let opened_vals: Vec<F<C>> = leaf_openings[*batch_idx].opened_values[*mat_idx]
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect();
                let val = Utils::<C, SC>::compute_dotproduct_mix(builder, coeffs, &opened_vals);
                reduce_sum = builder.eval(reduce_sum + val);
            });
            ro.insert(log_height + config.log_blowup, reduce_sum);
        });

        // Step 3: Verify IOPP folding (basefold variant)
        Self::verify_iopp_query_basefold(
            builder,
            config,
            commit_phase_commits,
            challenge_point,
            ro,
            query_proof,
            &[],
            folding_rs,
            merge_betas,
            opening_point,
            codeword,
            final_codeword,
            final_branch_codewords,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_iopp_query_basefold(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        commit_phase_commits: &[SC::DigestVariable],
        challenge_point: &[C::Bit],
        ro: BTreeMap<usize, EF<C>>,
        query_proof: &FriQueryProofVariable<C, SC>,
        commit_log_foldings: &[usize],
        folding_rs: &[EF<C>],
        merge_betas: &[EF<C>],
        opening_point: &[EF<C>],
        codeword: &EF<C>,
        final_codeword: Option<&[EF<C>]>,
        final_branch_codewords: &BTreeMap<usize, Vec<EF<C>>>,
    ) {
        let num_vars = folding_rs.len();
        let mut ro_iter = ro.iter().rev().peekable();

        let one: EF<C> = builder.constant(C::EF::one());
        let mut folded_eval: EF<C> = builder.constant(C::EF::zero());
        let log_max_height = num_vars + config.log_blowup;
        let mut merge_idx: usize = 0;
        let mut eq_factor: EF<C> = one;
        let mut virtual_codeword = final_codeword.map(|codeword| codeword.to_vec());

        let mut committed_rounds = 0usize;
        for (opening_idx, (comm, opening)) in
            commit_phase_commits.iter().zip(query_proof.commit_phase_openings.iter()).enumerate()
        {
            if committed_rounds >= num_vars {
                break;
            }
            let log_folding = commit_log_foldings.get(opening_idx).copied().unwrap_or(1);
            assert!(log_folding > 0 && committed_rounds + log_folding <= num_vars);

            let current_codeword_log = log_max_height - committed_rounds;
            let log_row_height = current_codeword_log - log_folding;
            let log_folded_height = current_codeword_log - 1;

            if let Some((_, &leaf_sum)) = ro_iter.next_if(|(lh, _)| **lh == log_folded_height + 1) {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = builder
                        .eval(eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum);
                    eq_factor = one;
                }
                merge_idx += 1;
            }

            if log_folding == 1 && opening.leaf_values.is_empty() {
                let index_sibling = challenge_point[committed_rounds];
                let index_pair = &challenge_point[(committed_rounds + 1)..];

                let evals_ext = C::select_chain_ef(
                    builder,
                    index_sibling,
                    once(folded_eval),
                    once(opening.sibling_value),
                );
                let evals_felt = vec![
                    C::ext2felt(builder, evals_ext[0]).to_vec(),
                    C::ext2felt(builder, evals_ext[1]).to_vec(),
                ];
                let heights = &[1 << log_folded_height];
                Self::verify_batch(
                    builder,
                    *comm,
                    heights,
                    index_pair,
                    [evals_felt.clone()].to_vec(),
                    opening.opening_proof.clone(),
                );

                let generator: EF<C> =
                    builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
                let g1: EF<C> = C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
                let g2: EF<C> = builder.eval(-g1);

                let k = (evals_ext[1].clone() - evals_ext[0].clone()) / (g2 - g1);
                let b = evals_ext[0] - k * g1;
                folded_eval = builder.eval(b + k * folding_rs[committed_rounds]);

                let var_idx = num_vars - 1 - committed_rounds;
                let p_i = opening_point[var_idx];
                let fc_i = folding_rs[committed_rounds];
                eq_factor = builder.eval(eq_factor * (p_i * fc_i + (one - p_i) * (one - fc_i)));
                committed_rounds += 1;
                continue;
            }

            assert_eq!(opening.leaf_values.len(), 1usize << log_folding);
            let row_index_bits = &challenge_point[(committed_rounds + log_folding)..];
            let local_bits = &challenge_point[committed_rounds..(committed_rounds + log_folding)];
            let selected_leaf = Self::select_ext_by_bits(builder, &opening.leaf_values, local_bits);
            builder.assert_ext_eq(selected_leaf, folded_eval);

            let row_felts = opening
                .leaf_values
                .iter()
                .map(|&value| C::ext2felt(builder, value).to_vec())
                .collect::<Vec<_>>();
            Self::verify_batch(
                builder,
                *comm,
                &[1 << log_row_height],
                row_index_bits,
                vec![row_felts],
                opening.opening_proof.clone(),
            );

            let mut block_values = opening.leaf_values.clone();
            for local_round in 0..log_folding {
                let round_i = committed_rounds + local_round;
                let log_folded_height = log_max_height - round_i - 1;
                if local_round > 0 {
                    assert!(
                        ro_iter
                            .peek()
                            .map_or(true, |(lh, _)| **lh != log_folded_height + 1),
                        "cross-round Basefold does not support branch merges inside a skipped window",
                    );
                }

                let pair_select_bits =
                    &challenge_point[(round_i + 1)..(committed_rounds + log_folding)];
                let even_values =
                    block_values.chunks_exact(2).map(|pair| pair[0]).collect::<Vec<_>>();
                let odd_values =
                    block_values.chunks_exact(2).map(|pair| pair[1]).collect::<Vec<_>>();
                let evals_ext = [
                    Self::select_ext_by_bits(builder, &even_values, pair_select_bits),
                    Self::select_ext_by_bits(builder, &odd_values, pair_select_bits),
                ];
                let selected = C::select_chain_ef(
                    builder,
                    challenge_point[round_i],
                    once(evals_ext[0]),
                    once(evals_ext[1]),
                );
                builder.assert_ext_eq(selected[0], folded_eval);

                let index_pair = &challenge_point[(round_i + 1)..];
                let generator: EF<C> =
                    builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
                let g1: EF<C> = C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
                let g2: EF<C> = builder.eval(-g1);
                let k = (evals_ext[1] - evals_ext[0]) / (g2 - g1);
                let b = evals_ext[0] - k * g1;
                folded_eval = builder.eval(b + k * folding_rs[round_i]);

                if local_round + 1 < log_folding {
                    block_values = Self::fold_opened_row_block(
                        builder,
                        &block_values,
                        row_index_bits,
                        log_folding - local_round,
                        log_row_height,
                        folding_rs[round_i],
                    );
                }

                let var_idx = num_vars - 1 - round_i;
                let p_i = opening_point[var_idx];
                let fc_i = folding_rs[round_i];
                eq_factor = builder.eval(eq_factor * (p_i * fc_i + (one - p_i) * (one - fc_i)));
            }

            committed_rounds += log_folding;
        }

        for round_i in committed_rounds..num_vars {
            let log_folded_height = log_max_height - round_i - 1;

            if let Some(branch_codeword) = final_branch_codewords.get(&(log_folded_height + 1)) {
                let codeword_values = virtual_codeword
                    .as_mut()
                    .expect("missing Basefold final codeword for early-stop branch merge");
                assert_eq!(codeword_values.len(), branch_codeword.len());
                assert!(merge_idx > 0, "invalid Basefold branch merge before first height group");
                let merge_beta = merge_betas[merge_idx - 1];
                for (value, branch_value) in codeword_values.iter_mut().zip(branch_codeword.iter())
                {
                    *value = builder.eval(eq_factor * *value + merge_beta * *branch_value);
                }
            }

            if let Some((_, &leaf_sum)) = ro_iter.next_if(|(lh, _)| **lh == log_folded_height + 1) {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = builder
                        .eval(eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum);
                    eq_factor = one;
                }
                merge_idx += 1;
            }

            let codeword_values = virtual_codeword
                .as_ref()
                .expect("missing Basefold final codeword for early-stop query");
            let even_values: Vec<_> = codeword_values.chunks_exact(2).map(|pair| pair[0]).collect();
            let odd_values: Vec<_> = codeword_values.chunks_exact(2).map(|pair| pair[1]).collect();
            let index_sibling = challenge_point[round_i];
            let index_pair = &challenge_point[(round_i + 1)..];
            let evals_ext = [
                Self::select_ext_by_bits(builder, &even_values, index_pair),
                Self::select_ext_by_bits(builder, &odd_values, index_pair),
            ];
            if round_i == committed_rounds {
                let selected = C::select_chain_ef(
                    builder,
                    index_sibling,
                    once(evals_ext[0]),
                    once(evals_ext[1]),
                );
                builder.assert_ext_eq(selected[0], folded_eval);
            }

            let generator: EF<C> =
                builder.constant(C::EF::two_adic_generator(log_folded_height + 1));
            let g1: EF<C> = C::exp_reverse_bits_ext(builder, generator, index_pair.to_vec());
            let g2: EF<C> = builder.eval(-g1);

            let k = (evals_ext[1] - evals_ext[0]) / (g2 - g1);
            let b = evals_ext[0] - k * g1;
            folded_eval = builder.eval(b + k * folding_rs[round_i]);

            if let Some(ref mut codeword_values) = virtual_codeword {
                *codeword_values =
                    Self::fold_codeword(builder, codeword_values, folding_rs[round_i]);
            }

            let var_idx = num_vars - 1 - round_i;
            let p_i = opening_point[var_idx];
            let fc_i = folding_rs[round_i];
            eq_factor = builder.eval(eq_factor * (p_i * fc_i + (one - p_i) * (one - fc_i)));
        }

        builder.assert_ext_eq(*codeword, folded_eval);
    }

    fn encode_final_poly_to_codeword(
        builder: &mut Builder<C>,
        config: &FriConfig<SCFriMmcs<SC>>,
        final_poly: &[EF<C>],
    ) -> Vec<EF<C>> {
        let repeat_times = 1 << config.log_blowup;
        let mut codeword = Vec::with_capacity(final_poly.len() * repeat_times);
        for _ in 0..repeat_times {
            codeword.extend_from_slice(final_poly);
        }
        Self::dft_evals_no_bit_reverse(builder, &mut codeword);
        codeword
    }

    fn dft_evals_no_bit_reverse(builder: &mut Builder<C>, values: &mut [EF<C>]) {
        if values.len() <= 1 {
            return;
        }
        let log_h = log2_strict_usize(values.len());
        let generator = C::F::two_adic_generator(log_h);
        let half_n = 1 << (log_h - 1);
        let nth_roots: Vec<_> = generator.powers().take(half_n).collect();
        let mut root_table: Vec<Vec<C::F>> = (0..log_h)
            .map(|i| {
                let mut twiddles: Vec<_> = nth_roots.iter().step_by(1 << i).copied().collect();
                reverse_slice_index_bits(&mut twiddles);
                twiddles
            })
            .collect();

        for twiddles in root_table.iter_mut().rev() {
            let size = values.len();
            let num_blocks = twiddles.len();
            let block_size = size / num_blocks;
            let half_block_size = block_size / 2;

            for (block_idx, &twiddle) in twiddles.iter().enumerate() {
                let twiddle_ext: EF<C> = builder.constant(C::EF::from_base(twiddle));
                let block_start = block_idx * block_size;
                for offset in 0..half_block_size {
                    let hi_idx = block_start + offset;
                    let lo_idx = block_start + half_block_size + offset;
                    let x_1 = values[hi_idx];
                    let x_2 = values[lo_idx];
                    let x_2_twiddle: EF<C> = builder.eval((x_2 - x_1) * twiddle_ext);
                    values[hi_idx] = builder.eval(x_1 + x_2_twiddle);
                    values[lo_idx] = builder.eval(x_1 - x_2_twiddle);
                }
            }
        }
    }

    fn verify_whir_first_round_inputs(
        builder: &mut Builder<C>,
        commitment_batch: &[SC::DigestVariable],
        layouts: &[StackedBatchLayout],
        coeffs_by_batch: &[StackedBatchCoefficients<C>],
        query_openings: &[Vec<BatchOpeningVariable<C, SC>>],
        query_points: &[Vec<C::Bit>],
        stack_log_height: usize,
        log_blowup: usize,
    ) -> Vec<EF<C>> {
        assert_eq!(query_openings.len(), query_points.len());
        let stacked_height = (1usize << stack_log_height) << log_blowup;
        let stacked_heights = vec![stacked_height];

        query_openings
            .iter()
            .zip(query_points.iter())
            .map(|(leaf_opening, query_bits)| {
                assert_eq!(leaf_opening.len(), coeffs_by_batch.len());
                for (batch_idx, opening) in leaf_opening.iter().enumerate() {
                    Self::verify_batch(
                        builder,
                        commitment_batch[batch_idx],
                        &stacked_heights,
                        query_bits,
                        opening.opened_values.clone(),
                        opening.opening_proof.clone(),
                    );
                    assert_eq!(layouts[batch_idx].width, opening.opened_values[0].len());
                }

                let mut leaf_sum: EF<C> = builder.constant(C::EF::zero());
                for (coeffs, opening) in coeffs_by_batch.iter().zip(leaf_opening.iter()) {
                    let opened_vals: Vec<F<C>> =
                        opening.opened_values[0].clone().into_iter().flatten().collect();
                    let val = Utils::<C, SC>::compute_dotproduct_mix(
                        builder,
                        &coeffs.column_coeffs,
                        &opened_vals,
                    );
                    leaf_sum = builder.eval(leaf_sum + val);
                }
                leaf_sum
            })
            .collect()
    }

    fn verify_whir_first_round_inputs_pruned(
        builder: &mut Builder<C>,
        commitment_batch: &[SC::DigestVariable],
        layouts: &[StackedBatchLayout],
        coeffs_by_batch: &[StackedBatchCoefficients<C>],
        input_pruned: &crate::sumcheck::types::InputPrunedVariable<C, SC>,
        query_points: &[Vec<C::Bit>],
        stack_log_height: usize,
        log_blowup: usize,
    ) -> Vec<EF<C>> {
        let num_batches = coeffs_by_batch.len();
        assert_eq!(commitment_batch.len(), num_batches);
        assert_eq!(layouts.len(), num_batches);
        assert_eq!(input_pruned.round_pruned.len(), num_batches);
        assert_eq!(input_pruned.round_opened_values.len(), num_batches);
        assert_eq!(input_pruned.query_to_unique_slot.len(), num_batches);

        let stacked_height = (1usize << stack_log_height) << log_blowup;
        let stacked_heights = vec![stacked_height];
        let log_stacked_height = stack_log_height + log_blowup;

        for batch_idx in 0..num_batches {
            let opened_values = &input_pruned.round_opened_values[batch_idx];
            let q2u = &input_pruned.query_to_unique_slot[batch_idx];
            assert_eq!(q2u.len(), query_points.len());
            assert_eq!(
                opened_values.len(),
                input_pruned.round_pruned[batch_idx].sorted_indices.len()
            );
            Self::assert_pruned_query_slot_mapping(
                builder,
                query_points,
                q2u,
                &input_pruned.round_pruned[batch_idx].sorted_indices,
                0,
            );

            for opened in opened_values {
                assert_eq!(opened.len(), 1);
                assert_eq!(opened[0].len(), layouts[batch_idx].width);
            }

            let leaf_digests = opened_values
                .iter()
                .map(|opened| Self::compute_leaf_digest(builder, &stacked_heights, opened))
                .collect::<Vec<_>>();

            Self::verify_batch_pruned(
                builder,
                commitment_batch[batch_idx],
                log_stacked_height,
                leaf_digests,
                &input_pruned.round_pruned[batch_idx],
                &stacked_heights,
                opened_values,
            );
        }

        (0..query_points.len())
            .map(|query_idx| {
                let mut leaf_sum: EF<C> = builder.constant(C::EF::zero());
                for batch_idx in 0..num_batches {
                    let slot = input_pruned.query_to_unique_slot[batch_idx][query_idx];
                    let opened_values = &input_pruned.round_opened_values[batch_idx];
                    assert!(slot < opened_values.len());
                    let opened_vals = opened_values[slot][0].clone();
                    let val = Utils::<C, SC>::compute_dotproduct_mix(
                        builder,
                        &coeffs_by_batch[batch_idx].column_coeffs,
                        &opened_vals,
                    );
                    leaf_sum = builder.eval(leaf_sum + val);
                }
                leaf_sum
            })
            .collect()
    }

    fn verify_whir_iopp_step_full(
        builder: &mut Builder<C>,
        commitment: SC::DigestVariable,
        query_bits: &[C::Bit],
        current_codeword_log: usize,
        log_folding: usize,
        step: &FriCommitPhaseProofStepVariable<C, SC>,
    ) -> Vec<EF<C>> {
        assert!(log_folding > 0);
        assert!(current_codeword_log >= log_folding);
        assert_eq!(query_bits.len(), current_codeword_log);
        let row_width = 1usize << log_folding;
        let log_row_height = current_codeword_log - log_folding;
        assert_eq!(step.leaf_values.len(), row_width);

        let row_felts = step
            .leaf_values
            .iter()
            .map(|value| C::ext2felt(builder, *value).to_vec())
            .collect::<Vec<_>>();
        Self::verify_batch(
            builder,
            commitment,
            &[1usize << log_row_height],
            &query_bits[log_folding..],
            vec![row_felts],
            step.opening_proof.clone(),
        );

        step.leaf_values.clone()
    }

    fn verify_whir_iopp_round_pruned(
        builder: &mut Builder<C>,
        commitment: SC::DigestVariable,
        query_points: &[Vec<C::Bit>],
        current_codeword_log: usize,
        log_folding: usize,
        pruned_round: &WhirPrunedIoppRoundVariable<C, SC>,
    ) -> Vec<Vec<EF<C>>> {
        assert!(log_folding > 0);
        assert!(current_codeword_log >= log_folding);
        let row_width = 1usize << log_folding;
        let log_row_height = current_codeword_log - log_folding;
        assert_eq!(pruned_round.query_to_unique_slot.len(), query_points.len());
        assert_eq!(pruned_round.opened_rows.len(), pruned_round.pruned_proof.sorted_indices.len());

        for query_bits in query_points {
            assert_eq!(query_bits.len(), current_codeword_log);
        }
        Self::assert_pruned_query_slot_mapping(
            builder,
            query_points,
            &pruned_round.query_to_unique_slot,
            &pruned_round.pruned_proof.sorted_indices,
            log_folding,
        );
        for opened_row in &pruned_round.opened_rows {
            assert_eq!(opened_row.len(), 1);
            assert_eq!(opened_row[0].len(), row_width);
        }

        let leaf_digests = pruned_round
            .opened_rows
            .iter()
            .map(|slot| {
                let felt_slice: Vec<Felt<C::F>> = slot[0]
                    .iter()
                    .flat_map(|value| C::ext2felt(builder, *value).to_vec())
                    .collect();
                SC::hash(builder, &felt_slice[..])
            })
            .collect::<Vec<_>>();

        Self::verify_batch_pruned(
            builder,
            commitment,
            log_row_height,
            leaf_digests,
            &pruned_round.pruned_proof,
            &[],
            &[],
        );

        pruned_round
            .query_to_unique_slot
            .iter()
            .map(|&slot| {
                assert!(slot < pruned_round.opened_rows.len());
                pruned_round.opened_rows[slot][0].clone()
            })
            .collect()
    }

    fn assert_pruned_query_slot_mapping(
        builder: &mut Builder<C>,
        query_points: &[Vec<C::Bit>],
        query_to_unique_slot: &[usize],
        sorted_indices: &[Felt<C::F>],
        skipped_low_bits: usize,
    ) {
        assert_eq!(query_to_unique_slot.len(), query_points.len());
        for (query_bits, &slot) in query_points.iter().zip(query_to_unique_slot.iter()) {
            assert!(slot < sorted_indices.len());
            assert!(skipped_low_bits <= query_bits.len());
            let opened_row_index =
                C::bits2num(builder, query_bits[skipped_low_bits..].iter().cloned());
            builder.assert_felt_eq(opened_row_index, sorted_indices[slot]);
        }
    }

    fn fold_whir_opened_iopp_row(
        builder: &mut Builder<C>,
        mut opened_row: Vec<EF<C>>,
        row_index_bits: &[C::Bit],
        log_folding: usize,
        log_row_height: usize,
        folding_challenges: &[EF<C>],
    ) -> EF<C> {
        assert_eq!(opened_row.len(), 1usize << log_folding);
        assert_eq!(row_index_bits.len(), log_row_height);
        assert_eq!(folding_challenges.len(), log_folding);
        for local_round in 0..log_folding {
            opened_row = Self::fold_opened_row_block(
                builder,
                &opened_row,
                row_index_bits,
                log_folding - local_round,
                log_row_height,
                folding_challenges[local_round],
            );
        }
        opened_row[0]
    }

    fn whir_eq_eval(builder: &mut Builder<C>, left: EF<C>, right: EF<C>) -> EF<C> {
        let one: EF<C> = builder.constant(C::EF::one());
        builder.eval(left * right + (one - left) * (one - right))
    }

    fn fold_whir_symbolic_weight_terms(
        builder: &mut Builder<C>,
        terms: &mut [(EF<C>, Vec<EF<C>>)],
        alpha: EF<C>,
    ) {
        for (coeff, point) in terms.iter_mut() {
            let point_value = point.pop().expect("WHIR symbolic point underflow");
            let eq = Self::whir_eq_eval(builder, point_value, alpha);
            *coeff = builder.eval(*coeff * eq);
        }
    }

    fn whir_pow2_ext_point(builder: &mut Builder<C>, z: EF<C>, len: usize) -> Vec<EF<C>> {
        let mut powers = Vec::with_capacity(len);
        let mut cur = z;
        for _ in 0..len {
            powers.push(cur);
            cur = builder.eval(cur * cur);
        }
        powers
    }

    fn whir_codeword_query_point(
        builder: &mut Builder<C>,
        row_index_bits: &[C::Bit],
        mle_vars: usize,
        log_blowup: usize,
    ) -> Vec<EF<C>> {
        if mle_vars == 0 {
            return Vec::new();
        }
        assert!(log_blowup > 0, "WHIR recursion currently expects nonzero blowup");
        let z = if row_index_bits.is_empty() {
            builder.constant(C::EF::one())
        } else {
            let generator: EF<C> =
                builder.constant(C::EF::two_adic_generator(row_index_bits.len()));
            C::exp_reverse_bits_ext(builder, generator, row_index_bits.to_vec())
        };
        let mut point = Self::whir_pow2_ext_point(builder, z, mle_vars);
        point.reverse();
        point
    }

    fn evaluate_mle_evals_at_point(
        builder: &mut Builder<C>,
        evals: &[EF<C>],
        point: &[EF<C>],
    ) -> EF<C> {
        assert_eq!(evals.len(), 1usize << point.len());
        let mut folded = evals.to_vec();
        for alpha in point.iter().rev() {
            let half = folded.len() / 2;
            for idx in 0..half {
                let even = folded[2 * idx];
                let odd = folded[2 * idx + 1];
                folded[idx] = builder.eval(even + *alpha * (odd - even));
            }
            folded.truncate(half);
        }
        folded[0]
    }

    fn whir_symbolic_final_accumulator(
        builder: &mut Builder<C>,
        final_poly: &[EF<C>],
        terms: &[(EF<C>, Vec<EF<C>>)],
    ) -> EF<C> {
        let mut acc: EF<C> = builder.constant(C::EF::zero());
        for (coeff, point) in terms {
            let value = Self::evaluate_mle_evals_at_point(builder, final_poly, point);
            acc = builder.eval(acc + *coeff * value);
        }
        acc
    }

    fn select_ext_by_bits(builder: &mut Builder<C>, values: &[EF<C>], bits: &[C::Bit]) -> EF<C> {
        assert_eq!(values.len(), 1 << bits.len());
        let mut layer = values.to_vec();
        let mut len = layer.len();
        for bit in bits {
            for i in 0..(len / 2) {
                layer[i] =
                    C::select_chain_ef(builder, *bit, once(layer[2 * i]), once(layer[2 * i + 1]))
                        [0];
            }
            len /= 2;
        }
        layer[0]
    }

    fn fold_opened_row_block(
        builder: &mut Builder<C>,
        block: &[EF<C>],
        row_index_bits: &[C::Bit],
        remaining_log_folding: usize,
        log_row_height: usize,
        beta: EF<C>,
    ) -> Vec<EF<C>> {
        assert!(remaining_log_folding > 0);
        assert_eq!(block.len(), 1usize << remaining_log_folding);
        assert_eq!(row_index_bits.len(), log_row_height);

        let pair_count = block.len() / 2;
        let log_current = log_row_height + remaining_log_folding;
        let generator_value = C::EF::two_adic_generator(log_current);
        let generator: EF<C> = builder.constant(generator_value);
        let row_factor: EF<C> = if row_index_bits.is_empty() {
            builder.constant(C::EF::one())
        } else {
            C::exp_reverse_bits_ext(builder, generator, row_index_bits.to_vec())
        };

        (0..pair_count)
            .map(|i| {
                let local_exp = reverse_bits_len(i, remaining_log_folding - 1) << log_row_height;
                let local_factor: EF<C> =
                    builder.constant(generator_value.exp_u64(local_exp as u64));
                let g1: EF<C> = builder.eval(row_factor * local_factor);
                let g2: EF<C> = builder.eval(-g1);
                let k = (block[2 * i + 1] - block[2 * i]) / (g2 - g1);
                let b = block[2 * i] - k * g1;
                builder.eval(b + k * beta)
            })
            .collect()
    }

    fn fold_codeword(builder: &mut Builder<C>, codeword: &[EF<C>], beta: EF<C>) -> Vec<EF<C>> {
        let n = codeword.len();
        assert!(n >= 2 && n.is_power_of_two());
        let half = n / 2;
        let log_n = log2_strict_usize(n);
        let g_inv = C::EF::two_adic_generator(log_n).inverse();
        let one_half_value = C::EF::two().inverse();
        let one_half: EF<C> = builder.constant(one_half_value);
        let half_beta: EF<C> = builder.eval(beta * one_half);

        (0..half)
            .map(|i| {
                let power_value =
                    g_inv.exp_u64(reverse_bits_len(i, half.trailing_zeros() as usize) as u64);
                let power_const: EF<C> = builder.constant(power_value);
                let power: EF<C> = builder.eval(power_const * half_beta);
                let r0 = codeword[2 * i];
                let r1 = codeword[2 * i + 1];
                builder.eval((one_half + power) * r0 + (one_half - power) * r1)
            })
            .collect()
    }
}

#[cfg(all(test, not(feature = "koalabear")))]
mod tests {
    use super::*;
    use crate::witness::{WitnessBlock, Witnessable};
    use dt_core_machine::{
        shape::{chip_log_height_threshold, num_skip_rounds},
        utils::setup_logger,
    };
    use dt_recursion_compiler::{config::InnerConfig, ir::Builder};
    use dt_stark::InnerChallenge;

    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
    use std::{collections::BTreeMap, process::exit};
    use tracing_subscriber::field::debug;

    use crate::{challenger::DuplexChallengerVariable, utils::sc_tests::run_test_recursion};
    use dt_stark::baby_bear_poseidon2::{
        my_perm, ChallengeMmcs, Challenger, Dft, MyCompress, MyHash, ValMmcs,
    };
    use log::debug;
    use p3_fri::FriConfig;
    use p3_matrix::{compressed::CompressedMatrix, dense::RowMajorMatrix, Dimensions, Matrix};
    use pcs::{
        basefold::{basefold_pcs::BaseFoldPcs, mlpcs::MlPCS},
        utils::mlpoly::{MultilinearExtension, MultilinearPolynomial},
    };
    use rand::{distributions::Uniform, Rng};
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type C = InnerConfig;

    fn get_col(mat: &RowMajorMatrix<F>, col: usize) -> Vec<F> {
        (0..mat.height()).map(|row| mat.get(row, col)).collect()
    }

    fn get_ef_col(mat: &RowMajorMatrix<EF>, col: usize) -> Vec<EF> {
        (0..mat.height()).map(|row| mat.get(row, col)).collect()
    }

    fn generate_vectors() -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        let range_len = Uniform::from(1..=5); // 子向量长度范围
        let lengths: [usize; 3] =
            [rng.sample(range_len), rng.sample(range_len), rng.sample(range_len)];

        let mut vec1 = Vec::with_capacity(3);
        let mut vec2 = Vec::with_capacity(3);

        for i in 0..3 {
            let len = lengths[i];
            let base_range = Uniform::from(1..=4);
            let mut v1 = Vec::with_capacity(len);
            let mut v2 = Vec::with_capacity(len);

            for _ in 0..len {
                v1.push(rng.sample(base_range) * 2);
                v2.push(rng.sample(base_range) * 4);
            }

            if i != 2 {
                for j in 0..len {
                    v1[j] += rng.sample(Uniform::from(1..=3));
                    v2[j] += rng.sample(Uniform::from(1..=10));
                }
            }

            vec1.push(v1);
            vec2.push(v2);
        }
        (vec1, vec2)
    }

    fn test_pcs_verify(log_heights: Vec<Vec<usize>>, widths: Vec<Vec<usize>>) {
        let batch_size = log_heights.len();
        let num_vars =
            log_heights.iter().flat_map(|batch| batch.iter()).copied().max().unwrap_or(0);

        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft::default();
        let mut challenger = Challenger::new(perm.clone());
        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 90,
            grinding_bits_query: 10,
            grinding_bits_batching: 10,
            grinding_bits_folding: 0,
            log_final_poly_len: 0,
            cross_round_log_foldings: Vec::new(),
            num_committed_groups: None,
            mmcs: challenge_mmcs.clone(),
        };
        type Pcs = BaseFoldPcs<F, ValMmcs, ChallengeMmcs, EF, Challenger>;
        let pcs = Pcs::new(val_mmcs, fri_config);

        // Generate random matrices
        let matrices: Vec<Vec<RowMajorMatrix<F>>> = widths
            .iter()
            .enumerate()
            .map(|(batch_idx, widths_in_batch)| {
                widths_in_batch
                    .iter()
                    .enumerate()
                    .map(|(i, &width)| {
                        let height = 1 << log_heights[batch_idx][i];
                        let vec: Vec<F> = (0..width * height).map(|_| rand::random()).collect();
                        RowMajorMatrix::<F>::new(vec, width)
                    })
                    .collect()
            })
            .collect();

        // Build EF matrices for the last batch
        let ef_matrices: Vec<RowMajorMatrix<EF>> = {
            let last_batch = &matrices[batch_size - 1];
            last_batch
                .iter()
                .map(|mat| {
                    let height = mat.height();
                    let ef_width = mat.width() / 4;
                    let mut ef_values = Vec::with_capacity(height * ef_width);
                    for row in 0..height {
                        for ef_col in 0..ef_width {
                            let idx = row * mat.width() + ef_col * 4;
                            let mut arr = [F::zero(); 4];
                            for d in 0..4 {
                                arr[d] = mat.values[idx + d];
                            }
                            ef_values.push(EF::from_base_slice(&arr));
                        }
                    }
                    RowMajorMatrix::<EF>::new(ef_values, ef_width)
                })
                .collect()
        };

        let max_log_height = log_heights.iter().flat_map(|v| v.iter()).max().copied().unwrap();
        let opening_point: Vec<EF> = (0..max_log_height).map(|_| rand::random()).collect();

        // Compute opened_values (no shifts, just direct evaluations)
        let opened_values: Vec<Vec<Vec<EF>>> = {
            let mut result: Vec<Vec<Vec<EF>>> = matrices
                .iter()
                .take(batch_size - 1)
                .enumerate()
                .map(|(batch_idx, batch)| {
                    batch
                        .iter()
                        .enumerate()
                        .map(|(mat_idx, matrix)| {
                            (0..matrix.width())
                                .map(|j| {
                                    let poly = get_col(matrix, j);
                                    let poly = MultilinearPolynomial::from_evals(poly);
                                    let log_h = log_heights[batch_idx][mat_idx];
                                    poly.evaluate_mix(&opening_point[..log_h])
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            // Last batch: EF matrices
            let ef_batch: Vec<Vec<EF>> = ef_matrices
                .iter()
                .enumerate()
                .map(|(mat_idx, matrix)| {
                    (0..matrix.width())
                        .map(|j| {
                            let poly = get_ef_col(matrix, j);
                            let poly = MultilinearPolynomial::from_evals(poly);
                            let log_h = log_heights[batch_size - 1][mat_idx];
                            poly.evaluate_mix(&opening_point[..log_h])
                        })
                        .collect()
                })
                .collect();
            result.push(ef_batch);
            result
        };

        // Convert to CompressedMatrix for the PCS API
        let compressed_matrices: Vec<Vec<CompressedMatrix<F>>> = matrices
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|mat| CompressedMatrix::from_full_matrix_no_padding(mat.clone()))
                    .collect()
            })
            .collect();

        let (com, prover_data): (Vec<_>, Vec<_>) =
            (0..batch_size).map(|i| pcs.commit(compressed_matrices[i].iter().collect())).unzip();

        let proof = pcs
            .open(compressed_matrices, prover_data, &opening_point, &opened_values, &mut challenger)
            .unwrap();

        // Build circuit
        let mut builder = Builder::<InnerConfig>::default();
        let proof_variable = proof.read(&mut builder);
        let commit_variable = com.read(&mut builder);
        let opening_point_variable = Vec::<InnerChallenge>::read(&opening_point, &mut builder);
        let opened_values_variable = opened_values.read(&mut builder);
        let mut challenger2 = DuplexChallengerVariable::new(&mut builder);
        let fri_config2 = FriConfig {
            log_blowup: 1,
            num_queries: 90,
            grinding_bits_query: 10,
            grinding_bits_batching: 10,
            grinding_bits_folding: 0,
            log_final_poly_len: 0,
            cross_round_log_foldings: Vec::new(),
            num_committed_groups: None,
            mmcs: challenge_mmcs.clone(),
        };

        let dims: Vec<Vec<Dimensions>> = widths
            .iter()
            .zip(log_heights.iter())
            .map(|(ws, hs)| {
                ws.iter()
                    .zip(hs.iter())
                    .map(|(&w, &log_h)| Dimensions { width: w, height: 1 << log_h })
                    .collect()
            })
            .collect();

        PcsVerifyTools::verify_basefold_pcs(
            &mut builder,
            &fri_config2,
            commit_variable,
            &dims,
            &opening_point_variable,
            &opened_values_variable,
            &proof_variable,
            &mut challenger2,
        );

        let mut witness_stream = Vec::<WitnessBlock<C>>::new();
        Witnessable::<C>::write(&proof, &mut witness_stream);
        Witnessable::<C>::write(&com, &mut witness_stream);
        Witnessable::<C>::write(&opening_point, &mut witness_stream);
        Witnessable::<C>::write(&opened_values, &mut witness_stream);
        run_test_recursion(
            builder.into_root_block(),
            witness_stream,
            num_skip_rounds(),
            chip_log_height_threshold(),
        );
    }

    #[test]
    fn multi_pcs_verify_tests() {
        setup_logger();
        for i in 0..2 {
            let (log_heights, widths) = generate_vectors();
            test_pcs_verify(log_heights, widths);
            println!("test {} passed!", i);
        }
    }
}
