use core::{borrow::Borrow, fmt};
use std::collections::{BTreeMap, BTreeSet};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir},
    septic_curve_params::{compute_beta_septix, KoalaBearCurveParams},
    sumcheck::{config::SCStarkGenericConfig, keys::SCStarkProvingKey, trace::CompressedMatrix},
    MachineRecord, StarkGenericConfig,
};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
use p3_matrix::{
    compressed::{padding_row_to_base_vec, padding_row_to_challenge_vec},
    dense::RowMajorMatrixView,
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use polyair::{
    permutation::generate_permutation_trace_,
    precompute::{collect_reserved_poly, precompute_linear_combination},
};

use crate::{
    config::{D_EF, EF, F, POSEIDON2_WIDTH},
    constraint_replay_dt::{ConstraintBetaLadderCols, ConstraintDagEvalCols, ConstraintFoldCols},
    interaction::{validate_recursion_interaction_budget, RecursionInteractionBudget},
    interaction_registry_dt::validate_recursion_interaction_registry,
    machine_dt::{NativeProverFor, NativeRecursionAir},
    native_air_dt::NativeAirFamily,
    proof_shape_dt::ProofShapeBinderCols,
    statement_hash_air_dt::{
        statement_hash_poseidon2_inputs, statement_hash_rows_cached, StatementDigestMode,
    },
    system_dt::{RecursionPoseidon2Pool, RecursionRecord},
    transcript_dt::{
        merkle_path::trace::merkle_row_iter, poseidon2::Poseidon2PermuteTraceGenerator,
        sponge::trace::transcript_sponge_rows_cached,
    },
    whir_dt::{
        columns::{
            WhirQueryFoldPackedCols, WhirQueryFoldReservedCols,
            NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS, NUM_WHIR_QUERY_FOLD_PACKED_COLS,
        },
        trace::whir_round_rows,
        WhirBatchEvalCols, WhirLeafStreamCols,
    },
    Instant,
};
use polyair::prover::SCMachineProver;

pub const NATIVE_RECURSION_ALLOWED_RANGE_BITS: [usize; 2] = [8, 21];
const ASSERT_PROVIDER_POOL_EQ_ENV: &str = "DT_NATIVE_RECURSION_ASSERT_PROVIDER_POOL_EQ";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRecursionValidationError {
    message: String,
}

impl NativeRecursionValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for NativeRecursionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NativeRecursionValidationError {}

pub type NativeRecursionValidationResult<T> = Result<T, NativeRecursionValidationError>;

macro_rules! validation_bail {
    ($($arg:tt)*) => {
        return Err($crate::validate::NativeRecursionValidationError::new(format!($($arg)*)))
    };
}

macro_rules! validation_ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            validation_bail!($($arg)*);
        }
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRecursionPoolStats {
    pub poseidon2_unique: usize,
    pub poseidon2_total: usize,
    pub poseidon2_height: usize,
    pub range_bits: BTreeSet<usize>,
    pub pow_unique: usize,
}

/// Bound for configs the record-side validators accept: the machine's PCS/
/// transcript config is irrelevant to record preparation, only the concrete
/// field pinning (`Val = F`, `Challenge = EF`) matters. Both the Poseidon2
/// `SC` (lift/L2/L3) and the SHA256 `RootSC` (L4) satisfy it.
pub trait NativeValidateConfig:
    SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>
{
}
impl<C> NativeValidateConfig for C where
    C: SCStarkGenericConfig + StarkGenericConfig<Val = F, Challenge = EF>
{
}

fn build_direct_provider_poseidon2_pool(
    record: &RecursionRecord,
    statement_hash_mode: StatementDigestMode,
) -> RecursionPoseidon2Pool {
    let mut aggregate = RecursionPoseidon2Pool::default();
    for row in merkle_row_iter(record) {
        aggregate.record_poseidon2(row.input);
    }
    for row in whir_round_rows(record) {
        let mult = row.final_root_poseidon2_recv_mult;
        if row.is_final_perm && mult != 0 {
            aggregate.record_poseidon2_count(row.final_root_poseidon2_input, mult);
        }
    }
    if record.proof_records.iter().all(|proof| !proof.transcript.sponge_blocks.is_empty()) {
        for row in transcript_sponge_rows_cached(record).iter() {
            aggregate.record_poseidon2(row.input16);
        }
    }
    for row in statement_hash_rows_cached(record, statement_hash_mode).iter() {
        aggregate.record_poseidon2(row.perm_input);
    }
    aggregate
}

/// Complete source-time provider registration after the statement has reached its final form.
/// Merkle, WHIR-final-root, and transcript requests are already captured while their source
/// records are built; statement hashing is the only record-wide source and is registered here.
pub fn finalize_provider_requests_at_source(
    record: &mut RecursionRecord,
    statement_hash_mode: StatementDigestMode,
) {
    assert!(
        !record.provider_requests_finalized,
        "provider requests finalized more than once for one record generation"
    );
    for input in statement_hash_poseidon2_inputs(record, statement_hash_mode) {
        record.poseidon2.record_poseidon2(input);
    }
    if should_assert_provider_pool_eq(statement_hash_mode) {
        let rebuilt = build_direct_provider_poseidon2_pool(record, statement_hash_mode);
        assert_provider_poseidon2_pool_eq(&record.poseidon2, &rebuilt);
    }
    record.mark_provider_requests_finalized();
}

fn should_assert_provider_pool_eq(_statement_hash_mode: StatementDigestMode) -> bool {
    match crate::env_var(ASSERT_PROVIDER_POOL_EQ_ENV) {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

fn assert_provider_poseidon2_pool_eq(
    left: &RecursionPoseidon2Pool,
    right: &RecursionPoseidon2Pool,
) {
    let left_canonical = canonical_poseidon2_requests(left);
    let right_canonical = canonical_poseidon2_requests(right);
    if left_canonical != right_canonical {
        let first_diff = left_canonical
            .iter()
            .zip(right_canonical.iter())
            .position(|(left, right)| left != right)
            .or_else(|| {
                if left_canonical.len() != right_canonical.len() {
                    Some(left_canonical.len().min(right_canonical.len()))
                } else {
                    None
                }
            });
        panic!(
            "provider-pool Poseidon2 multiset mismatch: left_unique={} left_total={} right_unique={} right_total={} first_diff_index={:?} left_at_diff={:?} right_at_diff={:?}",
            left.unique_count(),
            left.total_count_usize(),
            right.unique_count(),
            right.total_count_usize(),
            first_diff,
            first_diff.and_then(|idx| left_canonical.get(idx)),
            first_diff.and_then(|idx| right_canonical.get(idx))
        );
    }
    eprintln!(
        "provider-pool multiset dev gate passed: unique={} total={}",
        left.unique_count(),
        left.total_count_usize()
    );
}

#[cfg(test)]
pub(crate) fn assert_provider_requests_match_sources_for_test(
    record: &RecursionRecord,
    statement_hash_mode: StatementDigestMode,
) {
    let rebuilt = build_direct_provider_poseidon2_pool(record, statement_hash_mode);
    assert_provider_poseidon2_pool_eq(&record.poseidon2, &rebuilt);
}

fn canonical_poseidon2_requests(
    pool: &RecursionPoseidon2Pool,
) -> Vec<([u32; POSEIDON2_WIDTH], u32)> {
    let mut requests = pool
        .requests()
        .map(|request| (request.input.map(u32_value), request.count))
        .collect::<Vec<_>>();
    requests.sort_unstable();
    requests
}

pub fn check_provider_pools(
    record: &RecursionRecord,
) -> NativeRecursionValidationResult<NativeRecursionPoolStats> {
    let poseidon2_height = Poseidon2PermuteTraceGenerator::trace_height(&record.poseidon2);
    validation_ensure!(record.poseidon2.unique_count() > 0, "poseidon2 pool is empty");

    let range_bits =
        record.range.requests().map(|request| request.max_bits).collect::<BTreeSet<_>>();
    let allowed = NATIVE_RECURSION_ALLOWED_RANGE_BITS.iter().copied().collect::<BTreeSet<_>>();
    validation_ensure!(
        range_bits.is_subset(&allowed),
        "range pool contains unsupported widths: actual={range_bits:?} allowed={allowed:?}"
    );
    validation_ensure!(record.pow.unique_count() == 0, "pow pool must be empty");

    Ok(NativeRecursionPoolStats {
        poseidon2_unique: record.poseidon2.unique_count(),
        poseidon2_total: record.poseidon2.total_count_usize(),
        poseidon2_height,
        range_bits,
        pow_unique: record.pow.unique_count(),
    })
}

/// One chip's admitted trace shape: the exact height and width every later
/// stage must realize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedChipTrace {
    pub chip: String,
    pub total_height: usize,
    pub width: usize,
}

/// The exact-gate output: per-chip planned shapes for one admitted record.
/// Produced only by [`exact_pre_trace_gate`]; the single planned-shape
/// authority realized traces are asserted against.
#[derive(Debug)]
pub struct ExactTracePlan {
    pub chips: Vec<PlannedChipTrace>,
    /// Parallel admission of row counts from already-owned workspace artifacts/events.
    pub row_count_admission_ms: u128,
    /// Ordered capacity/interaction fold after all row counts are available.
    pub plan_fold_ms: u128,
}

/// One device-derived active-row claim admitted against the CPU-owned static
/// chip schema. The exact gate alone converts it to the symbolic power-of-two
/// domain used by the downstream plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactTraceActiveRowOverride {
    pub chip: String,
    pub active_rows: usize,
}

/// S4b/S6 exact pre-trace gate (V3 §8.7/§9): derive every included chip's
/// planned total height from the shared `num_rows` authority and run the
/// capacity checks — the interaction budget and the pinned stack height —
/// BEFORE any trace matrix is allocated. Rejection therefore costs zero
/// trace cells. The returned [`ExactTracePlan`] is what
/// [`check_traces_match_plan`] asserts the realized traces against.
pub fn exact_pre_trace_gate<C: NativeValidateConfig, PROV>(
    prover: &PROV,
    record: &RecursionRecord,
    stack_log_height: Option<usize>,
) -> NativeRecursionValidationResult<ExactTracePlan>
where
    PROV: SCMachineProver<C, crate::machine_dt::NativeRecursionAir, D_EF>,
{
    validate_recursion_interaction_registry();
    let chips = prover.machine().shard_chips(record).collect::<Vec<_>>();
    let warmup_start = Instant::now();
    let override_map: BTreeMap<&str, usize> = BTreeMap::new();
    let admission_start = Instant::now();
    let measured = chips
        .par_iter()
        .map(|chip| {
            let chip_name = chip.name();
            let rows = if let Some(&active_rows) = override_map.get(chip_name.as_str()) {
                active_rows.max(1).next_power_of_two()
            } else {
                chip.num_rows(record).ok_or_else(|| {
                    NativeRecursionValidationError::new(format!(
                        "chip {} has no num_rows authority; every native chip must plan its \
                         height before allocation",
                        chip.name()
                    ))
                })?
            };
            Ok((
                chip_name,
                rows,
                chip.symbolic_builder.main.len(),
                chip.symbolic_builder.lookup_infos.iter().filter(|lookup| lookup.is_send).count(),
                chip.symbolic_builder.lookup_infos.len(),
            ))
        })
        .collect::<NativeRecursionValidationResult<Vec<_>>>()?;
    let row_count_admission_ms = admission_start.elapsed().as_millis();

    let fold_start = Instant::now();
    let mut budgets = Vec::new();
    let mut planned = Vec::new();
    let mut tallest = 0usize;
    for (name, rows, width, sends, lookups) in measured {
        validation_ensure!(
            rows.is_power_of_two(),
            "planned height for {} is not padded: {rows}",
            name
        );
        tallest = tallest.max(rows.next_power_of_two().trailing_zeros() as usize);
        let receives = lookups - sends;
        budgets.push(RecursionInteractionBudget::new(sends, receives, budget_log_height(rows)));
        planned.push(PlannedChipTrace { chip: name, total_height: rows, width });
    }
    if let Some(stack_h) = stack_log_height {
        validation_ensure!(
            tallest <= stack_h,
            "planned tallest main matrix 2^{tallest} exceeds the pinned stack_log_height              H = {stack_h}; the H freeze must be reopened (R-M2-5)"
        );
    }
    validate_recursion_interaction_budget::<F>(budgets).map_err(|err| {
        NativeRecursionValidationError::new(format!(
            "pre-trace interaction budget validation failed: {err:?}"
        ))
    })?;
    Ok(ExactTracePlan {
        chips: planned,
        row_count_admission_ms,
        plan_fold_ms: fold_start.elapsed().as_millis(),
    })
}

/// Planned == realized, per chip and bidirectional: every generated trace
/// sits exactly on its planned (height, width), and every planned chip was
/// generated. The plan is the only shape authority here — nothing is
/// re-derived from the record.
pub fn check_traces_match_plan(
    plan: &ExactTracePlan,
    traces: &[(String, CompressedMatrix<F>)],
) -> NativeRecursionValidationResult<()> {
    validation_ensure!(
        plan.chips.len() == traces.len(),
        "trace count mismatch: planned {} chips, generated {}",
        plan.chips.len(),
        traces.len()
    );
    for (name, trace) in traces {
        let planned = plan.chips.iter().find(|chip| chip.chip == *name).ok_or_else(|| {
            NativeRecursionValidationError::new(format!("trace for unplanned chip {name}"))
        })?;
        validation_ensure!(
            trace.total_height == planned.total_height,
            "total height mismatch for {name}: trace={} planned={}",
            trace.total_height,
            planned.total_height
        );
        validation_ensure!(
            trace.main.width() == planned.width,
            "trace width mismatch for {name}: trace={} planned={}",
            trace.main.width(),
            planned.width
        );
        validation_ensure!(
            trace.stored_height() <= trace.total_height,
            "stored height exceeds total height for {name}: stored={} total={}",
            trace.stored_height(),
            trace.total_height
        );
    }
    Ok(())
}

pub fn check_real_trace_constraints<C: NativeValidateConfig>(
    prover: &NativeProverFor<C>,
    pk: &SCStarkProvingKey<C>,
    record: &RecursionRecord,
    traces: &[(String, CompressedMatrix<F>)],
) -> NativeRecursionValidationResult<()> {
    let perm_alpha = EF::from_base(F::from_canonical_u32(7));
    let perm_beta = EF::from_base(F::from_canonical_u32(13));
    let max_beta_power =
        prover.machine.chips.iter().map(|chip| chip.required_max_beta_power()).max().unwrap_or(0);
    let beta_powers = {
        let mut powers = perm_beta.powers();
        (0..=max_beta_power).map(|_| powers.next().expect("infinite powers")).collect::<Vec<_>>()
    };
    let beta_septix = compute_beta_septix::<F, EF, KoalaBearCurveParams>(perm_beta);
    let public_values = record.public_values::<F>();
    let chip_names = traces.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    let preprocessed = pk.get_preprocessed_compressed_for_chips(&chip_names);

    for (idx, (name, trace)) in traces.iter().enumerate() {
        let chip =
            prover.machine.chips.iter().find(|chip| chip.name() == *name).ok_or_else(|| {
                NativeRecursionValidationError::new(format!("trace for unknown chip {name}"))
            })?;
        let precompute = precompute_linear_combination(
            &chip.air,
            preprocessed[idx],
            trace,
            &public_values,
            perm_alpha,
            &beta_powers,
            beta_septix,
            chip.num_precompute(),
        );
        let reserved = collect_reserved_poly(preprocessed[idx], trace, chip.reserved_poly());
        let (permutation, _) = generate_permutation_trace_(
            &chip.air,
            &reserved,
            &precompute,
            perm_alpha,
            &beta_powers,
            chip.logup_batch_size(),
            chip.num_lookup(),
        );
        for row_idx in 0..reserved.stored_height() {
            let pre_row = precompute.main.row_slice(row_idx);
            let reserved_row = reserved.main.row_slice(row_idx);
            let perm_row = permutation.main.row_slice(row_idx);
            let is_first_row = if row_idx == 0 { F::one() } else { F::zero() };
            let is_last_row =
                if row_idx == reserved.total_height - 1 { F::one() } else { F::zero() };
            check_one_row(
                &chip.air,
                name,
                row_idx,
                &public_values,
                RowMajorMatrixView::new_row(&pre_row),
                RowMajorMatrixView::new_row(&reserved_row),
                RowMajorMatrixView::new_row(&perm_row),
                perm_alpha,
                &beta_powers,
                beta_septix,
                chip.logup_batch_size(),
                is_first_row,
                is_last_row,
            )?;
        }
        if reserved.total_height > reserved.stored_height() {
            let pre_pad = padding_row_to_challenge_vec(&precompute.padding_row);
            let reserved_pad = padding_row_to_base_vec(&reserved.padding_row);
            let perm_pad = padding_row_to_challenge_vec(&permutation.padding_row);
            check_one_row(
                &chip.air,
                name,
                reserved.stored_height(),
                &public_values,
                RowMajorMatrixView::new_row(&pre_pad),
                RowMajorMatrixView::new_row(&reserved_pad),
                RowMajorMatrixView::new_row(&perm_pad),
                perm_alpha,
                &beta_powers,
                beta_septix,
                chip.logup_batch_size(),
                F::zero(),
                F::one(),
            )?;
        }
    }
    Ok(())
}

pub fn check_lookup_residuals<C: NativeValidateConfig>(
    prover: &NativeProverFor<C>,
    pk: &SCStarkProvingKey<C>,
    record: &RecursionRecord,
    traces: &[(String, CompressedMatrix<F>)],
) -> NativeRecursionValidationResult<()> {
    let max_beta_power =
        prover.machine.chips.iter().map(|chip| chip.required_max_beta_power()).max().unwrap_or(0);
    let public_values = record.public_values::<F>();
    let chip_names = traces.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    let preprocessed = pk.get_preprocessed_compressed_for_chips(&chip_names);
    let challenge_pairs = [(7, 13), (3, 5), (12_345, 67_890), (314_159, 271_828)];

    for (alpha_raw, beta_raw) in challenge_pairs {
        let perm_alpha = EF::from_base(F::from_canonical_u32(alpha_raw));
        let perm_beta = EF::from_base(F::from_canonical_u32(beta_raw));
        let beta_powers = {
            let mut powers = perm_beta.powers();
            (0..=max_beta_power)
                .map(|_| powers.next().expect("infinite powers"))
                .collect::<Vec<_>>()
        };
        let beta_septix = compute_beta_septix::<F, EF, KoalaBearCurveParams>(perm_beta);
        let mut totals = BTreeMap::<String, F>::new();
        let mut contributors = BTreeMap::<String, Vec<String>>::new();

        for (idx, (name, trace)) in traces.iter().enumerate() {
            let chip =
                prover.machine.chips.iter().find(|chip| chip.name() == *name).ok_or_else(|| {
                    NativeRecursionValidationError::new(format!("trace for unknown chip {name}"))
                })?;
            let precompute = precompute_linear_combination(
                &chip.air,
                preprocessed[idx],
                trace,
                &public_values,
                perm_alpha,
                &beta_powers,
                beta_septix,
                chip.num_precompute(),
            );
            let reserved = collect_reserved_poly(preprocessed[idx], trace, chip.reserved_poly());
            for row_idx in 0..reserved.stored_height() {
                let pre_row = precompute.main.row_slice(row_idx);
                let reserved_row = reserved.main.row_slice(row_idx);
                collect_lookup_row_residual(
                    &chip.air,
                    name,
                    row_idx,
                    1,
                    RowMajorMatrixView::new_row(pre_row.as_ref()),
                    RowMajorMatrixView::new_row(reserved_row.as_ref()),
                    &public_values,
                    perm_alpha,
                    &beta_powers,
                    beta_septix,
                    &mut totals,
                    &mut contributors,
                )?;
            }
            if reserved.total_height > reserved.stored_height() {
                let repeat = reserved.total_height - reserved.stored_height();
                let pre_pad = padding_row_to_challenge_vec(&precompute.padding_row);
                let reserved_pad = padding_row_to_base_vec(&reserved.padding_row);
                collect_lookup_row_residual(
                    &chip.air,
                    name,
                    reserved.stored_height(),
                    repeat,
                    RowMajorMatrixView::new_row(&pre_pad),
                    RowMajorMatrixView::new_row(&reserved_pad),
                    &public_values,
                    perm_alpha,
                    &beta_powers,
                    beta_septix,
                    &mut totals,
                    &mut contributors,
                )?;
            }
        }

        let residual = totals
            .iter()
            .filter(|(_, value)| **value != F::zero())
            .map(|(key, value)| {
                let sample = contributors
                    .get(key)
                    .into_iter()
                    .flatten()
                    .take(6)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{key}=>{:?} [{}]", value, sample)
            })
            .take(20)
            .collect::<Vec<_>>();
        validation_ensure!(
            residual.is_empty(),
            "G-S1e lookup residual mismatch for alpha={alpha_raw} beta={beta_raw}: {} residual keys, sample={}",
            totals.values().filter(|value| **value != F::zero()).count(),
            residual.join("; ")
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_lookup_row_residual(
    air: &NativeRecursionAir,
    chip_name: &str,
    row_idx: usize,
    repeat: usize,
    precomputed: RowMajorMatrixView<'_, EF>,
    reserved_poly: RowMajorMatrixView<'_, F>,
    public_values: &[F],
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    totals: &mut BTreeMap<String, F>,
    contributors: &mut BTreeMap<String, Vec<String>>,
) -> NativeRecursionValidationResult<()> {
    let empty_perm = Vec::<EF>::new();
    let denominator_binding = precomputed.row_slice(0);
    let denominator_values = denominator_binding.as_ref().to_vec();
    let mut folder = ExactConstraintFolder {
        air_family: air.family(),
        chip_name,
        row_idx,
        public_values,
        alpha,
        beta_powers,
        beta_septix,
        precomputed,
        reserved_poly,
        is_first_row: F::zero(),
        is_last_row: F::zero(),
        permutation: RowMajorMatrixView::new_row(&empty_perm),
        multiplicities: Vec::new(),
        batch_size: 1,
        constraint_index: 0,
        failures: Vec::new(),
    };
    air.lookup(&mut folder);
    validation_ensure!(
        folder.multiplicities.len() <= denominator_values.len(),
        "lookup count exceeds denominator count for {chip_name} row {row_idx}: lookups={} denominators={}",
        folder.multiplicities.len(),
        denominator_values.len()
    );
    let repeat_f = F::from_canonical_usize(repeat);
    for (lookup_idx, mult) in folder.multiplicities.iter().copied().enumerate() {
        if mult == F::zero() {
            continue;
        }
        let scaled = mult * repeat_f;
        let key = format!("{:?}", ef_limbs(denominator_values[lookup_idx]));
        *totals.entry(key.clone()).or_insert(F::zero()) += scaled;
        let context = lookup_context(air.family(), lookup_idx, &reserved_poly)
            .map(|context| format!(":{context}"))
            .unwrap_or_default();
        contributors.entry(key).or_default().push(format!(
            "{chip_name}:row{row_idx}:lookup{lookup_idx}:repeat{repeat}:mult={scaled:?}{context}"
        ));
    }
    Ok(())
}

fn lookup_context(
    family: NativeAirFamily,
    lookup_idx: usize,
    reserved_poly: &RowMajorMatrixView<'_, F>,
) -> Option<String> {
    let row_binding = reserved_poly.row_slice(0);
    match family {
        NativeAirFamily::ConstraintDagEval => {
            let local: &ConstraintDagEvalCols<F> = row_binding.as_ref().borrow();
            let leaf_kind = local
                .leaf_flags
                .iter()
                .enumerate()
                .find_map(|(idx, flag)| (*flag == F::one()).then_some(idx as u32))
                .unwrap_or(0);
            Some(format!(
                "dag proof={} chip={} static={} node={} lhs={} rhs={} third={} aux={} leaf_kind={} value={:?} fanout={} flags=[const:{} add:{} sub:{} mul:{} fused:{}]",
                u32_value(local.proof_idx),
                u32_value(local.chip_idx),
                u32_value(local.static_chip_id),
                u32_value(local.node_idx),
                u32_value(local.lhs_idx),
                u32_value(local.rhs_idx),
                u32_value(local.third_idx),
                u32_value(local.aux),
                leaf_kind,
                local.value,
                u32_value(local.fanout),
                u32_value(local.is_const),
                u32_value(local.is_add),
                u32_value(local.is_sub),
                u32_value(local.is_mul),
                u32_value(local.is_fused),
            ))
        }
        NativeAirFamily::ConstraintFold => {
            let local: &ConstraintFoldCols<F> = row_binding.as_ref().borrow();
            let slot = lookup_idx.checked_sub(6).filter(|slot| *slot < local.root_nodes.len());
            Some(format!(
                "fold proof={} remaining_chips={} static={} cursor={} local_ord={} root_ord={} kinds=[skip:{} gate:{} batch:{}] slot={:?} root_node={:?} root_value={:?} perm_value={:?}",
                u32_value(local.proof_idx),
                u32_value(local.remaining_chips),
                u32_value(local.static_chip_id),
                u32_value(local.cursor),
                u32_value(local.local_ord),
                u32_value(local.root_ord),
                u32_value(local.is_skip),
                u32_value(local.is_gate),
                u32_value(local.is_batch),
                slot,
                slot.map(|idx| u32_value(local.root_nodes[idx])),
                slot.map(|idx| local.root_values[idx]),
                local.perm_value,
            ))
        }
        NativeAirFamily::ConstraintBetaLadder => {
            let local: &ConstraintBetaLadderCols<F> = row_binding.as_ref().borrow();
            Some(format!(
                "beta_ladder proof={} power={} flags=[valid:{} seed:{} last:{}] serve={} recv={} alpha_serve={} septix_serve={} beta={:?} prev_or_alpha={:?} power={:?}",
                u32_value(local.proof_idx),
                u32_value(local.power_idx),
                u32_value(local.is_valid),
                u32_value(local.is_seed),
                u32_value(local.is_last),
                u32_value(local.serve_mult),
                u32_value(local.challenges_recv_mult),
                u32_value(local.alpha_serve_mult),
                u32_value(local.septix_serve_mult),
                local.beta,
                local.prev_power_or_alpha,
                local.power,
            ))
        }
        NativeAirFamily::WhirBatchEval => {
            let local: &WhirBatchEvalCols<F> = row_binding.as_ref().borrow();
            Some(format!(
                "whir proof={} batch={} batch_pos={} chip={} value_idx={} is_value={} opened_mult={} value={:?}",
                u32_value(local.proof_idx),
                u32_value(local.batch_id),
                u32_value(local.batch_pos),
                u32_value(local.chip_idx),
                u32_value(local.value_idx),
                u32_value(local.is_value),
                u32_value(local.opened_eval_send_mult),
                local.value,
            ))
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn check_one_row(
    air: &NativeRecursionAir,
    chip_name: &str,
    row_idx: usize,
    public_values: &[F],
    precomputed: RowMajorMatrixView<'_, EF>,
    reserved_poly: RowMajorMatrixView<'_, F>,
    permutation: RowMajorMatrixView<'_, EF>,
    alpha: EF,
    beta_powers: &[EF],
    beta_septix: EF,
    batch_size: usize,
    is_first_row: F,
    is_last_row: F,
) -> NativeRecursionValidationResult<()> {
    let mut folder = ExactConstraintFolder {
        air_family: air.family(),
        chip_name,
        row_idx,
        public_values,
        alpha,
        beta_powers,
        beta_septix,
        precomputed,
        reserved_poly,
        is_first_row,
        is_last_row,
        permutation,
        multiplicities: Vec::new(),
        batch_size,
        constraint_index: 0,
        failures: Vec::new(),
    };
    air.eval(&mut folder);
    air.lookup(&mut folder);
    folder.constrain_lookup();
    if !folder.failures.is_empty() {
        let context =
            folder.failure_context().map(|context| format!("; {context}")).unwrap_or_default();
        validation_bail!(
            "G-S1e constraint check failed for {chip_name} row {row_idx}: {}{}",
            folder.failures.into_iter().take(6).collect::<Vec<_>>().join("; "),
            context
        );
    }
    Ok(())
}

struct ExactConstraintFolder<'a> {
    air_family: NativeAirFamily,
    chip_name: &'a str,
    row_idx: usize,
    public_values: &'a [F],
    alpha: EF,
    beta_powers: &'a [EF],
    beta_septix: EF,
    precomputed: RowMajorMatrixView<'a, EF>,
    reserved_poly: RowMajorMatrixView<'a, F>,
    is_first_row: F,
    is_last_row: F,
    permutation: RowMajorMatrixView<'a, EF>,
    multiplicities: Vec<F>,
    batch_size: usize,
    constraint_index: usize,
    failures: Vec<String>,
}

impl ExactConstraintFolder<'_> {
    fn failure_context(&self) -> Option<String> {
        if self.air_family == NativeAirFamily::WhirLeafStream {
            let row_binding = self.reserved_poly.row_slice(0);
            let local: &WhirLeafStreamCols<F> = row_binding.as_ref().borrow();
            let mut expected_acc = if local.is_unit_start == F::one() {
                EF::zero()
            } else {
                EF::from_base_slice(&local.acc_in)
            };
            for slot in 0..8 {
                if local.chunk_mask[slot] == F::one() {
                    expected_acc += EF::from_base_slice(&local.slot_pows[slot]) *
                        EF::from_base(local.values[slot]);
                }
            }
            return Some(format!(
                "WhirLeafStream context is_unit_start={:?} is_unit_key_start={:?} is_unit_end={:?} log_height={:?} batch_id={:?} mask={:?} acc_in={:?} acc_out={:?} expected_acc={:?} pow_in={:?} pow_out={:?}",
                local.is_unit_start,
                local.is_unit_key_start,
                local.is_unit_end,
                local.log_height,
                local.batch_id,
                local.chunk_mask,
                local.acc_in,
                local.acc_out,
                ef_limbs(expected_acc),
                local.pow_in,
                local.pow_out,
            ));
        }
        if self.air_family == NativeAirFamily::ProofShapeBinder {
            let row_binding = self.reserved_poly.row_slice(0);
            let local: &ProofShapeBinderCols<F> = row_binding.as_ref().borrow();
            return Some(format!(
                "ProofShapeBinder context valid={:?} kinds=[vk:{:?} meta:{:?} pv:{:?} e1:{:?} chip:{:?} e5:{:?}] proof={:?} role={:?} chip_idx={:?} static_chip_id={:?} log_height={:?} prep_width={:?} main_width={:?} perm_width={:?} constraint_count={:?} gate_count={:?} has_prep={:?} prev_chip_idx={:?} prev_log_height={:?} prev_static_chip_id={:?} is_group_start={:?} range_val={:?} send_mults=[chip_meta:{:?} prep:{:?} perm:{:?} fold_plan:{:?} summary:{:?}]",
                local.is_valid,
                local.is_vk_commit,
                local.is_vk_meta,
                local.is_public_values,
                local.is_e1,
                local.is_chip,
                local.is_e5,
                local.proof_idx,
                local.role_id,
                local.chip_idx,
                local.static_chip_id,
                local.log_height,
                local.prep_width,
                local.main_width,
                local.perm_width,
                local.constraint_count,
                local.gate_count,
                local.has_prep,
                local.prev_chip_idx,
                local.prev_log_height,
                local.prev_static_chip_id,
                local.is_group_start,
                local.range_val,
                local.chip_meta_send_mult,
                local.batch_dim_prep_send_mult,
                local.batch_dim_perm_send_mult,
                local.fold_plan_source_mult,
                local.summary_send_mult,
            ));
        }
        if self.air_family != NativeAirFamily::WhirQueryFold {
            return None;
        }
        let row_binding = self.reserved_poly.row_slice(0);
        let local: &WhirQueryFoldReservedCols<F> = row_binding.as_ref().borrow();
        let precomputed_binding = self.precomputed.row_slice(0);
        let packed: &WhirQueryFoldPackedCols<EF> = precomputed_binding.as_ref()
            [NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS..
                NUM_WHIR_QUERY_FOLD_DENOMINATOR_COLS + NUM_WHIR_QUERY_FOLD_PACKED_COLS]
            .borrow();
        let x = EF::from_base(local.x);
        let f0 = packed.f0;
        let f1 = packed.f1;
        let r = packed.r_fold;
        let folded = packed.chain_send_folded;
        let lhs =
            EF::from_base(F::from_canonical_u32(2)) * x * folded - x * (f0 + f1) - r * (f0 - f1);
        Some(format!(
            "WhirQueryFold context is_seed={:?} is_round={:?} cursor={:?} idx={:?} idx_bit={:?} x={:?} chain_send_folded={:?} fold_lhs={:?}",
            local.is_seed,
            local.is_round,
            local.cursor,
            local.idx,
            local.idx_bit,
            local.x,
            ef_limbs(folded),
            ef_limbs(lhs),
        ))
    }

    fn constrain_lookup(&mut self) {
        let values_binding = self.precomputed.row_slice(0);
        let values: &[EF] = values_binding.as_ref();
        let perm_binding = self.permutation.row_slice(0);
        let perm_local: &[EF] = perm_binding.as_ref();
        if self.multiplicities.len() > values.len() {
            self.failures.push(format!(
                "lookup multiplicity count {} exceeds precomputed values {}",
                self.multiplicities.len(),
                values.len()
            ));
            return;
        }
        for (lookup_index, (value, multiplicity)) in values
            .chunks(self.batch_size)
            .zip(self.multiplicities.chunks(self.batch_size))
            .enumerate()
        {
            let denominator = value.iter().copied().product::<EF>();
            let mut numerator = EF::zero();
            for (i, m) in multiplicity.iter().copied().enumerate() {
                let mut all_but_current = EF::one();
                for other_rlc in
                    value.iter().enumerate().filter(|(j, _)| i != *j).map(|(_, rlc)| rlc)
                {
                    all_but_current *= *other_rlc;
                }
                numerator += all_but_current * m;
            }
            let expected =
                denominator * perm_local.get(lookup_index).copied().unwrap_or_else(EF::zero);
            if numerator != expected {
                self.failures.push(format!(
                    "lookup batch {lookup_index} mismatch numerator={numerator:?} denominator*perm={expected:?}"
                ));
            }
        }
    }
}

impl<'a> FullAirBuilder for ExactConstraintFolder<'a> {
    type F = F;
    type EF = EF;
    type VarBase = F;
    type VarMaybeExt = F;
    type VarExt = EF;
    type MatMaybeExt = RowMajorMatrixView<'a, F>;
    type MatExt = RowMajorMatrixView<'a, EF>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        unreachable!("exact checker evaluates from reserved/precomputed rows")
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        unreachable!("exact checker evaluates from reserved/precomputed rows")
    }

    fn public(&self) -> &[Self::VarBase] {
        self.public_values
    }

    fn alpha(&self) -> Self::VarExt {
        self.alpha
    }

    fn beta_powers(&self) -> &[Self::VarExt] {
        self.beta_powers
    }

    fn beta_septix(&self) -> Self::VarExt {
        self.beta_septix
    }

    fn retain_precomputed(&mut self, _x: Self::VarExt) {
        unreachable!("precompute is already materialized")
    }

    fn precomputed(&self) -> Self::MatExt {
        self.precomputed
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        self.reserved_poly
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        self.multiplicities.push(if is_send { multiplicity } else { -multiplicity });
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        self.is_last_row
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        F::one() - self.is_first_row - self.is_last_row
    }

    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        a * b
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        ef
    }

    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        let value = x.into();
        if value != F::zero() {
            self.failures.push(format!(
                "constraint {} base nonzero {value:?} ({} row {})",
                self.constraint_index, self.chip_name, self.row_idx
            ));
        }
        self.constraint_index += 1;
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        let value = x.into();
        if value != EF::zero() {
            self.failures.push(format!(
                "constraint {} ext nonzero {value:?} ({} row {})",
                self.constraint_index, self.chip_name, self.row_idx
            ));
        }
        self.constraint_index += 1;
    }
}

fn ef_limbs(value: EF) -> [F; D_EF] {
    let limbs = value.as_base_slice();
    core::array::from_fn(|idx| limbs[idx])
}

fn u32_value(value: F) -> u32 {
    value.as_canonical_u32()
}

fn log2_strict(value: usize) -> usize {
    assert!(value.is_power_of_two(), "value is not a power of two");
    value.trailing_zeros() as usize
}

fn budget_log_height(total_height: usize) -> usize {
    #[cfg(test)]
    {
        let override_value = BUDGET_LOG_HEIGHT_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst);
        if override_value != usize::MAX {
            return override_value;
        }
    }
    log2_strict(total_height)
}

#[cfg(test)]
static BUDGET_LOG_HEIGHT_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

#[cfg(test)]
pub(crate) fn set_budget_log_height_override_for_test(log_height: Option<usize>) {
    BUDGET_LOG_HEIGHT_OVERRIDE
        .store(log_height.unwrap_or(usize::MAX), std::sync::atomic::Ordering::SeqCst);
}
