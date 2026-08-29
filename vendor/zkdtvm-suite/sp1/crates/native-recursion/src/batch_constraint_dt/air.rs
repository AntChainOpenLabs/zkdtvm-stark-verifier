use core::{borrow::Borrow, ops::Deref};

use dt_stark::{
    air::{FullAir, FullAirBuilder, MachineAir, PairCol},
    sumcheck::trace::CompressedMatrix,
};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    batch_constraint_dt::{
        bus::{
            BatchOpeningPointBus, BatchSumcheckClaimChainBus, SumcheckOutBus, SUMCHECK_OUT_EQ,
            SUMCHECK_OUT_PERM_ALPHA, SUMCHECK_OUT_PERM_BETA,
        },
        columns::{
            batch_seed_prefix_limbs, batch_sumcheck_width, BatchSumcheckCols,
            BatchSumcheckPackedCols, BatchSumcheckReservedCols, BatchTranscriptInputsCols,
            BATCH_ACTIVE_SHAPE_ENTRY_LIMBS, BATCH_ACTIVE_SHAPE_HEADER_LIMBS,
            BATCH_COMMITMENT_LIMBS, BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS, BATCH_SUMCHECK_EVALS,
            BATCH_VK_TAG_V1, BATCH_VK_VERSION_V1, NUM_BATCH_SUMCHECK_PACKED_COLS,
            NUM_BATCH_TRANSCRIPT_INPUTS_COLS,
        },
        trace::{BatchSumcheckTraceGenerator, BatchTranscriptInputsTraceGenerator},
    },
    config::{D_EF, F},
    constraint_replay_dt::{ConstraintFoldChainBus, ConstraintFoldPlanChainBus},
    proof_shape_dt::ProofShapeSummaryBus,
    system_dt::{RecursionNativeProgram, RecursionRecord},
    transcript_dt::sponge::TranscriptEventBus,
};

const BATCH_TRANSCRIPT_INPUT_LOOKUP_COUNT: usize = 21;
const BATCH_SUMCHECK_LOOKUP_COUNT: usize = 40;

#[derive(Debug, Clone, Copy)]
pub struct BatchTranscriptInputsAir {
    pub num_public_values: usize,
    pub seed_prefix_limbs: usize,
    pub transcript_event_bus: TranscriptEventBus,
    pub sumcheck_out_bus: SumcheckOutBus,
    pub fold_chain_bus: ConstraintFoldChainBus,
    pub fold_plan_chain_bus: ConstraintFoldPlanChainBus,
}

impl BatchTranscriptInputsAir {
    pub const fn new(num_public_values: usize, contains_global_bus: bool) -> Self {
        Self {
            num_public_values,
            seed_prefix_limbs: batch_seed_prefix_limbs(contains_global_bus),
            transcript_event_bus: TranscriptEventBus::new(),
            sumcheck_out_bus: SumcheckOutBus::new(),
            fold_chain_bus: ConstraintFoldChainBus::new(),
            fold_plan_chain_bus: ConstraintFoldPlanChainBus::new(),
        }
    }
}

impl BaseAir<F> for BatchTranscriptInputsAir {
    fn width(&self) -> usize {
        NUM_BATCH_TRANSCRIPT_INPUTS_COLS
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for BatchTranscriptInputsAir {
    fn width(&self) -> usize {
        NUM_BATCH_TRANSCRIPT_INPUTS_COLS
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.transcript_event_bus.required_max_beta_power_floor(),
            self.sumcheck_out_bus.required_max_beta_power_floor(),
            self.fold_chain_bus.required_max_beta_power_floor(),
            self.fold_plan_chain_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![PairCol::Main(core::mem::offset_of!(BatchTranscriptInputsCols<u8>, is_valid))]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let denominators = {
            let main = builder.main();
            let local: &BatchTranscriptInputsCols<AB::VarMaybeExt> = main.borrow();
            input_denominators(self, builder, local)
        };
        debug_assert_eq!(denominators.len(), BATCH_TRANSCRIPT_INPUT_LOOKUP_COUNT);
        for denominator in denominators {
            builder.retain_precomputed(denominator);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local = reserved.row_slice(0);
        assert_bool(builder, local[0].clone());
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local = reserved.row_slice(0);
        let is_valid = local[0].clone();

        // VK tag/version, E3/E4(10), E7(5), authenticated c_chips, then perm-alpha,
        // perm-beta and the unique FoldChain alpha seed.
        builder.recv(is_valid.clone());
        builder.recv(is_valid.clone());
        for _ in 0..(3 * D_EF) {
            builder.recv(is_valid.clone());
        }
        builder.recv(is_valid.clone());
        builder.send(is_valid.clone());
        builder.send(is_valid.clone());
        builder.send(is_valid);
    }
}

impl MachineAir<F> for BatchTranscriptInputsAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeBatchTranscriptInputs".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(BatchTranscriptInputsTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        BatchTranscriptInputsTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BatchSumcheckAir {
    pub num_public_values: usize,
    pub seed_prefix_limbs: usize,
    pub transcript_event_bus: TranscriptEventBus,
    pub summary_bus: ProofShapeSummaryBus,
    pub chain_bus: BatchSumcheckClaimChainBus,
    pub opening_point_bus: BatchOpeningPointBus,
    pub sumcheck_out_bus: SumcheckOutBus,
}

impl BatchSumcheckAir {
    pub const fn new(num_public_values: usize, contains_global_bus: bool) -> Self {
        Self {
            num_public_values,
            seed_prefix_limbs: batch_seed_prefix_limbs(contains_global_bus),
            transcript_event_bus: TranscriptEventBus::new(),
            summary_bus: ProofShapeSummaryBus::new(),
            chain_bus: BatchSumcheckClaimChainBus::new(),
            opening_point_bus: BatchOpeningPointBus::new(),
            sumcheck_out_bus: SumcheckOutBus::new(),
        }
    }
}

impl Default for BatchSumcheckAir {
    fn default() -> Self {
        Self::new(0, false)
    }
}

impl BaseAir<F> for BatchSumcheckAir {
    fn width(&self) -> usize {
        batch_sumcheck_width()
    }
}

impl<AB: FullAirBuilder> FullAir<AB> for BatchSumcheckAir {
    fn width(&self) -> usize {
        batch_sumcheck_width()
    }

    fn required_max_beta_power(&self) -> usize {
        [
            self.transcript_event_bus.required_max_beta_power_floor(),
            self.summary_bus.required_max_beta_power_floor(),
            self.chain_bus.required_max_beta_power_floor(),
            self.opening_point_bus.required_max_beta_power_floor(),
            self.sumcheck_out_bus.required_max_beta_power_floor(),
        ]
        .into_iter()
        .max()
        .expect("non-empty bus list")
    }

    fn reserved_poly(&self) -> Vec<PairCol> {
        vec![
            PairCol::Main(core::mem::offset_of!(BatchSumcheckCols<u8>, is_seed)),
            PairCol::Main(core::mem::offset_of!(BatchSumcheckCols<u8>, is_round)),
            PairCol::Main(core::mem::offset_of!(BatchSumcheckCols<u8>, round_idx)),
        ]
    }

    fn precompute_lc(&self, builder: &mut AB) {
        let values = {
            let main = builder.main();
            let local: &BatchSumcheckCols<AB::VarMaybeExt> = main.borrow();
            sumcheck_precomputed(self, builder, local)
        };
        debug_assert_eq!(
            values.len(),
            BATCH_SUMCHECK_LOOKUP_COUNT + NUM_BATCH_SUMCHECK_PACKED_COLS
        );
        for value in values {
            builder.retain_precomputed(value);
        }
    }

    fn eval(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &BatchSumcheckReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();
        let precomputed = builder.precomputed();
        let precomputed_binding = precomputed.row_slice(0);
        let packed: &BatchSumcheckPackedCols<AB::VarExt> = precomputed_binding.deref()
            [BATCH_SUMCHECK_LOOKUP_COUNT..
                BATCH_SUMCHECK_LOOKUP_COUNT + NUM_BATCH_SUMCHECK_PACKED_COLS]
            .borrow();

        assert_bool(builder, local.is_seed.clone());
        assert_bool(builder, local.is_round.clone());
        assert_bool(builder, local.is_seed.clone() + local.is_round.clone());
        builder.assert_zero(local.is_seed.clone() * local.round_idx.clone());
        constrain_sumcheck_horner(builder, local, packed);
        builder.assert_zero_ext(packed.claim_out.clone() * local.is_seed.clone());
    }

    fn lookup(&self, builder: &mut AB) {
        let reserved = builder.reserved_poly();
        let local_binding = reserved.row_slice(0);
        let local: &BatchSumcheckReservedCols<AB::VarMaybeExt> = local_binding.deref().borrow();

        builder.recv(local.is_seed.clone());
        builder.recv(local.is_round.clone());
        builder.send(local.is_seed.clone() + local.is_round.clone());
        for _ in 0..(BATCH_SUMCHECK_EVALS * D_EF) {
            builder.recv(local.is_round.clone());
        }
        for _ in 0..D_EF {
            builder.recv(local.is_round.clone());
        }
        builder.send(local.is_round.clone() * const_maybe::<AB>(2));
        for _ in 0..D_EF {
            builder.recv(local.is_round.clone());
        }
        builder.send(local.is_round.clone());
    }
}

impl MachineAir<F> for BatchSumcheckAir {
    type Record = RecursionRecord;
    type Program = RecursionNativeProgram<F>;

    fn name(&self) -> String {
        "NativeBatchSumcheck".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(BatchSumcheckTraceGenerator::trace_height(input))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> CompressedMatrix<F> {
        BatchSumcheckTraceGenerator::generate_trace_compressed(input)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

fn input_denominators<AB: FullAirBuilder>(
    air: &BatchTranscriptInputsAir,
    builder: &AB,
    local: &BatchTranscriptInputsCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    let mut denominators = Vec::with_capacity(BATCH_TRANSCRIPT_INPUT_LOOKUP_COUNT);
    denominators.push(air.transcript_event_bus.denominator(
        builder,
        proof_idx.clone(),
        AB::zero_maybe(),
        AB::zero_maybe(),
        AB::VarMaybeExt::from(AB::F::from_canonical_u32(BATCH_VK_TAG_V1)),
    ));
    denominators.push(air.transcript_event_bus.denominator(
        builder,
        proof_idx.clone(),
        AB::one_maybe(),
        AB::zero_maybe(),
        AB::VarMaybeExt::from(AB::F::from_canonical_u32(BATCH_VK_VERSION_V1)),
    ));
    for i in 0..(2 * D_EF) {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            e3_tidx::<AB>(air.seed_prefix_limbs, air.num_public_values, local.c_chips.clone(), i),
            AB::one_maybe(),
            local.event_values[i].clone(),
        ));
    }
    for i in 0..D_EF {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            e7_tidx::<AB>(air.seed_prefix_limbs, air.num_public_values, local.c_chips.clone(), i),
            AB::one_maybe(),
            local.event_values[2 * D_EF + i].clone(),
        ));
    }
    denominators.push(air.fold_plan_chain_bus.denominator(
        builder,
        proof_idx.clone(),
        AB::zero_maybe(),
        local.c_chips.clone(),
        AB::zero_maybe(),
    ));
    denominators.push(air.sumcheck_out_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(SUMCHECK_OUT_PERM_ALPHA),
        AB::zero_maybe(),
        arr5(&local.event_values[..D_EF]),
    ));
    denominators.push(air.sumcheck_out_bus.denominator(
        builder,
        proof_idx.clone(),
        const_maybe::<AB>(SUMCHECK_OUT_PERM_BETA),
        AB::zero_maybe(),
        arr5(&local.event_values[D_EF..2 * D_EF]),
    ));
    let zero_ext = core::array::from_fn(|_| AB::zero_maybe());
    denominators.push(air.fold_chain_bus.denominator(
        builder,
        proof_idx,
        AB::zero_maybe(),
        arr5(&local.event_values[2 * D_EF..3 * D_EF]),
        zero_ext.clone(),
        zero_ext.clone(),
        zero_ext,
    ));
    denominators
}

fn sumcheck_precomputed<AB: FullAirBuilder>(
    air: &BatchSumcheckAir,
    builder: &AB,
    local: &BatchSumcheckCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let mut values = sumcheck_denominators(air, builder, local);
    values.extend(local.coefficients.iter().map(|coeff| AB::pack_ext_limbs(coeff)));
    values.push(AB::pack_ext_limbs(&local.challenge));
    values.push(AB::pack_ext_limbs(&local.claim_in));
    values.push(AB::pack_ext_limbs(&local.acc_3));
    values.push(AB::pack_ext_limbs(&local.acc_2));
    values.push(AB::pack_ext_limbs(&local.acc_1));
    values.push(AB::pack_ext_limbs(&local.claim_out));
    values
}

fn sumcheck_denominators<AB: FullAirBuilder>(
    air: &BatchSumcheckAir,
    builder: &AB,
    local: &BatchSumcheckCols<AB::VarMaybeExt>,
) -> Vec<AB::VarExt> {
    let proof_idx = local.proof_idx.clone();
    let opening_idx = local.r_rounds.clone() - local.round_idx.clone() - AB::one_maybe();
    let coefficients = sumcheck_coefficients_expr::<AB>(local);
    let evals = core::array::from_fn::<_, BATCH_SUMCHECK_EVALS, _>(|node| {
        eval_coefficient_polynomial::<AB>(&coefficients, node)
    });
    let e9_base = e9_tidx::<AB>(
        air.seed_prefix_limbs,
        air.num_public_values,
        local.r_rounds.clone(),
        local.c_chips.clone(),
        local.round_idx.clone(),
    );

    let mut denominators = Vec::with_capacity(BATCH_SUMCHECK_LOOKUP_COUNT);
    denominators.push(air.summary_bus.denominator(
        builder,
        proof_idx.clone(),
        local.r_rounds.clone(),
        local.c_chips.clone(),
        const_maybe::<AB>(air.num_public_values),
        local.summary_id_base.clone(),
    ));
    denominators.push(air.chain_bus.denominator(
        builder,
        proof_idx.clone(),
        local.round_idx.clone(),
        local.r_rounds.clone(),
        local.c_chips.clone(),
        local.claim_in.clone(),
    ));
    denominators.push(air.chain_bus.denominator(
        builder,
        proof_idx.clone(),
        local.round_idx.clone() + local.is_round.clone(),
        local.r_rounds.clone(),
        local.c_chips.clone(),
        local.claim_out.clone(),
    ));
    for (eval_idx, eval) in evals.iter().enumerate() {
        for (limb_idx, limb) in eval.iter().enumerate() {
            denominators.push(air.transcript_event_bus.denominator(
                builder,
                proof_idx.clone(),
                e9_base.clone() + const_maybe::<AB>(eval_idx * D_EF + limb_idx),
                AB::zero_maybe(),
                limb.clone(),
            ));
        }
    }
    for limb_idx in 0..D_EF {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            e9_base.clone() + const_maybe::<AB>(BATCH_SUMCHECK_EVALS * D_EF + limb_idx),
            AB::one_maybe(),
            local.challenge[limb_idx].clone(),
        ));
    }
    denominators.push(air.opening_point_bus.denominator(
        builder,
        proof_idx.clone(),
        opening_idx.clone(),
        local.challenge.clone(),
    ));
    for limb_idx in 0..D_EF {
        denominators.push(air.transcript_event_bus.denominator(
            builder,
            proof_idx.clone(),
            e8_tidx::<AB>(
                air.seed_prefix_limbs,
                air.num_public_values,
                local.c_chips.clone(),
                opening_idx.clone(),
                limb_idx,
            ),
            AB::one_maybe(),
            local.eq_challenge[limb_idx].clone(),
        ));
    }
    denominators.push(air.sumcheck_out_bus.denominator(
        builder,
        proof_idx,
        const_maybe::<AB>(SUMCHECK_OUT_EQ),
        opening_idx,
        local.eq_challenge.clone(),
    ));
    denominators
}

fn sumcheck_coefficients_expr<AB: FullAirBuilder>(
    local: &BatchSumcheckCols<AB::VarMaybeExt>,
) -> [[AB::VarMaybeExt; D_EF]; BATCH_SUMCHECK_EVALS] {
    let half = AB::F::two().inverse();
    let c0 = core::array::from_fn(|limb| {
        let higher = local
            .coefficients
            .iter()
            .fold(AB::zero_maybe(), |sum, coeff| sum + coeff[limb].clone());
        AB::mul_base(local.claim_in[limb].clone() - higher, half)
    });
    core::array::from_fn(
        |idx| {
            if idx == 0 {
                c0.clone()
            } else {
                local.coefficients[idx - 1].clone()
            }
        },
    )
}

fn eval_coefficient_polynomial<AB: FullAirBuilder>(
    coefficients: &[[AB::VarMaybeExt; D_EF]; BATCH_SUMCHECK_EVALS],
    node: usize,
) -> [AB::VarMaybeExt; D_EF] {
    let x = AB::F::from_canonical_usize(node);
    core::array::from_fn(|limb| {
        coefficients
            .iter()
            .rev()
            .fold(AB::zero_maybe(), |acc, coeff| AB::mul_base(acc, x) + coeff[limb].clone())
    })
}

fn constrain_sumcheck_horner<AB: FullAirBuilder>(
    builder: &mut AB,
    local: &BatchSumcheckReservedCols<AB::VarMaybeExt>,
    packed: &BatchSumcheckPackedCols<AB::VarExt>,
) {
    let zero = AB::pack_ext_limbs(&[AB::zero_maybe()]);
    let c0 = (packed.claim_in.clone() -
        packed.coefficients.iter().cloned().fold(zero, |sum, coeff| sum + coeff)) *
        AB::VarMaybeExt::from(AB::F::two().inverse());
    builder.assert_zero_ext(
        (packed.acc_3.clone() -
            (packed.coefficients[2].clone() +
                packed.challenge.clone() * packed.coefficients[3].clone())) *
            local.is_round.clone(),
    );
    builder.assert_zero_ext(
        (packed.acc_2.clone() -
            (packed.coefficients[1].clone() + packed.challenge.clone() * packed.acc_3.clone())) *
            local.is_round.clone(),
    );
    builder.assert_zero_ext(
        (packed.acc_1.clone() -
            (packed.coefficients[0].clone() + packed.challenge.clone() * packed.acc_2.clone())) *
            local.is_round.clone(),
    );
    builder.assert_zero_ext(
        (packed.claim_out.clone() - (c0 + packed.challenge.clone() * packed.acc_1.clone())) *
            local.is_round.clone(),
    );
}

fn e3_tidx<AB: FullAirBuilder>(
    seed_prefix_limbs: usize,
    num_public_values: usize,
    c_chips: AB::VarMaybeExt,
    offset: usize,
) -> AB::VarMaybeExt {
    const_maybe::<AB>(seed_prefix_limbs + num_public_values + BATCH_COMMITMENT_LIMBS) +
        const_maybe::<AB>(BATCH_ACTIVE_SHAPE_HEADER_LIMBS) +
        c_chips * const_maybe::<AB>(BATCH_ACTIVE_SHAPE_ENTRY_LIMBS) +
        const_maybe::<AB>(offset)
}

fn e7_tidx<AB: FullAirBuilder>(
    seed_prefix_limbs: usize,
    num_public_values: usize,
    c_chips: AB::VarMaybeExt,
    offset: usize,
) -> AB::VarMaybeExt {
    e3_tidx::<AB>(
        seed_prefix_limbs,
        num_public_values,
        c_chips.clone(),
        BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS,
    ) + c_chips * const_maybe::<AB>(D_EF) +
        const_maybe::<AB>(offset)
}

fn e8_tidx<AB: FullAirBuilder>(
    seed_prefix_limbs: usize,
    num_public_values: usize,
    c_chips: AB::VarMaybeExt,
    round_idx: AB::VarMaybeExt,
    offset: usize,
) -> AB::VarMaybeExt {
    e7_tidx::<AB>(seed_prefix_limbs, num_public_values, c_chips, D_EF) +
        round_idx * const_maybe::<AB>(D_EF) +
        const_maybe::<AB>(offset)
}

fn e9_tidx<AB: FullAirBuilder>(
    seed_prefix_limbs: usize,
    num_public_values: usize,
    r_rounds: AB::VarMaybeExt,
    c_chips: AB::VarMaybeExt,
    round_idx: AB::VarMaybeExt,
) -> AB::VarMaybeExt {
    e3_tidx::<AB>(seed_prefix_limbs, num_public_values, c_chips.clone(), 0) +
        const_maybe::<AB>(BATCH_PERM_CHALLENGE_AND_COMMIT_LIMBS) +
        c_chips * const_maybe::<AB>(D_EF) +
        const_maybe::<AB>(D_EF) +
        r_rounds * const_maybe::<AB>(D_EF) +
        round_idx * const_maybe::<AB>(BATCH_SUMCHECK_EVALS * D_EF + D_EF)
}

fn arr5<T: Clone>(slice: &[T]) -> [T; D_EF] {
    debug_assert_eq!(slice.len(), D_EF);
    core::array::from_fn(|idx| slice[idx].clone())
}

fn assert_bool<AB: FullAirBuilder>(builder: &mut AB, value: AB::VarMaybeExt) {
    builder.assert_zero(value.clone() * (value - AB::one_maybe()));
}

fn const_maybe<AB: FullAirBuilder>(value: usize) -> AB::VarMaybeExt {
    AB::VarMaybeExt::from(AB::F::from_canonical_usize(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polyair::Chip;

    use crate::{
        batch_constraint_dt::columns::NUM_BATCH_SUMCHECK_RESERVED_COLS,
        symbolic_expr_fixed_dt::RecursionFixedSymbolicChip, symbolic_ir_dt::RecursionPolyAirChipIr,
    };

    fn print_shape<A: MachineAir<F>>(
        label: &str,
        chip: &Chip<A, F, D_EF>,
    ) -> (usize, usize, usize) {
        let fixed =
            RecursionFixedSymbolicChip::from_polyair_chip(0, chip).expect("fixed batch chip");
        let ir = RecursionPolyAirChipIr::compile(&fixed).expect("batch chip IR");
        let roots = ir.gate_roots.len() + 2 * ir.lookup_multiplicity_roots.len();
        let folds = ir.gate_roots.len() +
            ir.lookup_multiplicity_roots.len().div_ceil(ir.logup_batch_size.max(1)) +
            1;
        eprintln!(
            "{label}_SHAPE main={} reserved={} precomputed={} permutation={} active={} \
             lookups={} gates={} alpha={} nodes={} roots={} folds={}",
            chip.width(),
            chip.reserved_poly().len(),
            chip.num_precompute(),
            chip.perm_width(),
            chip.reserved_poly().len() + chip.num_precompute() + chip.perm_width(),
            chip.num_lookup(),
            chip.symbolic_builder.gate.len(),
            chip.num_alpha,
            ir.node_table.len(),
            roots,
            folds,
        );
        (ir.node_table.len(), roots, folds)
    }

    #[test]
    fn symbolic_analysis() {
        let inputs =
            Chip::<BatchTranscriptInputsAir, F, D_EF>::new(BatchTranscriptInputsAir::new(52, true));
        assert_eq!(NUM_BATCH_TRANSCRIPT_INPUTS_COLS, 18);
        assert_eq!(inputs.reserved_poly().len(), 1);
        assert_eq!(inputs.num_lookup(), BATCH_TRANSCRIPT_INPUT_LOOKUP_COUNT);
        assert_eq!(inputs.num_precompute(), BATCH_TRANSCRIPT_INPUT_LOOKUP_COUNT);
        assert_eq!(inputs.perm_width(), 11);
        assert!(inputs.required_max_beta_power() >= 13);
        assert!(inputs.degree <= 3);
        assert_eq!(inputs.symbolic_builder.gate.len(), 1);
        assert_eq!(inputs.num_alpha, 13);
        let (input_nodes, input_roots, input_folds) = print_shape("BATCH_INPUTS", &inputs);
        assert_eq!((input_nodes, input_roots, input_folds), (142, 43, 13));
        assert_eq!(
            (
                input_nodes.next_power_of_two(),
                input_roots.next_power_of_two(),
                input_folds.next_power_of_two(),
            ),
            (256, 64, 16)
        );

        let sumcheck = Chip::<BatchSumcheckAir, F, D_EF>::new(BatchSumcheckAir::new(52, true));
        assert_eq!(batch_sumcheck_width(), 62);
        assert_eq!(sumcheck.reserved_poly().len(), NUM_BATCH_SUMCHECK_RESERVED_COLS);
        assert_eq!(sumcheck.num_lookup(), BATCH_SUMCHECK_LOOKUP_COUNT);
        assert_eq!(
            sumcheck.num_precompute(),
            BATCH_SUMCHECK_LOOKUP_COUNT + NUM_BATCH_SUMCHECK_PACKED_COLS
        );
        assert_eq!(sumcheck.perm_width(), 20);
        assert!(sumcheck.required_max_beta_power() >= 13);
        assert!(sumcheck.degree <= 3);
        assert_eq!(sumcheck.symbolic_builder.gate.len(), 9);
        assert_eq!(sumcheck.num_alpha, 30);
        let (sumcheck_nodes, sumcheck_roots, sumcheck_folds) =
            print_shape("BATCH_SUMCHECK", &sumcheck);
        assert_eq!((sumcheck_nodes, sumcheck_roots, sumcheck_folds), (470, 89, 30));
        assert_eq!(
            (
                sumcheck_nodes.next_power_of_two(),
                sumcheck_roots.next_power_of_two(),
                sumcheck_folds.next_power_of_two(),
            ),
            (512, 128, 32)
        );
    }
}
