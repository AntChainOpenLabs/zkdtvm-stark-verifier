use std::marker::PhantomData;

use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, ExtensionField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrixView;

use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, Felt, SymbolicExt},
};
use dt_stark::{
    air::{derive_active_shape_v1, FullAir, MachineAir, PairCol, PolyAirExtendable},
    sumcheck::{
        config::SCStarkGenericConfig,
        proof::{SCChipOpenedValues, SCShardCommitment},
        verifier::Selectors,
    },
    Val,
};
use polyair::Chip;

use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    sumcheck::{
        types::{SCShardProofVariable, SCVerifyingKeyVariable},
        SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};

use super::{
    polyair_folder::RecursivePolyAirConstraintFolder,
    polyair_precompute::RecursivePolyAirPrecomputeRowBuilder,
};

const POLYAIR_NUM_SKIP_ROUNDS: usize = 1;
const POLYAIR_CHIP_LOG_HEIGHT_THRESHOLD: usize = 0;

pub struct PolyAirSumcheckVerifier<C, SC, A, const D: usize> {
    _phantom: PhantomData<(C, SC, A)>,
}

impl<C, SC, A, const D: usize> PolyAirSumcheckVerifier<C, SC, A, D>
where
    C: CircuitConfig<F = Val<SC>>,
    C::F: Field + PrimeField32,
    C::EF: ExtensionField<C::F>,
    SC: SCStarkGenericConfig + SCBabyBearFriConfigVariable<C>,
    A: MachineAir<Val<SC>>,
    Val<SC>: PolyAirExtendable<D>,
    Builder<C>: CircuitV2Builder<C>,
{
    /// Verify sumcheck unipolys (all linear since chip_log_height_threshold = 0).
    ///
    /// Mirrors polyair/src/verifier.rs:253-287.
    fn verify_sumcheck_unipolys(
        builder: &mut Builder<C>,
        claim: &mut Ext<C::F, C::EF>,
        unipolys: &[crate::sumcheck::types::UniPolyVariable<C>],
        challenger: &mut SC::FriChallengerVariable,
    ) -> Vec<Ext<C::F, C::EF>> {
        let mut sumcheck_challenges = Vec::with_capacity(unipolys.len());

        for unipoly in unipolys.iter() {
            // All rounds are linear (chip_log_height_threshold = 0)
            unipoly.observe_into(builder, challenger);
            let challenge = challenger.sample_ext(builder);
            sumcheck_challenges.push(challenge);
            *claim = unipoly.verify_and_evaluate(builder, &challenge, *claim);
        }

        sumcheck_challenges
    }

    /// Verify a single PolyAir shard.
    ///
    /// Challenger sequence follows polyair/src/verifier.rs:82-149 exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_shard(
        builder: &mut Builder<C>,
        vk: &SCVerifyingKeyVariable<C, SC>,
        chips: &[&Chip<A, Val<SC>, D>],
        challenger: &mut SC::FriChallengerVariable,
        proof: &SCShardProofVariable<C, SC>,
    ) where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        let SCShardProofVariable {
            commitment,
            opened_values,
            sumcheck_proof,
            chip_ordering,
            public_values,
            ..
        } = proof;

        let log_heights: Vec<usize> =
            opened_values.chips.iter().map(|val| val.log_height).collect();
        if chips.len() != opened_values.chips.len() {
            let machine_chips = chips.iter().map(|chip| chip.name()).collect::<Vec<_>>();
            let proof_chips = chip_ordering
                .iter()
                .map(|(name, index)| (name.as_str(), *index))
                .collect::<Vec<_>>();
            panic!(
                "polyair verifier chip/opening mismatch: machine_chips={machine_chips:?}, proof_chips={proof_chips:?}, openings={}, log_heights={log_heights:?}",
                opened_values.chips.len()
            );
        }

        let SCShardCommitment { main_commit, permutation_commit } = *commitment;

        // === Challenger sequence (polyair/src/verifier.rs:82-149) ===

        // 1. observe main_commit (v.rs:82)
        challenger.observe(builder, main_commit);

        let active_shape = derive_active_shape_v1(
            chips
                .iter()
                .zip(log_heights.iter())
                .map(|(chip, &log_height)| (chip.name(), chip.width(), log_height)),
        )
        .expect("recursive PolyAir active shape must be canonical");
        crate::global_claim::observe_active_shape(builder, challenger, &active_shape);

        for (chip, opening) in chips.iter().zip(opened_values.chips.iter()) {
            if chip.num_lookup() == 0 {
                let zero_ef: Ext<_, _> = builder.constant(C::EF::zero());
                builder.assert_ext_eq(opening.local_cumulative_sum, zero_ef);
            }
        }

        // 3. sample perm challenges AFTER cumsum (v.rs:105-106) NOTE: dt_stark samples perm
        //    challenges BEFORE cumsum — order is reversed here
        let perm_alpha: Ext<C::F, C::EF> = challenger.sample_ext(builder);
        let perm_beta: Ext<C::F, C::EF> = challenger.sample_ext(builder);

        // 4. observe permutation_commit if present (v.rs:129-131)
        if let Some(permutation_commit) = permutation_commit {
            challenger.observe(builder, permutation_commit);
        }

        // 5. per-chip observe local_cumulative_sum (v.rs:133-138) NOTE: This step does not exist in
        //    dt_stark
        for opening in opened_values.chips.iter() {
            let local_sum = C::ext2felt(builder, opening.local_cumulative_sum);
            challenger.observe_slice(builder, local_sum);
        }

        // 6. sample alpha (v.rs:140)
        let alpha: Ext<C::F, C::EF> = challenger.sample_ext(builder);

        // 7. compute num_rounds (v.rs:142-148) With chip_log_height_threshold = 0:
        //    num_rounds_nonlinear = 0, all rounds linear
        let max_height = *log_heights
            .iter()
            .max()
            .expect("a PolyAir shard must contain at least one chip opening");
        let num_rounds = max_height; // num_rounds_linear = max_height, nonlinear = 0

        // 8. sample eq_challenges (v.rs:148-149)
        let eq_challenges: Vec<Ext<C::F, C::EF>> =
            (0..num_rounds).map(|_| challenger.sample_ext(builder)).collect();

        let mut claim: Ext<_, _> = builder.constant(C::EF::zero());

        let sumcheck_challenges = Self::verify_sumcheck_unipolys(
            builder,
            &mut claim,
            &sumcheck_proof.unipolys[..num_rounds],
            challenger,
        );

        // extended_sumcheck_challenges = sumcheck_challenges (no nonlinear expansion needed)
        let extended_sumcheck_challenges = sumcheck_challenges.clone();

        // Compute beta_powers and beta_septix in circuit
        let max_beta_powers = chips.iter().map(|c| c.required_max_beta_power()).max().unwrap_or(0);
        let beta_powers_ext = Self::compute_beta_powers(builder, perm_beta, max_beta_powers);
        let beta_septix = Self::compute_beta_septix(builder, perm_beta);

        // Convert beta_powers to SymbolicExt for builders
        let beta_powers_sym: Vec<SymbolicExt<C::F, C::EF>> =
            beta_powers_ext.iter().map(|&e| SymbolicExt::from(e)).collect();
        let perm_alpha_sym = SymbolicExt::from(perm_alpha);
        let beta_septix_sym = SymbolicExt::from(beta_septix);

        // Compute num_constraints from VK
        let num_constraints: Vec<usize> = chips
            .iter()
            .map(|chip| {
                *vk.constraints_map.get(&chip.name()).expect("chip not found in constraints map")
            })
            .collect();

        // Compute selectors (is_first_row, is_last_row) per chip
        let one: Ext<_, _> = builder.constant(C::EF::one());
        let mut selector_cache: std::collections::HashMap<
            usize,
            (Ext<C::F, C::EF>, Ext<C::F, C::EF>),
        > = std::collections::HashMap::new();
        let selectors: Vec<Selectors<Ext<C::F, C::EF>>> = opened_values
            .chips
            .iter()
            .map(|chip_opening| {
                let (is_first_row, is_last_row) =
                    *selector_cache.entry(chip_opening.log_height).or_insert_with(|| {
                        let start = extended_sumcheck_challenges.len() - chip_opening.log_height;
                        extended_sumcheck_challenges[start..].iter().fold(
                            (one, one),
                            |(first, last): (Ext<C::F, C::EF>, Ext<C::F, C::EF>), rand| {
                                let one_minus_rand: Ext<C::F, C::EF> = builder.eval(one - *rand);
                                (builder.eval(first * one_minus_rand), builder.eval(last * *rand))
                            },
                        )
                    });
                Selectors { is_first_row, is_last_row }
            })
            .collect();

        // Compute alpha_shifts
        let max_constraints = num_constraints.iter().copied().max().unwrap_or(0);
        let mut alpha_powers = Vec::with_capacity(max_constraints + 1);
        alpha_powers.push(one);
        for _ in 1..=max_constraints {
            let prev = *alpha_powers.last().unwrap();
            alpha_powers.push(builder.eval(prev * alpha));
        }
        let mut alpha_shifts = Vec::with_capacity(chips.len());
        alpha_shifts.push(one);
        for &num in num_constraints.iter().take(chips.len().saturating_sub(1)) {
            let shift: Ext<_, _> =
                builder.eval(alpha_shifts[alpha_shifts.len() - 1] * alpha_powers[num]);
            alpha_shifts.push(shift);
        }

        // Compute perm_eval
        let zero_ext: Ext<_, _> = builder.constant(C::EF::zero());
        let mut perm_eval: Ext<C::F, C::EF> = zero_ext;
        for (i, chip_opening) in opened_values.chips.iter().enumerate() {
            if chip_opening.permutation.local.is_empty() {
                continue;
            }
            let mut row_sum: Ext<C::F, C::EF> = zero_ext;
            for &val in &chip_opening.permutation.local {
                row_sum = builder.eval(row_sum + val);
            }
            assert!(chip_opening.log_height < 31, "log_height too large");
            let inv: C::EF = C::EF::from_canonical_usize(1 << chip_opening.log_height).inverse();
            let inv_ext: Ext<_, _> = builder.constant(inv);
            let adjusted: Ext<_, _> =
                builder.eval(row_sum - chip_opening.local_cumulative_sum * inv_ext);
            let term: Ext<_, _> = builder.eval(adjusted * alpha_shifts[i]);
            perm_eval = builder.eval(perm_eval + term);
        }

        // Five-phase eval_constraints
        let main_eval = Self::eval_constraints(
            builder,
            chips,
            &opened_values.chips,
            &selectors,
            alpha,
            &alpha_shifts,
            perm_alpha_sym,
            &beta_powers_sym,
            beta_septix_sym,
            public_values,
        );

        // Compute eq polynomial (all linear rounds since num_rounds_nonlinear = 0)
        let sumcheck_challenges_rev: Vec<Ext<C::F, C::EF>> =
            sumcheck_challenges.iter().rev().copied().collect();
        let eq = C::eq_poly(builder, eq_challenges, sumcheck_challenges_rev);

        // Final check: main_eval * eq + perm_eval == last_claim
        let main_eq: Ext<_, _> = builder.eval(main_eval * eq);
        let lhs: Ext<_, _> = builder.eval(main_eq + perm_eval);
        builder.assert_ext_eq(lhs, claim);

        let zero_local_sum: Ext<C::F, C::EF> = builder.constant(C::EF::zero());
        let local_sum = opened_values
            .chips
            .iter()
            .map(|opening| opening.local_cumulative_sum)
            .fold(zero_local_sum, |acc, value| builder.eval(acc + value));
        let expected = crate::global_claim::expected_local_imbalance(
            builder,
            public_values,
            perm_alpha,
            perm_beta,
        );
        builder.assert_ext_eq(local_sum, expected);
    }

    /// Five-phase constraint evaluation (mirrors polyair/src/verifier.rs:411-476).
    #[allow(clippy::too_many_arguments)]
    fn eval_constraints(
        builder: &mut Builder<C>,
        chips: &[&Chip<A, Val<SC>, D>],
        openings: &[SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>],
        selectors: &[Selectors<Ext<C::F, C::EF>>],
        alpha: Ext<C::F, C::EF>,
        alpha_shifts: &[Ext<C::F, C::EF>],
        perm_alpha: SymbolicExt<C::F, C::EF>,
        beta_powers: &[SymbolicExt<C::F, C::EF>],
        beta_septix: SymbolicExt<C::F, C::EF>,
        public_values: &[Felt<C::F>],
    ) -> Ext<C::F, C::EF>
    where
        A: for<'a> FullAir<RecursivePolyAirConstraintFolder<'a, C>>,
        A: for<'a> FullAir<RecursivePolyAirPrecomputeRowBuilder<'a, C>>,
    {
        const CHUNK_SIZE: usize = 4;
        let all_items: Vec<_> = izip!(chips, openings, selectors, alpha_shifts).collect();
        let mut total: Ext<C::F, C::EF> = builder.constant(C::EF::zero());

        for chunk in all_items.chunks(CHUNK_SIZE) {
            let partial_expr = chunk
                .iter()
                .map(|(&chip, opening, sels, shift)| {
                    // Convert opened values from Ext to SymbolicExt
                    let preprocessed_sym: Vec<SymbolicExt<C::F, C::EF>> =
                        opening.preprocessed.local.iter().map(|&e| SymbolicExt::from(e)).collect();
                    let main_sym: Vec<SymbolicExt<C::F, C::EF>> =
                        opening.main.local.iter().map(|&e| SymbolicExt::from(e)).collect();

                    // Phase 1: Precompute linear combinations (v.rs:424-437)
                    let num_precompute = chip.num_precompute();
                    let mut precompute_lc: Vec<SymbolicExt<C::F, C::EF>> =
                        vec![SymbolicExt::zero(); num_precompute];
                    let mut precompute_builder = RecursivePolyAirPrecomputeRowBuilder::<C> {
                        preprocessed: &preprocessed_sym,
                        main: &main_sym,
                        beta_powers,
                        public: public_values,
                        alpha: perm_alpha,
                        beta_septix,
                        row: &mut precompute_lc,
                        col_index: 0,
                    };
                    chip.air.precompute_lc(&mut precompute_builder);

                    // Phase 2: Reserved polynomials (v.rs:438-445)
                    let reserved_poly: Vec<SymbolicExt<C::F, C::EF>> = chip
                        .reserved_poly()
                        .iter()
                        .map(|col| match *col {
                            PairCol::Prep(i) => preprocessed_sym[i],
                            PairCol::Main(i) => main_sym[i],
                        })
                        .collect();

                    // Phase 4+5: Constraint folding + lookup (v.rs:448-468)
                    let permutation_sym: Vec<SymbolicExt<C::F, C::EF>> =
                        opening.permutation.local.iter().map(|&e| SymbolicExt::from(e)).collect();

                    let alpha_sym = SymbolicExt::from(alpha);
                    let mut folder = RecursivePolyAirConstraintFolder::<C> {
                        public: public_values,
                        alpha: perm_alpha,
                        beta_powers,
                        beta_septix,
                        precomputed: RowMajorMatrixView::new_row(&precompute_lc),
                        reserved_poly: RowMajorMatrixView::new_row(&reserved_poly),
                        is_first_row: SymbolicExt::from(sels.is_first_row),
                        is_last_row: SymbolicExt::from(sels.is_last_row),
                        local_sum: SymbolicExt::from(opening.local_cumulative_sum),
                        permutation: RowMajorMatrixView::new_row(&permutation_sym),
                        multiplicities: vec![],
                        batch_size: chip.logup_batch_size(),
                        accumulator: SymbolicExt::zero(),
                        constraint_reducer: alpha_sym,
                    };

                    chip.air.eval(&mut folder);
                    chip.air.lookup(&mut folder);
                    folder.constrain_lookup();

                    // Apply alpha shift and extra alpha multiplication (v.rs:469-473)
                    if opening.permutation.local.is_empty() {
                        folder.accumulator * SymbolicExt::from(**shift)
                    } else {
                        folder.accumulator * SymbolicExt::from(**shift) * alpha_sym
                    }
                })
                .sum::<SymbolicExt<_, _>>();
            let partial: Ext<C::F, C::EF> = builder.eval(partial_expr);
            total = builder.eval(total + partial);
        }
        total
    }

    /// Compute beta^0, beta^1, ..., beta^max_power in circuit.
    fn compute_beta_powers(
        builder: &mut Builder<C>,
        beta: Ext<C::F, C::EF>,
        max_power: usize,
    ) -> Vec<Ext<C::F, C::EF>> {
        let mut powers = Vec::with_capacity(max_power + 1);
        let one_ext: Ext<_, _> = builder.constant(C::EF::one());
        powers.push(one_ext);
        for _ in 1..=max_power {
            let prev = *powers.last().unwrap();
            powers.push(builder.eval(prev * beta));
        }
        powers
    }

    /// Compute beta raised to the 7th power in the septic extension in circuit.
    ///
    /// Uses the irreducible polynomial to reduce beta^7 modulo the septic polynomial.
    fn compute_beta_septix(builder: &mut Builder<C>, beta: Ext<C::F, C::EF>) -> Ext<C::F, C::EF> {
        #[cfg(feature = "koalabear")]
        {
            use dt_stark::septic_curve_params::KoalaBearCurveParams;
            Self::compute_beta_septix_with_params::<KoalaBearCurveParams>(builder, beta)
        }
        #[cfg(feature = "babybear")]
        {
            use dt_stark::septic_curve_params::BabyBearCurveParams;
            Self::compute_beta_septix_with_params::<BabyBearCurveParams>(builder, beta)
        }
    }

    fn compute_beta_septix_with_params<P: dt_stark::septic_curve_params::SepticCurveParams>(
        builder: &mut Builder<C>,
        beta: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let neg_const = C::EF::from_canonical_u32(
            (-P::POLY_COEFFS[0]).try_into().expect("negative septic constant coefficient"),
        );
        let neg_linear = C::EF::from_canonical_u32(
            (-P::POLY_COEFFS[1]).try_into().expect("negative septic linear coefficient"),
        );

        // Compute beta^7 via squaring chain: b^2, b^4, b^7 = b^4 * b^2 * b
        let beta2: Ext<_, _> = builder.eval(beta * beta);
        let beta4: Ext<_, _> = builder.eval(beta2 * beta2);
        let beta7: Ext<_, _> = builder.eval(beta4 * beta2 * beta);

        // beta_septix = beta^7 - neg_linear * beta - neg_const
        // Mirrors: dt_stark::septic_curve_params::compute_beta_septix
        let neg_const_ext: Ext<_, _> = builder.constant(neg_const);
        let neg_linear_ext: Ext<_, _> = builder.constant(neg_linear);
        let result: Ext<_, _> = builder.eval(beta7 - neg_linear_ext * beta - neg_const_ext);
        result
    }
}
