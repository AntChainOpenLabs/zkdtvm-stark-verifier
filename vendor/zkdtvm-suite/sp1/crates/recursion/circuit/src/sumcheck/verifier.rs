use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    constraints::RecursiveSumcheckConstraintFolder,
    sumcheck::{
        pcs::PcsVerifyTools,
        types::{SCShardProofVariable, SCVerifyingKeyVariable, UniPolyVariable},
        utils::Utils,
        SCBabyBearFriConfigVariable,
    },
    CircuitConfig,
};
use dt_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Ext, Felt, SymbolicExt},
};
use dt_stark::{
    air::{derive_active_shape_v1, InteractionScope, MachineAir},
    sumcheck::{
        config::SCStarkGenericConfig,
        proof::{SCChipOpenedValues, SCShardCommitment},
        verifier::{OpeningShapeError, Selectors},
    },
    Challenge, InteractionKind, MachineChip, SCStarkMachine, Val,
};
use itertools::{izip, Itertools};
use p3_air::{Air, BaseAir};
use p3_commit::Mmcs;
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use std::iter::zip;

#[derive(Debug, Clone, Copy)]
pub struct SumcheckVerifier<C: Config, SC: SCStarkGenericConfig, A, AE> {
    _phantom: std::marker::PhantomData<(C, SC, A, AE)>,
}

impl<C, SC, A, AE> SumcheckVerifier<C, SC, A, AE>
where
    C::F: TwoAdicField + PrimeField32,
    C: CircuitConfig<F = SC::Val>,
    SC: SCBabyBearFriConfigVariable<C>,
    SC::ValMmcs: Mmcs<Val<SC>, ProverData<RowMajorMatrix<Val<SC>>>: Clone>,
    A: MachineAir<Val<SC>>,
    AE: MachineAir<Challenge<SC>>,
{
    #[allow(clippy::type_complexity)]
    pub fn verify_opening_shape(
        chip: &MachineChip<SC, A>,
        opening: &SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    ) -> Result<(), OpeningShapeError> {
        // Verify that the preprocessed width matches the expected value for the chip.
        if opening.preprocessed.local.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.local.len(),
            ));
        }

        // Verify that the main width matches the expected value for the chip.
        if opening.main.local.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(
                chip.width(),
                opening.main.local.len(),
            ));
        }

        // Verify that the permutation width matches the expected value for the chip.
        if opening.permutation.local.len() != chip.permutation_width() {
            return Err(OpeningShapeError::PermutationWidthMismatch(
                chip.permutation_width(),
                opening.permutation.local.len(),
            ));
        }

        Ok(())
    }
}

// impl verifier (includes verify_sumcheck_unipolys)
impl<C, SC, A, AE> SumcheckVerifier<C, SC, A, AE>
where
    C::F: TwoAdicField,
    C: CircuitConfig<F = SC::Val>,
    SC: SCBabyBearFriConfigVariable<C>,
    SC::ValMmcs: Mmcs<Val<SC>, ProverData<RowMajorMatrix<Val<SC>>>: Clone>,
    A: MachineAir<Val<SC>>,
    AE: MachineAir<Challenge<SC>>,
    Builder<C>: CircuitV2Builder<C>,
{
    fn verify_sumcheck_unipolys(
        builder: &mut Builder<C>,
        claim: &mut Ext<C::F, C::EF>,
        unipolys: &[UniPolyVariable<C>],
        challenger: &mut SC::FriChallengerVariable,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
    ) -> Vec<Ext<C::F, C::EF>> {
        let mut sumcheck_challenges = Vec::with_capacity(unipolys.len());

        unipolys.iter().enumerate().for_each(|(idx, unipoly)| {
            if idx < num_rounds_linear {
                unipoly.observe_into(builder, challenger);
                let challenge = challenger.sample_ext(builder);
                sumcheck_challenges.push(challenge);
                *claim = unipoly.verify_and_evaluate(builder, &challenge, *claim);
            } else {
                // Nonlinear round: claim == sum of evals[0..2^k] (direct addition).
                let mut sum: Ext<_, _> = builder.constant(C::EF::zero());
                for i in 0..(1 << num_skip_rounds) {
                    sum = builder.eval(sum + unipoly.evals[i]);
                }
                builder.assert_ext_eq(sum, *claim);
                unipoly.observe_into(builder, challenger);
                let challenge = challenger.sample_ext(builder);
                sumcheck_challenges.push(challenge);
                *claim = unipoly.evaluate(builder, &challenge);
            }
        });

        sumcheck_challenges
    }

    pub fn verify_shard(
        builder: &mut Builder<C>,
        vk: &SCVerifyingKeyVariable<C, SC>,
        machine: &SCStarkMachine<SC, A, AE>,
        challenger: &mut SC::FriChallengerVariable,
        proof: &SCShardProofVariable<C, SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
    ) where
        A: for<'a> Air<RecursiveSumcheckConstraintFolder<'a, C>>,
    {
        let chips = machine.shard_chips_ordered(&proof.chip_ordering).collect::<Vec<_>>();

        let SCShardProofVariable {
            commitment,
            opened_values,
            opening_proof,
            sumcheck_proof,
            dimensions,
            public_values,
            ..
        } = proof;

        tracing::debug_span!("assert lookup multiplicities").in_scope(|| {
            for kind in InteractionKind::all_kinds() {
                let mut max_lookup_mult = 0u64;
                chips.iter().zip(opened_values.chips.iter()).for_each(|(chip, val)| {
                    max_lookup_mult = max_lookup_mult
                        .checked_add(
                            (chip.num_sends_by_kind(kind) as u64 +
                                chip.num_receives_by_kind(kind) as u64)
                                .checked_mul(1u64.checked_shl(val.log_height as u32).unwrap())
                                .unwrap(),
                        )
                        .unwrap();
                });
                assert!(max_lookup_mult < SC::Val::ORDER_U64, "Lookup multiplicities overflow");
            }
        });

        let mut log_heights = Vec::with_capacity(opened_values.chips.len());
        log_heights.extend(opened_values.chips.iter().map(|val| val.log_height));

        let SCShardCommitment { main_commit, permutation_commit } = *commitment;

        challenger.observe(builder, main_commit);

        let active_shape = derive_active_shape_v1(
            chips
                .iter()
                .zip(log_heights.iter())
                .map(|(chip, &log_height)| (chip.name(), chip.width(), log_height)),
        )
        .expect("recursive active shape must be canonical");
        crate::global_claim::observe_active_shape(builder, challenger, &active_shape);

        let local_permutation_challenges =
            (0..2).map(|_| challenger.sample_ext(builder)).collect::<Vec<_>>();

        if let Some(permutation_commit) = permutation_commit {
            challenger.observe(builder, permutation_commit);
        }

        tracing::debug_span!("observe all cumulative sums").in_scope(|| {
            for (opening, chip) in opened_values.chips.iter().zip_eq(chips.iter()) {
                let local_sum = C::ext2felt(builder, opening.local_cumulative_sum);
                challenger.observe_slice(builder, local_sum);

                let has_local_interactions = chip
                    .sends()
                    .iter()
                    .chain(chip.receives())
                    .any(|i| matches!(i.scope, InteractionScope::Local));
                if !has_local_interactions {
                    let zero_ef: Ext<_, _> = builder.constant(C::EF::zero());
                    builder.assert_ext_eq(opening.local_cumulative_sum, zero_ef);
                }
            }
        });

        let alpha = challenger.sample_ext(builder);

        let max_height = log_heights[0];
        let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold);
        let num_rounds_nonlinear =
            std::cmp::min(max_height, chip_log_height_threshold) / num_skip_rounds;
        let num_rounds = num_rounds_linear + num_rounds_nonlinear;

        let eq_challenges =
            (0..num_rounds).map(|_| challenger.sample_ext(builder)).collect::<Vec<_>>();

        let mut claim: Ext<_, _> = builder.constant(C::EF::zero());

        let sumcheck_challenges_for_linear_and_skip =
            tracing::debug_span!("verify sumcheck unipolys").in_scope(|| {
                Self::verify_sumcheck_unipolys(
                    builder,
                    &mut claim,
                    &sumcheck_proof.unipolys[..num_rounds],
                    challenger,
                    num_rounds_linear,
                    num_skip_rounds,
                )
            });

        // Build extended sumcheck challenges: linear challenges as-is, then each nonlinear
        // challenge expanded to num_skip_rounds challenges via BitExpandPoly
        let mut extended_sumcheck_challenges =
            Vec::with_capacity(num_rounds_linear + num_rounds_nonlinear * num_skip_rounds);
        extended_sumcheck_challenges
            .extend(sumcheck_challenges_for_linear_and_skip[..num_rounds_linear].iter().copied());

        let sumcheck_challenges_nonlinear =
            sumcheck_challenges_for_linear_and_skip[num_rounds_linear..].to_vec();
        let extended_challenges_nonlinear: Vec<Ext<_, _>> =
            Utils::<C, SC>::extend_challenges_with_skips(
                builder,
                &sumcheck_challenges_nonlinear,
                num_skip_rounds,
                chip_log_height_threshold,
            );
        extended_sumcheck_challenges.extend(extended_challenges_nonlinear);

        // opening_point = reversed extended_sumcheck_challenges (matches original verifier)
        let mut opening_point: Vec<Ext<C::F, C::EF>> = extended_sumcheck_challenges.clone();
        opening_point.reverse();

        let permutation_challenges = local_permutation_challenges;

        tracing::debug_span!("verify opening shapes").in_scope(|| {
            for (chip, values) in zip(chips.iter(), opened_values.chips.iter()) {
                Self::verify_opening_shape(chip, values).unwrap();
            }
        });

        let prep_shifts_and_open: Vec<Vec<Ext<C::F, C::EF>>> = opened_values
            .chips
            .iter()
            .filter(|chip| !chip.preprocessed.local.is_empty())
            .map(|chip| chip.preprocessed.to_vec_values())
            .collect();

        let main_shifts_and_open: Vec<Vec<Ext<C::F, C::EF>>> =
            opened_values.chips.iter().map(|chip| chip.main.to_vec_values()).collect();

        let config = machine.config().fri_config();

        if permutation_commit.is_some() {
            let permutation_shifts_and_open: Vec<Vec<Ext<C::F, C::EF>>> =
                opened_values.chips.iter().map(|chip| chip.permutation.to_vec_values()).collect();

            let shifts_and_open =
                vec![prep_shifts_and_open, main_shifts_and_open, permutation_shifts_and_open];

            tracing::debug_span!("basefold pcs").in_scope(|| {
                PcsVerifyTools::verify_basefold_pcs(
                    builder,
                    config,
                    vec![vk.commitment, main_commit, *permutation_commit.as_ref().unwrap()],
                    dimensions,
                    &opening_point,
                    &shifts_and_open,
                    opening_proof,
                    challenger,
                );
            });
        } else {
            let shifts_and_open = vec![prep_shifts_and_open, main_shifts_and_open];

            tracing::debug_span!("basefold pcs").in_scope(|| {
                PcsVerifyTools::verify_basefold_pcs(
                    builder,
                    config,
                    vec![vk.commitment, main_commit],
                    dimensions,
                    &opening_point,
                    &shifts_and_open,
                    opening_proof,
                    challenger,
                );
            });
        }

        tracing::debug_span!("verify polynomial evaluations").in_scope(|| {
            let mut num_constraints = Vec::with_capacity(chips.len());
            num_constraints.extend(chips.iter().map(|chip| {
                *vk.constraints_map.get(&chip.name()).expect("chip not found in constraints map")
            }));

            Self::verify_constraints(
                builder,
                claim,
                &chips,
                num_constraints,
                &opened_values.chips,
                &eq_challenges,
                alpha,
                &extended_sumcheck_challenges,
                &sumcheck_challenges_for_linear_and_skip,
                &permutation_challenges,
                public_values,
                num_skip_rounds,
                num_rounds_nonlinear,
            );
        });

        // Verify local_cumulative_sum equals the expected state imbalance.
        let local_cumulative_sum: Ext<C::F, C::EF> = opened_values
            .chips
            .iter()
            .map(|val| val.local_cumulative_sum)
            .fold(builder.constant(C::EF::zero()), |acc, x| builder.eval(acc + x));

        let expected_imbalance = if machine.has_global_bus() {
            crate::global_claim::expected_local_imbalance(
                builder,
                public_values,
                permutation_challenges[0],
                permutation_challenges[1],
            )
        } else {
            builder.constant(C::EF::zero())
        };
        builder.assert_ext_eq(local_cumulative_sum, expected_imbalance);
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_constraints(
        builder: &mut Builder<C>,
        last_claim: Ext<C::F, C::EF>,
        chips: &[&MachineChip<SC, A>],
        num_constraints: Vec<usize>,
        openings: &[SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>],
        eq_challenges: &[Ext<C::F, C::EF>],
        alpha: Ext<C::F, C::EF>,
        extended_sumcheck_challenges: &[Ext<C::F, C::EF>],
        sumcheck_challenges: &[Ext<C::F, C::EF>],
        permutation_challenges: &[Ext<C::F, C::EF>],
        public_values: &[Felt<C::F>],
        num_skip_rounds: usize,
        num_rounds_nonlinear: usize,
    ) where
        A: for<'a> Air<RecursiveSumcheckConstraintFolder<'a, C>>,
    {
        let degree = (1 << num_skip_rounds) - 1;

        let sumcheck_challenges_rev = sumcheck_challenges.iter().rev().copied().collect::<Vec<_>>();

        let one: Ext<_, _> = builder.constant(C::EF::one());

        // Compute is_first_row and is_last_row, deduplicated by log_height.
        // Chips sharing the same log_height produce identical values.
        let mut selector_cache: std::collections::HashMap<
            usize,
            (Ext<C::F, C::EF>, Ext<C::F, C::EF>),
        > = std::collections::HashMap::new();
        let selectors: Vec<Selectors<Ext<C::F, C::EF>>> = openings
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

        // Calculate alpha_shifts: alpha_shifts[i] = alpha^(sum of num_constraints[0..i])
        // Precompute alpha powers up to the max constraint count and compose shifts
        // by incremental multiplication, avoiding repeated exp_e calls.
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

        // Compute perm_eval independently (matches original verifier)
        let zero_ext: Ext<_, _> = builder.constant(C::EF::zero());
        let mut perm_eval: Ext<C::F, C::EF> = zero_ext;
        for (i, chip_opening) in openings.iter().enumerate() {
            if chip_opening.permutation.local.is_empty() {
                continue;
            }
            // row_sum = sum of permutation.local
            let mut row_sum: Ext<C::F, C::EF> = zero_ext;
            for &val in &chip_opening.permutation.local {
                row_sum = builder.eval(row_sum + val);
            }
            // row_sum -= local_cumulative_sum * (1 << log_height)^{-1}
            assert!(chip_opening.log_height < 31, "log_height too large");
            let inv: C::EF = C::EF::from_canonical_usize(1 << chip_opening.log_height).inverse();
            let inv_ext: Ext<_, _> = builder.constant(inv);
            let adjusted: Ext<_, _> =
                builder.eval(row_sum - chip_opening.local_cumulative_sum * inv_ext);
            let term: Ext<_, _> = builder.eval(adjusted * alpha_shifts[i]);

            perm_eval = builder.eval(perm_eval + term);
        }

        // Calculate the folded evaluations of the constraints (with alpha correction)
        let main_eval = Self::eval_constraints(
            builder,
            chips,
            openings,
            &selectors,
            alpha,
            &alpha_shifts,
            permutation_challenges,
            public_values,
        );

        // Compute eq polynomial: linear rounds use standard eq, nonlinear rounds use EqPoly
        let mut eq = C::eq_poly(
            builder,
            eq_challenges[num_rounds_nonlinear..].to_vec(),
            sumcheck_challenges_rev[num_rounds_nonlinear..].to_vec(),
        );
        for (&r, &s) in eq_challenges[..num_rounds_nonlinear]
            .iter()
            .zip(sumcheck_challenges_rev[..num_rounds_nonlinear].iter())
        {
            let mult = Utils::<C, SC>::calculate_eq(builder, r, s, degree);
            eq = builder.eval(eq * mult);
        }

        // Final verification: main_eval * eq + perm_eval == last_claim
        let main_eq: Ext<_, _> = builder.eval(main_eval * eq);
        let lhs: Ext<_, _> = builder.eval(main_eq + perm_eval);

        builder.assert_ext_eq(lhs, last_claim);
    }

    fn eval_constraints(
        builder: &mut Builder<C>,
        chips: &[&MachineChip<SC, A>],
        openings: &[SCChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>],
        selectors: &[Selectors<Ext<C::F, C::EF>>],
        alpha: Ext<C::F, C::EF>,
        alpha_shifts: &[Ext<C::F, C::EF>],
        permutation_challenges: &[Ext<C::F, C::EF>],
        public_values: &[Felt<C::F>],
    ) -> Ext<C::F, C::EF>
    where
        A: for<'a> Air<RecursiveSumcheckConstraintFolder<'a, C>>,
    {
        const CHUNK_SIZE: usize = 4;
        let all_items: Vec<_> = izip!(chips, openings, selectors, alpha_shifts).collect();
        let mut total: Ext<C::F, C::EF> = builder.constant(C::EF::zero());

        for chunk in all_items.chunks(CHUNK_SIZE) {
            let partial_expr = chunk
                .iter()
                .map(|(&chip, opening, sels, shift)| {
                    let mut folder = RecursiveSumcheckConstraintFolder::<C> {
                        preprocessed: RowMajorMatrixView::new_row(&opening.preprocessed.local),
                        main: RowMajorMatrixView::new_row(&opening.main.local),
                        permutation: RowMajorMatrixView::new_row(&opening.permutation.local),
                        perm_challenges: permutation_challenges,
                        local_cumulative_sum: &opening.local_cumulative_sum,
                        is_first_row: sels.is_first_row,
                        is_last_row: sels.is_last_row,
                        alpha,
                        accumulator: SymbolicExt::zero(),
                        public_values,
                    };

                    chip.eval(&mut folder);

                    if opening.permutation.local.is_empty() {
                        folder.accumulator * SymbolicExt::from(**shift)
                    } else {
                        folder.accumulator * SymbolicExt::from(**shift) * SymbolicExt::from(alpha)
                    }
                })
                .sum::<SymbolicExt<_, _>>();
            let partial: Ext<C::F, C::EF> = builder.eval(partial_expr);
            total = builder.eval(total + partial);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        challenger::CanObserveVariable,
        sumcheck::{
            types::SCVerifyingKeyVariable, verifier::SumcheckVerifier, SCBabyBearFriConfigVariable,
        },
        utils::sc_tests::run_test_recursion,
        witness::{WitnessBlock, Witnessable},
        CircuitConfig,
    };
    use dt_core_machine::{
        shape::{chip_log_height_threshold, num_skip_rounds},
        utils::setup_logger,
    };
    use dt_recursion_compiler::ir::{Builder, Felt};
    use dt_stark::{
        air::MachineAir,
        sumcheck::{
            config::SCStarkGenericConfig,
            prover::{SCMachineProver, SumcheckProver},
            test::{DummyRecord, SimpleAddChip},
        },
        Chip, DTCoreOpts, MachineProver, SCStarkMachine, StarkGenericConfig,
    };
    use log::debug;
    use p3_field::extension::BinomialExtensionField;

    #[cfg(not(feature = "koalabear"))]
    use dt_recursion_compiler::config::InnerConfig;
    #[cfg(not(feature = "koalabear"))]
    use dt_stark::baby_bear_poseidon2::SCBabyBearPoseidon2;
    #[cfg(not(feature = "koalabear"))]
    use dt_stark::{InnerChallenge, InnerVal};

    #[cfg(feature = "koalabear")]
    use dt_recursion_compiler::config::SCInnerConfig as InnerConfig;
    #[cfg(feature = "koalabear")]
    use dt_stark::koalabear_poseidon2::koala_bear_poseidon2::SCKoalaBearPoseidon2 as SCBabyBearPoseidon2;
    #[cfg(feature = "koalabear")]
    use dt_stark::koalabear_poseidon2::{InnerChallenge, InnerVal};

    type F = InnerVal;
    type EF = InnerChallenge;

    #[test]
    fn verify_sumcheck() {
        setup_logger();
        let config = SCBabyBearPoseidon2::new();
        let mut challenger_prover = config.mlchallenger();
        let mut challenger_verifier = challenger_prover.clone();

        // should align to num_skip_rounds() and chip_log_height_threshold() in dt-core-machine
        let num_skip_rounds = num_skip_rounds();
        let chip_log_height_threshold = chip_log_height_threshold();

        let log_heights = [17, 16, 8];
        let (chips, chips_ext): (Vec<_>, Vec<_>) = log_heights
            .iter()
            .enumerate()
            .map(|(index, log_height)| {
                let chip = Chip::<F, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                let chip_ext = Chip::<EF, SimpleAddChip>::new(SimpleAddChip {
                    index,
                    log_height: Some(*log_height),
                    ..Default::default()
                });
                (chip, chip_ext)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        let machine =
            SCStarkMachine::new(config.clone(), chips.clone(), chips_ext.clone(), 0, true);

        let prover = SumcheckProver { machine };

        // Generate pk and vk.
        let (pk, vk) = prover.setup(&<SimpleAddChip as MachineAir<F>>::Program::default());

        // Prove.
        let prove_result = prover.prove(
            &pk,
            vec![DummyRecord::default(); 1],
            &mut challenger_prover,
            DTCoreOpts::default(),
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = prove_result {
            panic!("prove failed: {}", e);
        }
        let proof = prove_result.unwrap();

        // Verify.
        let verify_result = prover.machine().verify(
            &vk,
            &proof,
            &mut challenger_verifier,
            num_skip_rounds,
            chip_log_height_threshold,
        );
        if let Err(e) = verify_result {
            panic!("verify failed: {}", e);
        }

        debug!("verify success");

        let mut builder = Builder::<InnerConfig>::default();
        let mut witness_stream = Vec::<WitnessBlock<InnerConfig>>::new();

        // Add a hash invocation, since the poseidon2 table expects that it's in the first row.
        let mut challenger = config.clone().challenger_variable(&mut builder);
        Witnessable::<InnerConfig>::write(&vk, &mut witness_stream);
        let vk: SCVerifyingKeyVariable<_, _> = vk.read(&mut builder);
        vk.observe_into(&mut builder, &mut challenger);

        debug!("proofs len : {}", proof.shard_proofs.len());
        let proofs = proof.shard_proofs.read(&mut builder);
        Witnessable::<InnerConfig>::write(&proof.shard_proofs, &mut witness_stream);

        // Verify each shard proof using the circuit verifier.
        for proof in proofs.into_iter() {
            let mut challenger = challenger.copy(&mut builder);
            let machine =
                SCStarkMachine::new(config.clone(), chips.clone(), chips_ext.clone(), 0, true);
            let pv_slice = &proof.public_values[..machine.num_pv_elts()];
            challenger.observe_slice(&mut builder, pv_slice.iter().cloned());
            debug!("start recursion verify");
            SumcheckVerifier::verify_shard(
                &mut builder,
                &vk,
                &machine,
                &mut challenger,
                &proof,
                num_skip_rounds,
                chip_log_height_threshold,
            );
        }
        debug!("end recursion verify");

        run_test_recursion(
            builder.into_root_block(),
            witness_stream,
            num_skip_rounds,
            chip_log_height_threshold,
        );
    }
}
