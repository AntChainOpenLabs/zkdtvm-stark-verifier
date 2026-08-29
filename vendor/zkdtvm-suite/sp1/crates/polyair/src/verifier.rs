use crate::{evaluator::uinit_vec, precompute::PrecomputeRowBuilder, Chip};
use dt_stark::{
    air::{
        derive_active_shape_v1, observe_active_shape_v1, FullAir, FullAirBuilder, MachineAir,
        PairCol, PolyAirExtendable, PublicValues,
    },
    global_d11::{compute_expected_global_claim_imbalance_v2, validate_global_claim},
    machine::compute_expected_state_imbalance,
    septic_curve_params::compute_beta_septix,
    sumcheck::{
        config::{MlCom, MlPcsOpeningProof, MlPcsVerificationTrace, SCStarkGenericConfig},
        keys::SCStarkVerifyingKey,
        proof::{SCChipOpenedValues, SCShardCommitment, SCShardProof},
        types::{EqPoly, UniPolyEvals},
        verifier::{OpeningShapeError, SumcheckVerificationError},
    },
    Challenge, Val, Word,
};
use itertools::izip;
use p3_air::BaseAir;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrixView;
use pcs::basefold::mlpcs::MlPCS;
use std::{borrow::Borrow, marker::PhantomData, mem::take, slice};
pub struct Selectors<T> {
    pub is_first_row: T,
    pub is_last_row: T,
}

/// Compact semantic facts produced by the mandatory sumcheck verification pass.
///
/// These are deliberately not trace rows.  A recorder can retain them as the
/// authenticated source for later trace generation without evaluating the same
/// univariate polynomial a second time after verification.
#[derive(Clone, Debug)]
pub struct VerifiedSumcheckRound<Challenge> {
    pub round_idx: usize,
    pub evals: Vec<Challenge>,
    pub challenge: Challenge,
    pub claim_in: Challenge,
    pub claim_out: Challenge,
}

/// Successful first-pass verifier output used by native-recursion recording.
/// Construction is local to `verify_shard_with_semantics`; it is returned only
/// after PCS, constraint, and cumulative-sum verification have all succeeded.
#[derive(Clone, Debug)]
pub struct VerifiedShardSemantics<Val, Challenge, PcsVerification> {
    pub _value_marker: PhantomData<Val>,
    pub local_cumulative_sums: Vec<Challenge>,
    pub perm_alpha: Challenge,
    pub perm_beta: Challenge,
    pub alpha: Challenge,
    pub eq_challenges: Vec<Challenge>,
    pub sumcheck_rounds: Vec<VerifiedSumcheckRound<Challenge>>,
    pub last_claim: Challenge,
    pub pcs_verification: PcsVerification,
}

pub struct Verifier<SC, A, const D: usize>(PhantomData<SC>, PhantomData<A>);

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>, const D: usize> Verifier<SC, A, D>
where
    Val<SC>: PolyAirExtendable<D> + PrimeField32,
    A: for<'a> FullAir<PrecomputeRowBuilder<'a, Val<SC>, Challenge<SC>, Challenge<SC>>>,
    A: for<'a> FullAir<SumcheckVerifierConstraintFolder<'a, Val<SC>, Challenge<SC>>>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn verify_shard<ProofSC>(
        config: &SC,
        vk: &SCStarkVerifyingKey<ProofSC>,
        chips: &[&Chip<A, Val<SC>, D>],
        challenger: &mut SC::MlChallenger,
        proof: &SCShardProof<ProofSC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        contains_global_bus: bool,
    ) -> Result<(), SumcheckVerificationError<SC>>
    where
        ProofSC: SCStarkGenericConfig<Val = Val<SC>, Challenge = Challenge<SC>>,
        ProofSC::Mlpcs: MlPCS<Commitment = MlCom<SC>, BatchProof = MlPcsOpeningProof<SC>>,
    {
        Self::verify_shard_with_semantics(
            config,
            vk,
            chips,
            challenger,
            proof,
            num_skip_rounds,
            chip_log_height_threshold,
            contains_global_bus,
        )?;
        Ok(())
    }

    /// Verify a child shard once and return the compact semantic facts computed
    /// by that same pass.  The returned value is the commit point: callers must
    /// not publish partial recorder state before this function succeeds.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_shard_with_semantics<ProofSC>(
        config: &SC,
        vk: &SCStarkVerifyingKey<ProofSC>,
        chips: &[&Chip<A, Val<SC>, D>],
        challenger: &mut SC::MlChallenger,
        proof: &SCShardProof<ProofSC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        contains_global_bus: bool,
    ) -> Result<
        VerifiedShardSemantics<Val<SC>, Challenge<SC>, MlPcsVerificationTrace<SC>>,
        SumcheckVerificationError<SC>,
    >
    where
        ProofSC: SCStarkGenericConfig<Val = Val<SC>, Challenge = Challenge<SC>>,
        ProofSC::Mlpcs: MlPCS<Commitment = MlCom<SC>, BatchProof = MlPcsOpeningProof<SC>>,
    {
        assert!(chip_log_height_threshold == 0);
        let SCShardProof {
            commitment,
            opened_values,
            sumcheck_proof,
            public_values,
            opening_proof,
            dimensions,
            ..
        } = proof;

        let pcs = config.mlpcs();

        if chips.len() != opened_values.chips.len() {
            return Err(SumcheckVerificationError::ChipOpeningLengthMismatch);
        }

        let log_heights = opened_values.chips.iter().map(|val| val.log_height).collect::<Vec<_>>();

        // TODO: check LookupMultiplicityOverflow

        let phase_pre_pcs = crate::Instant::now();
        let SCShardCommitment { main_commit, permutation_commit } = commitment;
        challenger.observe(main_commit.clone());
        let active_shape = derive_active_shape_v1(
            chips
                .iter()
                .zip(opened_values.chips.iter())
                .map(|(chip, opening)| (chip.name(), chip.width(), opening.log_height)),
        )
        .map_err(|_| SumcheckVerificationError::CumulativeSumsError("invalid active shape"))?;
        observe_active_shape_v1::<Val<SC>, _>(challenger, &active_shape);

        let global_claim = if contains_global_bus {
            let pv: &PublicValues<Word<Val<SC>>, Val<SC>> = public_values.as_slice().borrow();
            let claim = pv.global;
            let owners = chips.iter().filter(|chip| chip.global_boundary_owner().is_some()).count();
            if owners > 1 {
                return Err(SumcheckVerificationError::CumulativeSumsError(
                    "duplicate Global opening",
                ));
            }
            validate_global_claim(&claim, owners == 1).map_err(|_| {
                SumcheckVerificationError::CumulativeSumsError(
                    "Global claim/opening admission failed",
                )
            })?;
            claim
        } else {
            Default::default()
        };

        for (chip, opening) in chips.iter().zip(opened_values.chips.iter()) {
            let local_sum = opening.local_cumulative_sum;
            if chip.num_lookup() == 0 && !local_sum.is_zero() {
                return Err(SumcheckVerificationError::CumulativeSumsError(
                    "local cumulative sum is non-zero, but no local interactions",
                ));
            }
        }

        let perm_alpha: Challenge<SC> = challenger.sample_ext_element();
        let perm_beta: Challenge<SC> = challenger.sample_ext_element();
        let max_beta_powers = chips.iter().map(|c| c.required_max_beta_power()).max().unwrap_or(0);
        let beta_powers = {
            let mut res = Vec::with_capacity(max_beta_powers + 1);
            let mut powers = perm_beta.powers();
            for _ in 0..(max_beta_powers + 1) {
                res.push(powers.next().unwrap());
            }
            res
        };
        #[cfg(feature = "koalabear")]
        let beta_septix = compute_beta_septix::<
            Val<SC>,
            Challenge<SC>,
            dt_stark::septic_curve_params::KoalaBearCurveParams,
        >(perm_beta);
        #[cfg(feature = "babybear")]
        let beta_septix = compute_beta_septix::<
            Val<SC>,
            Challenge<SC>,
            dt_stark::septic_curve_params::BabyBearCurveParams,
        >(perm_beta);

        if permutation_commit.is_some() {
            challenger.observe(permutation_commit.as_ref().unwrap().clone());
        }

        for opening in &opened_values.chips {
            <SC::MlChallenger as CanObserve<Val<SC>>>::observe_slice(
                challenger,
                opening.local_cumulative_sum.as_base_slice(),
            );
        }

        let alpha = challenger.sample_ext_element::<Challenge<SC>>();

        let max_height = log_heights[0];
        let num_rounds_linear = max_height.saturating_sub(chip_log_height_threshold);
        let num_rounds_nonlinear =
            std::cmp::min(max_height, chip_log_height_threshold) / num_skip_rounds;
        let num_rounds = num_rounds_linear + num_rounds_nonlinear;

        let eq_challenges: Vec<Challenge<SC>> =
            (0..num_rounds).map(|_| challenger.sample_ext_element()).collect();

        let mut claim = Challenge::<SC>::zero();

        let (sumcheck_challenges, sumcheck_rounds) = Self::verify_sumcheck_unipolys(
            &mut claim,
            &sumcheck_proof.unipolys[..num_rounds],
            challenger,
            num_rounds_linear,
            num_skip_rounds,
        )?;
        // Keep one owned verifier challenge vector. The PCS consumes a reverse
        // indexed view materialized once; constraint verification reuses both
        // orders instead of clone -> reverse -> clone.
        let opening_point = sumcheck_challenges.iter().rev().copied().collect::<Vec<_>>();

        for (chip, values) in chips.iter().zip(opened_values.chips.iter()) {
            Self::verify_opening_shape(chip, values)
                .map_err(|e| SumcheckVerificationError::OpeningShapeError(chip.name(), e))?;
        }

        let prep_opened_values: Vec<Vec<Challenge<SC>>> = opened_values
            .chips
            .iter()
            .filter(|chip| !chip.preprocessed.local.is_empty())
            .map(|chip| chip.preprocessed.to_vec_values())
            .collect();
        let main_opened_values: Vec<Vec<Challenge<SC>>> =
            opened_values.chips.iter().map(|chip| chip.main.to_vec_values()).collect();
        pcs::whir::profile::add_ms("verify.shard_pre_pcs_us", phase_pre_pcs.elapsed().as_micros());
        let phase_pcs = crate::Instant::now();
        let pcs_verification = if permutation_commit.is_some() {
            let permutation_opened_values: Vec<Vec<Challenge<SC>>> =
                opened_values.chips.iter().map(|chip| chip.permutation.to_vec_values()).collect();
            let pcs_opened_values =
                vec![prep_opened_values, main_opened_values, permutation_opened_values];
            pcs.verify(
                vec![
                    vk.commit.clone(),
                    main_commit.clone(),
                    permutation_commit.as_ref().unwrap().clone(),
                ],
                dimensions,
                &opening_point,
                &pcs_opened_values,
                opening_proof,
                challenger,
            )
            .map_err(|e| SumcheckVerificationError::MlPcsOpeningError(format!("{e:?}")))?
        } else {
            let pcs_opened_values = vec![prep_opened_values, main_opened_values];
            pcs.verify(
                vec![vk.commit.clone(), main_commit.clone()],
                dimensions,
                &opening_point,
                &pcs_opened_values,
                opening_proof,
                challenger,
            )
            .map_err(|e| SumcheckVerificationError::MlPcsOpeningError(format!("{e:?}")))?
        };

        pcs::whir::profile::add_ms("verify.shard_pcs_us", phase_pcs.elapsed().as_micros());
        let phase_constraints = crate::Instant::now();
        let num_constraints = chips
            .iter()
            .map(|chip| {
                *vk.constraints_map.get(&chip.name()).expect("chip not found in constraints map")
            })
            .collect::<Vec<usize>>();
        Self::verify_constraints(
            claim,
            chips,
            num_constraints,
            &opened_values.chips,
            &eq_challenges,
            alpha,
            &sumcheck_challenges,
            &opening_point,
            perm_alpha,
            &beta_powers,
            beta_septix,
            public_values,
            num_skip_rounds,
            num_rounds_nonlinear,
        )?;
        pcs::whir::profile::add_ms(
            "verify.shard_constraints_us",
            phase_constraints.elapsed().as_micros(),
        );

        let permutation_challenges = [perm_alpha, perm_beta];
        let expected_local_sum = if contains_global_bus {
            let mut expected =
                compute_expected_state_imbalance::<SC>(public_values, &permutation_challenges);
            expected += compute_expected_global_claim_imbalance_v2(
                permutation_challenges[0],
                permutation_challenges[1],
                &global_claim,
            )
            .map_err(|_| {
                SumcheckVerificationError::CumulativeSumsError(
                    "invalid projective Global-chain boundary",
                )
            })?;
            expected
        } else {
            Challenge::<SC>::zero()
        };
        let local_cumulative_sum = proof.local_cumulative_sum();
        if local_cumulative_sum != expected_local_sum {
            return Err(SumcheckVerificationError::CumulativeSumsError(
                "local cumulative sum does not match expected state imbalance",
            ));
        }

        Ok(VerifiedShardSemantics {
            _value_marker: PhantomData,
            local_cumulative_sums: opened_values
                .chips
                .iter()
                .map(|opening| opening.local_cumulative_sum)
                .collect(),
            perm_alpha,
            perm_beta,
            alpha,
            eq_challenges,
            sumcheck_rounds,
            last_claim: claim,
            pcs_verification,
        })
    }

    fn verify_sumcheck_unipolys(
        claim: &mut Challenge<SC>,
        unipolys: &[UniPolyEvals<Challenge<SC>>],
        challenger: &mut SC::MlChallenger,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
    ) -> Result<
        (Vec<Challenge<SC>>, Vec<VerifiedSumcheckRound<Challenge<SC>>>),
        SumcheckVerificationError<SC>,
    > {
        let mut sumcheck_challenges = Vec::with_capacity(unipolys.len());
        let mut verified_rounds = Vec::with_capacity(unipolys.len());
        let mut challenge = Challenge::<SC>::zero();
        unipolys.iter().enumerate().try_for_each(|(idx, unipoly)| {
            let claim_in = *claim;
            if idx < num_rounds_linear {
                let sum = unipoly.evals[0] + unipoly.evals[1];
                if sum != *claim {
                    return Err(SumcheckVerificationError::SumcheckUniPolyError);
                }
            } else {
                let full_sum: Challenge<SC> =
                    unipoly.evals[..1 << num_skip_rounds].iter().copied().sum();
                if full_sum != *claim {
                    return Err(SumcheckVerificationError::SumcheckUniPolyError);
                }
            }
            unipoly.evals.iter().for_each(|eval| {
                <SC::MlChallenger as CanObserve<SC::Val>>::observe_slice(
                    challenger,
                    eval.as_base_slice(),
                );
            });
            challenge = challenger.sample_ext_element();
            sumcheck_challenges.push(challenge);
            *claim = unipoly.eval_at_point(challenge);
            verified_rounds.push(VerifiedSumcheckRound {
                round_idx: idx,
                evals: unipoly.evals.clone(),
                challenge,
                claim_in,
                claim_out: *claim,
            });
            Ok(())
        })?;
        Ok((sumcheck_challenges, verified_rounds))
    }

    fn verify_opening_shape(
        chip: &Chip<A, Val<SC>, D>,
        opening: &SCChipOpenedValues<Val<SC>, Challenge<SC>>,
    ) -> Result<(), OpeningShapeError> {
        if opening.preprocessed.local.len() != chip.preprocessed_width() {
            return Err(OpeningShapeError::PreprocessedWidthMismatch(
                chip.preprocessed_width(),
                opening.preprocessed.local.len(),
            ));
        }
        if opening.main.local.len() != chip.width() {
            return Err(OpeningShapeError::MainWidthMismatch(
                chip.width(),
                opening.main.local.len(),
            ));
        }
        if opening.permutation.local.len() != chip.perm_width() {
            return Err(OpeningShapeError::PermutationWidthMismatch(
                chip.perm_width(),
                opening.permutation.local.len(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_constraints(
        last_claim: Challenge<SC>,
        chips: &[&Chip<A, Val<SC>, D>],
        num_constraints: Vec<usize>,
        openings: &[SCChipOpenedValues<Val<SC>, Challenge<SC>>],
        eq_challenges: &[Challenge<SC>],
        alpha: Challenge<SC>,
        sumcheck_challenges_forward: &[Challenge<SC>],
        opening_point: &[Challenge<SC>],
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
        num_skip_rounds: usize,
        num_rounds_nonlinear: usize,
    ) -> Result<(), SumcheckVerificationError<SC>> {
        let (is_first_row, is_last_row): (Vec<Challenge<SC>>, Vec<Challenge<SC>>) = openings
            .iter()
            .map(|opening| {
                let start = sumcheck_challenges_forward.len() - opening.log_height;
                let (first, last) = sumcheck_challenges_forward[start..]
                    .iter()
                    .fold((Challenge::<SC>::one(), Challenge::<SC>::one()), |(first, last), &x| {
                        (first * (Challenge::<SC>::one() - x), last * x)
                    });
                (first, last)
            })
            .unzip();

        let selectors = is_first_row
            .into_iter()
            .zip(is_last_row)
            .map(|(is_first_row, is_last_row)| Selectors { is_first_row, is_last_row })
            .collect::<Vec<_>>();

        let mut alpha_shifts = Vec::with_capacity(chips.len());
        alpha_shifts.push(Challenge::<SC>::one());
        for num in num_constraints.into_iter().take(chips.len().saturating_sub(1)) {
            alpha_shifts.push(alpha_shifts[alpha_shifts.len() - 1] * alpha.exp_u64(num as u64));
        }

        let perm_eval: Challenge<SC> = openings
            .iter()
            .enumerate()
            .map(|(i, chip)| {
                if chip.permutation.local.is_empty() {
                    return Challenge::<SC>::zero();
                }
                let mut row_sum: Challenge<SC> = chip.permutation.local.iter().copied().sum();
                assert!(chip.log_height < 31, "log_height too large");
                row_sum -= chip.local_cumulative_sum *
                    Val::<SC>::from_canonical_usize(1 << chip.log_height).inverse();
                row_sum * alpha_shifts[i]
            })
            .sum();

        let main_eval = Self::eval_constraints(
            chips,
            openings,
            &selectors,
            alpha,
            alpha_shifts,
            perm_alpha,
            beta_powers,
            beta_septix,
            public_values,
        );

        let mut eq = calculate_eq(
            &eq_challenges[num_rounds_nonlinear..],
            &opening_point[num_rounds_nonlinear..],
        );
        let degree = (1 << num_skip_rounds) - 1;
        for (&r, &s) in eq_challenges[..num_rounds_nonlinear]
            .iter()
            .zip(opening_point[..num_rounds_nonlinear].iter())
        {
            eq *= EqPoly::<Val<SC>, Challenge<SC>>::eval_eq(r, s, degree);
        }

        let lhs = main_eval * eq + perm_eval;
        if lhs == last_claim {
            Ok(())
        } else {
            tracing::error!("[VERIFY FAIL] EvaluationsInconsistent");
            tracing::error!("  lhs (main_eval*eq + perm_eval) = {:?}", lhs);
            tracing::error!("  last_claim                     = {:?}", last_claim);
            tracing::error!("  main_eval                      = {:?}", main_eval);
            tracing::error!("  perm_eval                      = {:?}", perm_eval);
            tracing::error!("  eq                             = {:?}", eq);
            tracing::error!("  chips: {:?}", chips.iter().map(|c| c.name()).collect::<Vec<_>>());
            Err(SumcheckVerificationError::EvaluationsInconsistent)
        }
    }

    fn eval_constraints(
        chips: &[&Chip<A, Val<SC>, D>],
        openings: &[SCChipOpenedValues<Val<SC>, Challenge<SC>>],
        selectors: &[Selectors<Challenge<SC>>],
        alpha: Challenge<SC>,
        alpha_shifts: Vec<Challenge<SC>>,
        perm_alpha: Challenge<SC>,
        beta_powers: &[Challenge<SC>],
        beta_septix: Challenge<SC>,
        public_values: &[Val<SC>],
    ) -> Challenge<SC> {
        izip!(chips, openings, selectors, &alpha_shifts)
            .map(|(chip, opening, sels, &shift)| {
                let mut precompute_lc = uinit_vec::<Challenge<SC>>(chip.num_precompute());
                let preprocessed = &opening.preprocessed.local[..];
                let main = &opening.main.local[..];
                let mut builder = PrecomputeRowBuilder::<Val<SC>, Challenge<SC>, Challenge<SC>> {
                    preprocessed,
                    main,
                    beta_powers,
                    public: public_values,
                    alpha: perm_alpha,
                    beta_septix,
                    row: &mut precompute_lc,
                    col_index: 0,
                };
                chip.air.precompute_lc(&mut builder);
                let reserved_poly: Vec<Challenge<SC>> = chip
                    .reserved_poly()
                    .into_iter()
                    .map(|i| match i {
                        PairCol::Prep(i) => preprocessed[*i],
                        PairCol::Main(i) => main[*i],
                    })
                    .collect();
                let mut accumulator = Challenge::<SC>::zero();
                let mut folder = SumcheckVerifierConstraintFolder::<Val<SC>, Challenge<SC>> {
                    public: public_values,
                    alpha: perm_alpha,
                    beta_powers,
                    beta_septix,
                    precomputed: RowMajorMatrixView::new_row(&precompute_lc),
                    reserved_poly: RowMajorMatrixView::new_row(&reserved_poly),
                    is_first_row: sels.is_first_row,
                    is_last_row: sels.is_last_row,
                    local_sum: opening.local_cumulative_sum,
                    permutation: RowMajorMatrixView::new_row(&opening.permutation.local),
                    multiplicitys: vec![],
                    batch_size: chip.logup_batch_size(),
                    accumulator: &mut accumulator,
                    constraint_reducer: alpha,
                };
                chip.air.eval(&mut folder);
                chip.air.lookup(&mut folder);
                folder.constrain_lookup();
                let contribution = if opening.permutation.local.is_empty() {
                    accumulator * shift
                } else {
                    accumulator * shift * alpha
                };
                contribution
            })
            .sum()
    }
}

pub(crate) fn calculate_eq<F: Field>(r: &[F], s: &[F]) -> F {
    debug_assert_eq!(r.len(), s.len(), "r and s must have the same length");
    r.iter().zip(s).fold(F::one(), |acc, (&r, &s)| acc * ((r * s).double() - r - s + F::one()))
}

pub struct SumcheckVerifierConstraintFolder<'a, F: Field, EF: Field + ExtensionField<F>> {
    /// Public values.
    pub public: &'a [F],
    /// The alpha challenge.
    pub alpha: EF,
    /// Powers of the beta challenge.
    pub beta_powers: &'a [EF],
    /// Beta raised to the 7th power in the septic extension.
    pub beta_septix: EF,
    /// Precomputed linear combinations (no next row).
    pub precomputed: RowMajorMatrixView<'a, EF>,
    /// Reserved polynomial values (current and next row).
    pub reserved_poly: RowMajorMatrixView<'a, EF>,
    /// Indicator for the first row.
    pub is_first_row: EF,
    /// Indicator for the last row.
    pub is_last_row: EF,
    /// Indicator for transition rows.

    /// The local cumulative sum.
    pub local_sum: EF,
    /// Permutation trace (no next row).
    pub permutation: RowMajorMatrixView<'a, EF>,
    /// Cached multiplicities from lookup operations.
    pub multiplicitys: Vec<EF>,
    /// Number of lookups per batch.
    pub batch_size: usize,

    /// Accumulator for constraint values.
    pub accumulator: &'a mut EF,
    /// Coefficients for combining constraints.
    pub constraint_reducer: EF,
}

impl<'a, F: Field, EF: Field + ExtensionField<F>> SumcheckVerifierConstraintFolder<'a, F, EF> {
    /// Verify lookup constraints.
    ///
    /// This method is called after `air.lookup()` and `air.eval()` to verify
    /// that the permutation trace running products match the precomputed
    /// lookup denominators.
    pub fn constrain_lookup(&mut self) {
        let perm_local = unsafe {
            slice::from_raw_parts(self.permutation.values.as_ptr(), self.permutation.values.len())
        };
        let multiplicitys = take(&mut self.multiplicitys);
        let values =
            unsafe { slice::from_raw_parts(self.precomputed.values.as_ptr(), multiplicitys.len()) };

        // Verify each batch of lookups
        for (lookup_index, (value, multiplicity)) in
            values.chunks(self.batch_size).zip(multiplicitys.chunks(self.batch_size)).enumerate()
        {
            if self.batch_size == 2 {
                let residual = match (value, multiplicity) {
                    ([d0, d1], [m0, m1]) => {
                        *d1 * *m0 + *d0 * (*m1 - *d1 * perm_local[lookup_index])
                    }
                    ([d0], [m0]) => *m0 - *d0 * perm_local[lookup_index],
                    _ => unreachable!("batch-size-two chunks contain one or two lookups"),
                };
                self.assert_zero_ext(residual);
                continue;
            }
            let denumerator: EF = value.iter().copied().product();
            let mut numerator = EF::zero();
            for (i, m) in multiplicity.iter().copied().enumerate() {
                let mut all_but_current = EF::one();
                for other_rlc in
                    value.iter().enumerate().filter(|(j, _)| i != *j).map(|(_, rlc)| rlc)
                {
                    all_but_current = all_but_current.clone() * other_rlc.clone();
                }
                numerator += all_but_current * m;
            }
            self.assert_eq_ext(numerator, denumerator * perm_local[lookup_index]);
        }
    }
}

impl<'a, F: Field, EF: Field + ExtensionField<F>> FullAirBuilder
    for SumcheckVerifierConstraintFolder<'a, F, EF>
{
    type F = F;
    type EF = EF;
    type VarBase = F;
    type VarMaybeExt = EF;
    type VarExt = EF;
    type MatMaybeExt = RowMajorMatrixView<'a, EF>;
    type MatExt = RowMajorMatrixView<'a, EF>;

    fn preprocessed(&self) -> &[Self::VarMaybeExt] {
        unreachable!("preprocessed should not be used in constraint evaluation phase")
    }

    fn main(&self) -> &[Self::VarMaybeExt] {
        unreachable!("main should not be used in constraint evaluation phase")
    }

    fn public(&self) -> &[Self::VarBase] {
        self.public
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

    fn retain_precomputed(&mut self, _: Self::VarExt) {
        unreachable!("retain_precomputed should not be called in constraint evaluation phase")
    }

    fn precomputed(&self) -> Self::MatExt {
        self.precomputed
    }

    fn reserved_poly(&self) -> Self::MatMaybeExt {
        self.reserved_poly
    }

    fn local_lookup(&mut self, multiplicity: Self::VarMaybeExt, is_send: bool) {
        let multiplicity = if is_send { multiplicity } else { multiplicity.neg() };
        self.multiplicitys.push(multiplicity);
    }

    fn is_first_row(&self) -> Self::VarMaybeExt {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::VarMaybeExt {
        self.is_last_row
    }

    fn is_transition(&self) -> Self::VarMaybeExt {
        unreachable!("is_transition is deprecated")
    }

    fn mul_base(a: Self::VarMaybeExt, b: Self::F) -> Self::VarMaybeExt {
        a * b
    }

    fn from_ef(ef: Self::EF) -> Self::VarExt {
        ef
    }

    fn assert_zero<I: Into<Self::VarMaybeExt>>(&mut self, x: I) {
        let x = x.into();
        *self.accumulator = *self.accumulator * self.constraint_reducer + x;
    }

    fn assert_zero_ext<I: Into<Self::VarExt>>(&mut self, x: I) {
        let x = x.into();
        *self.accumulator = *self.accumulator * self.constraint_reducer + x;
    }
}
