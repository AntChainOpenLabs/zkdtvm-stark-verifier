use hashbrown::HashMap;
use itertools::{izip, Itertools};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{AbstractExtensionField, AbstractField, Field, PrimeField32, PrimeField64};
use p3_matrix::dense::RowMajorMatrixView;
use pcs::basefold::mlpcs::MlPCS;
use std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter},
    iter::zip,
    marker::PhantomData,
};

use crate::{
    air::{
        derive_active_shape_v1, observe_active_shape_v1, InteractionScope, MachineAir, PublicValues,
    },
    global_d11::{compute_expected_global_claim_imbalance_v2, validate_global_claim},
    machine::compute_expected_state_imbalance,
    sumcheck::{
        config::SCStarkGenericConfig,
        folder::SumcheckVerifierConstraintFolder,
        keys::SCStarkVerifyingKey,
        proof::{SCChipOpenedValues, SCShardCommitment, SCShardProof},
        types::{BitExpandPoly, EqPoly, UniPolyEvals},
    },
    InteractionKind, MachineChip, OpeningError, StarkGenericConfig, Val, Word,
};

use crate::config::Challenge;

pub struct Selectors<T> {
    pub is_first_row: T,
    pub is_last_row: T,
}

/// A verifier for a collection of air chips.
pub struct Verifier<SC, A>(PhantomData<SC>, PhantomData<A>);

impl<SC: SCStarkGenericConfig, A: MachineAir<Val<SC>>> Verifier<SC, A> {
    /// Verify a proof for a collection of air chips.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn verify_shard(
        config: &SC,
        vk: &SCStarkVerifyingKey<SC>,
        chips: &[&MachineChip<SC, A>],
        challenger: &mut SC::MlChallenger,
        proof: &SCShardProof<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        contains_global_bus: bool,
    ) -> Result<(), SumcheckVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        A: for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        let interaction_kinds = InteractionKind::all_kinds();
        Self::verify_shard_with_interaction_kinds(
            config,
            vk,
            chips,
            challenger,
            proof,
            num_skip_rounds,
            chip_log_height_threshold,
            contains_global_bus,
            &interaction_kinds,
        )
    }

    /// Verify a native-recursion shard proof, checking recursion-only interaction kinds.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn verify_shard_with_recursion_interactions(
        config: &SC,
        vk: &SCStarkVerifyingKey<SC>,
        chips: &[&MachineChip<SC, A>],
        challenger: &mut SC::MlChallenger,
        proof: &SCShardProof<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        contains_global_bus: bool,
    ) -> Result<(), SumcheckVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        A: for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        Self::verify_shard_with_interaction_kinds(
            config,
            vk,
            chips,
            challenger,
            proof,
            num_skip_rounds,
            chip_log_height_threshold,
            contains_global_bus,
            InteractionKind::recursion_kinds(),
        )
    }

    /// Verify a proof for a collection of air chips with an explicit active interaction kind set.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn verify_shard_with_interaction_kinds(
        config: &SC,
        vk: &SCStarkVerifyingKey<SC>,
        chips: &[&MachineChip<SC, A>],
        challenger: &mut SC::MlChallenger,
        proof: &SCShardProof<SC>,
        num_skip_rounds: usize,
        chip_log_height_threshold: usize,
        contains_global_bus: bool,
        interaction_kinds: &[InteractionKind],
    ) -> Result<(), SumcheckVerificationError<SC>>
    where
        Val<SC>: PrimeField32,
        A: for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>,
    {
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

        for &kind in interaction_kinds {
            let mut max_lookup_mult = 0u64;
            chips.iter().zip(opened_values.chips.iter()).for_each(|(chip, val)| {
                max_lookup_mult = max_lookup_mult.saturating_add(
                    (chip.num_sends_by_kind(kind) as u64 + chip.num_receives_by_kind(kind) as u64)
                        .saturating_mul(1u64 << val.log_height),
                );
            });
            if max_lookup_mult >= SC::Val::ORDER_U64 {
                return Err(SumcheckVerificationError::LookupMultiplicityOverflow);
            }
        }

        let SCShardCommitment { main_commit, permutation_commit } = commitment;

        challenger.observe_slice(public_values);
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

        let local_permutation_challenges =
            (0..2).map(|_| challenger.sample_ext_element::<Challenge<SC>>()).collect::<Vec<_>>();

        if permutation_commit.is_some() {
            challenger.observe(permutation_commit.as_ref().unwrap().clone());
        }

        for (opening, chip) in opened_values.chips.iter().zip_eq(chips.iter()) {
            let local_sum = opening.local_cumulative_sum;

            <SC::MlChallenger as CanObserve<Val<SC>>>::observe_slice(
                challenger,
                local_sum.as_base_slice(),
            );

            let has_local_interactions = chip
                .sends()
                .iter()
                .chain(chip.receives())
                .any(|i| i.scope == InteractionScope::Local);
            if !has_local_interactions && !local_sum.is_zero() {
                return Err(SumcheckVerificationError::CumulativeSumsError(
                    "local cumulative sum is non-zero, but no local interactions",
                ));
            }
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

        let sumcheck_challenges =
            tracing::debug_span!("verify sumcheck unipolys v2").in_scope(|| {
                Self::verify_sumcheck_unipolys(
                    &mut claim,
                    &sumcheck_proof.unipolys[..num_rounds],
                    challenger,
                    num_rounds_linear,
                    num_skip_rounds,
                )
            })?;

        // extended_sumcheck_challenges: first num_rounds_linear as-is, then each nonlinear
        // challenge expanded to num_skip_rounds challenges via BitExpandPoly
        let bit_expand_poly = BitExpandPoly::new(
            (0..(1 << num_skip_rounds)).map(Val::<SC>::from_canonical_usize).collect(),
        );
        let mut extended_sumcheck_challenges = Vec::with_capacity(max_height);
        extended_sumcheck_challenges
            .extend(sumcheck_challenges[..num_rounds_linear].iter().copied());
        for challenge in &sumcheck_challenges[num_rounds_linear..] {
            extended_sumcheck_challenges.extend(bit_expand_poly.evals_all(*challenge));
        }

        let mut opening_point: Vec<Challenge<SC>> = extended_sumcheck_challenges.clone();
        opening_point.reverse();

        let permutation_challenges = local_permutation_challenges;

        for (chip, values) in zip(chips.iter(), opened_values.chips.iter()) {
            Self::verify_opening_shape(chip, values)
                .map_err(|e| SumcheckVerificationError::OpeningShapeError(chip.name(), e))?;
        }

        tracing::debug_span!("verify opening proof v2").in_scope(
            || -> Result<(), SumcheckVerificationError<SC>> {
                let prep_opened_values: Vec<Vec<Challenge<SC>>> = opened_values
                    .chips
                    .iter()
                    .filter(|chip| !chip.preprocessed.local.is_empty())
                    .map(|chip| chip.preprocessed.to_vec_values())
                    .collect();

                let main_opened_values: Vec<Vec<Challenge<SC>>> =
                    opened_values.chips.iter().map(|chip| chip.main.to_vec_values()).collect();

                if permutation_commit.is_some() {
                    let permutation_opened_values: Vec<Vec<Challenge<SC>>> = opened_values
                        .chips
                        .iter()
                        .map(|chip| chip.permutation.to_vec_values())
                        .collect();

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
                    .map_err(|e| SumcheckVerificationError::MlPcsOpeningError(format!("{e:?}")))?;
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
                    .map_err(|e| SumcheckVerificationError::MlPcsOpeningError(format!("{e:?}")))?;
                }

                Ok(())
            },
        )?;

        tracing::debug_span!("verify polynomial evaluations v2").in_scope(|| {
            let num_constraints = chips
                .iter()
                .map(|chip| {
                    let nc = *HashMap::<String, usize>::get(&vk.constraints_map, &chip.name())
                        .expect("chip not found in constraints map");
                    nc
                })
                .collect::<Vec<usize>>();
            Self::verify_constraints(
                claim,
                chips,
                num_constraints,
                &opened_values.chips,
                &eq_challenges,
                alpha,
                &extended_sumcheck_challenges,
                &sumcheck_challenges,
                &permutation_challenges,
                public_values,
                num_skip_rounds,
                num_rounds_nonlinear,
            )
        })?;

        // Compute the expected local cumulative sum from dangling interactions.
        // Only core proofs have dangling State/MemoryGlobalAddr/projective Global interactions.
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

        Ok(())
    }

    fn verify_sumcheck_unipolys(
        claim: &mut Challenge<SC>,
        unipolys: &[UniPolyEvals<Challenge<SC>>],
        challenger: &mut SC::MlChallenger,
        num_rounds_linear: usize,
        num_skip_rounds: usize,
    ) -> Result<Vec<Challenge<SC>>, SumcheckVerificationError<SC>> {
        let mut sumcheck_challenges = Vec::with_capacity(unipolys.len());

        let mut challenge = Challenge::<SC>::zero();

        unipolys.iter().enumerate().try_for_each(|(idx, unipoly)| {
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

            Ok(())
        })?;

        Ok(sumcheck_challenges)
    }

    fn verify_opening_shape(
        chip: &MachineChip<SC, A>,
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
        if opening.permutation.local.len() != chip.permutation_width() {
            return Err(OpeningShapeError::PermutationWidthMismatch(
                chip.permutation_width(),
                opening.permutation.local.len(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_constraints(
        last_claim: Challenge<SC>,
        chips: &[&MachineChip<SC, A>],
        num_constraints: Vec<usize>,
        openings: &[SCChipOpenedValues<Val<SC>, Challenge<SC>>],
        eq_challenges: &[Challenge<SC>],
        alpha: Challenge<SC>,
        extended_sumcheck_challenges: &[Challenge<SC>],
        sumcheck_challenges: &[Challenge<SC>],
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
        num_skip_rounds: usize,
        num_rounds_nonlinear: usize,
    ) -> Result<(), SumcheckVerificationError<SC>>
    where
        A: for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        let (is_first_row, is_last_row): (Vec<Challenge<SC>>, Vec<Challenge<SC>>) = openings
            .iter()
            .map(|opening| {
                let start = extended_sumcheck_challenges.len() - opening.log_height;
                let (first, last) = extended_sumcheck_challenges[start..]
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
                let perm_empty = chip.permutation.local.is_empty();
                if perm_empty {
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
            permutation_challenges,
            public_values,
        );

        let sumcheck_challenges = sumcheck_challenges.iter().rev().copied().collect::<Vec<_>>();

        let mut eq = calculate_eq(
            &eq_challenges[num_rounds_nonlinear..],
            &sumcheck_challenges[num_rounds_nonlinear..],
        );
        let degree = (1 << num_skip_rounds) - 1;
        for (&r, &s) in eq_challenges[..num_rounds_nonlinear]
            .iter()
            .zip(sumcheck_challenges[..num_rounds_nonlinear].iter())
        {
            eq *= EqPoly::<Val<SC>, Challenge<SC>>::eval_eq(r, s, degree);
        }
        let lhs = main_eval * eq + perm_eval;
        if lhs == last_claim {
            Ok(())
        } else {
            tracing::debug!("[VERIFY FAIL] EvaluationsInconsistent");
            tracing::debug!("  lhs (main_eval*eq + perm_eval) = {:?}", lhs);
            tracing::debug!("  last_claim                     = {:?}", last_claim);
            tracing::debug!("  main_eval                      = {:?}", main_eval);
            tracing::debug!("  perm_eval                      = {:?}", perm_eval);
            tracing::debug!("  eq                             = {:?}", eq);
            tracing::debug!("  chips: {:?}", chips.iter().map(|c| c.name()).collect::<Vec<_>>());
            Err(SumcheckVerificationError::EvaluationsInconsistent)
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn eval_constraints(
        chips: &[&MachineChip<SC, A>],
        openings: &[SCChipOpenedValues<Val<SC>, Challenge<SC>>],
        selectors: &[Selectors<Challenge<SC>>],
        alpha: Challenge<SC>,
        alpha_shifts: Vec<Challenge<SC>>,
        permutation_challenges: &[Challenge<SC>],
        public_values: &[Val<SC>],
    ) -> Challenge<SC>
    where
        A: for<'a> Air<SumcheckVerifierConstraintFolder<'a, SC>>,
    {
        izip!(chips, openings, selectors, &alpha_shifts)
            .map(|(chip, opening, sels, &shift)| {
                let mut folder = SumcheckVerifierConstraintFolder::<SC> {
                    preprocessed: RowMajorMatrixView::new_row(&opening.preprocessed.local),
                    main: RowMajorMatrixView::new_row(&opening.main.local),
                    permutation: RowMajorMatrixView::new_row(&opening.permutation.local),
                    perm_challenges: permutation_challenges,
                    local_cumulative_sum: &opening.local_cumulative_sum,
                    is_first_row: sels.is_first_row,
                    is_last_row: sels.is_last_row,
                    alpha,
                    accumulator: Challenge::<SC>::zero(),
                    public_values,
                    constraint_count: 0,
                };

                chip.eval(&mut folder);

                if opening.permutation.local.is_empty() {
                    folder.accumulator * shift
                } else {
                    folder.accumulator * shift * alpha
                }
            })
            .sum()
    }
}

pub(crate) fn calculate_eq<F: Field>(r: &[F], s: &[F]) -> F {
    debug_assert_eq!(r.len(), s.len(), "r and s must have the same length");
    r.iter().zip(s).fold(F::one(), |acc, (&r, &s)| acc * ((r * s).double() - r - s + F::one()))
}

/// An error that occurs when the shape of the openings does not match the expected shape.
#[allow(clippy::enum_variant_names)]
pub enum OpeningShapeError {
    /// The width of the preprocessed trace does not match the expected width.
    PreprocessedWidthMismatch(usize, usize),
    /// The width of the main trace does not match the expected width.
    MainWidthMismatch(usize, usize),
    /// The width of the permutation trace does not match the expected width.
    PermutationWidthMismatch(usize, usize),
}

/// An error that occurs during the verification.
pub enum SumcheckVerificationError<SC: StarkGenericConfig> {
    /// opening proof is invalid.
    InvalidopeningArgument(OpeningError<SC>),
    /// evaluations are inconsistent.
    EvaluationsInconsistent,
    /// The shape of the opening arguments is invalid.
    OpeningShapeError(String, OpeningShapeError),
    /// The cpu chip is missing.
    MissingCpuChip,
    /// The length of the chip opening does not match the expected length.
    ChipOpeningLengthMismatch,
    /// The preprocessed chip id does not match the claimed opening id.
    PreprocessedChipIdMismatch(String, String),
    /// Cumulative sums error
    CumulativeSumsError(&'static str),
    /// The log degree of a chip is invalid.
    InvalidLogDegree(String, usize),
    /// The lookup multiplicity can overflow.
    LookupMultiplicityOverflow,
    /// The sum of the evaluations at zero and one of sumcheck unipoly is not equal to claim.
    SumcheckUniPolyError,
    /// `MlPCS` opening proof verification failed.
    MlPcsOpeningError(String),
}

impl Debug for OpeningShapeError {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            OpeningShapeError::PreprocessedWidthMismatch(expected, actual) => {
                write!(f, "Preprocessed width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::MainWidthMismatch(expected, actual) => {
                write!(f, "Main width mismatch: expected {}, got {}", expected, actual)
            }
            OpeningShapeError::PermutationWidthMismatch(expected, actual) => {
                write!(f, "Permutation width mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl Display for OpeningShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<SC: StarkGenericConfig> Debug for SumcheckVerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            SumcheckVerificationError::InvalidopeningArgument(e) => {
                write!(f, "Invalid opening argument: {:?}", e)
            }
            SumcheckVerificationError::EvaluationsInconsistent => {
                write!(f, "Evaluations are inconsistent")
            }
            SumcheckVerificationError::OpeningShapeError(chip, e) => {
                write!(f, "Invalid opening shape for chip {}: {:?}", chip, e)
            }
            SumcheckVerificationError::MissingCpuChip => {
                write!(f, "Missing CPU chip")
            }
            SumcheckVerificationError::ChipOpeningLengthMismatch => {
                write!(f, "Chip opening length mismatch")
            }
            SumcheckVerificationError::PreprocessedChipIdMismatch(expected, actual) => {
                write!(f, "Preprocessed chip id mismatch: expected {}, got {}", expected, actual)
            }
            SumcheckVerificationError::CumulativeSumsError(s) => {
                write!(f, "cumulative sums error: {}", s)
            }
            SumcheckVerificationError::InvalidLogDegree(chip, log_degree) => {
                write!(f, "Invalid log degree for chip {}: got {}", chip, log_degree)
            }
            SumcheckVerificationError::LookupMultiplicityOverflow => {
                write!(f, "Lookup multiplicity overflow")
            }
            SumcheckVerificationError::SumcheckUniPolyError => {
                write!(f, "Sumcheck unipoly error")
            }
            SumcheckVerificationError::MlPcsOpeningError(e) => {
                write!(f, "MlPCS opening proof verification failed: {}", e)
            }
        }
    }
}

impl<SC: StarkGenericConfig> Display for SumcheckVerificationError<SC> {
    #[allow(clippy::uninlined_format_args)]
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            SumcheckVerificationError::InvalidopeningArgument(_) => {
                write!(f, "Invalid opening argument")
            }
            SumcheckVerificationError::EvaluationsInconsistent => {
                write!(f, "Evaluations are inconsistent")
            }
            SumcheckVerificationError::OpeningShapeError(chip, e) => {
                write!(f, "Invalid opening shape for chip {}: {}", chip, e)
            }
            SumcheckVerificationError::MissingCpuChip => {
                write!(f, "Missing CPU chip in shard")
            }
            SumcheckVerificationError::ChipOpeningLengthMismatch => {
                write!(f, "Chip opening length mismatch")
            }
            SumcheckVerificationError::CumulativeSumsError(s) => {
                write!(f, "cumulative sums error: {}", s)
            }
            SumcheckVerificationError::PreprocessedChipIdMismatch(expected, actual) => {
                write!(f, "Preprocessed chip id mismatch: expected {}, got {}", expected, actual)
            }
            SumcheckVerificationError::InvalidLogDegree(chip, log_degree) => {
                write!(f, "Invalid log degree for chip {}: got {}", chip, log_degree)
            }
            SumcheckVerificationError::LookupMultiplicityOverflow => {
                write!(f, "Lookup multiplicity overflow")
            }
            SumcheckVerificationError::SumcheckUniPolyError => {
                write!(f, "Sumcheck unipoly error")
            }
            SumcheckVerificationError::MlPcsOpeningError(e) => {
                write!(f, "MlPCS opening proof verification failed: {}", e)
            }
        }
    }
}

impl<SC: StarkGenericConfig> std::error::Error for SumcheckVerificationError<SC> {}
