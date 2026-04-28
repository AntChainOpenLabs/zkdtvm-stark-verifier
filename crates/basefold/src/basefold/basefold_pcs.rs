use crate::basefold::mlpcs::MlPCS;
use crate::basefold::sumcheck::SumcheckInstanceProof;
use crate::utils::eqpoly::EqPolynomial;
use crate::utils::math::compute_dotproduct_mix;
use core::fmt::{Debug, Display, Formatter};
use itertools::izip;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, Powers, TwoAdicField};
use p3_fri::{BatchOpening, FriConfig, QueryProof};
use p3_matrix::compressed::CompressedMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::marker::PhantomData;

#[derive(Debug)]
pub enum BaseFoldError<CommitMmcsErr, InputError> {
    CommitPhaseMmcsError(CommitMmcsErr),
    CommitmentCheckFailed,
    SumcheckPhaseError,
    FinalPolyMismatch,
    InvalidPowWitness,
    InvalidInputError,
    FriFinalStepMisMatch,
    _PhantomInputError(PhantomData<InputError>),
}

/// Batch opening proof for the Basefold PCS.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct BasefoldProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    pub sumcheck_transcript: SumcheckInstanceProof<F>,
    pub iopp_oracles: Vec<M::Commitment>,
    pub iopp_queries: Vec<QueryProof<F, M>>,
    pub query_openings: InputProof,
    pub grinding_batching_witness: Vec<Witness>,
    pub grinding_query_witness: Vec<Witness>,
    pub out_of_domain_responses: Option<Vec<F>>,
}

/// A matrix's dimensions paired with its global index across all batches.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DimAndNo {
    pub dim: Dimensions,
    pub num: usize,
}

impl Debug for DimAndNo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "dim: {:?}, No: {}", self.dim, self.num)
    }
}

impl Display for DimAndNo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "dim: {:?}, No: {}", self.dim, self.num)
    }
}

#[derive(Debug)]
pub struct BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger> {
    mmcs: InputMmcs,
    pub fri: FriConfig<FriMmcs>,
    _phantom: PhantomData<(F, EF, Challenger)>,
}

struct MatricesSizeIndex {
    prefix_sums: Vec<usize>,
}

impl MatricesSizeIndex {
    fn new(matrices_size: &Vec<Vec<Dimensions>>) -> Self {
        let mut prefix_sums = Vec::with_capacity(matrices_size.len() + 1);
        prefix_sums.push(0);
        let mut sum = 0;
        for vec in matrices_size {
            sum += vec.len();
            prefix_sums.push(sum);
        }
        Self { prefix_sums }
    }

    fn find_position(&self, index: usize) -> (usize, usize) {
        let i = match self.prefix_sums.binary_search(&index) {
            Ok(exact) => exact,
            Err(insert_pos) => insert_pos - 1,
        };
        let j = index - self.prefix_sums[i];
        (i, j)
    }
}

impl<Val, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<Val, InputMmcs, FriMmcs, EF, Challenger>
{
    pub const fn new(mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }

    pub fn mmcs_ref(&self) -> &InputMmcs {
        &self.mmcs
    }

    pub fn fri_ref(&self) -> &FriConfig<FriMmcs> {
        &self.fri
    }
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> MlPCS
    for BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    type Field = F;
    type ExtensionField = EF;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<F>>;
    type Commitment = InputMmcs::Commitment;
    type BatchProof = BasefoldProof<EF, FriMmcs, F, Vec<Vec<BatchOpening<F, InputMmcs>>>>;
    type Challenger = Challenger;
    type Error = BaseFoldError<FriMmcs::Error, InputMmcs::Error>;

    fn commit(
        &self,
        _evaluations: Vec<&CompressedMatrix<Self::Field>>,
    ) -> (Self::Commitment, Self::ProverData) {
        unimplemented!("commit is not available in verifier-only build")
    }

    fn open(
        &self,
        _polynomials_batch: Vec<Vec<CompressedMatrix<Self::Field>>>,
        _prover_data: Vec<Self::ProverData>,
        _opening_point: &[Self::ExtensionField],
        _opened_values: &Vec<Vec<Vec<Self::ExtensionField>>>,
        _challenger: &mut Self::Challenger,
    ) -> Result<Self::BatchProof, Self::Error> {
        unimplemented!("open is not available in verifier-only build")
    }

    #[tracing::instrument(skip_all, level = "debug", name = "BaseFold::verify")]
    fn verify(
        &self,
        commitment_batch: Vec<Self::Commitment>,
        matrices_size_batch: &Vec<Vec<Dimensions>>,
        opening_point: &[Self::ExtensionField],
        opened_values_batch: &Vec<Vec<Vec<Self::ExtensionField>>>,
        proof: &Self::BatchProof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error> {
        self.validate_verify_inputs(&commitment_batch, matrices_size_batch, opened_values_batch)?;

        let BasefoldProof {
            sumcheck_transcript,
            iopp_oracles,
            iopp_queries,
            query_openings,
            grinding_batching_witness,
            grinding_query_witness,
            out_of_domain_responses: _,
        } = proof;

        if grinding_batching_witness.len() != 2 || grinding_query_witness.len() != 2 {
            return Err(BaseFoldError::InvalidInputError);
        }

        let num_vars = opening_point.len();

        let flat_dims: Vec<DimAndNo> = matrices_size_batch
            .iter()
            .flat_map(|batch| batch.iter())
            .enumerate()
            .map(|(idx, dim)| DimAndNo {
                dim: dim.clone(),
                num: idx,
            })
            .collect();
        let flat_opened_values: Vec<&Vec<EF>> =
            opened_values_batch.iter().flat_map(|v| v.iter()).collect();

        let matrices_by_log_height =
            Self::group_dims_by_log_height(&flat_dims, &flat_opened_values);
        let log_max_height = matrices_by_log_height.keys().max().cloned().unwrap_or(0);
        debug_assert_eq!(log_max_height, num_vars);

        let size_index = MatricesSizeIndex::new(matrices_size_batch);

        let alpha: EF = challenger.sample_ext_element();
        let beta: EF = challenger.sample_ext_element();
        let mut alpha_powers = Powers::<EF> {
            base: alpha,
            current: EF::one(),
        };

        let mut claims_by_height: BTreeMap<usize, EF> = BTreeMap::new();
        let mut coefficients_by_height: BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>> =
            BTreeMap::new();

        let mut beta_power = EF::one();
        let mut is_first_group = true;

        for (&log_height, group) in matrices_by_log_height.iter().rev() {
            let (dims, values): (Vec<&DimAndNo>, Vec<&Vec<EF>>) =
                group.iter().map(|(d, v)| (*d, *v)).unzip();

            let coeffs: Vec<Vec<EF>> = values
                .iter()
                .map(|vals| vals.iter().map(|_| alpha_powers.next().unwrap()).collect())
                .collect();

            if is_first_group {
                is_first_group = false;
            } else {
                beta_power *= beta;
            }

            let claimed_sum: EF = values
                .iter()
                .zip(coeffs.iter())
                .flat_map(|(vals, cs)| vals.iter().zip(cs.iter()).map(|(v, c)| *v * *c))
                .sum::<EF>()
                * beta_power;
            claims_by_height.insert(log_height, claimed_sum);

            let scaled: Vec<((usize, usize), Vec<EF>)> = coeffs
                .iter()
                .enumerate()
                .map(|(i, cs)| {
                    let scaled_cs: Vec<EF> = cs.iter().map(|c| *c * beta_power).collect();
                    (size_index.find_position(dims[i].num), scaled_cs)
                })
                .collect();
            coefficients_by_height.insert(log_height, scaled);
        }

        challenger.observe(grinding_batching_witness[0]);
        if !challenger.check_witness(
            self.fri.grinding_bits_batching,
            grinding_batching_witness[1],
        ) {
            return Err(BaseFoldError::InvalidPowWitness);
        }

        let mut poly_iter = sumcheck_transcript.uni_polys.iter();
        let mut current_claim = claims_by_height
            .remove(&num_vars)
            .ok_or(BaseFoldError::SumcheckPhaseError)?;

        challenger.observe(iopp_oracles[0].clone());

        let mut folding_challenges: Vec<EF> = Vec::with_capacity(num_vars);
        let mut merge_betas: Vec<EF> = Vec::new();
        let mut eq_factor = EF::one();

        for round in (0..=num_vars).rev() {
            if round < num_vars {
                challenger.observe(iopp_oracles[num_vars - round].clone());
            }
            if round == 0 {
                break;
            }

            let uni_poly = poly_iter.next().ok_or(BaseFoldError::SumcheckPhaseError)?;
            if uni_poly.eval_at_zero() + uni_poly.eval_at_one() != current_claim {
                return Err(BaseFoldError::SumcheckPhaseError);
            }
            uni_poly
                .coeffs
                .iter()
                .for_each(|c| challenger.observe_ext_element(*c));
            let r_fold: EF = challenger.sample_ext_element();
            folding_challenges.push(r_fold);
            current_claim = uni_poly.evaluate(&r_fold);

            let p_i = opening_point[round - 1];
            eq_factor *= p_i * r_fold + (EF::one() - p_i) * (EF::one() - r_fold);

            if let Some(branch_claim) = claims_by_height.remove(&(round - 1)) {
                let merge_beta: EF = challenger.sample_ext_element();
                merge_betas.push(merge_beta);

                current_claim = current_claim + merge_beta * branch_claim;

                eq_factor = EF::one();
            }
        }

        let fc_rev: Vec<EF> = folding_challenges.iter().rev().cloned().collect();

        let min_log_height = matrices_by_log_height
            .keys()
            .min()
            .cloned()
            .unwrap_or(num_vars);

        let combined_eq_sum = EqPolynomial::new(opening_point[..min_log_height].to_vec())
            .evaluate(&fc_rev[..min_log_height]);

        let combined_f_r: EF = current_claim / combined_eq_sum;

        let expected_codeword = vec![combined_f_r; 1 << self.fri.log_blowup];
        let (expected_commitment, _) = self
            .fri
            .mmcs
            .commit_matrix(RowMajorMatrix::new(expected_codeword, 2));

        let last_oracle = iopp_oracles
            .last()
            .ok_or(BaseFoldError::CommitmentCheckFailed)?;
        let last_bytes =
            bincode::serialize(last_oracle).map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        let expected_bytes = bincode::serialize(&expected_commitment)
            .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        if last_bytes != expected_bytes {
            return Err(BaseFoldError::CommitmentCheckFailed);
        }

        challenger.observe(grinding_query_witness[0]);
        if !challenger.check_witness(self.fri.grinding_bits_query, grinding_query_witness[1]) {
            return Err(BaseFoldError::InvalidPowWitness);
        }

        let query_points: Vec<usize> = (0..self.fri.num_queries)
            .map(|_| challenger.sample_bits(num_vars + self.fri.log_blowup))
            .collect();

        let all_queries_valid = iopp_queries
            .par_iter()
            .zip(query_openings.par_iter())
            .enumerate()
            .all(|(i, (query, leaf_opening))| {
                self.verify_query_p3_batch(
                    &commitment_batch,
                    iopp_oracles.as_slice(),
                    query_points[i],
                    matrices_size_batch,
                    query,
                    leaf_opening,
                    &coefficients_by_height,
                    &folding_challenges,
                    &merge_betas,
                    opening_point,
                    &combined_f_r,
                )
                .is_ok()
            });

        if !all_queries_valid {
            return Err(BaseFoldError::FriFinalStepMisMatch);
        }

        Ok(())
    }
}

impl<F, InputMmcs, FriMmcs, EF, Challenger> BaseFoldPcs<F, InputMmcs, FriMmcs, EF, Challenger>
where
    F: TwoAdicField,
    InputMmcs: Mmcs<F> + Send + Sync,
    FriMmcs: Mmcs<EF> + Send + Sync,
    EF: TwoAdicField + ExtensionField<F>,
    Challenger:
        FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = F>,
{
    pub fn verify_iopp_query(
        &self,
        iopp_commitments: &[FriMmcs::Commitment],
        mut query_point: usize,
        leaf_sums_by_log_height: BTreeMap<usize, EF>,
        query_proof: &QueryProof<EF, FriMmcs>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        let num_vars = folding_challenges.len();
        let log_max_height = num_vars + self.fri.log_blowup;

        let mut folded_eval = EF::zero();
        let mut merge_idx: usize = 0;
        let mut eq_factor = EF::one();
        let mut height_iter = leaf_sums_by_log_height.iter().rev().peekable();

        for (round, (&_r, commitment, opening)) in izip!(
            folding_challenges,
            iopp_commitments,
            &query_proof.commit_phase_openings
        )
        .enumerate()
        {
            let log_folded_height = log_max_height - round - 1;

            if let Some((_, &leaf_sum)) =
                height_iter.next_if(|(lh, _)| **lh == log_folded_height + 1)
            {
                if merge_idx == 0 {
                    folded_eval = leaf_sum;
                } else {
                    folded_eval = eq_factor * folded_eval + merge_betas[merge_idx - 1] * leaf_sum;
                    eq_factor = EF::one();
                }
                merge_idx += 1;
            }

            let sibling_index = query_point ^ 1;
            let pair_index = query_point >> 1;

            let mut pair_evals = vec![folded_eval; 2];
            pair_evals[sibling_index % 2] = opening.sibling_value;

            self.fri
                .mmcs
                .verify_batch(
                    commitment,
                    &[Dimensions {
                        width: 2,
                        height: 1 << log_folded_height,
                    }],
                    pair_index,
                    &[pair_evals.clone()],
                    &opening.opening_proof,
                )
                .map_err(BaseFoldError::CommitPhaseMmcsError)?;

            query_point = pair_index;
            let generator = EF::two_adic_generator(log_folded_height + 1)
                .exp_u64(reverse_bits_len(query_point, log_folded_height) as u64);

            let slope = (pair_evals[1] - pair_evals[0]) / (-generator - generator);
            let intercept = pair_evals[0] - slope * generator;
            folded_eval = intercept + slope * folding_challenges[round];

            let var_idx = num_vars - 1 - round;
            let p_i = opening_point[var_idx];
            let fc_i = folding_challenges[round];
            eq_factor *= p_i * fc_i + (EF::one() - p_i) * (EF::one() - fc_i);
        }

        if folded_eval != *expected_final_value {
            return Err(BaseFoldError::FinalPolyMismatch);
        }
        Ok(())
    }

    pub fn verify_query_p3_batch(
        &self,
        commitments: &[InputMmcs::Commitment],
        iopp_commitments: &[FriMmcs::Commitment],
        query_point: usize,
        matrices_size_batch: &[Vec<Dimensions>],
        query_proof: &QueryProof<EF, FriMmcs>,
        leaf_openings: &[BatchOpening<F, InputMmcs>],
        coefficients_by_height: &BTreeMap<usize, Vec<((usize, usize), Vec<EF>)>>,
        folding_challenges: &[EF],
        merge_betas: &[EF],
        opening_point: &[EF],
        expected_final_value: &EF,
    ) -> Result<(), BaseFoldError<FriMmcs::Error, InputMmcs::Error>> {
        for (batch_dims, (commitment, opening)) in matrices_size_batch
            .iter()
            .zip(commitments.iter().zip(leaf_openings.iter()))
        {
            let max_log_height = batch_dims
                .iter()
                .map(|dim| log2_strict_usize(dim.height))
                .max()
                .unwrap_or(0);

            let codeword_dims: Vec<Dimensions> = batch_dims
                .iter()
                .map(|dim| Dimensions {
                    width: 0,
                    height: dim.height << self.fri.log_blowup,
                })
                .collect();

            self.mmcs
                .verify_batch(
                    commitment,
                    &codeword_dims,
                    query_point >> (folding_challenges.len() - max_log_height),
                    &opening.opened_values,
                    &opening.opening_proof,
                )
                .map_err(|_| BaseFoldError::CommitmentCheckFailed)?;
        }

        let leaf_sums_by_log_height: BTreeMap<usize, EF> = coefficients_by_height
            .iter()
            .map(|(&log_height, entries)| {
                let sum = entries
                    .par_iter()
                    .map(|((batch_idx, mat_idx), coeffs)| {
                        compute_dotproduct_mix(
                            coeffs,
                            &leaf_openings[*batch_idx].opened_values[*mat_idx],
                        )
                    })
                    .reduce(|| EF::zero(), |acc, val| acc + val);
                (log_height + self.fri.log_blowup, sum)
            })
            .collect();

        self.verify_iopp_query(
            iopp_commitments,
            query_point,
            leaf_sums_by_log_height,
            query_proof,
            folding_challenges,
            merge_betas,
            opening_point,
            expected_final_value,
        )
    }
}
